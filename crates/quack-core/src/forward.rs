//! Reads through the process that holds a workspace (`docs/desktop.md`,
//! section 3.6).
//!
//! `DuckDB` allows one process with read-write access to a file, or any
//! number of read-only processes, never a writer beside readers: a second
//! `quack` cannot open a workspace the terminal, `quack serve`, or the
//! desktop app holds, even for `SELECT 1`. There is one lock and one
//! `DuckDB` instance per workspace, and the way through it is to run the
//! second process's reads inside the holder.
//!
//! Whichever process holds a workspace open also listens on a Unix socket
//! in the workspace directory ([`SOCKET_FILE`]) with a per-launch token in
//! a 0600 file beside it ([`TOKEN_FILE`]); both go when the workspace
//! closes. A command that hits the lock ([`Error::WorkspaceLocked`]) looks
//! for the socket and runs its read there, on the holder's reader pool in
//! a read-only transaction, and gets the same rows it would have read
//! itself. A write is refused naming the holder: forwarded blind it would
//! run under the holder's policy with nobody to ask and no one to audit it
//! as.
//!
//! Neither end trusts the files or the peer on faith. A client uses the
//! socket and the token file only when each is the right kind of file,
//! owned by the current user, and unreachable by anyone else; both ends
//! check the peer's credentials and refuse a process running as another
//! user; a holder never removes a stale socket that is not its own user's.
//!
//! One request per connection, one JSON object per line each way. The
//! module is Unix-only; elsewhere [`connect`] finds no holder and
//! [`Listener::bind`] listens on nothing.

#[cfg(unix)]
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::error::Error;
use crate::ids::WorkspaceId;
use crate::storage::workspace::QueryResults;

/// The process that holds workspaces open, as the token file names it to
/// the person reading `quack -q`'s stderr, and the one workspace it holds:
/// `quack serve` opens any workspace a request names; every other holder
/// opened exactly one and refuses the rest (`engine::Engine::workspace`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Holder {
    Serve,
    Terminal(WorkspaceId),
    Print(WorkspaceId),
    Mcp(WorkspaceId),
    Search(WorkspaceId),
    Saved(WorkspaceId),
}

impl Holder {
    /// The one workspace this holder opened, or `None` for the server.
    #[must_use]
    pub fn held(&self) -> Option<&WorkspaceId> {
        match self {
            Self::Serve => None,
            Self::Terminal(id)
            | Self::Print(id)
            | Self::Mcp(id)
            | Self::Search(id)
            | Self::Saved(id) => Some(id),
        }
    }
}

impl std::fmt::Display for Holder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Serve => "quack serve",
            Self::Terminal(_) => "the quack terminal session",
            Self::Print(_) => "quack -p",
            Self::Mcp(_) => "quack mcp",
            Self::Search(_) => "quack search",
            Self::Saved(_) => "quack saved",
        })
    }
}

/// The socket a workspace's holder listens on, inside the workspace
/// directory.
pub const SOCKET_FILE: &str = "quack.sock";
/// The token a request must carry, beside the socket, mode 0600.
pub const TOKEN_FILE: &str = "quack.sock.token";
/// The most bytes one request line may run to.
#[cfg(unix)]
const MAX_REQUEST_BYTES: u64 = 1 << 20;

/// What the token file holds: the token, and who listens.
#[cfg(unix)]
#[derive(Debug, Serialize, Deserialize)]
struct TokenFile {
    token: String,
    /// The holder as a person reads it: `quack serve`, `Quack.app`, the
    /// terminal session.
    holder: String,
    pid: u32,
}

/// One request from a process that could not take the lock.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "request", rename_all = "snake_case")]
pub enum Request {
    /// Who holds the workspace.
    Hello,
    /// A read statement, run as `quack -q` would run it.
    Query { sql: String },
}

/// The holder's answer.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "reply", rename_all = "snake_case")]
pub enum Reply {
    Hello {
        holder: String,
    },
    Results {
        results: QueryResults,
    },
    /// The request was a write, which the holder runs for nobody else.
    Refused,
    /// The request failed on the holder's side.
    Error {
        message: String,
    },
}

#[cfg(unix)]
#[derive(Debug, Serialize, Deserialize)]
struct Envelope {
    token: String,
    request: Request,
}

/// Where a workspace's socket and token live.
#[cfg(unix)]
#[derive(Debug, Clone)]
struct Paths {
    socket: PathBuf,
    token: PathBuf,
}

#[cfg(unix)]
impl Paths {
    /// The socket and token inside `workspace_dir`
    /// (`Config::workspace_dir`).
    fn of(workspace_dir: &Path) -> Self {
        Self {
            socket: workspace_dir.join(SOCKET_FILE),
            token: workspace_dir.join(TOKEN_FILE),
        }
    }
}

#[cfg(unix)]
mod unix {
    use std::io::Write as _;
    use std::os::unix::fs::{FileTypeExt, MetadataExt, OpenOptionsExt, PermissionsExt};
    use std::path::Path;
    use std::sync::{Arc, Mutex};

    use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader};
    use tokio::net::{UnixListener, UnixStream};

    use super::{Envelope, Holder, MAX_REQUEST_BYTES, Paths, Reply, Request, TokenFile};
    use crate::analysis::tools::ReaderDb;
    use crate::error::{Error, Result};
    use crate::storage::control::random_bytes;
    use crate::storage::workspace::{QueryResults, StatementKind};

    /// What a refusal names as the holder before the token file has been
    /// read, or when the other end is not who it should be.
    const SOMEONE: &str = "another process";

    /// The user this process runs as.
    fn current_uid() -> u32 {
        rustix::process::getuid().as_raw()
    }

    /// Which kind of file a path must be to be used.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum Kind {
        Socket,
        Regular,
    }

    fn refuse(path: &Path, what: &str) -> Error {
        Error::Forward(format!(
            "{} {what}; refusing to use it (the workspace is open in {SOMEONE})",
            path.display()
        ))
    }

    /// `path` is a `kind` file this user owns. Symlinks are not followed, so
    /// a link planted at the path is refused rather than resolved.
    #[expect(
        clippy::filetype_is_file,
        reason = "a regular file is exactly what the token file must be: not a link, a directory, a device, or a pipe"
    )]
    fn check_owned(path: &Path, kind: Kind) -> Result<std::fs::Metadata> {
        let meta = std::fs::symlink_metadata(path)?;
        let is_kind = match kind {
            Kind::Socket => meta.file_type().is_socket(),
            Kind::Regular => meta.file_type().is_file(),
        };
        if !is_kind {
            let want = match kind {
                Kind::Socket => "a socket",
                Kind::Regular => "a regular file",
            };
            return Err(refuse(path, &format!("is not {want}")));
        }
        if meta.uid() != current_uid() {
            return Err(refuse(
                path,
                &format!("is owned by uid {}, not this user", meta.uid()),
            ));
        }
        Ok(meta)
    }

    /// `path` is a `kind` file this user owns and nobody else can reach.
    fn check_private(path: &Path, kind: Kind) -> Result<()> {
        let meta = check_owned(path, kind)?;
        let mode = meta.mode() & 0o777;
        if mode & 0o077 != 0 {
            return Err(refuse(
                path,
                &format!("is reachable by others (mode {mode:o})"),
            ));
        }
        Ok(())
    }

    /// The other end of `stream` runs as this user.
    fn check_peer(stream: &UnixStream) -> Result<()> {
        let cred = stream.peer_cred()?;
        if cred.uid() != current_uid() {
            return Err(Error::Forward(format!(
                "the other end of the socket runs as uid {}, not this user; refusing it",
                cred.uid()
            )));
        }
        Ok(())
    }

    /// The holder's end: the socket, the token file, and the accept loop.
    /// Dropping it stops listening and removes both files.
    pub struct Listener {
        paths: Paths,
        accept: tokio::task::JoinHandle<()>,
        /// The reader the accept loop answers on, held here rather than in
        /// the loop's future: an aborted task drops its future only when the
        /// runtime next polls it, and the reader's connections must close with
        /// the holder's, so that closing the workspace checkpoints its log.
        /// Dropping the listener takes it out at once.
        reader: Arc<Mutex<Option<ReaderDb>>>,
    }

    impl Listener {
        /// Listen in `workspace_dir` (`Config::workspace_dir`) for the
        /// workspace's reads as `holder` (the name a refused writer is told),
        /// answering them on `reader`. Takes the directory, not the `Config`,
        /// so a caller's future holds a path, not a configuration. A socket left by a
        /// holder of this user that died is replaced; one that answers is an
        /// error, since two holders cannot both have the file open; one that
        /// another user owns is an error and is left alone.
        ///
        /// # Errors
        ///
        /// Returns an error when the socket cannot be bound, the token cannot
        /// be written, another process already answers on the socket, or a
        /// file in the way is not this user's.
        pub async fn bind(workspace_dir: &Path, holder: &Holder, reader: ReaderDb) -> Result<Self> {
            let holder = holder.to_string();
            let paths = Paths::of(workspace_dir);
            if paths.socket.exists() {
                check_owned(&paths.socket, Kind::Socket)?;
                if UnixStream::connect(&paths.socket).await.is_ok() {
                    return Err(Error::Forward(format!(
                        "{holder} cannot listen: another process already answers on {}",
                        paths.socket.display()
                    )));
                }
                std::fs::remove_file(&paths.socket)?;
            }
            let listener = UnixListener::bind(&paths.socket)?;
            std::fs::set_permissions(&paths.socket, std::fs::Permissions::from_mode(0o600))?;
            let token = Self::write_token(&paths.token, &holder)?;
            let token: Arc<str> = Arc::from(token);
            let holder: Arc<str> = Arc::from(holder);
            let reader = Arc::new(Mutex::new(Some(reader)));
            let shared = Arc::clone(&reader);
            let accept = tokio::spawn(async move {
                loop {
                    match listener.accept().await {
                        Ok((stream, _)) => {
                            // A connection that arrives as the holder closes
                            // finds no reader and is dropped unanswered.
                            let Some(reader) = shared.lock().ok().and_then(|r| r.clone()) else {
                                return;
                            };
                            let (token, holder) = (Arc::clone(&token), Arc::clone(&holder));
                            tokio::spawn(async move {
                                if let Err(e) = serve_one(stream, &token, &holder, &reader).await {
                                    tracing::debug!(error = %e, "a forwarded read ended early");
                                }
                            });
                        }
                        Err(e) => {
                            tracing::warn!(error = %e, "accepting a forwarded read failed");
                            tokio::time::sleep(std::time::Duration::from_millis(100)).await;
                        }
                    }
                }
            });
            Ok(Self {
                paths,
                accept,
                reader,
            })
        }

        /// A fresh token in a 0600 file, replacing one this user left behind.
        fn write_token(path: &Path, holder: &str) -> Result<String> {
            let mut bytes = [0u8; 32];
            random_bytes(&mut bytes)?;
            let token =
                base64::Engine::encode(&base64::engine::general_purpose::URL_SAFE_NO_PAD, bytes);
            let record = TokenFile {
                token: token.clone(),
                holder: holder.to_owned(),
                pid: std::process::id(),
            };
            if path.exists() {
                check_owned(path, Kind::Regular)?;
                std::fs::remove_file(path)?;
            }
            let mut file = std::fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .mode(0o600)
                .open(path)?;
            file.write_all(serde_json::to_string(&record)?.as_bytes())?;
            Ok(token)
        }
    }

    impl std::fmt::Debug for Listener {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.debug_struct("Listener")
                .field("socket", &self.paths.socket)
                .field("token", &self.paths.token)
                .finish_non_exhaustive()
        }
    }

    impl Drop for Listener {
        fn drop(&mut self) {
            self.accept.abort();
            // Release the reader now, not when the aborted task is next polled.
            if let Ok(mut reader) = self.reader.lock() {
                drop(reader.take());
            }
            for path in [&self.paths.socket, &self.paths.token] {
                if let Err(e) = std::fs::remove_file(path)
                    && e.kind() != std::io::ErrorKind::NotFound
                {
                    tracing::warn!(path = %path.display(), error = %e, "could not remove the forwarding file");
                }
            }
        }
    }

    /// Answer one connection: one request line, one reply line. A peer that
    /// is not this user gets nothing, not even a reply.
    async fn serve_one(
        stream: UnixStream,
        token: &str,
        holder: &str,
        reader: &ReaderDb,
    ) -> Result<()> {
        if let Err(e) = check_peer(&stream) {
            tracing::warn!(error = %e, "refused a connection to the forwarding socket");
            return Ok(());
        }
        let (read, mut write) = stream.into_split();
        let mut line = String::new();
        BufReader::new(read.take(MAX_REQUEST_BYTES))
            .read_line(&mut line)
            .await?;
        if line.trim().is_empty() {
            // A probe (`connect` checking that someone answers), not a request.
            return Ok(());
        }
        let reply = match serde_json::from_str::<Envelope>(&line) {
            Ok(envelope) if token_matches(token, &envelope.token) => {
                answer(envelope.request, holder, reader).await
            }
            Ok(_) => Reply::Error {
                message: String::from("the token does not match the holder's"),
            },
            Err(e) => Reply::Error {
                message: format!("malformed request: {e}"),
            },
        };
        let mut text = serde_json::to_string(&reply)?;
        text.push('\n');
        write.write_all(text.as_bytes()).await?;
        write.shutdown().await?;
        Ok(())
    }

    fn token_matches(expected: &str, presented: &str) -> bool {
        aws_lc_rs::constant_time::verify_slices_are_equal(expected.as_bytes(), presented.as_bytes())
            .is_ok()
    }

    async fn answer(request: Request, holder: &str, reader: &ReaderDb) -> Reply {
        match request {
            Request::Hello => Reply::Hello {
                holder: holder.to_owned(),
            },
            Request::Query { sql } => match read(reader, sql).await {
                Ok(Some(results)) => Reply::Results { results },
                Ok(None) => Reply::Refused,
                Err(e) => Reply::Error {
                    message: e.to_string(),
                },
            },
        }
    }

    /// Run a read on the holder's reader pool; `None` when the statement
    /// is a write.
    async fn read(reader: &ReaderDb, sql: String) -> Result<Option<QueryResults>> {
        reader
            .with_db(move |db| match db.classify_statement(&sql)? {
                StatementKind::Read => db.execute_query(&sql).map(Some),
                StatementKind::Write => Ok(None),
                StatementKind::Invalid(message) => Err(Error::Analysis(message)),
            })
            .await
    }

    /// The other end: a connection to the holder of the workspace in
    /// `workspace_dir`, when one listens. `None` when there is no token file
    /// or no socket, or nothing answers on the socket (a holder that died
    /// leaves both behind).
    ///
    /// # Errors
    ///
    /// Returns an error when the token file or the socket exists but is not
    /// a private file of this user's, cannot be read, or the process
    /// answering runs as another user.
    pub async fn connect(workspace_dir: &Path) -> Result<Option<Client>> {
        let paths = Paths::of(workspace_dir);
        if !paths.token.exists() || !paths.socket.exists() {
            return Ok(None);
        }
        check_private(&paths.token, Kind::Regular)?;
        let text = std::fs::read_to_string(&paths.token)?;
        let token: TokenFile = serde_json::from_str(&text)
            .map_err(|e| refuse(&paths.token, &format!("is malformed ({e})")))?;
        check_private(&paths.socket, Kind::Socket)?;
        let Ok(probe) = UnixStream::connect(&paths.socket).await else {
            return Ok(None);
        };
        check_peer(&probe)?;
        drop(probe);
        Ok(Some(Client {
            paths,
            token: token.token,
            holder: token.holder,
        }))
    }

    /// A process that could not take the lock, reading through the holder.
    #[derive(Debug)]
    pub struct Client {
        paths: Paths,
        token: String,
        holder: String,
    }

    impl Client {
        /// Who holds the workspace, as the token file names it.
        #[must_use]
        pub fn holder(&self) -> &str {
            &self.holder
        }

        /// Run a read statement inside the holder and get its rows.
        ///
        /// # Errors
        ///
        /// Returns [`Error::ForwardedWriteRefused`] when the statement is a
        /// write, [`Error::Forward`] when the holder failed, the protocol
        /// broke, or the other end is not this user, and an I/O error when
        /// the socket is gone.
        pub async fn query(&self, sql: &str) -> Result<QueryResults> {
            match self
                .send(Request::Query {
                    sql: sql.to_owned(),
                })
                .await?
            {
                Reply::Results { results } => Ok(results),
                Reply::Refused => Err(Error::ForwardedWriteRefused {
                    holder: self.holder.clone(),
                }),
                Reply::Error { message } => Err(self.failed(&message)),
                Reply::Hello { .. } => Err(self.failed("unexpected reply")),
            }
        }

        async fn send(&self, request: Request) -> Result<Reply> {
            let stream = UnixStream::connect(&self.paths.socket).await?;
            check_peer(&stream)?;
            let (read, mut write) = stream.into_split();
            let mut text = serde_json::to_string(&Envelope {
                token: self.token.clone(),
                request,
            })?;
            text.push('\n');
            write.write_all(text.as_bytes()).await?;
            write.shutdown().await?;
            let mut line = String::new();
            BufReader::new(read).read_line(&mut line).await?;
            if line.is_empty() {
                return Err(self.failed("the holder closed without answering"));
            }
            serde_json::from_str(&line).map_err(|e| self.failed(&format!("malformed reply: {e}")))
        }

        fn failed(&self, message: &str) -> Error {
            Error::Forward(format!("reading through {} failed: {message}", self.holder))
        }
    }
}

#[cfg(unix)]
pub use unix::{Client, Listener, connect};

#[cfg(not(unix))]
mod other {
    use std::path::Path;

    use super::Holder;
    use crate::analysis::tools::ReaderDb;
    use crate::error::Result;
    use crate::storage::workspace::QueryResults;

    /// No holder listens on this platform.
    #[derive(Debug)]
    pub struct Listener;

    impl Listener {
        /// Listens on nothing: Unix sockets are the only transport so far.
        ///
        /// # Errors
        ///
        /// Never fails.
        pub async fn bind(
            _workspace_dir: &Path,
            _holder: &Holder,
            _reader: ReaderDb,
        ) -> Result<Self> {
            Ok(Self)
        }
    }

    /// A holder's client; never constructed on this platform.
    #[derive(Debug)]
    pub struct Client {
        holder: String,
    }

    impl Client {
        #[must_use]
        pub fn holder(&self) -> &str {
            &self.holder
        }

        /// # Errors
        ///
        /// Never constructed, so never called.
        pub async fn query(&self, _sql: &str) -> Result<QueryResults> {
            Err(crate::error::Error::Forward(format!(
                "reading through {} failed: read forwarding needs Unix sockets",
                self.holder
            )))
        }
    }

    /// No holder is ever found on this platform.
    ///
    /// # Errors
    ///
    /// Never fails.
    pub async fn connect(_workspace_dir: &Path) -> Result<Option<Client>> {
        Ok(None)
    }
}

#[cfg(not(unix))]
pub use other::{Client, Listener, connect};

/// Whether `err` is the lock another process holds on the workspace file:
/// the case a command answers by reading through the holder.
#[must_use]
pub fn is_locked(err: &Error) -> bool {
    matches!(err, Error::WorkspaceLocked { .. })
}

#[cfg(all(test, unix))]
mod tests {
    use std::os::unix::fs::PermissionsExt;
    use std::path::Path;
    use std::sync::Arc;

    use super::{Holder, Listener, Paths, SOCKET_FILE, TOKEN_FILE, connect};
    use crate::analysis::tools::{ReaderDb, SharedDb};
    use crate::config::Config;
    use crate::error::Error;
    use crate::storage::workspace::WorkspaceDb;
    use crate::storage::writer::Writer;

    fn config_in(dir: &Path) -> Config {
        let mut config = Config::default();
        config.general.data_dir = dir.join("data");
        config
    }

    #[expect(clippy::unwrap_used, reason = "test")]
    fn holder_reader(config: &Config) -> ReaderDb {
        let db = WorkspaceDb::open(config, "ws").unwrap();
        db.execute_query("CREATE TABLE t AS SELECT 1 AS n UNION ALL SELECT 2")
            .unwrap();
        let shared: SharedDb = Arc::new(Writer::spawn(db).unwrap());
        ReaderDb::new(shared)
    }

    #[expect(clippy::unwrap_used, reason = "test")]
    fn chmod(path: &Path, mode: u32) {
        std::fs::set_permissions(path, std::fs::Permissions::from_mode(mode)).unwrap();
    }

    /// A read goes through the holder and comes back as the holder's rows;
    /// a write is refused naming the holder.
    #[tokio::test]
    #[expect(clippy::unwrap_used, reason = "test")]
    async fn reads_go_through_and_writes_are_refused() {
        let dir = tempfile::tempdir().unwrap();
        let config = config_in(dir.path());
        let reader = holder_reader(&config);
        let listener = Listener::bind(&config.workspace_dir("ws"), &Holder::Serve, reader)
            .await
            .unwrap();
        let client = connect(&config.workspace_dir("ws")).await.unwrap().unwrap();
        assert_eq!(client.holder(), "quack serve");

        let results = client.query("SELECT n FROM t ORDER BY n").await.unwrap();
        assert_eq!(results.columns, vec!["n"]);
        assert_eq!(results.rows.len(), 2);

        let refused = client
            .query("CREATE TABLE u (x INTEGER)")
            .await
            .unwrap_err();
        assert!(
            matches!(&refused, Error::ForwardedWriteRefused { holder } if holder == "quack serve"),
            "{refused}"
        );
        let failed = client.query("SELECT FROM").await.unwrap_err();
        assert!(matches!(failed, Error::Forward(_)), "{failed}");

        let paths = Paths::of(&config.workspace_dir("ws"));
        for path in [&paths.socket, &paths.token] {
            assert_eq!(
                std::fs::metadata(path).unwrap().permissions().mode() & 0o777,
                0o600,
                "{}",
                path.display()
            );
        }
        drop(listener);
        assert!(!paths.socket.exists());
        assert!(!config.workspace_dir("ws").join(TOKEN_FILE).exists());
        assert!(
            connect(&config.workspace_dir("ws"))
                .await
                .unwrap()
                .is_none()
        );
    }

    /// A socket or token file that others can reach is refused, not used;
    /// a token file that is a symlink is refused too.
    #[tokio::test]
    #[expect(clippy::unwrap_used, reason = "test")]
    async fn files_others_can_reach_are_refused() {
        let dir = tempfile::tempdir().unwrap();
        let config = config_in(dir.path());
        let reader = holder_reader(&config);
        let _listener = Listener::bind(&config.workspace_dir("ws"), &Holder::Serve, reader)
            .await
            .unwrap();
        let paths = Paths::of(&config.workspace_dir("ws"));

        chmod(&paths.socket, 0o666);
        let err = connect(&config.workspace_dir("ws")).await.unwrap_err();
        assert!(
            matches!(&err, Error::Forward(message) if message.contains("reachable by others (mode 666)")),
            "{err}"
        );
        chmod(&paths.socket, 0o600);

        chmod(&paths.token, 0o640);
        let err = connect(&config.workspace_dir("ws")).await.unwrap_err();
        assert!(
            matches!(&err, Error::Forward(message) if message.contains("reachable by others (mode 640)")),
            "{err}"
        );
        chmod(&paths.token, 0o600);

        let real = paths.token.with_extension("real");
        std::fs::rename(&paths.token, &real).unwrap();
        std::os::unix::fs::symlink(&real, &paths.token).unwrap();
        let err = connect(&config.workspace_dir("ws")).await.unwrap_err();
        assert!(
            matches!(&err, Error::Forward(message) if message.contains("is not a regular file")),
            "{err}"
        );
        std::fs::remove_file(&paths.token).unwrap();
        std::fs::rename(&real, &paths.token).unwrap();

        assert!(
            connect(&config.workspace_dir("ws"))
                .await
                .unwrap()
                .is_some()
        );
    }

    /// A request whose token is not the holder's is answered with an error
    /// and runs nothing.
    #[tokio::test]
    #[expect(clippy::unwrap_used, reason = "test")]
    async fn a_wrong_token_is_refused() {
        let dir = tempfile::tempdir().unwrap();
        let config = config_in(dir.path());
        let reader = holder_reader(&config);
        let _listener = Listener::bind(&config.workspace_dir("ws"), &Holder::Serve, reader)
            .await
            .unwrap();
        let paths = Paths::of(&config.workspace_dir("ws"));
        let mut record: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&paths.token).unwrap()).unwrap();
        if let Some(fields) = record.as_object_mut() {
            fields.insert(
                String::from("token"),
                serde_json::Value::String(String::from("not-it")),
            );
        }
        std::fs::write(&paths.token, record.to_string()).unwrap();
        chmod(&paths.token, 0o600);
        let client = connect(&config.workspace_dir("ws")).await.unwrap().unwrap();
        let err = client.query("SELECT 1").await.unwrap_err();
        assert!(
            matches!(&err, Error::Forward(message) if message.contains("token")),
            "{err}"
        );
    }

    /// A socket a dead holder of this user left behind is replaced, not an
    /// error; no token file means no holder.
    #[tokio::test]
    #[expect(clippy::unwrap_used, reason = "test")]
    async fn a_stale_socket_is_replaced() {
        let dir = tempfile::tempdir().unwrap();
        let config = config_in(dir.path());
        let reader = holder_reader(&config);
        let socket = config.workspace_dir("ws").join(SOCKET_FILE);
        let stale = tokio::net::UnixListener::bind(&socket).unwrap();
        drop(stale);
        assert!(socket.exists());
        assert!(
            connect(&config.workspace_dir("ws"))
                .await
                .unwrap()
                .is_none()
        );
        let _listener = Listener::bind(&config.workspace_dir("ws"), &Holder::Serve, reader)
            .await
            .unwrap();
        let client = connect(&config.workspace_dir("ws")).await.unwrap().unwrap();
        assert_eq!(client.holder(), "quack serve");
    }
}
