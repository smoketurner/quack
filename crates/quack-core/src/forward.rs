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
//! One request per connection, one JSON object per line each way. The
//! module is Unix-only; elsewhere [`connect`] finds no holder and
//! [`Listener::bind`] listens on nothing.

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use crate::config::Config;
use crate::error::Error;
use crate::storage::workspace::QueryResults;

/// The socket a workspace's holder listens on, inside the workspace
/// directory.
pub const SOCKET_FILE: &str = "quack.sock";
/// The token a request must carry, beside the socket, mode 0600.
pub const TOKEN_FILE: &str = "quack.sock.token";
/// The most bytes one request line may run to.
const MAX_REQUEST_BYTES: u64 = 1 << 20;

/// What the token file holds: the token, and who listens.
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

#[derive(Debug, Serialize, Deserialize)]
struct Envelope {
    token: String,
    request: Request,
}

/// Where a workspace's socket and token live.
#[derive(Debug, Clone)]
struct Paths {
    socket: PathBuf,
    token: PathBuf,
}

impl Paths {
    fn of(config: &Config, workspace_id: &str) -> Self {
        let dir = config.workspace_dir(workspace_id);
        Self {
            socket: dir.join(SOCKET_FILE),
            token: dir.join(TOKEN_FILE),
        }
    }

    fn read_token(&self) -> Option<TokenFile> {
        let text = std::fs::read_to_string(&self.token).ok()?;
        serde_json::from_str(&text).ok()
    }
}

#[cfg(unix)]
mod unix {
    use std::io::Write as _;
    use std::os::unix::fs::{OpenOptionsExt, PermissionsExt};
    use std::path::Path;
    use std::sync::Arc;

    use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader};
    use tokio::net::{UnixListener, UnixStream};

    use super::{Envelope, MAX_REQUEST_BYTES, Paths, Reply, Request, TokenFile};
    use crate::analysis::tools::ReaderDb;
    use crate::config::Config;
    use crate::error::{Error, Result};
    use crate::storage::control::random_bytes;
    use crate::storage::workspace::{QueryResults, StatementKind};

    /// The holder's end: the socket, the token file, and the accept loop.
    /// Dropping it stops listening and removes both files.
    #[derive(Debug)]
    pub struct Listener {
        paths: Paths,
        accept: tokio::task::JoinHandle<()>,
    }

    impl Listener {
        /// Listen for the workspace's reads as `holder` (the name a refused
        /// writer is told), answering them on `reader`. A socket left by a
        /// holder that died is replaced; one that answers is an error, since
        /// two holders cannot both have the file open.
        ///
        /// # Errors
        ///
        /// Returns an error when the socket cannot be bound, the token cannot
        /// be written, or another process already answers on the socket.
        pub async fn bind(
            config: &Config,
            workspace_id: &str,
            holder: &str,
            reader: ReaderDb,
        ) -> Result<Self> {
            let paths = Paths::of(config, workspace_id);
            if paths.socket.exists() {
                if UnixStream::connect(&paths.socket).await.is_ok() {
                    return Err(Error::Forward {
                        holder: holder.to_owned(),
                        message: format!(
                            "another process already answers on {}",
                            paths.socket.display()
                        ),
                    });
                }
                std::fs::remove_file(&paths.socket)?;
            }
            let listener = UnixListener::bind(&paths.socket)?;
            std::fs::set_permissions(&paths.socket, std::fs::Permissions::from_mode(0o600))?;
            let token = Self::write_token(&paths.token, holder)?;
            let token: Arc<str> = Arc::from(token);
            let holder: Arc<str> = Arc::from(holder);
            let accept = tokio::spawn(async move {
                loop {
                    match listener.accept().await {
                        Ok((stream, _)) => {
                            let (token, holder, reader) =
                                (Arc::clone(&token), Arc::clone(&holder), reader.clone());
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
            Ok(Self { paths, accept })
        }

        /// A fresh token in a 0600 file, replacing any left behind.
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

    impl Drop for Listener {
        fn drop(&mut self) {
            self.accept.abort();
            for path in [&self.paths.socket, &self.paths.token] {
                if let Err(e) = std::fs::remove_file(path)
                    && e.kind() != std::io::ErrorKind::NotFound
                {
                    tracing::warn!(path = %path.display(), error = %e, "could not remove the forwarding file");
                }
            }
        }
    }

    /// Answer one connection: one request line, one reply line.
    async fn serve_one(
        stream: UnixStream,
        token: &str,
        holder: &str,
        reader: &ReaderDb,
    ) -> Result<()> {
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

    /// The other end: a connection to the holder of `workspace_id`, when one
    /// listens. `None` when there is no token file, or nothing answers on
    /// the socket (a holder that died leaves both behind).
    ///
    /// # Errors
    ///
    /// Returns an error when the token file exists but cannot be read.
    pub async fn connect(config: &Config, workspace_id: &str) -> Result<Option<Client>> {
        let paths = Paths::of(config, workspace_id);
        let Some(token) = paths.read_token() else {
            return Ok(None);
        };
        if UnixStream::connect(&paths.socket).await.is_err() {
            return Ok(None);
        }
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
        /// write, [`Error::Forward`] when the holder failed or the protocol
        /// broke, and an I/O error when the socket is gone.
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
                Reply::Error { message } => Err(self.failed(message)),
                Reply::Hello { .. } => Err(self.failed(String::from("unexpected reply"))),
            }
        }

        async fn send(&self, request: Request) -> Result<Reply> {
            let stream = UnixStream::connect(&self.paths.socket).await?;
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
                return Err(self.failed(String::from("the holder closed without answering")));
            }
            serde_json::from_str(&line).map_err(|e| self.failed(format!("malformed reply: {e}")))
        }

        fn failed(&self, message: String) -> Error {
            Error::Forward {
                holder: self.holder.clone(),
                message,
            }
        }
    }
}

#[cfg(unix)]
pub use unix::{Client, Listener, connect};

#[cfg(not(unix))]
mod other {
    use crate::analysis::tools::ReaderDb;
    use crate::config::Config;
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
            _config: &Config,
            _workspace_id: &str,
            _holder: &str,
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
            Err(crate::error::Error::Forward {
                holder: self.holder.clone(),
                message: String::from("read forwarding needs Unix sockets"),
            })
        }
    }

    /// No holder is ever found on this platform.
    ///
    /// # Errors
    ///
    /// Never fails.
    pub async fn connect(_config: &Config, _workspace_id: &str) -> Result<Option<Client>> {
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

/// The socket path a holder of `workspace_id` would listen on.
#[must_use]
pub fn socket_path(config: &Config, workspace_id: &str) -> PathBuf {
    Paths::of(config, workspace_id).socket
}

#[cfg(all(test, unix))]
mod tests {
    use std::path::Path;
    use std::sync::Arc;

    use super::{Listener, Paths, SOCKET_FILE, TOKEN_FILE, connect};
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

    /// A read goes through the holder and comes back as the holder's rows;
    /// a write is refused naming the holder.
    #[tokio::test]
    #[expect(clippy::unwrap_used, reason = "test")]
    async fn reads_go_through_and_writes_are_refused() {
        let dir = tempfile::tempdir().unwrap();
        let config = config_in(dir.path());
        let reader = holder_reader(&config);
        let listener = Listener::bind(&config, "ws", "the test holder", reader)
            .await
            .unwrap();
        let client = connect(&config, "ws").await.unwrap().unwrap();
        assert_eq!(client.holder(), "the test holder");

        let results = client.query("SELECT n FROM t ORDER BY n").await.unwrap();
        assert_eq!(results.columns, vec!["n"]);
        assert_eq!(results.rows.len(), 2);

        let refused = client
            .query("CREATE TABLE u (x INTEGER)")
            .await
            .unwrap_err();
        assert!(
            matches!(&refused, Error::ForwardedWriteRefused { holder } if holder == "the test holder"),
            "{refused}"
        );
        let failed = client.query("SELECT FROM").await.unwrap_err();
        assert!(matches!(failed, Error::Forward { .. }), "{failed}");

        let socket = Paths::of(&config, "ws").socket;
        assert_eq!(
            std::os::unix::fs::PermissionsExt::mode(
                &std::fs::metadata(&socket).unwrap().permissions()
            ) & 0o777,
            0o600
        );
        drop(listener);
        assert!(!socket.exists());
        assert!(!config.workspace_dir("ws").join(TOKEN_FILE).exists());
        assert!(connect(&config, "ws").await.unwrap().is_none());
    }

    /// A request whose token is not the holder's is answered with an error
    /// and runs nothing.
    #[tokio::test]
    #[expect(clippy::unwrap_used, reason = "test")]
    async fn a_wrong_token_is_refused() {
        let dir = tempfile::tempdir().unwrap();
        let config = config_in(dir.path());
        let reader = holder_reader(&config);
        let _listener = Listener::bind(&config, "ws", "holder", reader)
            .await
            .unwrap();
        let paths = Paths::of(&config, "ws");
        let mut record: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&paths.token).unwrap()).unwrap();
        record["token"] = serde_json::Value::String(String::from("not-it"));
        std::fs::write(&paths.token, record.to_string()).unwrap();
        let client = connect(&config, "ws").await.unwrap().unwrap();
        let err = client.query("SELECT 1").await.unwrap_err();
        assert!(
            matches!(&err, Error::Forward { message, .. } if message.contains("token")),
            "{err}"
        );
    }

    /// A socket a dead holder left behind is replaced, not an error; no
    /// token file means no holder.
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
        assert!(connect(&config, "ws").await.unwrap().is_none());
        let _listener = Listener::bind(&config, "ws", "holder", reader)
            .await
            .unwrap();
        let client = connect(&config, "ws").await.unwrap().unwrap();
        assert_eq!(client.holder(), "holder");
    }
}
