//! The socket a workspace's holder serves the workspace on (design doc
//! 4.1 and 11.3).
//!
//! `DuckDB` allows one process with read-write access to a file, or any
//! number of read-only processes, never a writer beside readers: a second
//! `quack` cannot open a workspace the terminal session or a command
//! holds, even for `SELECT 1`. There is one lock and one `DuckDB` instance
//! per workspace, and the way through it is to reach the holder.
//!
//! Whichever process holds a workspace open also listens on a Unix socket
//! ([`SocketPath`]) and serves MCP on it, nothing else: `quack mcp` relays
//! a client's requests to it, and `quack -q` calls its `sql` tool
//! (`quack_mcp`). The socket goes when the workspace closes. `quack serve`
//! serves no socket: a login-mode server would hand any same-user process
//! the whole agent with no audit row.
//!
//! Neither end trusts the file or the peer on faith. A client uses the
//! socket only when it is a socket, owned by the current user, and
//! unreachable by anyone else; both ends check the peer's credentials and
//! refuse a process running as another user; a holder never removes a
//! stale socket that is not its own user's.
//!
//! This module is the transport: who binds, who connects, and the checks.
//! The module is Unix-only; elsewhere [`connect`] finds no holder and
//! [`Listener::bind`] fails.

use std::path::{Path, PathBuf};

use crate::error::Error;
use crate::ids::WorkspaceId;

/// The process that holds workspaces open, as the MCP `initialize` reply
/// names it to the person reading `quack -q`'s stderr, and the one
/// workspace it holds: `quack serve` opens any workspace a request names;
/// the terminal session and every command opened exactly one and refuse
/// the rest (`engine::Engine::workspace`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Holder {
    Serve,
    Terminal(WorkspaceId),
    /// A `quack` command, named by its verb (`-p`, `ingest`, `mcp`, ...).
    Command {
        workspace: WorkspaceId,
        verb: &'static str,
    },
}

impl Holder {
    /// The holder a command with `verb` becomes once its workspace is
    /// known: `OpenedWorkspace::resolve(name, Holder::command("ingest"))`.
    pub fn command(verb: &'static str) -> impl FnOnce(WorkspaceId) -> Self {
        move |workspace| Self::Command { workspace, verb }
    }

    /// The one workspace this holder opened, or `None` for the server.
    #[must_use]
    pub fn held(&self) -> Option<&WorkspaceId> {
        match self {
            Self::Serve => None,
            Self::Terminal(id) | Self::Command { workspace: id, .. } => Some(id),
        }
    }
}

impl std::fmt::Display for Holder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Serve => f.write_str("quack serve"),
            Self::Terminal(_) => f.write_str("the quack terminal session"),
            Self::Command { verb, .. } => write!(f, "quack {verb}"),
        }
    }
}

/// The socket a workspace's holder listens on, inside the workspace
/// directory.
pub const SOCKET_FILE: &str = "quack.sock";

/// Where a workspace's socket is. A Unix socket address holds about a
/// hundred bytes of path (104 on macOS, 108 on Linux), and a data
/// directory under a long path leaves no room beside the workspace file,
/// so the socket then goes under a per-user directory in the temp
/// directory instead. One rule, worked out by both ends from the same
/// inputs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SocketPath {
    /// `{workspace_dir}/quack.sock`.
    Beside(PathBuf),
    /// `{temp_dir}/quack-{uid}/{workspace_id}.sock`, since the workspace
    /// directory's path is too long for a socket address.
    Fallback(PathBuf),
}

impl SocketPath {
    /// The socket for the workspace in `workspace_dir`
    /// (`Config::workspace_dir`).
    #[must_use]
    pub fn of(workspace_dir: &Path, workspace_id: &WorkspaceId) -> Self {
        let beside = workspace_dir.join(SOCKET_FILE);
        if unix::fits_socket_address(&beside) {
            return Self::Beside(beside);
        }
        Self::Fallback(unix::fallback_dir().join(format!("{workspace_id}.sock")))
    }

    /// The socket file.
    #[must_use]
    pub fn path(&self) -> &Path {
        match self {
            Self::Beside(path) | Self::Fallback(path) => path,
        }
    }

    /// The directory a fallback socket is in, which the holder creates
    /// private to this user; `None` beside the workspace file.
    #[must_use]
    pub fn fallback_dir(&self) -> Option<&Path> {
        match self {
            Self::Beside(_) => None,
            Self::Fallback(path) => path.parent(),
        }
    }
}

#[cfg(unix)]
mod unix {
    use std::os::unix::fs::{DirBuilderExt, FileTypeExt, MetadataExt, PermissionsExt};
    use std::path::{Path, PathBuf};

    use tokio::net::{UnixListener, UnixStream};

    use super::SocketPath;
    use crate::error::{Error, Result};
    use crate::ids::WorkspaceId;

    /// What a refusal names as the holder when the other end is not who it
    /// should be.
    const SOMEONE: &str = "another process";

    /// The user this process runs as.
    fn current_uid() -> u32 {
        rustix::process::getuid().as_raw()
    }

    /// Whether `path` fits a Unix socket address on this platform.
    pub(super) fn fits_socket_address(path: &Path) -> bool {
        rustix::net::SocketAddrUnix::new(path).is_ok()
    }

    /// The per-user directory a fallback socket goes in.
    pub(super) fn fallback_dir() -> PathBuf {
        std::env::temp_dir().join(format!("quack-{}", current_uid()))
    }

    /// Which kind of file a path must be to be used.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum Kind {
        Socket,
        Directory,
    }

    fn refuse(path: &Path, what: &str) -> Error {
        Error::Forward(format!(
            "{} {what}; refusing to use it (the workspace is open in {SOMEONE})",
            path.display()
        ))
    }

    /// `path` is a `kind` file this user owns. Symlinks are not followed, so
    /// a link planted at the path is refused rather than resolved.
    fn check_owned(path: &Path, kind: Kind) -> Result<std::fs::Metadata> {
        let meta = std::fs::symlink_metadata(path)?;
        let is_kind = match kind {
            Kind::Socket => meta.file_type().is_socket(),
            Kind::Directory => meta.file_type().is_dir(),
        };
        if !is_kind {
            let want = match kind {
                Kind::Socket => "a socket",
                Kind::Directory => "a directory",
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

    /// A connection to or from a holder.
    pub type Stream = UnixStream;

    /// The holder's end: the bound socket. Dropping it stops listening
    /// and removes the file.
    pub struct Listener {
        at: SocketPath,
        listener: UnixListener,
    }

    impl Listener {
        /// Listen for the workspace in `workspace_dir`
        /// (`Config::workspace_dir`). A socket left by a holder of this
        /// user that died is replaced; one that answers is an error, since
        /// two holders cannot both have the file open; one that another
        /// user owns is an error and is left alone.
        ///
        /// # Errors
        ///
        /// Returns an error when the socket cannot be bound, another
        /// process already answers on it, or a file in the way is not this
        /// user's.
        pub async fn bind(workspace_dir: &Path, workspace_id: &WorkspaceId) -> Result<Self> {
            let at = SocketPath::of(workspace_dir, workspace_id);
            if let Some(dir) = at.fallback_dir() {
                let created = std::fs::DirBuilder::new().mode(0o700).create(dir);
                if let Err(e) = created
                    && e.kind() != std::io::ErrorKind::AlreadyExists
                {
                    return Err(Error::Io(e));
                }
                check_private(dir, Kind::Directory)?;
            }
            let path = at.path();
            if path.exists() {
                check_owned(path, Kind::Socket)?;
                if UnixStream::connect(path).await.is_ok() {
                    return Err(Error::Forward(format!(
                        "another process already answers on {}",
                        path.display()
                    )));
                }
                std::fs::remove_file(path)?;
            }
            let listener = UnixListener::bind(path)?;
            std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600))?;
            Ok(Self { at, listener })
        }

        /// Where it listens.
        #[must_use]
        pub fn at(&self) -> &SocketPath {
            &self.at
        }

        /// The next connection, from a process running as this user. One
        /// from another user is refused here and never seen by the caller.
        ///
        /// # Errors
        ///
        /// Returns the accept's own error.
        pub async fn accept(&self) -> Result<UnixStream> {
            loop {
                let (stream, _) = self.listener.accept().await?;
                match check_peer(&stream) {
                    Ok(()) => return Ok(stream),
                    Err(e) => {
                        tracing::warn!(error = %e, "refused a connection to the workspace socket");
                    }
                }
            }
        }
    }

    impl std::fmt::Debug for Listener {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.debug_struct("Listener")
                .field("at", &self.at)
                .finish_non_exhaustive()
        }
    }

    impl Drop for Listener {
        fn drop(&mut self) {
            if let Err(e) = std::fs::remove_file(self.at.path())
                && e.kind() != std::io::ErrorKind::NotFound
            {
                tracing::warn!(path = %self.at.path().display(), error = %e, "could not remove the workspace socket");
            }
        }
    }

    /// The other end: a connection to the holder of the workspace in
    /// `workspace_dir`, when one listens. `None` when there is no socket
    /// or nothing answers on it (a holder that died leaves it behind).
    ///
    /// # Errors
    ///
    /// Returns an error when the socket or its directory exists but is not
    /// a private file of this user's, or the process answering runs as
    /// another user.
    pub async fn connect(
        workspace_dir: &Path,
        workspace_id: &WorkspaceId,
    ) -> Result<Option<UnixStream>> {
        let at = SocketPath::of(workspace_dir, workspace_id);
        if let Some(dir) = at.fallback_dir() {
            if !dir.exists() {
                return Ok(None);
            }
            check_private(dir, Kind::Directory)?;
        }
        let path = at.path();
        if !path.exists() {
            return Ok(None);
        }
        check_private(path, Kind::Socket)?;
        let Ok(stream) = UnixStream::connect(path).await else {
            return Ok(None);
        };
        check_peer(&stream)?;
        Ok(Some(stream))
    }
}

#[cfg(unix)]
pub use unix::{Listener, Stream, connect};

#[cfg(not(unix))]
mod unix {
    use std::path::{Path, PathBuf};
    use std::pin::Pin;
    use std::task::{Context, Poll};

    use tokio::io::{AsyncRead, AsyncWrite, ReadBuf};

    use crate::error::{Error, Result};
    use crate::ids::WorkspaceId;

    /// A connection to a holder: never made on this platform.
    #[derive(Debug)]
    pub enum Stream {}

    impl AsyncRead for Stream {
        fn poll_read(
            self: Pin<&mut Self>,
            _cx: &mut Context<'_>,
            _buf: &mut ReadBuf<'_>,
        ) -> Poll<std::io::Result<()>> {
            match *self {}
        }
    }

    impl AsyncWrite for Stream {
        fn poll_write(
            self: Pin<&mut Self>,
            _cx: &mut Context<'_>,
            _buf: &[u8],
        ) -> Poll<std::io::Result<usize>> {
            match *self {}
        }

        fn poll_flush(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<std::io::Result<()>> {
            match *self {}
        }

        fn poll_shutdown(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<std::io::Result<()>> {
            match *self {}
        }
    }

    /// Every path fits: nothing is ever bound here.
    pub(super) fn fits_socket_address(_path: &Path) -> bool {
        true
    }

    pub(super) fn fallback_dir() -> PathBuf {
        std::env::temp_dir()
    }

    /// No holder listens on this platform.
    #[derive(Debug)]
    pub struct Listener;

    impl Listener {
        /// Unix sockets are the only transport so far.
        ///
        /// # Errors
        ///
        /// Always fails.
        pub async fn bind(_workspace_dir: &Path, _workspace_id: &WorkspaceId) -> Result<Self> {
            Err(Error::Forward(String::from(
                "the workspace socket needs Unix sockets",
            )))
        }
    }

    /// No holder is ever found on this platform.
    ///
    /// # Errors
    ///
    /// Never fails.
    pub async fn connect(
        _workspace_dir: &Path,
        _workspace_id: &WorkspaceId,
    ) -> Result<Option<Stream>> {
        Ok(None)
    }
}

#[cfg(not(unix))]
pub use unix::{Listener, Stream, connect};

/// Whether `err` is the lock another process holds on the workspace file:
/// the case a command answers by reaching the holder.
#[must_use]
pub fn is_locked(err: &Error) -> bool {
    matches!(err, Error::WorkspaceLocked { .. })
}

#[cfg(all(test, unix))]
mod tests {
    use std::os::unix::fs::PermissionsExt;
    use std::path::Path;

    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    use super::{Listener, SOCKET_FILE, SocketPath, connect};
    use crate::error::Error;
    use crate::ids::WorkspaceId;

    fn id() -> WorkspaceId {
        WorkspaceId::from("0199d1f0-0000-7000-8000-000000000001")
    }

    #[expect(clippy::unwrap_used, reason = "test")]
    fn chmod(path: &Path, mode: u32) {
        std::fs::set_permissions(path, std::fs::Permissions::from_mode(mode)).unwrap();
    }

    /// A connection reaches the holder and carries bytes both ways; the
    /// socket is private and goes with the listener.
    #[tokio::test]
    #[expect(clippy::unwrap_used, reason = "test")]
    async fn a_connection_reaches_the_holder() {
        let dir = tempfile::tempdir().unwrap();
        let listener = Listener::bind(dir.path(), &id()).await.unwrap();
        let at = listener.at().clone();
        assert_eq!(at, SocketPath::Beside(dir.path().join(SOCKET_FILE)));
        assert_eq!(
            std::fs::metadata(at.path()).unwrap().permissions().mode() & 0o777,
            0o600
        );
        let echo = tokio::spawn(async move {
            let mut stream = listener.accept().await.unwrap();
            let mut line = [0_u8; 5];
            stream.read_exact(&mut line).await.unwrap();
            stream.write_all(&line).await.unwrap();
            listener
        });
        let mut client = connect(dir.path(), &id()).await.unwrap().unwrap();
        client.write_all(b"hello").await.unwrap();
        let mut back = [0_u8; 5];
        client.read_exact(&mut back).await.unwrap();
        assert_eq!(&back, b"hello");
        let listener = echo.await.unwrap();
        drop(listener);
        assert!(!at.path().exists());
        assert!(connect(dir.path(), &id()).await.unwrap().is_none());
    }

    /// A socket that others can reach is refused, not used.
    #[tokio::test]
    #[expect(clippy::unwrap_used, reason = "test")]
    async fn a_socket_others_can_reach_is_refused() {
        let dir = tempfile::tempdir().unwrap();
        let listener = Listener::bind(dir.path(), &id()).await.unwrap();
        chmod(listener.at().path(), 0o666);
        let err = connect(dir.path(), &id()).await.unwrap_err();
        assert!(
            matches!(&err, Error::Forward(message) if message.contains("reachable by others (mode 666)")),
            "{err}"
        );
        chmod(listener.at().path(), 0o600);
        assert!(connect(dir.path(), &id()).await.unwrap().is_some());
    }

    /// A socket a dead holder of this user left behind is replaced, not an
    /// error; one a live holder answers on is an error.
    #[tokio::test]
    #[expect(clippy::unwrap_used, reason = "test")]
    async fn a_stale_socket_is_replaced_and_a_live_one_refused() {
        let dir = tempfile::tempdir().unwrap();
        let socket = dir.path().join(SOCKET_FILE);
        let stale = tokio::net::UnixListener::bind(&socket).unwrap();
        drop(stale);
        chmod(&socket, 0o600);
        assert!(socket.exists());
        assert!(connect(dir.path(), &id()).await.unwrap().is_none());
        let listener = Listener::bind(dir.path(), &id()).await.unwrap();
        let err = Listener::bind(dir.path(), &id()).await.unwrap_err();
        assert!(
            matches!(&err, Error::Forward(message) if message.contains("already answers")),
            "{err}"
        );
        drop(listener);
    }

    /// A workspace directory whose path is too long for a socket address
    /// gets its socket in the per-user temp directory, found by both ends.
    #[tokio::test]
    #[expect(clippy::unwrap_used, reason = "test")]
    async fn a_long_path_falls_back_to_the_temp_directory() {
        let dir = tempfile::tempdir().unwrap();
        let long = dir.path().join("x".repeat(120));
        std::fs::create_dir(&long).unwrap();
        let at = SocketPath::of(&long, &id());
        assert!(matches!(at, SocketPath::Fallback(_)), "{at:?}");
        assert!(at.path().ends_with(format!("{}.sock", id())));
        let listener = Listener::bind(&long, &id()).await.unwrap();
        assert_eq!(listener.at(), &at);
        assert_eq!(
            std::fs::metadata(at.fallback_dir().unwrap())
                .unwrap()
                .permissions()
                .mode()
                & 0o777,
            0o700
        );
        assert!(connect(&long, &id()).await.unwrap().is_some());
        drop(listener);
        assert!(!at.path().exists());
    }
}
