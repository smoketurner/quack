//! Opening the workspace a command runs in: the one `-w` names, which
//! must exist, or the default, created on first use. The binary and the
//! terminal's setup open a workspace through this one type, so every
//! command resolves names, enters the provider egress scope, holds the
//! file the same way (through the process's [`Engine`], the one open
//! path), and serves it on the workspace socket while it holds it, so
//! `quack mcp` and `quack -q` in another shell reach it (`quack_mcp::socket`).

use std::fmt;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result};
use quack_core::analysis::permissions::Approvals;
use quack_core::analysis::policy::WritePolicy;
use quack_core::config::Config;
use quack_core::engine::{Engine, WorkspaceHandle};
use quack_core::error::Error as CoreError;
use quack_core::forward::{Holder, Listener};
use quack_core::ids::WorkspaceId;
use quack_core::llm::egress::Egress;
use quack_core::storage::control::{ControlPlane, WorkspaceRow};
use quack_core::storage::workspace::WorkspaceDb;
use quack_mcp::socket::SocketServer;
use quack_mcp::{McpServer, McpSetup, Transport, client};
use tokio::sync::OnceCell;

/// Why a command could not open its workspace: another `quack` process
/// holds the file. It names the holder when the holder's socket answers
/// (`initialize` names it), and what still works, so the person knows
/// which process to quit or which command to use instead; when no socket
/// answers it says what that means. `Error::WorkspaceLocked` stays its
/// source, so `-q` and `jobs`, which answer a lock by reaching the
/// holder, still recognize it.
#[derive(Debug)]
pub struct HeldElsewhere {
    /// Who holds the file, as its socket named it; `None` when no socket
    /// answered.
    pub holder: Option<String>,
    /// What this process is: the command that was refused.
    wanted: Holder,
    path: PathBuf,
    locked: CoreError,
}

impl fmt::Display for HeldElsewhere {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let Some(holder) = &self.holder else {
            return f.write_str(&client::locked_without_socket(&self.path));
        };
        write!(
            f,
            "the workspace is open in {holder}; `quack -q` and `quack jobs` read through it; \
             quit it to "
        )?;
        match &self.wanted {
            Holder::Command { verb, .. } => write!(f, "run `quack {verb}`"),
            Holder::Terminal(_) => f.write_str("start a session here"),
            Holder::Serve | Holder::Mcp => f.write_str("open it here"),
        }
    }
}

impl std::error::Error for HeldElsewhere {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.locked)
    }
}

/// The engine and the workspace a command runs in.
pub struct OpenedWorkspace {
    pub engine: Arc<Engine>,
    pub control: ControlPlane,
    pub workspace: WorkspaceRow,
    pub name: String,
    /// Where a socket turn's write asks: the terminal session listens
    /// (`SessionSetup::approvals`); for a command nobody does, so a write
    /// is refused.
    pub approvals: Approvals,
    /// The socket server, started by the first `handle()`; `None` when the
    /// socket could not be bound (said once in the log).
    socket: OnceCell<Option<SocketServer>>,
}

impl OpenedWorkspace {
    /// Load the configuration and open the workspace `workspace_name`
    /// selects: the named one, which must exist
    /// (`Error::NoWorkspaceNamed`), or with `None` the
    /// `[general].default_workspace`, created on first use. `holder` names
    /// the command to other processes (`Holder::Terminal`,
    /// `Holder::command("ingest")`).
    ///
    /// # Errors
    ///
    /// Returns an error when the configuration does not load, the control
    /// plane does not open, or the named workspace does not exist.
    pub async fn resolve(
        workspace_name: Option<&str>,
        holder: impl FnOnce(WorkspaceId) -> Holder,
    ) -> Result<Self> {
        let config = Config::load().context("failed to load configuration")?;
        Self::in_config(config, workspace_name, holder).await
    }

    /// The workspace `workspace_name` selects under `config`. Enters the
    /// `Egress` scope for the workspace's allowed providers, so everything
    /// the command sends afterwards goes only to them.
    ///
    /// # Errors
    ///
    /// Returns an error when the control plane does not open or the named
    /// workspace does not exist.
    pub async fn in_config(
        config: Config,
        workspace_name: Option<&str>,
        holder: impl FnOnce(WorkspaceId) -> Holder,
    ) -> Result<Self> {
        let control = ControlPlane::open(&config)
            .await
            .context("failed to open control plane")?;
        let workspace = control
            .workspace_or_default(workspace_name, &config.general.default_workspace)
            .await?;
        let name = workspace.name.clone();
        // The command sends only to the model providers the workspace allows.
        Egress::Workspace(workspace.allowed_providers.clone()).enter();
        let engine = Engine::new(config, holder(workspace.id.clone()));
        Ok(Self {
            engine,
            control,
            workspace,
            name,
            approvals: Approvals::new(),
            socket: OnceCell::new(),
        })
    }

    /// The configuration the engine runs under.
    #[must_use]
    pub fn config(&self) -> &Config {
        &self.engine.config
    }

    /// The workspace's writer, reader pool, and audit connection, the file
    /// opened on first call (`Engine::workspace`).
    ///
    /// # Errors
    ///
    /// Returns [`HeldElsewhere`] when another process holds the file (who,
    /// when its socket answers, and what still works), or an error when the
    /// workspace database does not open (the file is too new, ...).
    pub async fn handle(&self) -> Result<WorkspaceHandle> {
        let handle = match self.engine.workspace(&self.workspace.id).await {
            Ok(handle) => handle,
            Err(locked @ CoreError::WorkspaceLocked { .. }) => {
                return Err(self.held_elsewhere(locked).await.into());
            }
            Err(e) => {
                return Err(anyhow::Error::from(e).context("failed to open workspace database"));
            }
        };
        self.socket.get_or_init(|| self.serve_socket(&handle)).await;
        Ok(handle)
    }

    /// The refusal for a file another process holds: one connection to
    /// the workspace socket asks who, and the answer (or that nothing
    /// answered) goes on the error.
    async fn held_elsewhere(&self, locked: CoreError) -> HeldElsewhere {
        let id = self.workspace.id.as_str();
        let holder = match client::holder_of(
            &self.engine.config.workspace_dir(id),
            &self.workspace.id,
        )
        .await
        {
            Ok(holder) => holder,
            Err(e) => {
                tracing::warn!(error = %e, "could not ask the workspace socket who holds the file");
                None
            }
        };
        HeldElsewhere {
            holder,
            wanted: self.engine.holder().clone(),
            path: self.engine.config.workspace_db_path(id),
            locked,
        }
    }

    /// Serve the workspace on its socket for other `quack` processes,
    /// unaudited like stdio. Under the terminal session a `query` turn's
    /// write asks at the session's prompt (`Ask`, whatever `--allow-write`
    /// said: the asker is another process); under a command it is refused.
    /// A socket that cannot be bound is logged once; the command runs
    /// without it.
    async fn serve_socket(&self, handle: &WorkspaceHandle) -> Option<SocketServer> {
        let policy = match self.engine.holder() {
            Holder::Terminal(_) => WritePolicy::Ask,
            Holder::Command { .. } => WritePolicy::Deny,
            Holder::Serve | Holder::Mcp => return None,
        };
        let dir = self.engine.config.workspace_dir(self.workspace.id.as_str());
        let listener = match Listener::bind(&dir, &self.workspace.id).await {
            Ok(listener) => listener,
            Err(e) => {
                tracing::warn!(workspace = %self.workspace.id, error = %e, "other quack processes cannot reach this workspace through this one");
                return None;
            }
        };
        let server = McpServer::new(McpSetup {
            config: self.engine.config.clone(),
            db: Arc::clone(&handle.writer),
            reader: handle.reader.clone(),
            workspace: self.workspace.clone(),
            policy,
            user_id: None,
            transport: Transport::Socket,
            jobs: self.engine.jobs.clone(),
            stopping: self.engine.stopping(),
            holder: self.engine.holder().to_string(),
            approvals: self.approvals.clone(),
        });
        Some(SocketServer::serve(listener, server))
    }

    /// Stop serving the socket and wait up to `grace` for its sessions;
    /// the socket file goes with it. Before the engine closes, so a
    /// session in flight ends on an open workspace.
    pub async fn close_socket(&mut self, grace: Duration) {
        if let Some(Some(socket)) = self.socket.take() {
            socket.close(grace).await;
        }
    }

    /// Run `f` on the workspace's writer connection and await it: what a
    /// one-shot command does with the database, in one step on the
    /// writer's thread (a `-q` statement, a listing written to stdout).
    ///
    /// # Errors
    ///
    /// Returns the open's error or `f`'s own.
    pub async fn with_db<T, F>(&self, f: F) -> Result<T>
    where
        T: Send + 'static,
        F: FnOnce(&WorkspaceDb) -> Result<T> + Send + 'static,
    {
        self.handle().await?.writer.run(move |db| Ok(f(db))).await?
    }

    /// Let the engine's jobs end under `grace` (a labelling or a title the
    /// command started finishes; nothing is cancelled), end the socket's
    /// sessions, and close the workspace, so the writer checkpoints before
    /// the process exits. Every handle must be dropped first.
    pub async fn finish(mut self, grace: Duration) {
        self.engine.drain_jobs(grace).await;
        self.close_socket(grace).await;
        self.engine.close().await;
    }
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use quack_core::config::Config;
    use quack_core::error::Error as CoreError;
    use quack_core::forward::{Holder, SocketPath};
    use quack_core::storage::control::ControlPlane;

    use super::OpenedWorkspace;

    fn config_in(dir: &Path) -> Config {
        let mut config = Config::default();
        config.general.data_dir = dir.join("data");
        config
    }

    /// `-w` with a name no workspace has is the typed not-found error that
    /// says how to create it, and it creates nothing.
    #[tokio::test]
    #[expect(clippy::unwrap_used, reason = "test")]
    async fn an_unknown_workspace_is_refused_and_creates_nothing() {
        let dir = tempfile::tempdir().unwrap();
        let config = config_in(dir.path());
        let refused = OpenedWorkspace::in_config(config.clone(), Some("slaes"), Holder::Terminal)
            .await
            .err()
            .unwrap();
        assert_eq!(
            format!("{refused:#}"),
            "no workspace named 'slaes'; create it with: quack workspace create slaes"
        );
        assert!(matches!(
            refused.downcast_ref::<CoreError>(),
            Some(CoreError::NoWorkspaceNamed(name)) if name == "slaes"
        ));
        let control = ControlPlane::open(&config).await.unwrap();
        assert!(control.list_workspaces().await.unwrap().is_empty());
    }

    /// With no `-w` the default workspace is created on first use, held by
    /// the command's engine, and reached through one writer.
    #[tokio::test]
    #[expect(clippy::unwrap_used, reason = "test")]
    async fn the_default_workspace_is_created_and_held() {
        let dir = tempfile::tempdir().unwrap();
        let opened = OpenedWorkspace::in_config(config_in(dir.path()), None, Holder::command("-q"))
            .await
            .unwrap();
        assert_eq!(
            opened.name,
            opened.config().general.default_workspace.to_string()
        );
        let n = opened
            .with_db(|db| Ok(db.execute_query("SELECT 1 AS n")?.rows.len()))
            .await
            .unwrap();
        assert_eq!(n, 1);
        let first = opened.handle().await.unwrap();
        let second = opened.handle().await.unwrap();
        assert!(std::sync::Arc::ptr_eq(&first.writer, &second.writer));
        drop((first, second));
        // The first handle started the socket; finishing at once, as every
        // one-shot command does, takes the socket file with it.
        let socket = SocketPath::of(
            &opened.config().workspace_dir(opened.workspace.id.as_str()),
            &opened.workspace.id,
        );
        assert!(socket.path().exists(), "the command serves its workspace");
        opened.finish(std::time::Duration::ZERO).await;
        assert!(!socket.path().exists(), "the socket file was left behind");
    }
}
