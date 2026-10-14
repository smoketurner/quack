//! Opening the workspace a command runs in: the one `-w` names, which
//! must exist, or the default, created on first use. The binary and the
//! terminal's setup open a workspace through this one type, so every
//! command resolves names, enters the provider egress scope, holds the
//! file the same way (through the process's [`Engine`], the one open
//! path), and serves it on the workspace socket while it holds it, so
//! `quack mcp` and `quack -q` in another shell reach it (`quack_mcp::socket`).

use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result};
use quack_core::analysis::permissions::Approvals;
use quack_core::analysis::policy::WritePolicy;
use quack_core::config::Config;
use quack_core::engine::{Drain, Engine, Holder, Stop, WorkspaceHandle};
use quack_core::error::Error as CoreError;
use quack_core::forward::Listener;
use quack_core::ids::WorkspaceId;
use quack_core::llm::egress::Egress;
use quack_core::storage::control::{ControlPlane, WorkspaceRow};
use quack_core::storage::workspace::WorkspaceDb;
use quack_mcp::client::HeldElsewhere;
use quack_mcp::socket::SocketServer;
use quack_mcp::{McpServer, McpSetup, Transport};
use tokio::sync::OnceCell;

/// The engine and the workspace a command runs in.
pub struct OpenedWorkspace {
    pub engine: Arc<Engine>,
    pub control: ControlPlane,
    pub workspace: WorkspaceRow,
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
        // The command sends only to the model providers the workspace allows.
        Egress::Workspace(workspace.allowed_providers.clone()).enter();
        let engine = Engine::new(config, holder(workspace.id.clone()));
        let approvals = Approvals::with_timeout(engine.config.server.permission_timeout());
        Ok(Self {
            engine,
            control,
            workspace,
            approvals,
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
        HeldElsewhere::find(
            &self.engine.config.workspace_dir(id),
            &self.workspace.id,
            format!(
                "the workspace file {}",
                self.engine.config.workspace_db_path(id).display()
            ),
            self.engine.holder().clone(),
            locked,
        )
        .await
    }

    /// Serve the workspace on its socket for other `quack` processes,
    /// unaudited like stdio. Under the terminal session a `query` turn's
    /// write asks at the session's prompt (`Ask`, whatever `--allow-write`
    /// said: the asker is another process); under a command it is refused.
    /// A socket that cannot be bound is logged once; the command runs
    /// without it.
    async fn serve_socket(&self, handle: &WorkspaceHandle) -> Option<SocketServer> {
        let holder = self.engine.holder();
        if !holder.serves_socket() {
            return None;
        }
        let policy = if matches!(holder, Holder::Terminal(_)) {
            WritePolicy::Ask
        } else {
            WritePolicy::Deny
        };
        let dir = self.engine.config.workspace_dir(self.workspace.id.as_str());
        let listener = match Listener::bind(&dir, &self.workspace.id).await {
            Ok(listener) => listener,
            Err(e) => {
                tracing::warn!(workspace = %self.workspace.id, error = %e, "other quack processes cannot reach this workspace through this one");
                return None;
            }
        };
        let server = McpServer::new(McpSetup::local(
            &self.engine,
            handle.clone(),
            self.workspace.clone(),
            policy,
            Transport::Socket,
            Some(self.approvals.clone()),
        ));
        Some(SocketServer::serve(listener, server))
    }

    /// Whether a statement or call another process sent over the workspace
    /// socket is in progress, so a quit can say what it waits for.
    #[must_use]
    pub fn socket_busy(&self) -> bool {
        self.socket
            .get()
            .and_then(Option::as_ref)
            .is_some_and(SocketServer::has_sessions)
    }

    /// The socket's drain for [`Engine::shutdown`]: stop serving it and
    /// wait up to `grace` for its sessions; the socket file goes with it.
    /// Before the engine closes, so a session in flight ends on an open
    /// workspace. `None` when no socket was served.
    fn socket_drain(&mut self, grace: Duration) -> Option<Drain> {
        let socket = self.socket.take().flatten()?;
        Some(Box::pin(socket.close(grace)))
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
    /// the process exits (`Engine::shutdown`). Every handle must be
    /// dropped first.
    pub async fn finish(self, grace: Duration) {
        self.stop(Stop::Drain, grace, None).await;
    }

    /// The process's stop sequence (`Engine::shutdown`) with the socket's
    /// drain and, for a session that served the web beside itself, the
    /// server's: cancel or drain the jobs as `how` says, end the socket's
    /// sessions and the server's requests under `grace`, then close the
    /// workspace.
    pub async fn stop(mut self, how: Stop, grace: Duration, server: Option<Drain>) {
        let drains = self.socket_drain(grace).into_iter().chain(server).collect();
        self.engine.shutdown(how, grace, drains).await;
    }
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use quack_core::config::Config;
    use quack_core::engine::Holder;
    use quack_core::error::Error as CoreError;
    use quack_core::forward::SocketPath;
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
            opened.workspace.name,
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
