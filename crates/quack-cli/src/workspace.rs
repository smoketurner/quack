//! Opening the workspace a command runs in: the one `-w` names, which
//! must exist, or the default, created on first use. The binary and the
//! terminal's setup open a workspace through this one type, so every
//! interface resolves names, enters the provider egress scope, and holds
//! the file the same way: a one-shot command on its own connection, and a
//! command that runs jobs through the process's [`Engine`].

use std::sync::Arc;

use anyhow::{Context, Result};
use quack_core::config::Config;
use quack_core::engine::Engine;
use quack_core::forward::Holder;
use quack_core::ids::WorkspaceId;
use quack_core::llm::egress::Egress;
use quack_core::storage::control::{ControlPlane, WorkspaceRow};
use quack_core::storage::workspace::WorkspaceDb;
use quack_core::storage::writer::Writer;

/// The configuration and the workspace a command runs in.
pub struct OpenedWorkspace {
    pub config: Config,
    pub control: ControlPlane,
    pub workspace: WorkspaceRow,
    pub name: String,
}

/// A workspace held by the process's engine: the one way to open the file
/// for a command that runs jobs (`-p`, `mcp`, `search`, `saved run`, the
/// terminal). It consumes the [`OpenedWorkspace`], so nothing can open
/// the same file a second time beside the engine's handle.
pub struct HeldWorkspace {
    pub engine: Arc<Engine>,
    pub control: ControlPlane,
    pub workspace: WorkspaceRow,
    pub name: String,
}

impl OpenedWorkspace {
    /// Load the configuration and open the workspace `workspace_name`
    /// selects: the named one, which must exist
    /// (`Error::NoWorkspaceNamed`), or with `None` the
    /// `[general].default_workspace`, created on first use.
    ///
    /// # Errors
    ///
    /// Returns an error when the configuration does not load, the control
    /// plane does not open, or the named workspace does not exist.
    pub async fn resolve(workspace_name: Option<&str>) -> Result<Self> {
        let config = Config::load().context("failed to load configuration")?;
        Self::in_config(config, workspace_name).await
    }

    /// The workspace `workspace_name` selects under `config`. Enters the
    /// `Egress` scope for the workspace's allowed providers, so everything
    /// the command sends afterwards goes only to them.
    ///
    /// # Errors
    ///
    /// Returns an error when the control plane does not open or the named
    /// workspace does not exist.
    pub async fn in_config(config: Config, workspace_name: Option<&str>) -> Result<Self> {
        let control = ControlPlane::open(&config)
            .await
            .context("failed to open control plane")?;
        let workspace = control
            .workspace_or_default(workspace_name, &config.general.default_workspace)
            .await?;
        let name = workspace.name.clone();
        // The command sends only to the model providers the workspace allows.
        Egress::Workspace(workspace.allowed_providers.clone()).enter();
        Ok(Self {
            config,
            control,
            workspace,
            name,
        })
    }

    /// Hold the workspace through an engine as `holder` (`Holder::Print`,
    /// `Holder::Terminal`, ...), the holder named after this workspace.
    #[must_use]
    pub fn into_engine(self, holder: fn(WorkspaceId) -> Holder) -> HeldWorkspace {
        let Self {
            config,
            control,
            workspace,
            name,
        } = self;
        let engine = Engine::new(config, holder(workspace.id.clone()));
        HeldWorkspace {
            engine,
            control,
            workspace,
            name,
        }
    }

    /// The workspace's connection.
    ///
    /// # Errors
    ///
    /// Returns an error when the workspace database does not open.
    pub fn open_db(&self) -> Result<WorkspaceDb> {
        WorkspaceDb::open(&self.config, self.workspace.id.as_str())
            .context("failed to open workspace database")
    }

    /// The workspace's connection on a writer thread of its own: the
    /// commands that run core's multi-step work (ingestion, import, graph
    /// and ontology commands) send it their steps exactly as the server
    /// and the terminal do.
    ///
    /// # Errors
    ///
    /// Returns an error when the database does not open or the writer
    /// thread does not start.
    pub fn writer(&self) -> Result<Writer> {
        Writer::spawn(self.open_db()?).context("failed to start the workspace writer")
    }
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use quack_core::config::Config;
    use quack_core::error::Error as CoreError;
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
        let refused = OpenedWorkspace::in_config(config.clone(), Some("slaes"))
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

    /// With no `-w`, a new data directory gets the default workspace, and
    /// naming it afterwards opens the same one.
    #[tokio::test]
    #[expect(clippy::unwrap_used, reason = "test")]
    async fn no_workspace_name_creates_the_default_on_a_fresh_data_dir() {
        let dir = tempfile::tempdir().unwrap();
        let config = config_in(dir.path());
        let opened = OpenedWorkspace::in_config(config.clone(), None)
            .await
            .unwrap();
        assert_eq!(opened.name, "default");
        opened.open_db().unwrap();
        let again = OpenedWorkspace::in_config(config, Some("default"))
            .await
            .unwrap();
        assert_eq!(again.workspace.id, opened.workspace.id);
    }
}
