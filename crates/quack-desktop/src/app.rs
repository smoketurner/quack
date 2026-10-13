//! The process's one state, managed by Tauri: the configuration, the
//! control plane, the job queue, the workspaces the app holds open, and
//! what each running turn needs to be cancelled (design 3.1).

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex, PoisonError};

use anyhow::{Context, Result};
use quack_cli::OpenedWorkspace;
use quack_core::analysis::tools::{ReaderDb, SharedDb};
use quack_core::config::Config;
use quack_core::forward::Listener;
use quack_core::ids::{SessionId, WorkspaceId};
use quack_core::jobs::{JobId, JobQueue};
use quack_core::llm::egress::Egress;
use quack_core::storage::audit::AuditLog;
use quack_core::storage::control::{ControlPlane, WorkspaceRow};

use crate::audit::Auditor;
use crate::error::{CommandError, CommandResult};

/// What the refused writer of another process is told holds the file.
pub(crate) const HOLDER: &str = "the quack desktop app";

/// A workspace the app holds open: what `OpenedWorkspace::shared`
/// produces, plus the audit connection and the socket other processes'
/// reads come in on. Dropping it closes the file (the writer checkpoints)
/// and removes the socket.
pub(crate) struct OpenWorkspace {
    pub(crate) row: WorkspaceRow,
    pub(crate) db: SharedDb,
    pub(crate) reader: ReaderDb,
    pub(crate) audit: AuditLog,
    _forward: Option<Listener>,
}

impl OpenWorkspace {
    /// The providers this workspace's model requests may reach.
    pub(crate) fn egress(&self) -> Egress {
        Egress::Workspace(self.row.allowed_providers.clone())
    }
}

/// The app.
pub(crate) struct AppState {
    pub(crate) config: Config,
    pub(crate) control: ControlPlane,
    pub(crate) jobs: JobQueue,
    pub(crate) auditor: Auditor,
    /// Opened once per process and shared by every command; `DuckDB`'s
    /// lock is per process, so a second open would fork the database.
    workspaces: tokio::sync::Mutex<HashMap<WorkspaceId, Arc<OpenWorkspace>>>,
    /// The job of each session's running turn, for `cancel_turn`.
    turns: Mutex<HashMap<SessionId, JobId>>,
    /// Workspaces where "Run changes without asking" is on; off at launch.
    allow_writes: Mutex<HashSet<WorkspaceId>>,
    /// Native dialogs show one at a time, in event order.
    pub(crate) dialogs: tokio::sync::Mutex<()>,
}

impl AppState {
    /// Open the control plane and start the job queue.
    ///
    /// # Errors
    ///
    /// When the control plane does not open.
    pub(crate) async fn open(config: Config) -> Result<Self> {
        let control = ControlPlane::open(&config)
            .await
            .context("failed to open the control plane")?;
        let jobs = JobQueue::new(config.jobs.history);
        let auditor = Auditor::new(control.clone());
        Ok(Self {
            config,
            control,
            jobs,
            auditor,
            workspaces: tokio::sync::Mutex::new(HashMap::new()),
            turns: Mutex::new(HashMap::new()),
            allow_writes: Mutex::new(HashSet::new()),
            dialogs: tokio::sync::Mutex::new(()),
        })
    }

    /// The workspace `id`, opened on first use and held until closed.
    ///
    /// # Errors
    ///
    /// When no workspace has that id, or its file does not open (another
    /// process holds it: `ErrorKind::Locked`).
    pub(crate) async fn workspace(&self, id: &WorkspaceId) -> CommandResult<Arc<OpenWorkspace>> {
        let mut open = self.workspaces.lock().await;
        if let Some(workspace) = open.get(id) {
            return Ok(Arc::clone(workspace));
        }
        let row = self
            .control
            .get_workspace(id)
            .await?
            .ok_or_else(|| CommandError::not_found(format!("no workspace has the id {id}")))?;
        let workspace = Arc::new(self.open_workspace(row).await?);
        open.insert(id.clone(), Arc::clone(&workspace));
        Ok(workspace)
    }

    async fn open_workspace(&self, row: WorkspaceRow) -> CommandResult<OpenWorkspace> {
        let opened = OpenedWorkspace {
            config: self.config.clone(),
            name: row.name.clone(),
            workspace: row.clone(),
        };
        let (opened, db, audit) = tokio::task::spawn_blocking(move || {
            let db = opened.open_db()?;
            // Uploads an earlier process took and never finished are failed,
            // so their documents do not sit `queued` forever.
            let stale = db.fail_stale_uploads()?;
            if stale > 0 {
                tracing::warn!(workspace = %opened.workspace.id, stale, "failed uploads left queued by an earlier process");
            }
            let audit = AuditLog::open(&db)?;
            Ok::<_, anyhow::Error>((opened, db, audit))
        })
        .await
        .map_err(|e| anyhow::anyhow!("workspace open task failed: {e}"))??;
        let (db, reader) = opened.shared(db).await?;
        let dir = self.config.workspace_dir(row.id.as_str());
        let forward = match Listener::bind(&dir, HOLDER, reader.clone()).await {
            Ok(listener) => Some(listener),
            Err(e) => {
                tracing::warn!(error = %e, workspace = %row.id, "other quack processes cannot read this workspace through the app");
                None
            }
        };
        Ok(OpenWorkspace {
            row,
            db,
            reader,
            audit,
            _forward: forward,
        })
    }

    /// Let go of `id`: the file closes once the last command using it
    /// finishes. `false` when it was not open.
    pub(crate) async fn close_workspace(&self, id: &WorkspaceId) -> bool {
        self.workspaces.lock().await.remove(id).is_some()
    }

    /// The ids of the workspaces held open.
    pub(crate) async fn open_ids(&self) -> HashSet<WorkspaceId> {
        self.workspaces.lock().await.keys().cloned().collect()
    }

    /// Whether "Run changes without asking" is on for `id`.
    pub(crate) fn allows_writes(&self, id: &WorkspaceId) -> bool {
        self.allow_writes
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .contains(id)
    }

    pub(crate) fn set_allow_writes(&self, id: &WorkspaceId, allow: bool) {
        let mut set = self
            .allow_writes
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        if allow {
            set.insert(id.clone());
        } else {
            set.remove(id);
        }
    }

    /// Remember the job running `session`'s turn.
    pub(crate) fn turn_started(&self, session: &SessionId, job: JobId) {
        self.turns
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .insert(session.clone(), job);
    }

    /// Forget `session`'s turn once its events have ended, unless a newer
    /// turn has taken its place.
    pub(crate) fn turn_ended(&self, session: &SessionId, job: JobId) {
        let mut turns = self.turns.lock().unwrap_or_else(PoisonError::into_inner);
        if turns.get(session) == Some(&job) {
            turns.remove(session);
        }
    }

    /// The job running `session`'s turn, if one is.
    pub(crate) fn running_turn(&self, session: &SessionId) -> Option<JobId> {
        self.turns
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .get(session)
            .copied()
    }

    /// Stop the jobs within the configured grace and close every
    /// workspace, so each writer checkpoints before the process exits.
    pub(crate) async fn shutdown(&self) {
        let stopped = self
            .jobs
            .shutdown(self.config.server.shutdown_grace())
            .await;
        if !stopped.is_empty() {
            tracing::info!(jobs = stopped.len(), "jobs cancelled at quit");
        }
        self.workspaces.lock().await.clear();
    }
}
