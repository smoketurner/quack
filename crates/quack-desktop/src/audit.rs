//! Every workspace-touching command writes an `audit_log` row on channel
//! `desktop` as the one local user, plus the `_quack_audit` detail row
//! inside the workspace under the same id (design doc section 12): the
//! shape of the server's `Access::audit` without the authorization half.

use anyhow::Result;
use quack_core::ids::{AuditId, UserId};
use quack_core::storage::audit::AuditDetail;
use quack_core::storage::control::{
    AuditAction, AuditEntry, AuditResource, Channel, ControlPlane, Outcome,
};
use serde_json::Value;

use crate::app::OpenWorkspace;

/// The one person at the keyboard, as the server's `--local` mode names them.
pub(crate) const LOCAL_USER: &str = "local";

/// Writes the app's audit rows.
#[derive(Clone)]
pub(crate) struct Auditor {
    control: ControlPlane,
    user: UserId,
}

impl Auditor {
    pub(crate) fn new(control: ControlPlane) -> Self {
        Self {
            control,
            user: UserId::from(LOCAL_USER),
        }
    }

    /// An `audit_log` entry for a change the control plane commits with
    /// the change itself (a workspace created, renamed, or deleted).
    pub(crate) fn entry(&self, action: AuditAction, outcome: Outcome) -> AuditEntry {
        let mut entry = AuditEntry::new(action, outcome, Channel::Desktop);
        entry.user_id = Some(self.user.clone());
        entry
    }

    /// Both halves of one audit row for `workspace`: the `audit_log` row,
    /// then the `_quack_audit` detail under its id.
    ///
    /// # Errors
    ///
    /// When either row cannot be written.
    pub(crate) async fn record(
        &self,
        workspace: &OpenWorkspace,
        action: AuditAction,
        resource: Option<AuditResource<'_>>,
        outcome: Outcome,
        detail: Option<Value>,
    ) -> Result<AuditId> {
        let mut entry = self.entry(action, outcome).in_workspace(&workspace.row.id);
        if let Some(resource) = resource {
            entry = entry.on(resource);
        }
        self.control.record_audit(&entry).await?;
        workspace
            .audit
            .record(AuditDetail {
                id: entry.id.clone(),
                user_id: Some(self.user.clone()),
                action: entry.action.to_string(),
                detail: detail.unwrap_or_else(|| serde_json::json!({})),
            })
            .await?;
        Ok(entry.id)
    }
}
