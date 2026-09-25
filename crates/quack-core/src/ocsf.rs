//! Access-audit rows as OCSF events (Open Cybersecurity Schema Framework),
//! for SIEMs and archives. Rows stay in `control.db`'s shape; this renders
//! them, so a new OCSF version only changes this module.

use jiff::civil::DateTime;
use jiff::tz::TimeZone;
use serde_json::{Value, json};

use crate::error::{Error, Result};
use crate::storage::control::{AuditAction, AuditRow, Outcome};

/// The OCSF release the events conform to.
pub const OCSF_VERSION: &str = "1.9.0";

/// An OCSF event class and the activity within it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum EventClass {
    /// Authentication [3002], in Identity & Access Management [3].
    Authentication(AuthActivity),
    /// API Activity [6003], in Application Activity [6].
    Api(ApiActivity),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AuthActivity {
    Logon,
    Logoff,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ApiActivity {
    Create,
    Read,
    Update,
    Delete,
    /// No CRUD verb fits; `activity_name` carries quack's action.
    Other,
}

impl EventClass {
    /// Where a row's action lands. A denied `token` is a rejected bearer,
    /// so it is a failed logon rather than token management.
    fn of(action: Option<AuditAction>, outcome: Outcome) -> Self {
        use ApiActivity::{Create, Delete, Other, Read, Update};
        let Some(action) = action else {
            return Self::Api(Other);
        };
        match action {
            AuditAction::Login | AuditAction::Session => Self::Authentication(AuthActivity::Logon),
            AuditAction::Logout => Self::Authentication(AuthActivity::Logoff),
            AuditAction::Token => match outcome {
                Outcome::Denied => Self::Authentication(AuthActivity::Logon),
                Outcome::Allowed | Outcome::Error => Self::Api(Other),
            },
            AuditAction::Open
            | AuditAction::List
            | AuditAction::Page
            | AuditAction::Show
            | AuditAction::Stream
            | AuditAction::Query
            | AuditAction::Search
            | AuditAction::Sql
            | AuditAction::Export
            | AuditAction::Graph
            | AuditAction::SessionRead
            | AuditAction::EmbeddingsStatus => Self::Api(Read),
            AuditAction::Ingest
            | AuditAction::Import
            | AuditAction::Propose
            | AuditAction::GraphExtract
            | AuditAction::Admin => Self::Api(Create),
            AuditAction::Context
            | AuditAction::Member
            | AuditAction::Share
            | AuditAction::Mode
            | AuditAction::Cancel
            | AuditAction::GraphReview
            | AuditAction::GraphRevalidate
            | AuditAction::GraphMerge
            | AuditAction::EmbeddingsRefresh => Self::Api(Update),
            AuditAction::Delete => Self::Api(Delete),
            AuditAction::Workspace | AuditAction::Ontology => Self::Api(Other),
        }
    }

    fn class(self) -> (u32, &'static str, u32, &'static str) {
        match self {
            Self::Authentication(_) => (3002, "Authentication", 3, "Identity & Access Management"),
            Self::Api(_) => (6003, "API Activity", 6, "Application Activity"),
        }
    }

    fn activity(self) -> (u32, &'static str) {
        match self {
            Self::Authentication(AuthActivity::Logon) => (1, "Logon"),
            Self::Authentication(AuthActivity::Logoff) => (2, "Logoff"),
            Self::Api(ApiActivity::Create) => (1, "Create"),
            Self::Api(ApiActivity::Read) => (2, "Read"),
            Self::Api(ApiActivity::Update) => (3, "Update"),
            Self::Api(ApiActivity::Delete) => (4, "Delete"),
            Self::Api(ApiActivity::Other) => (99, "Other"),
        }
    }
}

impl AuditRow {
    /// The row as an OCSF event. Only `control.db` fields go in: ids,
    /// outcome, and the client, never the workspace's audit detail.
    ///
    /// # Errors
    ///
    /// Returns an error if the stored timestamp does not parse.
    pub fn to_ocsf(&self) -> Result<Value> {
        let action = self.action.parse::<AuditAction>().ok();
        let class = EventClass::of(action, self.outcome);
        let (class_uid, class_name, category_uid, category_name) = class.class();
        let (activity_id, known_activity) = class.activity();
        let activity_name = match class {
            EventClass::Api(ApiActivity::Other) => self.action.as_str(),
            EventClass::Authentication(_) | EventClass::Api(_) => known_activity,
        };
        let (status_id, status) = match self.outcome {
            Outcome::Allowed => (1, "Success"),
            Outcome::Denied | Outcome::Error => (2, "Failure"),
        };
        let time = DateTime::strptime("%Y-%m-%d %H:%M:%S", &self.timestamp)
            .and_then(|dt| dt.to_zoned(TimeZone::UTC))
            .map_err(|e| Error::Config(format!("audit timestamp {:?}: {e}", self.timestamp)))?
            .timestamp();
        let user = self.user_id.as_ref().map(|id| json!({ "uid": id }));
        // An operator at the shell has no user row: the CLI is the actor.
        let actor = user.clone().map_or_else(
            || json!({ "application": { "name": format!("quack {}", self.channel) } }),
            |user| json!({ "user": user }),
        );
        // CLI and terminal rows have no client address: they ran on the host.
        let src_endpoint = self
            .client_addr
            .as_ref()
            .map_or_else(|| json!({ "name": "local" }), |ip| json!({ "ip": ip }));
        let mut resources = Vec::new();
        if let Some(workspace) = &self.workspace_id {
            resources.push(json!({ "type": "workspace", "uid": workspace }));
        }
        if let (Some(kind), Some(uid)) = (&self.resource_type, &self.resource_id) {
            resources.push(json!({ "type": kind, "uid": uid }));
        }
        let mut event = json!({
            "class_uid": class_uid,
            "class_name": class_name,
            "category_uid": category_uid,
            "category_name": category_name,
            "activity_id": activity_id,
            "activity_name": activity_name,
            "type_uid": class_uid.saturating_mul(100).saturating_add(activity_id),
            "severity_id": 1,
            "severity": "Informational",
            "status_id": status_id,
            "status": status,
            "status_detail": self.outcome.as_str(),
            "time": time.as_millisecond(),
            "time_dt": time.to_string(),
            "metadata": {
                "version": OCSF_VERSION,
                "uid": self.id,
                "correlation_uid": self.request_id,
                "profiles": ["datetime"],
                "product": {
                    "name": "quack",
                    "vendor_name": "quack",
                    "version": env!("CARGO_PKG_VERSION"),
                },
            },
            "src_endpoint": src_endpoint,
            "unmapped": {
                "channel": self.channel.as_str(),
                "token_hash": self.token_hash,
            },
        });
        let fields = match class {
            // Authentication requires a user; an unknown account is type 0.
            EventClass::Authentication(_) => json!({
                "user": user.unwrap_or_else(|| json!({ "name": "unknown", "type_id": 0 })),
                "service": { "name": "quack" },
            }),
            EventClass::Api(_) => json!({
                "actor": actor,
                "api": { "operation": self.action },
                "resources": resources,
            }),
        };
        if let (Some(event), Some(fields)) = (event.as_object_mut(), fields.as_object()) {
            event.extend(fields.clone());
        }
        Ok(without_nulls(event))
    }
}

/// OCSF leaves an attribute out rather than setting it to null.
fn without_nulls(value: Value) -> Value {
    match value {
        Value::Object(map) => Value::Object(
            map.into_iter()
                .filter(|(_, v)| !v.is_null())
                .map(|(k, v)| (k, without_nulls(v)))
                .collect(),
        ),
        Value::Array(items) => Value::Array(items.into_iter().map(without_nulls).collect()),
        other => other,
    }
}

#[cfg(test)]
#[expect(
    clippy::indexing_slicing,
    reason = "tests read fixed keys of the rendered JSON"
)]
mod tests {
    use super::*;
    use crate::ids::{AuditId, UserId, WorkspaceId};
    use crate::storage::control::Channel;

    fn row(action: &str, outcome: Outcome) -> AuditRow {
        AuditRow {
            id: AuditId::from("0199aaaa-0000-7000-8000-000000000001"),
            timestamp: String::from("2026-09-24 12:34:56"),
            user_id: Some(UserId::from("u1")),
            token_hash: None,
            workspace_id: Some(WorkspaceId::from("w1")),
            action: action.to_owned(),
            resource_type: Some(String::from("document")),
            resource_id: Some(String::from("d1")),
            outcome,
            channel: Channel::Api,
            client_addr: Some(String::from("10.0.0.7")),
            request_id: Some(String::from("req-1")),
        }
    }

    fn render(row: &AuditRow) -> Value {
        row.to_ocsf().unwrap_or_else(|e| panic_with(&e.to_string()))
    }

    #[expect(clippy::panic, reason = "test failure path")]
    fn panic_with(msg: &str) -> ! {
        panic!("{msg}")
    }

    /// The base event's required attributes, and each class's own.
    fn assert_required(event: &Value) {
        for key in [
            "activity_id",
            "category_uid",
            "class_uid",
            "metadata",
            "severity_id",
            "time",
            "type_uid",
        ] {
            assert!(event.get(key).is_some(), "{key} missing: {event}");
        }
        assert_eq!(event["metadata"]["version"], OCSF_VERSION);
        assert!(event["metadata"]["product"]["name"].is_string());
        match event["class_uid"].as_u64() {
            Some(3002) => {
                assert!(event["user"].is_object() && event["service"]["name"].is_string());
            }
            Some(6003) => {
                assert!(event["actor"].is_object(), "{event}");
                assert!(event["api"]["operation"].is_string(), "{event}");
                assert!(event["src_endpoint"].is_object(), "{event}");
            }
            other => panic_with(&format!("unexpected class {other:?}")),
        }
    }

    #[test]
    fn logins_are_authentication_events() {
        let event = render(&row("login", Outcome::Allowed));
        assert_required(&event);
        assert_eq!(event["type_uid"], 300_201);
        assert_eq!(event["status_id"], 1);
        assert_eq!(event["user"]["uid"], "u1");

        let mut unknown = row("token", Outcome::Denied);
        unknown.user_id = None;
        let event = render(&unknown);
        assert_required(&event);
        assert_eq!(
            event["type_uid"], 300_201,
            "a rejected bearer is a failed logon"
        );
        assert_eq!(event["status_id"], 2);
        assert_eq!(event["user"]["type_id"], 0);

        assert_eq!(
            render(&row("logout", Outcome::Allowed))["type_uid"],
            300_202
        );
    }

    #[test]
    fn workspace_actions_are_api_activity() {
        let event = render(&row("sql", Outcome::Allowed));
        assert_required(&event);
        assert_eq!(event["type_uid"], 600_302);
        assert_eq!(event["api"]["operation"], "sql");
        assert_eq!(event["src_endpoint"]["ip"], "10.0.0.7");
        assert_eq!(
            event["metadata"]["uid"],
            "0199aaaa-0000-7000-8000-000000000001"
        );
        assert_eq!(event["metadata"]["correlation_uid"], "req-1");
        assert_eq!(
            event["resources"],
            json!([{ "type": "workspace", "uid": "w1" }, { "type": "document", "uid": "d1" }])
        );
        assert_eq!(event["time"], 1_790_253_296_000_i64);
        assert_eq!(event["time_dt"], "2026-09-24T12:34:56Z");
        assert_eq!(
            render(&row("ingest", Outcome::Allowed))["type_uid"],
            600_301
        );
        assert_eq!(render(&row("delete", Outcome::Error))["type_uid"], 600_304);
        assert_eq!(
            render(&row("delete", Outcome::Error))["status_detail"],
            "error"
        );
    }

    #[test]
    fn unmapped_actions_and_local_rows_stay_valid() {
        let mut cli = row("workspace", Outcome::Allowed);
        cli.client_addr = None;
        cli.channel = Channel::Cli;
        cli.user_id = None;
        cli.request_id = None;
        let event = render(&cli);
        assert_required(&event);
        assert_eq!(event["type_uid"], 600_399);
        assert_eq!(event["activity_name"], "workspace");
        assert_eq!(event["src_endpoint"]["name"], "local");
        assert_eq!(event["unmapped"]["channel"], "cli");
        assert_eq!(event["actor"]["application"]["name"], "quack cli");
        assert!(
            event["metadata"].get("correlation_uid").is_none(),
            "{event}"
        );
        assert!(event["unmapped"].get("token_hash").is_none(), "{event}");

        let retired = render(&row("retired_action", Outcome::Allowed));
        assert_eq!(retired["type_uid"], 600_399);
        assert_eq!(retired["activity_name"], "retired_action");

        let mut broken = row("open", Outcome::Allowed);
        broken.timestamp = String::from("yesterday");
        assert!(broken.to_ocsf().is_err());
    }
}
