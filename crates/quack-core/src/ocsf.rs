//! Access-audit rows as OCSF events (Open Cybersecurity Schema Framework),
//! for SIEMs and archives. Rows stay in `control.db`'s shape; this renders
//! them, so a new OCSF version only changes this module.

use jiff::civil::DateTime;
use jiff::tz::TimeZone;
use serde_json::{Value, json};

use crate::error::{Error, Result};
use crate::storage::audit::AuditDetailRow;
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
    fn of(action: &AuditAction, outcome: Outcome) -> Self {
        use ApiActivity::{Create, Delete, Other, Read, Update};
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
            | AuditAction::SavedRun
            | AuditAction::EmbeddingsStatus => Self::Api(Read),
            AuditAction::Ingest
            | AuditAction::Import
            | AuditAction::Propose
            | AuditAction::GraphExtract
            | AuditAction::Save
            | AuditAction::BreakGlass
            | AuditAction::Admin => Self::Api(Create),
            AuditAction::Context
            | AuditAction::Member
            | AuditAction::Share
            | AuditAction::Mode
            | AuditAction::Cancel
            | AuditAction::Permission
            | AuditAction::GraphReview
            | AuditAction::GraphRevalidate
            | AuditAction::GraphMerge
            | AuditAction::GraphEdit
            | AuditAction::Password
            | AuditAction::EmbeddingsRefresh => Self::Api(Update),
            AuditAction::Delete => Self::Api(Delete),
            // A name a newer build wrote is an API event under its own name.
            AuditAction::Workspace | AuditAction::Ontology | AuditAction::Unknown(_) => {
                Self::Api(Other)
            }
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
        let entry = &self.entry;
        let class = EventClass::of(&entry.action, entry.outcome);
        let (class_uid, class_name, category_uid, category_name) = class.class();
        let (activity_id, known_activity) = class.activity();
        let activity_name = match class {
            EventClass::Api(ApiActivity::Other) => entry.action.as_str(),
            EventClass::Authentication(_) | EventClass::Api(_) => known_activity,
        };
        let (status_id, status) = match entry.outcome {
            Outcome::Allowed => (1, "Success"),
            Outcome::Denied | Outcome::Error => (2, "Failure"),
        };
        let time = DateTime::strptime("%Y-%m-%d %H:%M:%S", &self.timestamp)
            .and_then(|dt| dt.to_zoned(TimeZone::UTC))
            .map_err(|e| Error::Config(format!("audit timestamp {:?}: {e}", self.timestamp)))?
            .timestamp();
        let user = entry.user_id.as_ref().map(|id| json!({ "uid": id }));
        // An operator at the shell has no user row: the CLI is the actor.
        let actor = user.clone().map_or_else(
            || json!({ "application": { "name": format!("quack {}", entry.origin.channel) } }),
            |user| json!({ "user": user }),
        );
        // CLI and terminal rows have no client address: they ran on the host.
        let src_endpoint = entry
            .origin
            .client_addr
            .as_ref()
            .map_or_else(|| json!({ "name": "local" }), |ip| json!({ "ip": ip }));
        let mut resources = Vec::new();
        if let Some(workspace) = &entry.workspace_id {
            resources.push(json!({ "type": "workspace", "uid": workspace }));
        }
        if let (Some(kind), Some(uid)) = (&entry.resource_type, &entry.resource_id) {
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
            "status_detail": entry.outcome.as_str(),
            "time": time.as_millisecond(),
            "time_dt": time.to_string(),
            "metadata": {
                "version": OCSF_VERSION,
                "uid": entry.id,
                "correlation_uid": entry.origin.request_id,
                "profiles": ["datetime"],
                "product": {
                    "name": "quack",
                    "vendor_name": "quack",
                    "version": env!("CARGO_PKG_VERSION"),
                },
            },
            "src_endpoint": src_endpoint,
            "unmapped": {
                "channel": entry.origin.channel.as_str(),
                "token_hash": entry.token_hash,
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
                "api": { "operation": entry.action },
                "resources": resources,
            }),
        };
        if let (Some(event), Some(fields)) = (event.as_object_mut(), fields.as_object()) {
            event.extend(fields.clone());
        }
        Ok(without_nulls(event))
    }
}

/// Whether an exported event carries the question's text. It is workspace
/// content, so the default leaves it out.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PromptText {
    Omit,
    Include,
}

impl AuditRow {
    /// The row as an OCSF event joined to its `_quack_audit` detail, the
    /// half that lives inside the workspace: for a query, the
    /// `ai_operation` profile with the model that answered (`ai_model`),
    /// the tool calls with their durations, and the documents and chunks
    /// the answer cited as resources; for any other action, the detail
    /// under `unmapped.detail`. A `break_glass` row is raised to Medium.
    /// The prompt goes in only when asked.
    ///
    /// # Errors
    ///
    /// Returns an error if the stored timestamp does not parse.
    pub fn to_ocsf_with_detail(
        &self,
        detail: Option<&AuditDetailRow>,
        prompt: PromptText,
    ) -> Result<Value> {
        let mut event = self.to_ocsf()?;
        if self.entry.action == AuditAction::BreakGlass
            && let Some(object) = event.as_object_mut()
        {
            object.insert("severity_id".into(), json!(3));
            object.insert("severity".into(), json!("Medium"));
        }
        let Some(detail) = detail.and_then(|d| d.detail.as_ref()) else {
            return Ok(event);
        };
        let Some(object) = event.as_object_mut() else {
            return Ok(event);
        };
        if self.entry.action != AuditAction::Query {
            object.insert(
                "unmapped".into(),
                unmapped_with(object, "detail", detail.clone()),
            );
            return Ok(without_nulls(event));
        }
        if let Some(profiles) = object
            .get_mut("metadata")
            .and_then(|m| m.get_mut("profiles"))
            .and_then(Value::as_array_mut)
        {
            profiles.push(json!("ai_operation"));
        }
        if let Some(model) = detail.get("model") {
            object.insert(
                "ai_model".into(),
                json!({
                    "name": model.get("name"),
                    "vendor_name": model.get("provider"),
                    "type": "Large Language Model",
                }),
            );
        }
        let tools: Vec<Value> = detail
            .get("steps")
            .and_then(Value::as_array)
            .map(|steps| {
                steps
                    .iter()
                    .map(|s| {
                        json!({
                            "name": s.get("tool"),
                            "duration_ms": s.get("duration_ms"),
                            "rows": s.get("rows"),
                            "summary": s.get("summary"),
                        })
                    })
                    .collect()
            })
            .unwrap_or_default();
        let mut resources = object
            .get("resources")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        for cited in detail
            .get("citations")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
        {
            if let Some(document) = cited.get("document_id") {
                resources.push(json!({ "type": "document", "uid": document }));
            }
            if let Some(chunk) = cited.get("chunk_id") {
                resources.push(json!({ "type": "chunk", "uid": chunk }));
            }
        }
        let mut seen = Vec::new();
        resources.retain(|r| {
            if seen.contains(r) {
                false
            } else {
                seen.push(r.clone());
                true
            }
        });
        object.insert("resources".into(), Value::Array(resources));
        let mut extra = json!({ "tools": tools, "session_id": self.entry.resource_id });
        if prompt == PromptText::Include
            && let Some(text) = detail.get("prompt")
            && let Some(extra) = extra.as_object_mut()
        {
            extra.insert("prompt".into(), text.clone());
        }
        object.insert("unmapped".into(), unmapped_with(object, "ai", extra));
        Ok(without_nulls(event))
    }
}

/// The event's `unmapped` object with `key` set to `value`.
fn unmapped_with(event: &serde_json::Map<String, Value>, key: &str, value: Value) -> Value {
    let mut unmapped = event
        .get("unmapped")
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default();
    unmapped.insert(key.to_owned(), value);
    Value::Object(unmapped)
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
    use crate::storage::control::{AuditEntry, Channel, Origin, ResourceKind};

    fn row(action: AuditAction, outcome: Outcome) -> AuditRow {
        AuditRow {
            timestamp: String::from("2026-09-24 12:34:56"),
            entry: AuditEntry {
                id: AuditId::from("0199aaaa-0000-7000-8000-000000000001"),
                user_id: Some(UserId::from("u1")),
                token_hash: None,
                workspace_id: Some(WorkspaceId::from("w1")),
                action,
                resource_type: Some(ResourceKind::Document),
                resource_id: Some(String::from("d1")),
                outcome,
                origin: Origin {
                    channel: Channel::Api,
                    client_addr: Some(String::from("10.0.0.7")),
                    request_id: Some(String::from("req-1")),
                },
            },
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
        let event = render(&row(AuditAction::Login, Outcome::Allowed));
        assert_required(&event);
        assert_eq!(event["type_uid"], 300_201);
        assert_eq!(event["status_id"], 1);
        assert_eq!(event["user"]["uid"], "u1");

        let mut unknown = row(AuditAction::Token, Outcome::Denied);
        unknown.entry.user_id = None;
        let event = render(&unknown);
        assert_required(&event);
        assert_eq!(
            event["type_uid"], 300_201,
            "a rejected bearer is a failed logon"
        );
        assert_eq!(event["status_id"], 2);
        assert_eq!(event["user"]["type_id"], 0);

        assert_eq!(
            render(&row(AuditAction::Logout, Outcome::Allowed))["type_uid"],
            300_202
        );
    }

    #[test]
    fn a_query_with_its_detail_carries_the_ai_operation_profile() {
        let mut query = row(AuditAction::Query, Outcome::Allowed);
        query.entry.resource_type = Some(ResourceKind::Session);
        query.entry.resource_id = Some(String::from("s1"));
        let detail = AuditDetailRow {
            id: query.entry.id.clone(),
            timestamp: String::from("2026-09-24 12:34:56"),
            user_id: query.entry.user_id.clone(),
            action: String::from("query"),
            detail: Some(json!({
                "prompt": "how many orders per region?",
                "model": { "provider": "ollama", "name": "gpt-oss:20b" },
                "steps": [
                    { "tool": "run_sql", "duration_ms": 9, "rows": 4, "summary": "4 rows" },
                    { "tool": "search_documents", "duration_ms": 41, "summary": "8 chunks" }
                ],
                "citations": [
                    { "document_id": "d1", "chunk_id": "c1", "chunk_index": 3 },
                    { "document_id": "d1", "chunk_id": "c2", "chunk_index": 4 }
                ]
            })),
        };
        let event = query
            .to_ocsf_with_detail(Some(&detail), PromptText::Omit)
            .unwrap_or_else(|e| panic_with(&e.to_string()));
        assert_required(&event);
        assert_eq!(
            event["metadata"]["profiles"],
            json!(["datetime", "ai_operation"])
        );
        assert_eq!(event["ai_model"]["name"], "gpt-oss:20b");
        assert_eq!(event["ai_model"]["vendor_name"], "ollama");
        assert_eq!(event["unmapped"]["ai"]["tools"][0]["name"], "run_sql");
        assert_eq!(event["unmapped"]["ai"]["tools"][1]["duration_ms"], 41);
        assert_eq!(event["unmapped"]["ai"]["session_id"], "s1");
        assert!(event["unmapped"]["ai"].get("prompt").is_none(), "{event}");
        let resources = event["resources"].as_array().cloned().unwrap_or_default();
        assert!(resources.contains(&json!({ "type": "document", "uid": "d1" })));
        assert!(resources.contains(&json!({ "type": "chunk", "uid": "c2" })));
        assert_eq!(
            resources.iter().filter(|r| r["type"] == "document").count(),
            1,
            "a document cited twice is one resource"
        );
        let with_prompt = query
            .to_ocsf_with_detail(Some(&detail), PromptText::Include)
            .unwrap_or_else(|e| panic_with(&e.to_string()));
        assert_eq!(
            with_prompt["unmapped"]["ai"]["prompt"],
            "how many orders per region?"
        );
        // Without a detail row the event is the plain one; another action's
        // detail rides along unmapped.
        let plain = query
            .to_ocsf_with_detail(None, PromptText::Include)
            .unwrap_or_else(|e| panic_with(&e.to_string()));
        assert!(plain.get("ai_model").is_none());
        let sql = row(AuditAction::Sql, Outcome::Allowed);
        let sql_detail = AuditDetailRow {
            detail: Some(json!({ "sql": "SELECT 1" })),
            action: String::from("sql"),
            ..detail
        };
        let event = sql
            .to_ocsf_with_detail(Some(&sql_detail), PromptText::Omit)
            .unwrap_or_else(|e| panic_with(&e.to_string()));
        assert_eq!(event["unmapped"]["detail"]["sql"], "SELECT 1");
    }

    #[test]
    fn a_break_glass_grant_is_a_medium_create_event() {
        let event = render(&row(AuditAction::BreakGlass, Outcome::Allowed));
        assert_required(&event);
        assert_eq!(event["type_uid"], 600_301);
        assert_eq!(event["api"]["operation"], "break_glass");
        let raised = row(AuditAction::BreakGlass, Outcome::Allowed)
            .to_ocsf_with_detail(None, PromptText::Omit)
            .unwrap_or_else(|e| panic_with(&e.to_string()));
        assert_eq!(raised["severity_id"], 3);
    }

    #[test]
    fn workspace_actions_are_api_activity() {
        let event = render(&row(AuditAction::Sql, Outcome::Allowed));
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
            render(&row(AuditAction::Ingest, Outcome::Allowed))["type_uid"],
            600_301
        );
        assert_eq!(
            render(&row(AuditAction::Delete, Outcome::Error))["type_uid"],
            600_304
        );
        assert_eq!(
            render(&row(AuditAction::Delete, Outcome::Error))["status_detail"],
            "error"
        );
        // A membership change is API activity/Update; a failed no-op removal
        // audited as `Error` is a `Failure` event, never a `Success` one.
        let allowed = render(&row(AuditAction::Member, Outcome::Allowed));
        assert_eq!(allowed["type_uid"], 600_303);
        assert_eq!(allowed["activity_name"], "Update");
        assert_eq!(allowed["status_id"], 1);
        assert_eq!(allowed["status"], "Success");
        let failed = render(&row(AuditAction::Member, Outcome::Error));
        assert_eq!(failed["type_uid"], 600_303);
        assert_eq!(failed["status_id"], 2);
        assert_eq!(failed["status"], "Failure");
        assert_eq!(failed["status_detail"], "error");
    }

    #[test]
    fn unmapped_actions_and_local_rows_stay_valid() {
        let mut cli = row(AuditAction::Workspace, Outcome::Allowed);
        cli.entry.origin.client_addr = None;
        cli.entry.origin.channel = Channel::Cli;
        cli.entry.user_id = None;
        cli.entry.origin.request_id = None;
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

        let retired = render(&row(
            AuditAction::Unknown(String::from("retired_action")),
            Outcome::Allowed,
        ));
        assert_eq!(retired["type_uid"], 600_399);
        assert_eq!(retired["activity_name"], "retired_action");
        assert_eq!(retired["api"]["operation"], "retired_action");

        let mut broken = row(AuditAction::Open, Outcome::Allowed);
        broken.timestamp = String::from("yesterday");
        assert!(broken.to_ocsf().is_err());
    }
}
