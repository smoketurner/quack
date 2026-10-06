//! `control.db`: who may open which workspace, and the access audit.
//!
//! Users, workspaces (name and label), membership, API tokens, and the
//! append-only `audit_log`. Nothing here can reveal workspace content
//! (design doc 5.5); the detail of what was done lives in `_quack_audit`
//! inside the workspace file, keyed by the same UUID v7.

use argon2::Argon2;
use argon2::password_hash::phc::PasswordHash;
use argon2::password_hash::{PasswordHasher, PasswordVerifier};
use sea_query::{Cond, DynIden, Expr, ExprTrait, IntoIden, OnConflict, Order, Query};
use serde::{Serialize, Serializer};
use sqlx::sqlite::{SqliteConnectOptions, SqlitePoolOptions, SqliteRow};
use sqlx::{FromRow, Row, SqlitePool};
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fmt;
use std::str::FromStr;

use jiff::{SignedDuration, Timestamp};

use super::queries::{
    ApiTokens, AuditLog, Bound, ClientKeys, ClientRegistrations, GroupRoles, Members,
    ProviderTokens, SealedColumns, UserTokens, Users, Workspaces,
};
use crate::config::{Config, Lockout, ProviderName};
use crate::crypto::sha256_hex;
use crate::error::{Error, Result};
use crate::ids::{AuditId, UserId, WorkspaceId};
use crate::ocsf::PromptText;
use crate::oidc::OidcSubject;
use crate::storage::audit::AuditDetailRow;
use crate::text::blank_as_none;
use crate::vault::Sealed;

/// The `control.db` schema, as plain SQL files embedded at compile time.
///
/// One file per version under `crates/quack-core/migrations/`. A file that has
/// shipped is frozen: sqlx records a SHA-384 checksum per version in
/// `_sqlx_migrations` and refuses a database whose recorded checksum no longer
/// matches. Migrations are never regenerated from the `Iden` enums in
/// `storage::queries`, because those track the current schema, not history
/// (`docs/migrations.md`).
static MIGRATOR: sqlx::migrate::Migrator = sqlx::migrate!("./migrations");

/// A name a new workspace may take: trimmed, non-empty, and without `/`,
/// `\`, or `.`. Lookups take plain text, since older rows predate the rule.
#[derive(Debug, Clone, PartialEq, Eq, serde::Deserialize)]
#[serde(try_from = "String")]
pub struct WorkspaceName(String);

impl WorkspaceName {
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl FromStr for WorkspaceName {
    type Err = Error;

    fn from_str(name: &str) -> Result<Self> {
        let name = name.trim();
        if name.is_empty() || name.contains(['/', '\\', '.']) {
            return Err(Error::InvalidWorkspaceName);
        }
        Ok(Self(name.to_owned()))
    }
}

impl TryFrom<String> for WorkspaceName {
    type Error = Error;

    fn try_from(name: String) -> Result<Self> {
        name.parse()
    }
}

/// The name `[general].default_workspace` takes when unset.
impl Default for WorkspaceName {
    fn default() -> Self {
        Self(String::from("default"))
    }
}

impl fmt::Display for WorkspaceName {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// A workspace row from the control plane.
#[derive(Debug, Clone, serde::Serialize)]
pub struct WorkspaceRow {
    pub id: WorkspaceId,
    pub name: String,
    pub classification: String,
    pub allowed_providers: AllowedProviders,
}

impl FromRow<'_, SqliteRow> for WorkspaceRow {
    fn from_row(row: &SqliteRow) -> sqlx::Result<Self> {
        Ok(Self {
            id: row.try_get("id")?,
            name: row.try_get("name")?,
            classification: row.try_get("classification")?,
            allowed_providers: AllowedProviders::from_column(
                row.try_get::<Option<String>, _>("allowed_providers")?
                    .as_deref(),
            ),
        })
    }
}

/// A stored text column parsed into a typed value; a value that does not
/// parse is a decode error for the column, not a silent default.
fn parsed<T>(r: &SqliteRow, column: &str) -> sqlx::Result<T>
where
    T: FromStr<Err = Error>,
{
    r.try_get::<String, _>(column)?
        .parse()
        .map_err(|e: Error| sqlx::Error::ColumnDecode {
            index: column.to_owned(),
            source: Box::new(e),
        })
}

/// Which configured providers a workspace's questions may use. Stored as a
/// JSON array of names, or NULL for all; decoded once, when the row is read.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub enum AllowedProviders {
    #[default]
    All,
    Only(BTreeSet<String>),
}

impl AllowedProviders {
    /// Decode the stored column. A value that does not parse allows no
    /// provider rather than every one, and says so in the log.
    fn from_column(stored: Option<&str>) -> Self {
        let Some(text) = stored else {
            return Self::All;
        };
        match serde_json::from_str(text) {
            Ok(names) => Self::Only(names),
            Err(e) => {
                tracing::warn!(error = %e, "unreadable allowed_providers; allowing none");
                Self::Only(BTreeSet::new())
            }
        }
    }

    #[must_use]
    pub fn permits(&self, provider: &str) -> bool {
        match self {
            Self::All => true,
            Self::Only(names) => names.contains(provider),
        }
    }

    /// The names, or `None` when every provider is allowed.
    #[must_use]
    pub fn names(&self) -> Option<&BTreeSet<String>> {
        match self {
            Self::All => None,
            Self::Only(names) => Some(names),
        }
    }
}

/// As a refusal names it: `every provider`, `no provider`, or `only: a, b`.
impl fmt::Display for AllowedProviders {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::All => f.write_str("every provider"),
            Self::Only(names) if names.is_empty() => f.write_str("no provider"),
            Self::Only(names) => {
                f.write_str("only: ")?;
                for (i, name) in names.iter().enumerate() {
                    if i > 0 {
                        f.write_str(", ")?;
                    }
                    f.write_str(name)?;
                }
                Ok(())
            }
        }
    }
}

impl Serialize for AllowedProviders {
    fn serialize<S: Serializer>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error> {
        self.names().serialize(serializer)
    }
}

/// Settings a workspace owner may change.
#[derive(Debug, Clone, Default)]
pub struct WorkspaceChanges {
    pub classification: Option<String>,
    pub allowed_providers: ProviderAllowList,
}

/// The provider allow-list for a workspace update.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub enum ProviderAllowList {
    /// Leave it as it is.
    #[default]
    Keep,
    /// Clear it: every configured provider is allowed.
    All,
    /// Only these provider names.
    Only(BTreeSet<String>),
}

/// Whether a server user administers the server. Serializes as the
/// `is_admin` boolean.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(from = "bool", into = "bool")]
pub enum UserKind {
    #[default]
    Standard,
    /// Manages users and every workspace; reads workspace content only as
    /// a member.
    Admin,
}

flag_enum!(UserKind, false => Standard, true => Admin);

/// A token just minted: the secret, shown to its owner once and stored
/// only as a hash, and its row.
#[derive(Debug, Clone)]
pub struct IssuedToken {
    pub secret: TokenSecret,
    pub row: TokenRow,
}

/// An API token's secret. `Debug` never prints it; `expose` is the one way
/// to read it.
#[derive(Clone)]
pub struct TokenSecret(String);

impl TokenSecret {
    #[must_use]
    pub fn expose(&self) -> &str {
        &self.0
    }
}

impl fmt::Debug for TokenSecret {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("TokenSecret(<redacted>)")
    }
}

/// A workspace and where one person stands in it. Serializes as the
/// workspace's fields plus `role`: the role, or `null` for an admin
/// without membership.
#[derive(Debug, Clone, serde::Serialize)]
pub struct Membership {
    #[serde(flatten)]
    pub workspace: WorkspaceRow,
    #[serde(rename = "role")]
    pub standing: Standing,
}

/// Where a person stands in a workspace: a member with a role, or a
/// server admin without membership, who reaches settings and members but
/// never content.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(into = "Option<Role>")]
pub enum Standing {
    Member(Role),
    Admin,
}

impl Standing {
    /// A membership's role, or `Admin` where there is none.
    #[must_use]
    pub fn of(role: Option<Role>) -> Self {
        role.map_or(Self::Admin, Self::Member)
    }
}

/// The role, when there is a membership.
impl From<Standing> for Option<Role> {
    fn from(standing: Standing) -> Self {
        match standing {
            Standing::Member(role) => Some(role),
            Standing::Admin => None,
        }
    }
}

impl fmt::Display for Standing {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Member(role) => role.fmt(f),
            Self::Admin => f.write_str("admin"),
        }
    }
}

/// When a workspace was created and last reached, as stored UTC text.
#[derive(Debug, Clone)]
pub struct WorkspaceTimes {
    pub created_at: String,
    /// `None` until a request first touches the workspace.
    pub last_accessed_at: Option<String>,
}

/// A server user. The password hash never leaves this module.
#[derive(Debug, Clone, serde::Serialize)]
pub struct UserRow {
    pub id: UserId,
    pub username: String,
    #[serde(rename = "is_admin")]
    pub kind: UserKind,
    pub created_at: String,
    /// When an admin disabled the account; a disabled user cannot log in,
    /// and every credential they hold is refused.
    pub disabled_at: Option<String>,
    /// While a lockout for wrong passwords lasts.
    pub locked_until: Option<String>,
}

impl UserRow {
    #[must_use]
    pub fn is_disabled(&self) -> bool {
        self.disabled_at.is_some()
    }
}

impl FromRow<'_, SqliteRow> for UserRow {
    fn from_row(row: &SqliteRow) -> sqlx::Result<Self> {
        Ok(Self {
            id: row.try_get("id")?,
            username: row.try_get("username")?,
            kind: UserKind::from(row.try_get::<bool, _>("is_admin")?),
            created_at: row.try_get("created_at")?,
            disabled_at: row.try_get("disabled_at")?,
            locked_until: row.try_get("locked_until")?,
        })
    }
}

/// What checking a password found.
#[derive(Debug)]
pub enum PasswordCheck {
    Verified(UserRow),
    /// No such user, or a wrong password: the user when there is one, so
    /// the denied row can name the account aimed at.
    Wrong(Option<UserId>),
    Disabled(UserId),
    /// Too many wrong passwords in a row: refused until `until`.
    Locked {
        user_id: UserId,
        until: String,
    },
}

/// What a member may do in a workspace (design doc 12).
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, serde::Serialize, serde::Deserialize,
)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    /// Asks questions and searches.
    Viewer,
    /// Also uploads, pins, deletes own uploads, grants write, edits context.
    Member,
    /// Also manages members and tokens and sees every session.
    Owner,
}

text_enum!(Role, "role", {
    Viewer => "viewer",
    Member => "member",
    Owner => "owner",
});

/// One membership, with the username for listings.
#[derive(Debug, Clone, serde::Serialize)]
pub struct MemberRow {
    pub workspace_id: WorkspaceId,
    pub user_id: UserId,
    pub username: String,
    pub role: Role,
    pub created_at: String,
    pub granted_by: GrantedBy,
}

impl FromRow<'_, SqliteRow> for MemberRow {
    fn from_row(row: &SqliteRow) -> sqlx::Result<Self> {
        Ok(Self {
            workspace_id: row.try_get("workspace_id")?,
            user_id: row.try_get("user_id")?,
            username: row.try_get("username")?,
            role: parsed(row, "role")?,
            created_at: row.try_get("created_at")?,
            granted_by: parsed(row, "granted_by")?,
        })
    }
}

/// Who gave a member their role: a person, or the identity provider's
/// group claim at sign-in. Only the latter is revoked when the groups
/// change.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum GrantedBy {
    User,
    Idp,
}

text_enum!(GrantedBy, "membership grant", {
    User => "user",
    Idp => "idp",
});

/// A role an identity provider's group carries in a workspace.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub struct GroupRoleRow {
    pub workspace_id: WorkspaceId,
    pub group_name: String,
    pub role: Role,
    pub created_at: String,
}

impl FromRow<'_, SqliteRow> for GroupRoleRow {
    fn from_row(row: &SqliteRow) -> sqlx::Result<Self> {
        Ok(Self {
            workspace_id: row.try_get("workspace_id")?,
            group_name: row.try_get("group_name")?,
            role: parsed(row, "role")?,
            created_at: row.try_get("created_at")?,
        })
    }
}

/// What reconciling a person's groups changed.
#[derive(Debug, Clone, Default, PartialEq, Eq, serde::Serialize)]
pub struct Reconciled {
    pub granted: usize,
    pub changed: usize,
    pub revoked: usize,
}

/// What an API token may do (design doc 12).
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Scope {
    Read,
    Write,
    Admin,
}

text_enum!(Scope, "scope", {
    Read => "read",
    Write => "write",
    Admin => "admin",
});

/// An API token row; the token itself is shown once at creation.
#[derive(Debug, Clone, serde::Serialize)]
pub struct TokenRow {
    pub token_hash: String,
    pub workspace_id: WorkspaceId,
    pub user_id: UserId,
    pub name: String,
    pub scopes: Vec<Scope>,
    pub created_at: String,
    pub expires_at: Option<String>,
    pub last_used_at: Option<String>,
}

impl FromRow<'_, SqliteRow> for TokenRow {
    fn from_row(row: &SqliteRow) -> sqlx::Result<Self> {
        let scopes: String = row.try_get("scopes")?;
        Ok(Self {
            token_hash: row.try_get("token_hash")?,
            workspace_id: row.try_get("workspace_id")?,
            user_id: row.try_get("user_id")?,
            name: row.try_get("name")?,
            scopes: serde_json::from_str(&scopes).map_err(|e| sqlx::Error::ColumnDecode {
                index: String::from("scopes"),
                source: Box::new(e),
            })?,
            created_at: row.try_get("created_at")?,
            expires_at: row.try_get("expires_at")?,
            last_used_at: row.try_get("last_used_at")?,
        })
    }
}

impl TokenRow {
    #[must_use]
    pub fn has_scope(&self, scope: Scope) -> bool {
        self.scopes.contains(&scope)
    }

    /// Whether `expires_at` is before `now`. An expiry that does not
    /// parse counts as passed, so a damaged row never grants access.
    #[must_use]
    pub fn is_expired(&self, now: Timestamp) -> bool {
        self.expires_at
            .as_deref()
            .is_some_and(|at| at.parse::<Expiry>().map_or(true, |at| at.0 < now))
    }
}

/// When a token stops working, as `control.db` stores it: UTC
/// `YYYY-MM-DD HH:MM:SS`, the form `SQLite`'s own `datetime()` writes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Expiry(Timestamp);

/// How `control.db` writes a moment.
const SQLITE_TIME: &str = "%Y-%m-%d %H:%M:%S";

impl Expiry {
    /// `days` from now.
    ///
    /// # Errors
    ///
    /// Returns an error when that is past the last moment `jiff` can hold.
    pub fn after_days(days: u32) -> Result<Self> {
        Timestamp::now()
            .checked_add(SignedDuration::from_hours(
                i64::from(days).saturating_mul(24),
            ))
            .map(Self)
            .map_err(|_| Error::Config(format!("an expiry {days} days away is too far off")))
    }
}

impl fmt::Display for Expiry {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0.strftime(SQLITE_TIME))
    }
}

impl FromStr for Expiry {
    type Err = jiff::Error;

    fn from_str(text: &str) -> std::result::Result<Self, Self::Err> {
        let at = jiff::fmt::strtime::parse(SQLITE_TIME, text)?.to_datetime()?;
        Ok(Self(at.to_zoned(jiff::tz::TimeZone::UTC)?.timestamp()))
    }
}

/// Where a request came from: the channel it arrived over, and the
/// client address and request id the server saw, when it recorded them.
#[derive(Debug, Clone, serde::Serialize)]
pub struct Origin {
    pub channel: Channel,
    pub client_addr: Option<String>,
    pub request_id: Option<String>,
}

/// A request over `channel` with no address or id: the CLI, or a row the
/// server writes before it has read either.
impl From<Channel> for Origin {
    fn from(channel: Channel) -> Self {
        Self {
            channel,
            client_addr: None,
            request_id: None,
        }
    }
}

impl Origin {
    /// The denied `session` row for `user`, when their sign-in has ended.
    #[must_use]
    pub fn denied_session(&self, user: &UserId) -> AuditEntry {
        let mut entry = AuditEntry::new(AuditAction::Session, Outcome::Denied, self.clone());
        entry.user_id = Some(user.clone());
        entry
    }
}

/// One access-audit row to record: who, what, outcome, origin.
#[derive(Debug, Clone, serde::Serialize)]
pub struct AuditEntry {
    /// UUID v7; the same id keys `_quack_audit` inside the workspace.
    pub id: AuditId,
    pub user_id: Option<UserId>,
    pub token_hash: Option<String>,
    pub workspace_id: Option<WorkspaceId>,
    pub action: AuditAction,
    pub resource_type: Option<ResourceKind>,
    /// An opaque id or a table name; never content.
    pub resource_id: Option<String>,
    pub outcome: Outcome,
    #[serde(flatten)]
    pub origin: Origin,
}

impl AuditEntry {
    /// A fresh entry with a new UUID v7 and nothing else set.
    #[must_use]
    pub fn new(action: AuditAction, outcome: Outcome, origin: impl Into<Origin>) -> Self {
        Self {
            id: AuditId::generate(),
            user_id: None,
            token_hash: None,
            workspace_id: None,
            action,
            resource_type: None,
            resource_id: None,
            outcome,
            origin: origin.into(),
        }
    }

    /// The workspace the action concerned.
    #[must_use]
    pub fn in_workspace(mut self, workspace_id: &WorkspaceId) -> Self {
        self.workspace_id = Some(workspace_id.clone());
        self
    }

    /// The resource the action touched.
    #[must_use]
    pub fn on(mut self, resource: AuditResource<'_>) -> Self {
        self.resource_type = Some(resource.kind);
        self.resource_id = Some(resource.id.to_owned());
        self
    }
}

/// What an access-audit row records was done. The log is history: a
/// stored name this build does not define reads back as `Unknown`, which
/// is never written.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(into = "String", from = "String")]
pub enum AuditAction {
    Login,
    Logout,
    /// A person changed their own password.
    Password,
    /// A session that expired or no longer exists was presented.
    Session,
    /// Server administration: users.
    Admin,
    Workspace,
    Member,
    /// An admin granted themself a role in a workspace they were not a
    /// member of: membership taken by admin right, with a reason.
    BreakGlass,
    Token,
    /// Opened one resource: a document, table, version, or session.
    Open,
    /// Listed resources of a kind.
    List,
    /// Rendered a web page.
    Page,
    Show,
    Stream,
    Query,
    Search,
    Sql,
    Ingest,
    Import,
    Export,
    Delete,
    Context,
    Ontology,
    Propose,
    Graph,
    GraphExtract,
    GraphReview,
    GraphRevalidate,
    GraphMerge,
    /// A workspace written out as a snapshot.
    Snapshot,
    /// A workspace restored from a snapshot.
    Restore,
    /// A person added, corrected, or deleted a graph node or edge.
    GraphEdit,
    EmbeddingsRefresh,
    EmbeddingsStatus,
    SessionRead,
    Share,
    Mode,
    Cancel,
    /// A person's decision on a write the agent wanted to run.
    Permission,
    /// An answer's SQL saved as a question.
    Save,
    /// A saved question's SQL run again without the model.
    SavedRun,
    /// A stored name this build does not define, as a newer build wrote
    /// it. Read only: the one write path refuses it.
    Unknown(String),
}

history_enum!(AuditAction, Unknown, {
    Login => "login",
    Logout => "logout",
    Password => "password",
    Session => "session",
    Admin => "admin",
    Workspace => "workspace",
    Member => "member",
    BreakGlass => "break_glass",
    Token => "token",
    Open => "open",
    List => "list",
    Page => "page",
    Show => "show",
    Stream => "stream",
    Query => "query",
    Search => "search",
    Sql => "sql",
    Ingest => "ingest",
    Import => "import",
    Export => "export",
    Delete => "delete",
    Context => "context",
    Ontology => "ontology",
    Propose => "propose",
    Graph => "graph",
    GraphExtract => "graph_extract",
    GraphReview => "graph_review",
    GraphRevalidate => "graph_revalidate",
    GraphMerge => "graph_merge",
    Snapshot => "snapshot",
    Restore => "restore",
    GraphEdit => "graph_edit",
    EmbeddingsRefresh => "embeddings_refresh",
    EmbeddingsStatus => "embeddings_status",
    SessionRead => "session_read",
    Share => "share",
    Mode => "mode",
    Cancel => "cancel",
    Permission => "permission",
    Save => "save",
    SavedRun => "saved_run",
});

/// The kinds of resource an audit row names by opaque id. Like
/// [`AuditAction`], a stored kind this build does not define reads back as
/// `Unknown` and is never written.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(into = "String", from = "String")]
pub enum ResourceKind {
    Session,
    /// A message in a session, named by its sequence number.
    Message,
    Document,
    /// A chunk of a document, named by the document and its position.
    Chunk,
    User,
    Workspace,
    Token,
    Job,
    Candidate,
    Context,
    OntologyVersion,
    InductionRun,
    GraphRun,
    GraphMerge,
    GraphNode,
    GraphEdge,
    EmbeddingsRun,
    /// An MCP resource URI.
    Resource,
    Audit,
    SavedQuestion,
    /// An identity provider's group, named in a workspace's group roles.
    Group,
    /// A stored name this build does not define, as a newer build wrote
    /// it. Read only: the one write path refuses it.
    Unknown(String),
}

history_enum!(ResourceKind, Unknown, {
    Session => "session",
    Message => "message",
    Document => "document",
    Chunk => "chunk",
    User => "user",
    Workspace => "workspace",
    Token => "token",
    Job => "job",
    Candidate => "candidate",
    Context => "context",
    OntologyVersion => "ontology_version",
    InductionRun => "induction_run",
    GraphRun => "graph_run",
    GraphMerge => "graph_merge",
    GraphNode => "graph_node",
    GraphEdge => "graph_edge",
    EmbeddingsRun => "embeddings_run",
    Resource => "resource",
    Audit => "audit",
    SavedQuestion => "saved_question",
    Group => "group",
});

impl ResourceKind {
    /// The text form with spaces, for a sentence: "ontology version".
    #[must_use]
    pub fn label(&self) -> String {
        self.as_str().replace('_', " ")
    }

    /// The error for this kind of resource with `id` missing.
    #[must_use]
    pub fn missing(self, id: impl Into<String>) -> Error {
        Error::NotFound {
            kind: self,
            id: id.into(),
        }
    }

    /// This kind of resource with `id`, as an audit row names it.
    #[must_use]
    pub fn id(self, id: &(impl AsRef<str> + ?Sized)) -> AuditResource<'_> {
        AuditResource {
            kind: self,
            id: id.as_ref(),
        }
    }
}

/// The resource an audit row names: its kind and opaque id, never content.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AuditResource<'a> {
    pub kind: ResourceKind,
    pub id: &'a str,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Outcome {
    Allowed,
    Denied,
    Error,
}

text_enum!(Outcome, "outcome", {
    Allowed => "allowed",
    Denied => "denied",
    Error => "error",
});

impl Outcome {
    /// `Allowed` for work that succeeded, `Error` for work that failed.
    #[must_use]
    pub fn of<T, E>(result: &std::result::Result<T, E>) -> Self {
        match result {
            Ok(_) => Self::Allowed,
            Err(_) => Self::Error,
        }
    }

    /// `Denied` for work that was refused, `Error` for any other failure.
    #[must_use]
    pub const fn of_failure(error: &Error) -> Self {
        if error.is_provider_refusal() {
            Self::Denied
        } else {
            Self::Error
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Channel {
    Web,
    Api,
    Mcp,
    Tui,
    Desktop,
    Cli,
}

text_enum!(Channel, "channel", {
    Web => "web",
    Api => "api",
    Mcp => "mcp",
    Tui => "tui",
    Desktop => "desktop",
    Cli => "cli",
});

/// A stored access-audit row: the entry as it was recorded, and when.
/// Serializes flat, the timestamp beside the entry's fields.
#[derive(Debug, Clone, serde::Serialize)]
pub struct AuditRow {
    pub timestamp: String,
    #[serde(flatten)]
    pub entry: AuditEntry,
}

impl FromRow<'_, SqliteRow> for AuditRow {
    fn from_row(row: &SqliteRow) -> sqlx::Result<Self> {
        Ok(Self {
            timestamp: row.try_get("timestamp")?,
            entry: AuditEntry {
                id: row.try_get("id")?,
                user_id: row.try_get("user_id")?,
                token_hash: row.try_get("token_hash")?,
                workspace_id: row.try_get("workspace_id")?,
                action: AuditAction::from(row.try_get::<String, _>("action")?),
                resource_type: row
                    .try_get::<Option<String>, _>("resource_type")?
                    .map(ResourceKind::from),
                resource_id: row.try_get("resource_id")?,
                outcome: parsed(row, "outcome")?,
                origin: Origin {
                    channel: parsed(row, "channel")?,
                    client_addr: row.try_get("client_addr")?,
                    request_id: row.try_get("request_id")?,
                },
            },
        })
    }
}

/// Filters for reading the audit log, as a query string or a form sends
/// them: every field is optional, a blank one is "any", and `limit`
/// defaults to 100 rows a page.
#[derive(Debug, Clone, serde::Deserialize)]
#[serde(default)]
pub struct AuditFilter {
    #[serde(deserialize_with = "blank_as_none")]
    pub user_id: Option<UserId>,
    #[serde(deserialize_with = "blank_as_none")]
    pub workspace_id: Option<WorkspaceId>,
    #[serde(deserialize_with = "blank_as_none")]
    pub action: Option<String>,
    #[serde(deserialize_with = "blank_as_none")]
    pub outcome: Option<Outcome>,
    /// Inclusive lower bound on `timestamp` (SQLite text form).
    #[serde(deserialize_with = "blank_as_none")]
    pub since: Option<String>,
    /// Exclusive upper bound on `timestamp`.
    #[serde(deserialize_with = "blank_as_none")]
    pub until: Option<String>,
    pub limit: u32,
    #[serde(deserialize_with = "blank_as_none")]
    pub cursor: Option<AuditCursor>,
}

impl Default for AuditFilter {
    fn default() -> Self {
        Self {
            user_id: None,
            workspace_id: None,
            action: None,
            outcome: None,
            since: None,
            until: None,
            limit: Self::DEFAULT_LIMIT,
            cursor: None,
        }
    }
}

/// A page of the audit log, newest first; `next` is `None` on the last page.
#[derive(Debug, Clone)]
pub struct AuditPage {
    pub rows: Vec<AuditRow>,
    pub next: Option<AuditCursor>,
}

/// Where a page ended: the last row's timestamp and id (the id breaks ties
/// between rows in the same second), and a digest of the filter it was
/// read under. Opaque to callers.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(into = "String", try_from = "String")]
pub struct AuditCursor {
    timestamp: String,
    id: AuditId,
    filter: String,
}

impl AuditCursor {
    const SEPARATOR: char = '\n';
}

impl fmt::Display for AuditCursor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let raw = format!(
            "{}{sep}{}{sep}{}",
            self.timestamp,
            self.id,
            self.filter,
            sep = Self::SEPARATOR
        );
        f.write_str(&base64::Engine::encode(
            &base64::engine::general_purpose::URL_SAFE_NO_PAD,
            raw,
        ))
    }
}

impl FromStr for AuditCursor {
    type Err = Error;

    fn from_str(text: &str) -> Result<Self> {
        let not_a_cursor = || Error::Config(String::from("not an audit cursor"));
        let raw = base64::Engine::decode(
            &base64::engine::general_purpose::URL_SAFE_NO_PAD,
            text.trim(),
        )
        .map_err(|_| not_a_cursor())?;
        let raw = String::from_utf8(raw).map_err(|_| not_a_cursor())?;
        let mut parts = raw.split(Self::SEPARATOR);
        match (parts.next(), parts.next(), parts.next(), parts.next()) {
            (Some(timestamp), Some(id), Some(filter), None) => Ok(Self {
                timestamp: timestamp.to_owned(),
                id: AuditId::from(id),
                filter: filter.to_owned(),
            }),
            _ => Err(not_a_cursor()),
        }
    }
}

impl From<AuditCursor> for String {
    fn from(cursor: AuditCursor) -> Self {
        cursor.to_string()
    }
}

impl TryFrom<String> for AuditCursor {
    type Error = Error;

    fn try_from(text: String) -> Result<Self> {
        text.parse()
    }
}

/// Fill `bytes` from the process's CSPRNG (aws-lc-rs).
///
/// # Errors
///
/// Returns an error when the random source fails.
pub fn random_bytes(bytes: &mut [u8]) -> Result<()> {
    aws_lc_rs::rand::fill(bytes)
        .map_err(|_| Error::Config(String::from("random generation failed")))
}

/// A password as `users.password_hash` stores it: argon2id with default
/// parameters. The hash never leaves this module.
struct StoredPasswordHash(String);

impl StoredPasswordHash {
    /// A valid argon2id hash of a random string, verified against when the
    /// username is unknown so login timing does not reveal which usernames
    /// exist.
    const DUMMY: &str = "$argon2id$v=19$m=19456,t=2,p=1$c2FsdHNhbHRzYWx0c2FsdA$Pm2ZmwZKKGUUtY5t1M2p5iP5B0KJ5FhBwa0zDf5Tr3Y";

    /// Hash `password`.
    fn new(password: &str) -> Result<Self> {
        Argon2::default()
            .hash_password(password.as_bytes())
            .map(|h| Self(h.to_string()))
            .map_err(|e| Error::Config(format!("password hashing failed: {e}")))
    }

    /// The stored hash, or [`Self::DUMMY`] for a user with none.
    fn stored_or_dummy(stored: Option<String>) -> Self {
        Self(stored.unwrap_or_else(|| String::from(Self::DUMMY)))
    }

    /// Whether `password` is the one hashed. A stored value that is not a
    /// hash verifies nothing.
    fn verifies(&self, password: &str) -> bool {
        PasswordHash::new(&self.0).is_ok_and(|parsed| {
            Argon2::default()
                .verify_password(password.as_bytes(), &parsed)
                .is_ok()
        })
    }
}

/// Whose sealed token a row is, which decides its table.
#[derive(Debug, Clone, Copy)]
pub enum SealedOwner<'a> {
    /// A signed-in user's identity-provider token (`user_tokens`, deleted
    /// with the user; `vault::Purpose::UserToken`).
    User(&'a UserId),
    /// A model provider's token from `quack auth login` or its
    /// client-credentials grant (`provider_tokens`;
    /// `vault::Purpose::ProviderToken`).
    Provider(&'a ProviderName),
    /// An OAuth client's `private_key_jwt` signing key, by key name
    /// (`<issuer> <client_id>`, or a key not yet in use: see
    /// `llm::oauth::client_key::ClientKeyName`; `client_keys`;
    /// `vault::Purpose::ClientKey`).
    ClientKey(&'a str),
}

impl SealedOwner<'_> {
    /// The table, its key column, this owner's key, and the column that
    /// records when the row was written.
    fn row(self) -> (DynIden, DynIden, String, DynIden) {
        match self {
            Self::User(user) => (
                UserTokens::Table.into_iden(),
                UserTokens::UserId.into_iden(),
                user.to_string(),
                SealedColumns::UpdatedAt.into_iden(),
            ),
            Self::Provider(provider) => (
                ProviderTokens::Table.into_iden(),
                ProviderTokens::Provider.into_iden(),
                provider.to_string(),
                SealedColumns::UpdatedAt.into_iden(),
            ),
            Self::ClientKey(name) => (
                ClientKeys::Table.into_iden(),
                ClientKeys::Name.into_iden(),
                name.to_owned(),
                ClientKeys::CreatedAt.into_iden(),
            ),
        }
    }
}

/// A client quack registered with an issuer itself (`client_registrations`,
/// RFC 7591): found by `name`, the issuer, before its `client_id` is known.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RegistrationRow {
    /// The issuer, without a trailing slash.
    pub name: String,
    pub client_id: String,
    /// Where RFC 7592 reads, updates, and deletes the registration, when
    /// the issuer said.
    pub registration_client_uri: Option<String>,
    /// The `registration_access_token`, sealed for
    /// `vault::Purpose::RegistrationToken`, when the issuer returned one.
    pub token: Option<Sealed>,
}

impl FromRow<'_, SqliteRow> for RegistrationRow {
    fn from_row(row: &SqliteRow) -> sqlx::Result<Self> {
        let key_id: Option<String> = row.try_get("key_id")?;
        let enc: Option<Vec<u8>> = row.try_get("enc")?;
        let ciphertext: Option<Vec<u8>> = row.try_get("ciphertext")?;
        Ok(Self {
            name: row.try_get("name")?,
            client_id: row.try_get("client_id")?,
            registration_client_uri: row.try_get("registration_client_uri")?,
            token: match (key_id, enc, ciphertext) {
                (Some(key_id), Some(enc), Some(ciphertext)) => Some(Sealed {
                    key_id,
                    enc,
                    ciphertext,
                }),
                _ => None,
            },
        })
    }
}

/// A change to the client keys that must land together with a
/// registration's (or on its own, all or nothing): the key to write under
/// its name, replacing any there, and the names whose keys to delete.
#[derive(Debug, Clone, Copy, Default)]
pub struct KeyChange<'a> {
    pub put: Option<(&'a str, &'a Sealed)>,
    pub delete: &'a [&'a str],
}

/// What a registration's name must still hold for
/// [`ControlPlane::save_registration`] to write.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Previous<'a> {
    /// No registration: a new one.
    Nothing,
    /// The registration of this client: an update or a replacement.
    Client(&'a str),
    /// Whatever is there.
    Any,
}

/// A sealed-token row: the sealed value, as `vault::Sealed` is.
impl FromRow<'_, SqliteRow> for Sealed {
    fn from_row(row: &SqliteRow) -> sqlx::Result<Self> {
        Ok(Self {
            key_id: row.try_get("key_id")?,
            enc: row.try_get("enc")?,
            ciphertext: row.try_get("ciphertext")?,
        })
    }
}

/// Manages the SQLite control plane database. Cloning shares the pool.
#[derive(Clone)]
pub struct ControlPlane {
    pool: SqlitePool,
}

impl ControlPlane {
    /// Open (or create) the control plane database and run migrations.
    ///
    /// # Errors
    ///
    /// Returns an error if the database cannot be opened or migrations fail.
    pub async fn open(config: &Config) -> Result<Self> {
        config.ensure_dirs()?;

        let db_path = config.control_db_path();
        let url = format!("sqlite:{}?mode=rwc", db_path.display());
        let options = SqliteConnectOptions::from_str(&url)?;

        let pool = SqlitePoolOptions::new()
            .max_connections(1)
            .connect_with(options)
            .await?;

        let cp = Self { pool };
        cp.run_migrations().await?;
        Ok(cp)
    }

    async fn run_migrations(&self) -> Result<()> {
        sqlx::query("PRAGMA journal_mode=WAL")
            .execute(&self.pool)
            .await?;
        sqlx::query("PRAGMA foreign_keys=ON")
            .execute(&self.pool)
            .await?;

        self.adopt_legacy_versions().await?;

        MIGRATOR.run(&self.pool).await.map_err(sqlx::Error::from)?;

        Ok(())
    }

    /// Record, without running, the versions a pre-sqlx binary already applied.
    ///
    /// Versions 1 to 3 were applied by sea-query DDL builders that tracked
    /// progress in a `schema_version` table. Replaying them is not idempotent:
    /// version 2 drops and recreates `audit_log`, which would destroy the
    /// access record. `schema_version` is left in place, inert, so an older
    /// binary opening the same file still sees its own versions as applied.
    async fn adopt_legacy_versions(&self) -> Result<()> {
        if self.table_exists("_sqlx_migrations").await?
            || !self.table_exists("schema_version").await?
        {
            return Ok(());
        }

        let legacy: i64 =
            sqlx::query_scalar("SELECT COALESCE(MAX(version), 0) FROM schema_version")
                .fetch_one(&self.pool)
                .await?;
        if legacy <= 0 {
            return Ok(());
        }

        tracing::info!(
            version = legacy,
            "adopting control.db created before sqlx migrations"
        );
        MIGRATOR
            .skip(&self.pool, Some(legacy))
            .await
            .map_err(sqlx::Error::from)?;
        Ok(())
    }

    async fn table_exists(&self, name: &str) -> Result<bool> {
        let found: Option<i64> =
            sqlx::query_scalar("SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?")
                .bind(name)
                .fetch_optional(&self.pool)
                .await?;
        Ok(found.is_some())
    }

    /// The newest schema version this binary carries: the version an open
    /// database reaches.
    #[must_use]
    pub fn latest_schema_version() -> i64 {
        MIGRATOR.iter().map(|m| m.version).max().unwrap_or(0)
    }

    /// Highest applied schema version.
    ///
    /// # Errors
    ///
    /// Returns an error if the query fails.
    pub async fn schema_version(&self) -> Result<i64> {
        Ok(sqlx::query_scalar(
            "SELECT COALESCE(MAX(version), 0) FROM _sqlx_migrations WHERE success",
        )
        .fetch_one(&self.pool)
        .await?)
    }

    // --- workspaces -------------------------------------------------------

    fn workspace_select() -> sea_query::SelectStatement {
        Query::select()
            .column(Workspaces::Id)
            .column(Workspaces::Name)
            .column(Workspaces::Classification)
            .column(Workspaces::AllowedProviders)
            .from(Workspaces::Table)
            .to_owned()
    }

    /// Look up a workspace by name.
    ///
    /// # Errors
    ///
    /// Returns an error if the query fails.
    pub async fn find_workspace_by_name(&self, name: &str) -> Result<Option<WorkspaceRow>> {
        let bound =
            Bound::new(Self::workspace_select().and_where(Expr::col(Workspaces::Name).eq(name)))?;
        Ok(bound.query_as().fetch_optional(&self.pool).await?)
    }

    /// Look up a workspace by id.
    ///
    /// # Errors
    ///
    /// Returns an error if the query fails.
    pub async fn get_workspace(&self, id: &WorkspaceId) -> Result<Option<WorkspaceRow>> {
        let bound =
            Bound::new(Self::workspace_select().and_where(Expr::col(Workspaces::Id).eq(id)))?;
        Ok(bound.query_as().fetch_optional(&self.pool).await?)
    }

    /// Create a new workspace, with `owner` as its owner when given, and
    /// record `audit` in the same transaction. The name's unique index
    /// decides whether it is taken, so two callers cannot both create it.
    ///
    /// # Errors
    ///
    /// Returns [`Error::WorkspaceExists`] when the name is taken, or an
    /// error if the insert fails; nothing is then written.
    pub async fn create_workspace(
        &self,
        name: &WorkspaceName,
        owner: Option<&UserId>,
        audit: AuditEntry,
    ) -> Result<WorkspaceRow> {
        let (ws, insert) = Self::new_workspace(name)?;
        let mut change = vec![insert];
        if let Some(owner) = owner {
            change.push(Self::member_insert(
                &ws.id,
                owner,
                Role::Owner,
                GrantedBy::User,
            )?);
        }
        let audit = audit
            .in_workspace(&ws.id)
            .on(ResourceKind::Workspace.id(ws.id.as_str()));
        self.commit_audited(change, audit)
            .await
            .map_err(|e| match &e {
                Error::Sqlite(sqlx::Error::Database(db)) if db.is_unique_violation() => {
                    Error::WorkspaceExists(name.as_str().to_owned())
                }
                _ => e,
            })?;
        tracing::info!(workspace_name = %name, workspace_id = %ws.id, "created workspace");
        Ok(ws)
    }

    /// One round trip to `control.db`, for a readiness check.
    ///
    /// # Errors
    ///
    /// Returns an error when the database does not answer.
    pub async fn ping(&self) -> Result<()> {
        sqlx::query("SELECT 1").execute(&self.pool).await?;
        Ok(())
    }

    /// The workspace called `name`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::NoWorkspaceNamed`] when there is none, or an error
    /// if the query fails.
    pub async fn workspace_named(&self, name: &str) -> Result<WorkspaceRow> {
        self.find_workspace_by_name(name)
            .await?
            .ok_or_else(|| Error::NoWorkspaceNamed(name.to_owned()))
    }

    /// The workspace a command line runs in: the one it `named`, which
    /// must exist, or `default`. The default is created, audited on the
    /// `cli` channel, the first time it is used, so a new install needs no
    /// setup step. When two processes use it first at once, one creates it
    /// and the other opens that row.
    ///
    /// # Errors
    ///
    /// Returns [`Error::NoWorkspaceNamed`] for a named workspace other than
    /// the default that does not exist, or an error if a query fails.
    pub async fn workspace_or_default(
        &self,
        named: Option<&str>,
        default: &WorkspaceName,
    ) -> Result<WorkspaceRow> {
        let name = named.unwrap_or(default.as_str());
        if name != default.as_str() {
            return self.workspace_named(name).await;
        }
        if let Some(ws) = self.find_workspace_by_name(name).await? {
            return Ok(ws);
        }
        let audit = AuditEntry::new(AuditAction::Workspace, Outcome::Allowed, Channel::Cli);
        match self.create_workspace(default, None, audit).await {
            Err(Error::WorkspaceExists(_)) => self.workspace_named(name).await,
            created => created,
        }
    }

    fn new_workspace(name: &WorkspaceName) -> Result<(WorkspaceRow, Bound)> {
        let id = WorkspaceId::generate();
        let insert = Bound::new(
            Query::insert()
                .into_table(Workspaces::Table)
                .columns([Workspaces::Id, Workspaces::Name, Workspaces::Classification])
                .values([id.clone().into(), name.as_str().into(), "internal".into()])?,
        )?;
        let ws = WorkspaceRow {
            id,
            name: name.as_str().to_owned(),
            classification: String::from("internal"),
            allowed_providers: AllowedProviders::All,
        };
        Ok((ws, insert))
    }

    /// Change a workspace's classification label and provider allow-list.
    /// Fields left `None` keep their value.
    ///
    /// # Errors
    ///
    /// Returns an error if the update fails or the workspace does not exist.
    pub async fn update_workspace(
        &self,
        id: &WorkspaceId,
        changes: &WorkspaceChanges,
    ) -> Result<WorkspaceRow> {
        // The builder is dropped before the await so the future stays Send.
        let bound = {
            let mut update = Query::update();
            update
                .table(Workspaces::Table)
                .value(Workspaces::UpdatedAt, Expr::cust("CURRENT_TIMESTAMP"))
                .and_where(Expr::col(Workspaces::Id).eq(id));
            if let Some(c) = &changes.classification {
                update.value(Workspaces::Classification, c.as_str());
            }
            match &changes.allowed_providers {
                ProviderAllowList::Keep => {}
                ProviderAllowList::All => {
                    update.value(Workspaces::AllowedProviders, Option::<String>::None);
                }
                ProviderAllowList::Only(names) => {
                    update.value(Workspaces::AllowedProviders, serde_json::to_string(names)?);
                }
            }
            Bound::new(&update)?
        };
        bound.query().execute(&self.pool).await?;
        self.get_workspace(id)
            .await?
            .ok_or_else(|| ResourceKind::Workspace.missing(id.to_string()))
    }

    /// Delete a workspace's row, which takes its members and API tokens
    /// with it, and commit the audit row with it. The files are the
    /// caller's to remove once nothing holds them.
    ///
    /// # Errors
    ///
    /// Returns an error when the workspace is missing or the write fails.
    pub async fn delete_workspace(&self, id: &WorkspaceId, audit: AuditEntry) -> Result<()> {
        let delete = Bound::new(
            Query::delete()
                .from_table(Workspaces::Table)
                .and_where(Expr::col(Workspaces::Id).eq(id)),
        )?;
        let audit = audit
            .in_workspace(id)
            .on(ResourceKind::Workspace.id(id.as_str()));
        if !self.commit_audited(vec![delete], audit).await? {
            return Err(ResourceKind::Workspace.missing(id.to_string()));
        }
        tracing::info!(workspace_id = %id, "deleted workspace");
        Ok(())
    }

    /// Give a workspace a new name, audited in the same transaction.
    ///
    /// # Errors
    ///
    /// Returns [`Error::WorkspaceExists`] when the name is taken, an error
    /// when the workspace is missing or the write fails.
    pub async fn rename_workspace(
        &self,
        id: &WorkspaceId,
        name: &WorkspaceName,
        audit: AuditEntry,
    ) -> Result<WorkspaceRow> {
        let update = Bound::new(
            Query::update()
                .table(Workspaces::Table)
                .value(Workspaces::Name, name.as_str())
                .value(Workspaces::UpdatedAt, Expr::cust("CURRENT_TIMESTAMP"))
                .and_where(Expr::col(Workspaces::Id).eq(id)),
        )?;
        let audit = audit
            .in_workspace(id)
            .on(ResourceKind::Workspace.id(id.as_str()));
        let changed = self
            .commit_audited(vec![update], audit)
            .await
            .map_err(|e| match &e {
                Error::Sqlite(sqlx::Error::Database(db)) if db.is_unique_violation() => {
                    Error::WorkspaceExists(name.as_str().to_owned())
                }
                _ => e,
            })?;
        if !changed {
            return Err(ResourceKind::Workspace.missing(id.to_string()));
        }
        self.get_workspace(id)
            .await?
            .ok_or_else(|| ResourceKind::Workspace.missing(id.to_string()))
    }

    /// List all workspaces.
    ///
    /// # Errors
    ///
    /// Returns an error if the query fails.
    pub async fn list_workspaces(&self) -> Result<Vec<WorkspaceRow>> {
        let bound = Bound::new(Self::workspace_select().order_by(Workspaces::Name, Order::Asc))?;
        Ok(bound.query_as().fetch_all(&self.pool).await?)
    }

    /// Workspaces the user is a member of, with the role.
    ///
    /// # Errors
    ///
    /// Returns an error if the query fails.
    pub async fn workspaces_for_user(&self, user_id: &UserId) -> Result<Vec<Membership>> {
        let bound = Bound::new(
            Query::select()
                .columns([
                    (Workspaces::Table, Workspaces::Id),
                    (Workspaces::Table, Workspaces::Name),
                    (Workspaces::Table, Workspaces::Classification),
                    (Workspaces::Table, Workspaces::AllowedProviders),
                ])
                .column((Members::Table, Members::Role))
                .from(Workspaces::Table)
                .inner_join(
                    Members::Table,
                    Expr::col((Members::Table, Members::WorkspaceId))
                        .equals((Workspaces::Table, Workspaces::Id)),
                )
                .and_where(Expr::col((Members::Table, Members::UserId)).eq(user_id))
                .order_by((Workspaces::Table, Workspaces::Name), Order::Asc),
        )?;
        let rows = bound.query().fetch_all(&self.pool).await?;
        rows.iter()
            .map(|r| {
                Ok(Membership {
                    workspace: WorkspaceRow::from_row(r)?,
                    standing: Standing::Member(parsed(r, "role")?),
                })
            })
            .collect()
    }

    /// When each workspace was created and last reached, by id. The last
    /// access is the newest `audit_log` row naming the workspace, allowed or
    /// denied, so each lookup is one seek on `audit_log_workspace_ts`.
    ///
    /// # Errors
    ///
    /// Returns an error if the query fails.
    pub async fn workspace_times(&self) -> Result<HashMap<WorkspaceId, WorkspaceTimes>> {
        let last_access = Query::select()
            .expr(Expr::col((AuditLog::Table, AuditLog::Timestamp)).max())
            .from(AuditLog::Table)
            .and_where(
                Expr::col((AuditLog::Table, AuditLog::WorkspaceId))
                    .equals((Workspaces::Table, Workspaces::Id)),
            )
            .to_owned();
        let bound = Bound::new(
            Query::select()
                .column((Workspaces::Table, Workspaces::Id))
                .column((Workspaces::Table, Workspaces::CreatedAt))
                .expr_as(
                    Expr::SubQuery(None, Box::new(last_access.into())),
                    "last_accessed_at",
                )
                .from(Workspaces::Table),
        )?;
        let rows = bound.query().fetch_all(&self.pool).await?;
        rows.iter()
            .map(|r| {
                Ok((
                    r.try_get("id")?,
                    WorkspaceTimes {
                        created_at: r.try_get("created_at")?,
                        last_accessed_at: r.try_get("last_accessed_at")?,
                    },
                ))
            })
            .collect()
    }

    // --- users --------------------------------------------------------------

    fn user_select() -> sea_query::SelectStatement {
        Query::select()
            .columns([
                Users::Id,
                Users::Username,
                Users::IsAdmin,
                Users::CreatedAt,
                Users::DisabledAt,
                Users::LockedUntil,
            ])
            .from(Users::Table)
            .to_owned()
    }

    /// One change to a user's row, bound before any await so the future
    /// stays `Send` (the builder is not).
    fn user_change(id: &UserId, change: &mut sea_query::UpdateStatement) -> Result<Bound> {
        Ok(Bound::new(
            change
                .table(Users::Table)
                .and_where(Expr::col(Users::Id).eq(id)),
        )?)
    }

    /// One change to a user's row, audited in the same transaction.
    async fn update_user(&self, id: &UserId, change: Bound, audit: AuditEntry) -> Result<()> {
        let audit = audit.on(ResourceKind::User.id(id.as_str()));
        if !self.commit_audited(vec![change], audit).await? {
            return Err(ResourceKind::User.missing(id.to_string()));
        }
        Ok(())
    }

    /// `user`, unless an admin disabled them: then a denied row under
    /// `action` from `origin`, and [`Error::AccountDisabled`], whatever the
    /// credential that named them.
    ///
    /// # Errors
    ///
    /// Returns [`Error::AccountDisabled`] for a disabled user, or an error
    /// when the denied row cannot be written.
    pub async fn admit(
        &self,
        user: UserRow,
        action: AuditAction,
        origin: &Origin,
    ) -> Result<UserRow> {
        if !user.is_disabled() {
            return Ok(user);
        }
        let mut entry = AuditEntry::new(action, Outcome::Denied, origin.clone());
        entry.user_id = Some(user.id.clone());
        self.record_audit(&entry).await?;
        Err(Error::AccountDisabled)
    }

    /// Refuse the user's logins and credentials from now on.
    ///
    /// # Errors
    ///
    /// Returns an error when the user is missing or the write fails.
    pub async fn disable_user(&self, id: &UserId, audit: AuditEntry) -> Result<()> {
        let change = Self::user_change(
            id,
            Query::update().value(Users::DisabledAt, Expr::cust("CURRENT_TIMESTAMP")),
        )?;
        self.update_user(id, change, audit).await?;
        tracing::info!(user_id = %id, "disabled user");
        Ok(())
    }

    /// Let a disabled user log in again, and clear any lockout.
    ///
    /// # Errors
    ///
    /// Returns an error when the user is missing or the write fails.
    pub async fn enable_user(&self, id: &UserId, audit: AuditEntry) -> Result<()> {
        let change = Self::user_change(
            id,
            Query::update()
                .value(Users::DisabledAt, Option::<String>::None)
                .value(Users::LockedUntil, Option::<String>::None)
                .value(Users::FailedLogins, 0_i64),
        )?;
        self.update_user(id, change, audit).await?;
        tracing::info!(user_id = %id, "enabled user");
        Ok(())
    }

    /// Give or take the server-wide admin flag.
    ///
    /// # Errors
    ///
    /// Returns an error when the user is missing or the write fails.
    pub async fn set_admin(&self, id: &UserId, kind: UserKind, audit: AuditEntry) -> Result<()> {
        let change = Self::user_change(
            id,
            Query::update().value(Users::IsAdmin, i64::from(bool::from(kind))),
        )?;
        self.update_user(id, change, audit).await
    }

    /// Replace the user's password; the caller ends their sessions.
    ///
    /// # Errors
    ///
    /// Returns an error for an empty password, a missing user, or a failed
    /// write.
    pub async fn set_password(&self, id: &UserId, password: &str, audit: AuditEntry) -> Result<()> {
        if password.is_empty() {
            return Err(Error::Config(String::from("password must not be empty")));
        }
        let password = password.to_owned();
        let hash = tokio::task::spawn_blocking(move || StoredPasswordHash::new(&password))
            .await
            .map_err(|e| Error::Config(format!("password hashing task failed: {e}")))??;
        let change = Self::user_change(
            id,
            Query::update()
                .value(Users::PasswordHash, hash.0.as_str())
                .value(Users::PasswordChangedAt, Expr::cust("CURRENT_TIMESTAMP"))
                .value(Users::FailedLogins, 0_i64)
                .value(Users::LockedUntil, Option::<String>::None),
        )?;
        self.update_user(id, change, audit).await
    }

    /// Delete a user: their memberships and stored sign-in go with the row,
    /// and their API tokens are deleted here, since that table carries no
    /// foreign key to users. The audit log keeps every row that names them.
    ///
    /// # Errors
    ///
    /// Returns an error when the user is missing or the write fails.
    pub async fn delete_user(&self, id: &UserId, audit: AuditEntry) -> Result<()> {
        let tokens = Bound::new(
            Query::delete()
                .from_table(ApiTokens::Table)
                .and_where(Expr::col(ApiTokens::UserId).eq(id)),
        )?;
        let user = Bound::new(
            Query::delete()
                .from_table(Users::Table)
                .and_where(Expr::col(Users::Id).eq(id)),
        )?;
        let audit = audit.on(ResourceKind::User.id(id.as_str()));
        // The token delete may touch no row; only the user's row decides.
        let mut tx = self.pool.begin().await?;
        tokens.query().execute(&mut *tx).await?;
        let deleted = user.query().execute(&mut *tx).await?.rows_affected() > 0;
        if !deleted {
            return Err(ResourceKind::User.missing(id.to_string()));
        }
        Self::audit_insert(&audit)?
            .query()
            .execute(&mut *tx)
            .await?;
        tx.commit().await?;
        tracing::info!(user_id = %id, "deleted user");
        Ok(())
    }

    /// Create a user with an argon2id password hash, and record `audit` in
    /// the same transaction.
    ///
    /// # Errors
    ///
    /// Returns an error for an empty username or password, a duplicate
    /// username, or a failed insert; nothing is then written.
    pub async fn create_user(
        &self,
        username: &str,
        password: &str,
        kind: UserKind,
        audit: AuditEntry,
    ) -> Result<UserRow> {
        let username = username.trim();
        if username.is_empty() {
            return Err(Error::Config(String::from("username must not be empty")));
        }
        if password.is_empty() {
            return Err(Error::Config(String::from("password must not be empty")));
        }
        let password = password.to_owned();
        let hash = tokio::task::spawn_blocking(move || StoredPasswordHash::new(&password))
            .await
            .map_err(|e| Error::Config(format!("password hashing task failed: {e}")))??;
        let id = UserId::generate();
        let bound = Bound::new(
            Query::insert()
                .into_table(Users::Table)
                .columns([
                    Users::Id,
                    Users::Username,
                    Users::PasswordHash,
                    Users::IsAdmin,
                ])
                .values([
                    (&id).into(),
                    username.into(),
                    hash.0.as_str().into(),
                    i64::from(bool::from(kind)).into(),
                ])?,
        )?;
        let audit = audit.on(ResourceKind::User.id(id.as_str()));
        self.commit_audited(vec![bound], audit)
            .await
            .map_err(|e| match &e {
                Error::Sqlite(sqlx::Error::Database(db)) if db.is_unique_violation() => {
                    Error::Config(format!("user '{username}' already exists"))
                }
                _ => e,
            })?;
        tracing::info!(username, ?kind, "created user");
        self.get_user(&id)
            .await?
            .ok_or_else(|| Error::Config(String::from("user vanished after insert")))
    }

    /// The user with the given id.
    ///
    /// # Errors
    ///
    /// Returns an error if the query fails.
    pub async fn get_user(&self, id: &UserId) -> Result<Option<UserRow>> {
        let bound = Bound::new(Self::user_select().and_where(Expr::col(Users::Id).eq(id)))?;
        Ok(bound.query_as().fetch_optional(&self.pool).await?)
    }

    /// The user with the given username.
    ///
    /// # Errors
    ///
    /// Returns an error if the query fails.
    pub async fn find_user_by_username(&self, username: &str) -> Result<Option<UserRow>> {
        let bound = Bound::new(
            Self::user_select().and_where(Expr::col(Users::Username).eq(username.trim())),
        )?;
        Ok(bound.query_as().fetch_optional(&self.pool).await?)
    }

    /// Every user, by name.
    ///
    /// # Errors
    ///
    /// Returns an error if the query fails.
    pub async fn list_users(&self) -> Result<Vec<UserRow>> {
        let bound = Bound::new(Self::user_select().order_by(Users::Username, Order::Asc))?;
        Ok(bound.query_as().fetch_all(&self.pool).await?)
    }

    /// The user a sign-in through the server's issuer belongs to.
    ///
    /// # Errors
    ///
    /// Returns an error if the query fails.
    pub async fn find_user_by_oidc_subject(
        &self,
        subject: &OidcSubject,
    ) -> Result<Option<UserRow>> {
        let bound = Bound::new(
            Self::user_select().and_where(Expr::col(Users::OidcSubject).eq(subject.as_str())),
        )?;
        Ok(bound.query_as().fetch_optional(&self.pool).await?)
    }

    /// The user for a sign-in through the server's issuer, created on the
    /// first one: no password, not an admin, and no workspace memberships
    /// until an owner adds them. The username is the issuer's name for them;
    /// when another user already has it, the new user gets it with a suffix,
    /// so a sign-in never takes over an account by name.
    ///
    /// # Errors
    ///
    /// Returns an error if the name is empty or a query fails.
    pub async fn oidc_user(&self, subject: &OidcSubject, username: &str) -> Result<UserRow> {
        if let Some(user) = self.find_user_by_oidc_subject(subject).await? {
            return Ok(user);
        }
        let username = username.trim();
        if username.is_empty() {
            return Err(Error::Config(String::from("username must not be empty")));
        }
        let id = UserId::generate();
        // The random tail of a UUID v7; its head is the creation time, shared
        // by users made in the same millisecond.
        let text = id.as_str();
        let suffix = text.get(text.len().saturating_sub(8)..).unwrap_or(text);
        for name in [username.to_owned(), format!("{username}-{suffix}")] {
            let bound = Bound::new(
                Query::insert()
                    .into_table(Users::Table)
                    .columns([
                        Users::Id,
                        Users::Username,
                        Users::OidcSubject,
                        Users::IsAdmin,
                    ])
                    .values([
                        (&id).into(),
                        name.as_str().into(),
                        subject.as_str().into(),
                        0_i64.into(),
                    ])?,
            )?;
            match bound.query().execute(&self.pool).await {
                Ok(_) => {
                    tracing::info!(username = %name, "created user on first sign-in");
                    return self
                        .get_user(&id)
                        .await?
                        .ok_or_else(|| Error::Config(String::from("user vanished after insert")));
                }
                Err(sqlx::Error::Database(db)) if db.is_unique_violation() => {
                    // A concurrent first sign-in by the same person got there
                    // first; otherwise the name is taken and the next one is
                    // tried.
                    if let Some(user) = self.find_user_by_oidc_subject(subject).await? {
                        return Ok(user);
                    }
                }
                Err(e) => return Err(e.into()),
            }
        }
        Err(Error::Config(format!(
            "no free username for '{username}'; both it and its suffixed form are taken"
        )))
    }

    /// The sealed token `owner` keeps, if any.
    ///
    /// # Errors
    ///
    /// Returns an error if the query fails.
    pub async fn sealed(&self, owner: SealedOwner<'_>) -> Result<Option<Sealed>> {
        let (table, key, id, _) = owner.row();
        let bound = Bound::new(
            Query::select()
                .columns([
                    SealedColumns::KeyId,
                    SealedColumns::Enc,
                    SealedColumns::Ciphertext,
                ])
                .from(table)
                .and_where(Expr::col(key).eq(id)),
        )?;
        Ok(bound.query_as().fetch_optional(&self.pool).await?)
    }

    /// Keep `owner`'s sealed token, replacing the one before it.
    ///
    /// # Errors
    ///
    /// Returns an error if the owner cannot hold one (a user that does not
    /// exist) or the write fails.
    pub async fn put_sealed(&self, owner: SealedOwner<'_>, sealed: &Sealed) -> Result<()> {
        self.insert_sealed(owner, sealed, true).await.map(drop)
    }

    /// Keep `owner`'s sealed value only when it has none yet; whether this
    /// one was kept. Two processes making the same key at once both call
    /// this, and both then use whichever row won.
    ///
    /// # Errors
    ///
    /// Returns an error if the owner cannot hold one or the write fails.
    pub async fn add_sealed(&self, owner: SealedOwner<'_>, sealed: &Sealed) -> Result<bool> {
        self.insert_sealed(owner, sealed, false).await
    }

    /// Replace `owner`'s sealed value only while it is still `expected`;
    /// whether this one was kept. Two processes replacing the same value at
    /// once both call this, and both then use whichever row won.
    ///
    /// # Errors
    ///
    /// Returns an error if the write fails.
    pub async fn replace_sealed(
        &self,
        owner: SealedOwner<'_>,
        expected: &Sealed,
        sealed: &Sealed,
    ) -> Result<bool> {
        let (table, key, id, stamp) = owner.row();
        let bound = Bound::new(
            Query::update()
                .table(table)
                .values([
                    (
                        SealedColumns::KeyId.into_iden(),
                        sealed.key_id.as_str().into(),
                    ),
                    (SealedColumns::Enc.into_iden(), sealed.enc.clone().into()),
                    (
                        SealedColumns::Ciphertext.into_iden(),
                        sealed.ciphertext.clone().into(),
                    ),
                    (stamp, Expr::current_timestamp()),
                ])
                .and_where(Expr::col(key).eq(id))
                .and_where(Expr::col(SealedColumns::KeyId).eq(expected.key_id.as_str()))
                .and_where(Expr::col(SealedColumns::Enc).eq(expected.enc.clone())),
        )?;
        let done = bound.query().execute(&self.pool).await?;
        Ok(done.rows_affected() > 0)
    }

    async fn insert_sealed(
        &self,
        owner: SealedOwner<'_>,
        sealed: &Sealed,
        replace: bool,
    ) -> Result<bool> {
        let done = Self::sealed_insert(owner, sealed, replace)?
            .query()
            .execute(&self.pool)
            .await?;
        Ok(done.rows_affected() > 0)
    }

    /// The insert that keeps `owner`'s sealed value: replacing the one
    /// before it, or only when there is none.
    fn sealed_insert(owner: SealedOwner<'_>, sealed: &Sealed, replace: bool) -> Result<Bound> {
        let (table, key, id, stamp) = owner.row();
        let conflict = if replace {
            OnConflict::column(key.clone())
                .update_columns([
                    SealedColumns::KeyId.into_iden(),
                    SealedColumns::Enc.into_iden(),
                    SealedColumns::Ciphertext.into_iden(),
                    stamp.clone(),
                ])
                .to_owned()
        } else {
            OnConflict::column(key.clone()).do_nothing().to_owned()
        };
        let bound = Bound::new(
            Query::insert()
                .into_table(table)
                .columns([
                    key,
                    SealedColumns::KeyId.into_iden(),
                    SealedColumns::Enc.into_iden(),
                    SealedColumns::Ciphertext.into_iden(),
                    stamp,
                ])
                .values([
                    id.into(),
                    sealed.key_id.as_str().into(),
                    sealed.enc.clone().into(),
                    sealed.ciphertext.clone().into(),
                    Expr::current_timestamp(),
                ])?
                .on_conflict(conflict),
        )?;
        Ok(bound)
    }

    /// The delete that forgets `owner`'s sealed value.
    fn sealed_delete(owner: SealedOwner<'_>) -> Result<Bound> {
        let (table, key, id, _) = owner.row();
        Ok(Bound::new(
            Query::delete()
                .from_table(table)
                .and_where(Expr::col(key).eq(id)),
        )?)
    }

    /// Forget `owner`'s sealed token; one without is already done.
    ///
    /// # Errors
    ///
    /// Returns an error if the delete fails.
    pub async fn delete_sealed(&self, owner: SealedOwner<'_>) -> Result<()> {
        Self::sealed_delete(owner)?
            .query()
            .execute(&self.pool)
            .await?;
        Ok(())
    }

    // --- client registrations (RFC 7591) ------------------------------------

    /// The client registered at the issuer `name`, if any.
    ///
    /// # Errors
    ///
    /// Returns an error if the query fails.
    pub async fn registration(&self, name: &str) -> Result<Option<RegistrationRow>> {
        let bound = Bound::new(
            Query::select()
                .columns([
                    ClientRegistrations::Name.into_iden(),
                    ClientRegistrations::ClientId.into_iden(),
                    ClientRegistrations::RegistrationClientUri.into_iden(),
                    SealedColumns::KeyId.into_iden(),
                    SealedColumns::Enc.into_iden(),
                    SealedColumns::Ciphertext.into_iden(),
                ])
                .from(ClientRegistrations::Table)
                .and_where(Expr::col(ClientRegistrations::Name).eq(name)),
        )?;
        Ok(bound.query_as().fetch_optional(&self.pool).await?)
    }

    /// Every kept registration: quack's clients, one per issuer, and the
    /// records of temporary sign-in clients not yet deleted.
    ///
    /// # Errors
    ///
    /// Returns an error if the query fails.
    pub async fn registrations(&self) -> Result<Vec<RegistrationRow>> {
        let bound = Bound::new(
            Query::select()
                .columns([
                    ClientRegistrations::Name.into_iden(),
                    ClientRegistrations::ClientId.into_iden(),
                    ClientRegistrations::RegistrationClientUri.into_iden(),
                    SealedColumns::KeyId.into_iden(),
                    SealedColumns::Enc.into_iden(),
                    SealedColumns::Ciphertext.into_iden(),
                ])
                .from(ClientRegistrations::Table)
                .order_by(ClientRegistrations::Name, Order::Asc),
        )?;
        Ok(bound.query_as().fetch_all(&self.pool).await?)
    }

    /// Keep `row` at its name and apply `keys` in the same transaction, but
    /// only while the name still holds what the caller read: nothing
    /// ([`Previous::Nothing`]), the client it names ([`Previous::Client`]),
    /// or anything ([`Previous::Any`]). Returns `false`, having changed
    /// nothing, when another writer got there first, so two registrations
    /// racing each other never overwrite one another's record.
    ///
    /// # Errors
    ///
    /// Returns an error if a write fails; nothing is then changed.
    pub async fn save_registration(
        &self,
        row: &RegistrationRow,
        previous: Previous<'_>,
        keys: KeyChange<'_>,
    ) -> Result<bool> {
        let (key_id, enc, ciphertext) = match &row.token {
            Some(sealed) => (
                Some(sealed.key_id.clone()),
                Some(sealed.enc.clone()),
                Some(sealed.ciphertext.clone()),
            ),
            None => (None, None, None),
        };
        let values = [
            row.name.as_str().into(),
            row.client_id.as_str().into(),
            row.registration_client_uri.clone().into(),
            key_id.clone().into(),
            enc.clone().into(),
            ciphertext.clone().into(),
            Expr::current_timestamp(),
        ];
        let columns = [
            ClientRegistrations::Name.into_iden(),
            ClientRegistrations::ClientId.into_iden(),
            ClientRegistrations::RegistrationClientUri.into_iden(),
            SealedColumns::KeyId.into_iden(),
            SealedColumns::Enc.into_iden(),
            SealedColumns::Ciphertext.into_iden(),
            SealedColumns::UpdatedAt.into_iden(),
        ];
        let updated = [
            ClientRegistrations::ClientId.into_iden(),
            ClientRegistrations::RegistrationClientUri.into_iden(),
            SealedColumns::KeyId.into_iden(),
            SealedColumns::Enc.into_iden(),
            SealedColumns::Ciphertext.into_iden(),
            SealedColumns::UpdatedAt.into_iden(),
        ];
        let write = match previous {
            Previous::Nothing => Bound::new(
                Query::insert()
                    .into_table(ClientRegistrations::Table)
                    .columns(columns)
                    .values(values)?
                    .on_conflict(
                        OnConflict::column(ClientRegistrations::Name)
                            .do_nothing()
                            .to_owned(),
                    ),
            )?,
            Previous::Any => Bound::new(
                Query::insert()
                    .into_table(ClientRegistrations::Table)
                    .columns(columns)
                    .values(values)?
                    .on_conflict(
                        OnConflict::column(ClientRegistrations::Name)
                            .update_columns(updated)
                            .to_owned(),
                    ),
            )?,
            Previous::Client(client_id) => Bound::new(
                Query::update()
                    .table(ClientRegistrations::Table)
                    .values([
                        (
                            ClientRegistrations::ClientId.into_iden(),
                            row.client_id.as_str().into(),
                        ),
                        (
                            ClientRegistrations::RegistrationClientUri.into_iden(),
                            row.registration_client_uri.clone().into(),
                        ),
                        (SealedColumns::KeyId.into_iden(), key_id.into()),
                        (SealedColumns::Enc.into_iden(), enc.into()),
                        (SealedColumns::Ciphertext.into_iden(), ciphertext.into()),
                        (
                            SealedColumns::UpdatedAt.into_iden(),
                            Expr::current_timestamp(),
                        ),
                    ])
                    .and_where(Expr::col(ClientRegistrations::Name).eq(row.name.as_str()))
                    .and_where(Expr::col(ClientRegistrations::ClientId).eq(client_id)),
            )?,
        };
        let mut tx = self.pool.begin().await?;
        if write.query().execute(&mut *tx).await?.rows_affected() == 0 {
            return Ok(false);
        }
        for statement in Self::key_statements(keys)? {
            statement.query().execute(&mut *tx).await?;
        }
        tx.commit().await?;
        Ok(true)
    }

    /// Forget the registration at the issuer `name` and apply `keys`, in one
    /// transaction; a registration that is not there is already forgotten.
    ///
    /// # Errors
    ///
    /// Returns an error if a delete fails; nothing is then changed.
    pub async fn delete_registration(&self, name: &str, keys: KeyChange<'_>) -> Result<()> {
        let delete = Bound::new(
            Query::delete()
                .from_table(ClientRegistrations::Table)
                .and_where(Expr::col(ClientRegistrations::Name).eq(name)),
        )?;
        let mut statements = vec![delete];
        statements.extend(Self::key_statements(keys)?);
        self.in_transaction(statements).await
    }

    /// Apply `keys` alone, all or nothing: a key moved from one name to
    /// another is never in both places, nor in neither.
    ///
    /// # Errors
    ///
    /// Returns an error if a write fails; nothing is then changed.
    pub async fn change_client_keys(&self, keys: KeyChange<'_>) -> Result<()> {
        self.in_transaction(Self::key_statements(keys)?).await
    }

    fn key_statements(keys: KeyChange<'_>) -> Result<Vec<Bound>> {
        let mut statements = Vec::new();
        for name in keys.delete {
            statements.push(Self::sealed_delete(SealedOwner::ClientKey(name))?);
        }
        if let Some((name, sealed)) = keys.put {
            statements.push(Self::sealed_insert(
                SealedOwner::ClientKey(name),
                sealed,
                true,
            )?);
        }
        Ok(statements)
    }

    async fn in_transaction(&self, statements: Vec<Bound>) -> Result<()> {
        let mut tx = self.pool.begin().await?;
        for statement in statements {
            statement.query().execute(&mut *tx).await?;
        }
        tx.commit().await?;
        Ok(())
    }

    /// Check a password login. Returns the user on success; `None` for an
    /// unknown user, a user without a password, or a wrong password. The
    /// unknown-user path still runs a hash verification so the two cases
    /// take the same time.
    ///
    /// # Errors
    ///
    /// Returns an error if the query fails.
    pub async fn verify_password(&self, username: &str, password: &str) -> Result<Option<UserRow>> {
        match self
            .check_password(username, password, Lockout::default())
            .await?
        {
            PasswordCheck::Verified(user) => Ok(Some(user)),
            PasswordCheck::Wrong(_) | PasswordCheck::Disabled(_) | PasswordCheck::Locked { .. } => {
                Ok(None)
            }
        }
    }

    /// Check a password under `lockout`, and record the outcome on the
    /// user's row: a wrong one counts toward the lock, a right one clears
    /// the count. A disabled or locked account is refused before the hash
    /// is checked; the dummy hash is still verified when there is no such
    /// user, so timing says nothing about who exists.
    ///
    /// # Errors
    ///
    /// Returns an error if a query fails.
    pub async fn check_password(
        &self,
        username: &str,
        password: &str,
        lockout: Lockout,
    ) -> Result<PasswordCheck> {
        let bound = Bound::new(
            Query::select()
                .columns([
                    Users::Id,
                    Users::PasswordHash,
                    Users::DisabledAt,
                    Users::LockedUntil,
                    Users::FailedLogins,
                ])
                .from(Users::Table)
                .and_where(Expr::col(Users::Username).eq(username.trim())),
        )?;
        let row = bound.query().fetch_optional(&self.pool).await?;
        let mut found = None;
        let mut hash = None;
        if let Some(r) = row {
            let id: UserId = r.try_get("id")?;
            let disabled: Option<String> = r.try_get("disabled_at")?;
            let locked: Option<String> = r.try_get("locked_until")?;
            let failed: i64 = r.try_get("failed_logins")?;
            hash = r.try_get("password_hash")?;
            if disabled.is_some() {
                return Ok(PasswordCheck::Disabled(id));
            }
            if let Some(until) = locked.filter(|until| Self::still_locked(until)) {
                return Ok(PasswordCheck::Locked { user_id: id, until });
            }
            found = Some((id, u32::try_from(failed).unwrap_or(u32::MAX)));
        }
        let password = password.to_owned();
        let hash = StoredPasswordHash::stored_or_dummy(hash);
        let ok = tokio::task::spawn_blocking(move || hash.verifies(&password))
            .await
            .map_err(|e| Error::Config(format!("password verification task failed: {e}")))?;
        let Some((id, failed)) = found else {
            return Ok(PasswordCheck::Wrong(None));
        };
        if ok {
            if failed > 0 {
                self.set_failed_logins(&id, 0, None).await?;
            }
            return match self.get_user(&id).await? {
                Some(user) => Ok(PasswordCheck::Verified(user)),
                None => Ok(PasswordCheck::Wrong(None)),
            };
        }
        let failed = failed.saturating_add(1);
        let until = lockout.locks_after(failed).then(|| {
            Timestamp::now()
                .checked_add(SignedDuration::from_mins(i64::from(lockout.minutes)))
                .unwrap_or(Timestamp::MAX)
                .to_string()
        });
        self.set_failed_logins(&id, failed, until.as_deref())
            .await?;
        match until {
            Some(until) => {
                tracing::warn!(user_id = %id, until, "account locked after repeated wrong passwords");
                Ok(PasswordCheck::Locked { user_id: id, until })
            }
            None => Ok(PasswordCheck::Wrong(Some(id))),
        }
    }

    /// Whether a stored `locked_until` is still in the future.
    fn still_locked(until: &str) -> bool {
        until
            .parse::<Timestamp>()
            .is_ok_and(|until| until > Timestamp::now())
    }

    /// The login counter and lock, not audited: the login itself is.
    async fn set_failed_logins(&self, id: &UserId, failed: u32, until: Option<&str>) -> Result<()> {
        let bound = Bound::new(
            Query::update()
                .table(Users::Table)
                .value(Users::FailedLogins, i64::from(failed))
                .value(Users::LockedUntil, until)
                .and_where(Expr::col(Users::Id).eq(id)),
        )?;
        bound.query().execute(&self.pool).await?;
        Ok(())
    }

    // --- members ------------------------------------------------------------

    /// Add or change a membership, and record `audit` in the same
    /// transaction.
    ///
    /// # Errors
    ///
    /// Returns an error if the workspace or user does not exist or the
    /// write fails; nothing is then written.
    pub async fn set_member(
        &self,
        workspace_id: &WorkspaceId,
        user_id: &UserId,
        role: Role,
        audit: AuditEntry,
    ) -> Result<()> {
        let existing = self.member_role(workspace_id, user_id).await?;
        let bound = if existing.is_some() {
            Bound::new(
                Query::update()
                    .table(Members::Table)
                    .value(Members::Role, role.as_str())
                    .value(Members::GrantedBy, GrantedBy::User.as_str())
                    .and_where(Expr::col(Members::WorkspaceId).eq(workspace_id))
                    .and_where(Expr::col(Members::UserId).eq(user_id)),
            )?
        } else {
            Self::member_insert(workspace_id, user_id, role, GrantedBy::User)?
        };
        let audit = audit
            .in_workspace(workspace_id)
            .on(ResourceKind::User.id(user_id.as_str()));
        self.commit_audited(vec![bound], audit)
            .await
            .map_err(|e| match &e {
                Error::Sqlite(sqlx::Error::Database(db)) if db.is_foreign_key_violation() => {
                    Error::Config(String::from(
                        "membership needs an existing workspace and user",
                    ))
                }
                _ => e,
            })?;
        Ok(())
    }

    fn member_insert(
        workspace_id: &WorkspaceId,
        user_id: &UserId,
        role: Role,
        granted_by: GrantedBy,
    ) -> Result<Bound> {
        Ok(Bound::new(
            Query::insert()
                .into_table(Members::Table)
                .columns([
                    Members::WorkspaceId,
                    Members::UserId,
                    Members::Role,
                    Members::GrantedBy,
                ])
                .values([
                    workspace_id.into(),
                    user_id.into(),
                    role.as_str().into(),
                    granted_by.as_str().into(),
                ])?,
        )?)
    }

    /// Remove a membership, and record `audit` in the same transaction
    /// (as `Error` when there was none). Returns whether one existed.
    ///
    /// # Errors
    ///
    /// Returns an error if the delete fails; nothing is then written.
    pub async fn remove_member(
        &self,
        workspace_id: &WorkspaceId,
        user_id: &UserId,
        audit: AuditEntry,
    ) -> Result<bool> {
        let bound = Bound::new(
            Query::delete()
                .from_table(Members::Table)
                .and_where(Expr::col(Members::WorkspaceId).eq(workspace_id))
                .and_where(Expr::col(Members::UserId).eq(user_id)),
        )?;
        let audit = audit
            .in_workspace(workspace_id)
            .on(ResourceKind::User.id(user_id.as_str()));
        self.commit_audited(vec![bound], audit).await
    }

    /// The user's role in a workspace, if a member.
    ///
    /// # Errors
    ///
    /// Returns an error if the query fails.
    pub async fn member_role(
        &self,
        workspace_id: &WorkspaceId,
        user_id: &UserId,
    ) -> Result<Option<Role>> {
        let bound = Bound::new(
            Query::select()
                .column(Members::Role)
                .from(Members::Table)
                .and_where(Expr::col(Members::WorkspaceId).eq(workspace_id))
                .and_where(Expr::col(Members::UserId).eq(user_id)),
        )?;
        let row = bound.query().fetch_optional(&self.pool).await?;
        Ok(row.map(|r| parsed(&r, "role")).transpose()?)
    }

    /// Members of a workspace with their usernames.
    ///
    /// # Errors
    ///
    /// Returns an error if the query fails.
    pub async fn list_members(&self, workspace_id: &WorkspaceId) -> Result<Vec<MemberRow>> {
        let bound = Bound::new(
            Query::select()
                .columns([
                    (Members::Table, Members::WorkspaceId),
                    (Members::Table, Members::UserId),
                    (Members::Table, Members::Role),
                    (Members::Table, Members::CreatedAt),
                    (Members::Table, Members::GrantedBy),
                ])
                .column((Users::Table, Users::Username))
                .from(Members::Table)
                .inner_join(
                    Users::Table,
                    Expr::col((Users::Table, Users::Id)).equals((Members::Table, Members::UserId)),
                )
                .and_where(Expr::col((Members::Table, Members::WorkspaceId)).eq(workspace_id))
                .order_by((Users::Table, Users::Username), Order::Asc),
        )?;
        Ok(bound.query_as().fetch_all(&self.pool).await?)
    }

    // --- group roles --------------------------------------------------------

    /// Give an identity provider's group a role in the workspace, or change
    /// it, audited in the same transaction. Members already signed in get
    /// it at their next sign-in.
    ///
    /// # Errors
    ///
    /// Returns an error for an empty group name, a missing workspace, or a
    /// failed write.
    pub async fn set_group_role(
        &self,
        workspace_id: &WorkspaceId,
        group: &str,
        role: Role,
        audit: AuditEntry,
    ) -> Result<GroupRoleRow> {
        let group = group.trim();
        if group.is_empty() {
            return Err(Error::Config(String::from("group name must not be empty")));
        }
        let bound = Bound::new(
            Query::insert()
                .into_table(GroupRoles::Table)
                .columns([
                    GroupRoles::WorkspaceId,
                    GroupRoles::GroupName,
                    GroupRoles::Role,
                ])
                .values([workspace_id.into(), group.into(), role.as_str().into()])?
                .on_conflict(
                    OnConflict::columns([GroupRoles::WorkspaceId, GroupRoles::GroupName])
                        .update_column(GroupRoles::Role)
                        .to_owned(),
                ),
        )?;
        let audit = audit
            .in_workspace(workspace_id)
            .on(ResourceKind::Group.id(group));
        self.commit_audited(vec![bound], audit)
            .await
            .map_err(|e| match &e {
                Error::Sqlite(sqlx::Error::Database(db)) if db.is_foreign_key_violation() => {
                    ResourceKind::Workspace.missing(workspace_id.to_string())
                }
                _ => e,
            })?;
        let bound = Bound::new(
            Self::group_role_select()
                .and_where(Expr::col(GroupRoles::WorkspaceId).eq(workspace_id))
                .and_where(Expr::col(GroupRoles::GroupName).eq(group)),
        )?;
        bound
            .query_as()
            .fetch_optional(&self.pool)
            .await?
            .ok_or_else(|| ResourceKind::Group.missing(group))
    }

    /// Take a group's role away; whether it had one. Memberships it granted
    /// go at each member's next sign-in.
    ///
    /// # Errors
    ///
    /// Returns an error if the write fails.
    pub async fn remove_group_role(
        &self,
        workspace_id: &WorkspaceId,
        group: &str,
        audit: AuditEntry,
    ) -> Result<bool> {
        let group = group.trim();
        let bound = Bound::new(
            Query::delete()
                .from_table(GroupRoles::Table)
                .and_where(Expr::col(GroupRoles::WorkspaceId).eq(workspace_id))
                .and_where(Expr::col(GroupRoles::GroupName).eq(group)),
        )?;
        let audit = audit
            .in_workspace(workspace_id)
            .on(ResourceKind::Group.id(group));
        self.commit_audited(vec![bound], audit).await
    }

    fn group_role_select() -> sea_query::SelectStatement {
        Query::select()
            .columns([
                GroupRoles::WorkspaceId,
                GroupRoles::GroupName,
                GroupRoles::Role,
                GroupRoles::CreatedAt,
            ])
            .from(GroupRoles::Table)
            .to_owned()
    }

    /// The groups with a role in the workspace, by name.
    ///
    /// # Errors
    ///
    /// Returns an error if the query fails.
    pub async fn list_group_roles(&self, workspace_id: &WorkspaceId) -> Result<Vec<GroupRoleRow>> {
        let bound = Bound::new(
            Self::group_role_select()
                .and_where(Expr::col(GroupRoles::WorkspaceId).eq(workspace_id))
                .order_by(GroupRoles::GroupName, Order::Asc),
        )?;
        Ok(bound.query_as().fetch_all(&self.pool).await?)
    }

    /// Make the user's provider-granted memberships match `groups`, each
    /// change a `member` row from `origin`: in
    /// each workspace, the highest role among the groups with one, else
    /// none. Only rows granted by the provider are added, re-roled, or
    /// deleted; a row a person granted is never touched. Every change
    /// commits with its own `member` audit row in one transaction.
    ///
    /// # Errors
    ///
    /// Returns an error if a query or the commit fails.
    pub async fn reconcile_idp_memberships(
        &self,
        user: &UserRow,
        groups: &[String],
        origin: &Origin,
    ) -> Result<Reconciled> {
        let user_id = &user.id;
        let audit = || {
            let mut entry = AuditEntry::new(AuditAction::Member, Outcome::Allowed, origin.clone());
            entry.user_id = Some(user_id.clone());
            entry
        };
        let wanted = self.roles_for_groups(groups).await?;
        let bound = Bound::new(
            Query::select()
                .columns([Members::WorkspaceId, Members::Role, Members::GrantedBy])
                .from(Members::Table)
                .and_where(Expr::col(Members::UserId).eq(user_id)),
        )?;
        let mut current: BTreeMap<WorkspaceId, (Role, GrantedBy)> = BTreeMap::new();
        for row in bound.query().fetch_all(&self.pool).await? {
            current.insert(
                row.try_get("workspace_id")?,
                (parsed(&row, "role")?, parsed(&row, "granted_by")?),
            );
        }
        let mut changes: Vec<(Bound, AuditEntry)> = Vec::new();
        let mut outcome = Reconciled::default();
        let entry = |workspace: &WorkspaceId| {
            audit()
                .in_workspace(workspace)
                .on(ResourceKind::User.id(user_id.as_str()))
        };
        for (workspace, role) in &wanted {
            match current.get(workspace) {
                Some((_, GrantedBy::User)) => {}
                Some((have, GrantedBy::Idp)) if have == role => {}
                Some((_, GrantedBy::Idp)) => {
                    changes.push((
                        Bound::new(
                            Query::update()
                                .table(Members::Table)
                                .value(Members::Role, role.as_str())
                                .and_where(Expr::col(Members::WorkspaceId).eq(workspace))
                                .and_where(Expr::col(Members::UserId).eq(user_id)),
                        )?,
                        entry(workspace),
                    ));
                    outcome.changed = outcome.changed.saturating_add(1);
                }
                None => {
                    changes.push((
                        Self::member_insert(workspace, user_id, *role, GrantedBy::Idp)?,
                        entry(workspace),
                    ));
                    outcome.granted = outcome.granted.saturating_add(1);
                }
            }
        }
        for (workspace, (_, granted_by)) in &current {
            if *granted_by == GrantedBy::Idp && !wanted.contains_key(workspace) {
                changes.push((
                    Bound::new(
                        Query::delete()
                            .from_table(Members::Table)
                            .and_where(Expr::col(Members::WorkspaceId).eq(workspace))
                            .and_where(Expr::col(Members::UserId).eq(user_id)),
                    )?,
                    entry(workspace),
                ));
                outcome.revoked = outcome.revoked.saturating_add(1);
            }
        }
        if changes.is_empty() {
            return Ok(outcome);
        }
        let mut tx = self.pool.begin().await?;
        for (change, audit) in changes {
            change.query().execute(&mut *tx).await?;
            Self::audit_insert(&audit)?
                .query()
                .execute(&mut *tx)
                .await?;
        }
        tx.commit().await?;
        tracing::info!(user = %user.username, ?outcome, "memberships follow the identity provider's groups");
        Ok(outcome)
    }

    /// The highest role `groups` carry in each workspace.
    async fn roles_for_groups(&self, groups: &[String]) -> Result<BTreeMap<WorkspaceId, Role>> {
        let mut wanted = BTreeMap::new();
        if groups.is_empty() {
            return Ok(wanted);
        }
        let bound = Bound::new(
            Query::select()
                .columns([GroupRoles::WorkspaceId, GroupRoles::Role])
                .from(GroupRoles::Table)
                .and_where(
                    Expr::col(GroupRoles::GroupName).is_in(groups.iter().map(String::as_str)),
                ),
        )?;
        for row in bound.query().fetch_all(&self.pool).await? {
            let workspace: WorkspaceId = row.try_get("workspace_id")?;
            let role: Role = parsed(&row, "role")?;
            wanted
                .entry(workspace)
                .and_modify(|have: &mut Role| *have = (*have).max(role))
                .or_insert(role);
        }
        Ok(wanted)
    }

    // --- API tokens ---------------------------------------------------------

    fn token_select() -> sea_query::SelectStatement {
        Query::select()
            .columns([
                ApiTokens::TokenHash,
                ApiTokens::WorkspaceId,
                ApiTokens::UserId,
                ApiTokens::Name,
                ApiTokens::Scopes,
                ApiTokens::CreatedAt,
                ApiTokens::ExpiresAt,
                ApiTokens::LastUsedAt,
            ])
            .from(ApiTokens::Table)
            .to_owned()
    }

    /// Mint an API token scoped to one workspace, and record `audit` in the
    /// same transaction. Returns the token text, which is never stored, and
    /// the row.
    ///
    /// # Errors
    ///
    /// Returns an error if the workspace or user does not exist or the
    /// insert fails; nothing is then written.
    pub async fn create_token(
        &self,
        workspace_id: &WorkspaceId,
        user_id: &UserId,
        name: &str,
        scopes: &[Scope],
        expires_at: Option<Expiry>,
        audit: AuditEntry,
    ) -> Result<IssuedToken> {
        let mut secret = [0u8; 32];
        random_bytes(&mut secret)?;
        let token = format!(
            "qk_{}",
            base64::Engine::encode(&base64::engine::general_purpose::URL_SAFE_NO_PAD, secret)
        );
        let hash = sha256_hex(token.as_bytes());
        let scopes = if scopes.is_empty() {
            vec![Scope::Read]
        } else {
            scopes.to_vec()
        };
        let bound = Bound::new(
            Query::insert()
                .into_table(ApiTokens::Table)
                .columns([
                    ApiTokens::TokenHash,
                    ApiTokens::WorkspaceId,
                    ApiTokens::UserId,
                    ApiTokens::Name,
                    ApiTokens::Scopes,
                    ApiTokens::ExpiresAt,
                ])
                .values([
                    hash.as_str().into(),
                    workspace_id.into(),
                    user_id.into(),
                    name.into(),
                    serde_json::to_string(&scopes)?.into(),
                    expires_at.map(|at| at.to_string()).into(),
                ])?,
        )?;
        let audit = audit
            .in_workspace(workspace_id)
            .on(ResourceKind::Token.id(&hash));
        self.commit_audited(vec![bound], audit)
            .await
            .map_err(|e| match &e {
                Error::Sqlite(sqlx::Error::Database(db)) if db.is_foreign_key_violation() => {
                    Error::Config(String::from("a token needs an existing workspace and user"))
                }
                _ => e,
            })?;
        let row = self
            .find_token(&hash)
            .await?
            .ok_or_else(|| Error::Config(String::from("token vanished after insert")))?;
        Ok(IssuedToken {
            secret: TokenSecret(token),
            row,
        })
    }

    /// The token row for a presented token, by its hash.
    ///
    /// # Errors
    ///
    /// Returns an error if the query fails.
    pub async fn find_token(&self, token_hash: &str) -> Result<Option<TokenRow>> {
        let bound = Bound::new(
            Self::token_select().and_where(Expr::col(ApiTokens::TokenHash).eq(token_hash)),
        )?;
        Ok(bound.query_as().fetch_optional(&self.pool).await?)
    }

    /// Record that a token was just used.
    ///
    /// # Errors
    ///
    /// Returns an error if the update fails.
    pub async fn touch_token(&self, token_hash: &str) -> Result<()> {
        let bound = Bound::new(
            Query::update()
                .table(ApiTokens::Table)
                .value(ApiTokens::LastUsedAt, Expr::cust("CURRENT_TIMESTAMP"))
                .and_where(Expr::col(ApiTokens::TokenHash).eq(token_hash)),
        )?;
        bound.query().execute(&self.pool).await?;
        Ok(())
    }

    /// Tokens for a workspace, newest first.
    ///
    /// # Errors
    ///
    /// Returns an error if the query fails.
    pub async fn list_tokens(&self, workspace_id: &WorkspaceId) -> Result<Vec<TokenRow>> {
        let bound = Bound::new(
            Self::token_select()
                .and_where(Expr::col(ApiTokens::WorkspaceId).eq(workspace_id))
                .order_by(ApiTokens::CreatedAt, Order::Desc),
        )?;
        Ok(bound.query_as().fetch_all(&self.pool).await?)
    }

    /// Revoke a token, and record `audit` in the same transaction (as
    /// `Error` when there was none). Returns whether it existed.
    ///
    /// # Errors
    ///
    /// Returns an error if the delete fails; nothing is then written.
    pub async fn delete_token(&self, token_hash: &str, audit: AuditEntry) -> Result<bool> {
        let bound = Bound::new(
            Query::delete()
                .from_table(ApiTokens::Table)
                .and_where(Expr::col(ApiTokens::TokenHash).eq(token_hash)),
        )?;
        let audit = audit.on(ResourceKind::Token.id(token_hash));
        self.commit_audited(vec![bound], audit).await
    }

    // --- access audit (append-only) ------------------------------------------

    /// Append one row to `audit_log`. There is no update or delete path.
    ///
    /// # Errors
    ///
    /// Returns an error if the insert fails.
    pub async fn record_audit(&self, entry: &AuditEntry) -> Result<()> {
        Self::audit_insert(entry)?
            .query()
            .execute(&self.pool)
            .await?;
        Ok(())
    }

    /// Run `change`, then `audit`'s row, in one transaction: a change never
    /// stands unaudited. A change that touched no row is audited as `Error`.
    /// Returns whether every statement touched a row.
    async fn commit_audited(&self, change: Vec<Bound>, mut audit: AuditEntry) -> Result<bool> {
        let mut tx = self.pool.begin().await?;
        let mut changed = true;
        for statement in change {
            changed &= statement.query().execute(&mut *tx).await?.rows_affected() > 0;
        }
        if !changed {
            audit.outcome = Outcome::Error;
        }
        Self::audit_insert(&audit)?
            .query()
            .execute(&mut *tx)
            .await?;
        tx.commit().await?;
        Ok(changed)
    }

    fn audit_insert(entry: &AuditEntry) -> Result<Bound> {
        // Only what this build defines is written; a name read from history
        // never goes back in under this build's name.
        if !entry.action.is_defined() {
            return Err(Error::UnknownValue {
                what: "audit action",
                value: entry.action.as_str().to_owned(),
                allowed: AuditAction::ALL
                    .iter()
                    .map(AuditAction::as_str)
                    .collect::<Vec<_>>()
                    .join(", "),
            });
        }
        if let Some(kind) = entry.resource_type.as_ref().filter(|k| !k.is_defined()) {
            return Err(Error::UnknownValue {
                what: "resource kind",
                value: kind.as_str().to_owned(),
                allowed: ResourceKind::ALL
                    .iter()
                    .map(ResourceKind::as_str)
                    .collect::<Vec<_>>()
                    .join(", "),
            });
        }
        Ok(Bound::new(
            Query::insert()
                .into_table(AuditLog::Table)
                .columns([
                    AuditLog::Id,
                    AuditLog::UserId,
                    AuditLog::TokenHash,
                    AuditLog::WorkspaceId,
                    AuditLog::Action,
                    AuditLog::ResourceType,
                    AuditLog::ResourceId,
                    AuditLog::Outcome,
                    AuditLog::Channel,
                    AuditLog::ClientAddr,
                    AuditLog::RequestId,
                ])
                .values([
                    (&entry.id).into(),
                    entry.user_id.as_ref().map(UserId::as_str).into(),
                    entry.token_hash.as_deref().into(),
                    entry.workspace_id.as_ref().map(WorkspaceId::as_str).into(),
                    entry.action.as_str().into(),
                    entry
                        .resource_type
                        .as_ref()
                        .map(ResourceKind::as_str)
                        .into(),
                    entry.resource_id.as_deref().into(),
                    entry.outcome.as_str().into(),
                    entry.origin.channel.as_str().into(),
                    entry.origin.client_addr.as_deref().into(),
                    entry.origin.request_id.as_deref().into(),
                ])?,
        )?)
    }

    /// The access rows with these ids, for joining a workspace's detail
    /// rows to them; ids with no row are left out.
    ///
    /// # Errors
    ///
    /// Returns an error if the query fails.
    pub async fn audit_rows_by_ids(&self, ids: &[AuditId]) -> Result<Vec<AuditRow>> {
        if ids.is_empty() {
            return Ok(Vec::new());
        }
        let bound = AuditFilter::by_ids(ids)?;
        Ok(bound.query_as().fetch_all(&self.pool).await?)
    }

    /// A workspace's detail rows joined to their access rows by the shared
    /// id, as OCSF events in the details' order; a detail row whose access
    /// row is gone is left out.
    ///
    /// # Errors
    ///
    /// Returns an error if the query fails or a stored timestamp does not
    /// parse.
    pub async fn ocsf_events(
        &self,
        details: &[AuditDetailRow],
        prompt: PromptText,
    ) -> Result<Vec<serde_json::Value>> {
        let ids: Vec<AuditId> = details.iter().map(|d| d.id.clone()).collect();
        let rows = self.audit_rows_by_ids(&ids).await?;
        let by_id: HashMap<&AuditId, &AuditRow> = rows.iter().map(|r| (&r.entry.id, r)).collect();
        let mut events = Vec::with_capacity(details.len());
        for detail in details {
            if let Some(row) = by_id.get(&detail.id) {
                events.push(row.to_ocsf_with_detail(Some(detail), prompt)?);
            }
        }
        Ok(events)
    }

    /// One page of the access audit under `filter`, newest first.
    ///
    /// # Errors
    ///
    /// Returns an error when the cursor belongs to another filter or the
    /// query fails.
    pub async fn query_audit(&self, filter: &AuditFilter) -> Result<AuditPage> {
        let digest = filter.digest();
        if let Some(after) = &filter.cursor
            && after.filter != digest
        {
            return Err(Error::Config(String::from(
                "the audit cursor belongs to a different filter; ask again without it",
            )));
        }
        let limit = usize::try_from(filter.page_size()).unwrap_or(usize::MAX);
        let mut rows: Vec<AuditRow> = filter.select()?.query_as().fetch_all(&self.pool).await?;
        let next = if rows.len() > limit {
            rows.truncate(limit);
            rows.last().map(|last| AuditCursor {
                timestamp: last.timestamp.clone(),
                id: last.entry.id.clone(),
                filter: digest,
            })
        } else {
            None
        };
        Ok(AuditPage { rows, next })
    }
}

impl AuditFilter {
    pub const DEFAULT_LIMIT: u32 = 100;
    /// Rows a page holds at most, whatever `limit` asks.
    pub const MAX_LIMIT: u32 = 1_000;

    fn page_size(&self) -> u32 {
        self.limit.clamp(1, Self::MAX_LIMIT)
    }

    /// Identifies the filter so a cursor only continues the same query.
    fn digest(&self) -> String {
        let fields = [
            self.user_id.as_ref().map(UserId::as_str),
            self.workspace_id.as_ref().map(WorkspaceId::as_str),
            self.action.as_deref(),
            self.outcome.map(Outcome::as_str),
            self.since.as_deref(),
            self.until.as_deref(),
        ];
        let canonical = serde_json::json!(fields).to_string();
        let mut hex = sha256_hex(canonical.as_bytes());
        hex.truncate(16);
        hex
    }

    /// The access rows with these ids, in no order: the half of a
    /// workspace's audit that `control.db` holds, joined by the detail
    /// rows' ids.
    pub(crate) fn by_ids(ids: &[AuditId]) -> sqlx::Result<Bound> {
        Bound::new(
            Query::select()
                .columns([
                    AuditLog::Id,
                    AuditLog::Timestamp,
                    AuditLog::UserId,
                    AuditLog::TokenHash,
                    AuditLog::WorkspaceId,
                    AuditLog::Action,
                    AuditLog::ResourceType,
                    AuditLog::ResourceId,
                    AuditLog::Outcome,
                    AuditLog::Channel,
                    AuditLog::ClientAddr,
                    AuditLog::RequestId,
                ])
                .from(AuditLog::Table)
                .and_where(Expr::col(AuditLog::Id).is_in(ids.iter().map(AuditId::as_str))),
        )
    }

    /// The filtered audit select, built and bound in one scope so no builder
    /// lives across an await. One extra row says whether a next page exists.
    fn select(&self) -> sqlx::Result<Bound> {
        let filter = self;
        let mut select = Query::select();
        select
            .columns([
                AuditLog::Id,
                AuditLog::Timestamp,
                AuditLog::UserId,
                AuditLog::TokenHash,
                AuditLog::WorkspaceId,
                AuditLog::Action,
                AuditLog::ResourceType,
                AuditLog::ResourceId,
                AuditLog::Outcome,
                AuditLog::Channel,
                AuditLog::ClientAddr,
                AuditLog::RequestId,
            ])
            .from(AuditLog::Table)
            .order_by(AuditLog::Timestamp, Order::Desc)
            .order_by(AuditLog::Id, Order::Desc)
            .limit(u64::from(filter.page_size()).saturating_add(1));
        if let Some(after) = &filter.cursor {
            select.cond_where(
                Cond::any()
                    .add(Expr::col(AuditLog::Timestamp).lt(after.timestamp.as_str()))
                    .add(
                        Cond::all()
                            .add(Expr::col(AuditLog::Timestamp).eq(after.timestamp.as_str()))
                            .add(Expr::col(AuditLog::Id).lt(after.id.as_str())),
                    ),
            );
        }
        if let Some(v) = &filter.user_id {
            select.and_where(Expr::col(AuditLog::UserId).eq(v.as_str()));
        }
        if let Some(v) = &filter.workspace_id {
            select.and_where(Expr::col(AuditLog::WorkspaceId).eq(v.as_str()));
        }
        if let Some(v) = &filter.action {
            select.and_where(Expr::col(AuditLog::Action).eq(v.as_str()));
        }
        if let Some(v) = filter.outcome {
            select.and_where(Expr::col(AuditLog::Outcome).eq(v.as_str()));
        }
        if let Some(v) = &filter.since {
            select.and_where(Expr::col(AuditLog::Timestamp).gte(v.as_str()));
        }
        if let Some(v) = &filter.until {
            select.and_where(Expr::col(AuditLog::Timestamp).lt(v.as_str()));
        }
        Bound::new(&select)
    }
}

#[cfg(test)]
mod tests;
