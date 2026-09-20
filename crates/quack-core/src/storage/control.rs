//! `control.db`: who may open which workspace, and the access audit.
//!
//! Users, workspaces (name and label), membership, API tokens, and the
//! append-only `audit_log`. Nothing here can reveal workspace content
//! (design doc 5.5); the detail of what was done lives in `_quack_audit`
//! inside the workspace file, keyed by the same UUID v7.

use argon2::Argon2;
use argon2::password_hash::phc::PasswordHash;
use argon2::password_hash::{PasswordHasher, PasswordVerifier};
use sea_query::{Expr, ExprTrait, Order, Query, SqliteQueryBuilder};
use sqlx::sqlite::{SqliteConnectOptions, SqlitePoolOptions};
use sqlx::{AssertSqlSafe, Row, SqlitePool};
use std::str::FromStr;

use super::queries::{ApiTokens, AuditLog, Members, Users, Workspaces};
use crate::config::Config;
use crate::error::{Error, Result};

/// The `control.db` schema, as plain SQL files embedded at compile time.
///
/// One file per version under `crates/quack-core/migrations/`. A file that has
/// shipped is frozen: sqlx records a SHA-384 checksum per version in
/// `_sqlx_migrations` and refuses a database whose recorded checksum no longer
/// matches. Migrations are never regenerated from the `Iden` enums in
/// `storage::queries`, because those track the current schema, not history
/// (`docs/migrations.md`).
static MIGRATOR: sqlx::migrate::Migrator = sqlx::migrate!("./migrations");

/// A workspace row from the control plane.
#[derive(Debug, Clone, serde::Serialize)]
pub struct WorkspaceRow {
    pub id: String,
    pub name: String,
    pub classification: String,
    /// JSON array of provider names, or `None` for all.
    pub allowed_providers: Option<String>,
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
    Only(Vec<String>),
}

/// A server user. The password hash never leaves this module.
#[derive(Debug, Clone, serde::Serialize)]
pub struct UserRow {
    pub id: String,
    pub username: String,
    pub is_admin: bool,
    pub created_at: String,
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

impl Role {
    /// Parse `viewer`, `member`, or `owner`.
    ///
    /// # Errors
    ///
    /// Returns a `Config` error for any other text.
    pub fn parse(text: &str) -> Result<Self> {
        match text.trim().to_ascii_lowercase().as_str() {
            "viewer" => Ok(Self::Viewer),
            "member" => Ok(Self::Member),
            "owner" => Ok(Self::Owner),
            other => Err(Error::Config(format!(
                "unknown role '{other}'; use viewer, member, or owner"
            ))),
        }
    }

    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Viewer => "viewer",
            Self::Member => "member",
            Self::Owner => "owner",
        }
    }
}

impl std::fmt::Display for Role {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// One membership, with the username for listings.
#[derive(Debug, Clone, serde::Serialize)]
pub struct MemberRow {
    pub workspace_id: String,
    pub user_id: String,
    pub username: String,
    pub role: Role,
    pub created_at: String,
}

/// What an API token may do (design doc 12).
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Scope {
    Read,
    Write,
    Admin,
}

impl Scope {
    /// Parse `read`, `write`, or `admin`.
    ///
    /// # Errors
    ///
    /// Returns a `Config` error for any other text.
    pub fn parse(text: &str) -> Result<Self> {
        match text.trim().to_ascii_lowercase().as_str() {
            "read" => Ok(Self::Read),
            "write" => Ok(Self::Write),
            "admin" => Ok(Self::Admin),
            other => Err(Error::Config(format!(
                "unknown scope '{other}'; use read, write, or admin"
            ))),
        }
    }
}

/// An API token row; the token itself is shown once at creation.
#[derive(Debug, Clone, serde::Serialize)]
pub struct TokenRow {
    pub token_hash: String,
    pub workspace_id: String,
    pub user_id: String,
    pub name: String,
    pub scopes: Vec<Scope>,
    pub created_at: String,
    pub expires_at: Option<String>,
    pub last_used_at: Option<String>,
}

impl TokenRow {
    #[must_use]
    pub fn has_scope(&self, scope: Scope) -> bool {
        self.scopes.contains(&scope)
    }

    /// Whether `expires_at` is in the past.
    #[must_use]
    pub fn is_expired(&self, now: &str) -> bool {
        self.expires_at.as_deref().is_some_and(|e| e < now)
    }
}

/// One access-audit row to record: who, what, outcome, channel.
#[derive(Debug, Clone)]
pub struct AuditEntry {
    /// UUID v7; the same id keys `_quack_audit` inside the workspace.
    pub id: String,
    pub user_id: Option<String>,
    pub token_hash: Option<String>,
    pub workspace_id: Option<String>,
    pub action: String,
    pub resource_type: Option<String>,
    /// An opaque id or a table name; never content.
    pub resource_id: Option<String>,
    pub outcome: Outcome,
    pub channel: Channel,
    pub client_addr: Option<String>,
    pub request_id: Option<String>,
}

impl AuditEntry {
    /// A fresh entry with a new UUID v7 and nothing else set.
    #[must_use]
    pub fn new(action: &str, outcome: Outcome, channel: Channel) -> Self {
        Self {
            id: uuid::Uuid::now_v7().to_string(),
            user_id: None,
            token_hash: None,
            workspace_id: None,
            action: action.to_owned(),
            resource_type: None,
            resource_id: None,
            outcome,
            channel,
            client_addr: None,
            request_id: None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Outcome {
    Allowed,
    Denied,
    Error,
}

impl Outcome {
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Allowed => "allowed",
            Self::Denied => "denied",
            Self::Error => "error",
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

impl Channel {
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Web => "web",
            Self::Api => "api",
            Self::Mcp => "mcp",
            Self::Tui => "tui",
            Self::Desktop => "desktop",
            Self::Cli => "cli",
        }
    }
}

/// A stored access-audit row.
#[derive(Debug, Clone, serde::Serialize)]
pub struct AuditRow {
    pub id: String,
    pub timestamp: String,
    pub user_id: Option<String>,
    pub token_hash: Option<String>,
    pub workspace_id: Option<String>,
    pub action: String,
    pub resource_type: Option<String>,
    pub resource_id: Option<String>,
    pub outcome: String,
    pub channel: String,
    pub client_addr: Option<String>,
    pub request_id: Option<String>,
}

/// Filters for reading the audit log; every field is optional.
#[derive(Debug, Clone, Default)]
pub struct AuditFilter {
    pub user_id: Option<String>,
    pub workspace_id: Option<String>,
    pub action: Option<String>,
    pub outcome: Option<String>,
    /// Inclusive lower bound on `timestamp` (SQLite text form).
    pub since: Option<String>,
    /// Exclusive upper bound on `timestamp`.
    pub until: Option<String>,
    pub limit: u32,
}

/// Lowercase hex SHA-256, the form tokens are stored in.
#[must_use]
pub fn sha256_hex(bytes: &[u8]) -> String {
    let digest = aws_lc_rs::digest::digest(&aws_lc_rs::digest::SHA256, bytes);
    let mut out = String::with_capacity(64);
    for b in digest.as_ref() {
        out.push(char::from_digit(u32::from(b >> 4), 16).unwrap_or('0'));
        out.push(char::from_digit(u32::from(b & 0x0f), 16).unwrap_or('0'));
    }
    out
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

/// Hash a password with argon2id and default parameters.
///
/// # Errors
///
/// Returns an error when hashing fails (out of memory).
pub fn hash_password(password: &str) -> Result<String> {
    Argon2::default()
        .hash_password(password.as_bytes())
        .map(|h| h.to_string())
        .map_err(|e| Error::Config(format!("password hashing failed: {e}")))
}

fn verify_password_hash(password: &str, hash: &str) -> bool {
    PasswordHash::new(hash).is_ok_and(|parsed| {
        Argon2::default()
            .verify_password(password.as_bytes(), &parsed)
            .is_ok()
    })
}

/// Manages the SQLite control plane database.
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

    fn workspace_from_row(r: &sqlx::sqlite::SqliteRow) -> Result<WorkspaceRow> {
        Ok(WorkspaceRow {
            id: r.try_get("id")?,
            name: r.try_get("name")?,
            classification: r.try_get("classification")?,
            allowed_providers: r.try_get("allowed_providers")?,
        })
    }

    /// Look up a workspace by name.
    ///
    /// # Errors
    ///
    /// Returns an error if the query fails.
    pub async fn find_workspace_by_name(&self, name: &str) -> Result<Option<WorkspaceRow>> {
        let sql = Self::workspace_select()
            .and_where(Expr::col(Workspaces::Name).eq(name))
            .to_string(SqliteQueryBuilder);
        let row = sqlx::query(AssertSqlSafe(sql.as_str()))
            .fetch_optional(&self.pool)
            .await?;
        row.as_ref().map(Self::workspace_from_row).transpose()
    }

    /// Look up a workspace by id.
    ///
    /// # Errors
    ///
    /// Returns an error if the query fails.
    pub async fn get_workspace(&self, id: &str) -> Result<Option<WorkspaceRow>> {
        let sql = Self::workspace_select()
            .and_where(Expr::col(Workspaces::Id).eq(id))
            .to_string(SqliteQueryBuilder);
        let row = sqlx::query(AssertSqlSafe(sql.as_str()))
            .fetch_optional(&self.pool)
            .await?;
        row.as_ref().map(Self::workspace_from_row).transpose()
    }

    /// Create a new workspace and return its row.
    ///
    /// # Errors
    ///
    /// Returns an error if the insert fails (a duplicate name included).
    pub async fn create_workspace(&self, name: &str) -> Result<WorkspaceRow> {
        let id = uuid::Uuid::now_v7().to_string();

        let sql = Query::insert()
            .into_table(Workspaces::Table)
            .columns([Workspaces::Id, Workspaces::Name, Workspaces::Classification])
            .values([id.as_str().into(), name.into(), "internal".into()])?
            .to_string(SqliteQueryBuilder);

        sqlx::query(AssertSqlSafe(sql.as_str()))
            .execute(&self.pool)
            .await?;

        tracing::info!(workspace_name = name, workspace_id = %id, "created workspace");

        Ok(WorkspaceRow {
            id,
            name: name.to_owned(),
            classification: String::from("internal"),
            allowed_providers: None,
        })
    }

    /// Find a workspace by name, creating it if it does not exist.
    ///
    /// # Errors
    ///
    /// Returns an error if the lookup or creation fails.
    pub async fn find_or_create_workspace(&self, name: &str) -> Result<WorkspaceRow> {
        if let Some(ws) = self.find_workspace_by_name(name).await? {
            return Ok(ws);
        }
        self.create_workspace(name).await
    }

    /// Change a workspace's classification label and provider allow-list.
    /// Fields left `None` keep their value.
    ///
    /// # Errors
    ///
    /// Returns an error if the update fails or the workspace does not exist.
    pub async fn update_workspace(
        &self,
        id: &str,
        changes: &WorkspaceChanges,
    ) -> Result<WorkspaceRow> {
        // The builder is dropped before the await so the future stays Send.
        let sql = {
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
            update.to_string(SqliteQueryBuilder)
        };
        sqlx::query(AssertSqlSafe(sql.as_str()))
            .execute(&self.pool)
            .await?;
        self.get_workspace(id)
            .await?
            .ok_or_else(|| Error::WorkspaceNotFound(id.to_owned()))
    }

    /// List all workspaces.
    ///
    /// # Errors
    ///
    /// Returns an error if the query fails.
    pub async fn list_workspaces(&self) -> Result<Vec<WorkspaceRow>> {
        let sql = Self::workspace_select()
            .order_by(Workspaces::Name, Order::Asc)
            .to_string(SqliteQueryBuilder);
        let rows = sqlx::query(AssertSqlSafe(sql.as_str()))
            .fetch_all(&self.pool)
            .await?;
        rows.iter().map(Self::workspace_from_row).collect()
    }

    /// Workspaces the user is a member of, with the role.
    ///
    /// # Errors
    ///
    /// Returns an error if the query fails.
    pub async fn workspaces_for_user(&self, user_id: &str) -> Result<Vec<(WorkspaceRow, Role)>> {
        let sql = Query::select()
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
            .order_by((Workspaces::Table, Workspaces::Name), Order::Asc)
            .to_string(SqliteQueryBuilder);
        let rows = sqlx::query(AssertSqlSafe(sql.as_str()))
            .fetch_all(&self.pool)
            .await?;
        rows.iter()
            .map(|r| {
                let role: String = r.try_get("role")?;
                Ok((Self::workspace_from_row(r)?, Role::parse(&role)?))
            })
            .collect()
    }

    // --- users --------------------------------------------------------------

    fn user_from_row(r: &sqlx::sqlite::SqliteRow) -> Result<UserRow> {
        let is_admin: i64 = r.try_get("is_admin")?;
        Ok(UserRow {
            id: r.try_get("id")?,
            username: r.try_get("username")?,
            is_admin: is_admin != 0,
            created_at: r.try_get("created_at")?,
        })
    }

    fn user_select() -> sea_query::SelectStatement {
        Query::select()
            .columns([Users::Id, Users::Username, Users::IsAdmin, Users::CreatedAt])
            .from(Users::Table)
            .to_owned()
    }

    /// Create a user with an argon2id password hash.
    ///
    /// # Errors
    ///
    /// Returns an error for an empty username or password, a duplicate
    /// username, or a failed insert.
    pub async fn create_user(
        &self,
        username: &str,
        password: &str,
        is_admin: bool,
    ) -> Result<UserRow> {
        let username = username.trim();
        if username.is_empty() {
            return Err(Error::Config(String::from("username must not be empty")));
        }
        if password.is_empty() {
            return Err(Error::Config(String::from("password must not be empty")));
        }
        let password = password.to_owned();
        let hash = tokio::task::spawn_blocking(move || hash_password(&password))
            .await
            .map_err(|e| Error::Config(format!("password hashing task failed: {e}")))??;
        let id = uuid::Uuid::now_v7().to_string();
        let sql = Query::insert()
            .into_table(Users::Table)
            .columns([
                Users::Id,
                Users::Username,
                Users::PasswordHash,
                Users::IsAdmin,
            ])
            .values([
                id.as_str().into(),
                username.into(),
                hash.as_str().into(),
                i64::from(is_admin).into(),
            ])?
            .to_string(SqliteQueryBuilder);
        sqlx::query(AssertSqlSafe(sql.as_str()))
            .execute(&self.pool)
            .await
            .map_err(|e| match &e {
                sqlx::Error::Database(db) if db.is_unique_violation() => {
                    Error::Config(format!("user '{username}' already exists"))
                }
                _ => e.into(),
            })?;
        tracing::info!(username, is_admin, "created user");
        self.get_user(&id)
            .await?
            .ok_or_else(|| Error::Config(String::from("user vanished after insert")))
    }

    /// The user with the given id.
    ///
    /// # Errors
    ///
    /// Returns an error if the query fails.
    pub async fn get_user(&self, id: &str) -> Result<Option<UserRow>> {
        let sql = Self::user_select()
            .and_where(Expr::col(Users::Id).eq(id))
            .to_string(SqliteQueryBuilder);
        let row = sqlx::query(AssertSqlSafe(sql.as_str()))
            .fetch_optional(&self.pool)
            .await?;
        row.as_ref().map(Self::user_from_row).transpose()
    }

    /// The user with the given username.
    ///
    /// # Errors
    ///
    /// Returns an error if the query fails.
    pub async fn find_user_by_username(&self, username: &str) -> Result<Option<UserRow>> {
        let sql = Self::user_select()
            .and_where(Expr::col(Users::Username).eq(username.trim()))
            .to_string(SqliteQueryBuilder);
        let row = sqlx::query(AssertSqlSafe(sql.as_str()))
            .fetch_optional(&self.pool)
            .await?;
        row.as_ref().map(Self::user_from_row).transpose()
    }

    /// Every user, by name.
    ///
    /// # Errors
    ///
    /// Returns an error if the query fails.
    pub async fn list_users(&self) -> Result<Vec<UserRow>> {
        let sql = Self::user_select()
            .order_by(Users::Username, Order::Asc)
            .to_string(SqliteQueryBuilder);
        let rows = sqlx::query(AssertSqlSafe(sql.as_str()))
            .fetch_all(&self.pool)
            .await?;
        rows.iter().map(Self::user_from_row).collect()
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
        let sql = Query::select()
            .columns([Users::Id, Users::PasswordHash])
            .from(Users::Table)
            .and_where(Expr::col(Users::Username).eq(username.trim()))
            .to_string(SqliteQueryBuilder);
        let row = sqlx::query(AssertSqlSafe(sql.as_str()))
            .fetch_optional(&self.pool)
            .await?;
        let (id, hash): (Option<String>, Option<String>) = match row {
            Some(r) => (Some(r.try_get("id")?), r.try_get("password_hash")?),
            None => (None, None),
        };
        let password = password.to_owned();
        let hash = hash.unwrap_or_else(|| DUMMY_HASH.to_owned());
        let ok = tokio::task::spawn_blocking(move || verify_password_hash(&password, &hash))
            .await
            .map_err(|e| Error::Config(format!("password verification task failed: {e}")))?;
        match (ok, id) {
            (true, Some(id)) => self.get_user(&id).await,
            _ => Ok(None),
        }
    }

    // --- members ------------------------------------------------------------

    /// Add or change a membership.
    ///
    /// # Errors
    ///
    /// Returns an error if the workspace or user does not exist or the
    /// write fails.
    pub async fn set_member(&self, workspace_id: &str, user_id: &str, role: Role) -> Result<()> {
        let existing = self.member_role(workspace_id, user_id).await?;
        let sql = if existing.is_some() {
            Query::update()
                .table(Members::Table)
                .value(Members::Role, role.as_str())
                .and_where(Expr::col(Members::WorkspaceId).eq(workspace_id))
                .and_where(Expr::col(Members::UserId).eq(user_id))
                .to_string(SqliteQueryBuilder)
        } else {
            Query::insert()
                .into_table(Members::Table)
                .columns([Members::WorkspaceId, Members::UserId, Members::Role])
                .values([workspace_id.into(), user_id.into(), role.as_str().into()])?
                .to_string(SqliteQueryBuilder)
        };
        sqlx::query(AssertSqlSafe(sql.as_str()))
            .execute(&self.pool)
            .await
            .map_err(|e| match &e {
                sqlx::Error::Database(db) if db.is_foreign_key_violation() => Error::Config(
                    String::from("membership needs an existing workspace and user"),
                ),
                _ => e.into(),
            })?;
        Ok(())
    }

    /// Remove a membership. Returns whether one existed.
    ///
    /// # Errors
    ///
    /// Returns an error if the delete fails.
    pub async fn remove_member(&self, workspace_id: &str, user_id: &str) -> Result<bool> {
        let sql = Query::delete()
            .from_table(Members::Table)
            .and_where(Expr::col(Members::WorkspaceId).eq(workspace_id))
            .and_where(Expr::col(Members::UserId).eq(user_id))
            .to_string(SqliteQueryBuilder);
        let done = sqlx::query(AssertSqlSafe(sql.as_str()))
            .execute(&self.pool)
            .await?;
        Ok(done.rows_affected() > 0)
    }

    /// The user's role in a workspace, if a member.
    ///
    /// # Errors
    ///
    /// Returns an error if the query fails.
    pub async fn member_role(&self, workspace_id: &str, user_id: &str) -> Result<Option<Role>> {
        let sql = Query::select()
            .column(Members::Role)
            .from(Members::Table)
            .and_where(Expr::col(Members::WorkspaceId).eq(workspace_id))
            .and_where(Expr::col(Members::UserId).eq(user_id))
            .to_string(SqliteQueryBuilder);
        let row = sqlx::query(AssertSqlSafe(sql.as_str()))
            .fetch_optional(&self.pool)
            .await?;
        row.map(|r| {
            let role: String = r.try_get("role")?;
            Role::parse(&role)
        })
        .transpose()
    }

    /// Members of a workspace with their usernames.
    ///
    /// # Errors
    ///
    /// Returns an error if the query fails.
    pub async fn list_members(&self, workspace_id: &str) -> Result<Vec<MemberRow>> {
        let sql = Query::select()
            .columns([
                (Members::Table, Members::WorkspaceId),
                (Members::Table, Members::UserId),
                (Members::Table, Members::Role),
                (Members::Table, Members::CreatedAt),
            ])
            .column((Users::Table, Users::Username))
            .from(Members::Table)
            .inner_join(
                Users::Table,
                Expr::col((Users::Table, Users::Id)).equals((Members::Table, Members::UserId)),
            )
            .and_where(Expr::col((Members::Table, Members::WorkspaceId)).eq(workspace_id))
            .order_by((Users::Table, Users::Username), Order::Asc)
            .to_string(SqliteQueryBuilder);
        let rows = sqlx::query(AssertSqlSafe(sql.as_str()))
            .fetch_all(&self.pool)
            .await?;
        rows.iter()
            .map(|r| {
                let role: String = r.try_get("role")?;
                Ok(MemberRow {
                    workspace_id: r.try_get("workspace_id")?,
                    user_id: r.try_get("user_id")?,
                    username: r.try_get("username")?,
                    role: Role::parse(&role)?,
                    created_at: r.try_get("created_at")?,
                })
            })
            .collect()
    }

    // --- API tokens ---------------------------------------------------------

    fn token_from_row(r: &sqlx::sqlite::SqliteRow) -> Result<TokenRow> {
        let scopes: String = r.try_get("scopes")?;
        Ok(TokenRow {
            token_hash: r.try_get("token_hash")?,
            workspace_id: r.try_get("workspace_id")?,
            user_id: r.try_get("user_id")?,
            name: r.try_get("name")?,
            scopes: serde_json::from_str(&scopes)?,
            created_at: r.try_get("created_at")?,
            expires_at: r.try_get("expires_at")?,
            last_used_at: r.try_get("last_used_at")?,
        })
    }

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

    /// Mint an API token scoped to one workspace. Returns the token text,
    /// which is never stored, and the row.
    ///
    /// # Errors
    ///
    /// Returns an error if the workspace or user does not exist or the
    /// insert fails.
    pub async fn create_token(
        &self,
        workspace_id: &str,
        user_id: &str,
        name: &str,
        scopes: &[Scope],
        expires_at: Option<&str>,
    ) -> Result<(String, TokenRow)> {
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
        let sql = Query::insert()
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
                expires_at.into(),
            ])?
            .to_string(SqliteQueryBuilder);
        sqlx::query(AssertSqlSafe(sql.as_str()))
            .execute(&self.pool)
            .await
            .map_err(|e| match &e {
                sqlx::Error::Database(db) if db.is_foreign_key_violation() => {
                    Error::Config(String::from("a token needs an existing workspace and user"))
                }
                _ => e.into(),
            })?;
        let row = self
            .find_token(&hash)
            .await?
            .ok_or_else(|| Error::Config(String::from("token vanished after insert")))?;
        Ok((token, row))
    }

    /// The token row for a presented token, by its hash.
    ///
    /// # Errors
    ///
    /// Returns an error if the query fails.
    pub async fn find_token(&self, token_hash: &str) -> Result<Option<TokenRow>> {
        let sql = Self::token_select()
            .and_where(Expr::col(ApiTokens::TokenHash).eq(token_hash))
            .to_string(SqliteQueryBuilder);
        let row = sqlx::query(AssertSqlSafe(sql.as_str()))
            .fetch_optional(&self.pool)
            .await?;
        row.as_ref().map(Self::token_from_row).transpose()
    }

    /// Record that a token was just used.
    ///
    /// # Errors
    ///
    /// Returns an error if the update fails.
    pub async fn touch_token(&self, token_hash: &str) -> Result<()> {
        let sql = Query::update()
            .table(ApiTokens::Table)
            .value(ApiTokens::LastUsedAt, Expr::cust("CURRENT_TIMESTAMP"))
            .and_where(Expr::col(ApiTokens::TokenHash).eq(token_hash))
            .to_string(SqliteQueryBuilder);
        sqlx::query(AssertSqlSafe(sql.as_str()))
            .execute(&self.pool)
            .await?;
        Ok(())
    }

    /// Tokens for a workspace, newest first.
    ///
    /// # Errors
    ///
    /// Returns an error if the query fails.
    pub async fn list_tokens(&self, workspace_id: &str) -> Result<Vec<TokenRow>> {
        let sql = Self::token_select()
            .and_where(Expr::col(ApiTokens::WorkspaceId).eq(workspace_id))
            .order_by(ApiTokens::CreatedAt, Order::Desc)
            .to_string(SqliteQueryBuilder);
        let rows = sqlx::query(AssertSqlSafe(sql.as_str()))
            .fetch_all(&self.pool)
            .await?;
        rows.iter().map(Self::token_from_row).collect()
    }

    /// Revoke a token. Returns whether it existed.
    ///
    /// # Errors
    ///
    /// Returns an error if the delete fails.
    pub async fn delete_token(&self, token_hash: &str) -> Result<bool> {
        let sql = Query::delete()
            .from_table(ApiTokens::Table)
            .and_where(Expr::col(ApiTokens::TokenHash).eq(token_hash))
            .to_string(SqliteQueryBuilder);
        let done = sqlx::query(AssertSqlSafe(sql.as_str()))
            .execute(&self.pool)
            .await?;
        Ok(done.rows_affected() > 0)
    }

    // --- access audit (append-only) ------------------------------------------

    /// Append one row to `audit_log`. There is no update or delete path.
    ///
    /// # Errors
    ///
    /// Returns an error if the insert fails.
    pub async fn record_audit(&self, entry: &AuditEntry) -> Result<()> {
        let sql = Query::insert()
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
                entry.id.as_str().into(),
                entry.user_id.as_deref().into(),
                entry.token_hash.as_deref().into(),
                entry.workspace_id.as_deref().into(),
                entry.action.as_str().into(),
                entry.resource_type.as_deref().into(),
                entry.resource_id.as_deref().into(),
                entry.outcome.as_str().into(),
                entry.channel.as_str().into(),
                entry.client_addr.as_deref().into(),
                entry.request_id.as_deref().into(),
            ])?
            .to_string(SqliteQueryBuilder);
        sqlx::query(AssertSqlSafe(sql.as_str()))
            .execute(&self.pool)
            .await?;
        Ok(())
    }

    /// Audit rows matching the filter, newest first.
    ///
    /// # Errors
    ///
    /// Returns an error if the query fails.
    pub async fn query_audit(&self, filter: &AuditFilter) -> Result<Vec<AuditRow>> {
        let sql = audit_query_sql(filter);
        let rows = sqlx::query(AssertSqlSafe(sql.as_str()))
            .fetch_all(&self.pool)
            .await?;
        rows.iter()
            .map(|r| {
                Ok(AuditRow {
                    id: r.try_get("id")?,
                    timestamp: r.try_get("timestamp")?,
                    user_id: r.try_get("user_id")?,
                    token_hash: r.try_get("token_hash")?,
                    workspace_id: r.try_get("workspace_id")?,
                    action: r.try_get("action")?,
                    resource_type: r.try_get("resource_type")?,
                    resource_id: r.try_get("resource_id")?,
                    outcome: r.try_get("outcome")?,
                    channel: r.try_get("channel")?,
                    client_addr: r.try_get("client_addr")?,
                    request_id: r.try_get("request_id")?,
                })
            })
            .collect()
    }
}

/// The filtered audit select, built and rendered in one scope so no
/// builder lives across an await.
fn audit_query_sql(filter: &AuditFilter) -> String {
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
        .limit(u64::from(Ord::max(filter.limit, 1)));
    if let Some(v) = &filter.user_id {
        select.and_where(Expr::col(AuditLog::UserId).eq(v.as_str()));
    }
    if let Some(v) = &filter.workspace_id {
        select.and_where(Expr::col(AuditLog::WorkspaceId).eq(v.as_str()));
    }
    if let Some(v) = &filter.action {
        select.and_where(Expr::col(AuditLog::Action).eq(v.as_str()));
    }
    if let Some(v) = &filter.outcome {
        select.and_where(Expr::col(AuditLog::Outcome).eq(v.as_str()));
    }
    if let Some(v) = &filter.since {
        select.and_where(Expr::col(AuditLog::Timestamp).gte(v.as_str()));
    }
    if let Some(v) = &filter.until {
        select.and_where(Expr::col(AuditLog::Timestamp).lt(v.as_str()));
    }
    select.to_string(SqliteQueryBuilder)
}

/// A valid argon2id hash of a random string, verified against when the
/// username is unknown so login timing does not reveal which usernames exist.
const DUMMY_HASH: &str = "$argon2id$v=19$m=19456,t=2,p=1$c2FsdHNhbHRzYWx0c2FsdA$Pm2ZmwZKKGUUtY5t1M2p5iP5B0KJ5FhBwa0zDf5Tr3Y";

#[cfg(test)]
mod tests {
    use super::*;

    async fn open() -> (tempfile::TempDir, ControlPlane) {
        let dir = tempfile::tempdir().unwrap_or_else(|e| fail(&e.to_string()));
        let mut config = Config::default();
        config.general.data_dir = dir.path().to_path_buf();
        let cp = ControlPlane::open(&config)
            .await
            .unwrap_or_else(|e| fail(&e.to_string()));
        (dir, cp)
    }

    #[expect(clippy::panic, reason = "test failure path")]
    fn fail(msg: &str) -> ! {
        panic!("{msg}")
    }

    /// A database from before sqlx migrations: the three sea-query versions
    /// applied, progress recorded in `schema_version`, and rows in the tables
    /// that replaying those versions would destroy.
    async fn legacy_control_db(path: &std::path::Path) {
        let url = format!("sqlite:{}?mode=rwc", path.display());
        let options = SqliteConnectOptions::from_str(&url).unwrap_or_else(|e| fail(&e.to_string()));
        let pool = SqlitePoolOptions::new()
            .max_connections(1)
            .connect_with(options)
            .await
            .unwrap_or_else(|e| fail(&e.to_string()));

        let mut sql = String::from(
            "CREATE TABLE IF NOT EXISTS schema_version (\
                 version INTEGER PRIMARY KEY, \
                 applied_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP);",
        );
        for file in [
            "0001_access_control",
            "0002_audit_log_access_record",
            "0003_users_and_token_scopes",
        ] {
            sql.push_str(
                &std::fs::read_to_string(
                    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                        .join("migrations")
                        .join(format!("{file}.sql")),
                )
                .unwrap_or_else(|e| fail(&e.to_string())),
            );
        }
        sql.push_str("INSERT INTO schema_version (version) VALUES (1), (2), (3);");
        sql.push_str(
            "INSERT INTO audit_log (id, action, outcome, channel) \
             VALUES ('01890000-0000-7000-8000-000000000001', 'login', 'allowed', 'cli');",
        );
        sqlx::raw_sql(AssertSqlSafe(sql))
            .execute(&pool)
            .await
            .unwrap_or_else(|e| fail(&e.to_string()));
        pool.close().await;
    }

    #[tokio::test]
    async fn legacy_schema_version_is_adopted_without_replaying_migrations() {
        let dir = tempfile::tempdir().unwrap_or_else(|e| fail(&e.to_string()));
        let mut config = Config::default();
        config.general.data_dir = dir.path().to_path_buf();
        config
            .ensure_dirs()
            .unwrap_or_else(|e| fail(&e.to_string()));
        legacy_control_db(&config.control_db_path()).await;

        let cp = ControlPlane::open(&config)
            .await
            .unwrap_or_else(|e| fail(&e.to_string()));

        assert!(cp.schema_version().await.is_ok_and(|v| v == 3));
        // Replaying v2 would have dropped and recreated audit_log.
        let rows: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM audit_log")
            .fetch_one(&cp.pool)
            .await
            .unwrap_or_else(|e| fail(&e.to_string()));
        assert_eq!(rows, 1, "the access record must survive adoption");
    }

    #[tokio::test]
    async fn a_shipped_migration_may_not_change_under_an_existing_database() {
        let (dir, cp) = open().await;
        drop(cp);

        // Simulate an edit to a migration that has already been applied.
        sqlx::raw_sql("UPDATE _sqlx_migrations SET checksum = x'00' WHERE version = 1")
            .execute(
                &SqlitePoolOptions::new()
                    .max_connections(1)
                    .connect_with(
                        SqliteConnectOptions::from_str(&format!(
                            "sqlite:{}",
                            dir.path().join("control.db").display()
                        ))
                        .unwrap_or_else(|e| fail(&e.to_string())),
                    )
                    .await
                    .unwrap_or_else(|e| fail(&e.to_string())),
            )
            .await
            .unwrap_or_else(|e| fail(&e.to_string()));

        let mut config = Config::default();
        config.general.data_dir = dir.path().to_path_buf();
        assert!(
            ControlPlane::open(&config).await.is_err(),
            "a changed checksum must refuse the database, not migrate it"
        );
    }

    #[tokio::test]
    async fn migrations_reach_v3_and_rerun_idempotently() {
        let (dir, cp) = open().await;
        assert!(cp.schema_version().await.is_ok_and(|v| v == 3));
        drop(cp);
        let mut config = Config::default();
        config.general.data_dir = dir.path().to_path_buf();
        let again = ControlPlane::open(&config).await;
        assert!(again.is_ok());
    }

    #[tokio::test]
    async fn workspace_updates_keep_unset_fields() {
        let (_dir, cp) = open().await;
        let ws = cp
            .create_workspace("w")
            .await
            .unwrap_or_else(|e| fail(&e.to_string()));
        let changed = cp
            .update_workspace(
                &ws.id,
                &WorkspaceChanges {
                    classification: Some(String::from("secret")),
                    allowed_providers: ProviderAllowList::Only(vec![String::from("ollama")]),
                },
            )
            .await;
        assert!(changed.is_ok_and(|w| {
            w.classification == "secret" && w.allowed_providers.as_deref() == Some("[\"ollama\"]")
        }));
        let kept = cp
            .update_workspace(&ws.id, &WorkspaceChanges::default())
            .await;
        assert!(kept.is_ok_and(|w| w.classification == "secret" && w.allowed_providers.is_some()));
        let cleared = cp
            .update_workspace(
                &ws.id,
                &WorkspaceChanges {
                    classification: None,
                    allowed_providers: ProviderAllowList::All,
                },
            )
            .await;
        assert!(cleared.is_ok_and(|w| w.allowed_providers.is_none()));
        assert!(
            cp.update_workspace("missing", &WorkspaceChanges::default())
                .await
                .is_err()
        );
        assert!(cp.get_workspace(&ws.id).await.is_ok_and(|w| w.is_some()));
        assert!(cp.create_workspace("w").await.is_err());
    }

    #[tokio::test]
    async fn users_hash_verify_and_reject_duplicates() {
        let (_dir, cp) = open().await;
        let alice = cp.create_user("alice", "hunter42", true).await;
        assert!(
            alice
                .as_ref()
                .is_ok_and(|u| u.is_admin && u.username == "alice")
        );
        assert!(
            cp.verify_password("alice", "hunter42")
                .await
                .is_ok_and(|u| u.is_some())
        );
        assert!(
            cp.verify_password("alice", "wrong")
                .await
                .is_ok_and(|u| u.is_none())
        );
        assert!(
            cp.verify_password("nobody", "hunter42")
                .await
                .is_ok_and(|u| u.is_none())
        );
        let dup = cp.create_user("alice", "x", false).await.err();
        assert!(dup.is_some_and(|e| e.to_string().contains("already exists")));
        assert!(cp.create_user("", "x", false).await.is_err());
        assert!(cp.create_user("bob", "", false).await.is_err());
        assert!(cp.list_users().await.is_ok_and(|u| u.len() == 1));
        assert!(
            cp.find_user_by_username(" alice ")
                .await
                .is_ok_and(|u| u.is_some())
        );
    }

    #[tokio::test]
    async fn members_need_a_user_and_a_workspace() {
        let (_dir, cp) = open().await;
        let ws = cp
            .create_workspace("w")
            .await
            .unwrap_or_else(|e| fail(&e.to_string()));
        let bob = cp
            .create_user("bob", "pw", false)
            .await
            .unwrap_or_else(|e| fail(&e.to_string()));
        assert!(cp.set_member(&ws.id, "ghost", Role::Member).await.is_err());
        assert!(cp.set_member(&ws.id, &bob.id, Role::Viewer).await.is_ok());
        assert!(
            cp.member_role(&ws.id, &bob.id)
                .await
                .is_ok_and(|r| r == Some(Role::Viewer))
        );
        assert!(cp.set_member(&ws.id, &bob.id, Role::Owner).await.is_ok());
        assert!(
            cp.member_role(&ws.id, &bob.id)
                .await
                .is_ok_and(|r| r == Some(Role::Owner))
        );
        let listed = cp.list_members(&ws.id).await;
        assert!(
            listed.is_ok_and(|m| m.len() == 1 && m.first().is_some_and(|m| m.username == "bob"))
        );
        let mine = cp.workspaces_for_user(&bob.id).await;
        assert!(mine.is_ok_and(|w| {
            w.len() == 1
                && w.first()
                    .is_some_and(|(w, r)| w.name == "w" && *r == Role::Owner)
        }));
        assert!(
            cp.remove_member(&ws.id, &bob.id)
                .await
                .is_ok_and(|removed| removed)
        );
        assert!(
            cp.remove_member(&ws.id, &bob.id)
                .await
                .is_ok_and(|removed| !removed)
        );
        assert!(Role::Viewer < Role::Member && Role::Member < Role::Owner);
        assert!(Role::parse("boss").is_err());
    }

    #[tokio::test]
    async fn tokens_are_stored_hashed_with_scopes_and_expiry() {
        let (_dir, cp) = open().await;
        let ws = cp
            .create_workspace("w")
            .await
            .unwrap_or_else(|e| fail(&e.to_string()));
        let bob = cp
            .create_user("bob", "pw", false)
            .await
            .unwrap_or_else(|e| fail(&e.to_string()));
        let minted = cp
            .create_token(
                &ws.id,
                &bob.id,
                "ci",
                &[Scope::Read, Scope::Write],
                Some("2000-01-01 00:00:00"),
            )
            .await;
        let Ok((token, row)) = minted else {
            fail("token creation failed");
        };
        assert!(token.starts_with("qk_"));
        assert_eq!(row.token_hash, sha256_hex(token.as_bytes()));
        assert!(row.has_scope(Scope::Write) && !row.has_scope(Scope::Admin));
        assert!(row.is_expired("2001-01-01 00:00:00"));
        assert!(!row.is_expired("1999-01-01 00:00:00"));
        assert!(
            cp.find_token(&row.token_hash)
                .await
                .is_ok_and(|t| t.is_some())
        );
        assert!(cp.touch_token(&row.token_hash).await.is_ok());
        assert!(
            cp.find_token(&row.token_hash)
                .await
                .is_ok_and(|t| t.is_some_and(|t| t.last_used_at.is_some()))
        );
        assert!(cp.list_tokens(&ws.id).await.is_ok_and(|t| t.len() == 1));
        assert!(
            cp.create_token("nope", &bob.id, "x", &[], None)
                .await
                .is_err()
        );
        assert!(cp.delete_token(&row.token_hash).await.is_ok_and(|d| d));
        assert!(
            cp.find_token(&row.token_hash)
                .await
                .is_ok_and(|t| t.is_none())
        );
        assert!(Scope::parse("root").is_err());
    }

    #[tokio::test]
    async fn audit_rows_append_and_filter() {
        let (_dir, cp) = open().await;
        let mut allowed = AuditEntry::new("open", Outcome::Allowed, Channel::Api);
        allowed.user_id = Some(String::from("u1"));
        allowed.workspace_id = Some(String::from("w1"));
        let mut denied = AuditEntry::new("open", Outcome::Denied, Channel::Web);
        denied.user_id = Some(String::from("u2"));
        denied.workspace_id = Some(String::from("w1"));
        let login = AuditEntry::new("login", Outcome::Error, Channel::Web);
        for e in [&allowed, &denied, &login] {
            assert!(cp.record_audit(e).await.is_ok());
        }
        let all = cp
            .query_audit(&AuditFilter {
                limit: 10,
                ..AuditFilter::default()
            })
            .await;
        assert!(all.is_ok_and(|r| r.len() == 3 && r.first().is_some_and(|r| r.action == "login")));
        let denied_only = cp
            .query_audit(&AuditFilter {
                outcome: Some(String::from("denied")),
                limit: 10,
                ..AuditFilter::default()
            })
            .await;
        assert!(denied_only.is_ok_and(|r| {
            r.len() == 1
                && r.first()
                    .is_some_and(|r| r.user_id.as_deref() == Some("u2"))
        }));
        let for_ws = cp
            .query_audit(&AuditFilter {
                workspace_id: Some(String::from("w1")),
                user_id: Some(String::from("u1")),
                limit: 10,
                ..AuditFilter::default()
            })
            .await;
        assert!(
            for_ws.is_ok_and(|r| r.len() == 1 && r.first().is_some_and(|r| r.id == allowed.id))
        );
        let none = cp
            .query_audit(&AuditFilter {
                until: Some(String::from("1990-01-01")),
                limit: 10,
                ..AuditFilter::default()
            })
            .await;
        assert!(none.is_ok_and(|r| r.is_empty()));
    }

    #[test]
    fn sha256_hex_is_the_known_digest_of_abc() {
        assert_eq!(
            sha256_hex(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    #[test]
    fn dummy_hash_parses_as_argon2id() {
        assert!(PasswordHash::new(DUMMY_HASH).is_ok());
        assert!(!verify_password_hash("anything", DUMMY_HASH));
    }
}
