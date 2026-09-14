use sea_query::{Expr, ExprTrait, Query, SqliteQueryBuilder};
use sqlx::sqlite::{SqliteConnectOptions, SqlitePoolOptions};
use sqlx::{AssertSqlSafe, Row, SqlitePool};
use std::str::FromStr;

use super::queries::Workspaces;
use crate::config::Config;

/// A workspace row from the control plane.
#[derive(Debug, Clone)]
pub struct WorkspaceRow {
    pub id: String,
    pub name: String,
    pub classification: String,
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
    pub async fn open(config: &Config) -> crate::error::Result<Self> {
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

    async fn run_migrations(&self) -> crate::error::Result<()> {
        sqlx::query("PRAGMA journal_mode=WAL")
            .execute(&self.pool)
            .await?;
        sqlx::query("PRAGMA foreign_keys=ON")
            .execute(&self.pool)
            .await?;

        // schema_version itself is created by v1, so bootstrap it first.
        let bootstrap = super::migrations::v1_statements();
        if let Some(create_schema_version) = bootstrap.first() {
            sqlx::query(AssertSqlSafe(create_schema_version.as_str()))
                .execute(&self.pool)
                .await?;
        }

        let current = self.schema_version().await?;

        for (version, statements) in super::migrations::versions() {
            if version <= current {
                continue;
            }
            for sql in &statements {
                sqlx::query(AssertSqlSafe(sql.as_str()))
                    .execute(&self.pool)
                    .await?;
            }
            sqlx::query("INSERT INTO schema_version (version) VALUES (?)")
                .bind(version)
                .execute(&self.pool)
                .await?;
            tracing::info!(version, "applied control.db migration");
        }

        Ok(())
    }

    /// Highest applied schema version.
    ///
    /// # Errors
    ///
    /// Returns an error if the query fails.
    pub async fn schema_version(&self) -> crate::error::Result<i64> {
        Ok(
            sqlx::query_scalar("SELECT COALESCE(MAX(version), 0) FROM schema_version")
                .fetch_one(&self.pool)
                .await?,
        )
    }

    /// Look up a workspace by name.
    ///
    /// # Errors
    ///
    /// Returns an error if the query fails.
    pub async fn find_workspace_by_name(
        &self,
        name: &str,
    ) -> crate::error::Result<Option<WorkspaceRow>> {
        let sql = Query::select()
            .column(Workspaces::Id)
            .column(Workspaces::Name)
            .column(Workspaces::Classification)
            .from(Workspaces::Table)
            .and_where(Expr::col(Workspaces::Name).eq(name))
            .to_string(SqliteQueryBuilder);

        let row = sqlx::query(AssertSqlSafe(sql.as_str()))
            .fetch_optional(&self.pool)
            .await?;

        row.map(|r| {
            Ok(WorkspaceRow {
                id: r.try_get("id")?,
                name: r.try_get("name")?,
                classification: r.try_get("classification")?,
            })
        })
        .transpose()
    }

    /// Create a new workspace and return its row.
    ///
    /// # Errors
    ///
    /// Returns an error if the insert fails.
    pub async fn create_workspace(&self, name: &str) -> crate::error::Result<WorkspaceRow> {
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
        })
    }

    /// Find a workspace by name, creating it if it does not exist.
    ///
    /// # Errors
    ///
    /// Returns an error if the lookup or creation fails.
    pub async fn find_or_create_workspace(&self, name: &str) -> crate::error::Result<WorkspaceRow> {
        if let Some(ws) = self.find_workspace_by_name(name).await? {
            return Ok(ws);
        }
        self.create_workspace(name).await
    }

    /// List all workspaces.
    ///
    /// # Errors
    ///
    /// Returns an error if the query fails.
    pub async fn list_workspaces(&self) -> crate::error::Result<Vec<WorkspaceRow>> {
        let sql = Query::select()
            .column(Workspaces::Id)
            .column(Workspaces::Name)
            .column(Workspaces::Classification)
            .from(Workspaces::Table)
            .to_string(SqliteQueryBuilder);

        let rows = sqlx::query(AssertSqlSafe(sql.as_str()))
            .fetch_all(&self.pool)
            .await?;

        rows.iter()
            .map(|r| {
                Ok(WorkspaceRow {
                    id: r.try_get("id")?,
                    name: r.try_get("name")?,
                    classification: r.try_get("classification")?,
                })
            })
            .collect()
    }
}
