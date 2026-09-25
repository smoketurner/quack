use sea_query::{Iden, QueryStatementWriter, SqliteQueryBuilder, Value};
use sqlx::sqlite::{Sqlite, SqliteArguments, SqliteRow};
use sqlx::{Arguments as _, AssertSqlSafe, FromRow};

/// A sea-query statement ready for sqlx: the SQL with `?` placeholders and
/// its values bound to them in order, never written into the text.
pub struct Bound {
    sql: String,
    arguments: SqliteArguments,
}

impl Bound {
    /// Build `statement` for `SQLite` and bind its values.
    ///
    /// # Errors
    ///
    /// Returns an error for a value of a type `control.db` never stores (a
    /// float, a date), which binding does not cover.
    pub fn new(statement: &impl QueryStatementWriter) -> sqlx::Result<Self> {
        let (sql, values) = statement.build(SqliteQueryBuilder);
        let mut arguments = SqliteArguments::default();
        for value in values {
            let bound = match value {
                Value::Bool(v) => arguments.add(v),
                Value::Int(v) => arguments.add(v),
                Value::BigInt(v) => arguments.add(v),
                Value::Unsigned(v) => arguments.add(v),
                // SQLite has no unsigned 64-bit integer; LIMIT arrives as one.
                Value::BigUnsigned(v) => arguments.add(
                    v.map(i64::try_from)
                        .transpose()
                        .map_err(|e| sqlx::Error::Encode(Box::new(e)))?,
                ),
                Value::String(v) => arguments.add(v),
                Value::Bytes(v) => arguments.add(v),
                other => {
                    return Err(sqlx::Error::Encode(
                        format!("control.db does not bind {other:?}").into(),
                    ));
                }
            };
            bound.map_err(sqlx::Error::Encode)?;
        }
        Ok(Self { sql, arguments })
    }

    /// The statement as a query to execute or fetch rows from.
    pub fn query(self) -> sqlx::query::Query<'static, Sqlite, SqliteArguments> {
        // Only placeholders are in the text: every value is bound.
        sqlx::query_with(AssertSqlSafe(self.sql), self.arguments)
    }

    /// The statement as a query that reads each row as a `T`.
    pub fn query_as<T>(self) -> sqlx::query::QueryAs<'static, Sqlite, T, SqliteArguments>
    where
        T: for<'r> FromRow<'r, SqliteRow>,
    {
        sqlx::query_as_with(AssertSqlSafe(self.sql), self.arguments)
    }
}

// --- Control plane tables (SQLite) ---

#[derive(Iden)]
pub enum Users {
    Table,
    Id,
    Username,
    PasswordHash,
    OidcSubject,
    IsAdmin,
    CreatedAt,
}

/// The columns every sealed-token table shares.
#[derive(Iden)]
pub enum SealedColumns {
    KeyId,
    Enc,
    Ciphertext,
    UpdatedAt,
}

#[derive(Iden)]
pub enum UserTokens {
    Table,
    UserId,
}

#[derive(Iden)]
pub enum ProviderTokens {
    Table,
    Provider,
}

#[derive(Iden)]
pub enum Workspaces {
    Table,
    Id,
    Name,
    Classification,
    AllowedProviders,
    CreatedAt,
    UpdatedAt,
}

#[derive(Iden)]
pub enum Members {
    Table,
    WorkspaceId,
    UserId,
    Role,
    CreatedAt,
}

#[derive(Iden)]
pub enum ApiTokens {
    Table,
    TokenHash,
    WorkspaceId,
    UserId,
    Name,
    Scopes,
    CreatedAt,
    ExpiresAt,
    LastUsedAt,
}

/// Access audit: who accessed what, when, how, and whether it was allowed.
/// Never holds content; detail lives inside the workspace (design doc 5.5).
#[derive(Iden)]
pub enum AuditLog {
    Table,
    Id,
    Timestamp,
    UserId,
    TokenHash,
    WorkspaceId,
    Action,
    ResourceType,
    ResourceId,
    Outcome,
    Channel,
    ClientAddr,
    RequestId,
}
