use sea_query::{ColumnDef, Expr, ForeignKey, ForeignKeyAction, SqliteQueryBuilder, Table};

use super::queries::{ApiTokens, AuditLog, Members, Messages, SchemaVersion, Threads, Workspaces};

/// Generate the DDL statements for control plane schema v1.
///
/// Each statement creates a table if it does not already exist.
#[must_use]
pub fn v1_statements() -> Vec<String> {
    vec![
        create_schema_version(),
        create_workspaces(),
        create_members(),
        create_api_tokens(),
        create_threads(),
        create_messages(),
        create_audit_log(),
    ]
}

fn create_schema_version() -> String {
    Table::create()
        .table(SchemaVersion::Table)
        .if_not_exists()
        .col(
            ColumnDef::new(SchemaVersion::Version)
                .integer()
                .not_null()
                .primary_key(),
        )
        .col(
            ColumnDef::new(SchemaVersion::AppliedAt)
                .text()
                .not_null()
                .default(Expr::cust("CURRENT_TIMESTAMP")),
        )
        .to_string(SqliteQueryBuilder)
}

fn create_workspaces() -> String {
    Table::create()
        .table(Workspaces::Table)
        .if_not_exists()
        .col(
            ColumnDef::new(Workspaces::Id)
                .text()
                .not_null()
                .primary_key(),
        )
        .col(ColumnDef::new(Workspaces::Name).text().not_null())
        .col(
            ColumnDef::new(Workspaces::Classification)
                .text()
                .not_null()
                .default("internal"),
        )
        .col(ColumnDef::new(Workspaces::AllowedProviders).text())
        .col(
            ColumnDef::new(Workspaces::CreatedAt)
                .text()
                .not_null()
                .default(Expr::cust("CURRENT_TIMESTAMP")),
        )
        .col(
            ColumnDef::new(Workspaces::UpdatedAt)
                .text()
                .not_null()
                .default(Expr::cust("CURRENT_TIMESTAMP")),
        )
        .to_string(SqliteQueryBuilder)
}

fn create_members() -> String {
    Table::create()
        .table(Members::Table)
        .if_not_exists()
        .col(ColumnDef::new(Members::WorkspaceId).text().not_null())
        .col(ColumnDef::new(Members::UserId).text().not_null())
        .col(
            ColumnDef::new(Members::Role)
                .text()
                .not_null()
                .default("member"),
        )
        .col(
            ColumnDef::new(Members::CreatedAt)
                .text()
                .not_null()
                .default(Expr::cust("CURRENT_TIMESTAMP")),
        )
        .foreign_key(
            ForeignKey::create()
                .from(Members::Table, Members::WorkspaceId)
                .to(Workspaces::Table, Workspaces::Id)
                .on_delete(ForeignKeyAction::Cascade),
        )
        .to_string(SqliteQueryBuilder)
}

fn create_api_tokens() -> String {
    Table::create()
        .table(ApiTokens::Table)
        .if_not_exists()
        .col(
            ColumnDef::new(ApiTokens::TokenHash)
                .text()
                .not_null()
                .primary_key(),
        )
        .col(ColumnDef::new(ApiTokens::WorkspaceId).text().not_null())
        .col(ColumnDef::new(ApiTokens::UserId).text().not_null())
        .col(ColumnDef::new(ApiTokens::Name).text().not_null())
        .col(
            ColumnDef::new(ApiTokens::CreatedAt)
                .text()
                .not_null()
                .default(Expr::cust("CURRENT_TIMESTAMP")),
        )
        .col(ColumnDef::new(ApiTokens::ExpiresAt).text())
        .foreign_key(
            ForeignKey::create()
                .from(ApiTokens::Table, ApiTokens::WorkspaceId)
                .to(Workspaces::Table, Workspaces::Id)
                .on_delete(ForeignKeyAction::Cascade),
        )
        .to_string(SqliteQueryBuilder)
}

fn create_threads() -> String {
    Table::create()
        .table(Threads::Table)
        .if_not_exists()
        .col(ColumnDef::new(Threads::Id).text().not_null().primary_key())
        .col(ColumnDef::new(Threads::WorkspaceId).text().not_null())
        .col(ColumnDef::new(Threads::Title).text())
        .col(ColumnDef::new(Threads::CreatedBy).text().not_null())
        .col(
            ColumnDef::new(Threads::CreatedAt)
                .text()
                .not_null()
                .default(Expr::cust("CURRENT_TIMESTAMP")),
        )
        .foreign_key(
            ForeignKey::create()
                .from(Threads::Table, Threads::WorkspaceId)
                .to(Workspaces::Table, Workspaces::Id)
                .on_delete(ForeignKeyAction::Cascade),
        )
        .to_string(SqliteQueryBuilder)
}

fn create_messages() -> String {
    Table::create()
        .table(Messages::Table)
        .if_not_exists()
        .col(ColumnDef::new(Messages::Id).text().not_null().primary_key())
        .col(ColumnDef::new(Messages::ThreadId).text().not_null())
        .col(ColumnDef::new(Messages::Role).text().not_null())
        .col(ColumnDef::new(Messages::Content).text().not_null())
        .col(ColumnDef::new(Messages::Metadata).text())
        .col(
            ColumnDef::new(Messages::CreatedAt)
                .text()
                .not_null()
                .default(Expr::cust("CURRENT_TIMESTAMP")),
        )
        .foreign_key(
            ForeignKey::create()
                .from(Messages::Table, Messages::ThreadId)
                .to(Threads::Table, Threads::Id)
                .on_delete(ForeignKeyAction::Cascade),
        )
        .to_string(SqliteQueryBuilder)
}

fn create_audit_log() -> String {
    Table::create()
        .table(AuditLog::Table)
        .if_not_exists()
        .col(
            ColumnDef::new(AuditLog::Id)
                .integer()
                .not_null()
                .auto_increment()
                .primary_key(),
        )
        .col(
            ColumnDef::new(AuditLog::Timestamp)
                .text()
                .not_null()
                .default(Expr::cust("CURRENT_TIMESTAMP")),
        )
        .col(ColumnDef::new(AuditLog::WorkspaceId).text().not_null())
        .col(ColumnDef::new(AuditLog::UserId).text())
        .col(ColumnDef::new(AuditLog::Action).text().not_null())
        .col(ColumnDef::new(AuditLog::Detail).text())
        .to_string(SqliteQueryBuilder)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn v1_generates_valid_sql() {
        let stmts = v1_statements();
        assert_eq!(stmts.len(), 7);
        for stmt in &stmts {
            assert!(
                stmt.starts_with("CREATE TABLE IF NOT EXISTS"),
                "unexpected DDL: {stmt}"
            );
        }
    }
}
