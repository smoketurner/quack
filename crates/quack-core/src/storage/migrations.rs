use sea_query::{
    ColumnDef, Expr, ForeignKey, ForeignKeyAction, Iden, Index, SqliteQueryBuilder, Table,
};

use super::queries::{ApiTokens, AuditLog, Members, SchemaVersion, Users, Workspaces};

/// Every schema version in order. `ControlPlane::open` applies the ones the
/// database has not recorded yet.
#[must_use]
pub fn versions() -> Vec<(i64, Vec<String>)> {
    vec![
        (1, v1_statements()),
        (2, v2_statements()),
        (3, v3_statements()),
    ]
}

/// v1: access-control tables.
#[must_use]
pub fn v1_statements() -> Vec<String> {
    vec![
        create_schema_version(),
        create_workspaces(),
        create_members_v1(),
        create_api_tokens(),
    ]
}

/// Tables from the first draft that held workspace content in `control.db`.
/// Sessions and messages belong inside the workspace file (design doc 5.4).
#[derive(Iden)]
enum LegacyThreads {
    #[iden = "threads"]
    Table,
}

#[derive(Iden)]
enum LegacyMessages {
    #[iden = "messages"]
    Table,
}

/// v2: drop the content tables and give the audit log its access-record
/// shape (who, what resource, outcome, channel, when; no detail column).
#[must_use]
pub fn v2_statements() -> Vec<String> {
    vec![
        Table::drop()
            .table(LegacyMessages::Table)
            .if_exists()
            .to_string(SqliteQueryBuilder),
        Table::drop()
            .table(LegacyThreads::Table)
            .if_exists()
            .to_string(SqliteQueryBuilder),
        Table::drop()
            .table(AuditLog::Table)
            .if_exists()
            .to_string(SqliteQueryBuilder),
        create_audit_log(),
        Index::create()
            .if_not_exists()
            .name("audit_log_user_ts")
            .table(AuditLog::Table)
            .col(AuditLog::UserId)
            .col(AuditLog::Timestamp)
            .to_string(SqliteQueryBuilder),
        Index::create()
            .if_not_exists()
            .name("audit_log_workspace_ts")
            .table(AuditLog::Table)
            .col(AuditLog::WorkspaceId)
            .col(AuditLog::Timestamp)
            .to_string(SqliteQueryBuilder),
    ]
}

/// The v1 members table, kept only so v3 can copy it into the one that
/// references `users`.
#[derive(Iden)]
enum LegacyMembers {
    #[iden = "members_v1"]
    Table,
}

/// v3: server users; token scopes and last use; members reference users.
#[must_use]
pub fn v3_statements() -> Vec<String> {
    vec![
        create_users(),
        Table::alter()
            .table(ApiTokens::Table)
            .add_column(
                ColumnDef::new(ApiTokens::Scopes)
                    .text()
                    .not_null()
                    .default("[\"read\"]"),
            )
            .to_string(SqliteQueryBuilder),
        Table::alter()
            .table(ApiTokens::Table)
            .add_column(ColumnDef::new(ApiTokens::LastUsedAt).text())
            .to_string(SqliteQueryBuilder),
        Table::rename()
            .table(Members::Table, LegacyMembers::Table)
            .to_string(SqliteQueryBuilder),
        create_members(),
        format!(
            "INSERT INTO \"members\" (workspace_id, user_id, role, created_at) \
             SELECT m.workspace_id, m.user_id, m.role, m.created_at FROM \"{}\" m \
             WHERE EXISTS (SELECT 1 FROM \"users\" u WHERE u.id = m.user_id)",
            LegacyMembers::Table.to_string()
        ),
        Table::drop()
            .table(LegacyMembers::Table)
            .to_string(SqliteQueryBuilder),
        Index::create()
            .if_not_exists()
            .unique()
            .name("workspaces_name")
            .table(Workspaces::Table)
            .col(Workspaces::Name)
            .to_string(SqliteQueryBuilder),
    ]
}

fn create_users() -> String {
    Table::create()
        .table(Users::Table)
        .if_not_exists()
        .col(ColumnDef::new(Users::Id).text().not_null().primary_key())
        .col(
            ColumnDef::new(Users::Username)
                .text()
                .not_null()
                .unique_key(),
        )
        .col(ColumnDef::new(Users::PasswordHash).text())
        .col(ColumnDef::new(Users::OidcSubject).text().unique_key())
        .col(
            ColumnDef::new(Users::IsAdmin)
                .integer()
                .not_null()
                .default(0),
        )
        .col(
            ColumnDef::new(Users::CreatedAt)
                .text()
                .not_null()
                .default(Expr::cust("CURRENT_TIMESTAMP")),
        )
        .to_string(SqliteQueryBuilder)
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

/// v1 shape: no reference to `users`, which did not exist yet.
fn create_members_v1() -> String {
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
        .primary_key(
            Index::create()
                .col(Members::WorkspaceId)
                .col(Members::UserId),
        )
        .foreign_key(
            ForeignKey::create()
                .from(Members::Table, Members::WorkspaceId)
                .to(Workspaces::Table, Workspaces::Id)
                .on_delete(ForeignKeyAction::Cascade),
        )
        .foreign_key(
            ForeignKey::create()
                .from(Members::Table, Members::UserId)
                .to(Users::Table, Users::Id)
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

fn create_audit_log() -> String {
    Table::create()
        .table(AuditLog::Table)
        .if_not_exists()
        .col(ColumnDef::new(AuditLog::Id).text().not_null().primary_key())
        .col(
            ColumnDef::new(AuditLog::Timestamp)
                .text()
                .not_null()
                .default(Expr::cust("CURRENT_TIMESTAMP")),
        )
        .col(ColumnDef::new(AuditLog::UserId).text())
        .col(ColumnDef::new(AuditLog::TokenHash).text())
        .col(ColumnDef::new(AuditLog::WorkspaceId).text())
        .col(ColumnDef::new(AuditLog::Action).text().not_null())
        .col(ColumnDef::new(AuditLog::ResourceType).text())
        .col(ColumnDef::new(AuditLog::ResourceId).text())
        .col(ColumnDef::new(AuditLog::Outcome).text().not_null())
        .col(ColumnDef::new(AuditLog::Channel).text().not_null())
        .col(ColumnDef::new(AuditLog::ClientAddr).text())
        .col(ColumnDef::new(AuditLog::RequestId).text())
        .to_string(SqliteQueryBuilder)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn v1_creates_only_access_control_tables() {
        let stmts = v1_statements();
        assert_eq!(stmts.len(), 4);
        for stmt in &stmts {
            assert!(
                stmt.starts_with("CREATE TABLE IF NOT EXISTS"),
                "unexpected DDL: {stmt}"
            );
            assert!(!stmt.contains("threads") && !stmt.contains("messages"));
        }
    }

    #[test]
    fn v2_drops_content_tables_and_reshapes_audit_log() {
        let stmts = v2_statements();
        assert!(
            stmts
                .iter()
                .any(|s| s == "DROP TABLE IF EXISTS \"messages\"")
        );
        assert!(
            stmts
                .iter()
                .any(|s| s == "DROP TABLE IF EXISTS \"threads\"")
        );
        let create = stmts
            .iter()
            .find(|s| s.starts_with("CREATE TABLE IF NOT EXISTS \"audit_log\""))
            .map(String::as_str)
            .unwrap_or_default();
        for col in ["outcome", "channel", "resource_id", "token_hash"] {
            assert!(create.contains(col), "missing {col}: {create}");
        }
        assert!(!create.contains("detail"));
        assert_eq!(
            stmts
                .iter()
                .filter(|s| s.starts_with("CREATE INDEX"))
                .count(),
            2
        );
    }

    #[test]
    fn v3_adds_users_scopes_and_members_that_reference_users() {
        let stmts = v3_statements();
        assert!(
            stmts
                .first()
                .is_some_and(|s| s.starts_with("CREATE TABLE IF NOT EXISTS \"users\""))
        );
        assert!(stmts.iter().any(|s| s.contains("ADD COLUMN \"scopes\"")));
        assert!(
            stmts
                .iter()
                .any(|s| s.contains("ADD COLUMN \"last_used_at\""))
        );
        let members = stmts
            .iter()
            .find(|s| s.starts_with("CREATE TABLE IF NOT EXISTS \"members\""))
            .map(String::as_str)
            .unwrap_or_default();
        assert!(members.contains("REFERENCES \"users\""), "{members}");
        assert!(
            members.contains("PRIMARY KEY (\"workspace_id\", \"user_id\")"),
            "{members}"
        );
        assert!(
            stmts
                .iter()
                .any(|s| s.starts_with("INSERT INTO \"members\""))
        );
        assert!(stmts.iter().any(|s| s == "DROP TABLE \"members_v1\""));
        assert!(
            stmts
                .last()
                .is_some_and(|s| s.starts_with("CREATE UNIQUE INDEX"))
        );
        assert!(!stmts.iter().any(|s| s.contains("audit_log")));
    }

    #[test]
    fn versions_are_ascending_and_start_at_one() {
        let v = versions();
        assert_eq!(v.first().map(|(n, _)| *n), Some(1));
        let numbers: Vec<i64> = v.iter().map(|(n, _)| *n).collect();
        let mut sorted = numbers.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(numbers, sorted, "versions must be strictly ascending");
    }
}
