use sea_query::Iden;

// --- Control plane tables (SQLite) ---

#[derive(Iden)]
pub enum SchemaVersion {
    Table,
    Version,
    AppliedAt,
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
    CreatedAt,
    ExpiresAt,
}

#[derive(Iden)]
pub enum Threads {
    Table,
    Id,
    WorkspaceId,
    Title,
    CreatedBy,
    CreatedAt,
}

#[derive(Iden)]
pub enum Messages {
    Table,
    Id,
    ThreadId,
    Role,
    Content,
    Metadata,
    CreatedAt,
}

#[derive(Iden)]
pub enum AuditLog {
    Table,
    Id,
    Timestamp,
    WorkspaceId,
    UserId,
    Action,
    Detail,
}
