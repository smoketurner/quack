use sea_query::Iden;

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
