-- v1: access-control tables.
--
-- Frozen history: this file has shipped. Never edit it — add a new version
-- instead. sqlx records a checksum per file and refuses a database whose
-- recorded checksum no longer matches.

CREATE TABLE IF NOT EXISTS "workspaces" (
    "id"                text NOT NULL PRIMARY KEY,
    "name"              text NOT NULL,
    "classification"    text NOT NULL DEFAULT 'internal',
    "allowed_providers" text,
    "created_at"        text NOT NULL DEFAULT (CURRENT_TIMESTAMP),
    "updated_at"        text NOT NULL DEFAULT (CURRENT_TIMESTAMP)
);

-- No reference to "users", which does not exist until 0003.
CREATE TABLE IF NOT EXISTS "members" (
    "workspace_id" text NOT NULL,
    "user_id"      text NOT NULL,
    "role"         text NOT NULL DEFAULT 'member',
    "created_at"   text NOT NULL DEFAULT (CURRENT_TIMESTAMP),
    FOREIGN KEY ("workspace_id") REFERENCES "workspaces" ("id") ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS "api_tokens" (
    "token_hash"   text NOT NULL PRIMARY KEY,
    "workspace_id" text NOT NULL,
    "user_id"      text NOT NULL,
    "name"         text NOT NULL,
    "created_at"   text NOT NULL DEFAULT (CURRENT_TIMESTAMP),
    "expires_at"   text,
    FOREIGN KEY ("workspace_id") REFERENCES "workspaces" ("id") ON DELETE CASCADE
);
