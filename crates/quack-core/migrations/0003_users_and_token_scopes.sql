-- v3: server users; token scopes and last use; members reference users.
--
-- "members" is rebuilt rather than altered because SQLite cannot add a foreign
-- key or a composite primary key to an existing table. Rows whose user_id has
-- no matching user are dropped on the way across.
--
-- Frozen history: this file has shipped. Never edit it — add a new version
-- instead.

CREATE TABLE IF NOT EXISTS "users" (
    "id"            text NOT NULL PRIMARY KEY,
    "username"      text NOT NULL UNIQUE,
    "password_hash" text,
    "oidc_subject"  text UNIQUE,
    "is_admin"      integer NOT NULL DEFAULT 0,
    "created_at"    text NOT NULL DEFAULT (CURRENT_TIMESTAMP)
);

ALTER TABLE "api_tokens" ADD COLUMN "scopes" text NOT NULL DEFAULT '["read"]';
ALTER TABLE "api_tokens" ADD COLUMN "last_used_at" text;

ALTER TABLE "members" RENAME TO "members_v1";

CREATE TABLE IF NOT EXISTS "members" (
    "workspace_id" text NOT NULL,
    "user_id"      text NOT NULL,
    "role"         text NOT NULL DEFAULT 'member',
    "created_at"   text NOT NULL DEFAULT (CURRENT_TIMESTAMP),
    PRIMARY KEY ("workspace_id", "user_id"),
    FOREIGN KEY ("workspace_id") REFERENCES "workspaces" ("id") ON DELETE CASCADE,
    FOREIGN KEY ("user_id") REFERENCES "users" ("id") ON DELETE CASCADE
);

INSERT INTO "members" (workspace_id, user_id, role, created_at)
SELECT m.workspace_id, m.user_id, m.role, m.created_at
FROM "members_v1" m
WHERE EXISTS (SELECT 1 FROM "users" u WHERE u.id = m.user_id);

DROP TABLE "members_v1";

CREATE UNIQUE INDEX IF NOT EXISTS "workspaces_name" ON "workspaces" ("name");
