-- v10: workspace roles granted to an identity provider's groups, and which
-- memberships the provider's group claim granted (reconciled at sign-in)
-- as opposed to a person's hand.
--
-- Frozen history once shipped: add a new version instead of editing this.

CREATE TABLE IF NOT EXISTS "group_roles" (
    "workspace_id" text NOT NULL,
    "group_name"   text NOT NULL,
    "role"         text NOT NULL DEFAULT 'member',
    "created_at"   text NOT NULL DEFAULT (CURRENT_TIMESTAMP),
    PRIMARY KEY ("workspace_id", "group_name"),
    FOREIGN KEY ("workspace_id") REFERENCES "workspaces" ("id") ON DELETE CASCADE
);

ALTER TABLE "members" ADD COLUMN "granted_by" text NOT NULL DEFAULT 'user';
