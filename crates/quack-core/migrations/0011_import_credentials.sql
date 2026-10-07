-- v11: the secret a saved import refreshes with (its URL with the password,
-- and the header values it sends), sealed with HPKE under the vault key
-- when its owner saved it with `--store-credential`. The import itself, and
-- everything about what it reads, lives in the workspace file; this row
-- holds only the sealed bytes, keyed by the import's id, and goes with its
-- workspace.
--
-- Frozen history once shipped: add a new version instead of editing this.

CREATE TABLE IF NOT EXISTS "import_credentials" (
    "import_id"    text NOT NULL PRIMARY KEY,
    "workspace_id" text NOT NULL,
    "key_id"       text NOT NULL,
    "enc"          blob NOT NULL,
    "ciphertext"   blob NOT NULL,
    "updated_at"   text NOT NULL DEFAULT (CURRENT_TIMESTAMP),
    FOREIGN KEY ("workspace_id") REFERENCES "workspaces" ("id") ON DELETE CASCADE
);
