-- v12: a saved import's secret is keyed by its workspace and its id
-- together. A restored snapshot carries the same import ids as the workspace
-- it was taken from, so keyed by id alone a restored copy read, replaced,
-- and deleted the original's secret.
--
-- Frozen history once shipped: add a new version instead of editing this.

CREATE TABLE "import_credentials_v12" (
    "workspace_id" text NOT NULL,
    "import_id"    text NOT NULL,
    "key_id"       text NOT NULL,
    "enc"          blob NOT NULL,
    "ciphertext"   blob NOT NULL,
    "updated_at"   text NOT NULL DEFAULT (CURRENT_TIMESTAMP),
    PRIMARY KEY ("workspace_id", "import_id"),
    FOREIGN KEY ("workspace_id") REFERENCES "workspaces" ("id") ON DELETE CASCADE
);

INSERT INTO "import_credentials_v12"
    ("workspace_id", "import_id", "key_id", "enc", "ciphertext", "updated_at")
SELECT "workspace_id", "import_id", "key_id", "enc", "ciphertext", "updated_at"
FROM "import_credentials";

DROP TABLE "import_credentials";

ALTER TABLE "import_credentials_v12" RENAME TO "import_credentials";
