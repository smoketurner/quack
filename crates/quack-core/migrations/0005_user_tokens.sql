-- v5: the identity-provider token of each user who signed in through
-- [server.oidc], sealed with HPKE under the server's key (which is kept in
-- the OS keychain, never here). A row goes with its user.
--
-- Frozen history once shipped: add a new version instead of editing this.

CREATE TABLE IF NOT EXISTS "user_tokens" (
    "user_id"    text NOT NULL PRIMARY KEY,
    "key_id"     text NOT NULL,
    "enc"        blob NOT NULL,
    "ciphertext" blob NOT NULL,
    "updated_at" text NOT NULL DEFAULT (CURRENT_TIMESTAMP),
    FOREIGN KEY ("user_id") REFERENCES "users" ("id") ON DELETE CASCADE
);
