-- v6: the token `quack auth login` (or a provider's client-credentials
-- grant) obtained for each OAuth model provider, sealed with HPKE under the
-- vault key (the OS keychain or <data_dir>/vault.key, never here). Replaces
-- the per-provider encrypted files under <data_dir>/tokens/.
--
-- Frozen history once shipped: add a new version instead of editing this.

CREATE TABLE IF NOT EXISTS "provider_tokens" (
    "provider"   text NOT NULL PRIMARY KEY,
    "key_id"     text NOT NULL,
    "enc"        blob NOT NULL,
    "ciphertext" blob NOT NULL,
    "updated_at" text NOT NULL DEFAULT (CURRENT_TIMESTAMP)
);
