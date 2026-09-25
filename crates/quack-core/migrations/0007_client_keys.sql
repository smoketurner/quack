-- v7: the P-256 private key quack signs its client assertions with
-- (`client_auth = "private_key_jwt"`, RFC 7523), one per OAuth client, named
-- `<issuer> <client_id>` so `[server.oidc]` and a model provider registered
-- as the same client share one key and one registered JWKS. Each key is
-- PKCS#8, sealed with HPKE under the vault key (the OS keychain or
-- <data_dir>/vault.key, never here).
--
-- Frozen history once shipped: add a new version instead of editing this.

CREATE TABLE IF NOT EXISTS "client_keys" (
    "name"       text NOT NULL PRIMARY KEY,
    "key_id"     text NOT NULL,
    "enc"        blob NOT NULL,
    "ciphertext" blob NOT NULL,
    "created_at" text NOT NULL DEFAULT (CURRENT_TIMESTAMP)
);
