-- v8: a client quack registered with an issuer itself (`quack auth register`,
-- RFC 7591), named by the issuer (without a trailing slash): one registered
-- client serves every `[server.oidc]` and `[providers.NAME.oauth]` section at
-- that issuer that names no client_id, which is how they find it before they
-- know its id. `registration_access_token` is what RFC 7592 reads, updates,
-- and deletes the registration with: kept in `key_id`, `enc`, and
-- `ciphertext`, sealed with HPKE under the vault key (the OS keychain or
-- <data_dir>/vault.key, never here). An issuer that returns no such token
-- (or no `registration_client_uri`) leaves those columns NULL, and quack
-- cannot manage the client afterwards.
--
-- Frozen history once shipped: add a new version instead of editing this.

CREATE TABLE IF NOT EXISTS "client_registrations" (
    "name"                    text NOT NULL PRIMARY KEY,
    "client_id"               text NOT NULL,
    "key_id"                  text,
    "enc"                     blob,
    "ciphertext"              blob,
    "registration_client_uri" text,
    "created_at"              text NOT NULL DEFAULT (CURRENT_TIMESTAMP),
    "updated_at"              text NOT NULL DEFAULT (CURRENT_TIMESTAMP)
);
