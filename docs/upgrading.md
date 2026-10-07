# Upgrading

One section per release, newest first: the config keys it added, renamed, or removed; the
workspace schema version it moves to and what the first open of each workspace does; whether
`quack embeddings refresh` is needed; and any new `control.db` migration. The version-bump
pull request adds the release's section, and `release.yml` refuses to publish a tag this page
has no section for (`scripts/upgrading-section.sh`); the section leads the release notes.

Two rules hold for every release:

- Config loading is strict: a key quack no longer recognizes stops it at startup, so a removed
  or renamed key must change in `config.toml` before the new binary starts. `quack config`
  lists every key in force and the ones in the file it does not recognize.
- A workspace file records the schema version and the quack version that wrote it. A newer
  quack upgrades the file on first open; an older quack refuses a file a newer one wrote
  (`docs/migrations.md`). Back up before a release that moves the schema, or take a snapshot.

`quack doctor` after the upgrade says whether the workspace's vectors are stale
(`quack embeddings refresh -w NAME` is its fix line).

## v2026.10.4

- Breaking: `postgres://` import sources are removed (#458). Export the query to a SQLite
  file or a CSV or Parquet file and import that, or fetch the file over HTTP(S) or from S3.
  Imports already loaded stay as tables.
- Schema: workspace 14, from 11. The first open of each workspace upgrades it in place and
  does not run again: it profiles every user table (12), detects each document's language and
  rebuilds the keyword index, reading every chunk once (13), and adds `_quack_imports` for saved
  imports (14). Expect the first open of a large workspace to take longer. An older quack
  refuses the upgraded file, so back up or take a snapshot first.
- `control.db`: migrations 9 to 11 run on start: user lifecycle and login lockout, group roles,
  and the sealed credentials of saved imports.
- Config: keys added, each defaulted so an existing file keeps its behavior:
  - `[ingestion]`: `max_decompressed_mb` (1024), `table_rows_as_table` (20).
  - `[retrieval]`: `languages` (`["auto"]`).
  - `[analysis]`: `step_result_rows` (50), `title_sessions` (false).
  - `[graph]`: `follow_ingest` (`"off"`).
  - `[import]`: `allow_server_credentials` (false). With logins, S3 and `--bearer-env` imports
    are refused until it is set.
  - `[server]`: `shutdown_grace_seconds` (20), `log_format` (`"text"`), `login_lockout_attempts`
    (5) and `login_lockout_minutes` (15), `trusted_proxies` (empty). Five wrong passwords in a
    row now lock an account for 15 minutes; set `login_lockout_attempts = 0` to keep only the
    rate limit. Behind a reverse proxy, list it in `trusted_proxies` so client addresses come
    from its forwarded headers.
  - `[server.oidc]`: `groups_claim`. `[server.webhooks]`: `url`, `secret_env`, `kinds`,
    `timeout_seconds` (10).
  - `[providers.NAME]`: `max_retries` (3) and `retry_backoff_ms` (500).
  No keys were removed.
- Embeddings: no refresh needed.
- Config: `[ingestion].vision_model` added (unset by default): the model that describes
  uploaded images at ingest; without it, image uploads are refused. `images = true` on a
  provider or model gives the agent `view_image`.

## v2026.10.3

- Config: no keys added or removed. The proxy variables (`HTTP_PROXY`, `HTTPS_PROXY`,
  `ALL_PROXY`, `NO_PROXY`) are now read; `quack config` lists them.
- Schema: unchanged (workspace 11, `control.db` migration 8).
- Embeddings: no refresh needed.

## v2026.10.2

- No upgrade steps.

## v2026.10.1

- Config: added `[analysis].compact_history`, `[retrieval].rerank_model` (with
  `[retrieval].rerank = "reranker"`), and `[server].permission_timeout_seconds`. All have
  defaults.
- Schema: unchanged.
- Embeddings: no refresh needed.

## v2026.9.8

- No upgrade steps.

## v2026.9.7

- Config: added `temperature`, `effort`, and `background_effort` on `[providers.NAME]` and
  the `[providers.NAME.models."ID"]` tables (`models`). All optional.
- Schema: unchanged.

## v2026.9.6

- No upgrade steps.

## v2026.9.5

- No upgrade steps.

## v2026.9.4

- Config: added `[providers.NAME].headers`. Optional.
- Schema: unchanged.

## v2026.9.3

- Config: `quack config` arrived with this release, and with it the strict key list; a key
  the file carried that quack did not read now stops it at startup, with the key it
  resembles named. Run `quack config` once and fix what it reports.
- `control.db`: migrations 0004 (audit keyset index), 0005 (`user_tokens`), 0006
  (`provider_tokens`), 0007 (`client_keys`), and 0008 (`client_registrations`) run on the
  first start.
- Workspace schema: moves to 11. The first open of each workspace rebuilds the term index
  (versions 6 and 7: stemming, then the joined identifier term), tags every stored vector
  with the embedding profile the workspace had recorded (8), marks documents left in the
  old `pending` status as `error` (9), records each ontology version's acceptance (10), and
  collapses duplicated merge proposals (11). On a workspace with many chunks the first open
  takes minutes; the server answers that workspace once it finishes.
- Embeddings: `quack embeddings refresh` only if the embedding model, width, or prefixes in
  `config.toml` differ from what the workspace recorded; `quack doctor` says so.

## v2026.9.2

- No upgrade steps.

## v2026.9.1

The first release.
