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
  (`docs/migrations.md`). Before a release that moves the schema, copy the data directory or
  run `quack workspace snapshot` with the release you are leaving: a snapshot the new binary
  takes opens, and so upgrades, the workspace first.

`quack doctor` with the new binary reads each workspace's schema version without upgrading it,
says which will be upgraded on their next open, and, once they are current, whether their
vectors are stale (`quack embeddings refresh -w NAME` is its fix line).

## Unreleased

- Changed: the server and MCP code moved into the `quack-server` crate, so its log targets
  change from `quack::server::...` and `quack::mcp` to `quack_server::...`. Update `RUST_LOG`
  filters and any collector rules that match the old targets. The access log keeps its
  `quack::access` target.
- Added: `quack init` finds a running Ollama and the API keys and AWS credentials in the
  environment, asks which chat and embedding models to use, and sets them in `config.toml`
  once `quack doctor` passes the result. It edits an existing file in place, keeping every
  other setting and comment. `quack` with no config file offers to run it. The "no chat model"
  error, the terminal's first-run note, and `quack doctor`'s chat model fix now point to it.
- Changed: the terminal keeps typed input in each workspace's file (`_quack_input_history`,
  the newest 500 lines) instead of `<data_dir>/terminal_history`, which held every
  workspace's questions and SQL outside any workspace file. The first terminal session
  deletes that file; its lines are not carried over, so Up starts empty in every workspace.
  The table needs no backfill and the schema version does not change.
- Breaking: `[analysis].max_context_tokens` is removed. Delete it from `config.toml` before
  starting the new binary. Ollama chat requests no longer carry `num_ctx` or `keep_alive`, so
  set the window and the load time on the Ollama server: `OLLAMA_CONTEXT_LENGTH=32768` keeps
  the old default window, and `OLLAMA_KEEP_ALIVE=30m` keeps the old load time.
- Changed: `effort` and `background_effort` go out as rig's reasoning option, and rig's model
  catalog decides which levels a model takes. A level the model lacks fails when the model is
  built, and `quack doctor` reports it as a failure, where quack used to send nothing and warn
  (an OpenAI model on Converse, for example). Any Ollama model is now sent `think`, not only
  gpt-oss. `effort = "none"` on Claude turns thinking off where the model allows it. Adaptive
  Claude models are also sent `thinking: adaptive` with the effort.
- Changed: `quack ontology propose --auto-accept` exits 0 when every candidate has low support,
  as with a single document. It accepts nothing and points to
  `quack ontology review --low-support`. It used to exit 1 with "no pending candidates for run".
- Changed: in `quack config --format json`, each unrecognized key carries `hint` in place of
  `suggestion`: `{"did_you_mean": "top_k"}`, or `{"removed": "..."}` for a key an earlier
  release read, with what took its place.

- Breaking: `GET /api/v1/workspaces/{id}/documents` and MCP `list_documents` return one page,
  newest first: 100 documents unless `limit` asks for more (500 at most), with `total` and,
  when more remain, `next`. Pass `next` as `after` for the following page. A client that read
  `documents` as the whole list must follow `next`. The `quack://workspace/.../documents`
  resource is the first page.
- Changed: the agent's `list_documents` lists 50 documents a call and ends with the total and
  the `after` for the next 50. The web Documents page shows 100 at a time, with Older and
  Newest links.
- Changed: every document listing is ordered by document id (a UUID v7, the order documents
  were registered in) instead of `ingested_at`. The two orders agree for documents quack
  registered.

## v2026.10.4

- Breaking: `-w NAME` no longer creates a workspace that does not exist; it exits 2 (#434).
  Create one first with `quack workspace create NAME`. Only `[general].default_workspace` is
  still created on first use.
- Breaking: `[general].default_workspace` must be a valid workspace name (not blank, no `/`,
  `\`, or `.`), or the config is refused at startup (#434). A workspace already named so
  still opens with `-w`.
- Changed: the terminal and the web chat answer a write request the same way. In the
  terminal, `a` now allows the writes of the turn that asked, not the rest of the session;
  restart with `--allow-write` to allow every write. SQL typed in the terminal runs without
  asking, as on the web SQL page; only the agent's writes ask. `/schema TABLE` is now
  `/tables TABLE`, which also takes `--note` and `--retype` as `quack tables` does.
- Breaking: `postgres://` import sources are removed (#458). Export the query to a SQLite
  file or a CSV or Parquet file and import that, or fetch the file over HTTP(S) or from S3.
  Imports already loaded stay as tables.
- Schema: workspace 14, from 11. The first open of each workspace upgrades it in place and
  does not run again: it profiles every user table (12), detects each document's language and
  rebuilds the keyword index, reading every chunk once (13), and adds `_quack_imports` for saved
  imports (14). Expect the first open of a large workspace to take longer. An older quack
  refuses the upgraded file, so back up or take a snapshot first.
- `control.db`: migrations 9 to 12 run on start: user lifecycle and login lockout, group roles,
  and the sealed credentials of saved imports, kept per workspace.
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
