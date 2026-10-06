# Schema and migrations

Two databases carry schema, each versioned on its own side of the classification boundary
(design doc section 5).

## `control.db` (SQLite, server mode)

`sqlx::migrate!` embeds the SQL files under `crates/quack-core/migrations/` at compile
time, and `ControlPlane::open()` applies them. `_sqlx_migrations` records each applied
version with a SHA-384 checksum of its file.

- One file per version, named `NNNN_description.sql`. Add a version by adding a file;
  never edit a shipped one. sqlx compares checksums on open and, by design, refuses a
  database whose recorded checksum no longer matches.
- A migration file is frozen history, never regenerated from the `Iden` enums in
  `storage::queries`. Those track the *current* schema: renaming a column there would
  silently change an old version's output, and a fresh database and one migrated by an
  older binary would disagree while reporting the same version.
- Each file runs in one transaction (SQLite has transactional DDL), so a failed migration
  leaves nothing half-applied.
- UUID v7 primary keys, client-generated with `uuid::Uuid::now_v7()`; no `SERIAL`.
- Only DDL lives in files. Runtime queries use sea-query (`Iden` enums per table,
  `SqliteQueryBuilder`) run through sqlx wrapped in `AssertSqlSafe`, since sea-query emits
  dynamic SQL strings. Handlers call typed store methods and never see SQL.
- Nothing that reveals workspace content. The database holds users, workspaces (name and
  label), membership, tokens (API token hashes; signed-in users' and model providers' OAuth
  tokens in `user_tokens` and `provider_tokens`), each OAuth client's `private_key_jwt`
  signing key in `client_keys` (version 7), the clients quack registered itself (RFC 7591)
  with their registration access tokens in `client_registrations` (version 8), and the
  access `audit_log`. The vault seals every token and key. The audit log is append-only; no
  code path may `UPDATE` or `DELETE` it.

### Databases created before the switch

Versions 1 to 3 were applied by sea-query DDL builders that recorded progress in a
`schema_version` table. `ControlPlane::open()` adopts such a database: it reads
`MAX(version)` from `schema_version` and calls `Migrator::skip` for those versions, which
records them in `_sqlx_migrations` without running them. A replay would not be idempotent:
version 2 drops and recreates `audit_log`, destroying the access record. The inert
`schema_version` table stays, which keeps an older binary from re-running those migrations
on the same file.

## Workspace DuckDB files

Each workspace's `data.duckdb` holds the `_quack_` tables beside the user's tables and
views: documents, chunks, terms, ontology, graph, provenance, merges, sessions, messages,
saved questions and their runs, context, and audit detail (design doc section 5.4). `WorkspaceDb::open()` creates what is
missing and records `_quack_meta.schema_version`. A version bump can trigger a rebuild on
open:

- **6**: Rebuilt the term index when stemming arrived.
- **7**: Rebuilt it again to add the joined identifier term (`pol8841` alongside `pol` and
  `8841` for `POL-8841`; issue #77).
- **8**: Tagged every stored vector with the embedding profile it was made under (the model
  the workspace had recorded, no prefixes).
- **9**: Marks documents still carrying the old `pending` default (or no status) as `error`:
  a status is now one of `queued`, `processing`, `ready`, or `error`, and those rows were
  never processed.
- **10**: Adds `_quack_ontology_versions.acceptance` (`reviewed` or `auto`) and marks
  versions whose note starts `auto-accepted` as `auto`, the only earlier record of an
  auto-accept.
- **11**: Collapses opposing-orientation rows in `_quack_graph_merges`. Before the dedup
  matched a node pair in either orientation, a pair whose provenance flipped between
  resolution passes could land twice (`(keep, drop)` and `(drop, keep)`). Each pair keeps
  one row, the more-decided one, so a reviewer's rejection survives.

A column that needs no backfill needs no bump: `ADD COLUMN IF NOT EXISTS` on open adds it,
and rows written earlier read as `NULL`. `_quack_documents.page_count`, `pages_unreadable`,
`pages_empty`, `superseded_by` (the id of the document replacing this one; the status
`superseded` joined the four above with it), and `source_path` (the file's path under the
folder `quack ingest DIR` read it from) arrived this way.

Phrase search (`"..."` in a keyword query) needed no bump: it post-filters candidates by
substring instead of adding term positions to `_quack_terms`.

This side is *not* a numbered migration list. It stays in Rust because the DDL is
parameterized by the embedding width (`FLOAT[{dim}]`, `graph::ddl(dim)`) and the
version-keyed steps are data rebuilds (the term reindex), not SQL. It converges instead:
every open replays `CREATE TABLE IF NOT EXISTS` plus `ADD COLUMN IF NOT EXISTS`.

- Internal statements are constant strings with `duckdb::params!` bindings. The only
  interpolated values are identifiers through `quote_ident` and the validated `FLOAT[N]`
  embedding width. sea-query is not used here: its SQLite backend cannot express DuckDB's
  arrays or recursive CTEs.
- The `_quack_` prefix hides every internal table from the agent's table listing, and user
  and agent SQL referencing one is refused.
- The embedding dimension is fixed per workspace and recorded in `_quack_meta`; no schema
  version changes it. A new embedding model, width, or input prefix leaves vectors stale,
  never dropped on open; `quack embeddings refresh` updates them, retyping the vector
  columns first when the width changed.
- The ontology and the workspace context are versioned as data
  (`_quack_ontology_versions`, `_quack_context`), not by schema versions.
- A workspace directory is portable: every version must open a file an older binary created
  on another machine. User tables and views are never touched.

### Upgrading and rolling back

Upgrades run on open. A newer quack opens an older workspace file, runs the steps above,
and records three values in `_quack_meta`: `schema_version`, `written_by_quack` (its own
version), and `written_by_duckdb` (the DuckDB library it was built with). `quack doctor`
prints all three for the workspace it checks.

Rolling back is refused. An older quack that opens a file whose `schema_version` is above
its own stops with `Error::WorkspaceTooNew` before it runs any statement that changes the
file: no table definition, no rebuild, no version write. The message names the file, both
schema versions, and the quack that wrote it:

```
workspace file .../data.duckdb has schema version 12, written by quack 2026.11.0; this quack
(2026.10.3) reads up to version 11: run quack 2026.11.0 or newer, or restore the copy of the
workspace made before the upgrade
```

The CLI prints it and exits 1, the server answers 503, and `quack doctor` fails the
workspace check with that advice as its fix. Two ways out:

1. Run the quack version the message names, or a newer one.
2. Restore the copy of the workspace directory made before the upgrade.

So copy the data directory before upgrading quack if rolling back must stay possible. A
file last written before `written_by_quack` existed names no version; the message then
asks for "a newer quack". A `schema_version` that is not a number is refused the same way,
before any statement runs (`Error::WorkspaceSchemaUnreadable`): quack does not guess which
schema such a file holds.

`control.db` has the same guard from sqlx: the migrator refuses a database that holds a
migration the binary lacks.

### DuckDB's storage format

DuckDB versions its file format separately from quack's schema. Two facts, checked against
the bundled DuckDB 1.5.6:

- DuckDB picks the format when it creates a file, from `storage_compatibility_version`.
  The default is `v0.10.2`; `duckdb_databases()` reports such a file as storage version
  `v1.0.0+`.
- An existing file keeps its format. Files created as `v1.0.0+`, `v1.2.0+`, `v1.4.0+`, and
  `v1.5.0+` each reported the same storage version after a `SET` to `v1.5.0` or `latest`,
  a write, and a checkpoint. A `v1.0.0+` file kept its version when opened with the setting
  at `v0.10.2`, `v1.5.0`, or `latest`, and a `v1.5.0+` file keeps it when quack opens it
  (the test below).

`WorkspaceDb::open` passes `storage_compatibility_version = 'v0.10.2'` when it opens the
file (`STORAGE_COMPATIBILITY_VERSION`). The setting has to go in at open: a `SET` after the
file exists changes nothing, and after `lock_configuration` it is an error. Naming the
default changes no file today. It means a `duckdb` upgrade whose default differs cannot
change the format of new workspace files: the test
`naming_the_storage_compatibility_version_changes_no_file` fails until someone decides.

Raising the constant is a deliberate step. New files then need at least that DuckDB, so no
quack built on an older one can open them. Existing files keep their format. Record the
change here when it happens.
