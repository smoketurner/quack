# Schema and migrations

Two databases carry schema, each versioned on its own side of the classification boundary
(design doc section 5).

## `control.db` (SQLite, server mode)

Plain SQL files under `crates/quack-core/migrations/`, embedded at compile time by
`sqlx::migrate!` and applied by `ControlPlane::open()`. Applied versions are recorded in
`_sqlx_migrations` with a SHA-384 checksum of the file.

- One file per version, named `NNNN_description.sql`. Add a version by adding a file; never
  edit one that has shipped — sqlx compares checksums on open and refuses a database whose
  recorded checksum no longer matches, which is the point of the format.
- A migration file is frozen history. It must not be regenerated from the `Iden` enums in
  `storage::queries`, because those track the *current* schema: renaming a column there
  would silently change what an old version emits, so a freshly created database and one
  migrated by an older binary would disagree while both reported the same version.
- Each file runs in one transaction (SQLite has transactional DDL), so a failed migration
  leaves nothing half-applied.
- UUID v7 primary keys, client-generated with `uuid::Uuid::now_v7()`; no `SERIAL`.
- Runtime queries are still built with sea-query (`Iden` enums per table,
  `SqliteQueryBuilder`) and run through sqlx wrapped in `AssertSqlSafe`, because sea-query
  emits dynamic SQL strings. Handlers call typed store methods and never see SQL. Only DDL
  moved to files.
- Nothing that reveals workspace content: users, workspaces (name and label), membership,
  tokens, and the access `audit_log`. The audit log is append-only; no code path may
  `UPDATE` or `DELETE` it.

### Databases created before the switch

Versions 1 to 3 were applied by sea-query DDL builders that recorded progress in a
`schema_version` table. `ControlPlane::open()` adopts such a database: it reads
`MAX(version)` from `schema_version` and calls `Migrator::skip` for those versions, which
records them in `_sqlx_migrations` without executing them. Replaying them would not be
idempotent — version 2 drops and recreates `audit_log`, which would destroy the access
record. The `schema_version` table is left in place; it is inert, and it keeps an older
binary from re-running the same migrations against the same file.

## Workspace DuckDB files

Each workspace's `data.duckdb` carries the `_quack_` tables (documents, chunks, terms,
ontology, graph, provenance, merges, sessions, messages, context, audit detail; design doc
section 5.4) beside the user's tables and views. `WorkspaceDb::open()` creates what is
missing and records `_quack_meta.schema_version`; a bump can trigger a rebuild, as version
6 rebuilt the term index when stemming arrived.

This side is deliberately *not* a numbered migration list, and stays in Rust: the DDL is
parameterized by the workspace's embedding width (`FLOAT[{dim}]`, `graph::ddl(dim)`), and
the version-keyed steps are data rebuilds (the term reindex), not SQL. It converges instead
of migrating — `CREATE TABLE IF NOT EXISTS` plus `ADD COLUMN IF NOT EXISTS`, replayed on
every open.

- Internal statements are constant strings with `duckdb::params!` bindings. The only
  interpolated values are identifiers through `quote_ident` and the validated `FLOAT[N]`
  embedding width. sea-query is not used here: its SQLite backend cannot express DuckDB's
  arrays or recursive CTEs.
- Every internal table is prefixed `_quack_` so it is hidden from the agent's table listing
  and refused in user and agent SQL.
- The embedding dimension is fixed per workspace and recorded in `_quack_meta`; a schema
  version never changes it. Changing the embedding model is a re-embed.
- The ontology and the workspace context are versioned as data
  (`_quack_ontology_versions`, `_quack_context`), not by schema versions.
- A workspace directory is portable: every version must open a file created by an older
  binary on another machine. User tables and views are never touched.
