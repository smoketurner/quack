# Schema and migrations

Two databases carry schema, each versioned on its own side of the classification boundary
(design doc section 5).

## `control.db` (SQLite, server mode)

Managed by sea-query DDL builders in `quack-core/src/storage/migrations.rs`. Each schema
version is a set of `CREATE TABLE IF NOT EXISTS` statements, applied by
`ControlPlane::open()` and recorded in `schema_version`.

- UUID v7 primary keys, client-generated with `uuid::Uuid::now_v7()`; no `SERIAL`.
- One logical change per version; `if_not_exists()` so versions are idempotent.
- Queries are built with sea-query (`Iden` enums per table, `SqliteQueryBuilder`) and run
  through sqlx wrapped in `AssertSqlSafe`, because sea-query emits dynamic SQL strings.
  Handlers call typed store methods and never see SQL.
- Nothing that reveals workspace content: users, workspaces (name and label), membership,
  tokens, and the access `audit_log`. The audit log is append-only; no code path may
  `UPDATE` or `DELETE` it.

## Workspace DuckDB files

Each workspace's `data.duckdb` carries the `_quack_` tables (documents, chunks, terms,
ontology, graph, provenance, merges, sessions, messages, context, audit detail; design doc
section 5.4) beside the user's tables and views. `WorkspaceDb::open()` creates what is
missing and records `_quack_meta.schema_version`; a bump can trigger a rebuild, as version
6 rebuilt the term index when stemming arrived.

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
