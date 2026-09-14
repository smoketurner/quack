# Migrations

Two databases carry schema, and each is versioned on its own side of the classification
boundary (design doc section 5).

## `control.db` (SQLite, server mode)

Managed by sea-query DDL builders in `quack-core/src/storage/migrations.rs`. Each schema
version generates a set of `CREATE TABLE IF NOT EXISTS` statements, applied by
`ControlPlane::open()` and recorded in `schema_version`.

Authoring rules:

- **UUID v7 primary keys** (client-supplied, `uuid::Uuid::now_v7()`), no `SERIAL`.
- **One logical change per version** — keep each version's statements focused.
- Use sea-query's `Table::create()` builder rather than raw SQL strings.
- Use `if_not_exists()` so migrations are idempotent.
- **Nothing that reveals workspace content.** `control.db` holds users, workspaces (name and
  label), membership, tokens, and the access audit log. Content detail belongs in the
  workspace file.
- `audit_log` is append-only: no migration or code path may add an `UPDATE` or `DELETE`
  against it.

## Workspace DuckDB files

Each workspace's `data.duckdb` carries the `_quack_` tables (documents, chunks, ontology,
graph, provenance, sessions, messages, context, audit detail; design doc section 5.4)
alongside user tables and views. Their schema is versioned by
`_quack_meta.schema_version`, applied by `WorkspaceDb::open()`.

Authoring rules:

- Statements are parameterized with `duckdb::params!`; the only interpolated values are
  identifiers through `quote_ident` and the validated `FLOAT[N]` embedding width.
- Every internal table is prefixed `_quack_` so it can be hidden from the agent's table
  listing and refused in agent SQL.
- The embedding dimension is fixed per workspace and recorded in `_quack_meta`; a
  migration never changes it. Changing the embedding model is a guided re-embed.
- The ontology and the workspace context are versioned as data (`_quack_ontology_versions`,
  `_quack_context`), not by schema migrations.
- Migrations must be safe to run on a workspace that was created by an older binary and
  copied from another machine: a workspace directory is portable.

User tables and views are never touched by migrations.
