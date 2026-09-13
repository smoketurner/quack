# Migrations

The SQLite control plane schema is managed by sea-query DDL builders in
`quack-core/src/storage/migrations.rs`. Each schema version generates a set of
`CREATE TABLE IF NOT EXISTS` statements.

## Authoring rules

- **UUID v7 primary keys** (client-supplied, `uuid::Uuid::now_v7()`), no `SERIAL`.
- **One logical change per version** — keep each version's statements focused.
- Use sea-query's `Table::create()` builder rather than raw SQL strings.
- Use `if_not_exists()` so migrations are idempotent.

## Running them

Migrations run automatically when `ControlPlane::open()` is called. The `schema_version`
table tracks which versions have been applied.

## DuckDB workspaces

DuckDB workspace databases have no managed migrations — they accept arbitrary user SQL.
Each workspace gets its own isolated database file at
`{data_dir}/workspaces/{id}/data.duckdb`.
