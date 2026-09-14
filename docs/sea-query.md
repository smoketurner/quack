# sea-query as the query builder

sea-query builds type-safe SQL queries for `control.db`, the SQLite access-control
database used in server mode (design doc section 5.5). Handlers call typed store methods
and never see raw SQL.

It is **not** used for the workspace DuckDB files: sea-query's SQLite backend does not
produce DuckDB's array, extension, or recursive-CTE syntax. Internal DuckDB statements
(chunks, graph, ontology, sessions) are constant strings with `duckdb::params!` bindings,
and the only interpolated values are identifiers via `quote_ident` and the validated
`FLOAT[N]` width (design doc section 5.6). Agent- and user-written SQL is executed as-is
through the permission layer and is never assembled by application code.

Crates (from the workspace menu):

```toml
sea-query = { workspace = true, features = ["backend-sqlite", "derive"] }
sqlx      = { workspace = true, features = ["runtime-tokio", "sqlite"] }
```

## Schema as an `Iden` enum

```rust
use sea_query::Iden;

#[derive(Iden)]
enum Workspaces {
    Table,
    Id,
    Name,
    Classification,
    // ...
}
```

## Query building

```rust
use sea_query::{Expr, ExprTrait, Query, SqliteQueryBuilder};

let sql = Query::select()
    .column(Workspaces::Id)
    .column(Workspaces::Name)
    .from(Workspaces::Table)
    .and_where(Expr::col(Workspaces::Name).eq(name))
    .to_string(SqliteQueryBuilder);
```

## Executing with sqlx

Since sea-query generates dynamic SQL strings, wrap them with `AssertSqlSafe` for sqlx 0.9+:

```rust
use sqlx::AssertSqlSafe;

let row = sqlx::query(AssertSqlSafe(sql.as_str()))
    .fetch_optional(&pool)
    .await?;
```

Generate ids client-side with `uuid::Uuid::now_v7()` (UUID v7, time-ordered) so you know
the id before insert.
