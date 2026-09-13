# sea-query as the query builder

sea-query builds type-safe SQL queries for the SQLite control plane. Handlers call typed
store methods and never see raw SQL.

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
