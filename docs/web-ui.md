# Embedded server UI

How `quack serve` renders its web UI (design doc sections 11.1, 11.2, and 12): axum serves
the routes, rust-embed bakes the assets into the executable, askama renders the templates,
htmx refreshes the document list and the SQL grid, a small script streams the chat over SSE
and draws the charts and the graph, and Tailwind's standalone binary builds the CSS. The
code lives in `crates/quack/src/server/web/`, the templates in
`crates/quack/templates/`, and the assets in `crates/quack/static/`. Localization via
fluent is deferred (design doc section 18).

The web UI is a client of `quack-core` behind the API's checks. Every page handler calls
`Access::resolve` and the API's operations (`Access::execute_sql`, `enqueue`, `set_pinned`,
`delete_document`), so a script can do anything a page does and every page writes the same
audit rows.

Crates (from the workspace menu):

```toml
axum       = { workspace = true, features = ["http1", "json", "query", "form", "tokio", "multipart", "matched-path"] }
axum-extra = { workspace = true, features = ["cookie", "form"] }   # `form` handles repeated fields (checkboxes)
askama     = { workspace = true, features = ["derive", "std", "config"] }
rust-embed = { workspace = true, features = ["deterministic-timestamps"] }
mime_guess = { workspace = true }
```

## Identity for pages

`WebUser` wraps the API's `Identity` extractor and redirects a missing or bad credential to
`/login`; API routes answer 401 JSON instead. `HtmlError` renders errors through
`error.html` with the status and message.

With `[server.oidc]` set, `login.html` shows "Sign in with <issuer host>" above the password
form. `web::sign_in` serves `GET /auth/oidc` (redirect to the issuer, plus the state cookie)
and `GET /auth/oidc/callback`. A refusal lands back on `/login` with the reason in the flash
slot; a success opens the same session cookie a password login does.

## Templates

Every page struct carries a `page: Page`: title, username, admin flag, local flag, and the
current workspace with its role and what the caller may do. `base.html` reads it for the
header and the workspace tabs: chat, documents, tables (with the import form), SQL,
context, ontology, graph, settings.

- **`ontology.html`** shows the class tree, relations, properties, and mappings; the JSON
  editor; the version list with the diff to the previous version; the propose form; and the
  paged review queue with bulk accept and reject.
- **`graph.html`** shows status banners (provisional, stale, missing mapped tables,
  drift), the search and path forms, the ECharts result with a node inspector, the merge
  queue, and the extract, revalidate, and review buttons.
- **Dark only.** `styles/input.css` sets `color-scheme: dark`, so native controls and the
  file picker follow; panels are `slate-900` on a `slate-950` page and primary buttons are
  `blue-600`. Charts and the graph use ECharts' built-in `dark` theme.
- **Tables never let columns touch or overrun.** Listing tables carry `data-table` (padded
  cells, from `input.css`); prose and names wrap with `wrap-anywhere`; short fixed values
  (status, sizes, times) are `whitespace-nowrap`; data grids (sample rows, SQL results) and
  opaque ids keep one line per cell, cut at a width with "…", and carry the full value in a
  `title` tooltip. A sidebar layout's content track is `minmax(0,1fr)`, so a wide grid
  scrolls inside its own box instead of widening the page.
- **Times are localized in the browser.** A page renders `<time datetime="…Z"
  data-when="clock|relative">` with UTC text (`web::Moment`); `app.js` rewrites each in the
  viewer's zone, with the full date and time as the tooltip. `clock` (chat messages) is the
  time of day, with the date when not today; `relative` (the Jobs page) is "5 min ago" for
  today and the date before that, refreshed every 30 seconds and after every htmx swap.
  Chat messages show when they were asked or answered, and an answer how many milliseconds it
  took (`duration_ms` on the response object and the assistant message's metadata).
- **Fragments that htmx swaps** (`documents_rows.html`, `sql_result.html`) are their own
  structs, rendered to a string and inserted with `|safe`. askama escapes everything else.

A form handler calls the same typed operation as the REST handler (a method on `Access` or
`Identity` in `server::api`), so both apply the same validation and audit rows. The API
wraps the result in JSON; the web handler in a `web::flash::Flash` redirect. Redirects carry
outcomes in the query string: `?error=` renders red and `?notice=` green, so a started
background pass or a revalidation count does not look like a failure. The documents list polls its rows fragment only while a
row is processing (marked `data-pending`).

## Assets

- `static/css/output.css` is built from `styles/input.css` with `make css-build`
  (Tailwind v4.3.3 standalone, which scans `templates/` and `src/` for class names). It is
  **committed**: rust-embed needs it at compile time and CI has no Tailwind. Rebuild and
  commit it whenever a template changes classes.
- `static/js/htmx.min.js` (htmx 4.0.0) and `static/js/echarts.min.js` (ECharts 6.1.0) are
  vendored so the binary works air-gapped.
- `static/js/app.js` is quack's own. It posts to `/api/v1/workspaces/{id}/query/stream`,
  parses the SSE events, and renders the steps block, the answer, citations as links to the
  document list, and the chart spec as an ECharts option. On page load it renders stored
  charts and draws the graph page's result as an ECharts force graph (nodes coloured by
  class; a click scrolls to the inspector entry).

## Static handler

```rust
#[derive(rust_embed::Embed)]
#[folder = "static/"]
struct Assets;

async fn static_asset(Path(path): Path<String>) -> Response {
    let Some(file) = Assets::get(&path) else {
        return StatusCode::NOT_FOUND.into_response();
    };
    let mime = mime_guess::from_path(&path).first_or_octet_stream();
    ([(header::CONTENT_TYPE, mime.as_ref().to_owned())], file.data.into_owned()).into_response()
}
```

The real handler also sets an `ETag` from rust-embed's content hash and
`Cache-Control: no-cache`, and answers `If-None-Match` with 304, so a rebuilt stylesheet or
script loads without a hard refresh. `server::no_store` gives every
other response `Cache-Control: no-cache, no-store, must-revalidate`, `Expires: 0`, and
`Pragma: no-cache`, unless its handler already set `Cache-Control`.

## Tailwind pipeline

Tailwind v4 is CSS-first: `styles/input.css` is one `@import "tailwindcss";`. Build with:

```bash
make css-build     # minified, into crates/quack/static/css/output.css
make css-dev       # watch mode while editing templates
```
