# Embedded server UI

How `quack serve` renders its web UI (design doc sections 11.1, 11.2, and 12): axum serves
the routes, rust-embed bakes the assets into the executable, askama renders the templates,
htmx refreshes the document list and the SQL grid, and a small script streams the chat
over the SSE endpoint and draws the charts and the graph. Tailwind builds the CSS with the
standalone binary. The code lives in
`crates/quack/src/server/web/`, the templates in `crates/quack/templates/`, and the assets
in `crates/quack/static/`. Localization via fluent is deferred (design doc section 18).

The web UI is a client of `quack-core` through the same checks as the API: every page
handler calls `Access::resolve` and the API's operations (`Access::execute_sql`, `enqueue`, `set_pinned`,
`delete_document`), so nothing a page does is unavailable to a script, and every page
writes the same audit rows.

Crates (from the workspace menu):

```toml
axum       = { workspace = true, features = ["http1", "json", "query", "form", "tokio", "multipart", "matched-path"] }
axum-extra = { workspace = true, features = ["cookie", "form"] }   # `form` handles repeated fields (checkboxes)
askama     = { workspace = true, features = ["derive", "std", "config"] }
rust-embed = { workspace = true, features = ["deterministic-timestamps"] }
mime_guess = { workspace = true }
```

## Identity for pages

`WebUser` wraps the API's `Identity` extractor and turns a missing or bad credential into a
redirect to `/login`; API routes answer 401 JSON instead. Errors render `error.html` with
the status and message through `HtmlError`.

## Templates

Every page struct carries a `page: Page` (title, username, admin flag, local flag, and the
current workspace with its role and what the caller may do), which `base.html` reads for the
header and the workspace tabs: chat, documents, tables (with the import form), SQL, context,
ontology, graph, settings. The ontology page (`ontology.html`) shows the class tree,
relations, properties, and mappings, the JSON editor, the version list with the diff to the
previous version, the propose form, and the paged review queue with bulk accept and reject;
the graph page (`graph.html`) shows status banners (provisional, stale, missing mapped
tables, drift), the search and path forms, the ECharts result with a node inspector, the
merge queue, and the extract, revalidate, and review buttons. Fragments that htmx swaps (`documents_rows.html`,
`sql_result.html`) are their own structs, rendered to a string and inserted with `|safe`.
askama escapes everything else. A form handler calls the same operation the REST handler does
(a method on `Access` or `Identity` in `server::api`, returning a typed result), so both
interfaces apply the same validation and write the same audit rows; the API wraps the result in
JSON and the web handler in a `web::flash::Flash` redirect. Redirects carry outcomes in the
query string: `?error=` renders red, `?notice=` green, so a started background pass or a
revalidation count is not styled as a failure. The documents list polls its rows fragment only while a row is still processing
(the fragment marks that with `data-pending`).

## Assets

- `static/css/output.css` is built from `styles/input.css` with `make css-build`
  (Tailwind v4.3.3 standalone, which scans `templates/` and `src/` for class names) and
  **committed**, because rust-embed needs it at compile time and CI has no Tailwind.
  Rebuild and commit it whenever a template changes classes.
- `static/js/htmx.min.js` (htmx 4.0.0) and `static/js/echarts.min.js` (ECharts 6.1.0) are
  vendored so the binary works air-gapped. `static/js/app.js` is quack's own: it posts to
  `/api/v1/workspaces/{id}/query/stream`, parses the SSE events, renders the steps block,
  the answer, citations as links to the document list, and the chart spec as an ECharts
  option; on page load it renders stored charts and the graph page's result as an ECharts
  force graph (nodes coloured by class, click scrolls to the inspector entry).

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
`Cache-Control: no-cache`, answering `If-None-Match` with 304, so a rebuilt stylesheet
or script is picked up on the next load without a hard refresh. Every other response
gets `Cache-Control: no-cache, no-store, must-revalidate`, `Expires: 0`, and
`Pragma: no-cache` from `server::no_store`, which leaves a response alone when its
handler already set `Cache-Control`.

## Tailwind pipeline

Tailwind v4 is CSS-first: `styles/input.css` is one `@import "tailwindcss";`. Build with:

```bash
make css-build     # minified, into crates/quack/static/css/output.css
make css-dev       # watch mode while editing templates
```
