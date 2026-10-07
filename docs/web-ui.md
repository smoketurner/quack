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

Every page struct carries a `page: Page`: title, its `Tab`, username, admin flag, local flag,
and the current workspace with its role and what the caller may do. `base.html` reads it for
the header and the workspace tabs (`Tab::WORKSPACE`: chat, documents, search, tables with
the import form, SQL, context, ontology, graph, jobs, settings), highlighting the page's tab, or the
Users, Audit, or About link, with `aria-current="page"`. Users and Audit show to admins;
About (`/about`: the running version and the projects quack is built with) shows to every
signed-in user and reads no workspace, so it writes no audit row.

- **`workspaces.html`** lists each workspace with its classification, the caller's role,
  when it was created, and when a request last touched it (`ControlDb::workspace_times`:
  the newest `audit_log` row naming it, allowed or denied; "never used" for a workspace
  the CLI made and nothing has opened through the server), one column each.
- **`documents.html`** lists what the workspace holds now; "Show replaced documents"
  (`?all=true`) adds each superseded document with a link to the one that took its place. A
  ready row's Replace control (`POST /w/{id}/documents/{doc}/replace`, one file) queues the
  file through the same `documents::enqueue` as the upload form with the row as its
  predecessor; the row reads "being replaced by …" until the new document is ready. The
  htmx actions on a row (pin, unpin, delete) swap in the live listing. The upload form takes
  files, not a folder; `quack ingest DIR` is the folder path (design doc section 6.1).
- **`passage.html`** (`/w/{id}/documents/{doc}/chunks/{n}`) is where a citation link lands:
  one chunk of a document with its position and total, page and heading, when the document
  was ingested, the chunk's full text, and links to the chunks before and after. It reads
  through the same `documents::read_chunks` as `GET .../documents/{doc}/chunks`, so each
  visit is audited as opening the document; a position the document does not have is a 404
  page. The chunk is shown whatever the document's status, so a citation in an old answer
  still opens after its document was replaced.
- **`search.html`** (`/w/{id}/search`, `server::web::search`) runs one document search
  without the model. The form posts its query in the body (search text is workspace
  content, and a URL ends up in logs) with a multi-select of the newest 200 ready
  documents, the mode, and an optional graph entity. It goes through the same
  `Access::search` as `POST .../search`, audited as `search`, and shows each hit's fused
  score, its vector, keyword, and rerank rank and score, and a link to its passage page,
  then both legs' candidates in collapsed sections, the phrase note, and the rerank
  outcome. A search the core refuses (an unknown document, vector mode with no embedding
  model) is shown on the page, not as a failed page. The chat form offers the same
  documents in a multi-select; picked ones go out as the turn's `document_ids`.
- **`ontology.html`** shows the class tree, relations, properties, and mappings; the JSON
  editor; the version list with the diff to the previous version; the propose form; the
  Rename form, which gives a class or relation a new id and moves the graph's nodes and
  edges with it; the paged review queue with bulk accept and reject; and links to download
  the ontology's JSON and its JSON Schema.
- **`graph.html`** shows status banners (provisional, stale, missing mapped tables,
  chunks and tables the graph has not caught up with, drift), the search and path forms,
  the ECharts result with a node inspector (each entry carries an edit form and a delete
  button for people who may write; each edge a delete button), a "Correct the graph"
  section with add-node and add-edge forms, the merge queue, the extract, revalidate,
  and review buttons, and a Download form (a `GET` to `.../graph/export` with the format and
  whether to include provisional rows). The stale banner lists what a
  revalidation would drop (totals, per class id, per relation id). Its button names those
  totals and posts them, and the server drops only when they still match what it counts.
  When the preview cannot be counted, the banner says why and offers no drop.
- **Dark only.** `styles/input.css` sets `color-scheme: dark`, so native controls and the
  file picker follow; panels are `slate-900` on a `slate-950` page and primary buttons are
  `blue-600`. Charts and the graph use ECharts' built-in `dark` theme.
- **Tables never let columns touch or overrun.** Listing tables carry `data-table` (padded
  cells, from `input.css`); prose and names wrap with `wrap-anywhere`; short fixed values
  (status, sizes, times) are `whitespace-nowrap`; data grids (sample rows, SQL results) and
  opaque ids keep one line per cell, cut at a width with "…", and carry the full value in a
  `title` tooltip. Every grid track is `minmax(0,1fr)` (`grid-cols-1` below `md`), so a
  wide grid or a long name scrolls or truncates inside its own box instead of widening the
  page, on a phone too; the Documents and Jobs tables scroll sideways in their own box.
- **Times are localized in the browser; no page prints a raw timestamp.** Every stored time
  goes through `web::When` (`Clock` or `Relative`), which renders `<time datetime="…Z"
  data-when="clock|relative">` with UTC text from `web::Moment` (DuckDB and SQLite timestamp
  text, or RFC 3339), and falls back to the escaped text when a value is not a time. `app.js`
  rewrites each in the viewer's zone, with the full local date and time as the tooltip.
  `clock` (chat messages, versions, the audit log, token expiry) is the time of day, with the
  date when not today; `relative` (jobs, documents, sessions, token last use) is written by the
  server as "5 min ago", "3 h ago", or the date, and the page leaves it as written, so nothing
  changes under the reader. While a document processes, the Documents page polls
  `.../documents/status` and swaps in only each row's status and the note (`hx-swap-oob`),
  never the whole table.
  Chat messages show when they were asked or answered, and an answer how many milliseconds it
  took (`duration_ms` on the response object and the assistant message's metadata). A SQL
  result shows its row count and the statement's own run time in milliseconds
  (`SqlOutcome::duration_ms`, also in the `POST .../sql` body), timed on the connection's
  thread so a wait for the writer is not counted.
- **SQL results sort by rewriting the statement.** A header is a button that posts the
  statement that ran back with `sort` (the 1-based column) and `dir`. When the statement is
  one `SELECT`-shaped query (`WorkspaceDb::sortable`, from `json_serialize_sql`),
  `WorkspaceDb::sort_statement` sets its own top-level `ORDER BY` in `DuckDB`'s parse tree,
  replacing any it had and placing it before a `LIMIT`, and `json_deserialize_sql` prints it
  back; the column is named when that name is unique and resolves, and given by position
  otherwise. Nulls sort last both ways. The rewritten SQL runs, and the response swaps it
  into the editor (`hx-swap-oob`), so what ran is what the person sees and keeps editing.
  Without a click, rows come back in the statement's own order. Anything else (a write,
  several statements) runs as typed with plain headers. "Download CSV" is a `POST` of the
  same statement: SQL never travels in a URL, where request logs and proxies would keep it.
  The file holds the rows the grid holds, at most `[analysis].max_query_rows`. When that cap
  cut the result, the button reads "Download CSV (first 250 of 1000 rows)" and the file is
  named `query-first-250-of-1000.csv`; a complete result downloads as `query.csv`. The file
  itself carries no marker row, which would break a CSV parser.
- **Accessibility.** Every page starts with a "Skip to content" link to `<main id="main">`;
  the workspace tabs and the admin links are named `<nav>` landmarks, and the current tab or
  chat session carries `aria-current="page"`. A control with only a placeholder has an
  `aria-label`; table headers carry `scope="col"`, and a table with no header row has an
  `aria-label`. Errors are `role="alert"`, notices and "refreshes itself" lines
  `role="status"`, and the SQL result is `aria-live="polite"`. The conversation is a polite
  `role="log"`; a streaming answer stays `aria-busy` until it completes, so a screen reader
  reads it once. Charts and graphs are `role="img"` with a label naming what they show.
- **Fragments that htmx swaps** (`documents_rows.html`, `sql_result.html`) are their own
  structs, rendered to a string and inserted with `|safe`. askama escapes everything else.

A form handler calls the same typed operation as the REST handler (a method on `Access` or
`Identity` in `server::api`), so both apply the same validation and audit rows. The API
wraps the result in JSON; the web handler in a `web::flash::Flash` redirect. A redirect's
outcome (an error in red, a notice in green, so a started background pass or a revalidation
count does not look like a failure), and for the Tables page the table to open, can name
workspace content, so it never goes in the URL: `web::flash::keep` stashes it in this
process's memory (`Flashes`, one minute) under a random id, and the browser carries only the
id in an `HttpOnly` `quack_flash` cookie; the landing page's `Flashed` extractor takes it, so
it shows once. The graph page's searches and the Tables page's choice of table are posted
forms for the same reason. The documents list polls its rows fragment only while a
row is processing (marked `data-pending`).

## Assets

- `static/css/output.css` is built from `styles/input.css` with `make css-build`
  (Tailwind v4.3.3 standalone, which scans `templates/` and `src/` for class names). It is
  **committed**: rust-embed needs it at compile time and CI has no Tailwind. Rebuild and
  commit it whenever a template changes classes.
- `static/js/htmx.min.js` (htmx 4.0.0) and `static/js/echarts.min.js` (ECharts 6.1.0) are
  vendored so the binary works air-gapped.
- `static/js/redoc.standalone.js` (Redoc 2.5.4, MIT, the npm package's
  `bundles/redoc.standalone.js`, with its `redoc.standalone.js.LICENSE.txt` beside it) is
  vendored the same way. It renders the API reference at `/api/v1/docs`
  (`templates/api_docs.html`, a page of its own outside `base.html`) from
  `/api/v1/openapi.json`. Redoc was chosen over Swagger UI (Apache-2.0, a 1.6 MB bundle plus
  a stylesheet) and Scalar (MIT, 4.4 MB) for its size, 1.1 MB in one file, and because it
  is read-only: the page sends no requests to the API on the reader's behalf. Neither the
  page nor the document needs sign-in, like the static assets: the document describes
  routes and reveals no workspace content. The Settings page's API tokens section and the
  About page link to it. To update it, download the new version's two files from the npm
  package and replace both.
- `.pre-commit-config.yaml` keeps prek's whitespace and file-size hooks off the vendored
  scripts and `output.css`, so each stays byte for byte as built or downloaded.
- `static/js/sql-editor.min.js` is the SQL page's editor: CodeMirror 6 with
  `@codemirror/lang-sql`, highlighting SQL and completing the table and column names that
  `GET /api/v1/workspaces/{id}/tables/schema` returns (audited, `no-store`, never a
  `_quack_` table), each inserted as SQL writes it. Its source is `editor/sql-editor.js`
  with pinned versions in `editor/package.json` and `pnpm-lock.yaml`; `make editor-build`
  bundles it with esbuild, and the bundle is **committed** like the CSS. Completion follows
  the terminal's rules over CodeMirror's SQL syntax tree: table names alone after `FROM`,
  `JOIN`, `DESCRIBE`, `SUMMARIZE`, or a comma in a `FROM` list; after `t.`, that table's or
  alias's columns (lang-sql's own source); in an expression, the columns of every table the
  statement names (before or after the cursor) first, then table names, then keywords and
  functions, offered before a letter is typed where an expression must follow (after
  `SELECT`, `WHERE`, `AND`, a comma, `=`); nothing at an alias. The popup opens as you type;
  Ctrl+Space opens it anywhere. The editor is built at once, at the textarea's size and on
  its background, and the schema plugs in when it arrives, so the box does not flash. It
  hides the page's `textarea` and keeps it in sync, so the htmx run, the sort swap, and the CSV
  download post it as before, and without JavaScript the textarea is the editor.
  Ctrl/Cmd+Enter runs the statement.
- `static/js/app.js` is quack's own. It posts to `/api/v1/workspaces/{id}/query/stream`,
  parses the SSE events, and renders the steps block (a `run_sql` or `create_chart` step's
  kept rows as a collapsed grid with an Export full result form that posts the statement to
  the SQL page's download), the answer, citations as links to the
  passage each one cites, and the chart spec as an ECharts option with the toolbox's
  save-as-image and read-only data view, `stack` on bar and line series when the spec is
  stacked, and a Download CSV link built from the spec as a data URL. A `permission_required` event
  becomes a card with the statement, Run it, Don't run it, and Allow for this turn, posted to
  `.../sessions/{sid}/permissions/{request}`, and the time the turn stops waiting. When the event carries a `notice`
  (its `reason` is `read_documents`: the turn read document text, so the write is asked for even
  under "Run changes without asking"), the card shows that sentence; the text lives in
  `policy::Hold::notice` only. On page load it renders stored
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
