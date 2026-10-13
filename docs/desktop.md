# `quack-desktop`: Quack.app over `quack-core`

**Set aside on 2026-10-12.** This design was built as far as a window showing the web UI
through a Tauri custom scheme with `quack-server` in-process (PR #546, closed unmerged,
the branch kept as the record) and stopped there. What it cost: a webview's custom scheme
answers a request whole, so the two streaming routes needed an IPC bridge, downloads would
load whole into memory, WebKit does not follow a redirect from a custom scheme so the
handler followed `Location` itself and the window's URL drifted from the page, and every
web UI change would have needed a desktop check. What it would have given over `quack
serve --local` in a browser: a Dock icon, native pickers and drop, a native write dialog, a
menu bar, and one install carrying the CLI. The owner judged that not worth the upkeep.
If a desktop app is wanted later, the cheap shape is a window over a loopback listener
(`quack serve --local` on `127.0.0.1:0` with a per-launch token), where the web UI runs
unchanged. What stays from the work: `OpenedWorkspace` and session choice in `quack-cli`,
the CLI's reads through a held workspace (`quack_core::forward`, #543), and the background
runs, upload spooling, and held permissions in `quack-core` (#545). The rest of this
document is the design as written before the attempt.

The design for the desktop interface the design doc calls `quack desktop` (section 11.6,
issue #35). Nothing in it is built. Section 10 records the decisions taken on
2026-10-12 and what else was on the table; section 11 lists what the implementation must
confirm with the project owner before it starts. Where this departs from the design doc
(the subcommand, the embedded server, section 14's "not a separate binary") it says so, and
those sections are updated in a follow-up PR against #35.

The one-sentence version: `Quack.app` is a Tauri 2 application whose Rust side is an
interface crate like `quack-terminal`, calling `quack-core` in-process (the same
`TurnRequest`, `JobQueue`, `Writer`, and storage calls the terminal makes), whose frontend
is built for the desktop and bundled in the app, and which ships for macOS first with the
`quack` command-line binary inside the bundle. There is no HTTP server, no listening socket,
no token, and no cookie anywhere in it: the frontend talks to Rust over Tauri's IPC, and
Rust talks to the engine as a library.

## 1. Shape

```
      +--------------------+   Tauri IPC (commands, channels, events)
      |  Quack.app UI      | <------------------------------------------+
      |  (TypeScript over  |                                            |
      |   @tauri-apps/api) |                                            v
      +--------------------+                              +---------------------------+
                                                          |  quack-desktop (Rust)     |
                                                          |  commands, turn bridge,   |
                                                          |  jobs bridge, permission  |
                                                          |  dialogs, menu, updater   |
                                                          +-------------+-------------+
                                                                        | in-process calls
                                                          +-------------v-------------+
                                                          |  quack-core (+ quack-cli) |
                                                          +---------------------------+
```

- **A separate binary, `quack-desktop`**, in `crates/quack-desktop`. Tauri wants to own the
  binary crate (`tauri-build` in its `build.rs`, `tauri.conf.json` beside it), and on Linux
  Tauri links GTK and WebKitGTK dynamically, which the `quack` binary's static musl builds and
  the `scratch` image cannot carry. The CLI and its release targets are unchanged.
- **Depends on `quack-core` and `quack-cli`, and on one module of `quack-server`.** It is
  the fourth client of the engine beside the terminal, print mode, and the server, and it
  reaches the engine the way the terminal does. The one thing it takes from `quack-server` is
  its MCP server, served in-process and opt-in so Claude Code and editors can reach a
  workspace the app holds (3.7); no REST route, page, session, or `AppState` comes with it.
  `crates/README.md`'s rule holds: behaviour more than one interface needs goes in
  `quack-core`; what owns the screen goes here.
- **The frontend is new**, written for a desktop window, and bundled into the app
  (`frontendDist`). The web UI's askama pages and htmx are not reused; the two UIs share the
  engine, its types, and its words (`Decision`, `PageCounts::note`, citation labels), not
  markup.
- **macOS first.** The crate is Tauri, so Windows and Linux can follow from the same code;
  nothing in the first release is built or tested for them, and every `cfg(target_os)` in the
  crate is listed in section 7.

### What moves to make this possible

- `OpenedWorkspace` (today private in `crates/quack/src/main.rs`: load `Config`, open the
  `ControlPlane`, `workspace_or_default`, enter the `Egress` scope, open the `WorkspaceDb`,
  spawn the `Writer`, open the `ReaderDb`) moves to `quack-cli`, so the binary, the terminal's
  setup, and the app open a workspace through one function.
- Session choice (`resolve_session`: new, latest, or by prefix, with the mode) moves beside
  it.
- Anything the app needs that today lives only in a `quack-server` handler moves to
  `quack-core` first, as the architecture rule says. The known cases are listed in 3.4.
- `quack_core::forward` is new in `quack-core`: the socket a workspace's holder listens on so
  another process's reads run inside it (3.6), offered by the app, `quack serve`, and the
  terminal alike.

## 2. Crate layout

```
crates/quack-desktop/
  Cargo.toml          # [[bin]] quack-desktop; build = "build.rs"
  build.rs            # tauri_build::build()
  tauri.conf.json     # identifier, window defaults, frontendDist = ui/dist, externalBin (the CLI), updater, bundle
  capabilities/       # the app window's capability: the commands below, dialog, opener, process
  icons/              # generated by `cargo tauri icon` from one SVG
  README.md           # what it holds, the manual release checklist
  ui/                 # the frontend: package.json, pnpm-lock.yaml, vite.config.ts, src/ (bindings.ts generated, views/, transcript/, grid/), dist/
  src/
    main.rs           # crypto provider once, logging, Config, tokio runtime, tauri::Builder with every command
    app.rs            # AppState: config, control plane, open workspaces, job queue, pending permissions
    workspace.rs      # commands: list, create, rename, delete, snapshot, restore, open, close
    chat.rs           # commands: sessions, start_turn (a Channel of TurnEvent), cancel_turn, documents scope, mode
    permission.rs     # PermissionRequired -> the native dialog -> the oneshot; "allow for this turn"
    documents.rs      # commands: list (paged), add paths, replace, fields, chunks (passage view), stored image
    tables.rs         # commands: list, describe, note, retype, import, saved imports
    sql.rs            # commands: run (read as typed), stream export to a chosen file, complete
    search.rs         # commands: DocumentSearch with explain
    graph.rs          # commands: search, path, status, extract, revalidate (preview then drop), merges, edits, export
    ontology.rs       # commands: show, versions, diff, import/export, propose, review, accept/reject, rename, edit
    saved.rs          # commands: list, add from a session's last answer, run, show, remove
    jobs.rs           # JobQueue broadcast -> Tauri event "job"; cancel; the counts for the badge
    audit.rs          # audit_log rows on channel `desktop` plus _quack_audit detail (section 3.5)
    mcp.rs            # opt-in: quack-server's McpServer over streamable HTTP on loopback, per-launch token (3.7)
    menu.rs           # the native menu bar, the Dock menu, the menu-bar extra
    cli.rs            # "Install quack command": symlink Contents/MacOS/quack into /usr/local/bin
    update.rs         # tauri-plugin-updater: check on launch and on demand, offer, install, relaunch
    setup.rs          # first launch: quack_core::setup's discovery without a terminal, the config write
    log.rs            # tracing to stderr and a rolling file under the data dir
```

### Dependencies (added to the root `[workspace.dependencies]` menu, pinned, default features off)

| Crate | Version (crates.io, 2026-10-10) | Features the crate opts into |
|---|---|---|
| `tauri` | `=2.12.3` | `wry`, `tray-icon`, `image-png`, `macos-private-api` only if the menu-bar extra needs it |
| `tauri-build` | `=2.7.2` | (build-dependency) |
| `tauri-plugin-dialog` | `=2.8.2` | the write confirm dialog, file and folder pickers, save dialogs, the quit and update prompts |
| `tauri-plugin-single-instance` | `=2.5.3` | |
| `tauri-plugin-opener` | `=2.7.1` | external links in the browser, "Reveal in Finder" |
| `tauri-plugin-updater` | `=2.13.3` | self-update (7.3) |
| `tauri-plugin-process` | `=2.4.0` | relaunch after an update |
| `tauri-specta` | `=2.0.0-rc.25` (the Tauri 2 line has no stable release yet; `1.0.2` is Tauri 1) | `derive`, `typescript`: the typed frontend client |
| `specta` | `=2.0.0-rc.25` | `derive`; `Type` on every struct a command returns, beside `Serialize` |
| `tracing-appender` | `=0.2.5` | the rolling log file |

`tauri-cli` (`=2.12.2`) is a tool, installed with `cargo install tauri-cli --version
=2.12.2 --locked` in `make setup` and the workflow. Every crate the code names by a full
path goes on `.clippy.toml`'s `absolute-paths-allowed-crates`. `cargo deny` covers the
crate; Tauri's tree is MIT/Apache with MPL-2.0 leaves, all on the allow-list, with no
`openssl`, `native-tls`, or `ring`; the updater plugin's `rustls-tls` feature rides the
workspace's rustls pin on aws-lc-rs, and `cargo tree -i ring` is part of the gate when it
lands (`docs/crypto.md`).

### The frontend's stack

React 19 with TypeScript, on Vite, over `@tauri-apps/api` and the plugins'
`@tauri-apps/plugin-*` packages. The same Tailwind the web UI uses for styling, so the two
UIs share a look; TanStack Table and Virtual for the SQL and Tables grids; Radix primitives
for menus, dialogs, and popovers and cmdk for the ⌘K switcher; a sanitizing Markdown renderer
for the transcript; ECharts for charts and the graph canvas (the library the web UI vendors);
CodeMirror 6 for the SQL editor (the packages `crates/quack-server/editor` pins), each
attached to a `ref`. pnpm with a committed lockfile, as the editor already does.

The frontend's types come from Rust. `tauri-specta` generates a typed TypeScript client
from every `#[tauri::command]` and event (the way `utoipa` generates the OpenAPI document
from every handler), so `AgentResponseBody`, `ToolStep`, `ChartSpec`, `JobInfo`, and the
rest have one definition; `make ui-types` regenerates `ui/src/bindings.ts` and a test fails
when it is stale, as the OpenAPI tests do. The Tauri 2 line of `tauri-specta` is a release
candidate at the time of writing (see the dependency table); if that is not acceptable when
the crate lands, `ts-rs` generates the types alone and the `invoke` wrappers are written by
hand.

### Lints

The crate declares `[lints] workspace = true` like every member. Tauri command handlers
are ordinary functions, so the panic, indexing, and `absolute_paths` denials apply.
`tauri::generate_context!()` and `tauri::generate_handler!()` are macros from an allowed
crate. The frontend gets `tsc --strict`, ESLint with the React and hooks rules, and
Prettier in `ui/`, run by `make ui-lint` and the CI job (7.1).

## 3. The Rust side

### 3.1 State

One `AppState` for the process, managed by Tauri (`app.manage`): the `Config`, the
`ControlPlane`, the `JobQueue` (one per process, as the terminal and the server each have),
the open workspaces, and the pending permissions. (Since then this became
`quack_core::engine::Engine`, which every interface builds and shares.)

An open workspace is what `OpenedWorkspace::shared` produced then (today the engine's
`WorkspaceHandle`): the `SharedDb` (the
`Writer` actor on its own thread) and the `ReaderDb` pool, plus the `Egress` scope for its
`allowed_providers`, entered on every command that reaches a model. The app opens a workspace
once per process and keeps it open while any window shows it (the `OnceCell`-per-workspace
rule the server follows: DuckDB's lock is per process, so a second open inside the app
would silently fork the database). While the app holds a workspace, the `quack` command-line
binary's reads run inside the app over a socket (3.6).

### 3.2 Commands

Every frontend action is a `#[tauri::command]` in the module named in section 2, taking the
ids the frontend holds and returning the engine's own types serialized with serde: the same
`AgentResponseBody`, `DocumentPage`, `TableDescriptionBody`, `GraphResult`, `JobInfo`,
`ChartSpec`, `SavedQuestion`, and `Ontology` the REST API returns, so the frontend's types
are generated from one source and nothing is reshaped on the way. Long work (ingestion,
import, extraction, the ontology document pass, embeddings refresh, classification) is
submitted to the `JobQueue` with the lanes the terminal uses, and the command returns the
`JobInfo`; progress reaches the frontend over the jobs event (3.4). A command that fails
returns the core error's message and its `ErrorCode`-equivalent kind, so the frontend can
tell "no chat model" from "not found".

### 3.3 A turn

`start_turn(workspace, session, message, documents, channel)` builds a `TurnRequest` with
the workspace's `SharedDb` and `ReaderDb`, the session, the write policy (3.5), the person's
document scope, `user: None` (as on the command line), the app's `LabelJobs`, a fresh
`EventSink`, and a `CancellationToken` kept under the session so `cancel_turn` can fire it.
The turn runs as a job in the session's serial lane, exactly as the terminal submits one, so
a follow-up waits for the answer before it.

The events reach the frontend over a Tauri `Channel<TurnEvent>`, where `TurnEvent` is a
serializable projection of `AgentEvent`: `status`, `reasoning`, `text`, `tool_started`,
`tool_finished` (the `ToolStep`), `permission_required` (`request`, `sql`, `reason`,
`notice`, `heading`, `choices`, as the SSE event carries them), `complete`
(`AgentResponseBody`), `error` (`TurnFailure`). The names are the SSE stream's names on
purpose, so the two interfaces' event vocabularies are one and a reader of
`docs/web-ui.md` recognizes them. `PermissionRequired` carries a `oneshot` and is not
serializable; `permission.rs` takes it before the projection and holds it (3.5).

### 3.4 Jobs

`JobQueue::subscribe` is forwarded as a Tauri event `job` with the `JobInfo`, for every
workspace the app has open; the frontend's job strip, the Dock badge (running jobs), and
the finished-job notification all read it. `cancel_job(id)` is `JobQueue::cancel`. The
workspace lanes are the terminal's: a chat session is a serial lane, uploads, graph
extraction, and the document pass have their own.

Work that existed only as a `quack-server` handler and that the app also needs is in
`quack-core`, with the server and the app as its two callers: the audited background
runs' bookkeeping (`jobs::runs`: embeddings refresh, graph and ontology document passes,
an import refresh, and a classification as jobs that audit their start and end under one
run id, through whatever `RunAudit` the caller is: the server's `Access`, the app's
auditor of 3.5), the upload queue's spooling rules (`ingestion::upload`: a file's bytes on
disk until its job ends, the per-workspace lane width from `[server].workers_per_workspace`,
the graph follow-up as a closure the caller supplies), and the pending-permission holder
(`analysis::permissions`: `hold` under a timeout with an expiry hook to audit, `decide` by
the `Asker` whose turn asked; the app's dialog answers through it like the web chat does,
while the terminal keeps its on-screen prompt). The server's `run.rs` and `permissions.rs`
are the thin ends: `Access` as the `RunAudit`, the expiry audited as a denied `permission`,
and the follow-up after an ingest in the workspace's graph lane behind its extraction slot.

### 3.5 Writes and the native confirm dialog

The write policy is `WritePolicy::Ask` by default and `Allow(Approver::Person)` while the
person has "Run changes without asking" on (a per-workspace toggle in the chat view, off
again at launch), the same two settings the terminal has. Writes typed in the SQL view run
as typed, as everywhere.

A `PermissionRequired` event is answered in a native dialog, not in the page. `permission.rs`
takes the `PermissionRequest`, keeps its `oneshot` under a request id, tells the frontend
(so the transcript shows the statement under "Answer in the dialog"), and shows an NSAlert
attached to the window, shown and focused first if it was hidden:

- **Title:** `Decision::HEADING`: "This statement changes the workspace:"
- **Body:** the `Hold::notice` sentence when there is one ("This turn read document text,
  which may have asked for this statement. Check that it is what you asked for."), a blank
  line, then the SQL, cut at 1,500 characters with "… and N more lines, shown in the chat".
- **Buttons:** the three `Decision`s in order with their `label`s: **Run it**, **Allow for
  this turn**, **Don't run it**. The words come from `Decision`, never from the app, so the
  terminal, the web, and the dialog say the same thing.

The chosen `Decision` goes into the `oneshot`; the `Delivery` it returns says whether the
turn still waited, and the frontend renders `Decision::reply` ("Ran it.", "Not run.",
"Ran it; the rest of this turn's writes run too.") or "the turn stopped waiting". Closing
the dialog any other way is `Deny`. Dialogs queue one at a time in event order, like the
terminal's prompts. No HTTP, no request id on the wire, no timeout of the server's kind: the
turn's own cancel token ends the wait when a turn is cancelled.

The audit: every workspace-touching command writes an `audit_log` row through
`ControlPlane::record_audit` with `Origin::from(Channel::Desktop)` and user `local`, plus
the `_quack_audit` detail row inside the workspace under the same id, as the design doc's
section 12 asks of every interface and as its schema's `channel` column (`desktop`) has
anticipated; `audit.rs` is a small copy of the server's `Access::audit` shape without the
authorization half. Section 11 notes that the terminal and the CLI write no rows today.

### 3.6 The app and the command line on one data directory

The app uses the CLI's data directory (`~/.local/share/quack`, `QUACK_DATA_DIR`, config under
`~/.config/quack`), so `Quack.app`, `quack -p`, and `quack mcp` see one set of workspaces and
one keychain item.

DuckDB allows one process with read-write access to a file, or any number of read-only
processes, never a writer beside readers: a read-only open fails while another process holds
the file read-write, since DuckDB has no cross-process MVCC. The app holds its workspaces
read-write, so a second `quack` process cannot open one even for `SELECT 1`, as it cannot
today against `quack serve`. There is one lock and one DuckDB instance per workspace, and
the way through it is to run the second process's reads inside the holder:

- **The holder listens.** Whichever process has a workspace open (`Quack.app`, `quack
  serve`, the terminal) also listens on a Unix socket in the workspace directory,
  `{workspace_dir}/quack.sock`, with a per-launch token in a 0600 file beside it; both are
  removed when the workspace closes, and a stale pair from a crash is detected by a failed
  connect and replaced. Neither side trusts the files or the peer on faith: before using the
  socket or the token file a client checks that each is the right kind of file, owned by the
  current user, and unreadable by anyone else, and both ends check the peer's credentials
  (`SO_PEERCRED`, `getpeereid`) and refuse a process running as another user; the holder
  never removes a stale socket that is not its own user's. This is a `quack-core` module, `quack_core::forward`, since every holder
  offers it: framed JSON requests over the socket, a handful of request kinds (a read
  statement with its rows streamed back, a document search, a table or document listing, a
  description, a graph query), each run on the holder's reader pool in a read-only
  transaction as the server's `App::read` runs them, under the agent's row cap and timeout.
- **The CLI follows the lock.** When `WorkspaceDb::open` hits the lock conflict, a
  read-only command checks for the socket and runs there instead: `quack -q` with a read
  statement, `search`, `tables`, `docs`, `saved run` (its statements are reads by
  construction), `graph search` and `path`, `graph status`, `ontology show`, `sessions`,
  `config`. The output is identical, and one line on stderr says "read through Quack.app"
  (the holder names itself in the token file). Without the socket the error is today's.
- **Writes stay refused.** `quack ingest`, `-q` with a write, `graph extract`, `ontology
  accept`, and every other write fail as they do now, naming the holder: "the workspace is
  open in Quack.app; add the file there, or quit it". A write forwarded blind would run under
  the holder's policy with nobody to ask and no one to audit it as, which is what the design
  keeps out. A `-p` question is a write too (its session is recorded), so it is refused the
  same way with a pointer to the app's chat or its Connect Claude Code endpoint (3.7).
- **The app side is one line.** The app enters `forward::Listener::bind` for each workspace
  it opens and leaves it when the workspace closes; the holder's end has no settings.

This also removes today's friction between `quack serve` and the CLI on one laptop, which
the design doc accepts as a limit, since the server is a holder like any other.

### 3.7 MCP for other tools, opt-in

DuckDB's lock is per process, so while the app holds a workspace, `quack mcp` started by
Claude Code or an editor cannot open it. Its read tools follow the lock like the CLI's (3.6):
`sql` with a read statement, `search`, `list_tables`, `describe_table`, and `list_documents`
run inside the app over the socket. `query` and `classify` cannot, because an agent turn
records its session and a labelling writes a table, so on stdio they are refused with a
pointer to the endpoint below. The server answers the same need today by serving MCP over
HTTP from the process that holds the lock, and the app does the same, as a setting that is
off by default.

**Settings › Connect Claude Code.** Switching it on, per workspace:

1. Builds `quack-server`'s `McpServer` from the app's own handles: `McpSetup` takes the
   `Config`, the workspace's `SharedDb` and `ReaderDb`, its `WorkspaceRow`, a
   `WritePolicy` (`Deny` unless the person also turns on "Allow writes over MCP", which is
   `quack mcp --allow-write`'s policy), `user_id: None`, the app's auditor (3.5, channel
   `mcp`), and the app's `JobQueue`. Nothing is opened twice; the MCP tools run on the same
   writer thread as the window's commands.
2. Serves it over rmcp's streamable HTTP transport on `127.0.0.1:0`, under a per-launch
   bearer token (32 random bytes from aws-lc-rs, compared in constant time) that every
   request must carry; a request without it is 401 and nothing else listens on the port. No
   REST, no pages, no sessions: a router of one route.
3. Shows the URL and the one JSON snippet to paste into Claude Code's MCP configuration,
   with a Copy button, and a line naming the policy in force ("reads only" or "writes
   allowed, unasked").

Turning it off stops the listener and ends its MCP sessions; Quit does the same through the
app's drain. The token changes on every launch, so the snippet is a one-time paste into a
config that reads it from a file the app writes (`{data_dir}/mcp/{workspace}.json`, mode
0600) rather than a literal, and the Settings view says so. Writes over MCP that a turn's
retrieved text dictated are still refused (`Approver::Nobody`), as over `quack mcp`.

For this the `McpServer` and `McpSetup` types become `pub` in `quack-server`, and
`mcp_http.rs`'s transport construction is factored so the app and `quack serve` build the
same one. The app links `rmcp`'s server transport through `quack-server` and adds `axum`
for the one-route router; the templates, static assets, and the rest of the server are not
compiled in, since `quack-server`'s page and asset embedding sits behind modules the app's
build does not reach. If that proves not to hold (rust-embed runs at compile time for the
whole crate), the MCP server moves to its own crate, `quack-mcp`, with `quack-server` and the
app as its two dependents; that is a refactor of module boundaries, not of behaviour.

The CLI's reads need none of this: they go through the socket (3.6). The endpoint exists for
what a read cannot be, an agent turn or a labelling run, asked for by another process.

### 3.8 First launch

With no `config.toml`, `setup.rs` runs `quack_core::setup::Discovery::run` over
`Environment::from_env` (Ollama, API keys, AWS credentials), builds a `SetupPlan` from what
it found, applies it with `SetupPlan::apply` through the same `toml_edit` path `quack init`
uses, only when `doctor`'s checks pass the result, and opens the window on the workspace
list. When nothing is found, the window opens on Settings with `doctor`'s findings and what
to install or set. `setup` already separates discovery, the plan, and the write from
`init_cli`'s prompts, so no core change is needed for this.

## 4. The window and the screens

One main window, title `quack` or the workspace's name, bounds remembered in
`{data_dir}/desktop.json`. A sidebar on the left, the view on the right, a job strip along
the bottom while jobs run. Keyboard shortcuts follow macOS: ⌘N new session, ⌘O add files,
⌘⇧O add folder, ⌘K workspace switcher, ⌘, settings, ⌘W close window, ⌘Q quit.

| View | Shows | Core it calls |
|---|---|---|
| Workspaces | the list with document and table counts; New, Rename, Delete (confirm names the workspace), Snapshot to a file, Restore from one | `ControlPlane` workspaces, `storage::backup` |
| Chat | sessions (search, rename, resume), the transcript with streamed text, each step's SQL and rows, citations opening the passage view, the chart, the document picker for scope, the mode (chat or query), "Run changes without asking" | `TurnRequest`, `storage::sessions`, `analysis::citations`, `ChartSpec` |
| Documents | one page of `DocumentListing` with filter and "show all"; drop zone; Add files, Add folder; Replace on a row; tags, author, date; the passage view with previous and next; a stored image | `WorkspaceDb::documents`, `ingest_file`, `register_document`, `document_chunks`, `stored_image` |
| Tables | row counts, profile warnings with the retype that fixes each, notes, Import (SQLite, URL, S3) and saved imports with Refresh | `TableProfile`, `describe_table`, `profile::Retype`, `quack_core::import` |
| SQL | the CodeMirror editor with completion, results as a virtualized grid, Export every row to a chosen file | `WorkspaceDb::execute_query`, `stream_query`, `quack_core::completion` |
| Search | query, documents, keyword or vector or hybrid, each hit's ranks | `analysis::search::DocumentSearch`, `explain_search` |
| Graph | search by entity or class, paths, status (stale, provisional, drift, pending), extract, revalidate with the preview's counts before the drop, merges, add and edit forms, export | `graph::query`, `graph::store`, `Revalidation`, `graph::export` |
| Ontology | the current version, versions and diff, import and export JSON, propose from tables or documents (cost shown, then asked), review, accept and reject, rename, the property and measure forms | `quack_core::ontology`, `induction`, `documents`, `edit` |
| Saved | the saved questions, Run (changed or not), Add from a session's last answer, Remove | `quack_core::saved` |
| Jobs | every job with progress, status line, outcome, Cancel | `JobQueue` |
| Settings | the models and providers with `doctor`'s probe results, the data directory, Install quack command, Connect Claude Code (3.7), Check for updates, the log | `Config`, `quack_core::setup`, `doctor`, `quack_server::McpServer` |

Native menu bar: **Quack** (About, Settings…, Check for Updates…, Install quack Command…,
Quit), **File** (New Workspace…, New Session, Add Files…, Add Folder…, Import…, Snapshot
Workspace…, Close Window), **Edit** (the standard items), **View** (one item per view),
**Window**, **Help** (documentation in the browser, the log folder). The Dock icon's badge
counts running jobs and shows a dot while a write waits for an answer; the Dock menu lists
the open workspaces. Closing the window keeps the app running, as macOS apps do; the Dock
icon reopens it; a menu-bar extra shows the job count and offers Open and Quit. Quit with
jobs running asks "N jobs are running. Quit and cancel them?" (**Quit** / **Keep running**),
then `JobQueue::shutdown(grace)` and the workspace handles dropped so each writer
checkpoints, then exit.

Files and folders arrive by path: an OS drop on the Documents view or the sidebar's
workspace, Add Files… and Add Folder… (NSOpenPanel), both into `ingest_file` with
`FileData::Path` and `DocumentSource::Path`, so nothing is read into the frontend and a
folder gets `quack ingest DIR`'s behaviour (root recorded, unchanged files skipped). Exports
(SQL rows, graph, OKF, snapshot) stream to a file the person chose in NSSavePanel; nothing
is held in memory as a download.

External links open in the browser through `tauri-plugin-opener`. The frontend loads only
from the bundle (`tauri://localhost`); its CSP allows the bundle and Tauri's IPC and nothing
else.

## 5. Keychain, data directory, logging

- **Keychain:** nothing new. The vault key and OAuth tokens go through
  `KeySource::Keychain`, the macOS Keychain, named by service, so the app and the CLI share
  the item; a signed bundle has a stable code identity, so the Keychain stops asking on
  every launch.
- **Data directory:** the CLI's (3.6).
- **Logging:** `tracing` to stderr and a rolling daily file `{data_dir}/logs/desktop.log`
  (`tracing-appender`, 7 kept), in `[server].log_format`'s text or JSON. A failure before
  the window exists (control plane will not open, data directory unwritable) is a native
  error dialog with the `anyhow` chain and the log path, then exit 1. The crate is `anyhow`,
  like the other interface crates.

## 6. The bundle: Quack.app with the CLI inside

The macOS release package is one `.dmg` holding `Quack.app`; the `quack` command-line binary
is inside the bundle at `Contents/MacOS/quack` as a Tauri sidecar (`bundle.externalBin`),
signed with the app and notarized with it. **Quack › Install quack Command…** creates the
symlink `/usr/local/bin/quack` to the binary inside the app (asking for administrator rights
through the standard prompt when the directory is not writable), and tells the person to
reopen their terminal, as VS Code's and Docker Desktop's install commands do; the menu item
reads **quack Command Installed** with a Remove option when the link exists and points at
this app. An app update updates the CLI with it, and the symlink stays valid as long as the
app stays in `/Applications`; moving the app breaks the link, and the menu item says so and
offers to repair it.

The Homebrew tap today installs the CLI from the macOS tarball with a formula. With the CLI
inside the app, the tap offers a cask that installs `Quack.app` and links the binary
(`binary "Quack.app/Contents/MacOS/quack"`), so `brew install --cask smoketurner/tap/quack`
gives both. Whether the tarball and formula also stay is in section 11.

## 7. Build, CI, and release

### 7.1 Contributors and CI

- `make setup` installs `tauri-cli` and pnpm; `make ui-build` builds the frontend into
  `ui/dist`; `make ui-lint` runs `tsc`, ESLint, and Prettier; `make desktop` runs `cargo
  tauri dev`.
- `cargo clippy --workspace` and `cargo test --workspace` compile the crate on macOS with no
  extra packages. On Linux the crate needs the WebKitGTK development packages even though
  the release does not build it there; `make lint` and `make test` check `pkg-config
  --exists webkit2gtk-4.1` and pass `--exclude quack-desktop` with a printed note when it
  fails. The CI lint and test jobs on Linux install the packages and never skip.
- A PR-time `desktop` job on `macos-latest` builds the frontend, runs `make ui-lint`, runs
  `cargo check -p quack-desktop --locked`, and runs the crate's tests.
- `Dockerfile.build`, `docker-bake.hcl`, and the image are untouched.

### 7.2 Release

The macOS job in `.github/workflows/reusable-build.yml` already builds, signs, and
notarizes the `quack` binary on `macos-latest` for `aarch64-apple-darwin`. It gains, after
that:

1. Build the frontend (`pnpm install --frozen-lockfile && pnpm run build` in `ui/`).
2. Copy the signed `quack` to the sidecar path Tauri expects
   (`binaries/quack-aarch64-apple-darwin`).
3. `cargo tauri build --target aarch64-apple-darwin --ci`: Tauri builds `quack-desktop`,
   assembles `Quack.app` with the sidecar, signs it with the same identity
   (`bundle.macOS.signingIdentity` from `APPLE_CERTIFICATE_NAME`), notarizes and staples it
   (a bundle can be stapled, unlike the bare binary), and writes the `.dmg`, the updater's
   `Quack.app.tar.gz`, and its `.sig`.
4. Upload `quack-${VERSION}-macos-arm64.dmg`, the `.app.tar.gz`, and the `.sig` as
   artifacts, attested with `actions/attest-build-provenance` like the tarballs, and listed
   in `SHA256SUMS` by `publish`.

Unsigned with a warning when the secrets are absent, as today. Reproducibility is not
claimed for the `.dmg`; the binaries inside are built `--locked` from the same commit, and
the attestation covers the image.

### 7.3 Self-update

`tauri-plugin-updater`, in the first release:

- **Key.** One minisign key pair made once with `cargo tauri signer generate`; the public
  key in `tauri.conf.json` (`plugins.updater.pubkey`), the private key and its password as
  two repository secrets the macOS job reads (`TAURI_SIGNING_PRIVATE_KEY`,
  `TAURI_SIGNING_PRIVATE_KEY_PASSWORD`). Losing the key means installed apps cannot verify
  the next release and must be reinstalled, so it is backed up like the Apple certificate.
- **Manifest.** The `publish` job, which already downloads every artifact and writes
  `SHA256SUMS`, also assembles `latest.json` (version, notes, date, the macOS `url` and
  `signature`) and uploads it with the release. The endpoint in `tauri.conf.json` is
  `https://github.com/smoketurner/quack/releases/latest/download/latest.json`. The workspace
  version (`2026.10.7`) parses as semver, which the plugin requires.
- **In the app.** `update.rs` checks once after the window shows and on **Check for
  Updates…**. A found update is a native dialog, "quack 2026.11.1 is available. Install it
  and relaunch?" with **Install and relaunch** / **Later** and the release notes' first lines.
  Install downloads over rustls, verifies the signature, replaces the app (and so the CLI
  inside it), quits through the same drain as ⌘Q (jobs running get the same question), and
  relaunches with `tauri-plugin-process`. A failed check is one log line, never a dialog.
- **Channel.** Latest only.

### 7.4 Platforms after macOS

The crate is Tauri, so Windows and Linux follow from the same code when wanted: a matrix row
each in `reusable-build.yml`, WebView2's bootstrapper in the Windows installer, WebKitGTK
packages as `.deb`/`.rpm` dependencies, the CLI as a sidecar in each bundle, and a per-platform
"install command" path. The `cfg(target_os = "macos")` code in the first release is the
menu-bar extra, the Dock badge and menu, `cli.rs`'s symlink path, and the NSAlert details;
everything else is Tauri's job. None of it is built or tested until then.

## 8. Tests

- `quack-cli`: `OpenedWorkspace` and session choice, moved with their behaviour pinned
  (default workspace created on first use, `-w` must exist, `--resume` by prefix).
- `quack-core`: the modules moved out of the server (3.4) keep the server's tests and gain
  the app as a second caller in tests. `quack_core::forward`: a holder and a second process in
  one test binary over a temp workspace (a second `WorkspaceDb::open` fails on the lock, the
  forwarded read returns the holder's rows, a write request is refused, a wrong token is
  refused, a stale socket is replaced, and the socket is gone after close); `quack -q`,
  `search`, and `tables` against a held workspace in the binary's tests, with the stderr
  line; `quack ingest` against one, refused with the holder's name.
- `quack-desktop`, under `tauri::test`'s mock runtime with the dialogs stubbed: the
  `AgentEvent` to `TurnEvent` projection (every variant, the permission event's fields, the
  words from `Decision`); a scripted turn through `start_turn` against `quack-testkit`'s
  `ScriptedOllama`, with a write that asks, answered allow, deny, and allow-for-turn, each
  recorded in the session and audited on channel `desktop`; a turn dictated by retrieved
  text (`seed_dictating_note`) asks even under "run without asking"; `cancel_turn`;
  `ingest` by path yields a `ready` document with `DocumentSource::Path`; the jobs event
  sequence for one upload; the install-command symlink logic against a temp directory; the state file round
  trip; `latest.json` reading against a fixture; the MCP endpoint (3.7) answers
  `initialize` and `list_tables` with the token and 401 without it, refuses a write under the
  default policy, and stops listening when switched off.
- `ui/`: Vitest for the event reducer that turns `TurnEvent`s into transcript state, with
  recorded event sequences as fixtures, and React Testing Library for the views that render
  them (a step with its SQL and rows, a citation, a chart, the permission notice); a test
  that `bindings.ts` matches what `tauri-specta` generates from the current commands.
- The release workflow's `latest.json` assembly runs against a fixture of `.sig` files in
  CI.
- A manual pre-release checklist in the README on a clean macOS account: first launch with
  and without Ollama, drop, the pickers, the dialog, exports, the menu, quit with jobs,
  install command, update and relaunch.
- `make eval` is untouched; it drives the CLI, which the app ships unchanged.

## 9. Cost

| Part | Size |
|---|---|
| `quack-cli`: `OpenedWorkspace` and session choice moved | ~150 lines moved |
| `quack-core`: background runs, upload spooling, pending permissions moved from the server | ~600 lines moved, two callers each |
| `quack-core`: `quack_core::forward`, the holder's socket and the CLI's fallback through it | ~400 lines with tests, plus the listen call in three holders |
| `quack-desktop` Rust: state, 11 command modules, turn and jobs bridges, permission dialog, audit, the MCP endpoint, menu, install command, updater, setup, logging | ~2,200 lines |
| `quack-server`: `McpServer` and `McpSetup` made public, the HTTP transport factored for two callers | ~100 lines changed |
| `ui/`: 11 views, the transcript renderer with steps, citations, charts, the grid, the editor, the graph canvas, forms | ~6,000 lines of TypeScript and CSS |
| Workflow, `latest.json` assembly, the cask | ~200 lines |
| Docs: this file, `architecture.md`, `ci-cd.md`, `crates/README.md`, the crate README, design doc 11.6, 14, and the section 7 table | |

The frontend is the bulk, and it is where a thin first release is possible without changing
the architecture: Workspaces, Chat, Documents, and Settings first (what a person needs to
ask a question of their files), the other seven views after, each a command module and a
view at a time.

## 10. Decisions taken (2026-10-12)

Taken by the project owner, not by this document.

| Decision | Chosen | Not chosen |
|---|---|---|
| Interface | a new desktop frontend bundled in the app, over Rust commands that call `quack-core` in-process (section 1) | the web UI served to a webview from an embedded HTTP server; the web UI served through a Tauri custom protocol without a socket; a SwiftUI shell over a core binding |
| `quack-server` in the app | not for the app's own UI; its MCP server alone, in-process, opt-in per workspace, so Claude Code can reach a workspace the app holds (3.7) | embedding the whole server and pointing the window at it; no MCP while the app is open |
| Frontend stack | React 19 with TypeScript on Vite, typed bindings from `tauri-specta` (section 2) | Svelte 5; SolidJS; no framework over `@tauri-apps/api`; Leptos or Dioxus in Rust |
| Shape | a separate `quack-desktop` binary | `quack desktop` inside the one binary; a subcommand exec'ing a sibling |
| macOS package | one `.dmg` with `Quack.app`, the `quack` CLI inside it as a sidecar, and an Install quack Command menu item (section 6) | a `.pkg` installing both; a `.dmg` with the app and the binary side by side |
| Platforms | macOS only in the first release, Windows and Linux later from the same crate (7.4) | Windows in the same release; all three now |
| Data directory | the CLI's (3.6) | the platform app-data directory the design doc named |
| The CLI against a held workspace | reads run inside the holder over a Unix socket, writes refused naming the holder (3.6) | the app closing idle workspaces so the CLI can take the lock; today's error alone |
| Write prompts | a native confirm dialog built from `Decision`'s words (3.5) | an in-page card |
| macOS hardware | Apple silicon only, matching the CLI | a universal `.app`; two bundles |
| Self-update | `tauri-plugin-updater` in the first release (7.3) | releases page and the tap only |
| Window on close | the app keeps running in the Dock with a menu-bar extra; Quit drains jobs (section 4) | quit on close |
| Tracking | #35 reopened; a follow-up PR updates design doc 11.6, 14, and the section 7 table | a new issue |

Superseded by the architecture change: the earlier decision to give the page a scoped IPC
capability over an embedded server. The frontend is the app's own, and its commands are
simply its API.

## 11. To confirm before building

Questions the implementation must not answer on its own:

1. **Audit.** The design doc's schema names channel `desktop` and section 12 asks every
   interface to write rows, and this document does so (3.5). The terminal and the CLI write
   none today. Confirm the app audits, or that it matches the terminal.
2. **The built frontend.** The repo commits built assets (the CSS, the editor bundle)
   because rust-embed needs them and CI has no Node. For the app, `ui/dist` can be committed
   the same way, or built by the macOS job with pnpm pinned and never committed.
3. **Homebrew.** A cask for `Quack.app` that links the CLI (section 6); whether the macOS
   tarball and the formula also remain for people who want only the CLI.
4. **The thin first release** (section 9): four views first, or all eleven before the first
   tag.
5. **`tauri-specta`'s release candidate.** Pin `=2.0.0-rc.25`, or use `ts-rs` for the types
   and hand-written `invoke` wrappers until it goes stable (section 2).

## 12. Where it sits

In `docs/architecture.md`'s layering diagram the `quack desktop` box becomes `Quack.app`,
a client of `quack-core` beside the terminal and the server, not of the server. The note
under the diagram ("all subcommands of the single `quack` binary") gains "and `Quack.app`, a
second binary that ships the first inside it". `crates/README.md` gets an eighth row;
`docs/ci-cd.md` gets the macOS bundle steps, the update manifest, and the cask;
`docs/audit.md` already lists channel `desktop`; design doc 11.6 is rewritten to this, 14's
"not a separate binary" line goes, and the section 7 permission table's Desktop row reads
"a native confirm dialog with `Decision`'s words".
