# `quack-desktop`: Quack.app over `quack-core`

The design for the desktop interface the design doc calls `quack desktop` (section 11.6,
issue #35). Section 10 records the decisions taken on 2026-10-12 and what else was on the
table; section 11 lists what the implementation must confirm with the project owner. Where
this departs from the design doc (the subcommand, section 14's "not a separate binary") it
says so, and those sections are updated in a follow-up PR against #35.

The one-sentence version: `Quack.app` is a Tauri 2 application that shows the web UI,
served from `quack-server`'s router inside the app's own process through a custom URI
scheme, with the server's two streaming routes carried over Tauri's IPC, write prompts in a
native dialog, files added by path, and the `quack` command-line binary inside the bundle.
There is no HTTP listener, no socket, no token, and no cookie anywhere in it: the webview
asks the router directly, as a test does with `oneshot`, and the router calls the engine as
a library.

## 1. Shape

```
      +--------------------+   quack://localhost/...  (pages, forms, assets, uploads, API)
      |  Quack.app window  | ------------------------------------------------+
      |  (the web UI:      |   Tauri IPC: `stream` (query/stream, jobs/stream)|
      |   askama + htmx)   | <----------------------------------------+       |
      +--------------------+                                          |       |
                                                   +------------------v-------v------+
                                                   |  quack-desktop (Rust)            |
                                                   |  scheme handler, stream bridge,  |
                                                   |  permission dialog, drop, menu   |
                                                   +------------------+---------------+
                                                                      | Embedded::handle / stream
                                                   +------------------v---------------+
                                                   |  quack-server (Embedded: router, |
                                                   |  AppState in ServeMode::Local)   |
                                                   +------------------+---------------+
                                                                      | in-process calls
                                                   +------------------v---------------+
                                                   |  quack-core                      |
                                                   +----------------------------------+
```

- **One frontend.** The web UI (`crates/quack-server`'s askama templates, htmx, and
  `static/js/app.js`) is the app's UI. A page, a form, a static asset, an upload, and
  every API call the page makes go to `quack://localhost/...`, which the app's scheme
  handler answers by handing the request to `quack_server::Embedded` and returning the
  router's response whole. Nothing is rendered twice and no second UI exists.
- **The server, in-process, without a listener.** `Embedded` (`crates/quack-server/src/embedded.rs`)
  opens the control plane and builds the router once in `ServeMode::Local` (the one
  caller is the implicit owner, as `quack serve --local` treats loopback), and answers a
  request through tower's `oneshot`, with a loopback `ConnectInfo` so the limiter keys and
  the audit rows see what a loopback caller gives them. `handle` reads the body whole;
  `stream` returns the body as chunks for the two routes that never end on their own.
- **Two routes stream, so two go over IPC.** A webview's custom scheme answers a request
  whole (WebKit's `WKURLSchemeHandler` buffers; wry has no streaming responder), so
  `POST .../query/stream` and `GET .../jobs/stream` are read by the app's `stream`
  command, which parses the server-sent events as they arrive and sends each down a Tauri
  `Channel`; `app.js` opens them that way when `window.__TAURI__` exists and as `fetch` and
  `EventSource` in a browser. The events are the same names with the same JSON.
- **A separate binary, `quack-desktop`**, in `crates/quack-desktop`. Tauri wants to own the
  binary crate (`tauri-build` in its `build.rs`, `tauri.conf.json` beside it), and on Linux
  Tauri links GTK and WebKitGTK dynamically, which the `quack` binary's static musl builds and
  the `scratch` image cannot carry. The CLI and its release targets are unchanged.
- **macOS first.** The crate is Tauri, so Windows and Linux can follow from the same code;
  nothing in the first release is built or tested for them (7.4).

### What moved to make this possible

- `OpenedWorkspace` and session choice moved from the binary to `quack-cli` (#543).
- The audited background runs, the upload queue's spooling, and the pending-permission
  holder moved from `quack-server` handlers to `quack-core` (#545), so the server's
  handlers are thin and a second holder of a workspace could reuse them.
- `quack_core::forward` (#543): the socket a workspace's holder listens on so another
  process's reads run inside it (3.6), offered by the app, `quack serve`, and the terminal
  alike.
- `quack_server::Embedded`: the server opened and driven by a process, with no listener.

## 2. Crate layout

```
crates/quack-desktop/
  Cargo.toml          # [[bin]] quack-desktop; build = "build.rs"; depends on quack-core and quack-server
  build.rs            # tauri_build::build()
  tauri.conf.json     # identifier, the main window at quack://localhost/, withGlobalTauri, bundle
  capabilities/       # the main window's capability: the app's commands, dialog, opener
  icons/              # generated by `cargo tauri icon` from one SVG
  static/index.html   # a placeholder frontendDist; the window never loads it
  src/
    main.rs           # crypto provider once, logging, Config, tokio runtime, Embedded::open, the scheme handler, the window, quit
    bridge.rs         # the `stream` and `stop_stream` commands: a streaming route read event by event into a Channel; permission_required answered in the dialog
    permission.rs     # the native dialog built from Decision's words
    drop.rs           # files and folders by path (an OS drop, File > Add Files..., Add Folder...) uploaded through the router
    menu.rs           # the native menu bar
    log.rs            # tracing to stderr and a rolling file under the data dir
```

`gen/` (Tauri's generated capability schemas) is ignored by git.

### Dependencies (in the root `[workspace.dependencies]` menu, pinned, default features off)

| Crate | Version | What for |
|---|---|---|
| `tauri` | `=2.12.3` | `wry`, `image-png` |
| `tauri-build` | `=2.7.2` | (build-dependency) |
| `tauri-plugin-dialog` | `=2.8.2` | the write confirm dialog, the file and folder pickers, the messages |
| `tauri-plugin-opener` | `=2.7.1` | external links in the browser |
| `tauri-plugin-single-instance` | `=2.5.3` | not used yet |
| `tauri-plugin-updater` | `=2.13.3` | self-update (7.3), not used yet |
| `tauri-plugin-process` | `=2.4.0` | relaunch after an update, not used yet |
| `tracing-appender` | `=0.2.5` | the rolling log file |
| `futures`, `tokio-util` | the workspace's | reading the stream; the cancel token a Stop fires |

`tauri-cli` (`=2.12.2`) is a tool, installed with `cargo install tauri-cli --version
=2.12.2 --locked`. `cargo deny` covers the crate; Tauri's tree is MIT/Apache with MPL-2.0
leaves, all on the allow-list, with no `openssl`, `native-tls`, or `ring`.

### Lints

The crate declares `[lints] workspace = true` like every member. Tauri command handlers
are ordinary functions, so the panic, indexing, and `absolute_paths` denials apply. A
command takes `AppHandle` and reads its state from it rather than `State<'_, T>`, since
Tauri's macro adds a type check with `unreachable!` to an async command with a borrowed
argument, which the workspace denies.

## 3. The Rust side

### 3.1 State

Tauri manages three things: the `Embedded` server (the `AppState` `quack serve` would
have, with its `JobQueue`, its open workspaces, and its pending permissions), the
`Streams` the page has open (a cancel token per stream id, so Stop can end one), and the
log's writer guard. The server opens a workspace once per process and keeps it open while
anything uses it (the `OnceCell`-per-workspace rule: DuckDB's lock is per process, so a
second open inside the app would silently fork the database). While the app holds a
workspace, the `quack` command-line binary's reads run inside the app over a socket (3.6).

### 3.2 The scheme handler

`main.rs` registers the `quack` scheme with `register_asynchronous_uri_scheme_protocol`.
Each request from the webview is rebuilt with its path and query, headers, and body, handed
to `Embedded::handle`, and the response returned to the webview with the status and
headers the router set. Redirects, cookies the server sets (none in local mode), forms,
htmx fragments, uploads (the Documents page's multipart form), downloads (the SQL page's
CSV, exports), and the API behind `app.js` all go this way. The webview resolves
`/workspaces` against `quack://localhost/`, so every relative link in the templates works
unchanged.

### 3.3 The stream bridge

`bridge::stream(open: {id, method, url, body}, on_frame: Channel)` opens the route on
`Embedded::stream`, sends `open` with the status, parses the body's bytes as server-sent
events as they arrive (`SseParser`, split anywhere), sends each as `event {event, data}`,
then `close`; a refused route sends `error` with the API's own `error` field. `stop_stream(id)`
cancels the read, which drops the body and so the server's stream, cancelling a turn as a
closed connection does. In `app.js`, `openStream` picks the transport: this command under
Tauri, `fetch` with a reader for the chat and `EventSource` for the jobs in a browser; the
chat's Stop and the jobs' reconnect (3 seconds after a close) sit on it.

### 3.4 Jobs

The server's own job strip, toasts, and Jobs page work as they do in a browser: the
page follows `GET .../jobs/stream` through the bridge, and a job that starts or ends
refetches the strip through the scheme. The lanes are the server's (`[server].workers_per_workspace`
per workspace, a chat session serial).

### 3.5 Writes and the native confirm dialog

A `permission_required` event is answered in a native dialog, not in the page. The bridge
sees the event before the page does, and while the page's card shows the statement under
"Answer in the dialog", `permission::show` puts up an alert attached to the window, shown
and focused first if it was hidden:

- **Title:** `Decision::HEADING`: "This statement changes the workspace:"
- **Body:** the hold's `notice` sentence when the event carries one ("This turn read
  document text, which may have asked for this statement. Check that it is what you asked
  for."), a blank line, then the SQL, cut at 1,500 characters with "… and N more lines,
  shown in the chat".
- **Buttons:** the three `Decision`s in order with their `label`s: **Run it**, **Allow for
  this turn**, **Don't run it**. The words come from `Decision`, never from the app, so the
  terminal, the web, and the dialog say the same thing.

The chosen `Decision` goes to `POST .../sessions/{sid}/permissions/{request}` through the
router, exactly as the web chat's buttons send it, so the answer is audited as `permission`
and times out under `[server].permission_timeout_seconds` like any other; the bridge then
sends the page a `permission_answered` event with `Decision::reply` ("Ran it.", "Not run.",
"Ran it; the rest of this turn's writes run too.") or the API's refusal, and the card shows
it. Closing the dialog any other way is `Deny`. Dialogs show one at a time in event order.

The audit is the server's: every workspace-touching request writes its `audit_log` row and
`_quack_audit` detail as user `local` under the server's channels (`web`, `api`), since the
requests are the web UI's. Section 11 asks whether the app should name itself as channel
`desktop` instead.

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

The app already carries the server's MCP route (`/mcp/v1/{workspace}`) in `Embedded`'s
router; what this adds is a loopback listener for that one route under the per-launch
token. Not built yet.

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

One main window, title `quack`, 1200 by 800 at first, opened on `quack://localhost/`,
which the server redirects to the workspace list. The screens are the web UI's
(`docs/web-ui.md`): Workspaces, Chat, Documents, Tables, SQL, Search, Graph, Ontology,
Saved, Jobs, and the settings pages, each the same page a browser gets from `quack serve`.
Nothing is built twice; a page that improves in the web UI improves in the app.

Native menu bar: **Quack** (About, Hide, Quit), **File** (Add Files… ⌘O, Add Folder… ⌘⇧O,
Close Window), **Edit** (the standard items), **Window** (Minimize, Full Screen). The
items planned and not built: Settings…, Check for Updates…, Install quack Command…, New
Workspace…, New Session, Import…, Snapshot Workspace…, View, Help, the Dock badge and menu,
the menu-bar extra, and keeping the app running when the window closes.

Files and folders arrive by path: an OS drop anywhere on the window, Add Files… and Add
Folder… (the OS's own pickers). `drop.rs` reads each file (every supported file under a
folder, through `quack_core::ingestion::tree::Tree::walk`) and uploads it as the
Documents page's form does, `POST .../documents` with one `file` part through the router,
into the workspace the window shows (`/w/{id}/...`); with no workspace open a message says
to open one first. The upload is spooled, queued in the workspace's ingest lane, shown in the
job strip, and audited as any upload. Not yet: `FileData::Path` with `DocumentSource::Path`
(a folder's root recorded, unchanged files skipped on a re-run), and exports to a chosen
file instead of the page's download.

Quit (⌘Q, or the last window closing) calls `Embedded::shutdown` with
`[server].shutdown_grace_seconds`: the streams end, the jobs get the grace, and every
workspace closes so its writer checkpoints. The "N jobs are running" question is not built.

External links open in the browser through `tauri-plugin-opener`. The CSP is the server's
own (`default-src 'self'`, with `connect-src 'self'`), which on macOS covers Tauri's IPC
since the webview posts messages through `WKScriptMessageHandler`; Linux and Windows
would need `ipc: http://ipc.localhost` added (7.4).

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

- `make desktop` runs `cargo tauri dev` (`cargo install tauri-cli --version =2.12.2
  --locked`). There is no frontend build: the pages are the server's, and its CSS and
  editor bundles are committed as before.
- `cargo clippy --workspace` and `cargo test --workspace` compile the crate on macOS with no
  extra packages. On Linux the crate needs the WebKitGTK development packages even though
  the release does not build it there; `make lint` and `make test` check `pkg-config
  --exists webkit2gtk-4.1` and pass `--exclude quack-desktop` with a printed note when it
  fails. The CI lint and test jobs on Linux install the packages and never skip.
- A PR-time `desktop` job on `macos-latest` runs `cargo check -p quack-desktop --locked`
  and the crate's tests.
- `Dockerfile.build`, `docker-bake.hcl`, and the image are untouched.

### 7.2 Release

The macOS job in `.github/workflows/reusable-build.yml` already builds, signs, and
notarizes the `quack` binary on `macos-latest` for `aarch64-apple-darwin`. It gains, after
that:

1. Copy the signed `quack` to the sidecar path Tauri expects
   (`binaries/quack-aarch64-apple-darwin`).
2. `cargo tauri build --target aarch64-apple-darwin --ci`: Tauri builds `quack-desktop`,
   assembles `Quack.app` with the sidecar, signs it with the same identity
   (`bundle.macOS.signingIdentity` from `APPLE_CERTIFICATE_NAME`), notarizes and staples it
   (a bundle can be stapled, unlike the bare binary), and writes the `.dmg`, the updater's
   `Quack.app.tar.gz`, and its `.sig`.
3. Upload `quack-${VERSION}-macos-arm64.dmg`, the `.app.tar.gz`, and the `.sig` as
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
"install command" path. The crate has no `cfg(target_os)` code today; the Linux `tauri-plugin-dialog` feature
(`gtk3`) is the one platform line in its `Cargo.toml`. Beyond the bundle, the CSP needs
`connect-src ipc: http://ipc.localhost` on those platforms (section 4). None of it is built
or tested until then.

## 8. Tests

- `quack-cli`: `OpenedWorkspace` and session choice, moved with their behaviour pinned
  (default workspace created on first use, `-w` must exist, `--resume` by prefix).
- `quack-core`: the modules moved out of the server keep the server's tests.
  `quack_core::forward`: a holder and a second process over a temp workspace (a second
  `WorkspaceDb::open` fails on the lock, the forwarded read returns the holder's rows, a
  write request is refused, a wrong token is refused, a stale socket is replaced, and the
  socket is gone after close).
- `quack-server`: `Embedded` over a temp data directory serves the root as a redirect to
  the workspace list with no sign-in, and reads the job stream chunk by chunk until
  `shutdown` ends it.
- `quack-desktop`: the event parser across chunk boundaries (names, multi-line data,
  comments); a refused stream's message (the API's `error`, else the status); the
  workspace id read from a stream URL; the multipart part the uploads send; the added-paths
  summary; the dialog result to `Decision` mapping and the dialog body (the notice, the cut).
  Not yet: a turn through the bridge against `quack-testkit`'s `ScriptedOllama` under
  `tauri::test`'s mock runtime with the dialog stubbed, answered allow, deny, and
  allow-for-turn.
- A manual pre-release checklist in the README on a clean macOS account: first launch with
  and without Ollama, drop, the pickers, the dialog, exports, the menu, quit with jobs,
  install command, update and relaunch.
- `make eval` is untouched; it drives the CLI, which the app ships unchanged.

## 9. Cost

| Part | Size |
|---|---|
| `quack-cli`: `OpenedWorkspace` and session choice moved (#543) | ~150 lines moved |
| `quack-core`: background runs, upload spooling, pending permissions moved from the server (#545) | ~600 lines moved |
| `quack-core`: `quack_core::forward` (#543) | ~400 lines with tests |
| `quack-server`: `Embedded` | ~200 lines with tests |
| `quack-server`: `app.js` transport (`openStream`, the desktop card, the jobs reconnect) | ~90 lines |
| `quack-desktop`: scheme handler, bridge, dialog, drop, menu, logging | ~700 lines with tests |
| Still to build: the bundle with the CLI inside, the install command, the updater, the MCP listener, the rest of the menu, hide-to-Dock, the quit question | ~800 lines, plus the workflow |

## 10. Decisions taken (2026-10-12)

Taken by the project owner, not by this document.

| Decision | Chosen | Not chosen |
|---|---|---|
| Interface | the web UI, served in-process through a Tauri custom scheme with no listener, its two streams over IPC (section 1); "htmx for both", one frontend to maintain | a React frontend of the app's own over Tauri commands (built first, then removed); the web UI served from an embedded HTTP listener; a SwiftUI shell over a core binding |
| `quack-server` in the app | the router and `AppState` as `Embedded`, driven directly; its MCP route offered on a loopback listener later, opt-in per workspace (3.7) | a second set of commands over `quack-core` reshaping what the server already does |
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

Superseded: the React frontend with hand-typed `invoke` wrappers over a `utoipa` schema,
and the per-view command modules, both built in the first scaffold and removed when the
owner chose one frontend.

## 11. To confirm

Questions the implementation must not answer on its own:

1. **Audit channel.** The server audits the app's requests under its own channels (`web`,
   `api`) as user `local` (3.5). The design doc's schema names channel `desktop`. Confirm
   the app should pass itself as `desktop`, or that the server's channels stand.
2. **Homebrew.** A cask for `Quack.app` that links the CLI (section 6); whether the macOS
   tarball and the formula also remain for people who want only the CLI.
3. **Order of the rest** (section 9): the bundle and the install command, the updater, the
   MCP listener, the menu and Dock behaviour.

## 12. Where it sits

In `docs/architecture.md`'s layering diagram the `quack desktop` box becomes `Quack.app`,
a window over `quack-server`'s router, which it drives in-process. The note
under the diagram ("all subcommands of the single `quack` binary") gains "and `Quack.app`, a
second binary that ships the first inside it". `crates/README.md` gets an eighth row;
`docs/ci-cd.md` gets the macOS bundle steps, the update manifest, and the cask;
`docs/audit.md` already lists channel `desktop`; design doc 11.6 is rewritten to this, 14's
"not a separate binary" line goes, and the section 7 permission table's Desktop row reads
"a native confirm dialog with `Decision`'s words".
