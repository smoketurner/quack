# quack-desktop

The desktop app: a Tauri window showing the web UI, with `quack-server` inside the process
and no listener. The design is [`docs/desktop.md`](../../docs/desktop.md); this README says
what the crate holds today and where new code goes.

## What it holds

| File | Owns |
|---|---|
| `src/main.rs` | the crypto provider once, the configuration, logging, the tokio runtime Tauri runs on, `Embedded::open`, the `quack://` scheme handler that hands each request to `Embedded::handle`, the window, the File menu's pickers, and quit through `Embedded::shutdown` |
| `src/bridge.rs` | the `stream` and `stop_stream` commands: a streaming route (`query/stream`, `jobs/stream`) read from `Embedded::stream`, parsed into server-sent events, and sent down a `Channel`; a `permission_required` event answered in the dialog, the answer posted through the router, and `permission_answered` sent to the page |
| `src/permission.rs` | the native dialog built from `Decision`'s words |
| `src/drop.rs` | files and folders by path (an OS drop, Add Files…, Add Folder…), each uploaded through the router as the Documents form does, into the workspace the window shows |
| `src/menu.rs` | the native menu bar |
| `src/log.rs` | `tracing` to stderr and a rolling daily file under `{data_dir}/logs/` |
| `build.rs`, `tauri.conf.json`, `capabilities/`, `icons/`, `static/` | Tauri's build step, the window (opened on `quack://localhost/`, `withGlobalTauri` so `app.js` finds the IPC) and bundle settings, the main window's capability, the icons, and a placeholder `frontendDist` the window never loads |

The page side is `crates/quack-server/static/js/app.js`: `openStream` uses the `stream`
command when `window.__TAURI__` exists and `fetch` or `EventSource` otherwise, and the
chat's write prompt card shows "Answer in the dialog" instead of buttons.

## Building

- `cargo check --workspace --all-features` (with `--workspace`: a `-p quack-desktop` build
  resolves its own feature set, and `libduckdb-sys` compiles a second time, for an hour and
  six gigabytes). On Linux the crate links WebKitGTK: install `libwebkit2gtk-4.1-dev`
  first. Without it, `make lint` and `make test` leave the crate out and say so; CI
  installs it.
- `make desktop` runs `cargo tauri dev` (`cargo install tauri-cli --version =2.12.2 --locked`).
- `gen/`, which Tauri writes, is ignored by git.

## Where new code goes

A screen is a server page: change the template, the handler, or `app.js`, and both the
browser and the app get it. Only what needs the OS goes here: a dialog, a picker, the
menu, the scheme handler, a stream the webview cannot read itself.

## Not yet built

The release packaging (the `.dmg`, the CLI as a sidecar, signing and notarization), the
updater, the MCP listener, the install-command menu item, first-launch setup, the Dock
badge and menu-bar extra, hide-to-Dock on close, and the quit question while jobs run:
`docs/desktop.md` sections 3.7, 3.8, 4, 6, and 7.
