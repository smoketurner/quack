# quack-desktop

The desktop app: a Tauri window over `quack-core`, the fourth client of the engine beside
the terminal, print mode, and the server. The design is [`docs/desktop.md`](../../docs/desktop.md);
this README says what the crate holds today and where new code goes.

## What it holds

| File | Owns |
|---|---|
| `src/main.rs` | the crypto provider once, the configuration, logging, the tokio runtime Tauri runs on, the `tauri::Builder` with every command, and the quit path (`JobQueue::shutdown`, then every workspace closed so its writer checkpoints) |
| `src/app.rs` | `AppState`: the `Config`, the `ControlPlane`, the one `JobQueue`, the workspaces held open (once per process, each with its `SharedDb`, `ReaderDb`, audit connection, and the `quack_core::forward` socket other processes read through), each session's running turn, and the per-workspace "Run changes without asking" toggle |
| `src/workspace.rs` | commands: list, create, rename, delete, open, close |
| `src/chat.rs` | commands: sessions, a new session, the transcript, the mode, `start_turn` (the turn as a job in the session's serial lane, its events over a `Channel<TurnEvent>`), `cancel_turn` |
| `src/permission.rs` | a `PermissionRequired` event answered in the native dialog built from `Decision`'s words; one dialog at a time |
| `src/documents.rs` | commands: one page of the listing, files and folders added by path (one ingest job each), delete |
| `src/jobs.rs` | the job queue's broadcast as the Tauri event `job`; list and cancel |
| `src/audit.rs` | `audit_log` rows on channel `desktop` as user `local`, plus the `_quack_audit` detail under the same id |
| `src/settings.rs` | the Settings view's data: `quack config`'s inspection, the paths, the models |
| `src/menu.rs` | the native menu bar; chosen items reach the frontend as the event `menu` |
| `src/log.rs` | `tracing` to stderr and a rolling daily file under `{data_dir}/logs/` |
| `src/schema.rs` | the JSON Schema of every type a command returns or an event carries, written by `quack-desktop --schema` |
| `ui/` | the frontend: React 19 and TypeScript on Vite, Tailwind, `@tauri-apps/api`; `src/commands.ts` is the typed client, `src/transcript/reducer.ts` folds a turn's events into the transcript, `src/views/` holds the views |
| `build.rs`, `tauri.conf.json`, `capabilities/`, `icons/` | Tauri's build step (a placeholder page when `ui/dist` is absent), the window and bundle settings, the main window's capability, the icons |

The first release's views are Workspaces, Chat, Documents, and Settings; the other seven
(Tables, SQL, Search, Graph, Ontology, Saved, Jobs) follow, each a command module and a
view at a time, as `docs/desktop.md` section 4 lists them.

## The frontend's types

The frontend's types come from Rust. `make ui-types` writes the `utoipa` JSON Schema of
every command's types (the same derives the REST API's OpenAPI document uses) to
`ui/src/schema.json` (the schema test, run with `QUACK_WRITE_SCHEMA` set, writes it; the
binary prints the same with `quack-desktop --schema`), and `openapi-typescript` turns it into
`ui/src/schema.d.ts`; `src/commands.ts` types each `invoke` with them. Both generated files
are committed, and a Rust test fails when `schema.json` no longer matches the commands.
`tauri-specta`, which the design named, is still a release candidate for Tauri 2; this is
its "types alone, wrappers by hand" alternative, with no new derive on any core type.

## Building

- `make ui-install`, then `make ui-lint` (`tsc`, ESLint, Prettier), `make ui-test` (Vitest),
  and `make ui-build` (into `ui/dist`, which is not committed; `build.rs` stands a
  placeholder page in when it is absent, so `cargo check` needs no Node).
- `cargo check --workspace --all-features` (with `--workspace`: a `-p quack-desktop` build
  resolves its own feature set, and `libduckdb-sys` compiles a second time, for an hour and
  six gigabytes). On Linux the crate links WebKitGTK: install `libwebkit2gtk-4.1-dev` first. Without it, `make lint` and `make test` leave the
  crate out and say so; CI installs it.
- `make desktop` runs `cargo tauri dev` (`cargo install tauri-cli --version =2.12.2 --locked`).

## Where new code goes

Behaviour more than one interface needs goes in `quack-core` first (the architecture
rule), then a command here calls it. A command takes the ids the frontend holds and returns
the engine's own types, so the frontend reshapes nothing. Anything that owns the screen
(a view, a dialog, the menu) goes here or in `ui/`.

## Not yet built

The release packaging (the `.dmg`, the CLI as a sidecar, signing and notarization), the
updater, the MCP endpoint, the install-command menu item, first-launch setup, the Dock
badge and menu-bar extra, and the remaining seven views: `docs/desktop.md` sections 3.7,
3.8, 4, 6, and 7.
