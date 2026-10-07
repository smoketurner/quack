# Logged event types

Every request that touches a workspace, and every change to users, workspaces, membership,
or tokens, writes one row to the access audit log in `control.db` (`audit_log`), including
requests that were denied. The row names who (user id, token hash), what (the action below,
a resource kind and opaque id, the workspace), from where (channel, client address, request
id), and the outcome: `allowed`, `denied` (the caller may not), or `error` (the caller may,
and the work failed). Rows are never updated or deleted. The content half of each row, what
was read or run, lives inside the workspace file (`_quack_audit`) under the same id and is
visible to the workspace's members only (design doc section 12).

`quack audit` and `GET /api/v1/admin/audit` read the access log; `--format ocsf` and
`?format=ocsf` render each row as an OCSF 1.9.0 event (`quack_core::ocsf`):
Authentication [3002] for sign-ins and sign-outs, API Activity [6003] for everything else,
with the activity below. `quack audit -w WORKSPACE --detail --format ocsf` and
`GET /api/v1/workspaces/{id}/audit?format=ocsf` join each row to its content half.

The table lists every action the code writes (`AuditAction` in
`crates/quack-core/src/storage/control.rs`); a test checks that no action is missing from
it. A row whose action a newer build wrote reads back under its own name and renders as an
API Activity event with that name.

| Action | Written when | Outcomes | OCSF activity |
|---|---|---|---|
| `login` | A password login or an identity-provider sign-in, on both routes | `allowed`, `denied` (wrong password, disabled or locked account, refused sign-in) | Authentication: Logon |
| `logout` | A session ends on request | `allowed` | Authentication: Logoff |
| `session` | A session cookie that has expired, or one the issuer no longer vouches for, is presented | `denied` | Authentication: Logon |
| `password` | A person changes their own password (`POST /auth/password`), proving the current one first | `allowed`, `denied` (the current password is wrong, or the account is disabled or locked) | API: Update |
| `admin` | A user is created or changed (`quack user`, the Users page, `/admin/users`) | `allowed` | API: Create |
| `workspace` | A workspace is created, renamed, or its settings change | `allowed` | API: Other |
| `member` | A membership is added, changed, or removed, by a person or by the identity provider's groups | `allowed`, `error` (removing a non-member) | API: Update |
| `break_glass` | An admin who is not a member of the workspace grants themself a role; the request must carry a reason, which the detail records with the role and `acting_as: admin` | `allowed` | API: Create, severity Medium |
| `token` | An API token is created, listed, or revoked; or a bearer is refused (unknown, expired, a rejected issuer token) | `allowed`, `denied` | API: Other; a denied row is Authentication: Logon |
| `open` | One resource is opened: a document, a table, a session, a passage, the workspace itself | `allowed`, `denied` | API: Read |
| `list` | Resources of a kind are listed (documents, tables, members, sessions, jobs) | `allowed`, `denied` | API: Read |
| `page` | A web console page is rendered | `allowed`, `denied` | API: Read |
| `show` | An ontology, its JSON Schema, or graph status is shown | `allowed` | API: Read |
| `stream` | A job or MCP event stream is opened | `allowed`, `denied` | API: Read |
| `query` | An agent turn, over REST, the web chat, or MCP (`-p` and the terminal are unaudited) | `allowed`, `denied` (the provider is not allowed for the workspace), `error` | API: Read; with the content half, the `ai_operation` profile |
| `search` | A document search without the agent (`POST .../search`, the web Search page, the MCP `search` tool); the detail holds the query and what it was limited to | `allowed`, `denied`, `error` | API: Read |
| `sql` | A statement run directly (`/sql`, the SQL page, the MCP `sql` tool) | `allowed`, `denied` (a write without the write scope, an internal table), `error` | API: Read |
| `ingest` | A file is uploaded, a document replaced, a bundle imported | `allowed`, `denied`, `error` | API: Create |
| `import` | A SQLite or URL import is started | `allowed`, `error` | API: Create |
| `export` | A workspace is exported as an OKF bundle, the graph as CSV, GraphML, or JSON-LD (the detail holds `format`, `nodes`, `edges`, `provenance`), or a statement's rows are streamed out | `allowed`, `denied` (a write statement), `error` | API: Read |
| `delete` | A document, session, saved question, workspace, or other resource is deleted | `allowed`, `denied`, `error` | API: Delete |
| `snapshot` | A workspace is written out as a snapshot (`quack workspace snapshot`, `GET .../snapshot`, the Settings page) | `allowed`, `error` | API: Update |
| `restore` | A snapshot is restored as a new workspace (`quack workspace restore`, `POST /workspaces/restore`) | `allowed` | API: Update |
| `context` | The workspace context is edited or restored | `allowed` | API: Update |
| `ontology` | The ontology is imported, edited, or restored to a version | `allowed`, `error` | API: Other |
| `propose` | An ontology proposal run starts or ends | `allowed`, `error` | API: Create |
| `graph` | The graph is searched or a path asked for | `allowed`, `error` | API: Read |
| `graph_extract` | A graph extraction run starts or ends | `allowed`, `error` | API: Create |
| `graph_review` | A graph review decision | `allowed` | API: Update |
| `graph_revalidate` | A revalidation drops what the ontology no longer allows | `allowed` | API: Update |
| `graph_merge` | A merge proposal is accepted or rejected | `allowed`, `error` | API: Update |
| `graph_edit` | A person adds, corrects, or deletes a node or edge | `allowed`, `error` | API: Update |
| `embeddings_refresh` | A vector refresh starts or ends | `allowed`, `error` | API: Update |
| `embeddings_status` | The vectors' status is read | `allowed` | API: Read |
| `session_read` | A session's transcript, or the workspace's audit detail, is read | `allowed`, `denied` | API: Read |
| `share` | A session is shared or unshared | `allowed`, `denied` | API: Update |
| `mode` | A session's mode changes | `allowed` | API: Update |
| `cancel` | A job is cancelled | `allowed`, `error` | API: Update |
| `permission` | A person answers a write the agent waits on, or the wait expires | `allowed`, `denied` | API: Update |
| `save` | An answer's SQL is saved as a question | `allowed`, `error` | API: Create |
| `saved_run` | A saved question runs without the model | `allowed`, `error` | API: Read |
| `table_note` | A table's note is set or removed (`PUT .../tables/note`, the Tables page) | `allowed`, `error` | API: Update |
| `retype` | A table column is given another type (`POST .../tables/retype`, the Tables page's Fix type) | `allowed`, `error` | API: Update |

Channels are `web` (a browser session or local mode), `api` (a bearer), `mcp`, `tui`,
`desktop`, and `cli` (an operator at the shell, whose rows carry no user). The client
address is the TCP peer, or behind `[server].trusted_proxies` the address the proxy
forwarded; CLI and terminal rows carry none.
