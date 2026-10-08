# Changelog

Every release, newest first, from the commit history. The upgrade steps a release
needs (config keys, schema moves, `quack embeddings refresh`) are in
[docs/upgrading.md](docs/upgrading.md).

## Unreleased

### Features

- **graph:** Report what the graph lags, follow ingests, and take a person's edits ([#440](https://github.com/smoketurner/quack/pull/440))
- **documents:** Open cited passages, replace changed files, and ingest folders ([#438](https://github.com/smoketurner/quack/pull/438))
- **saved:** Save an answer's SQL and re-run it without the model ([#435](https://github.com/smoketurner/quack/pull/435))

### Fixes

- **ingestion:** Stream workbook cells instead of building a dense grid ([#439](https://github.com/smoketurner/quack/pull/439))
- Close nine defects in ingestion, storage, the agent gate, and the server ([#434](https://github.com/smoketurner/quack/pull/434))

### Refactoring

- Merge nine duplicated types into the ones that already existed ([#437](https://github.com/smoketurner/quack/pull/437))
- Move inline tests of files over 1,000 lines beside their modules ([#436](https://github.com/smoketurner/quack/pull/436))

### Documentation

- Upgrade notes, contributor setup and recipes, review documents, rustdoc gate

### Build and CI

- **cliff:** End the changelog with exactly one newline
- **cliff:** End the changelog with a newline
## v2026.10.3 (2026-10-05)

### Features

- **terminal:** Select transcript text with the mouse and copy it ([#362](https://github.com/smoketurner/quack/pull/362))
- Honor proxy variables on every outbound client ([#360](https://github.com/smoketurner/quack/pull/360))

### Fixes

- **terminal:** Hash the chart plot in the message fingerprint ([#377](https://github.com/smoketurner/quack/pull/377))
- **terminal:** Refuse a line of files cut short by a # word ([#374](https://github.com/smoketurner/quack/pull/374))
- **agent:** Leave a rejected model call's text out of the answer ([#376](https://github.com/smoketurner/quack/pull/376))
- **agent:** Answer when the model refuses background_effort ([#375](https://github.com/smoketurner/quack/pull/375))
- **terminal:** Load files dropped with CR or CRLF separators at once ([#370](https://github.com/smoketurner/quack/pull/370))
- **terminal:** Make pie rendering scale linearly with slice count ([#369](https://github.com/smoketurner/quack/pull/369))

### Documentation

- **detail:** Record the 2026-10-05 batch in the volume log ([#378](https://github.com/smoketurner/quack/pull/378))
## v2026.10.2 (2026-10-05)

### Features

- **terminal:** Ratatui widgets, Markdown tables, and job and session lists ([#359](https://github.com/smoketurner/quack/pull/359))
- **terminal:** Load files dropped on the terminal at once ([#358](https://github.com/smoketurner/quack/pull/358))

### Fixes

- **ingestion:** Scrub absolute file paths from CSV/TSV reader errors ([#351](https://github.com/smoketurner/quack/pull/351))
- **server:** Refuse a write answer the turn no longer waits for ([#355](https://github.com/smoketurner/quack/pull/355))
- **sessions:** Re-summarize stale history after a budget increase ([#350](https://github.com/smoketurner/quack/pull/350))
- **sql:** Keep resolving tables in a comma FROM list after an unresolved name ([#349](https://github.com/smoketurner/quack/pull/349))
- **retrieval:** Make the model reranker honor background_effort ([#352](https://github.com/smoketurner/quack/pull/352))

### Build and CI

- Update Rust to 1.99.0 and dependencies ([#356](https://github.com/smoketurner/quack/pull/356))
- Bump zgosalvez/github-actions-ensure-sha-pinned-actions ([#353](https://github.com/smoketurner/quack/pull/353))

### Other

- Add the detail-triage skill for Detail bug batches ([#343](https://github.com/smoketurner/quack/pull/343))
## v2026.10.1 (2026-10-01)

### Features

- **web:** Workspace times, the duck in the header, select not tick ([#342](https://github.com/smoketurner/quack/pull/342))
- **server:** Let a person approve a write a streamed turn waits on ([#336](https://github.com/smoketurner/quack/pull/336))
- **llm:** Hold extraction and rerank answers to a JSON schema ([#334](https://github.com/smoketurner/quack/pull/334))
- **sessions:** Load history through rig's conversation memory ([#333](https://github.com/smoketurner/quack/pull/333))
- **sql:** Complete table and column names in the terminal and web UI ([#331](https://github.com/smoketurner/quack/pull/331))
- **retrieval:** Rerank with a dedicated rerank model ([#332](https://github.com/smoketurner/quack/pull/332))
- **doctor:** List provider models through the rig client a turn uses ([#330](https://github.com/smoketurner/quack/pull/330))
- **web:** Add an About page crediting the projects quack is built with ([#328](https://github.com/smoketurner/quack/pull/328))
- **web:** Dark UI, accessible and leak-free pages, turn timing, and rig 0.43 ([#319](https://github.com/smoketurner/quack/pull/319))

### Fixes

- **web:** Steady documents table, status-only polling, page help, earlier SQL columns, quieter request logs ([#341](https://github.com/smoketurner/quack/pull/341))
- **web:** Complete columns and only tables after FROM in the SQL editor ([#338](https://github.com/smoketurner/quack/pull/338))
- **agent:** Retry invalid tool calls and empty replies instead of ending the turn ([#329](https://github.com/smoketurner/quack/pull/329))

### Refactoring

- Put this round's free helpers on named types ([#337](https://github.com/smoketurner/quack/pull/337))
- **agent:** Hand per-turn state to tools through rig's ToolContext ([#335](https://github.com/smoketurner/quack/pull/335))

### Build and CI

- **lints:** Deny own-crate absolute paths and add hygiene lints ([#317](https://github.com/smoketurner/quack/pull/317))
## v2026.9.8 (2026-09-30)

### Features

- **oauth:** Show the quack duck on the browser login page ([#316](https://github.com/smoketurner/quack/pull/316))

### Fixes

- **doctor:** Surface the GPT-5.6 / Chat Completions tool-call refusal ([#315](https://github.com/smoketurner/quack/pull/315))
- Anthropic OAuth probe header and data directory mode ([#313](https://github.com/smoketurner/quack/pull/313))
## v2026.9.7 (2026-09-29)

### Fixes

- **llm:** Send temperature only to Ollama and let providers set effort ([#310](https://github.com/smoketurner/quack/pull/310))
## v2026.9.6 (2026-09-29)

### Fixes

- **llm:** Send Anthropic OAuth tokens as Authorization: Bearer ([#309](https://github.com/smoketurner/quack/pull/309))

### Documentation

- Split provider authentication into docs/providers.md ([#306](https://github.com/smoketurner/quack/pull/306))
## v2026.9.5 (2026-09-28)

### Fixes

- **build:** Pin office_oxide to 0.1.11 ([#305](https://github.com/smoketurner/quack/pull/305))
## v2026.9.4 (2026-09-28)

### Features

- **llm:** Send custom HTTP headers to model providers ([#304](https://github.com/smoketurner/quack/pull/304))
## v2026.9.3 (2026-09-28)

### Features

- **auth:** Register quack's client, signing the person in first at Vouch ([#285](https://github.com/smoketurner/quack/pull/285))
- **auth:** Two-step client key rotation, and a Vouch setup that works today ([#279](https://github.com/smoketurner/quack/pull/279))
- **oauth:** Private_key_jwt client auth and PAR for a secretless on-behalf-of setup ([#277](https://github.com/smoketurner/quack/pull/277))
- **oauth:** Act for each signed-in person at a provider (on-behalf-of) ([#220](https://github.com/smoketurner/quack/pull/220))
- **server:** Accept the issuer's access tokens and publish RFC 9728 metadata ([#218](https://github.com/smoketurner/quack/pull/218))
- **oauth:** Keep provider tokens in control.db, sealed by the vault ([#215](https://github.com/smoketurner/quack/pull/215))
- **llm:** Add Amazon Bedrock (bedrock and bedrock-mantle, VPC endpoints) with the AWS SDK credential chain ([#214](https://github.com/smoketurner/quack/pull/214))
- **server:** Sign in through the organization's OpenID Connect issuer ([#213](https://github.com/smoketurner/quack/pull/213))
- **oauth:** Add the client-credentials grant for providers ([#212](https://github.com/smoketurner/quack/pull/212))
- **audit:** Read the access log as OCSF 1.9.0 events ([#206](https://github.com/smoketurner/quack/pull/206))
- **audit:** Page through the whole access log with a cursor ([#205](https://github.com/smoketurner/quack/pull/205))
- Give core enums FromStr and Display and take them at every boundary ([#166](https://github.com/smoketurner/quack/pull/166))
- **terminal:** Autocomplete slash commands from their clap definition ([#122](https://github.com/smoketurner/quack/pull/122))
- **jobs:** Show a percentage beside job progress ([#121](https://github.com/smoketurner/quack/pull/121))
- **embedding:** Role prefixes, per-vector profiles, and quack embeddings refresh ([#114](https://github.com/smoketurner/quack/pull/114))
- **server:** Reads and audit rows never wait for the writer ([#113](https://github.com/smoketurner/quack/pull/113))
- Work-queue follow-ups: priority, cancellable ingest, writer actor, polish ([#112](https://github.com/smoketurner/quack/pull/112))
- Run background work on shared work queues ([#111](https://github.com/smoketurner/quack/pull/111))
- **cli:** Add `quack doctor` and run without a chat model ([#110](https://github.com/smoketurner/quack/pull/110))
- **cli:** Add `quack config` to show the configuration in force ([#109](https://github.com/smoketurner/quack/pull/109))
- **ingestion:** Pdf_oxide parser, cross-page PDF chunks, concurrent embedding ([#108](https://github.com/smoketurner/quack/pull/108))
- **analysis:** Make the ontology and the graph answerable from tool calling ([#99](https://github.com/smoketurner/quack/pull/99))

### Fixes

- **auth:** End sessions and audit an on-behalf-of token revocation at the moment of refusal ([#299](https://github.com/smoketurner/quack/pull/299))
- **jobs:** End a job, deliver its waiters, and evict under one lock ([#302](https://github.com/smoketurner/quack/pull/302))
- **terminal:** Keep write-approval SQL visible across session switches ([#301](https://github.com/smoketurner/quack/pull/301))
- **auth:** Resolve subject_claim from named Person fields, not just rest ([#300](https://github.com/smoketurner/quack/pull/300))
- **auth:** Serialize the vault key's first seal against cold-start races ([#298](https://github.com/smoketurner/quack/pull/298))
- **oidc:** Write the Login/Allowed audit row before committing the sign-in token ([#297](https://github.com/smoketurner/quack/pull/297))
- **llm:** Claude Opus 5.5 and GPT-5.6 requests, reasoning effort settings, and stale agent docs ([#290](https://github.com/smoketurner/quack/pull/290))
- **auth:** Harden client registration against interrupts, races, and key loss ([#286](https://github.com/smoketurner/quack/pull/286))
- Close the remaining Detail auth, audit, and token bugs ([#275](https://github.com/smoketurner/quack/pull/275))
- **audit:** Record failed authorized searches as Outcome::Error ([#270](https://github.com/smoketurner/quack/pull/270))
- **storage:** Preserve DECIMAL digits that do not round-trip as JSON numbers ([#260](https://github.com/smoketurner/quack/pull/260))
- **ontology:** Scope auto-accept to one run and report the count actually accepted ([#261](https://github.com/smoketurner/quack/pull/261))
- **cli:** Ignore control-key combos when reading passwords ([#255](https://github.com/smoketurner/quack/pull/255))
- **import:** Redact the whole password when it contains a raw @ ([#258](https://github.com/smoketurner/quack/pull/258))
- **graph:** Do not report never-built graphs as stale ([#273](https://github.com/smoketurner/quack/pull/273))
- **llm:** Honor AWS_ENDPOINT_URL so bedrock chat and embeddings agree ([#272](https://github.com/smoketurner/quack/pull/272))
- **audit:** Render token-bearer logout as API Activity, not OCSF Logoff ([#252](https://github.com/smoketurner/quack/pull/252))
- **storage:** Roll back the writer's DuckDB connection after a panic ([#271](https://github.com/smoketurner/quack/pull/271))
- **ingestion:** Drop nested script/style/SVG text from HTML headings ([#267](https://github.com/smoketurner/quack/pull/267))
- **server:** Filter removed non-admin members out of the workspaces list token branch ([#266](https://github.com/smoketurner/quack/pull/266))
- **ontology:** Carry property since_version per class membership ([#264](https://github.com/smoketurner/quack/pull/264))
- **jobs:** Run when_ended cleanup when the job is evicted from history ([#262](https://github.com/smoketurner/quack/pull/262))
- **audit:** Label a failed member removal as error, not allowed ([#259](https://github.com/smoketurner/quack/pull/259))
- **graph:** Dedup merge proposals by node pair, not orientation ([#268](https://github.com/smoketurner/quack/pull/268))
- **graph:** Resolve entry-point aliases case- and whitespace-insensitively ([#263](https://github.com/smoketurner/quack/pull/263))
- **okf:** Keep canonical class ids verbatim when proposing from quack's own entity files ([#274](https://github.com/smoketurner/quack/pull/274))
- **server:** Audit the bundle import's ontology restore and candidate run ([#269](https://github.com/smoketurner/quack/pull/269))
- **ingestion:** Attribute page-break chunks to the page their content begins on ([#265](https://github.com/smoketurner/quack/pull/265))
- **terminal:** Keep one assistant message per turn across tool calls ([#257](https://github.com/smoketurner/quack/pull/257))
- **ingestion:** Refuse colliding sanitized workbook sheet names ([#256](https://github.com/smoketurner/quack/pull/256))
- **storage:** Drop orphaned leading assistant from trimmed history ([#253](https://github.com/smoketurner/quack/pull/253))
- **embedding:** Count never-embedded graph nodes in the refresh plan ([#254](https://github.com/smoketurner/quack/pull/254))
- **config:** Provider names cannot contain '.' ([#221](https://github.com/smoketurner/quack/pull/221))
- **analysis:** Send tool schemas strict OpenAI-compatible servers accept ([#208](https://github.com/smoketurner/quack/pull/208))
- **okf:** Stream the export instead of holding the bundle in memory ([#204](https://github.com/smoketurner/quack/pull/204))
- Make the interfaces agree on config, settings, hops, prompts, CSV, import, and delete ([#171](https://github.com/smoketurner/quack/pull/171))
- **ontology:** Make propose always add only what the ontology lacks ([#163](https://github.com/smoketurner/quack/pull/163))
- Small CLI and web bugs, and dead code ([#164](https://github.com/smoketurner/quack/pull/164))
- **web:** Keep API tokens out of URLs and responses out of caches ([#161](https://github.com/smoketurner/quack/pull/161))
- **analysis:** Keep run_sql turns from looping one query per group ([#106](https://github.com/smoketurner/quack/pull/106))
- **storage:** Index joined identifiers and support quoted phrase search ([#104](https://github.com/smoketurner/quack/pull/104))
- **ingestion:** Read embedding_batch_size instead of a constant ([#102](https://github.com/smoketurner/quack/pull/102))

### Performance

- **bench:** Measure retrieval latency against workspace size with criterion ([#103](https://github.com/smoketurner/quack/pull/103))
- Bound the Ollama embedding window and hide the entity argument without a graph ([#101](https://github.com/smoketurner/quack/pull/101))
- Fix per-turn prompt cost, Ollama reload churn, and idle TUI redraws ([#100](https://github.com/smoketurner/quack/pull/100))

### Refactoring

- **auth:** Choose sign-in registration from discovery, not the issuer's host ([#287](https://github.com/smoketurner/quack/pull/287))
- **config:** Move the embedding model and its width into [embedding] ([#222](https://github.com/smoketurner/quack/pull/222))
- Finish the day's cleanups (dead okf code, --format, --source, token budgets) ([#207](https://github.com/smoketurner/quack/pull/207))
- Vectors and token counts are their own types ([#203](https://github.com/smoketurner/quack/pull/203))
- Storage, control, import, jobs, and server returns are named ([#202](https://github.com/smoketurner/quack/pull/202))
- Graph, ontology, and analysis returns are named structs ([#200](https://github.com/smoketurner/quack/pull/200))
- Interface flags that name a choice are enums ([#199](https://github.com/smoketurner/quack/pull/199))
- Core flags that name a choice are enums ([#198](https://github.com/smoketurner/quack/pull/198))
- Session message metadata and tool names are typed ([#197](https://github.com/smoketurner/quack/pull/197))
- Run and message ids are their own types ([#195](https://github.com/smoketurner/quack/pull/195))
- Ontology ids are their own types ([#194](https://github.com/smoketurner/quack/pull/194))
- Graph node and edge ids are their own types ([#193](https://github.com/smoketurner/quack/pull/193))
- Document and chunk ids are their own types ([#192](https://github.com/smoketurner/quack/pull/192))
- Session ids are their own type ([#190](https://github.com/smoketurner/quack/pull/190))
- Workspace, user, and audit ids are their own types ([#189](https://github.com/smoketurner/quack/pull/189))
- CLI and agent entry points on their types ([#188](https://github.com/smoketurner/quack/pull/188))
- Extract text only from text formats ([#187](https://github.com/smoketurner/quack/pull/187))
- Ingestion, import, OKF, jobs, and embedding on their types ([#185](https://github.com/smoketurner/quack/pull/185))
- Llm, OAuth, config inspector, doctor, and crypto on their types ([#184](https://github.com/smoketurner/quack/pull/184))
- **config:** Deserialize into validated types ([#183](https://github.com/smoketurner/quack/pull/183))
- Storage helpers onto their types, one RunControl, bounded long runs ([#182](https://github.com/smoketurner/quack/pull/182))
- Move graph and ontology helpers onto their types ([#181](https://github.com/smoketurner/quack/pull/181))
- Share the extraction machinery between graph and ontology ([#180](https://github.com/smoketurner/quack/pull/180))
- Share graph queries, id-prefix lookups, and small utilities across interfaces ([#179](https://github.com/smoketurner/quack/pull/179))
- **analysis:** Move analysis helpers onto their types ([#178](https://github.com/smoketurner/quack/pull/178))
- **analysis:** Shared tool deps, schemas, and error handling ([#177](https://github.com/smoketurner/quack/pull/177))
- **terminal:** Typed slash commands and methods on their types ([#176](https://github.com/smoketurner/quack/pull/176))
- **server:** Move the server's free helpers onto their types ([#174](https://github.com/smoketurner/quack/pull/174))
- **server:** One operation per action for the API and the web console ([#173](https://github.com/smoketurner/quack/pull/173))
- **server:** One BackgroundRun for the three audited background passes ([#172](https://github.com/smoketurner/quack/pull/172))
- **storage:** Make vector_type public instead of wrapping it ([#170](https://github.com/smoketurner/quack/pull/170))
- Type errors, statuses, kinds, and keys that were free text ([#167](https://github.com/smoketurner/quack/pull/167))

### Documentation

- Describe transactional audit, one-place revocation, and the vault key lock in CLAUDE.md ([#303](https://github.com/smoketurner/quack/pull/303))
- Tighten the README ([#289](https://github.com/smoketurner/quack/pull/289))
- Tighten docs/ in Amazon narrative style; ci(release): cancel same-tag runs ([#288](https://github.com/smoketurner/quack/pull/288))
- **auth:** Recommend a private_key_jwt client for the Vouch setup ([#282](https://github.com/smoketurner/quack/pull/282))
- **authentication:** Set up Vouch without DPoP, with actor = false for on-behalf-of ([#276](https://github.com/smoketurner/quack/pull/276))
- Explain how people sign in to quack and how quack signs in to providers ([#219](https://github.com/smoketurner/quack/pull/219))

### Tests

- **jobs:** Wait for lanes to drain instead of asserting at once ([#165](https://github.com/smoketurner/quack/pull/165))

### Build and CI

- **release:** Update the Homebrew tap on each release ([#162](https://github.com/smoketurner/quack/pull/162))
- **release:** Build .deb and .rpm packages and publish them ([#160](https://github.com/smoketurner/quack/pull/160))
- Name the repository for gh in the publish job ([#90](https://github.com/smoketurner/quack/pull/90))

### Other

- Merge imports from the same module into one use ([#169](https://github.com/smoketurner/quack/pull/169))
- Import quack's own items instead of spelling their paths ([#168](https://github.com/smoketurner/quack/pull/168))
- **docs:** Clean up readme
## v2026.9.2 (2026-09-21)

### Features

- **analysis:** Record the provider's token usage for a turn ([#89](https://github.com/smoketurner/quack/pull/89))

### Performance

- **analysis:** Give reads their own DuckDB connections ([#88](https://github.com/smoketurner/quack/pull/88))

### Other

- **rustc:** Update to 1.98.1 everywhere
## v2026.9.1 (2026-09-20)

### Tests

- **server:** Fire login throttle attempts concurrently ([#87](https://github.com/smoketurner/quack/pull/87))
## v0.1.1 (2026-09-20)

### Features

- **extraction:** Report progress per chunk, bound and parallelize the calls
- **terminal:** The operations the web has, rendered for a terminal
- **agent:** Cancel a running turn from every interface
- **examples:** Add the storms example on NOAA's 2024 Storm Events Database
- **import:** Snapshot Postgres, SQLite, and HTTP data files as tables
- **okf:** Export and import workspaces as Open Knowledge Format bundles
- **graph:** Build, resolve, traverse, and expose the knowledge graph
- **mcp:** Serve the workspace over MCP on stdio and streamable HTTP
- **storage:** Stem keyword-index tokens and record the vector index decision
- **ingestion:** Parse HTML, DOCX, PPTX, and load workbooks as tables
- **analysis:** Add a reranking hook with the chat model as a provider
- **sessions:** Share sessions with members and load piped stdin as a table
- **ingestion:** Dedup documents by sha256 and record source and title
- **analysis:** Confine the workspace connection and teach the agent DuckDB's dialect ([#37](https://github.com/smoketurner/quack/pull/37))
- **ontology:** Propose from document evidence with the chat model
- **ontology:** Propose from table evidence into a review queue
- **ontology:** Add the ontology model, versioned store, CLI, API, and page
- **print:** Show a progress indicator while the turn is waiting
- Make demo-data loads one public-domain supply chain dataset
- Add make demo-data with public sample tables and a document
- **web:** Render answers as Markdown
- **web:** Delete sessions and list a new one when its first answer lands
- **server:** Stop gracefully on SIGTERM as well as Ctrl-C
- **server:** Carry the request id in every request log line
- **server:** Log each request with tower-http's TraceLayer
- **server:** Print a startup banner with the effective configuration
- **server:** Add the askama and htmx web UI
- **server:** Add quack serve with the REST API, auth, roles, and audit
- **storage:** Add server users, tokens, members, and the split audit
- **llm:** Add OAuth PKCE and device-code provider auth
- **chart:** Replace the ECharts spec with quack's own chart spec
- **context:** Store and version the workspace context in the workspace
- **retrieval:** Hybrid search, citations, pinned documents, and query mode
- **storage:** Record sessions in the workspace and resume them
- Stream agent turns as events with visible steps and a write prompt
- Add TUI interface with file ingestion support ([#3](https://github.com/smoketurner/quack/pull/3))
- Add analysis engine with rig framework and XDG paths ([#2](https://github.com/smoketurner/quack/pull/2))
- Add ingestion pipeline with embeddings and vector search
- Add core foundation with config, control plane, and query CLI

### Fixes

- **server:** Expire web sessions and throttle the login form ([#81](https://github.com/smoketurner/quack/pull/81))
- **cli:** Skip a pipe on stdin that has nothing to read
- **cli:** End quietly when the reader closes stdout early
- **print:** Show the validated answer, not unvalidated streamed text
- **okf:** Stop proposing relations the ontology already has
- **import:** Refuse sqlite sources inside quack's own data directory
- **storage:** Keep columns that share a name in JSON and NDJSON output
- **graph:** Keep the merge candidate query inside the memory limit
- Propagate swallowed errors and cover graph writes with the statement timeout
- **graph:** Sample document extraction evenly and record extracted chunks
- **okf:** Make the bundle a documented one-way export that restores what it can
- **web:** Chat and page ergonomics
- **ontology:** Treat mapped tables as covered and review candidates in bulk
- **api:** Set a session's mode at creation only
- **terminal:** Let the 'a' answer cover the rest of the current turn
- **server:** Audit every allowed workspace read on every channel
- **storage:** Search only ready documents and keep chunks across a dimension change
- **ingestion:** One document per table and dedup that checks the table
- **graph:** Never merge keyed rows and bound the resolution pass
- **agent:** Size Ollama's context window and survive derailed turns
- **storage:** Take a document's graph rows and files with it on delete
- **import:** Hold server-side imports to an import policy
- **mcp:** Give each query call its own session and reset on failure
- **terminal:** Gate /sql like agent statements and list tables safely
- **storage:** Apply the row cap while reading query results
- **ontology:** Keep induced relations distinct when tables share a column name
- **storage:** Render every DuckDB value type faithfully in query results
- **ontology:** Stream extraction calls and keep low-support candidates aside
- **agent:** Accept file names and id prefixes in search document_ids
- **web:** Revalidate static assets instead of caching them for a day
- **web:** Show that the assistant is working during a turn
- **server:** Drop the session a failed first turn leaves empty
- **terminal:** Show full session ids and never blank the screen on resume
- **agent:** Live-model fixes for print mode, citations, and write policy
- **storage:** Drop DuckDB extensions so the static binary is self-contained
- **config:** Reject unknown keys and select models explicitly
- **storage:** Bind DuckDB values as parameters and quote identifiers
- **analysis:** Classify, gate, and limit agent SQL
- **crypto:** Install the aws-lc-rs rustls provider at startup
- **analysis:** Register search_documents as a real agent tool ([#12](https://github.com/smoketurner/quack/pull/12))

### Performance

- **graph:** Extract mapped tables in batched set-based writes

### Refactoring

- **storage:** Move control.db migrations to SQL files ([#71](https://github.com/smoketurner/quack/pull/71))
- One response object and one refusal contract across interfaces
- Import types and modules instead of spelling out nested paths
- **storage:** Make the workspace file the classification boundary
- Merge quack-cli and quack-tui into one quack binary
- Remove deps-lock crate and all DSQL references

### Documentation

- Derive stale strings from the code and align the design doc
- Simplify the README
- Rewrite the README and the stack docs for quack, drop the template ones
- Map the OKF export and import gap to issue #36
- Name the loader's workspace argument WORKSPACE
- Spell out the loader's workspace argument in the logistics README
- Move the demo data into examples/logistics with a README
- Track issue priority with the Priority field and issue types
- Record that the live-model verification (#20) is closed
- Map every design gap to a GitHub issue
- Update the design doc gap list for the cleanup commits
- Rewrite design doc around a knowledge-engine core ([#4](https://github.com/smoketurner/quack/pull/4))

### Tests

- Cover print mode's step output and close the review's test seams
- Add unit and integration tests for ingestion pipeline

### Build and CI

- Run CI on merge_group so the queue has checks to wait for ([#86](https://github.com/smoketurner/quack/pull/86))
- Queue releases and cancel superseded workflow runs ([#85](https://github.com/smoketurner/quack/pull/85))
- Fix dead build caches and get under the 10 GB cache budget ([#80](https://github.com/smoketurner/quack/pull/80))
- **release:** Sign Windows builds on tag runs only
- **release:** Build and sign every target in a reusable workflow
- Add the tag-triggered release pipeline, container images, and compose
- Add run-server target for a local server
- Bump rustls to 0.23.45 for RUSTSEC-2026-0285
- Bump the actions group across 1 directory with 3 updates ([#1](https://github.com/smoketurner/quack/pull/1))

### Deps

- Bump the rust-dependencies group across 1 directory with 3 updates ([#39](https://github.com/smoketurner/quack/pull/39))

### Docker

- Bump alpine from 3.22 to 3.24 ([#38](https://github.com/smoketurner/quack/pull/38))
