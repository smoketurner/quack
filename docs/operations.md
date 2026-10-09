# Operations

## Erasing a spilled document

When a document that should never have been uploaded lands in a workspace, remove it in
this order and know what each step leaves.

1. **Delete the document** (`quack docs --delete ID`, the Documents page, or
   `DELETE /api/v1/workspaces/{id}/documents/{doc}`). `WorkspaceDb::delete_document`
   removes, in one transaction: its rows in `_quack_chunks` (the text and every vector),
   `_quack_terms` (the keyword index), `_quack_graph_extracted`, the graph provenance that
   pointed at its chunks, the tables it loaded (unless a replacement now owns them), and the
   `_quack_documents` row. Nodes and edges the graph extracted from it stay until
   `quack graph revalidate` or an edit removes them: they are the graph's own claims, with
   their provenance gone.
2. **Know what still quotes it.** A chat session that cited the document keeps the answer
   text and the citation's excerpt (the first 500 characters of the chunk) in
   `_quack_messages`; delete those sessions (`quack sessions`, the chat page, or
   `DELETE .../sessions/{sid}`). A `search` or `query` audit detail row inside the workspace
   quotes the question, not the document. A saved question holds SQL, not rows.
3. **Reclaim the blocks.** DuckDB marks deleted rows free and reuses the space; the bytes
   stay in `data.duckdb` until the blocks are rewritten. To force it, take a snapshot
   (`quack workspace snapshot NAME --to FILE`) and restore it as a new workspace, or export
   the workspace as an OKF bundle and import it into a fresh one: both write a new file
   from live rows only. Then delete the old workspace.
4. **The access log keeps its rows.** `control.db`'s `audit_log` records that the document
   was ingested, read, and deleted, by opaque id, never by content. That is the record of
   the spill and its remediation; it is not erased.

## Taking a snapshot

`quack workspace snapshot NAME --to FILE`, `GET /api/v1/workspaces/{id}/snapshot`, and the
Settings page's download write the workspace as one tar. The workspace's file is closed
while it is copied: quack checkpoints it, closes every connection to it, copies it, and
opens it again. Windows refuses to let any other handle read a DuckDB file that is open, so
every OS takes this path.

- **Under `quack serve`**, requests to that workspace wait while the file is closed: writes
  and audit rows queue, reads wait for their connection. The window is the time to copy
  `data.duckdb` and `files/` to an unnamed temporary file in the data directory; the
  download then streams from that file with the workspace open again. Other workspaces do
  not wait. Leave free space in the data directory for one copy of the workspace.
- **From the command line**, the snapshot opens the workspace itself, so it fails while
  `quack serve` or another `quack` holds the workspace; use the API then.

## Retiring a workspace

`quack workspace delete NAME` (or `DELETE /api/v1/workspaces/{id}`, or the Settings page)
removes the row, its memberships and API tokens, and the directory with the file and the
stored originals under `files/`, once no job of the workspace is running. The access log
keeps the workspace's rows. Take a snapshot first if the content may be wanted again.

The vault key seals tokens, not workspace content; retiring a workspace does not touch it.
Retiring a whole deployment means deleting the data directory (`control.db`, every
workspace, `vault.key` if present) and the keychain entry `quack` / `vault`, and revoking
the OAuth clients it registered (`quack auth unregister`).

## Refreshing saved imports

quack has no scheduler. An import saved with `--save NAME` runs again when something calls
`quack import refresh NAME`. To refresh nightly, add a cron entry on the host that holds the
data directory:

```
# m h dom mon dow  command
15 2 * * *  quack import refresh orders -w sales
```

A refresh that finds the source unchanged prints `orders: source unchanged` and changes
nothing. A failed refresh exits 1, keeps the rows from the run before, and records the
error, which `quack import list` shows. Under `quack serve`, the Tables page's Refresh button
and `POST /api/v1/workspaces/{id}/imports/{import}/refresh` run one on demand.

## Labelling new rows

`quack classify TABLE --text COLUMN --questions FILE` labels the rows its output table does
not hold, so running it again after new rows arrive labels only those. quack has no
scheduler; to label nightly, add a cron entry on the host that holds the data directory:

```
# m h dom mon dow  command
30 2 * * *  quack classify tickets --text subject,body --questions triage.json -w support
```

Rows whose text changed after they were labelled keep their labels: `--all` labels every row
again, into a staging table that replaces the output only when it completes. A run that
stops (a cancel, a refusal from the model, a killed process) keeps the rows it wrote, and the
next run finishes; a run killed by a signal is marked `interrupted` by the next. A run into
an output labelled under other questions, or after `ollama pull` changed the model's weights,
is refused until `--all`. `quack classify list` shows the runs. Each run's definition and
counts are in `_quack_classifications` inside the workspace file, so deleting a user
rewrites `started_by` there to `removed` like every other record of who did what.

## Hardware sizing

Measured on 2026-10-06 with the release binary (`v2026.10.3` plus the changes of this page,
44 MB), the default settings (`[analysis].memory_limit_mb` 256, `threads` 4,
`reader_pool_size` 4, `[providers.ollama].max_concurrent_requests` 1), Ollama serving
`embeddinggemma` (768-wide vectors) on the same machine, an Apple M5 with 24 GB; the
documents are the evaluation set's Markdown files, copied with a distinct header to reach
the medium size. RSS is the resident set of the `quack` process alone; Ollama's own memory
(the embedding model, about 0.6 GB, plus a chat model of your choice: `gpt-oss:20b` takes
about 13 GB) comes on top.

| Workspace | Documents | Chunks | `data.duckdb` | Ingest RSS peak | Ingest time |
|---|---|---|---|---|---|
| Small | 27 | 46 | 3.9 MB | 77 MB | 2 s |
| Medium | 5,886 | 15,914 | 176 MB | 325 MB | 226 s (70 chunks/s, embedding-bound) |

`quack serve --local` with both workspaces: 22 MB resident idle, 152 MB after one search
and one SQL statement opened the medium workspace (DuckDB's buffer pool and the reader pool
fill on first use), 190 MB after six more searches. Memory per open workspace is bounded by
`memory_limit_mb` for query execution plus the vectors a search scans, so a server holding
several large workspaces open at once needs `memory_limit_mb` times the number of
workspaces in use, plus about 100 MB.

Disk is 11 KB per chunk at this width (vector, text, and the keyword index together), so a
100,000-chunk workspace (roughly 40,000 pages of text) is about 1.1 GB; that size was not
measured. The stored originals under `files/` add the uploads' own size. One processor core
is enough for quack itself; ingestion speed is the embedding model's throughput, which is the
GPU or the CPU Ollama runs on. A reasonable floor for the default local stack with a 20B
chat model is 4 cores, 16 GB of memory (the models take 14 GB of it), and 2 GB of disk per
100,000 chunks; without a local chat model (a remote provider), 2 cores and 2 GB of memory.
