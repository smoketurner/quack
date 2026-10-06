# Upgrading

One section per release, newest first: the config keys it added, renamed, or removed; the
workspace schema version it moves to and what the first open of each workspace does; whether
`quack embeddings refresh` is needed; and any new `control.db` migration. The version-bump
pull request adds the release's section, and `release.yml` refuses to publish a tag this page
has no section for (`scripts/upgrading-section.sh`); the section leads the release notes.

Two rules hold for every release:

- Config loading is strict: a key quack no longer recognizes stops it at startup, so a removed
  or renamed key must change in `config.toml` before the new binary starts. `quack config`
  lists every key in force and the ones in the file it does not recognize.
- A workspace file records the schema version and the quack version that wrote it. A newer
  quack upgrades the file on first open; an older quack refuses a file a newer one wrote
  (`docs/migrations.md`). Back up before a release that moves the schema, or take a snapshot.

`quack doctor` after the upgrade says whether the workspace's vectors are stale
(`quack embeddings refresh -w NAME` is its fix line).

## Unreleased

- Config: `[retrieval].languages` added (default `["auto"]`): what a document may be
  detected as for keyword stemming, `"auto"` or Snowball language names.
- Schema: workspace 12. The first open of each workspace detects every document's language
  from its first chunks and rebuilds the keyword index (`_quack_terms`), reading every chunk
  once. `control.db` unchanged (migration 8).
- Embeddings: no refresh needed.

The version-bump pull request renames this section to the tag.

## v2026.10.3

- Config: no keys added or removed. The proxy variables (`HTTP_PROXY`, `HTTPS_PROXY`,
  `ALL_PROXY`, `NO_PROXY`) are now read; `quack config` lists them.
- Schema: unchanged (workspace 11, `control.db` migration 8).
- Embeddings: no refresh needed.

## v2026.10.2

- No upgrade steps.

## v2026.10.1

- Config: added `[analysis].compact_history`, `[retrieval].rerank_model` (with
  `[retrieval].rerank = "reranker"`), and `[server].permission_timeout_seconds`. All have
  defaults.
- Schema: unchanged.
- Embeddings: no refresh needed.

## v2026.9.8

- No upgrade steps.

## v2026.9.7

- Config: added `temperature`, `effort`, and `background_effort` on `[providers.NAME]` and
  the `[providers.NAME.models."ID"]` tables (`models`). All optional.
- Schema: unchanged.

## v2026.9.6

- No upgrade steps.

## v2026.9.5

- No upgrade steps.

## v2026.9.4

- Config: added `[providers.NAME].headers`. Optional.
- Schema: unchanged.

## v2026.9.3

- Config: `quack config` arrived with this release, and with it the strict key list; a key
  the file carried that quack did not read now stops it at startup, with the key it
  resembles named. Run `quack config` once and fix what it reports.
- `control.db`: migrations 0004 (audit keyset index), 0005 (`user_tokens`), 0006
  (`provider_tokens`), 0007 (`client_keys`), and 0008 (`client_registrations`) run on the
  first start.
- Workspace schema: moves to 11. The first open of each workspace rebuilds the term index
  (versions 6 and 7: stemming, then the joined identifier term), tags every stored vector
  with the embedding profile the workspace had recorded (8), marks documents left in the
  old `pending` status as `error` (9), records each ontology version's acceptance (10), and
  collapses duplicated merge proposals (11). On a workspace with many chunks the first open
  takes minutes; the server answers that workspace once it finishes.
- Embeddings: `quack embeddings refresh` only if the embedding model, width, or prefixes in
  `config.toml` differ from what the workspace recorded; `quack doctor` says so.

## v2026.9.2

- No upgrade steps.

## v2026.9.1

The first release.
