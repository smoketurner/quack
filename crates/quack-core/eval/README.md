# Eval fixture

The in-tree fixture for `cargo run -p quack-core --example eval` (`make eval`), a
storms-like corpus written for this harness rather than downloaded: not the full NOAA
dataset `examples/storms/` uses.

- `documents/` — 27 short Markdown documents: a glossary, the EF and Saffir-Simpson
  scales, storm data preparation guidance, tornado safety and damage notation, ten
  episode narratives each carrying a unique `SR-nnnn` report identifier and at least one
  quotable phrase, four forecast office pages, and eight decoy documents (see below).
- `tables/` — `events.csv` (25 rows), `fatalities.csv` (39 rows, referencing
  `events.event_id`), and `states.csv` (6 rows), shaped like `examples/storms` but tiny.
- `gold_questions.json` — questions mapped to the document filename and a content
  substring that identifies the expected chunk (chunk ids are UUID v7, minted at ingest,
  so they cannot be fixed in the fixture ahead of time). Tagged `identifier`, `phrase`,
  or `semantic` so a tokenization change shows up in its own row instead of averaged
  away. One phrase question (`"catastrophic flood damage"`) sets `expect_empty: true`:
  every word in it appears somewhere in the corpus, but the exact phrase appears
  nowhere, so the correct answer is no result at all.

  **Decoys (issue #77).** Five `identifier` questions (`SR-8841`, `SR-4437`, `SR-7765`,
  `SR-2214`, and the `AKQ` office code) and two `phrase` questions (`"flash flood
  emergency"`, `"wall of water"`) are deliberately winnable by a decoy document under
  plain BM25: `survey-site-*.md` and `office-code-list.md` repeat the split identifier
  pieces (`SR` and a bare number, or `AKQ` and `Wakefield, Virginia`) densely enough to
  outscore the document that spells the identifier out; `emergency-notice.md` and
  `structural-notice.md` repeat the phrase's words apart rather than adjacent. Before
  identifier-joining and phrase filtering (main), keyword and hybrid search rank the
  decoy first; after, the joined-identifier term or the phrase substring filter should
  recover the true chunk. A change to any of this content should keep re-verifying that
  the decoy still wins on unpatched tokenization — see the PR that added these decoys
  for the before/after numbers.
- `expected_ontology.json` — the classes, properties, and relations
  `ontology::induction::propose_from_tables` is expected to propose from `tables/`.
- `graph_fixture.json` — a two-class, one-relation ontology; ten chunks named by an
  anchor substring (`SR-nnnn`) with a canned extraction answer for each (two of which
  add an out-of-ontology class and relation, to exercise drift dropping); and the node
  and edge set extraction should produce once invalid data is filtered out.
- `citation_fixture.json` — recorded answers with `[n]` markers, some valid, some
  referencing an unregistered chunk or a provider channel token, with how many markers
  each is expected to survive `analysis::citations::validate`.

Ingestion goes through the normal `ingestion::ingest_file` path into a temporary
workspace. There is no Ollama dependency: chunk embeddings come from a deterministic
hashing "embedding" (see the `HashEmbedder` doc comment in `examples/eval.rs`), and the
graph section uses a canned `GraphExtractor` instead of a chat model.
