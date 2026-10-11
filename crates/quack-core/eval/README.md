# Eval fixture

The in-tree fixture for `cargo run -p quack-core --example eval` (`make eval-offline`, no
model) and `... --example eval -- --ollama` (`make eval`, against Ollama), a
storms-like corpus written for this harness rather than downloaded: not the full NOAA
dataset `examples/storms/` uses.

- `documents/` — 29 short Markdown documents: a glossary, the EF and Saffir-Simpson
  scales, storm data preparation guidance, tornado safety and damage notation, ten
  episode narratives each carrying a unique `SR-nnnn` report identifier and at least one
  quotable phrase, four forecast office pages, and eight decoy documents (see below).
  Three more documents exercise the structured parsers: `readiness-review.html` (chrome to
  skip, headings, a `<table>` of office contacts, meta tags), `shelter-supply.docx`
  (outline-level headings, a table, core properties), and `dam-inspection.pdf` (a tagged
  PDF exported by LibreOffice from a Word document: headings set larger than the body, a
  ruled table, a running footer). Their gold questions ask for table cells, which must
  come back as the table's own chunk, and for section-scoped passages.
  Two of the Markdown documents are not in English (issue #395): `sturmschaeden.md`
  (German, whose gold questions use another inflection than the text, `Hagelschaden`
  for `Hagelschäden`) and `taifeng-lipei.md` (Chinese, no spaces between words, so its
  gold questions match only through character bigrams). Their questions are tagged
  `multilingual`.
- `tables/` — `events.csv` (25 rows), `fatalities.csv` (39 rows, referencing
  `events.event_id`), and `states.csv` (6 rows), shaped like `examples/storms` but tiny.
- `gold_questions.json` — questions mapped to the document filename and a content
  substring that identifies the expected chunk (chunk ids are UUID v7, minted at ingest,
  so they cannot be fixed in the fixture ahead of time). Tagged `identifier`, `phrase`,
  `semantic`, or `multilingual` so a tokenization change shows up in its own row instead of averaged
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
- `agent_questions.json` — questions for the agent under `make eval`, each with the values
  its answer must hold (every group matched by one of its strings, ignoring case), the
  tools it should come from (table questions only: passages `always_retrieve` puts in the
  prompt can answer a document question with no tool, so its citation shows the
  grounding), and whether it should cite a passage. `kind` is `table`
  (answered by SQL over `tables/`) or `document` (from `documents/`).
- `vision/shelter-notice.jpg` — a picture of a short notice for the vision model, read
  three ways under `make eval`: as an image, as the only picture on a PDF page, and as a
  figure inside a DOCX (both built at run time). No file name, title, or alt text the
  cases give holds one of the words it is scored on.

Ingestion goes through the normal `ingestion::ingest_file` path into a temporary
workspace. Under `make eval-offline` there is no Ollama dependency: chunk embeddings come
from a deterministic hashing "embedding" (see the `HashEmbedder` doc comment in
`examples/eval/main.rs`), and the graph section uses a canned `Extract<Extraction>` instead
of a chat model.

## Against Ollama (`make eval`)

`make eval` runs the same sections with a real embedding model, then puts
`agent_questions.json` to the chat model through a full agent turn and has the vision model
read the notice (`examples/eval/ollama.rs`). The agent table counts, per kind, correct answers,
answers through a tool the question names (`run_sql` for a table question),
cited answers, fallbacks (answers quack replaced by the query results because they stated a
figure no result held, design doc 7.7), and, under `QUACK_EVAL_REPEAT`, questions answered the
same every time. Nothing fails on a wrong answer: each case is scored and printed, with the
misses listed, and `QUACK_EVAL_OUT=eval.json` keeps the
numbers. Model output varies between runs, so compare a run with the last few, not with
one.

It needs Ollama running with the models pulled; each is a variable with a default small
enough for a CPU:

| Variable | Default |
|---|---|
| `QUACK_EVAL_OLLAMA_URL` | `http://127.0.0.1:11434` |
| `QUACK_EVAL_CHAT_MODEL` | `qwen2.5:3b` |
| `QUACK_EVAL_EMBEDDING_MODEL` | `embeddinggemma` |
| `QUACK_EVAL_EMBEDDING_DIMENSION` | `768` |
| `QUACK_EVAL_VISION_MODEL` | `qwen2.5vl:3b` (`none` skips the vision section) |
| `QUACK_EVAL_TURN_SECONDS` | `300`: a question past it is stopped and scored as timed out |
| `QUACK_EVAL_REPEAT` | `1`: past one, each question is asked that many times and the report counts the questions answered the same every time |

```bash
OLLAMA_CONTEXT_LENGTH=16384 ollama serve &
ollama pull qwen2.5:3b && ollama pull embeddinggemma && ollama pull qwen2.5vl:3b
make eval
```

The agent's prompt (tables, documents, tool guidance) runs past Ollama's default context
window, hence `OLLAMA_CONTEXT_LENGTH`. `.github/workflows/eval.yml` does the same nightly
and on dispatch (with the models as inputs), outside CI: it gates nothing, writes the
report to the job summary, and uploads `eval.json`.
