# quack-eval

The evaluation harness. It builds nothing and links nothing of quack: it runs the `quack`
binary as a person would, one process per command over a temporary data directory, and
scores what the binary's own `-f json` output says. A bug in argument parsing, print mode,
config loading, or the data directory layout shows here; a white-box harness over the
library would never see it.

```bash
make eval-offline   # no model: ingestion by format, keyword and hybrid retrieval, ontology induction
make eval           # with Ollama: real embeddings, the agent's answers, the vision model
make eval-scale     # make eval, then the agent over a real 70,000-row table
cargo run -p quack-eval -- --help
```

Nothing fails on a wrong answer. Every case is scored and printed as its section ends, so a
change to a prompt or a model shows up as a number to compare between runs; what a model
writes varies between runs, so one run is a sample, not a verdict. `QUACK_EVAL_OUT=eval.json`
(or `--out`) keeps the numbers. Progress goes to stderr, one line per model call, so a run of
minutes is never silent.

## Sections

| Section | Command driven | Scored by |
|---|---|---|
| Ingestion by format | `quack ingest FILE` for every file in `fixture/formats.json` | a phrase found by `quack search --keyword --in FILE`, or a table's row count by `quack -q` |
| Retrieval | `quack search` with `--keyword`, `--vector`, and neither, for 32 gold questions | recall@1/5/8 and MRR against the chunk the gold names by file and phrase |
| Ontology induction | `quack ontology propose --auto-accept`, then `quack ontology show --format json` | precision and recall of classes, properties, and relations against the hand-written expectation |
| Agent (`--ollama`) | `quack -p QUESTION -f json`, `-r SESSION` for a follow-up | the rows a `run_sql` step returned, the prose, the tool used, the citation, the exit sentence, a refused write |
| Vision (`--ollama`) | `quack ingest` of an image, a scanned PDF, and a DOCX with a figure | the notice's words found in the ingested text |
| Scale (`--scale FILE`) | `quack ingest` of one large CSV, then the agent | rows against `quack -q` of each question's `expect_sql` |

### Scoring the agent

The rule is the product's: `DuckDB`'s rows are the record of an answer. A table question in
`fixture/agent_questions.json` names the result set one `run_sql` step must have returned
(`rows`, any of `rows_any`, or the rows of `expect_sql`, which the harness runs itself for data
that is not fixed in the fixture): each expected row's cells all present in one actual row of
its own, numbers compared as numbers, text ignoring case, every dash a hyphen and every space
a plain one. Extra columns and extra rows pass, so a query that lists every month and an
answer that names the right one is right; the prose `expect` groups say whether the answer
picked the right row. A
question's `shape` says what else the reply must be:

- `not_in_data`: the exit sentence *I can't tell from the data in this workspace.*
- `refusal`: `write_refused` true, and `unchanged` (a statement) reading the same before and after.
- `reading`: an answer that takes a reading of an ambiguous question and asks nothing back.

A `follow_up` is asked on the same session (`-r`). `kind` groups the tally: `table`,
`document`, `join`, `date`, `wide` (a text column with more values than the prompt lists),
`mistyped` (figures stored as text), `unanswerable`, `ambiguous`, `write`, `follow_up`,
`scale`.

## Settings

| Variable | Default |
|---|---|
| `QUACK_EVAL_OLLAMA_URL` | `http://127.0.0.1:11434` |
| `QUACK_EVAL_CHAT_MODEL` | `gpt-oss:20b` |
| `QUACK_EVAL_EMBEDDING_MODEL` | `embeddinggemma-2` |
| `QUACK_EVAL_EMBEDDING_DIMENSION` | `768` |
| `QUACK_EVAL_VISION_MODEL` | `gemma4:e4b` (`none` skips the vision section) |
| `QUACK_EVAL_TURN_SECONDS` | `300`: a question past it is stopped and scored as timed out |
| `QUACK_EVAL_REPEAT` | `1`: past one, each question is asked that many times and the report counts the questions answered the same every time |
| `QUACK_EVAL_OUT` | unset: a path to write the report as JSON |

The defaults are what a developer machine runs; `.github/workflows/eval.yml` names smaller
models for its CPU runner and runs nightly, never gating a pull request.

## The fixture

`fixture/` is a storms-like corpus written for this harness, not the NOAA download
`examples/storms/` uses.

- `documents/` (34 files): a glossary, the EF and Saffir-Simpson scales, storm data
  preparation guidance, tornado safety and damage notation, ten episode narratives each
  carrying a unique `SR-nnnn` report identifier and at least one quotable phrase, four
  forecast office pages, eight decoy documents, `readiness-review.html`,
  `shelter-supply.docx`, `dam-inspection.pdf`, and two documents not in English
  (`sturmschaeden.md`, German; `taifeng-lipei.md`, Chinese, no spaces between words).
- `tables/`: `events.csv` (25 rows), `fatalities.csv` (39 rows, referencing
  `events.event_id`), and `states.csv` (6 rows). `agent_tables/`, loaded only for the
  agent: `offices.csv` (65 forecast offices, a text column wider than the prompt lists) and
  `insurance_claims.csv` (claim amounts as `4.20M` text and dates as text).
- `formats/`: one file per format quack ingests. Hand-written: RTF, EML, MBOX, VTT, SRT, two
  YAML files (a list of records loads as a table, a mapping is chunked), JSON, JSONL, TSV,
  plain text, Python source. Made once with office_oxide, zip, and `DuckDB` and committed:
  XLSX, PPTX, ODT, ODS, Parquet. Fetched once and committed: `yellow-wallpaper.epub`
  (Project Gutenberg #1952, public domain) and the Apache POI sample files `poi-sample.doc`,
  `poi-sample-show.ppt`, and `poi-sample-sheet.xls` (Apache License 2.0, from
  `test-data/` in the POI repository). `formats.json` says what each must become.
- `gold_questions.json`: 32 retrieval questions, each naming its chunk by file name and a
  phrase, tagged `identifier`, `phrase`, `semantic`, or `multilingual`. Decoys (issue #77):
  five identifier questions and two phrase questions are winnable by a decoy document under
  plain BM25 (`survey-site-*.md` and `office-code-list.md` repeat split identifier pieces;
  `emergency-notice.md` and `structural-notice.md` repeat a phrase's words apart), so
  identifier joining and phrase filtering show in their own rows. One question expects no
  result at all (`expect_empty`).
- `expected_ontology.json`: what `quack ontology propose` should propose from `tables/`.
- `agent_questions.json`: 32 questions for the agent (18 follow-ups and shapes included);
  `scale_questions.json`: 6 over the SPC tornado table, scored by `expect_sql`.
- `vision/`: `posted-sign.jpg`, a picture of a short notice; `records-scan.pdf`, a typed cover
  page and a page that is only that picture; `shelter-memo.docx`, a memo with the picture as a
  figure. No file name holds a word the cases are scored on.

`make eval-scale` downloads the SPC tornado database (every US tornado since 1950, one CSV of
about 8 MB, public domain) once into `target/eval-cache/`.
