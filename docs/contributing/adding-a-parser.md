# Adding a file parser

A parser turns an uploaded file into either tables (loaded into DuckDB) or chunks (text
under headings, vectorized). This recipe is for the chunk kind; a table format goes through
`load` instead of `extract`. The DOCX parser (`ingestion/office.rs`) is the worked pattern.

1. **`crates/quack-core/src/ingestion/parser.rs`**
   - Add the variant to `FileType` and its extension(s) to `EXTENSIONS`, which is what
     `FileType::from_path` and the upload form's accept list read.
   - `FileType::load` returns `Load::Chunks(TextFormat::X)` for a text format or
     `Load::Tables` for a tabular one.
   - `mime_type` names what the server sends the file back as.
   - Add the variant to `TextFormat` and the arm of `TextFormat::extract` that calls your
     module. `extract` takes the raw bytes and a `DecompressionBudget`; a format that
     inflates (a zip) must read through `budget.reader(...)` so a bomb stops at
     `[ingestion].upload_max_mb` worth of decompressed bytes.
2. **`crates/quack-core/src/ingestion/<format>.rs`**: `pub fn <format>(data: &[u8], budget:
   DecompressionBudget) -> Result<Extracted>`. Fill `Extracted { title, sections, flow,
   pages }`: one `Section` per heading (`heading`, `page`, `text`), `Flow::Sectioned` when
   the format has headings and `Flow::Continuous` when it is one running text with page
   breaks (PDF). Return `Error::Ingestion` with the reason when there is no text at all.
   Declare the module in `ingestion/mod.rs`.
3. **`crates/quack-core/tests/ingestion_integration.rs`**: a fixture file under
   `crates/quack-core/eval/documents/` (small, redistributable) and a test that ingests it
   and checks the title, the section headings, and that the chunks cite the right pages.
4. **`fuzz/`**: a target for the new `TextFormat` arm (copy `fuzz/fuzz_targets/docx.rs`), a
   `[[bin]]` entry in `fuzz/Cargo.toml`, the extension in `fuzz/seed.sh`, and the target in
   `.github/workflows/fuzz.yml`'s matrix.
5. **The format lists**: `README.md` (the ingest line), `CLAUDE.md` (the `ingest` comment),
   and the parser table in `docs/design-doc.md` section 6.1.

## What bites

- `clippy::absolute_paths`: `use` the type; `crate::a::b::Type` at a call site is denied.
- Every `unwrap`, `expect`, index, and slice is denied outside tests; in tests opt out
  narrowly with `#[expect(clippy::unwrap_used, reason = "...")]`.
- A file over 1,000 lines keeps its tests in a sibling `tests.rs`.
- Date and time through `jiff`; ids through `uuid::Uuid::now_v7()`.
- Nothing a parser reads may leave the workspace: it gets bytes, never a path, and never
  writes to `control.db`. Ingestion's audit rows are written by the caller, not the parser.
