use std::fmt;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::time::Duration;

use aws_lc_rs::digest;
use jiff::civil::DateTime;

use std::collections::BTreeMap;

use crate::analysis::table_search;
use crate::config::Config;
use crate::crypto;
use crate::embedding::{
    Dimension, EmbeddingStatus, Fingerprint, Input, Profile, Prompts, StaleVectors, Vector,
};
use crate::error::{Error, Result, WrittenBy};
use crate::graph;
use crate::ids::{ChunkId, DocumentId, NodeId, UserId};
use crate::ingestion::TableName;
use crate::ingestion::parser::{DocumentMeta, FileType, Load, PageCounts, SectionKind};
use crate::ontology::store::{self as ontology_store, Acceptance};
use crate::ontology::{Measure, Ontology, Property};
use crate::saved;
use crate::storage::control::ResourceKind;
use crate::storage::profile::{self, TableNote, TableProfile};
use crate::text::OneLine;

mod terms;

use terms::TermFrequencies;
pub use terms::{Analyzer, Language, LanguageSetting, Stemming, Unspaced};

/// BM25 parameters for the keyword index quack maintains in `_quack_terms`.
const BM25_K1: f64 = 1.2;
const BM25_B: f64 = 0.75;

/// How far a quoted-phrase query over-fetches BM25 candidates before the
/// substring post-filter, since `_quack_terms` carries no positions.
const PHRASE_OVER_FETCH: u32 = 4;
/// Absolute cap on phrase-search candidates, regardless of `top_k`.
const PHRASE_CANDIDATE_CAP: u32 = 500;

/// Chunks the keyword index rebuild reads at a time.
const REINDEX_PAGE: u32 = 1000;

/// Every internal table carries this prefix; anything starting with it is hidden.
pub const INTERNAL_PREFIX: &str = "_quack_";

/// What a deleted user's id and name become in a workspace file
/// (`WorkspaceDb::forget_user`).
pub const REMOVED_USER: &str = "removed";

/// Every stored vector records the profile it was made under.
const VECTOR_PROFILES: u32 = 8;
/// A document's status is one of four values.
const DOCUMENT_STATUSES: u32 = 9;
/// An ontology version records whether it was reviewed.
const ONTOLOGY_ACCEPTANCE: u32 = 10;
/// `_quack_graph_merges` is deduplicated by node pair, not by orientation:
/// a pair whose provenance flipped between passes could land twice, once
/// as `(keep, drop)` and once as `(drop, keep)`; collapse each pair to one
/// row, keeping the more-decided one so a reviewer's rejection is not lost.
const MERGE_DEDUP: u32 = 11;
/// Every user table is profiled (`storage::profile`), and the graph is
/// readable through `graph_` views.
const TABLE_PROFILES: u32 = 12;
/// Each document records the language it was detected as, and its chunks
/// are stemmed under it, with unspaced scripts as bigrams (issue #395):
/// every document is detected and every chunk reindexed.
const DOCUMENT_LANGUAGES: u32 = 13;

/// The documents table, and the columns older files gain on open.
const DOCUMENTS_DDL: &str = "
    CREATE TABLE IF NOT EXISTS _quack_documents (
        id TEXT PRIMARY KEY,
        filename TEXT NOT NULL,
        title TEXT,
        mime_type TEXT,
        size_bytes BIGINT,
        sha256 TEXT,
        source TEXT,
        ingested_at TIMESTAMP DEFAULT now(),
        status TEXT DEFAULT 'queued',
        error_message TEXT,
        pinned BOOLEAN NOT NULL DEFAULT false,
        chunk_count INTEGER,
        ingested_by TEXT,
        tables JSON,
        page_count INTEGER,
        pages_unreadable INTEGER,
        pages_empty INTEGER,
        superseded_by TEXT,
        source_root TEXT,
        source_path TEXT
    );
    ALTER TABLE _quack_documents ADD COLUMN IF NOT EXISTS pinned BOOLEAN DEFAULT false;
    ALTER TABLE _quack_documents ADD COLUMN IF NOT EXISTS title TEXT;
    ALTER TABLE _quack_documents ADD COLUMN IF NOT EXISTS sha256 TEXT;
    ALTER TABLE _quack_documents ADD COLUMN IF NOT EXISTS source TEXT;
    ALTER TABLE _quack_documents ADD COLUMN IF NOT EXISTS chunk_count INTEGER;
    ALTER TABLE _quack_documents ADD COLUMN IF NOT EXISTS ingested_by TEXT;
    ALTER TABLE _quack_documents ADD COLUMN IF NOT EXISTS tables JSON;
    ALTER TABLE _quack_documents ADD COLUMN IF NOT EXISTS page_count INTEGER;
    ALTER TABLE _quack_documents ADD COLUMN IF NOT EXISTS pages_unreadable INTEGER;
    ALTER TABLE _quack_documents ADD COLUMN IF NOT EXISTS pages_empty INTEGER;
    ALTER TABLE _quack_documents ADD COLUMN IF NOT EXISTS superseded_by TEXT;
    ALTER TABLE _quack_documents ADD COLUMN IF NOT EXISTS source_root TEXT;
    ALTER TABLE _quack_documents ADD COLUMN IF NOT EXISTS source_path TEXT;
    ALTER TABLE _quack_documents ADD COLUMN IF NOT EXISTS author TEXT;
    ALTER TABLE _quack_documents ADD COLUMN IF NOT EXISTS authored_at TIMESTAMP;
    ALTER TABLE _quack_documents ADD COLUMN IF NOT EXISTS modified_at TIMESTAMP;
    ALTER TABLE _quack_documents ADD COLUMN IF NOT EXISTS tags JSON;
    ALTER TABLE _quack_documents ADD COLUMN IF NOT EXISTS metadata JSON;
    ALTER TABLE _quack_documents ADD COLUMN IF NOT EXISTS language TEXT;";

/// Summaries of the turns a session's history window leaves out
/// (`[analysis].compact_history`), each with how many replayable messages,
/// oldest first, it covers.
const SESSION_SUMMARIES_DDL: &str = "CREATE TABLE IF NOT EXISTS _quack_session_summaries (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    covers INTEGER NOT NULL,
    summary TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT now()
);";

/// Schema version of the internal tables, recorded in `_quack_meta`: the
/// newest step above.
const WORKSPACE_SCHEMA_VERSION: u32 = DOCUMENT_LANGUAGES;

/// The oldest `DuckDB` that must read a file created here, given to `DuckDB`
/// when the file is opened. It is the bundled library's own default, named so
/// that a `duckdb` upgrade cannot change the format of new files unnoticed.
/// `DuckDB` uses it only when it creates a file; an existing one keeps its
/// format.
const STORAGE_COMPATIBILITY_VERSION: &str = "v0.10.2";

/// The tables that hold embedding vectors, with the same `embedding` and
/// `embedding_profile` columns.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum VectorTable {
    Chunks,
    GraphNodes,
}

impl VectorTable {
    const ALL: [Self; 2] = [Self::Chunks, Self::GraphNodes];

    const fn as_str(self) -> &'static str {
        match self {
            Self::Chunks => "_quack_chunks",
            Self::GraphNodes => "_quack_graph_nodes",
        }
    }
}

impl fmt::Display for VectorTable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// The keys of `_quack_meta`, the workspace's own settings.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MetaKey {
    /// `WORKSPACE_SCHEMA_VERSION` when the internal tables were last upgraded.
    SchemaVersion,
    /// The width of the vector columns.
    EmbeddingDimension,
    /// The embedding model before profiles; read once on upgrade and deleted.
    EmbeddingModel,
    /// The ontology version the graph was last built or revalidated with.
    GraphBuiltWithOntologyVersion,
    /// Extraction's unknown classes and relations, as a JSON count map.
    GraphDrift,
    /// The quack version that last opened the file for writing.
    WrittenByQuack,
    /// The `DuckDB` library version that quack was built with.
    WrittenByDuckDb,
    /// The stemmings the documents were indexed under, comma-separated: a
    /// query is tokenized under all of them.
    Languages,
}

text_enum!(MetaKey, "meta key", {
    SchemaVersion => "schema_version",
    EmbeddingDimension => "embedding_dimension",
    EmbeddingModel => "embedding_model",
    GraphBuiltWithOntologyVersion => "graph_built_with_ontology_version",
    GraphDrift => "graph_drift",
    WrittenByQuack => "written_by_quack",
    WrittenByDuckDb => "written_by_duckdb",
    Languages => "languages",
});
text_enum_sql!(MetaKey);

/// Width used when no embedding provider is configured and the workspace has
/// not recorded one yet.
const DEFAULT_EMBEDDING_DIMENSION: Dimension = Dimension::new(1024);

/// What a SQL statement would do if executed, decided by the `DuckDB` parser.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StatementKind {
    /// A `SELECT` (or an equivalent read-only statement such as `DESCRIBE`).
    Read,
    /// Anything that could mutate state, load code, or reach outside the
    /// workspace: DDL, DML, `COPY`, `ATTACH`, `SET`, `INSTALL`, `LOAD`, ...
    Write,
    /// The `DuckDB` parser rejected it; execution will report the syntax error.
    Invalid(String),
}

impl StatementKind {
    /// Whether a statement the parser accepted writes; the parser's message
    /// for one it rejected.
    ///
    /// # Errors
    ///
    /// Returns the syntax error of an `Invalid` statement.
    pub fn writes(self) -> std::result::Result<bool, String> {
        match self {
            Self::Read => Ok(false),
            Self::Write => Ok(true),
            Self::Invalid(message) => Err(message),
        }
    }
}

/// A statement as `DuckDB`'s `json_serialize_sql` returns it.
enum SerializedStatement {
    /// A `SELECT`-shaped statement, as its parse tree.
    Tree(serde_json::Value),
    /// The parser rejected it, with its message.
    SyntaxError(String),
    /// Parsed, but not a shape the serializer covers: any other statement
    /// type, or several statements.
    Unserializable,
}

impl SerializedStatement {
    /// The parse tree with every constant blanked and the source offsets
    /// dropped, so two statements that differ only in the values they
    /// filter on come out the same.
    fn shape(self) -> Option<String> {
        let Self::Tree(mut tree) = self else {
            return None;
        };
        blank_constants(&mut tree);
        Some(tree.to_string())
    }

    /// The names of the tables the statement reads.
    fn table_names(&self) -> Option<Vec<String>> {
        let Self::Tree(tree) = self else {
            return None;
        };
        let mut names = Vec::new();
        collect_table_names(tree, &mut names);
        Some(names)
    }
}

/// Which way a result is sorted. Nulls go last either way, so a column
/// with gaps shows its values first in both directions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Deserialize, serde::Serialize)]
#[serde(rename_all = "lowercase")]
pub enum SortDirection {
    Asc,
    Desc,
}

impl SortDirection {
    /// The other direction: what clicking a sorted column asks for next.
    #[must_use]
    pub const fn flipped(self) -> Self {
        match self {
            Self::Asc => Self::Desc,
            Self::Desc => Self::Asc,
        }
    }
}

/// A sort on a query's results: the column by its 1-based position, so a
/// name that needs quoting, or two columns sharing one, sorts the same.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResultSort {
    pub column: std::num::NonZeroUsize,
    pub direction: SortDirection,
}

/// A single `SELECT`-shaped statement's parse tree, which a sort rewrites:
/// its own `ORDER BY` is set, and `DuckDB` prints the statement back.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Sortable(serde_json::Value);

/// How a rewritten `ORDER BY` names its column.
enum OrderKey<'a> {
    Name(&'a str),
    Position(std::num::NonZeroUsize),
}

impl OrderKey<'_> {
    fn expression(&self) -> serde_json::Value {
        match self {
            Self::Name(name) => serde_json::json!({
                "class": "COLUMN_REF",
                "type": "COLUMN_REF",
                "alias": "",
                "column_names": [name],
            }),
            Self::Position(position) => serde_json::json!({
                "class": "CONSTANT",
                "type": "VALUE_CONSTANT",
                "alias": "",
                "value": {
                    "type": { "id": "BIGINT", "type_info": null },
                    "is_null": false,
                    "value": position.get(),
                },
            }),
        }
    }
}

impl Sortable {
    /// The top-level query's modifiers (`ORDER BY`, `LIMIT`, `DISTINCT`).
    fn modifiers(tree: &mut serde_json::Value) -> Option<&mut Vec<serde_json::Value>> {
        tree.get_mut("statements")?
            .get_mut(0)?
            .get_mut("node")?
            .get_mut("modifiers")?
            .as_array_mut()
    }

    /// The tree with its `ORDER BY` replaced by `sort` on `key`, placed
    /// before any `LIMIT`, so the rows are sorted and then cut, as the
    /// printed SQL reads.
    fn ordered_by(&self, sort: ResultSort, key: &OrderKey<'_>) -> serde_json::Value {
        let mut tree = self.0.clone();
        let order = serde_json::json!({
            "type": "ORDER_MODIFIER",
            "orders": [{
                "type": match sort.direction {
                    SortDirection::Asc => "ASCENDING",
                    SortDirection::Desc => "DESCENDING",
                },
                "null_order": "NULLS LAST",
                "expression": key.expression(),
            }],
        });
        if let Some(modifiers) = Self::modifiers(&mut tree) {
            let kind = |m: &serde_json::Value| {
                m.get("type")
                    .and_then(serde_json::Value::as_str)
                    .map(str::to_owned)
            };
            if let Some(existing) = modifiers
                .iter_mut()
                .find(|m| kind(m).as_deref() == Some("ORDER_MODIFIER"))
            {
                *existing = order;
            } else {
                let at = modifiers
                    .iter()
                    .position(|m| kind(m).as_deref() == Some("LIMIT_MODIFIER"))
                    .unwrap_or(modifiers.len());
                modifiers.insert(at, order);
            }
        }
        tree
    }
}

/// Message returned to the caller when a statement would create a temp
/// table or view.
pub const TEMP_OBJECT_REFUSED: &str = "Temporary tables and views are not visible to every reader \
     for the rest of this workspace's session; create a regular table instead \
     (CREATE TABLE, without TEMP or TEMPORARY).";

/// Whether `sql` is a `CREATE [OR REPLACE] {TEMP | TEMPORARY} ...`
/// statement. `DuckDB` temp objects are connection-local, so one created
/// on the writer would be invisible to every reader-routed tool for the
/// rest of the workspace handle's life (they run on other connections);
/// every path that can run a write (the agent's `run_sql`, the REST and
/// MCP `sql` handlers) refuses these outright with this as the friendly,
/// fail-fast message. It cannot be a complete check — a leading comment,
/// a semicolon before the real statement, or a multi-statement batch all
/// defeat a text match — so `ReaderDb::observe_write` is the actual
/// correctness backstop; this is the fast path for the obvious case.
#[must_use]
pub fn creates_temp_object(sql: &str) -> bool {
    let mut words = sql.split_whitespace().map(str::to_uppercase);
    if words.next().as_deref() != Some("CREATE") {
        return false;
    }
    let mut word = words.next();
    if word.as_deref() == Some("OR") {
        if words.next().as_deref() != Some("REPLACE") {
            return false;
        }
        word = words.next();
    }
    matches!(word.as_deref(), Some("TEMP" | "TEMPORARY"))
}

/// Leading keywords that mark terminal input as SQL to run directly rather
/// than a question for the agent.
const DIRECT_SQL_KEYWORDS: &[&str] = &[
    "SELECT",
    "WITH",
    "FROM",
    "DESCRIBE",
    "SHOW",
    "PIVOT",
    "UNPIVOT",
    "SUMMARIZE",
    "EXPLAIN",
];

/// Whether interactive input should run as SQL instead of going to the agent.
#[must_use]
pub fn looks_like_direct_sql(input: &str) -> bool {
    let word: String = input
        .trim_start()
        .chars()
        .take_while(char::is_ascii_alphabetic)
        .collect();
    !word.is_empty()
        && DIRECT_SQL_KEYWORDS
            .iter()
            .any(|k| k.eq_ignore_ascii_case(&word))
}

/// Read-only statements `DuckDB` cannot serialize to JSON but which never mutate.
const READ_ONLY_KEYWORDS: &[&str] = &[
    "DESCRIBE",
    "SHOW",
    "SUMMARIZE",
    "PIVOT",
    "UNPIVOT",
    "EXPLAIN",
];

/// How a streamed result set is written ([`WorkspaceDb::stream_query`]).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ExportFormat {
    #[default]
    Csv,
    Ndjson,
    Json,
}

text_enum!(ExportFormat, "export format", {
    Csv => "csv",
    Ndjson => "ndjson",
    Json => "json",
});

impl ExportFormat {
    /// The media type of what is written.
    #[must_use]
    pub const fn content_type(self) -> &'static str {
        match self {
            Self::Csv => "text/csv; charset=utf-8",
            Self::Ndjson => "application/x-ndjson",
            Self::Json => "application/json",
        }
    }
}

/// One row at a time in an [`ExportFormat`], the same shapes
/// [`QueryResults::write_csv`], `write_ndjson`, and `write_json` make.
enum RowWriter<'a, W: Write> {
    Csv(Box<csv::Writer<&'a mut W>>),
    Ndjson {
        out: &'a mut W,
        keys: Vec<String>,
    },
    Json {
        out: &'a mut W,
        keys: Vec<String>,
        first: bool,
    },
}

impl<'a, W: Write> RowWriter<'a, W> {
    fn start(format: ExportFormat, columns: &[String], out: &'a mut W) -> Result<Self> {
        let keys = QueryResults {
            columns: columns.to_vec(),
            rows: Vec::new(),
        }
        .json_keys();
        Ok(match format {
            ExportFormat::Csv => {
                let mut writer = csv::Writer::from_writer(out);
                writer.write_record(columns)?;
                Self::Csv(Box::new(writer))
            }
            ExportFormat::Ndjson => Self::Ndjson { out, keys },
            ExportFormat::Json => {
                out.write_all(b"[")?;
                Self::Json {
                    out,
                    keys,
                    first: true,
                }
            }
        })
    }

    fn row(&mut self, values: &[serde_json::Value]) -> Result<()> {
        match self {
            Self::Csv(writer) => {
                let cells: Vec<String> = values.iter().map(|v| Cell(v).text()).collect();
                writer.write_record(&cells)?;
            }
            Self::Ndjson { out, keys } => {
                writeln!(out, "{}", JsonRow { keys, values }.render()?)?;
            }
            Self::Json { out, keys, first } => {
                if !*first {
                    out.write_all(b",")?;
                }
                *first = false;
                write!(out, "\n{}", JsonRow { keys, values }.render()?)?;
            }
        }
        Ok(())
    }

    fn finish(self) -> Result<()> {
        match self {
            Self::Csv(mut writer) => writer.flush()?,
            Self::Ndjson { out, .. } => out.flush()?,
            Self::Json { out, .. } => {
                out.write_all(b"\n]\n")?;
                out.flush()?;
            }
        }
        Ok(())
    }
}

/// One row as a JSON object whose keys keep column order, written by hand
/// because `serde_json`'s map sorts its keys.
struct JsonRow<'a> {
    keys: &'a [String],
    values: &'a [serde_json::Value],
}

impl JsonRow<'_> {
    fn render(&self) -> Result<String> {
        let mut fields = Vec::with_capacity(self.keys.len());
        for (column, value) in self.keys.iter().zip(self.values) {
            fields.push(format!(
                "{}:{}",
                serde_json::to_string(column)?,
                serde_json::to_string(value)?
            ));
        }
        Ok(format!("{{{}}}", fields.join(",")))
    }
}

/// Query result set from a `DuckDB` workspace database.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct QueryResults {
    pub columns: Vec<String>,
    pub rows: Vec<Vec<serde_json::Value>>,
}

/// Query results with at most a caller's cap of rows kept, plus how many
/// the statement produced in all.
#[derive(Debug, Clone)]
pub struct CappedResults {
    pub results: QueryResults,
    pub total_rows: usize,
}

impl CappedResults {
    /// Whether rows were dropped to honor the cap.
    #[must_use]
    pub fn truncated(&self) -> bool {
        self.total_rows > self.results.rows.len()
    }

    /// Rows the cap dropped.
    #[must_use]
    pub fn omitted(&self) -> usize {
        self.total_rows.saturating_sub(self.results.rows.len())
    }

    /// The rows as a text table for the model, with a last row saying how
    /// many more there were and what to do instead when the cap cut it.
    ///
    /// # Errors
    ///
    /// Returns an error if formatting fails.
    pub fn to_model_text(&self) -> Result<String> {
        let mut table = self.results.clone();
        if self.truncated() {
            table.rows.push(vec![serde_json::Value::String(format!(
                "... ({} more rows not shown: the result was cut at the {}-row limit, so \
                 aggregate, filter, or ORDER BY and LIMIT it in one statement rather than \
                 re-running it per group)",
                self.omitted(),
                self.results.rows.len()
            ))]);
        }
        let mut buf = Vec::new();
        table.write_table(&mut buf)?;
        String::from_utf8(buf).map_err(|e| Error::Analysis(format!("UTF-8 error: {e}")))
    }
}

/// Query results with a digest of the whole result set (`ResultDigest`).
/// Rows past the cap are digested and counted but not kept.
#[derive(Debug, Clone)]
pub struct DigestedResults {
    pub results: CappedResults,
    pub digest: String,
}

/// A digest of a result set that does not depend on row order: the SHA-256
/// of the column names in order, then the sum modulo 2^256 of every row's
/// SHA-256, each row as one JSON array. A statement without `ORDER BY` can
/// return the same rows in a different order from one run to the next, so
/// row order must not count; a sum, unlike XOR, keeps a repeated row
/// distinct from a single one, and unlike sorting the row hashes needs no
/// memory per row.
struct ResultDigest {
    columns: digest::Context,
    /// The row-hash sum, 64-bit limbs least significant first.
    rows: [u64; 4],
}

impl ResultDigest {
    fn new(columns: &[String]) -> Result<Self> {
        let mut context = digest::Context::new(&digest::SHA256);
        context.update(&serde_json::to_vec(columns)?);
        Ok(Self {
            columns: context,
            rows: [0; 4],
        })
    }

    fn add_row(&mut self, values: &[serde_json::Value]) -> Result<()> {
        let hash = digest::digest(&digest::SHA256, &serde_json::to_vec(values)?);
        let mut carry = false;
        let (words, _) = hash.as_ref().as_chunks::<8>();
        for (limb, word) in self.rows.iter_mut().zip(words) {
            let (sum, overflowed) = limb.overflowing_add(u64::from_le_bytes(*word));
            let (sum, overflowed_again) = sum.overflowing_add(u64::from(carry));
            *limb = sum;
            carry = overflowed || overflowed_again;
        }
        Ok(())
    }

    fn finish(mut self) -> String {
        for limb in self.rows {
            self.columns.update(&limb.to_le_bytes());
        }
        crypto::hex_lower(self.columns.finish().as_ref())
    }
}

/// The ontology tables (design doc 5.4), created with the other internal
/// tables.
const ONTOLOGY_DDL: &str = "            CREATE TABLE IF NOT EXISTS _quack_ontology_versions (
                version INTEGER PRIMARY KEY,
                snapshot JSON NOT NULL,
                author TEXT,
                note TEXT,
                created_at TIMESTAMP DEFAULT now(),
                acceptance TEXT
            );
            ALTER TABLE _quack_ontology_versions ADD COLUMN IF NOT EXISTS acceptance TEXT;
            CREATE TABLE IF NOT EXISTS _quack_ontology_classes (
                id TEXT PRIMARY KEY,
                parent_id TEXT,
                label TEXT NOT NULL,
                description TEXT,
                key_property TEXT,
                since_version INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS _quack_ontology_relations (
                id TEXT PRIMARY KEY,
                label TEXT NOT NULL,
                description TEXT,
                domain_class TEXT NOT NULL,
                range_class TEXT NOT NULL,
                since_version INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS _quack_ontology_properties (
                id TEXT NOT NULL,
                class_id TEXT NOT NULL,
                label TEXT NOT NULL,
                type TEXT NOT NULL,
                enum_values JSON,
                since_version INTEGER NOT NULL,
                PRIMARY KEY (id, class_id)
            );
            ALTER TABLE _quack_ontology_properties ADD COLUMN IF NOT EXISTS description TEXT;
            ALTER TABLE _quack_ontology_properties ADD COLUMN IF NOT EXISTS unit TEXT;
            ALTER TABLE _quack_ontology_properties ADD COLUMN IF NOT EXISTS synonyms JSON;
            CREATE TABLE IF NOT EXISTS _quack_ontology_measures (
                id TEXT PRIMARY KEY,
                description TEXT,
                table_name TEXT NOT NULL,
                expression TEXT NOT NULL,
                since_version INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS _quack_ontology_mappings (
                id TEXT PRIMARY KEY,
                table_name TEXT NOT NULL,
                class_id TEXT NOT NULL,
                key_column TEXT NOT NULL,
                property_map JSON NOT NULL,
                relation_map JSON NOT NULL,
                since_version INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS _quack_ontology_candidates (
                id TEXT PRIMARY KEY,
                kind TEXT NOT NULL,
                proposal JSON NOT NULL,
                evidence JSON NOT NULL,
                confidence FLOAT NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending',
                proposed_by TEXT NOT NULL,
                decided_by TEXT,
                decided_at TIMESTAMP
            );";

/// The embedding side of a workspace, shared by every connection to it:
/// the width of its vector columns, which a refresh can change, and the
/// profile the configured model runs under.
#[derive(Debug)]
struct Vectors {
    /// Width of `_quack_chunks.embedding` and `_quack_graph_nodes.embedding`.
    column_dimension: AtomicU32,
    /// `None` without an embedding model.
    profile: Option<Profile>,
    /// `profile`'s fingerprint, stored beside each vector made under it.
    fingerprint: Option<Fingerprint>,
}

impl Vectors {
    fn new(column_dimension: Dimension, profile: Option<Profile>) -> Arc<Self> {
        let fingerprint = profile.as_ref().map(Profile::fingerprint);
        Arc::new(Self {
            column_dimension: AtomicU32::new(column_dimension.get()),
            profile,
            fingerprint,
        })
    }
}

/// A pinned document with its full text, chunks joined in order.
#[derive(Debug, Clone)]
pub struct PinnedDocument {
    pub document: DocumentInfo,
    pub text: String,
}

/// Whether a document is sent to the model in full on every turn.
/// Serializes as the `pinned` boolean.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(from = "bool", into = "bool")]
pub enum Pinning {
    Unpinned,
    /// Injected whole, within `[retrieval].pinned_token_budget`.
    Pinned,
}

flag_enum!(Pinning, false => Unpinned, true => Pinned);

/// Wraps a `DuckDB` connection for a single workspace.
pub struct WorkspaceDb {
    conn: duckdb::Connection,
    vectors: Arc<Vectors>,
    query_timeout: Duration,
    /// `files/` under the workspace directory, where ingested files are
    /// kept; `None` in memory.
    files_dir: Option<PathBuf>,
    /// `[retrieval].languages`: what a document may be detected as.
    languages: LanguageSetting,
}

impl WorkspaceDb {
    /// The schema version this build writes.
    #[must_use]
    pub const fn schema_version() -> u32 {
        WORKSPACE_SCHEMA_VERSION
    }

    /// Open an in-memory `DuckDB` database (for tests).
    ///
    /// # Errors
    ///
    /// Returns an error if the database cannot be created.
    pub fn open_in_memory(embedding_dimension: Dimension) -> Result<Self> {
        Self::in_memory(Vectors::new(embedding_dimension, None))
    }

    /// An in-memory database whose vectors are made under `profile` (for
    /// tests).
    ///
    /// # Errors
    ///
    /// Returns an error if the database cannot be created.
    pub fn open_in_memory_with_profile(profile: Profile) -> Result<Self> {
        let db = Self::in_memory(Vectors::new(profile.dimension, Some(profile)))?;
        db.record_profile()?;
        Ok(db)
    }

    fn in_memory(vectors: Arc<Vectors>) -> Result<Self> {
        let conn = duckdb::Connection::open_in_memory()?;
        let db = Self {
            conn,
            vectors,
            query_timeout: Duration::from_secs(30),
            files_dir: None,
            languages: LanguageSetting::Auto,
        };
        db.confine_to(None)?;
        db.create_internal_tables()?;
        Ok(db)
    }

    /// Detect documents among `languages` from now on (`[retrieval].languages`).
    #[must_use]
    pub fn with_languages(mut self, languages: LanguageSetting) -> Self {
        self.languages = languages;
        self
    }

    /// Override the per-statement timeout (tests and callers with special needs).
    #[must_use]
    pub fn with_query_timeout(mut self, timeout: Duration) -> Self {
        self.query_timeout = timeout;
        self
    }

    /// A `_quack_meta` value as a workspace file recorded it, before anything
    /// else runs on it: `None` for a new file or a key it never set.
    fn recorded(conn: &duckdb::Connection, key: MetaKey) -> Result<Option<String>> {
        let has_meta: bool = conn.query_row(
        "SELECT count(*) > 0 FROM duckdb_tables() WHERE table_name = '_quack_meta' AND NOT temporary",
        [],
        |row| row.get(0),
    )?;
        if !has_meta {
            return Ok(None);
        }
        let value: Option<String> = conn
            .query_row(
                "SELECT value FROM _quack_meta WHERE key = ?",
                duckdb::params![key],
                |row| row.get(0),
            )
            .map(Some)
            .or_else(|e| match e {
                duckdb::Error::QueryReturnedNoRows => Ok(None),
                other => Err(other),
            })?;
        Ok(value)
    }

    /// The vector width a workspace file recorded: `None` for a new file.
    fn recorded_dimension(conn: &duckdb::Connection) -> Result<Option<Dimension>> {
        Ok(Self::recorded(conn, MetaKey::EmbeddingDimension)?
            .and_then(|v| v.parse().ok())
            .map(Dimension::new))
    }

    /// Refuse a file a newer quack upgraded, before any statement changes
    /// it: this binary's table definitions and rebuilds do not know that
    /// schema, and recording its own lower version would hide the rollback.
    fn refuse_newer_schema(conn: &duckdb::Connection, path: &Path) -> Result<()> {
        // A file from before the version was recorded has none.
        let recorded = match Self::recorded(conn, MetaKey::SchemaVersion)? {
            None => 0,
            Some(text) => text
                .parse::<u32>()
                .map_err(|_| Error::WorkspaceSchemaUnreadable {
                    path: path.to_path_buf(),
                    recorded: text,
                })?,
        };
        if recorded <= WORKSPACE_SCHEMA_VERSION {
            return Ok(());
        }
        Err(Error::WorkspaceTooNew {
            path: path.to_path_buf(),
            recorded,
            supported: WORKSPACE_SCHEMA_VERSION,
            written_by: WrittenBy(Self::recorded(conn, MetaKey::WrittenByQuack)?),
        })
    }

    /// Open (or create) the `DuckDB` database for a workspace.
    ///
    /// Creates the `_quack_` internal tables if they do not exist, and
    /// reconciles the vector width recorded in `_quack_meta` with the
    /// configured embedding profile.
    ///
    /// # Errors
    ///
    /// Returns an error if the database file cannot be created or opened.
    pub fn open(config: &Config, workspace_id: &str) -> Result<Self> {
        let db_path = config.workspace_db_path(workspace_id);
        if let Some(parent) = db_path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        let files_dir = config.workspace_files_dir(workspace_id);
        std::fs::create_dir_all(&files_dir)?;

        let profile = Profile::from_config(config)?;
        // DuckDB reports a file held by another process only in its message
        // text ("Could not set lock on file ..."); it is classified here, once,
        // so callers match a variant instead.
        let storage = duckdb::Config::default().with(
            "storage_compatibility_version",
            STORAGE_COMPATIBILITY_VERSION,
        )?;
        let conn = duckdb::Connection::open_with_flags(&db_path, storage).map_err(|e| {
            if e.to_string().contains("Could not set lock") {
                Error::WorkspaceLocked {
                    path: db_path.clone(),
                }
            } else {
                Error::from(e)
            }
        })?;
        Self::refuse_newer_schema(&conn, &db_path)?;
        // The columns keep the width they were created with until the
        // reconciliation below decides otherwise.
        let column_dimension = Self::recorded_dimension(&conn)?
            .or_else(|| profile.as_ref().map(|p| p.dimension))
            .unwrap_or(DEFAULT_EMBEDDING_DIMENSION);

        let db = Self {
            conn,
            vectors: Vectors::new(column_dimension, profile),
            query_timeout: config.analysis.query_timeout(),
            files_dir: Some(files_dir),
            languages: config.retrieval.languages.clone(),
        };
        db.apply_resource_limits(config)?;
        let workspace_dir = std::fs::canonicalize(config.workspace_dir(workspace_id))?;
        db.confine_to(Some(&workspace_dir))?;
        db.create_internal_tables()?;
        db.reconcile_dimension()?;
        db.record_profile()?;
        Ok(db)
    }

    /// Open a second, reader connection to the same database:
    /// `duckdb::Connection::try_clone` opens a new connection against the
    /// already-open `DatabaseInstance`, so it succeeds even though
    /// `lock_configuration` is set — that lock freezes the *configuration*
    /// (`allowed_directories`, resource limits, ...), which is shared by
    /// every connection to the instance, not the ability to open more of
    /// them. `Connection::open` is not an option here: it would create a
    /// second `DatabaseInstance` and take the file's exclusive lock.
    ///
    /// The clone inherits the confinement and resource limits already
    /// locked in on `self`, so it must never call `confine_to` or
    /// `apply_resource_limits` again — both `SET`s would fail once
    /// the configuration is locked.
    ///
    /// # Errors
    ///
    /// Returns an error if `DuckDB` cannot open the new connection.
    pub fn try_clone_reader(&self) -> Result<Self> {
        Ok(Self {
            conn: self.conn.try_clone()?,
            vectors: Arc::clone(&self.vectors),
            query_timeout: self.query_timeout,
            files_dir: self.files_dir.clone(),
            languages: self.languages.clone(),
        })
    }

    /// Write everything committed into the database file, so a copy of the
    /// file once every connection has closed holds it all.
    ///
    /// # Errors
    ///
    /// Returns an error if the checkpoint fails.
    pub fn checkpoint(&self) -> Result<()> {
        self.conn.execute_batch("CHECKPOINT")?;
        Ok(())
    }

    /// Whether this connection has any temp tables: only ever the CLI's
    /// piped-stdin table, `ingestion::STDIN_TABLE`, loaded before a
    /// workspace handle's first turn — the agent itself is refused any
    /// statement that would create one (the agent's `SqlGate`),
    /// so none can appear later. `DuckDB` temp tables are connection-local,
    /// so a [`Self::try_clone_reader`] clone would not see one: a caller
    /// building a reader for a workspace handle's lifetime
    /// (`ReaderDb::open`) checks this once, at open, and
    /// reuses the writer instead when it's true.
    ///
    /// # Errors
    ///
    /// Returns an error if the catalog query fails.
    pub fn has_temp_tables(&self) -> Result<bool> {
        let n: i64 = self.conn.query_row(
            "SELECT count(*) FROM duckdb_tables() WHERE temporary",
            [],
            |row| row.get(0),
        )?;
        Ok(n > 0)
    }

    /// Cap memory and parallelism for every statement on this connection.
    fn apply_resource_limits(&self, config: &Config) -> Result<()> {
        let memory_limit = format!("{}MB", config.analysis.memory_limit_mb);
        self.conn
            .execute("SET memory_limit = ?", duckdb::params![memory_limit])?;
        self.conn.execute(
            "SET threads = ?",
            duckdb::params![i64::from(config.analysis.threads.max(1))],
        )?;
        Ok(())
    }

    /// Confine every statement on this connection to the workspace (design
    /// doc 7.4). `DuckDB`'s file readers, replacement scans (`FROM 'x.csv'`),
    /// `COPY`, `ATTACH`, `INSTALL`, and `LOAD` may touch nothing outside
    /// `allowed_dir` (the workspace directory, whose `files/` holds the
    /// ingested originals), secrets never persist, and the configuration is
    /// then locked so no later statement, agent-written or user-typed, can
    /// widen it or lift the resource limits. Read classification alone does
    /// not cover this: `SELECT * FROM read_text('/etc/passwd')` is a read.
    ///
    /// The allow-list must be set while external access is still enabled,
    /// and the lock must come last; `DuckDB` refuses both in any other order.
    fn confine_to(&self, allowed_dir: Option<&Path>) -> Result<()> {
        match allowed_dir {
            Some(dir) => {
                let dir = dir.to_string_lossy();
                self.conn.execute(
                    "SET allowed_directories = [?]",
                    duckdb::params![dir.as_ref()],
                )?;
            }
            None => {
                self.conn.execute("SET allowed_directories = []", [])?;
            }
        }
        // Errors as JSON, for every connection to the database (the reader
        // and audit clones included): it keeps the names `DuckDB` suggests
        // apart from the message, so `DuckDbMessage` can leave quack's
        // internal tables out of them.
        self.conn.execute_batch(
            "SET enable_external_access = false;\n\
             SET allow_persistent_secrets = false;\n\
             SET GLOBAL errors_as_json = true;\n\
             SET lock_configuration = true;",
        )?;
        Ok(())
    }

    /// Classify a statement as read, write, or invalid using `DuckDB`'s parser.
    ///
    /// `json_serialize_sql` succeeds only for `SELECT`-shaped statements; a
    /// "not implemented" error means some other statement type (or several
    /// statements) and is treated as a write. A short allowlist of read-only
    /// keywords (`DESCRIBE`, `SHOW`, `SUMMARIZE`, `PIVOT`, `UNPIVOT`,
    /// `EXPLAIN`) covers statements `DuckDB` cannot serialize but which never
    /// mutate, and only when the input is a single statement.
    ///
    /// # Errors
    ///
    /// Returns an error if the classification query itself fails.
    pub fn classify_statement(&self, sql: &str) -> Result<StatementKind> {
        Ok(match self.serialize(sql)? {
            SerializedStatement::Tree(tree) => {
                let statement_count = tree
                    .get("statements")
                    .and_then(serde_json::Value::as_array)
                    .map_or(0, Vec::len);
                if statement_count == 0 {
                    StatementKind::Invalid(String::from("empty statement"))
                } else {
                    StatementKind::Read
                }
            }
            SerializedStatement::SyntaxError(message) => StatementKind::Invalid(message),
            SerializedStatement::Unserializable if is_single_read_only_statement(sql) => {
                StatementKind::Read
            }
            SerializedStatement::Unserializable => StatementKind::Write,
        })
    }

    /// `sql` in a form that can be sorted, when it is exactly one
    /// `SELECT`-shaped statement; `None` for anything else (a write,
    /// several statements), which has no rows of its own to reorder.
    ///
    /// # Errors
    ///
    /// Returns an error if the parse fails.
    pub fn sortable(&self, sql: &str) -> Result<Option<Sortable>> {
        let SerializedStatement::Tree(mut tree) = self.serialize(sql)? else {
            return Ok(None);
        };
        let single = tree
            .get("statements")
            .and_then(serde_json::Value::as_array)
            .is_some_and(|statements| statements.len() == 1);
        if !single || Sortable::modifiers(&mut tree).is_none() {
            return Ok(None);
        }
        Ok(Some(Sortable(tree)))
    }

    /// The statement with its own `ORDER BY` set to `sort`, replacing any
    /// it had, as `DuckDB` prints it. The column is named when its name is
    /// unique among the results and `DuckDB` resolves it, which reads
    /// better in the editor; otherwise it is given by position, which
    /// always resolves (an unnamed expression, two columns sharing a name).
    ///
    /// # Errors
    ///
    /// Returns an error if printing or planning the statement fails.
    pub fn sort_statement(&self, sortable: &Sortable, sort: ResultSort) -> Result<String> {
        let names = self.result_columns(&self.print(&sortable.0)?)?;
        let index = sort.column.get().saturating_sub(1);
        if let Some(name) = names.get(index)
            && names.iter().filter(|n| *n == name).count() == 1
        {
            let named = self.print(&sortable.ordered_by(sort, &OrderKey::Name(name)))?;
            if self.result_columns(&named).is_ok() {
                return Ok(named);
            }
        }
        self.print(&sortable.ordered_by(sort, &OrderKey::Position(sort.column)))
    }

    /// A parse tree printed back as SQL by `DuckDB`.
    fn print(&self, tree: &serde_json::Value) -> Result<String> {
        Ok(self.conn.query_row(
            "SELECT json_deserialize_sql(?::JSON)",
            duckdb::params![tree.to_string()],
            |row| row.get(0),
        )?)
    }

    /// The result columns `sql` would return, from planning it with no rows.
    fn result_columns(&self, sql: &str) -> Result<Vec<String>> {
        Ok(self
            .read_rows(&format!("SELECT * FROM ({sql}) LIMIT 0"), Some(0))?
            .results
            .columns)
    }

    /// `sql` as `DuckDB`'s own parser sees it.
    fn serialize(&self, sql: &str) -> Result<SerializedStatement> {
        let serialized: String = self.conn.query_row(
            "SELECT json_serialize_sql(?::VARCHAR)",
            duckdb::params![sql],
            |row| row.get(0),
        )?;
        let parsed: serde_json::Value = serde_json::from_str(&serialized)?;
        let is_error = parsed
            .get("error")
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(true);
        if !is_error {
            return Ok(SerializedStatement::Tree(parsed));
        }
        let error_type = parsed.get("error_type").and_then(serde_json::Value::as_str);
        Ok(if error_type == Some("parser") {
            SerializedStatement::SyntaxError(
                parsed
                    .get("error_message")
                    .and_then(serde_json::Value::as_str)
                    .unwrap_or("syntax error")
                    .to_owned(),
            )
        } else {
            SerializedStatement::Unserializable
        })
    }

    /// The statement's parse tree with every constant blanked and the
    /// source offsets dropped, so two statements that differ only in the
    /// values they filter on serialize the same; `None` when `DuckDB`
    /// cannot serialize the statement (anything but a `SELECT` shape).
    ///
    /// # Errors
    ///
    /// Returns an error if the serialization query itself fails.
    pub fn statement_shape(&self, sql: &str) -> Result<Option<String>> {
        Ok(self.serialize(sql)?.shape())
    }

    /// Names of tables a statement references, as `DuckDB` parsed them, or
    /// `None` when `DuckDB` cannot serialize the statement.
    fn referenced_base_tables(&self, sql: &str) -> Result<Option<Vec<String>>> {
        Ok(self.serialize(sql)?.table_names())
    }

    /// Classify a statement a user or client wrote: `_quack_` tables are
    /// refused outright, then the statement is read, write, or invalid.
    ///
    /// # Errors
    ///
    /// Returns `Error::Analysis` when the statement reaches an internal
    /// table, or a storage error when classification fails.
    pub fn classify_user_statement(&self, sql: &str) -> Result<StatementKind> {
        if self.references_internal_table(sql)? {
            return Err(Error::Analysis(String::from(
                "internal tables are not accessible",
            )));
        }
        let kind = self.classify_statement(sql)?;
        if graph::views::write_names_reserved(sql, &kind) {
            return Err(Error::Analysis(String::from(
                graph::views::RESERVED_REFUSED,
            )));
        }
        Ok(kind)
    }

    /// Whether a statement touches any of quack's internal tables.
    ///
    /// Uses the parsed table references when `DuckDB` can serialize the
    /// statement, so string literals do not count; falls back to a
    /// conservative token scan for everything else.
    ///
    /// # Errors
    ///
    /// Returns an error if the classification query fails.
    pub fn references_internal_table(&self, sql: &str) -> Result<bool> {
        match self.referenced_base_tables(sql)? {
            Some(names) => Ok(names.iter().any(|n| is_internal_name(n))),
            None => Ok(mentions_internal_table_token(sql)),
        }
    }

    fn create_internal_tables(&self) -> Result<()> {
        self.rename_legacy_tables()?;
        let dim = self.embedding_dimension();
        let sql = format!(
            "CREATE TABLE IF NOT EXISTS _quack_meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS _quack_chunks (
                id TEXT PRIMARY KEY,
                document_id TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                content TEXT NOT NULL,
                heading TEXT,
                page INTEGER,
                embedding FLOAT[{dim}],
                token_count INTEGER
            );
            ALTER TABLE _quack_chunks ADD COLUMN IF NOT EXISTS heading TEXT;
            ALTER TABLE _quack_chunks ADD COLUMN IF NOT EXISTS page INTEGER;
            ALTER TABLE _quack_chunks ADD COLUMN IF NOT EXISTS embedding_profile TEXT;
            ALTER TABLE _quack_chunks ADD COLUMN IF NOT EXISTS kind TEXT;
            ALTER TABLE _quack_chunks ADD COLUMN IF NOT EXISTS locator TEXT;
            CREATE TABLE IF NOT EXISTS _quack_embedding_profiles (
                fingerprint TEXT PRIMARY KEY,
                profile JSON NOT NULL,
                first_used TIMESTAMP DEFAULT now()
            );
            CREATE TABLE IF NOT EXISTS _quack_terms (
                chunk_id TEXT NOT NULL,
                term TEXT NOT NULL,
                tf INTEGER NOT NULL
            );
            CREATE INDEX IF NOT EXISTS _quack_terms_term_idx ON _quack_terms (term);
            CREATE INDEX IF NOT EXISTS _quack_terms_chunk_idx ON _quack_terms (chunk_id);
            DROP SCHEMA IF EXISTS fts_main__quack_chunks CASCADE;
            CREATE TABLE IF NOT EXISTS _quack_context (
                version INTEGER PRIMARY KEY,
                content TEXT NOT NULL,
                edited_by TEXT,
                edited_at TIMESTAMP DEFAULT now()
            );
            CREATE TABLE IF NOT EXISTS _quack_sessions (
                id TEXT PRIMARY KEY,
                title TEXT,
                mode TEXT NOT NULL DEFAULT 'chat',
                model TEXT NOT NULL,
                created_by TEXT,
                shared BOOLEAN NOT NULL DEFAULT false,
                created_at TIMESTAMP DEFAULT now(),
                updated_at TIMESTAMP DEFAULT now()
            );
            CREATE TABLE IF NOT EXISTS _quack_messages (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                seq INTEGER NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                metadata JSON,
                created_at TIMESTAMP DEFAULT now(),
                UNIQUE (session_id, seq)
            );
            CREATE TABLE IF NOT EXISTS _quack_audit (
                id TEXT PRIMARY KEY,
                timestamp TIMESTAMP DEFAULT now(),
                user_id TEXT,
                action TEXT NOT NULL,
                detail JSON
            );"
        );
        self.conn.execute_batch(DOCUMENTS_DDL)?;
        self.conn.execute_batch(&sql)?;
        self.conn.execute_batch(SESSION_SUMMARIES_DDL)?;
        self.conn.execute_batch(saved::DDL)?;
        self.conn.execute_batch(ONTOLOGY_DDL)?;
        self.conn.execute_batch(&graph::ddl(dim))?;
        self.conn.execute_batch(profile::DDL)?;
        self.conn.execute_batch(table_search::DDL)?;
        self.upgrade_data(dim)?;
        if let Some(ontology) = ontology_store::current(self)? {
            graph::views::ensure(self, &ontology)?;
        }
        self.set_meta(MetaKey::EmbeddingDimension, &dim.to_string())?;
        self.set_meta(MetaKey::WrittenByQuack, env!("CARGO_PKG_VERSION"))?;
        self.set_meta(MetaKey::WrittenByDuckDb, &self.duckdb_version()?)?;
        // DuckDB cannot replay an `ADD COLUMN` from the write-ahead log (an
        // internal error on the next open), so a column added to an older
        // file goes into the database file before anything else runs.
        self.conn.execute_batch("CHECKPOINT")?;
        Ok(())
    }

    /// The data rebuilds a schema version asks of a workspace recorded
    /// under an older one, then the version it now matches. A file recorded
    /// under a newer one never gets here: [`Self::open`] refuses it first.
    fn upgrade_data(&self, dim: Dimension) -> Result<()> {
        let recorded = self
            .meta(MetaKey::SchemaVersion)?
            .and_then(|v| v.parse::<u32>().ok())
            .unwrap_or(0);
        // Version 4 introduced the term index, version 6 stemmed it, version
        // 7 added joined identifiers, and version 12 stems each document
        // under its own language: every one of them rebuilds the index.
        if recorded < DOCUMENT_LANGUAGES && self.chunk_count()? > 0 {
            tracing::info!("detecting document languages and indexing chunks for keyword search");
            self.detect_missing_languages()?;
            self.reindex_terms()?;
        }
        // Vectors made before profiles went to the model unprefixed, under
        // the model the workspace last recorded.
        if recorded < VECTOR_PROFILES {
            self.tag_legacy_vectors(dim)?;
        }
        // Rows written before every insert named its status took the
        // column's old default, `pending`, and were never processed.
        if recorded < DOCUMENT_STATUSES {
            self.conn.execute(
                "UPDATE _quack_documents SET status = ?, \
                 error_message = 'never finished processing; upload it again' \
                 WHERE status IS NULL OR status = 'pending'",
                duckdb::params![DocumentStatus::Error],
            )?;
        }
        // Before its own column, `--auto-accept` said so only in the note.
        if recorded < ONTOLOGY_ACCEPTANCE {
            self.conn.execute(
                "UPDATE _quack_ontology_versions SET acceptance = ? WHERE note LIKE 'auto-accepted%'",
                duckdb::params![Acceptance::Auto],
            )?;
        }
        // Before symmetric dedup, a node pair whose provenance flipped
        // between resolution passes could land in `_quack_graph_merges`
        // twice, once in each orientation. Collapse each pair to one row,
        // keeping the more-decided one (a reviewer's rejection survives),
        // else the earliest, so the queue no longer lists the same pair
        // twice and a rejected pair stays rejected.
        if recorded < MERGE_DEDUP {
            self.collapse_duplicate_merge_proposals()?;
        }
        if recorded < TABLE_PROFILES {
            let profiled = TableProfile::refresh_stale(self)?;
            if profiled > 0 {
                tracing::info!(tables = profiled, "profiled existing tables");
            }
        }
        self.set_meta(
            MetaKey::SchemaVersion,
            &WORKSPACE_SCHEMA_VERSION.to_string(),
        )
    }

    /// Collapse opposing-orientation duplicates in `_quack_graph_merges` to
    /// a single row per unordered node pair. Before symmetric dedup, a pair
    /// whose provenance flipped between resolution passes could be proposed
    /// twice — once as `(keep=A, drop=B)` and once as `(keep=B, drop=A)` —
    /// and an accepted merge of one orientation left the other dangling. Of
    /// each pair, keep the more-decided row (`accepted` over `rejected` over
    /// `superseded` over `pending`), breaking ties by earliest decision then
    /// earliest id, and drop the rest. Idempotent.
    fn collapse_duplicate_merge_proposals(&self) -> Result<()> {
        // Pick the surplus rows (the loser of each pair) first; deleting
        // while reading the same table in one statement is undefined.
        let mut stmt = self.conn.prepare(
            "SELECT id FROM ( \
                SELECT id, ROW_NUMBER() OVER ( \
                    PARTITION BY least(keep_node_id, drop_node_id), \
                                 greatest(keep_node_id, drop_node_id) \
                    ORDER BY CASE status \
                               WHEN 'accepted' THEN 0 \
                               WHEN 'rejected' THEN 1 \
                               WHEN 'superseded' THEN 2 \
                               ELSE 3 \
                             END, \
                             decided_at NULLS LAST, id \
                  ) AS rn \
                FROM _quack_graph_merges \
             ) WHERE rn > 1",
        )?;
        let mut rows = stmt.query([])?;
        let mut ids: Vec<String> = Vec::new();
        while let Some(row) = rows.next()? {
            ids.push(row.get(0)?);
        }
        drop(rows);
        drop(stmt);
        // The review queue is small, so a row-at-a-time delete is fine and
        // sidesteps any self-reference or bind-count limits.
        for id in ids {
            self.conn.execute(
                "DELETE FROM _quack_graph_merges WHERE id = ?",
                duckdb::params![id],
            )?;
        }
        Ok(())
    }
    fn tag_legacy_vectors(&self, dim: Dimension) -> Result<()> {
        let mut untagged: i64 = 0;
        for table in VectorTable::ALL {
            let count: i64 = self.conn.query_row(
                &format!(
                    "SELECT count(*) FROM {table} \
                     WHERE embedding IS NOT NULL AND embedding_profile IS NULL"
                ),
                [],
                |row| row.get(0),
            )?;
            untagged = untagged.saturating_add(count);
        }
        let model = self.meta(MetaKey::EmbeddingModel)?;
        // The profile table replaces this key.
        self.conn.execute(
            "DELETE FROM _quack_meta WHERE key = ?",
            duckdb::params![MetaKey::EmbeddingModel],
        )?;
        if untagged == 0 {
            return Ok(());
        }
        let model = model.unwrap_or_else(|| String::from("unknown"));
        let legacy = Profile::new(&model, dim, Prompts::default());
        self.insert_profile(&legacy)?;
        let fingerprint = legacy.fingerprint();
        for table in VectorTable::ALL {
            self.conn.execute(
                &format!(
                    "UPDATE {table} SET embedding_profile = ? \
                     WHERE embedding IS NOT NULL AND embedding_profile IS NULL"
                ),
                duckdb::params![fingerprint],
            )?;
        }
        tracing::info!(
            profile = %legacy,
            vectors = untagged,
            "recorded the profile of existing vectors"
        );
        Ok(())
    }

    fn insert_profile(&self, profile: &Profile) -> Result<()> {
        let json = serde_json::to_string(profile)?;
        self.conn.execute(
            "INSERT OR IGNORE INTO _quack_embedding_profiles (fingerprint, profile) VALUES (?, ?)",
            duckdb::params![profile.fingerprint(), json],
        )?;
        Ok(())
    }

    /// Record the current profile, so every stored fingerprint can be
    /// described.
    fn record_profile(&self) -> Result<()> {
        match &self.vectors.profile {
            Some(profile) => self.insert_profile(profile),
            None => Ok(()),
        }
    }

    /// Workspaces created before the `_quack_` prefix keep their data.
    fn rename_legacy_tables(&self) -> Result<()> {
        for (old, new) in [
            ("documents", "_quack_documents"),
            ("chunks", "_quack_chunks"),
        ] {
            let old_exists = self.table_exists(old)?;
            let new_exists = self.table_exists(new)?;
            if old_exists && !new_exists {
                self.conn
                    .execute(&format!("ALTER TABLE {old} RENAME TO {new}"), [])?;
                tracing::info!(old, new, "renamed legacy internal table");
            }
        }
        Ok(())
    }

    fn table_exists(&self, name: &str) -> Result<bool> {
        let count: i64 = self.conn.query_row(
            "SELECT count(*) FROM information_schema.tables WHERE table_schema = 'main' AND table_name = ?",
            duckdb::params![name],
            |row| row.get(0),
        )?;
        Ok(count > 0)
    }

    /// Read a `_quack_meta` value, if the table and key exist.
    ///
    /// # Errors
    ///
    /// Returns an error if the query fails.
    pub fn meta(&self, key: MetaKey) -> Result<Option<String>> {
        if !self.table_exists("_quack_meta")? {
            return Ok(None);
        }
        let mut stmt = self
            .conn
            .prepare("SELECT value FROM _quack_meta WHERE key = ?")?;
        let mut rows = stmt.query(duckdb::params![key])?;
        match rows.next()? {
            Some(row) => Ok(Some(row.get(0)?)),
            None => Ok(None),
        }
    }

    /// Write a `_quack_meta` entry.
    ///
    /// # Errors
    ///
    /// Returns an error if the write fails.
    pub fn set_meta(&self, key: MetaKey, value: &str) -> Result<()> {
        self.conn.execute(
            "INSERT OR REPLACE INTO _quack_meta (key, value) VALUES (?, ?)",
            duckdb::params![key, value],
        )?;
        Ok(())
    }

    /// Remove a `_quack_meta` entry.
    ///
    /// # Errors
    ///
    /// Returns an error if the delete fails.
    pub fn delete_meta(&self, key: MetaKey) -> Result<()> {
        self.conn.execute(
            "DELETE FROM _quack_meta WHERE key = ?",
            duckdb::params![key],
        )?;
        Ok(())
    }

    /// The width of the stored vectors: of the vector columns, which a
    /// refresh can change.
    #[must_use]
    pub fn embedding_dimension(&self) -> Dimension {
        Dimension::new(self.vectors.column_dimension.load(Ordering::Acquire))
    }

    /// Bring the vector columns to the configured profile's width when no
    /// chunk vector would be lost doing it. When chunk vectors of another
    /// width exist they stay as they are, unsearched, until `refresh`
    /// retypes the columns: a mistyped `[embedding].dimension` must not
    /// discard a workspace's embeddings.
    fn reconcile_dimension(&self) -> Result<()> {
        let Some(profile) = &self.vectors.profile else {
            return Ok(());
        };
        let column = self.embedding_dimension();
        if profile.dimension == column {
            return Ok(());
        }
        let stored: i64 = self.conn.query_row(
            "SELECT count(*) FROM _quack_chunks WHERE embedding IS NOT NULL",
            [],
            |row| row.get(0),
        )?;
        if stored > 0 {
            tracing::warn!(
                stored = %column,
                configured = %profile.dimension,
                "the workspace's vectors are a different width from the configured model's; \
                 vector search is off until `quack embeddings refresh` updates them"
            );
            return Ok(());
        }
        tracing::info!(
            from = %column,
            to = %profile.dimension,
            "no chunk embeddings stored; adopting the configured embedding dimension"
        );
        self.retype_vectors(profile.dimension)
    }

    /// Change the vector columns' width. Every stored vector is dropped:
    /// `DuckDB` cannot cast a vector to another width, not even a NULL one,
    /// so the columns are retyped through NULL. Chunks and their term
    /// index stay; node label vectors are recomputed by the next pass.
    ///
    /// # Errors
    ///
    /// Returns an error if a statement fails.
    pub fn retype_vectors(&self, dimension: Dimension) -> Result<()> {
        for table in VectorTable::ALL {
            if self.table_exists(table.as_str())? {
                self.conn.execute_batch(&format!(
                    "ALTER TABLE {table} ALTER embedding SET DATA TYPE FLOAT[{dimension}] \
                     USING NULL::FLOAT[{dimension}];
                     UPDATE {table} SET embedding_profile = NULL;"
                ))?;
            }
        }
        self.vectors
            .column_dimension
            .store(dimension.get(), Ordering::Release);
        self.set_meta(MetaKey::EmbeddingDimension, &dimension.to_string())
    }

    /// The profile the configured model runs under, `None` without one.
    #[must_use]
    pub fn embedding_profile(&self) -> Option<&Profile> {
        self.vectors.profile.as_ref()
    }

    /// The fingerprint a vector made now is stored under, and the only one
    /// vector search matches. `None` without a model: vectors stored then
    /// (only tests store any) are unprofiled and match each other.
    #[must_use]
    pub fn embedding_fingerprint(&self) -> Option<&Fingerprint> {
        self.vectors.fingerprint.as_ref()
    }

    /// The error for storing a vector the columns cannot hold: the
    /// configured width changed and `refresh` has not run yet.
    fn check_vector_width(&self, width: usize) -> Result<()> {
        let column = self.embedding_dimension();
        if column.fits(width) {
            return Ok(());
        }
        Err(Error::Embedding(format!(
            "the workspace stores {column}-dimensional vectors and this one has {width}; \
             run `quack embeddings refresh` to embed the workspace again at the new width"
        )))
    }

    /// Drop a document's chunks and their term index, and clear its chunk
    /// count: what a failed processing pass leaves behind must not stay
    /// searchable (issue #52).
    ///
    /// # Errors
    ///
    /// Returns an error if a delete fails.
    pub fn discard_chunks(&self, document_id: &DocumentId) -> Result<()> {
        self.conn.execute(
            "DELETE FROM _quack_graph_extracted WHERE chunk_id IN (SELECT id FROM _quack_chunks WHERE document_id = ?)",
            duckdb::params![document_id],
        )?;
        self.conn.execute(
            "DELETE FROM _quack_terms WHERE chunk_id IN (SELECT id FROM _quack_chunks WHERE document_id = ?)",
            duckdb::params![document_id],
        )?;
        self.conn.execute(
            "DELETE FROM _quack_chunks WHERE document_id = ?",
            duckdb::params![document_id],
        )?;
        self.conn.execute(
            "UPDATE _quack_documents SET chunk_count = NULL WHERE id = ?",
            duckdb::params![document_id],
        )?;
        Ok(())
    }

    /// Register a document row. Ingestion computes the hash and checks for
    /// a duplicate first; see `ingestion::register_document`.
    ///
    /// # Errors
    ///
    /// Returns an error if the insert fails.
    pub fn insert_document(&self, doc: &NewDocument<'_>) -> Result<()> {
        let size = i64::try_from(doc.size_bytes)
            .map_err(|_| Error::Ingestion("file size overflow".into()))?;

        self.conn.execute(
            "INSERT INTO _quack_documents (id, filename, title, mime_type, size_bytes, sha256, source, status, ingested_by, source_root, source_path) \
             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            duckdb::params![
                doc.id,
                doc.filename,
                doc.title,
                doc.mime_type,
                size,
                doc.sha256,
                doc.source.as_str(),
                doc.status,
                doc.ingested_by,
                doc.source_root,
                doc.source_path,
            ],
        )?;
        Ok(())
    }

    /// The document whose content hashes to `sha256`, if one was ingested
    /// and did not fail. Re-uploads of identical bytes are skipped through
    /// this lookup; a failed document is not a match so it can be retried.
    ///
    /// # Errors
    ///
    /// Returns an error if the query fails.
    pub fn document_by_sha256(&self, sha256: &str) -> Result<Option<DocumentInfo>> {
        let sql = format!(
            "{DOCUMENT_SELECT} WHERE sha256 = ? AND {LIVE_STATUS} ORDER BY ingested_at, id LIMIT 1"
        );
        let mut stmt = self.conn.prepare(&sql)?;
        let mut rows = stmt.query(duckdb::params![sha256])?;
        match rows.next()? {
            Some(row) => Ok(Some(DocumentInfo::try_from(row)?)),
            None => Ok(None),
        }
    }

    /// The live document that loaded `table`, if any: one document owns a
    /// table (issue #51).
    ///
    /// # Errors
    ///
    /// Returns an error if the query fails.
    pub fn table_owner(&self, table: &str) -> Result<Option<DocumentInfo>> {
        let sql = format!(
            "{DOCUMENT_SELECT} WHERE {LIVE_STATUS} AND tables IS NOT NULL \
             AND list_contains(CAST(tables AS VARCHAR[]), ?) ORDER BY ingested_at, id LIMIT 1"
        );
        let mut stmt = self.conn.prepare(&sql)?;
        let mut rows = stmt.query(duckdb::params![table])?;
        match rows.next()? {
            Some(row) => Ok(Some(DocumentInfo::try_from(row)?)),
            None => Ok(None),
        }
    }

    /// The newest ready document named `filename`: what `--replace` with
    /// no id replaces.
    ///
    /// # Errors
    ///
    /// Returns an error if the query fails.
    pub fn newest_document_named(&self, filename: &str) -> Result<Option<DocumentInfo>> {
        let sql = format!(
            "{DOCUMENT_SELECT} WHERE filename = ? AND status = ? \
             ORDER BY ingested_at DESC, id DESC LIMIT 1"
        );
        let mut stmt = self.conn.prepare(&sql)?;
        let mut rows = stmt.query(duckdb::params![filename, DocumentStatus::Ready])?;
        match rows.next()? {
            Some(row) => Ok(Some(DocumentInfo::try_from(row)?)),
            None => Ok(None),
        }
    }

    /// The ready document a run of the folder `source_root` stored from
    /// `source_path` under it, if one is: what a changed file at that path
    /// replaces. The same relative path under another folder is another
    /// document.
    ///
    /// # Errors
    ///
    /// Returns an error if the query fails.
    pub fn newest_document_at_path(
        &self,
        source_root: &str,
        source_path: &str,
    ) -> Result<Option<DocumentInfo>> {
        let sql = format!(
            "{DOCUMENT_SELECT} WHERE source_root = ? AND source_path = ? AND status = ? \
             ORDER BY ingested_at DESC, id DESC LIMIT 1"
        );
        let mut stmt = self.conn.prepare(&sql)?;
        let mut rows = stmt.query(duckdb::params![
            source_root,
            source_path,
            DocumentStatus::Ready
        ])?;
        match rows.next()? {
            Some(row) => Ok(Some(DocumentInfo::try_from(row)?)),
            None => Ok(None),
        }
    }

    /// Every ready document a run of the folder `source_root` stored, by
    /// its path: what a run compares that folder against to find the files
    /// that are gone. Documents from any other folder are not touched.
    ///
    /// # Errors
    ///
    /// Returns an error if the query fails.
    pub fn documents_under(&self, source_root: &str) -> Result<Vec<DocumentInfo>> {
        let sql = format!(
            "{DOCUMENT_SELECT} WHERE source_root = ? AND source_path IS NOT NULL AND status = ? \
             ORDER BY source_path, ingested_at DESC, id DESC"
        );
        let mut stmt = self.conn.prepare(&sql)?;
        let docs = stmt.query_map(duckdb::params![source_root, DocumentStatus::Ready], |row| {
            DocumentInfo::try_from(row)
        })?;
        Ok(docs.collect::<duckdb::Result<_>>()?)
    }

    /// Mark `old` as being replaced by `new`: `old` keeps serving until
    /// `new` is ready ([`Self::finish_replacement`]), and a failure of
    /// `new` undoes the mark ([`Self::mark_document_error`]).
    ///
    /// # Errors
    ///
    /// `old` must exist, be `ready`, and not already have a replacement
    /// on its way.
    pub fn begin_replacement(&self, old: &DocumentId, new: &DocumentId) -> Result<()> {
        let document = self
            .document(old)?
            .ok_or_else(|| ResourceKind::Document.missing(old.as_str()))?;
        if document.status != DocumentStatus::Ready {
            return Err(Error::Ingestion(format!(
                "cannot replace {} ({old}): it is {}, not ready",
                OneLine(&document.filename),
                document.status
            )));
        }
        if let Some(pending) = document.superseded_by {
            return Err(Error::Ingestion(format!(
                "cannot replace {} ({old}): a replacement ({pending}) is already being processed",
                OneLine(&document.filename)
            )));
        }
        self.conn.execute(
            "UPDATE _quack_documents SET superseded_by = ? WHERE id = ?",
            duckdb::params![new, old],
        )?;
        Ok(())
    }

    /// `new` is ready: the document it replaces becomes `superseded` and
    /// `new` takes over its pin. Returns the replaced document's id, if
    /// there was one.
    ///
    /// # Errors
    ///
    /// Returns an error if an update fails.
    pub fn finish_replacement(&self, new: &DocumentId) -> Result<Option<DocumentId>> {
        let mut stmt = self
            .conn
            .prepare("SELECT id, pinned FROM _quack_documents WHERE superseded_by = ?")?;
        let mut rows = stmt.query(duckdb::params![new])?;
        let Some(row) = rows.next()? else {
            return Ok(None);
        };
        let (old, pinned): (DocumentId, bool) = (row.get(0)?, row.get(1)?);
        drop(rows);
        self.conn.execute(
            "UPDATE _quack_documents SET status = ? WHERE id = ?",
            duckdb::params![DocumentStatus::Superseded, old],
        )?;
        if pinned {
            self.conn.execute(
                "UPDATE _quack_documents SET pinned = true WHERE id = ?",
                duckdb::params![new],
            )?;
        }
        Ok(Some(old))
    }

    /// Forget that `new` was to replace anything: the document it would
    /// have replaced stays as it was.
    fn abandon_replacement(&self, new: &DocumentId) -> Result<()> {
        self.conn.execute(
            "UPDATE _quack_documents SET superseded_by = NULL WHERE superseded_by = ?",
            duckdb::params![new],
        )?;
        Ok(())
    }

    /// Whether what a ready document loaded is still there: every table
    /// it recorded exists, and a chunked document still has chunks. A
    /// document still queued or processing counts as intact.
    ///
    /// # Errors
    ///
    /// Returns an error if a query fails.
    pub fn document_intact(&self, doc: &DocumentInfo) -> Result<bool> {
        if doc.status != DocumentStatus::Ready {
            return Ok(true);
        }
        if let Some(tables) = &doc.tables
            && !tables.is_empty()
        {
            let existing = self.list_tables()?;
            return Ok(tables.iter().all(|t| existing.contains(t)));
        }
        if doc.chunk_count.is_some_and(|n| n > 0) {
            let chunks: i64 = self.conn.query_row(
                "SELECT count(*) FROM _quack_chunks WHERE document_id = ?",
                duckdb::params![doc.id],
                |r| r.get(0),
            )?;
            return Ok(chunks > 0);
        }
        Ok(true)
    }

    /// Replace a deleted user's id and name in this file with
    /// [`REMOVED_USER`]: the sessions they made, the documents they
    /// ingested, the context versions they edited, and the detail audit
    /// rows that name them. The access audit in `control.db` keeps the id.
    /// Returns how many rows changed.
    ///
    /// # Errors
    ///
    /// Returns an error if an update fails.
    pub fn forget_user(&self, user_id: &UserId, username: &str) -> Result<usize> {
        let mut changed = 0_usize;
        for (sql, value) in [
            (
                "UPDATE _quack_sessions SET created_by = ? WHERE created_by = ?",
                user_id.as_str(),
            ),
            (
                "UPDATE _quack_documents SET ingested_by = ? WHERE ingested_by = ?",
                user_id.as_str(),
            ),
            (
                "UPDATE _quack_audit SET user_id = ? WHERE user_id = ?",
                user_id.as_str(),
            ),
            (
                "UPDATE _quack_context SET edited_by = ? WHERE edited_by = ?",
                username,
            ),
        ] {
            changed = changed.saturating_add(
                self.conn
                    .execute(sql, duckdb::params![REMOVED_USER, value])?,
            );
        }
        Ok(changed)
    }

    /// Fail every document still `queued` or `processing`: called once
    /// when a server opens the workspace, since upload bytes live only in
    /// the memory of the process that took them, so nothing can finish a
    /// row a restart interrupted (issue #51). Returns how many.
    ///
    /// # Errors
    ///
    /// Returns an error if the update fails.
    pub fn fail_stale_uploads(&self) -> Result<usize> {
        let changed = self.conn.execute(
            "UPDATE _quack_documents SET status = ?, \
             error_message = 'interrupted by a restart before it was processed; upload it again' \
             WHERE status IN (?, ?)",
            duckdb::params![
                DocumentStatus::Error,
                DocumentStatus::Queued,
                DocumentStatus::Processing
            ],
        )?;
        Ok(changed)
    }

    /// Set the title parsed from the content, when the caller gave none.
    ///
    /// # Errors
    ///
    /// Returns an error if the update fails.
    pub fn set_document_title_if_empty(&self, id: &DocumentId, title: &str) -> Result<()> {
        self.conn.execute(
            "UPDATE _quack_documents SET title = ? WHERE id = ? AND title IS NULL",
            duckdb::params![title, id],
        )?;
        Ok(())
    }

    /// Record the tables a structured document loaded into, so deleting
    /// the document drops them.
    ///
    /// # Errors
    ///
    /// Returns an error if the update fails.
    pub fn set_document_tables(&self, id: &DocumentId, tables: &[String]) -> Result<()> {
        let json = serde_json::to_string(tables)?;
        self.conn.execute(
            "UPDATE _quack_documents SET tables = ? WHERE id = ?",
            duckdb::params![json, id],
        )?;
        Ok(())
    }

    /// Record what a parse found the document says about itself. A date
    /// the database cannot read as a timestamp is kept in `metadata`
    /// under its key instead of being lost.
    ///
    /// # Errors
    ///
    /// Returns an error if the update fails.
    pub fn set_document_meta(&self, id: &DocumentId, meta: &DocumentMeta) -> Result<()> {
        let mut extra = meta.extra.clone();
        for (key, value) in [
            ("authored_at", &meta.authored_at),
            ("modified_at", &meta.modified_at),
        ] {
            if let Some(v) = value
                && !self.parses_as_timestamp(v)?
            {
                extra.insert(key.to_owned(), v.clone());
            }
        }
        self.conn.execute(
            "UPDATE _quack_documents SET \
                author = COALESCE(author, ?), \
                authored_at = COALESCE(authored_at, TRY_CAST(? AS TIMESTAMP)), \
                modified_at = COALESCE(modified_at, TRY_CAST(? AS TIMESTAMP)), \
                tags = CASE WHEN tags IS NULL OR CAST(tags AS VARCHAR) = '[]' THEN ?::JSON ELSE tags END, \
                metadata = ?::JSON \
             WHERE id = ?",
            duckdb::params![
                meta.author,
                meta.authored_at,
                meta.modified_at,
                serde_json::to_string(&meta.tags)?,
                serde_json::to_string(&extra)?,
                id
            ],
        )?;
        Ok(())
    }

    fn parses_as_timestamp(&self, text: &str) -> Result<bool> {
        let parsed: Option<String> = self.conn.query_row(
            "SELECT CAST(TRY_CAST(? AS TIMESTAMP) AS VARCHAR)",
            duckdb::params![text],
            |r| r.get(0),
        )?;
        Ok(parsed.is_some())
    }

    /// A person's edit of a document's own fields: each given value
    /// replaces the stored one (an empty text clears it), tags replace the
    /// list whole.
    ///
    /// # Errors
    ///
    /// Returns an error when the document is missing, a date does not read
    /// as one, or the update fails.
    pub fn set_document_fields(&self, id: &DocumentId, fields: &DocumentFields) -> Result<()> {
        if self.document(id)?.is_none() {
            return Err(ResourceKind::Document.missing(id.as_str()));
        }
        if let Some(date) = fields
            .authored_at
            .as_deref()
            .filter(|d| !d.trim().is_empty())
            && !self.parses_as_timestamp(date)?
        {
            return Err(Error::Analysis(format!(
                "'{date}' is not a date; give one as YYYY-MM-DD or an ISO 8601 timestamp"
            )));
        }
        let clear = |value: &Option<String>| value.as_deref().map(str::trim).map(str::is_empty);
        if let Some(title) = &fields.title {
            self.conn.execute(
                "UPDATE _quack_documents SET title = ? WHERE id = ?",
                duckdb::params![
                    (clear(&fields.title) != Some(true)).then_some(title.trim()),
                    id
                ],
            )?;
        }
        if let Some(author) = &fields.author {
            self.conn.execute(
                "UPDATE _quack_documents SET author = ? WHERE id = ?",
                duckdb::params![
                    (clear(&fields.author) != Some(true)).then_some(author.trim()),
                    id
                ],
            )?;
        }
        if let Some(date) = &fields.authored_at {
            self.conn.execute(
                "UPDATE _quack_documents SET authored_at = TRY_CAST(? AS TIMESTAMP) WHERE id = ?",
                duckdb::params![
                    (clear(&fields.authored_at) != Some(true)).then_some(date.trim()),
                    id
                ],
            )?;
        }
        if let Some(tags) = &fields.tags {
            let tags: Vec<String> = tags
                .iter()
                .map(|t| t.trim().to_owned())
                .filter(|t| !t.is_empty())
                .collect();
            self.conn.execute(
                "UPDATE _quack_documents SET tags = ?::JSON WHERE id = ?",
                duckdb::params![serde_json::to_string(&tags)?, id],
            )?;
        }
        Ok(())
    }

    /// Record how many chunks a processed document produced.
    ///
    /// # Errors
    ///
    /// Returns an error if the update fails.
    pub fn set_document_chunk_count(&self, id: &DocumentId, count: u32) -> Result<()> {
        self.conn.execute(
            "UPDATE _quack_documents SET chunk_count = ? WHERE id = ?",
            duckdb::params![count, id],
        )?;
        Ok(())
    }

    /// Record how a processed document's pages read; `None` for a source
    /// without pages.
    ///
    /// # Errors
    ///
    /// Returns an error if the update fails.
    pub fn set_document_pages(&self, id: &DocumentId, pages: Option<PageCounts>) -> Result<()> {
        self.conn.execute(
            "UPDATE _quack_documents SET page_count = ?, pages_unreadable = ?, pages_empty = ? \
             WHERE id = ?",
            duckdb::params![
                pages.map(|p| p.total),
                pages.map(|p| p.unreadable),
                pages.map(|p| p.empty),
                id
            ],
        )?;
        Ok(())
    }

    /// Update a document's status and clear any earlier error.
    ///
    /// # Errors
    ///
    /// Returns an error if the update fails.
    pub fn update_document_status(&self, id: &DocumentId, status: DocumentStatus) -> Result<()> {
        self.conn.execute(
            "UPDATE _quack_documents SET status = ?, error_message = NULL WHERE id = ?",
            duckdb::params![status, id],
        )?;
        Ok(())
    }

    /// Mark a document as failed with the reason.
    ///
    /// # Errors
    ///
    /// Returns an error if the update fails.
    pub fn mark_document_error(&self, id: &DocumentId, message: &str) -> Result<()> {
        self.conn.execute(
            "UPDATE _quack_documents SET status = ?, error_message = ? WHERE id = ?",
            duckdb::params![DocumentStatus::Error, message, id],
        )?;
        // A failed replacement leaves the document it was to replace as it was.
        self.abandon_replacement(id)
    }

    /// One document by id.
    ///
    /// # Errors
    ///
    /// Returns an error if the query fails.
    pub fn document(&self, id: &DocumentId) -> Result<Option<DocumentInfo>> {
        let sql = format!("{DOCUMENT_SELECT} WHERE id = ?");
        let mut stmt = self.conn.prepare(&sql)?;
        let mut rows = stmt.query(duckdb::params![id])?;
        match rows.next()? {
            Some(row) => Ok(Some(DocumentInfo::try_from(row)?)),
            None => Ok(None),
        }
    }

    /// Remove a document with its chunks and term index. The tables it
    /// loaded into are dropped too: those recorded on the row, or for rows
    /// from before that was recorded, [`DocumentInfo::fallback_tables`]. Graph nodes and
    /// edges whose only provenance was the document or its tables go with
    /// it (issue #43), as do its files under `files/`. Returns whether the
    /// document existed.
    ///
    /// # Errors
    ///
    /// Returns an error if any delete fails.
    pub fn delete_document(&self, id: &DocumentId) -> Result<bool> {
        let Some(doc) = self.document(id)? else {
            return Ok(false);
        };
        // A replaced document's tables and file now belong to its
        // replacement; only its own rows go.
        let tables = if doc.status == DocumentStatus::Superseded {
            Vec::new()
        } else if let Some(tables) = doc.tables.clone() {
            tables
        } else {
            // Never a table another document loaded: a row still queued, or
            // one that failed before loading, has no tables of its own yet.
            let mut unowned = Vec::new();
            for table in doc.fallback_tables() {
                if self
                    .table_owner(&table)?
                    .is_none_or(|owner| owner.id == doc.id)
                {
                    unowned.push(table);
                }
            }
            unowned
        };
        self.forget_graph_provenance(id, &tables)?;
        self.conn.execute(
            "DELETE FROM _quack_graph_extracted WHERE chunk_id IN (SELECT id FROM _quack_chunks WHERE document_id = ?)",
            duckdb::params![id],
        )?;
        self.conn.execute(
            "DELETE FROM _quack_terms WHERE chunk_id IN (SELECT id FROM _quack_chunks WHERE document_id = ?)",
            duckdb::params![id],
        )?;
        self.conn.execute(
            "DELETE FROM _quack_chunks WHERE document_id = ?",
            duckdb::params![id],
        )?;
        self.conn.execute(
            "DELETE FROM _quack_documents WHERE id = ?",
            duckdb::params![id],
        )?;
        for table in &tables {
            self.conn
                .execute_batch(&format!("DROP TABLE IF EXISTS {}", quote_ident(table)))?;
        }
        if doc.status != DocumentStatus::Superseded {
            self.remove_document_files(&doc.filename, &tables);
        }
        self.abandon_replacement(id)?;
        Ok(true)
    }

    /// Drop the provenance a document and its tables gave the graph, then
    /// the nodes and edges left without any provenance at all (an edge
    /// whose endpoint goes falls with it, as `graph::store::delete_nodes`
    /// does).
    fn forget_graph_provenance(&self, document_id: &DocumentId, tables: &[String]) -> Result<()> {
        let table_list = sql_text_list(tables);
        let mut stmt = self.conn.prepare(
            "SELECT DISTINCT subject_id FROM _quack_provenance \
             WHERE document_id = ? OR list_contains(?::VARCHAR[], table_name)",
        )?;
        let touched = stmt
            .query_map(duckdb::params![document_id, table_list], |row| {
                row.get::<_, String>(0)
            })?
            .collect::<duckdb::Result<Vec<_>>>()?;
        drop(stmt);
        if touched.is_empty() {
            return Ok(());
        }
        self.conn.execute(
            "DELETE FROM _quack_provenance \
             WHERE document_id = ? OR list_contains(?::VARCHAR[], table_name)",
            duckdb::params![document_id, table_list],
        )?;
        let touched_list = sql_text_list(&touched);
        let orphan_nodes = self.conn.execute(
            "DELETE FROM _quack_graph_nodes WHERE list_contains(?::VARCHAR[], id) \
             AND NOT EXISTS (SELECT 1 FROM _quack_provenance p WHERE p.subject_id = _quack_graph_nodes.id)",
            duckdb::params![touched_list],
        )?;
        let orphan_edges = self.conn.execute(
            "DELETE FROM _quack_graph_edges WHERE (list_contains(?::VARCHAR[], id) \
             AND NOT EXISTS (SELECT 1 FROM _quack_provenance p WHERE p.subject_id = _quack_graph_edges.id)) \
             OR NOT EXISTS (SELECT 1 FROM _quack_graph_nodes n WHERE n.id = source_node_id) \
             OR NOT EXISTS (SELECT 1 FROM _quack_graph_nodes n WHERE n.id = target_node_id)",
            duckdb::params![touched_list],
        )?;
        self.conn.execute_batch(
            "DELETE FROM _quack_provenance WHERE NOT EXISTS \
               (SELECT 1 FROM _quack_graph_nodes n WHERE n.id = subject_id) \
             AND NOT EXISTS (SELECT 1 FROM _quack_graph_edges e WHERE e.id = subject_id); \
             DELETE FROM _quack_graph_merges WHERE NOT EXISTS \
               (SELECT 1 FROM _quack_graph_nodes n WHERE n.id = keep_node_id) \
             OR NOT EXISTS (SELECT 1 FROM _quack_graph_nodes n WHERE n.id = drop_node_id);",
        )?;
        tracing::info!(
            document_id = %document_id,
            orphan_nodes,
            orphan_edges,
            "removed graph rows that only the deleted document supported"
        );
        Ok(())
    }

    /// Remove what ingestion wrote under `files/` for a document: the file
    /// itself and, for workbooks and imports, one CSV per table. A missing
    /// file is fine; any other failure is logged, since the rows are gone.
    fn remove_document_files(&self, filename: &str, tables: &[String]) {
        let Some(files_dir) = &self.files_dir else {
            return;
        };
        let mut candidates: Vec<PathBuf> = Vec::new();
        if let Some(name) = Path::new(filename).file_name() {
            candidates.push(files_dir.join(name));
        }
        for table in tables {
            candidates.push(files_dir.join(format!("{table}.csv")));
        }
        for path in candidates {
            match std::fs::remove_file(&path) {
                Ok(()) => tracing::debug!(path = %path.display(), "removed an ingested file"),
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
                Err(e) => {
                    tracing::warn!(path = %path.display(), error = %e, "could not remove an ingested file");
                }
            }
        }
    }

    /// Insert a text chunk, optionally with an embedding vector, and index
    /// its terms for keyword search.
    ///
    /// # Errors
    ///
    /// Returns an error if the insert fails.
    pub fn insert_chunk(&self, chunk: &NewChunk<'_>) -> Result<()> {
        let page = chunk.page.map(i64::from);
        let stemming = self.document_stemming(chunk.document_id, chunk.content)?;
        let terms = TermFrequencies::of(&Analyzer::of(stemming), chunk.content, chunk.heading);
        let length = terms.total();
        match chunk.embedding {
            Some(emb) => {
                self.check_vector_width(emb.len())?;
                let sql = format!(
                    "INSERT INTO _quack_chunks (id, document_id, chunk_index, content, heading, page, token_count, embedding, embedding_profile, kind, locator) \
                     VALUES (?, ?, ?, ?, ?, ?, ?, ?::{}, ?, ?, ?)",
                    self.vector_type()
                );
                self.conn.execute(
                    &sql,
                    duckdb::params![
                        chunk.id,
                        chunk.document_id,
                        chunk.chunk_index,
                        chunk.content,
                        chunk.heading,
                        page,
                        length,
                        emb.sql_literal(),
                        self.embedding_fingerprint(),
                        chunk.kind,
                        chunk.locator
                    ],
                )?;
            }
            None => {
                self.conn.execute(
                    "INSERT INTO _quack_chunks (id, document_id, chunk_index, content, heading, page, token_count, kind, locator) \
                     VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    duckdb::params![
                        chunk.id,
                        chunk.document_id,
                        chunk.chunk_index,
                        chunk.content,
                        chunk.heading,
                        page,
                        length,
                        chunk.kind,
                        chunk.locator
                    ],
                )?;
            }
        }
        self.insert_terms(chunk.id, &terms)?;
        Ok(())
    }

    fn insert_terms(&self, chunk_id: &ChunkId, terms: &TermFrequencies) -> Result<()> {
        if terms.0.is_empty() {
            return Ok(());
        }
        let mut appender = self.conn.appender("_quack_terms")?;
        for (term, tf) in &terms.0 {
            appender.append_row(duckdb::params![chunk_id, term, i64::from(*tf)])?;
        }
        appender.flush()?;
        Ok(())
    }

    /// The stemming a document's chunks are indexed under: the language it
    /// was detected as, detected now from `sample` when it has none yet.
    fn document_stemming(&self, id: &DocumentId, sample: &str) -> Result<Stemming> {
        let code: Option<String> = self
            .conn
            .query_row(
                "SELECT language FROM _quack_documents WHERE id = ?",
                duckdb::params![id],
                |row| row.get(0),
            )
            .or_else(|e| match e {
                duckdb::Error::QueryReturnedNoRows => Ok(None),
                other => Err(other),
            })?;
        match code {
            Some(code) => Ok(Stemming::of_code(Some(&code))),
            None => self.set_document_language(id, sample),
        }
    }

    /// Detect a document's language from `sample` (its opening text, under
    /// `[retrieval].languages`), record it, and add its stemming to the
    /// workspace's set: what its chunks are indexed under.
    ///
    /// # Errors
    ///
    /// Returns an error if a write fails.
    pub fn set_document_language(&self, id: &DocumentId, sample: &str) -> Result<Stemming> {
        let code = self.languages.detect(sample);
        self.conn.execute(
            "UPDATE _quack_documents SET language = ? WHERE id = ?",
            duckdb::params![code, id],
        )?;
        self.record_languages()?;
        Ok(Stemming::of_code(Some(code)))
    }

    /// Record the stemmings the documents were indexed under, which a
    /// query is tokenized under.
    fn record_languages(&self) -> Result<()> {
        let mut stmt = self
            .conn
            .prepare("SELECT DISTINCT language FROM _quack_documents WHERE language IS NOT NULL")?;
        let codes = stmt
            .query_map([], |row| row.get::<_, String>(0))?
            .collect::<duckdb::Result<Vec<_>>>()?;
        let recorded = Analyzer::record(codes.iter().map(|c| Some(c.as_str())));
        self.set_meta(MetaKey::Languages, &recorded)
    }

    /// The stemmings a query is tokenized under: every one a document of
    /// the workspace was indexed under.
    ///
    /// # Errors
    ///
    /// Returns an error if `_quack_meta` cannot be read.
    pub fn query_analyzer(&self) -> Result<Analyzer> {
        Ok(Analyzer::of_recorded(
            self.meta(MetaKey::Languages)?.as_deref(),
        ))
    }

    /// Characters of a document's opening text its language is detected
    /// from.
    pub const LANGUAGE_SAMPLE_CHARS: usize = 8000;

    /// Detect the language of every document with chunks that has none,
    /// from its first chunks: documents ingested before detection.
    fn detect_missing_languages(&self) -> Result<()> {
        let mut stmt = self.conn.prepare(
            "SELECT c.document_id, left(string_agg(c.content, ' ' ORDER BY c.chunk_index), ?) \
             FROM _quack_chunks c JOIN _quack_documents d ON d.id = c.document_id \
             WHERE d.language IS NULL AND c.chunk_index < 16 \
             GROUP BY c.document_id",
        )?;
        let samples = stmt
            .query_map(
                duckdb::params![i64::try_from(Self::LANGUAGE_SAMPLE_CHARS).unwrap_or(i64::MAX)],
                |row| Ok((row.get::<_, DocumentId>(0)?, row.get::<_, String>(1)?)),
            )?
            .collect::<duckdb::Result<Vec<_>>>()?;
        for (id, sample) in samples {
            self.conn.execute(
                "UPDATE _quack_documents SET language = ? WHERE id = ?",
                duckdb::params![self.languages.detect(&sample), id],
            )?;
        }
        self.record_languages()
    }

    fn chunk_count(&self) -> Result<i64> {
        Ok(self
            .conn
            .query_row("SELECT count(*) FROM _quack_chunks", [], |row| row.get(0))?)
    }

    /// Rebuild the keyword index from every stored chunk.
    ///
    /// # Errors
    ///
    /// Returns an error if reading chunks or writing terms fails.
    pub fn reindex_terms(&self) -> Result<()> {
        self.reindex_terms_by(REINDEX_PAGE)
    }

    /// [`Self::reindex_terms`], reading `page` chunks at a time.
    fn reindex_terms_by(&self, page: u32) -> Result<()> {
        self.conn.execute("DELETE FROM _quack_terms", [])?;
        let mut stmt = self.conn.prepare(
            "SELECT c.id, c.heading, c.content, d.language FROM _quack_chunks c \
             LEFT JOIN _quack_documents d ON d.id = c.document_id \
             WHERE ?::VARCHAR IS NULL OR c.id > ? ORDER BY c.id LIMIT ?",
        )?;
        // A page at a time by id, so a large workspace never holds every
        // chunk's text at once.
        let mut after: Option<ChunkId> = None;
        loop {
            let page = stmt
                .query_map(duckdb::params![after, after, i64::from(page)], |row| {
                    Ok((
                        PendingChunk::try_from(row)?,
                        row.get::<_, Option<String>>(3)?,
                    ))
                })?
                .collect::<duckdb::Result<Vec<_>>>()?;
            let Some((last, _)) = page.last() else {
                return Ok(());
            };
            after = Some(last.id.clone());
            for (chunk, language) in &page {
                let analyzer = Analyzer::of(Stemming::of_code(language.as_deref()));
                let terms =
                    TermFrequencies::of(&analyzer, &chunk.content, chunk.heading.as_deref());
                self.conn.execute(
                    "UPDATE _quack_chunks SET token_count = ? WHERE id = ?",
                    duckdb::params![terms.total(), chunk.id],
                )?;
                self.insert_terms(&chunk.id, &terms)?;
            }
        }
    }

    /// Store a chunk's vector, made under the current profile.
    ///
    /// # Errors
    ///
    /// Returns an error if the vector does not fit the columns or the
    /// update fails.
    pub fn set_chunk_embedding(&self, chunk_id: &ChunkId, embedding: &Vector) -> Result<()> {
        self.set_vector(VectorTable::Chunks, chunk_id.as_str(), embedding)
    }

    /// Store a graph node's label vector, made under the current profile.
    ///
    /// # Errors
    ///
    /// Returns an error if the vector does not fit the columns or the
    /// update fails.
    pub fn set_node_embedding(&self, node_id: &NodeId, embedding: &Vector) -> Result<()> {
        self.set_vector(VectorTable::GraphNodes, node_id.as_str(), embedding)
    }

    fn set_vector(&self, table: VectorTable, id: &str, embedding: &Vector) -> Result<()> {
        self.check_vector_width(embedding.len())?;
        let sql = format!(
            "UPDATE {table} SET embedding = ?::{}, embedding_profile = ? WHERE id = ?",
            self.vector_type()
        );
        self.conn.execute(
            &sql,
            duckdb::params![embedding.sql_literal(), self.embedding_fingerprint(), id],
        )?;
        Ok(())
    }

    /// Chunks of ready documents whose vector is missing or was made under
    /// another profile, oldest first: what `refresh` works through.
    ///
    /// # Errors
    ///
    /// Returns an error if the query fails.
    pub fn chunks_needing_embedding(&self, limit: u32) -> Result<Vec<PendingChunk>> {
        let mut stmt = self.conn.prepare(
            "SELECT c.id, c.heading, c.content FROM _quack_chunks c \
             JOIN _quack_documents d ON d.id = c.document_id \
             WHERE d.status = ? \
               AND (c.embedding IS NULL OR c.embedding_profile IS DISTINCT FROM ?) \
             ORDER BY c.id LIMIT ?",
        )?;
        let rows = stmt.query_map(
            duckdb::params![
                DocumentStatus::Ready,
                self.embedding_fingerprint(),
                i64::from(limit)
            ],
            |row| PendingChunk::try_from(row),
        )?;
        Ok(rows.collect::<duckdb::Result<_>>()?)
    }

    /// How this workspace's stored vectors stand against the current
    /// profile.
    ///
    /// # Errors
    ///
    /// Returns an error if a query fails.
    pub fn embedding_status(&self) -> Result<EmbeddingStatus> {
        let current = self.embedding_fingerprint();
        let (current_chunks, missing_chunks): (u64, u64) = self.conn.query_row(
            "SELECT count(*) FILTER (WHERE c.embedding IS NOT NULL AND c.embedding_profile IS NOT DISTINCT FROM ?), \
                    count(*) FILTER (WHERE c.embedding IS NULL) \
             FROM _quack_chunks c JOIN _quack_documents d ON d.id = c.document_id \
             WHERE d.status = ?",
            duckdb::params![current, DocumentStatus::Ready],
            |row| Ok((row.get(0)?, row.get(1)?)),
        )?;
        let mut stmt = self.conn.prepare(
            "SELECT CAST(p.profile AS VARCHAR), count(*) \
             FROM _quack_chunks c JOIN _quack_documents d ON d.id = c.document_id \
             LEFT JOIN _quack_embedding_profiles p ON p.fingerprint = c.embedding_profile \
             WHERE d.status = ? AND c.embedding IS NOT NULL \
               AND c.embedding_profile IS DISTINCT FROM ? \
             GROUP BY ALL ORDER BY 2 DESC",
        )?;
        let stale = stmt
            .query_map(duckdb::params![DocumentStatus::Ready, current], |row| {
                let profile: Option<String> = row.get(0)?;
                Ok(StaleVectors {
                    profile: profile.and_then(|json| serde_json::from_str(&json).ok()),
                    chunks: row.get(1)?,
                })
            })?
            .collect::<duckdb::Result<Vec<_>>>()?;
        let (stale_nodes, nodes_needing_embedding): (u64, u64) = if self
            .table_exists("_quack_graph_nodes")?
        {
            self.conn.query_row(
                "SELECT count(*) FILTER (WHERE embedding IS NOT NULL AND embedding_profile IS DISTINCT FROM ?), \
                        count(*) FILTER (WHERE embedding IS NULL OR embedding_profile IS DISTINCT FROM ?) \
                 FROM _quack_graph_nodes",
                duckdb::params![current, current],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )?
        } else {
            (0, 0)
        };
        Ok(EmbeddingStatus {
            profile: self.vectors.profile.clone(),
            column_dimension: self.embedding_dimension(),
            current_chunks,
            missing_chunks,
            stale,
            stale_nodes,
            nodes_needing_embedding,
        })
    }

    /// The `FLOAT[N]` type of this workspace's embedding column. `N` is a
    /// validated integer, the only value ever interpolated into vector SQL.
    #[must_use]
    pub fn vector_type(&self) -> String {
        format!("FLOAT[{}]", self.embedding_dimension())
    }

    /// Keyword search: BM25 over the terms quack indexed at ingest, scored in
    /// SQL from `_quack_terms` and each chunk's term count. Needs no
    /// extension and no rebuild step.
    ///
    /// A `"..."` quoted phrase in `query` is an adjacency requirement: `_quack_terms`
    /// carries no positions (`docs/migrations.md`), so BM25 still ranks by the
    /// phrase's own tokens, but the candidates are over-fetched and then
    /// post-filtered to those whose content or heading contains the phrase as a
    /// case-insensitive, whitespace-normalized substring. A phrase that matches no
    /// candidate returns an empty result rather than falling back to the
    /// unfiltered ranking. Unbalanced quotes are treated as ordinary text.
    ///
    /// # Errors
    ///
    /// Returns an error if the query fails.
    pub fn search_keyword_chunks(
        &self,
        query: &str,
        top_k: u32,
        scope: &ChunkScope,
    ) -> Result<Vec<ChunkSearchResult>> {
        if scope.is_empty() {
            return Ok(Vec::new());
        }
        let terms = TermFrequencies::distinct(&self.query_analyzer()?, query);
        if terms.is_empty() {
            return Ok(Vec::new());
        }
        let phrases = Phrases::parse(query);
        let fetch_k = phrases.fetch(top_k, top_k);
        // Quoted, so a token such as `null` stays a word and not a NULL
        // element (issue #62).
        let term_list = sql_text_list(&terms);
        let filter = scope.sql();
        let sql = format!(
            "WITH q AS (SELECT DISTINCT unnest(?::VARCHAR[]) AS term), \
                  stats AS (SELECT count(*) AS n, avg(token_count) AS avgdl \
                            FROM _quack_chunks WHERE token_count > 0), \
                  df AS (SELECT t.term, count(DISTINCT t.chunk_id) AS df \
                         FROM _quack_terms t JOIN q ON q.term = t.term GROUP BY t.term), \
                  scored AS (SELECT t.chunk_id, \
                         sum(ln(1 + (s.n - d.df + 0.5) / (d.df + 0.5)) \
                             * (t.tf * ({BM25_K1} + 1)) \
                             / (t.tf + {BM25_K1} * (1 - {BM25_B} + {BM25_B} * ch.token_count / s.avgdl))) AS score \
                         FROM _quack_terms t \
                         JOIN df d ON d.term = t.term \
                         JOIN _quack_chunks ch ON ch.id = t.chunk_id, stats s \
                         GROUP BY t.chunk_id) \
             SELECT c.id, c.content, c.document_id, c.chunk_index, d.filename, c.heading, c.page, sc.score, \
                    CAST(d.ingested_at AS VARCHAR), c.kind, c.locator \
             FROM scored sc \
             JOIN _quack_chunks c ON c.id = sc.chunk_id \
             JOIN _quack_documents d ON d.id = c.document_id \
             WHERE sc.score > 0 AND d.status = ?{filter} \
             ORDER BY sc.score DESC, c.chunk_index ASC \
             LIMIT ?"
        );
        let limit = i64::from(fetch_k);
        let mut stmt = self.conn.prepare(&sql)?;
        let mut params: Vec<&dyn duckdb::ToSql> = Vec::with_capacity(scope.len().saturating_add(3));
        params.push(&term_list);
        params.push(&DocumentStatus::Ready);
        scope.bind(&mut params);
        params.push(&limit);
        let mut results = stmt
            .query_map(params.as_slice(), |row| ChunkSearchResult::try_from(row))?
            .collect::<duckdb::Result<Vec<_>>>()?;
        phrases.retain_matching(&mut results, top_k);
        for (i, hit) in results.iter_mut().enumerate() {
            hit.ranks.keyword_rank = Some(Ranks::place(i));
            hit.ranks.bm25 = Some(hit.score);
        }
        Ok(results)
    }

    /// Hybrid retrieval: vector and keyword rankings fused with reciprocal
    /// rank fusion (`score = sum over rankings of 1 / (rrf_k + rank)`).
    ///
    /// A quoted phrase in `query_text` filters the fused result the same way
    /// [`Self::search_keyword_chunks`] filters its own: the vector leg knows
    /// nothing about phrases, so both legs are over-fetched and the phrase
    /// substring filter runs once on the fused ranking.
    ///
    /// # Errors
    ///
    /// Returns an error if either search fails.
    pub fn search_hybrid_chunks(
        &self,
        query_text: &str,
        query_embedding: &Vector,
        limits: HybridLimits,
        scope: &ChunkScope,
    ) -> Result<Vec<ChunkSearchResult>> {
        Ok(self
            .explain_search(query_text, query_embedding, limits, scope)?
            .fused)
    }

    /// One search in `mode`, with its workings. Hybrid without a query
    /// vector (no embedding model) runs the keyword leg alone; vector mode
    /// without one is an error, since nothing else would answer.
    ///
    /// # Errors
    ///
    /// Returns an error if a leg fails, or vector mode has no vector.
    pub fn search_chunks(
        &self,
        query: &str,
        embedding: Option<&Vector>,
        mode: SearchMode,
        limits: HybridLimits,
        scope: &ChunkScope,
    ) -> Result<SearchExplanation> {
        match (mode, embedding) {
            (SearchMode::Hybrid, Some(vector)) => self.explain_search(query, vector, limits, scope),
            (SearchMode::Hybrid | SearchMode::Keyword, _) => Ok(SearchExplanation::keyword_only(
                query,
                self.search_keyword_chunks(query, limits.top_k, scope)?,
            )),
            (SearchMode::Vector, Some(vector)) => {
                let phrases = Phrases::parse(query);
                let fetch = phrases.fetch(limits.top_k, limits.top_k);
                let hits = self.search_similar_chunks(vector, fetch, scope)?;
                Ok(SearchExplanation::vector_only(phrases, hits, limits.top_k))
            }
            (SearchMode::Vector, None) => Err(Error::Analysis(String::from(
                "vector search needs an embedding model ([embedding].model); search by keyword instead",
            ))),
        }
    }

    /// [`Self::search_hybrid_chunks`] with its workings: both legs as they
    /// ranked their (over-fetched) candidates, the fused ranking, and the
    /// quoted phrases that filtered it. Each hit carries its rank and score
    /// in every leg that found it.
    ///
    /// # Errors
    ///
    /// Returns an error if either search fails.
    pub fn explain_search(
        &self,
        query_text: &str,
        query_embedding: &Vector,
        limits: HybridLimits,
        scope: &ChunkScope,
    ) -> Result<SearchExplanation> {
        let phrases = Phrases::parse(query_text);
        let candidates = limits.top_k.saturating_mul(2).max(1);
        let fuse_k = phrases.fetch(limits.top_k, candidates);
        let vector = self.search_similar_chunks(query_embedding, fuse_k, scope)?;
        let keyword = self.search_keyword_chunks(query_text, fuse_k, scope)?;
        let mut fused = limits.fuse(vector.clone(), keyword.clone(), fuse_k);
        phrases.retain_matching(&mut fused, limits.top_k);
        Ok(SearchExplanation {
            vector,
            keyword,
            fused,
            phrases: phrases.0,
        })
    }

    /// Search for the most similar chunks to a query embedding. `score` is
    /// `1 / (1 + cosine distance)`.
    ///
    /// When `document_ids` is non-empty the search is restricted to those
    /// documents. Results carry the source filename for citations.
    ///
    /// # Errors
    ///
    /// Returns an error if the search query fails.
    pub fn search_similar_chunks(
        &self,
        query_embedding: &Vector,
        top_k: u32,
        scope: &ChunkScope,
    ) -> Result<Vec<ChunkSearchResult>> {
        if scope.is_empty() {
            return Ok(Vec::new());
        }
        if !self.embedding_dimension().fits(query_embedding.len()) {
            tracing::warn!(
                query = query_embedding.len(),
                stored = %self.embedding_dimension(),
                "vector search skipped: the workspace's vectors are another width until `quack embeddings refresh` runs"
            );
            return Ok(Vec::new());
        }
        let filter = scope.sql();
        let sql = format!(
            "SELECT c.id, c.content, c.document_id, c.chunk_index, d.filename, c.heading, c.page, \
                    1.0 / (1.0 + array_cosine_distance(c.embedding, ?::{})) AS score, \
                    CAST(d.ingested_at AS VARCHAR), c.kind, c.locator \
             FROM _quack_chunks c \
             JOIN _quack_documents d ON d.id = c.document_id \
             WHERE c.embedding IS NOT NULL AND c.embedding_profile IS NOT DISTINCT FROM ? \
               AND d.status = ?{filter} \
             ORDER BY score DESC \
             LIMIT ?",
            self.vector_type()
        );

        let query_literal = query_embedding.sql_literal();
        let fingerprint = self.embedding_fingerprint();
        let limit = i64::from(top_k);
        let mut stmt = self.conn.prepare(&sql)?;
        let mut params: Vec<&dyn duckdb::ToSql> = Vec::with_capacity(scope.len().saturating_add(4));
        params.push(&query_literal);
        params.push(&fingerprint);
        params.push(&DocumentStatus::Ready);
        scope.bind(&mut params);
        params.push(&limit);
        let mut results = stmt
            .query_map(params.as_slice(), |row| ChunkSearchResult::try_from(row))?
            .collect::<duckdb::Result<Vec<_>>>()?;
        for (i, hit) in results.iter_mut().enumerate() {
            hit.ranks.vector_rank = Some(Ranks::place(i));
            hit.ranks.vector_score = Some(hit.score);
        }
        Ok(results)
    }

    /// How many chunks `pool` holds.
    ///
    /// # Errors
    ///
    /// Returns an error if the query fails.
    pub fn pool_size(&self, pool: SamplePool) -> Result<u64> {
        Ok(self.conn.query_row(
            &format!(
                "SELECT count(*) FROM _quack_chunks c \
                 JOIN _quack_documents d ON d.id = c.document_id \
                 WHERE d.status = ? AND {}",
                pool.filter()
            ),
            duckdb::params![DocumentStatus::Ready],
            |row| row.get(0),
        )?)
    }

    /// Up to `limit` chunk ids from `pool`, spread evenly over its
    /// documents: each gets an equal quota, taken at evenly spaced
    /// positions through it, so a sample covers every document and not
    /// just the first one's front matter. Chosen in SQL, so only the ids
    /// of the chosen chunks leave the database.
    ///
    /// A document of `len` chunks with a quota of `take` keeps the chunks
    /// at `floor(k * len / take)` for `k` below `take`; the chunk at
    /// `pos` is one of them when `k = ceil(pos * take / len)` is below
    /// `take` and maps back to `pos`.
    ///
    /// # Errors
    ///
    /// Returns an error if the query fails.
    pub fn sample_chunk_ids(&self, pool: SamplePool, limit: u32) -> Result<Vec<ChunkId>> {
        if limit == 0 {
            return Ok(Vec::new());
        }
        let sql = format!(
            "WITH pool AS ( \
                 SELECT c.id, c.document_id, c.chunk_index, d.ingested_at, d.id AS doc, \
                        row_number() OVER (PARTITION BY c.document_id ORDER BY c.chunk_index) - 1 AS pos, \
                        count(*) OVER (PARTITION BY c.document_id) AS len \
                 FROM _quack_chunks c JOIN _quack_documents d ON d.id = c.document_id \
                 WHERE d.status = ? AND {filter}), \
             quota AS ( \
                 SELECT greatest(1, (?::BIGINT + count(DISTINCT document_id) - 1) \
                                    // greatest(count(DISTINCT document_id), 1)) AS q \
                 FROM pool), \
             placed AS ( \
                 SELECT p.*, least(quota.q, p.len) AS take FROM pool p, quota) \
             SELECT id FROM placed \
             WHERE (pos * take + len - 1) // len < take \
               AND (((pos * take + len - 1) // len) * len) // take = pos \
             ORDER BY {order}, chunk_index \
             LIMIT ?",
            filter = pool.filter(),
            order = pool.document_order(),
        );
        let mut stmt = self.conn.prepare(&sql)?;
        let ids = stmt.query_map(
            duckdb::params![DocumentStatus::Ready, i64::from(limit), i64::from(limit)],
            |row| row.get::<_, ChunkId>(0),
        )?;
        Ok(ids.collect::<duckdb::Result<_>>()?)
    }

    /// Up to `size` chunks of `pool` after the chunk `after`, by id, with
    /// the citation metadata a search hit carries (score 1): how a run
    /// over every chunk reads them a page at a time.
    ///
    /// # Errors
    ///
    /// Returns an error if the query fails.
    pub fn chunk_page(
        &self,
        pool: SamplePool,
        after: Option<&ChunkId>,
        size: u32,
    ) -> Result<Vec<ChunkSearchResult>> {
        let mut stmt = self.conn.prepare(&format!(
            "SELECT c.id, c.content, c.document_id, c.chunk_index, d.filename, c.heading, c.page, 1.0, \
                    CAST(d.ingested_at AS VARCHAR), c.kind, c.locator \
             FROM _quack_chunks c JOIN _quack_documents d ON d.id = c.document_id \
             WHERE d.status = ? AND {} AND (?::VARCHAR IS NULL OR c.id > ?) \
             ORDER BY c.id LIMIT ?",
            pool.filter()
        ))?;
        let rows = stmt.query_map(
            duckdb::params![DocumentStatus::Ready, after, after, i64::from(size)],
            |row| ChunkSearchResult::try_from(row),
        )?;
        Ok(rows.collect::<duckdb::Result<_>>()?)
    }

    /// Up to `limit` of a document's chunks from position `from` on, in
    /// document order, with the citation metadata a search hit carries
    /// (score 1). Status is not checked: a passage cited by an earlier
    /// answer must still open after its document was replaced.
    ///
    /// # Errors
    ///
    /// Returns an error if the query fails.
    pub fn document_chunks(
        &self,
        document_id: &DocumentId,
        from: u32,
        limit: u32,
    ) -> Result<Vec<ChunkSearchResult>> {
        let mut stmt = self.conn.prepare(
            "SELECT c.id, c.content, c.document_id, c.chunk_index, d.filename, c.heading, c.page, 1.0, \
                    CAST(d.ingested_at AS VARCHAR), c.kind, c.locator \
             FROM _quack_chunks c JOIN _quack_documents d ON d.id = c.document_id \
             WHERE c.document_id = ? AND c.chunk_index >= ? \
             ORDER BY c.chunk_index LIMIT ?",
        )?;
        let rows = stmt.query_map(
            duckdb::params![document_id, i64::from(from), i64::from(limit)],
            |row| ChunkSearchResult::try_from(row),
        )?;
        Ok(rows.collect::<duckdb::Result<_>>()?)
    }

    /// Chunks by id, in the order given, with the citation metadata a
    /// search hit carries (score 1).
    ///
    /// # Errors
    ///
    /// Returns an error if the query fails.
    pub fn chunks_by_ids(&self, ids: &[ChunkId]) -> Result<Vec<ChunkSearchResult>> {
        let mut out = Vec::with_capacity(ids.len());
        let mut stmt = self.conn.prepare(
            "SELECT c.id, c.content, c.document_id, c.chunk_index, d.filename, c.heading, c.page, 1.0, \
                    CAST(d.ingested_at AS VARCHAR), c.kind, c.locator \
             FROM _quack_chunks c JOIN _quack_documents d ON d.id = c.document_id WHERE c.id = ?",
        )?;
        for id in ids {
            let mut rows = stmt.query(duckdb::params![id])?;
            if let Some(row) = rows.next()? {
                out.push(ChunkSearchResult::try_from(row)?);
            }
        }
        Ok(out)
    }

    /// Pin or unpin a document. Pinned documents are injected in full into
    /// the system prompt.
    ///
    /// # Errors
    ///
    /// Returns an error if the document does not exist or the update fails.
    pub fn set_document_pinning(&self, document_id: &DocumentId, pinning: Pinning) -> Result<()> {
        let changed = self.conn.execute(
            "UPDATE _quack_documents SET pinned = ? WHERE id = ?",
            duckdb::params![pinning, document_id],
        )?;
        if changed == 0 {
            return Err(ResourceKind::Document.missing(document_id.as_str()));
        }
        Ok(())
    }

    /// Full text of every pinned document, in chunk order.
    ///
    /// # Errors
    ///
    /// Returns an error if the query fails.
    pub fn pinned_documents(&self) -> Result<Vec<PinnedDocument>> {
        let mut out = Vec::new();
        for doc in self
            .list_documents()?
            .into_iter()
            .filter(|d| d.pinning == Pinning::Pinned)
        {
            let mut stmt = self.conn.prepare(
                "SELECT content FROM _quack_chunks WHERE document_id = ? ORDER BY chunk_index",
            )?;
            let parts = stmt
                .query_map(duckdb::params![doc.id], |row| row.get::<_, String>(0))?
                .collect::<duckdb::Result<Vec<_>>>()?;
            out.push(PinnedDocument {
                document: doc,
                text: parts.join("\n"),
            });
        }
        Ok(out)
    }

    /// Execute an arbitrary SQL statement and return every row.
    ///
    /// The owner's tool (`quack -q`): nothing bounds the result. Every
    /// other caller uses [`Self::execute_query_capped`].
    ///
    /// # Errors
    ///
    /// Returns an error if the SQL is invalid or execution fails.
    pub fn execute_query(&self, sql: &str) -> Result<QueryResults> {
        Ok(self.read_rows(sql, None)?.results)
    }

    /// Execute a statement keeping at most `max_rows` rows. Rows past the
    /// cap are counted, never converted, so a `SELECT *` over a large
    /// table costs `max_rows` values of memory on the Rust side.
    ///
    /// # Errors
    ///
    /// Returns an error if the SQL is invalid or execution fails.
    pub fn execute_query_capped(&self, sql: &str, max_rows: u32) -> Result<CappedResults> {
        self.read_rows(sql, Some(max_rows as usize))
    }

    /// [`Self::execute_query_capped`], with a `ResultDigest` of the
    /// whole result set: every row is read and digested, and only the
    /// first `max_rows` are kept.
    ///
    /// # Errors
    ///
    /// Returns an error if the SQL is invalid or execution fails.
    pub fn execute_query_digested(&self, sql: &str, max_rows: u32) -> Result<DigestedResults> {
        self.under_timeout(|db| {
            let (results, digest) = db.read_rows_untimed(sql, Some(max_rows as usize), true)?;
            Ok(DigestedResults {
                results,
                digest: digest.unwrap_or_default(),
            })
        })
    }

    fn read_rows(&self, sql: &str, keep: Option<usize>) -> Result<CappedResults> {
        self.under_timeout(|db| Ok(db.read_rows_untimed(sql, keep, false)?.0))
    }

    /// Write every row of `sql` to `out` as it arrives, in `format`, under
    /// the query timeout: the whole result set with nothing held in memory
    /// but one row. The caller classifies the statement as a read and runs
    /// this inside [`Self::read_only`]. Returns how many rows were written.
    ///
    /// # Errors
    ///
    /// Returns an error if the SQL is invalid, execution fails, or a write
    /// to `out` fails.
    pub fn stream_query(
        &self,
        sql: &str,
        format: ExportFormat,
        out: &mut impl Write,
    ) -> Result<u64> {
        self.under_timeout(|db| db.stream_query_untimed(sql, format, out))
    }

    fn stream_query_untimed(
        &self,
        sql: &str,
        format: ExportFormat,
        out: &mut impl Write,
    ) -> Result<u64> {
        let mut stmt = self.conn.prepare(sql)?;
        let mut rows = stmt.query([])?;
        let (columns, column_count) = match rows.as_ref() {
            Some(stmt_ref) if stmt_ref.column_count() > 0 => {
                (stmt_ref.column_names(), stmt_ref.column_count())
            }
            _ => (Vec::new(), 0),
        };
        let mut writer = RowWriter::start(format, &columns, out)?;
        let mut written = 0_u64;
        if column_count > 0 {
            while let Some(row) = rows.next()? {
                let mut values = Vec::with_capacity(column_count);
                for i in 0..column_count {
                    values.push(extract_value(row, i));
                }
                writer.row(&values)?;
                written = written.saturating_add(1);
            }
        }
        writer.finish()?;
        Ok(written)
    }

    /// Read `sql`'s rows, keeping `keep` of them, and when `digested`,
    /// digest every row, kept or not: a digest covers the whole result, so
    /// rows past the cap are converted only then.
    fn read_rows_untimed(
        &self,
        sql: &str,
        keep: Option<usize>,
        digested: bool,
    ) -> Result<(CappedResults, Option<String>)> {
        let mut stmt = self.conn.prepare(sql)?;
        let mut rows = stmt.query([])?;

        let (columns, column_count) = match rows.as_ref() {
            Some(stmt_ref) if stmt_ref.column_count() > 0 => {
                (stmt_ref.column_names(), stmt_ref.column_count())
            }
            _ => (Vec::new(), 0),
        };
        let mut digest = digested.then(|| ResultDigest::new(&columns)).transpose()?;
        let mut results = CappedResults {
            results: QueryResults {
                columns,
                rows: Vec::new(),
            },
            total_rows: 0,
        };
        if column_count == 0 {
            return Ok((results, digest.map(ResultDigest::finish)));
        }

        while let Some(row) = rows.next()? {
            results.total_rows = results.total_rows.saturating_add(1);
            let kept = keep.is_none_or(|keep| results.results.rows.len() < keep);
            if !kept && digest.is_none() {
                continue;
            }
            let mut values = Vec::with_capacity(column_count);
            for i in 0..column_count {
                values.push(extract_value(row, i));
            }
            if let Some(digest) = digest.as_mut() {
                digest.add_row(&values)?;
            }
            if kept {
                results.results.rows.push(values);
            }
        }
        Ok((results, digest.map(ResultDigest::finish)))
    }

    /// Execute a SQL statement that does not return rows.
    ///
    /// # Errors
    ///
    /// Returns an error if the SQL is invalid or execution fails.
    pub fn execute_statement(&self, sql: &str) -> Result<()> {
        self.execute_with_params(sql, [])
    }

    /// Execute a parameterized statement that does not return rows.
    ///
    /// # Errors
    ///
    /// Returns an error if the SQL is invalid or execution fails.
    pub fn execute_with_params<P: duckdb::Params>(&self, sql: &str, params: P) -> Result<()> {
        self.under_timeout(|db| {
            db.conn.execute(sql, params)?;
            Ok(())
        })
    }

    /// Run `f` under the statement watchdog: if it is still going after
    /// the configured query timeout, the connection is interrupted and
    /// the statement inside fails. For multi-statement work (a graph
    /// batch) that would otherwise run unbounded.
    ///
    /// # Errors
    ///
    /// Returns `f`'s error, or [`Error::QueryTimeout`] when the watchdog
    /// interrupted it.
    pub fn under_timeout<R>(&self, f: impl FnOnce(&Self) -> Result<R>) -> Result<R> {
        let guard = self.arm_timeout();
        match f(self) {
            Err(_) if guard.fired() => Err(Error::QueryTimeout {
                timeout: self.query_timeout,
            }),
            other => other,
        }
    }

    /// Run `f` so that `canceller` can interrupt its statements while it
    /// runs; one cancelled before `f` starts fails without running it.
    ///
    /// # Errors
    ///
    /// Returns `f`'s error (an interrupted statement's included), or
    /// [`Error::Cancelled`] when the work was cancelled before it started.
    pub fn cancellable<R>(
        &self,
        canceller: &QueryCanceller,
        f: impl FnOnce(&Self) -> Result<R>,
    ) -> Result<R> {
        {
            let mut slot = canceller.slot();
            if slot.cancelled {
                return Err(Error::Cancelled);
            }
            slot.running = Some(self.conn.interrupt_handle());
        }
        let result = f(self);
        canceller.slot().running = None;
        result
    }

    /// An anonymous file, removed when closed, for an export to stage
    /// workspace content in: under the workspace's own directory, or the
    /// system's temporary directory for an in-memory database.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be created.
    pub fn spool_file(&self) -> Result<std::fs::File> {
        Ok(match &self.files_dir {
            Some(dir) => tempfile::tempfile_in(dir)?,
            None => tempfile::tempfile()?,
        })
    }

    /// Run `f`'s reads inside `BEGIN TRANSACTION READ ONLY`, scoped to `f`
    /// alone: reader connections use this for every query so no transaction
    /// outlives one piece of work, pinning an old snapshot and blocking
    /// checkpointing. `DuckDB`'s `unchecked_transaction` cannot pass the
    /// `READ ONLY` modifier, so the statements are issued directly; a write
    /// `f` attempts fails with `DuckDB`'s own error, before it reaches the
    /// write-gating the agent and API already apply. Safe on the writer
    /// connection too for a statement already known to be a read (`run_sql`
    /// does this); only a statement that might write must stay outside it.
    /// Not re-entrant: `f` must not call this again on the same connection
    /// (`DuckDB` rejects a nested `BEGIN`), and it never will through
    /// today's callers, all of which run one statement or a bounded set of
    /// them without transaction control of their own.
    ///
    /// # Errors
    ///
    /// Returns `f`'s error (the transaction is rolled back), or the error
    /// from beginning or committing the transaction itself.
    pub fn read_only<R>(&self, f: impl FnOnce(&Self) -> Result<R>) -> Result<R> {
        let mut guard = ReadOnlyGuard::begin(&self.conn)?;
        let value = f(self)?;
        guard.commit()?;
        Ok(value)
    }

    /// Run `f`'s writes as one transaction: committed when `f` returns
    /// `Ok`, rolled back when it errors. A document's chunks, or a batch
    /// of embeddings, then land in one commit instead of one per row. Not
    /// re-entrant, like [`Self::read_only`].
    ///
    /// # Errors
    ///
    /// Returns `f`'s error, or the error from beginning or committing the
    /// transaction itself.
    pub fn write_transaction<R>(&self, f: impl FnOnce(&Self) -> Result<R>) -> Result<R> {
        let tx = self.conn.unchecked_transaction()?;
        let value = f(self)?;
        tx.commit()?;
        Ok(value)
    }

    /// Start a watchdog that interrupts the connection if the statement runs
    /// past the configured timeout. Dropping the guard disarms it.
    fn arm_timeout(&self) -> TimeoutGuard {
        let (disarm, armed) = std::sync::mpsc::channel::<()>();
        let handle = self.conn.interrupt_handle();
        let timeout = self.query_timeout;
        let fired = Arc::new(AtomicBool::new(false));
        let flag = Arc::clone(&fired);
        // The watchdog sleeps on the channel: dropping the guard closes it
        // and wakes the thread at once, so nothing polls (issue #62).
        std::thread::spawn(move || {
            if armed.recv_timeout(timeout) == Err(std::sync::mpsc::RecvTimeoutError::Timeout) {
                tracing::warn!(?timeout, "statement exceeded timeout; interrupting");
                flag.store(true, Ordering::SeqCst);
                handle.interrupt();
            }
        });
        TimeoutGuard {
            _disarm: disarm,
            fired,
        }
    }

    /// The user tables and their columns, for SQL completion: at most
    /// [`SqlSchema::MAX_TABLES`] tables in name order and
    /// [`SqlSchema::MAX_COLUMNS`] columns each, never an internal table.
    ///
    /// # Errors
    ///
    /// Returns an error if the query fails.
    pub fn sql_schema(&self) -> Result<SqlSchema> {
        let reserved: Vec<String> = self
            .conn
            .prepare(
                "SELECT keyword_name FROM duckdb_keywords() WHERE keyword_category = 'reserved'",
            )?
            .query_map([], |row| row.get::<_, String>(0))?
            .collect::<duckdb::Result<_>>()?;
        let mut stmt = self.conn.prepare(
            "WITH columns AS (
                 SELECT table_name, column_name,
                        dense_rank() OVER (ORDER BY table_name) AS table_rank,
                        row_number() OVER (PARTITION BY table_name ORDER BY ordinal_position) AS column_rank
                 FROM information_schema.columns
                 WHERE table_schema = 'main' AND NOT starts_with(table_name, ?)
             )
             SELECT table_name, column_name, table_rank > ? OR column_rank > ? AS cut
             FROM columns
             WHERE table_rank <= ? + 1 AND column_rank <= ? + 1
             ORDER BY table_rank, column_rank",
        )?;
        let (tables, columns) = (SqlSchema::MAX_TABLES, SqlSchema::MAX_COLUMNS);
        let mut rows = stmt.query(duckdb::params![
            INTERNAL_PREFIX,
            tables,
            columns,
            tables,
            columns
        ])?;
        let mut schema = SqlSchema::default();
        while let Some(row) = rows.next()? {
            let (table, column, cut): (String, String, bool) =
                (row.get(0)?, row.get(1)?, row.get(2)?);
            if cut {
                schema.truncated = true;
                continue;
            }
            if schema.tables.last().is_none_or(|t| t.name.name != table) {
                schema.tables.push(TableColumns {
                    name: SqlName::new(table, &reserved),
                    columns: Vec::new(),
                });
            }
            if let Some(last) = schema.tables.last_mut() {
                last.columns.push(SqlName::new(column, &reserved));
            }
        }
        Ok(schema)
    }

    /// List all user-created tables in the workspace (excludes internal tables).
    ///
    /// # Errors
    ///
    /// Returns an error if the query fails.
    pub fn list_tables(&self) -> Result<Vec<String>> {
        let mut stmt = self.conn.prepare(
            "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main' ORDER BY table_name",
        )?;
        let names = stmt
            .query_map([], |row| row.get::<_, String>(0))?
            .collect::<duckdb::Result<Vec<_>>>()?;
        Ok(names
            .into_iter()
            .filter(|name| !is_internal_name(name))
            .collect())
    }

    /// A table's columns, name and `DuckDB` type, in order.
    ///
    /// # Errors
    ///
    /// Returns an error if the table does not exist or the query fails.
    pub fn describe_columns(&self, table_name: &str) -> Result<Vec<ColumnInfo>> {
        let describe_sql = format!("DESCRIBE {}", quote_ident(table_name));
        let mut stmt = self.conn.prepare(&describe_sql)?;
        let columns = stmt
            .query_map([], |row| {
                Ok(ColumnInfo {
                    name: row.get(0)?,
                    column_type: row.get(1)?,
                    meaning: None,
                })
            })?
            .collect::<duckdb::Result<Vec<_>>>()?;
        Ok(columns)
    }

    /// Describe a table: its columns with what the ontology says of them,
    /// up to 3 sample rows, the owner's note, its profile when current,
    /// and the measures defined over it.
    ///
    /// # Errors
    ///
    /// Returns an error if the table does not exist or the query fails.
    pub fn describe_table(&self, table_name: &str) -> Result<TableDescription> {
        let ontology = ontology_store::current(self)?;
        self.describe_table_under(table_name, ontology.as_ref())
    }

    /// [`Self::describe_table`] with the ontology already read, for a
    /// caller describing many tables.
    ///
    /// # Errors
    ///
    /// Returns an error if the table does not exist or the query fails.
    pub fn describe_table_under(
        &self,
        table_name: &str,
        ontology: Option<&Ontology>,
    ) -> Result<TableDescription> {
        let mut columns = self.describe_columns(table_name)?;
        let sample_sql = format!("SELECT * FROM {} LIMIT 3", quote_ident(table_name));
        let sample = self.execute_query(&sample_sql)?;
        let row_count = self.count_rows(table_name)?;
        let mapping = ontology.and_then(|o| o.mapping_for_table(table_name));
        if let (Some(ontology), Some(mapping)) = (ontology, mapping) {
            for column in &mut columns {
                column.meaning = mapping
                    .properties
                    .get(&column.name)
                    .and_then(|id| ontology.property(id))
                    .and_then(ColumnMeaning::of);
            }
        }
        let profile =
            TableProfile::current(self, table_name, u64::try_from(row_count).unwrap_or(0))?;
        let warnings = profile
            .as_ref()
            .map(|p| p.warnings(mapping.map(|m| m.key.as_str())))
            .unwrap_or_default();
        Ok(TableDescription {
            table_name: table_name.to_owned(),
            columns,
            row_count,
            sample_rows: sample,
            note: TableNote::get(self, table_name)?.map(|n| n.note),
            profile,
            warnings,
            measures: ontology
                .map(|o| o.measures_on(table_name).into_iter().cloned().collect())
                .unwrap_or_default(),
        })
    }

    /// Exact row count of one table.
    ///
    /// # Errors
    ///
    /// Returns an error if the table does not exist or the query fails.
    pub fn count_rows(&self, table_name: &str) -> Result<i64> {
        let sql = format!("SELECT count(*) FROM {}", quote_ident(table_name));
        let count: i64 = self.conn.query_row(&sql, [], |row| row.get(0))?;
        Ok(count)
    }

    /// The version of the `DuckDB` library compiled into this binary, such as
    /// `v1.5.0`; the system prompt pins its dialect notes to it.
    ///
    /// # Errors
    ///
    /// Returns an error if the version query fails.
    pub fn duckdb_version(&self) -> Result<String> {
        let version: String =
            self.conn
                .query_row("SELECT library_version FROM pragma_version()", [], |row| {
                    row.get(0)
                })?;
        Ok(version)
    }

    /// The `limit` most recently ingested documents, and how many there are
    /// in all: the bounded inventory the system prompt carries.
    ///
    /// # Errors
    ///
    /// Returns an error if the query fails.
    pub fn recent_documents(&self, limit: usize) -> Result<(Vec<DocumentInfo>, usize)> {
        let sql = format!(
            "{DOCUMENT_SELECT} WHERE {NOT_SUPERSEDED} ORDER BY ingested_at DESC, id DESC LIMIT ?"
        );
        let limit = i64::try_from(limit).unwrap_or(i64::MAX);
        let mut stmt = self.conn.prepare(&sql)?;
        let docs = stmt
            .query_map(duckdb::params![limit], |row| DocumentInfo::try_from(row))?
            .collect::<duckdb::Result<Vec<_>>>()?;
        let total: i64 = self.conn.query_row(
            &format!("SELECT count(*) FROM _quack_documents WHERE {NOT_SUPERSEDED}"),
            [],
            |row| row.get(0),
        )?;
        Ok((docs, usize::try_from(total).unwrap_or(usize::MAX)))
    }

    /// Every document with its status, newest first, failed ones with
    /// their reason: what the workspace holds now. A replaced document is
    /// left out ([`Self::list_all_documents`] has it).
    ///
    /// # Errors
    ///
    /// Returns an error if the query fails.
    pub fn list_documents(&self) -> Result<Vec<DocumentInfo>> {
        let sql =
            format!("{DOCUMENT_SELECT} WHERE {NOT_SUPERSEDED} ORDER BY ingested_at DESC, id DESC");
        let mut stmt = self.conn.prepare(&sql)?;
        let docs = stmt.query_map([], |row| DocumentInfo::try_from(row))?;
        Ok(docs.collect::<duckdb::Result<_>>()?)
    }

    /// The documents [`Self::list_documents`] lists that `filter` lets
    /// through.
    ///
    /// # Errors
    ///
    /// An unknown file type, `since` after `until`, or a failed query.
    pub fn list_documents_matching(&self, filter: &DocumentFilter) -> Result<Vec<DocumentInfo>> {
        let clause = filter.clause()?;
        let sql = format!(
            "{DOCUMENT_SELECT} d WHERE {NOT_SUPERSEDED}{} ORDER BY ingested_at DESC, id DESC",
            clause.sql
        );
        let mut stmt = self.conn.prepare(&sql)?;
        let mut params: Vec<&dyn duckdb::ToSql> = Vec::with_capacity(clause.params.len());
        for param in &clause.params {
            params.push(param);
        }
        let docs = stmt.query_map(params.as_slice(), |row| DocumentInfo::try_from(row))?;
        Ok(docs.collect::<duckdb::Result<_>>()?)
    }

    /// Every document row, replaced ones included, newest first: the
    /// listing behind `quack docs --all` and the Documents page's
    /// "show replaced" view.
    ///
    /// # Errors
    ///
    /// Returns an error if the query fails.
    pub fn list_all_documents(&self) -> Result<Vec<DocumentInfo>> {
        let sql = format!("{DOCUMENT_SELECT} ORDER BY ingested_at DESC, id DESC");
        let mut stmt = self.conn.prepare(&sql)?;
        let docs = stmt.query_map([], |row| DocumentInfo::try_from(row))?;
        Ok(docs.collect::<duckdb::Result<_>>()?)
    }

    /// Access the underlying `DuckDB` connection.
    #[must_use]
    pub fn connection(&self) -> &duckdb::Connection {
        &self.conn
    }
}

/// Column metadata from DESCRIBE, with what the ontology says of it.
#[derive(Debug, Clone, serde::Serialize)]
pub struct ColumnInfo {
    pub name: String,
    #[serde(rename = "type")]
    pub column_type: String,
    /// From the property a table mapping gives the column; `None` when the
    /// table is not mapped or the property says nothing.
    #[serde(flatten)]
    pub meaning: Option<ColumnMeaning>,
}

/// What a column means, from its ontology property.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub struct ColumnMeaning {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub unit: Option<String>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub synonyms: Vec<String>,
}

impl ColumnMeaning {
    /// What `property` says, or `None` when it says nothing.
    #[must_use]
    pub fn of(property: &Property) -> Option<Self> {
        let meaning = Self {
            description: property.description.clone(),
            unit: property.unit.clone(),
            synonyms: property.synonyms.clone(),
        };
        (meaning.description.is_some() || meaning.unit.is_some() || !meaning.synonyms.is_empty())
            .then_some(meaning)
    }
}

/// `: monthly revenue [USD] (also: sales, turnover)`, the suffix a column
/// line carries.
impl fmt::Display for ColumnMeaning {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(description) = &self.description {
            write!(f, ": {}", OneLine(description))?;
        }
        if let Some(unit) = &self.unit {
            write!(f, " [{}]", OneLine(unit))?;
        }
        if !self.synonyms.is_empty() {
            write!(f, " (also: {})", OneLine(&self.synonyms.join(", ")))?;
        }
        Ok(())
    }
}

/// The user tables and columns SQL completion offers.
#[derive(Debug, Clone, Default, PartialEq, Eq, serde::Serialize)]
pub struct SqlSchema {
    pub tables: Vec<TableColumns>,
    /// Some tables or columns were left out to stay within the caps.
    pub truncated: bool,
}

impl SqlSchema {
    /// The most tables one schema carries.
    pub const MAX_TABLES: u32 = 500;
    /// The most columns one table carries.
    pub const MAX_COLUMNS: u32 = 200;

    /// The table named `name`, compared without regard to case.
    #[must_use]
    pub fn table(&self, name: &str) -> Option<&TableColumns> {
        self.tables
            .iter()
            .find(|t| t.name.name.eq_ignore_ascii_case(name))
    }
}

/// One table's name and its columns, in column order.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub struct TableColumns {
    pub name: SqlName,
    pub columns: Vec<SqlName>,
}

/// An identifier and how a statement writes it: bare when `DuckDB` reads
/// it unquoted as the same name (lowercase letters, digits, and
/// underscores, not a reserved keyword), quoted otherwise.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub struct SqlName {
    pub name: String,
    pub sql: String,
}

impl SqlName {
    fn new(name: String, reserved: &[String]) -> Self {
        let plain = name
            .chars()
            .next()
            .is_some_and(|c| c.is_ascii_lowercase() || c == '_')
            && name
                .chars()
                .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_');
        let sql = if plain && !reserved.contains(&name) {
            name.clone()
        } else {
            quote_ident(&name)
        };
        Self { name, sql }
    }
}

/// Full table description with schema and sample data.
#[derive(Debug)]
pub struct TableDescription {
    pub table_name: String,
    pub columns: Vec<ColumnInfo>,
    /// Exact row count at describe time.
    pub row_count: i64,
    pub sample_rows: QueryResults,
    /// The owner's note on the table.
    pub note: Option<String>,
    /// The stored profile, when it was taken at the current row count.
    pub profile: Option<TableProfile>,
    /// The profile's warnings, the mapped key column held to the key rule.
    pub warnings: Vec<profile::Flagged>,
    /// The ontology's measures over this table.
    pub measures: Vec<Measure>,
}

impl TableDescription {
    /// The description as every interface sends it: columns with their
    /// meaning, the note, the profile's counts per column, each warning with
    /// its sentence and the type that fixes it, the measures, and the
    /// sample rows.
    #[must_use]
    pub fn to_json(&self) -> serde_json::Value {
        let warnings: Vec<serde_json::Value> = self
            .warnings
            .iter()
            .map(|f| {
                let mut value = serde_json::json!({
                    "column": f.column,
                    "message": f.warning.to_string(),
                    "fix": f.warning.fix(),
                });
                if let (Some(object), Ok(serde_json::Value::Object(kind))) =
                    (value.as_object_mut(), serde_json::to_value(f.warning))
                {
                    object.extend(kind);
                }
                value
            })
            .collect();
        serde_json::json!({
            "table": self.table_name,
            "row_count": self.row_count,
            "note": self.note,
            "columns": self.columns,
            "profile": self.profile,
            "warnings": warnings,
            "measures": self.measures,
            "sample": { "columns": self.sample_rows.columns, "rows": self.sample_rows.rows },
        })
    }
}

/// Where a document is in ingestion.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum DocumentStatus {
    /// Registered; its bytes wait on the work queue.
    Queued,
    /// Being parsed, chunked, and embedded, or loaded as a table.
    Processing,
    /// Searchable, or its table loaded.
    Ready,
    /// Failed; `error_message` says why.
    Error,
    /// Replaced by the document `superseded_by` names: no longer searched,
    /// listed, or read, though its chunks stay so earlier citations open.
    Superseded,
}

text_enum!(DocumentStatus, "document status", {
    Queued => "queued",
    Processing => "processing",
    Ready => "ready",
    Error => "error",
    Superseded => "superseded",
});

impl DocumentStatus {
    /// Still on its way to `ready` or `error`.
    #[must_use]
    pub fn is_in_flight(self) -> bool {
        match self {
            Self::Queued | Self::Processing => true,
            Self::Ready | Self::Error | Self::Superseded => false,
        }
    }
}

text_enum_sql!(DocumentStatus);

/// Document metadata row.
#[derive(Debug, Clone, serde::Serialize)]
pub struct DocumentInfo {
    pub id: DocumentId,
    pub filename: String,
    /// Given at upload or parsed from the content (first heading, HTML
    /// title); `None` when neither exists.
    pub title: Option<String>,
    pub mime_type: Option<String>,
    pub size_bytes: Option<i64>,
    /// Lowercase hex SHA-256 of the bytes; `None` only for rows written
    /// before the column existed.
    pub sha256: Option<String>,
    pub source: DocumentSource,
    pub status: DocumentStatus,
    pub error_message: Option<String>,
    #[serde(rename = "pinned")]
    pub pinning: Pinning,
    /// Chunks stored once processed; `None` until then and for tables.
    pub chunk_count: Option<i64>,
    /// Server user who uploaded it; `None` from the CLI.
    pub ingested_by: Option<String>,
    /// Tables a structured document loaded into; `None` until processed
    /// and for rows written before this was recorded.
    pub tables: Option<Vec<String>>,
    /// How a PDF's pages read; `None` for other sources, until processed,
    /// and for rows written before this was recorded.
    pub pages: Option<PageCounts>,
    pub ingested_at: String,
    /// The document replacing this one: on its way while this one is
    /// still `ready`, in place once this one is `superseded`.
    pub superseded_by: Option<DocumentId>,
    /// The folder the file was ingested from, as its canonical absolute
    /// path; `None` for a document from anywhere else.
    pub source_root: Option<String>,
    /// Where the file was under that folder, with `/` separators; a later
    /// run of the folder matches the file by root and path together.
    pub source_path: Option<String>,
    /// Who wrote the document, as the file says or a person set it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub author: Option<String>,
    /// When it was written (the file's creation date, front matter's
    /// `date`, a mail's `Date`), as `YYYY-MM-DD HH:MM:SS` when the source
    /// gave a date the store could parse.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub authored_at: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub modified_at: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tags: Vec<String>,
    /// Other named values the file carried (a subject, recipients, a
    /// description).
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub metadata: BTreeMap<String, String>,
    /// The ISO 639-3 code of the language its text was indexed under
    /// (`deu`, `cmn`); `None` for a table or a document not yet processed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub language: Option<String>,
}

impl DocumentInfo {
    /// The title when one exists, else the filename.
    #[must_use]
    pub fn display_name(&self) -> &str {
        self.title.as_deref().unwrap_or(&self.filename)
    }

    /// The document in `documents` the model named: by id, exact file
    /// name, or id prefix.
    ///
    /// # Errors
    ///
    /// A name that matches none is an error listing the documents there
    /// are, so the caller corrects it rather than reading an empty result
    /// as "the workspace has nothing on this".
    pub fn find<'a>(documents: &'a [Self], want: &str) -> Result<&'a Self> {
        let want = want.trim();
        let found = documents
            .iter()
            .find(|d| d.id.as_str() == want || d.filename == want)
            .or_else(|| {
                documents
                    .iter()
                    .find(|d| !want.is_empty() && d.id.as_str().starts_with(want))
            });
        found.ok_or_else(|| {
            let known: Vec<String> = documents
                .iter()
                .map(|d| format!("{} ({})", d.id, OneLine(&d.filename)))
                .collect();
            Error::Analysis(format!(
                "no document matches '{want}'; pass an id (a prefix is enough) or an exact \
                 file name from list_documents. Documents: {}",
                known.join(", ")
            ))
        })
    }

    /// The tables a row from before `tables` was recorded loaded into: the
    /// one named after the file, for a CSV, Parquet, or JSON file. Workbooks
    /// arrived with the `tables` column, so their rows always carry it.
    #[must_use]
    pub fn fallback_tables(&self) -> Vec<String> {
        match FileType::of(&self.filename).map(FileType::load) {
            Some(Load::Table(_)) => vec![TableName::of_file(&self.filename).into_string()],
            Some(Load::Workbook | Load::Chunks(_)) | None => Vec::new(),
        }
    }
}

/// How a document reached the workspace.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize, schemars::JsonSchema,
)]
#[serde(rename_all = "snake_case")]
pub enum DocumentSource {
    /// A file sent through the API or web UI.
    Upload,
    /// Text pasted through the API or web UI.
    Paste,
    /// A file path given to the CLI or terminal.
    Path,
    /// Bytes piped into the CLI.
    Stdin,
    /// Rows pulled from an external database or a URL (`quack import`).
    Import,
}

text_enum!(DocumentSource, "document source", {
    Upload => "upload",
    Paste => "paste",
    Path => "path",
    Stdin => "stdin",
    Import => "import",
});

impl DocumentSource {
    /// The stored column: rows written before it existed are uploads; any
    /// other unknown text is an error rather than a silent guess.
    fn from_column(value: Option<String>) -> duckdb::Result<Self> {
        value.map_or(Ok(Self::Upload), |text| {
            text.parse().map_err(|e: Error| {
                duckdb::Error::FromSqlConversionFailure(10, duckdb::types::Type::Text, Box::new(e))
            })
        })
    }
}

/// What a person may change on a document: each `Some` is applied, an
/// empty text clears the field.
#[derive(Debug, Clone, Default, PartialEq, Eq, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DocumentFields {
    pub title: Option<String>,
    pub author: Option<String>,
    pub authored_at: Option<String>,
    pub tags: Option<Vec<String>>,
}

impl DocumentFields {
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.title.is_none()
            && self.author.is_none()
            && self.authored_at.is_none()
            && self.tags.is_none()
    }
}

/// A document row to insert.
#[derive(Debug, Clone, Copy)]
pub struct NewDocument<'a> {
    pub id: &'a DocumentId,
    pub filename: &'a str,
    pub title: Option<&'a str>,
    pub mime_type: &'a str,
    pub size_bytes: usize,
    pub sha256: &'a str,
    pub source: DocumentSource,
    pub status: DocumentStatus,
    pub ingested_by: Option<&'a str>,
    /// The folder the file came from and its path under it, when it came
    /// from one.
    pub source_root: Option<&'a str>,
    pub source_path: Option<&'a str>,
}

impl<'a> NewDocument<'a> {
    /// A `queued` upload with no title, hash, or uploader; the fields are
    /// public for the rest.
    #[must_use]
    pub fn new(
        id: &'a DocumentId,
        filename: &'a str,
        mime_type: &'a str,
        size_bytes: usize,
    ) -> Self {
        Self {
            id,
            filename,
            title: None,
            mime_type,
            size_bytes,
            sha256: "",
            source: DocumentSource::Upload,
            status: DocumentStatus::Queued,
            ingested_by: None,
            source_root: None,
            source_path: None,
        }
    }

    #[must_use]
    pub fn with_status(mut self, status: DocumentStatus) -> Self {
        self.status = status;
        self
    }
}

const DOCUMENT_SELECT: &str = "SELECT id, filename, mime_type, size_bytes, status, error_message, \
     COALESCE(pinned, false), CAST(ingested_at AS VARCHAR), title, sha256, source, chunk_count, \
     ingested_by, CAST(tables AS VARCHAR), page_count, pages_unreadable, pages_empty, \
     superseded_by, source_root, source_path, author, CAST(authored_at AS VARCHAR), \
     CAST(modified_at AS VARCHAR), CAST(tags AS VARCHAR), CAST(metadata AS VARCHAR), language \
     FROM _quack_documents";

/// The `WHERE` clause that keeps a document that still stands for its
/// bytes, neither failed nor replaced: only such a document is a duplicate
/// of a re-upload or owns a table.
const LIVE_STATUS: &str = "status NOT IN ('error', 'superseded')";

/// The `WHERE` clause of every listing of what the workspace holds now: a
/// failed document stays listed with its reason, a replaced one does not.
const NOT_SUPERSEDED: &str = "status <> 'superseded'";

/// A row selected with `DOCUMENT_SELECT`.
impl TryFrom<&duckdb::Row<'_>> for DocumentInfo {
    type Error = duckdb::Error;

    fn try_from(row: &duckdb::Row<'_>) -> duckdb::Result<Self> {
        Ok(Self {
            id: row.get(0)?,
            filename: row.get(1)?,
            mime_type: row.get(2)?,
            size_bytes: row.get(3)?,
            status: row.get(4)?,
            error_message: row.get(5)?,
            pinning: row.get(6)?,
            ingested_at: row.get(7)?,
            title: row.get(8)?,
            sha256: row.get(9)?,
            source: DocumentSource::from_column(row.get(10)?)?,
            chunk_count: row.get(11)?,
            ingested_by: row.get(12)?,
            tables: row
                .get::<_, Option<String>>(13)?
                .and_then(|json| serde_json::from_str(&json).ok()),
            pages: match row.get::<_, Option<u32>>(14)? {
                Some(total) => Some(PageCounts {
                    total,
                    unreadable: row.get::<_, Option<u32>>(15)?.unwrap_or(0),
                    empty: row.get::<_, Option<u32>>(16)?.unwrap_or(0),
                }),
                None => None,
            },
            superseded_by: row.get(17)?,
            source_root: row.get(18)?,
            source_path: row.get(19)?,
            author: row.get(20)?,
            authored_at: row.get(21)?,
            modified_at: row.get(22)?,
            tags: row
                .get::<_, Option<String>>(23)?
                .and_then(|json| serde_json::from_str(&json).ok())
                .unwrap_or_default(),
            metadata: row
                .get::<_, Option<String>>(24)?
                .and_then(|json| serde_json::from_str(&json).ok())
                .unwrap_or_default(),
            language: row.get(25)?,
        })
    }
}

/// A chunk whose vector is missing or stale.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PendingChunk {
    pub id: ChunkId,
    pub heading: Option<String>,
    pub content: String,
}

impl PendingChunk {
    /// What the embedder is given for this chunk: its text under its
    /// heading, as at ingestion.
    #[must_use]
    pub fn embedding_input(&self) -> Input {
        Input::Document {
            title: self.heading.clone(),
            text: self.content.clone(),
        }
    }
}

/// A row of `id, heading, content`.
impl TryFrom<&duckdb::Row<'_>> for PendingChunk {
    type Error = duckdb::Error;

    fn try_from(row: &duckdb::Row<'_>) -> duckdb::Result<Self> {
        Ok(Self {
            id: row.get(0)?,
            heading: row.get(1)?,
            content: row.get(2)?,
        })
    }
}

/// A chunk to store.
#[derive(Debug, Clone, Copy)]
pub struct NewChunk<'a> {
    pub id: &'a ChunkId,
    pub document_id: &'a DocumentId,
    pub chunk_index: u32,
    pub content: &'a str,
    pub heading: Option<&'a str>,
    pub page: Option<u32>,
    pub embedding: Option<&'a Vector>,
    pub kind: SectionKind,
    /// Where the chunk sits in a source without pages (`line 40`, `12:04`,
    /// `chapter 3`, `message 2`).
    pub locator: Option<&'a str>,
}

/// A chunk returned from retrieval, with what a citation needs.
#[derive(Debug, Clone, serde::Serialize)]
pub struct ChunkSearchResult {
    pub id: ChunkId,
    pub content: String,
    pub document_id: DocumentId,
    pub chunk_index: u32,
    pub filename: String,
    pub heading: Option<String>,
    pub page: Option<u32>,
    /// Higher is better. Vector-only: `1 / (1 + distance)`; keyword-only:
    /// BM25; hybrid: reciprocal rank fusion.
    pub score: f64,
    /// When the chunk's document was ingested (UTC).
    pub ingested_at: DateTime,
    /// What the chunk holds: body text, a table, a note, or code.
    #[serde(default)]
    pub kind: SectionKind,
    /// Where it sits in a source without pages.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub locator: Option<String>,
    /// Where it stood in each ranking that found it.
    #[serde(flatten)]
    pub ranks: Ranks,
}

/// Where a hit stood in each ranking of one search, for a person checking
/// why retrieval found or missed a passage. Ranks count from 1; a ranking
/// that did not return the hit leaves its fields empty.
#[derive(Debug, Clone, Copy, Default, PartialEq, serde::Serialize)]
pub struct Ranks {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub vector_rank: Option<u32>,
    /// `1 / (1 + cosine distance)`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub vector_score: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub keyword_rank: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bm25: Option<f64>,
    /// Its place in the reranker's order.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rerank_rank: Option<u32>,
    /// The rerank model's relevance score; the chat model ranks without one.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rerank_score: Option<f64>,
}

impl Ranks {
    /// A 1-based rank from a 0-based position.
    #[must_use]
    pub fn place(index: usize) -> u32 {
        u32::try_from(index).unwrap_or(u32::MAX).saturating_add(1)
    }

    /// Keep every rank `other` has that this one lacks.
    fn merge(&mut self, other: Self) {
        self.vector_rank = self.vector_rank.or(other.vector_rank);
        self.vector_score = self.vector_score.or(other.vector_score);
        self.keyword_rank = self.keyword_rank.or(other.keyword_rank);
        self.bm25 = self.bm25.or(other.bm25);
        self.rerank_rank = self.rerank_rank.or(other.rerank_rank);
        self.rerank_score = self.rerank_score.or(other.rerank_score);
    }
}

/// A row of `id, content, document_id, chunk_index, filename, heading,
/// page, score, ingested_at, kind, locator`, the columns every search
/// selects. A chunk stored before `kind` existed reads as body text.
impl TryFrom<&duckdb::Row<'_>> for ChunkSearchResult {
    type Error = duckdb::Error;

    fn try_from(row: &duckdb::Row<'_>) -> duckdb::Result<Self> {
        let page: Option<i64> = row.get(6)?;
        let ingested_at: String = row.get(8)?;
        let ingested_at = ingested_at.parse().map_err(|e: jiff::Error| {
            duckdb::Error::FromSqlConversionFailure(8, duckdb::types::Type::Text, Box::new(e))
        })?;
        Ok(Self {
            id: row.get(0)?,
            content: row.get(1)?,
            document_id: row.get(2)?,
            chunk_index: row.get(3)?,
            filename: row.get(4)?,
            heading: row.get(5)?,
            page: page.and_then(|p| u32::try_from(p).ok()),
            score: row.get(7)?,
            ingested_at,
            kind: row.get::<_, Option<SectionKind>>(9)?.unwrap_or_default(),
            locator: row.get(10)?,
            ranks: Ranks::default(),
        })
    }
}

/// The quoted phrases of a keyword query: each `"..."` pair is an exact
/// adjacency requirement, checked as a substring after retrieval since
/// `_quack_terms` carries no positions.
struct Phrases(Vec<String>);

impl Phrases {
    /// An odd number of `"` characters is an unbalanced quote, so the whole
    /// query is left as ordinary text instead of guessing which quote was
    /// meant to close.
    fn parse(query: &str) -> Self {
        if !query.matches('"').count().is_multiple_of(2) {
            return Self(Vec::new());
        }
        Self(
            query
                .split('"')
                .enumerate()
                .filter_map(|(i, s)| (i % 2 == 1).then_some(s.trim()))
                .filter(|s| !s.is_empty())
                .map(String::from)
                .collect(),
        )
    }

    /// How many candidates to fetch for `top_k` results: `top_k` without a
    /// phrase, else `base` over-fetched, since the filter drops some.
    fn fetch(&self, top_k: u32, base: u32) -> u32 {
        if self.0.is_empty() {
            top_k
        } else {
            base.saturating_mul(PHRASE_OVER_FETCH)
                .min(PHRASE_CANDIDATE_CAP)
                .max(top_k)
        }
    }

    /// Keep the first `top_k` results whose content or heading contains
    /// every phrase, in their ranked order. Hybrid search runs this on the
    /// fused ranking, since its vector leg knows nothing of phrases.
    fn retain_matching(&self, results: &mut Vec<ChunkSearchResult>, top_k: u32) {
        if self.0.is_empty() {
            return;
        }
        results.retain(|r| {
            self.0.iter().all(|phrase| {
                Self::contains(&r.content, phrase)
                    || r.heading
                        .as_deref()
                        .is_some_and(|h| Self::contains(h, phrase))
            })
        });
        results.truncate(usize::try_from(top_k).unwrap_or(usize::MAX));
    }

    /// Whether `phrase` occurs in `text`, ignoring case and how the
    /// whitespace runs (a chunk may wrap a phrase across a line break).
    fn contains(text: &str, phrase: &str) -> bool {
        let normalize = |s: &str| {
            s.split_whitespace()
                .collect::<Vec<_>>()
                .join(" ")
                .to_lowercase()
        };
        normalize(text).contains(&normalize(phrase))
    }
}

/// The chunks of ready documents a long run draws from, and the order
/// its documents come in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SamplePool {
    /// Chunks the graph has not extracted yet, documents in ingest order.
    NotGraphExtracted,
    /// Chunks with more than a line of text, documents by id: what the
    /// ontology's document evidence reads.
    Substantive,
}

impl SamplePool {
    /// The `AND ...` condition on `_quack_chunks c`.
    const fn filter(self) -> &'static str {
        match self {
            Self::NotGraphExtracted => {
                "NOT EXISTS (SELECT 1 FROM _quack_graph_extracted x WHERE x.chunk_id = c.id)"
            }
            Self::Substantive => "length(c.content) > 40",
        }
    }

    /// How the documents are ordered, in the sampler's own columns: when
    /// the document was ingested, and its id.
    const fn document_order(self) -> &'static str {
        match self {
            Self::NotGraphExtracted => "ingested_at, doc",
            Self::Substantive => "doc",
        }
    }
}

/// Which rankings a search runs.
#[derive(
    Debug,
    Clone,
    Copy,
    Default,
    PartialEq,
    Eq,
    serde::Serialize,
    serde::Deserialize,
    schemars::JsonSchema,
)]
#[serde(rename_all = "lowercase")]
pub enum SearchMode {
    /// Vector and keyword, fused by reciprocal rank; keyword alone without
    /// an embedding model.
    #[default]
    Hybrid,
    /// BM25 over the term index only.
    Keyword,
    /// Cosine similarity only.
    Vector,
}

text_enum!(SearchMode, "search mode", {
    Hybrid => "hybrid",
    Keyword => "keyword",
    Vector => "vector",
});

/// What a document must be for a search or a listing to include it. Each
/// list keeps a document matching any of its entries; every field given
/// must hold.
#[derive(
    Debug, Clone, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize, schemars::JsonSchema,
)]
#[serde(deny_unknown_fields)]
pub struct DocumentFilter {
    /// File types, as extensions (`pdf`, `md`, `docx`) or MIME types
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub types: Vec<String>,
    /// How documents arrived: `upload`, `paste`, `path`, `stdin`, or `import`
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub sources: Vec<DocumentSource>,
    /// Tags, compared without regard to case
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tags: Vec<String>,
    /// Written on or after this date, `YYYY-MM-DD` (the ingest date for a
    /// document that gives none)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(with = "Option<String>")]
    pub since: Option<jiff::civil::Date>,
    /// Written on or before this date, `YYYY-MM-DD`
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(with = "Option<String>")]
    pub until: Option<jiff::civil::Date>,
    /// Text the author contains, compared without regard to case
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub author: Option<String>,
}

impl DocumentFilter {
    /// Whether it lets every document through.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        *self == Self::default()
    }

    /// The `AND ...` conditions on `_quack_documents d`, with their
    /// parameters in order.
    ///
    /// # Errors
    ///
    /// An unknown file type, or `since` after `until`.
    fn clause(&self) -> Result<FilterClause> {
        let mut clause = FilterClause::default();
        let mut mime_types: Vec<String> = Vec::new();
        for kind in &self.types {
            let kind = kind.trim().trim_start_matches('.').to_ascii_lowercase();
            let mime = if kind.contains('/') {
                kind
            } else {
                FileType::of(&format!("file.{kind}"))
                    .map(|t| t.mime_type().to_owned())
                    .ok_or_else(|| {
                        Error::Analysis(format!(
                            "unknown document type '{kind}'; give an extension such as pdf, md, \
                             or docx, or a MIME type"
                        ))
                    })?
            };
            if !mime_types.contains(&mime) {
                mime_types.push(mime);
            }
        }
        clause.any_of("d.mime_type", mime_types);
        clause.any_of(
            "COALESCE(d.source, 'upload')",
            self.sources.iter().map(ToString::to_string).collect(),
        );
        let tags: Vec<String> = self
            .tags
            .iter()
            .map(|t| t.trim().to_lowercase())
            .filter(|t| !t.is_empty())
            .collect();
        if !tags.is_empty() {
            clause.sql.push_str(
                " AND list_has_any(COALESCE(from_json(lower(CAST(d.tags AS VARCHAR)), '[\"VARCHAR\"]'), []), ?::VARCHAR[])",
            );
            clause.params.push(sql_text_list(&tags));
        }
        if let (Some(since), Some(until)) = (self.since, self.until)
            && since > until
        {
            return Err(Error::Analysis(format!(
                "since ({since}) is after until ({until})"
            )));
        }
        for (bound, op) in [(self.since, ">="), (self.until, "<=")] {
            if let Some(date) = bound {
                clause
                    .sql
                    .push_str(" AND CAST(COALESCE(d.authored_at, d.ingested_at) AS DATE) ");
                clause.sql.push_str(op);
                clause.sql.push_str(" ?::DATE");
                clause.params.push(date.to_string());
            }
        }
        if let Some(author) = self
            .author
            .as_deref()
            .map(str::trim)
            .filter(|a| !a.is_empty())
        {
            clause
                .sql
                .push_str(" AND contains(lower(COALESCE(d.author, '')), lower(?))");
            clause.params.push(author.to_owned());
        }
        Ok(clause)
    }
}

/// A [`DocumentFilter`] as SQL: conditions on `_quack_documents d` and their
/// text parameters, in order.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
struct FilterClause {
    sql: String,
    params: Vec<String>,
}

impl FilterClause {
    /// `AND column IN (...)` over `values`, unless there are none.
    fn any_of(&mut self, column: &str, values: Vec<String>) {
        if values.is_empty() {
            return;
        }
        self.sql.push_str(" AND ");
        self.sql.push_str(column);
        self.sql.push_str(" IN (");
        self.sql.push_str(&vec!["?"; values.len()].join(", "));
        self.sql.push(')');
        self.params.extend(values);
    }
}

/// Which chunks a search may return. Empty means the whole workspace; a
/// scope narrows it to certain documents, to an explicit set of chunks
/// (the chunks a graph entity was extracted from), or to both at once.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ChunkScope {
    /// Empty means every document: `ChunkScope::for_documents` refuses an id
    /// that matches nothing, so an empty list never means "none".
    documents: Vec<DocumentId>,
    /// `None` means no chunk restriction at all; `Some(ids)` means exactly
    /// those chunks, and `Some(empty)` means none — an entity whose chunks
    /// came back empty must return no rows, not the whole workspace.
    chunks: Option<Vec<ChunkId>>,
    /// What the chunks' documents must be.
    filter: FilterClause,
}

impl ChunkScope {
    /// Every ready chunk in the workspace.
    #[must_use]
    pub fn all() -> Self {
        Self::default()
    }

    /// Only chunks belonging to these documents.
    #[must_use]
    pub fn documents<I: IntoIterator<Item = DocumentId>>(ids: I) -> Self {
        Self {
            documents: ids.into_iter().collect(),
            chunks: None,
            filter: FilterClause::default(),
        }
    }

    /// Only chunks of the documents `wanted` names, each by id, id prefix,
    /// or file name; every document when `wanted` is empty.
    ///
    /// # Errors
    ///
    /// A name that matches no document is an error listing the documents
    /// there are, so the caller corrects it rather than reading an empty
    /// result as "the workspace has nothing on this".
    pub fn for_documents(db: &WorkspaceDb, wanted: &[String]) -> Result<Self> {
        if wanted.is_empty() {
            return Ok(Self::all());
        }
        let documents = db.list_documents()?;
        let mut resolved = Vec::with_capacity(wanted.len());
        for want in wanted {
            resolved.push(DocumentInfo::find(&documents, want)?.id.clone());
        }
        Ok(Self::documents(resolved))
    }

    /// Narrow further to the documents `filter` lets through.
    ///
    /// # Errors
    ///
    /// An unknown file type, or `since` after `until`.
    pub fn with_filter(mut self, filter: &DocumentFilter) -> Result<Self> {
        self.filter = filter.clause()?;
        Ok(self)
    }

    /// The documents it is limited to; empty for every document.
    #[must_use]
    pub fn document_ids(&self) -> &[DocumentId] {
        &self.documents
    }

    /// Narrow further to these chunk ids, however few.
    #[must_use]
    pub fn and_chunks<I: IntoIterator<Item = ChunkId>>(mut self, ids: I) -> Self {
        self.chunks = Some(ids.into_iter().collect());
        self
    }

    /// Whether the scope names nothing at all, so a search must return no
    /// rows instead of widening to the whole workspace.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.chunks.as_ref().is_some_and(Vec::is_empty)
    }

    /// The `AND ...` fragment for a query that has `_quack_chunks` as `c`.
    fn sql(&self) -> String {
        let mut clauses = Vec::new();
        if !self.documents.is_empty() {
            let placeholders = vec!["?"; self.documents.len()].join(", ");
            clauses.push(format!(" AND c.document_id IN ({placeholders})"));
        }
        if let Some(chunks) = self.chunks.as_ref().filter(|ids| !ids.is_empty()) {
            let placeholders = vec!["?"; chunks.len()].join(", ");
            clauses.push(format!(" AND c.id IN ({placeholders})"));
        }
        clauses.push(self.filter.sql.clone());
        clauses.concat()
    }

    /// How many parameters [`ChunkScope::bind`] will push.
    fn len(&self) -> usize {
        self.documents
            .len()
            .saturating_add(self.chunks.as_ref().map_or(0, Vec::len))
            .saturating_add(self.filter.params.len())
    }

    /// Push the scope's parameters, in the order [`ChunkScope::sql`] names
    /// them.
    fn bind<'a>(&'a self, params: &mut Vec<&'a dyn duckdb::ToSql>) {
        for id in &self.documents {
            params.push(id);
        }
        for id in self.chunks.iter().flatten() {
            params.push(id);
        }
        for param in &self.filter.params {
            params.push(param);
        }
    }
}

/// How many hits a hybrid search returns, and how steeply reciprocal rank
/// fusion discounts rank (`[retrieval].rrf_k`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HybridLimits {
    pub top_k: u32,
    pub rrf_k: u32,
}

impl HybridLimits {
    /// Reciprocal rank fusion of two rankings of the same chunk space, the
    /// first `keep` of them.
    fn fuse(
        self,
        vector: Vec<ChunkSearchResult>,
        keyword: Vec<ChunkSearchResult>,
        keep: u32,
    ) -> Vec<ChunkSearchResult> {
        let mut fused: Vec<ChunkSearchResult> = Vec::new();
        let k = f64::from(self.rrf_k);
        for ranking in [vector, keyword] {
            for (rank, mut hit) in ranking.into_iter().enumerate() {
                let contribution =
                    1.0 / (k + f64::from(u32::try_from(rank).unwrap_or(u32::MAX)) + 1.0);
                if let Some(existing) = fused.iter_mut().find(|h| h.id == hit.id) {
                    existing.score += contribution;
                    existing.ranks.merge(hit.ranks);
                } else {
                    hit.score = contribution;
                    fused.push(hit);
                }
            }
        }
        fused.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.chunk_index.cmp(&b.chunk_index))
        });
        fused.truncate(usize::try_from(keep).unwrap_or(usize::MAX));
        fused
    }
}

/// What one search did, for a person checking why it found or missed a
/// passage.
#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct SearchExplanation {
    /// The vector leg's candidates, best first; empty without one.
    pub vector: Vec<ChunkSearchResult>,
    /// The keyword (BM25) leg's candidates, best first.
    pub keyword: Vec<ChunkSearchResult>,
    /// What the search returns: the legs fused, or the one leg that ran.
    pub fused: Vec<ChunkSearchResult>,
    /// Quoted phrases in the query, which a hit must contain exactly.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub phrases: Vec<String>,
}

impl SearchExplanation {
    /// The keyword leg alone, as a search without vectors runs.
    fn keyword_only(query: &str, keyword: Vec<ChunkSearchResult>) -> Self {
        Self {
            fused: keyword.clone(),
            keyword,
            vector: Vec::new(),
            phrases: Phrases::parse(query).0,
        }
    }

    /// The vector leg alone. A quoted phrase still filters it.
    fn vector_only(phrases: Phrases, vector: Vec<ChunkSearchResult>, top_k: u32) -> Self {
        let mut fused = vector.clone();
        phrases.retain_matching(&mut fused, top_k);
        Self {
            vector,
            keyword: Vec::new(),
            fused,
            phrases: phrases.0,
        }
    }

    /// What the quoted phrases did, when there are any.
    #[must_use]
    pub fn phrase_note(&self) -> Option<String> {
        if self.phrases.is_empty() {
            return None;
        }
        let quoted: Vec<String> = self.phrases.iter().map(|p| format!("\"{p}\"")).collect();
        Some(format!(
            "{} must appear exactly: the candidates were over-fetched and only those containing {} kept",
            quoted.join(", "),
            if self.phrases.len() == 1 {
                "it"
            } else {
                "them all"
            }
        ))
    }
}

/// Lets another thread stop the statements one piece of work runs, and
/// only while that work runs: [`WorkspaceDb::cancellable`] registers the
/// connection's interrupt handle for the length of the work and removes it
/// before the connection is released, so a late cancel never interrupts
/// whatever the connection runs next. A cancel before the work starts makes
/// it fail at once. How the work queue stops a running SQL job.
#[derive(Clone, Default)]
pub struct QueryCanceller(Arc<std::sync::Mutex<CancelSlot>>);

#[derive(Default)]
struct CancelSlot {
    running: Option<Arc<duckdb::InterruptHandle>>,
    cancelled: bool,
}

impl QueryCanceller {
    fn slot(&self) -> std::sync::MutexGuard<'_, CancelSlot> {
        self.0
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    /// Interrupt the statement running under this canceller, if any, and
    /// refuse any work that has not started yet.
    pub fn cancel(&self) {
        let mut slot = self.slot();
        slot.cancelled = true;
        if let Some(handle) = &slot.running {
            handle.interrupt();
        }
    }
}

/// Disarms the watchdog when dropped: the closed channel wakes it.
struct TimeoutGuard {
    _disarm: std::sync::mpsc::Sender<()>,
    /// Set by the watchdog before it interrupts, so an error after it is
    /// known to be the timeout rather than the statement's own.
    fired: Arc<AtomicBool>,
}

impl TimeoutGuard {
    fn fired(&self) -> bool {
        self.fired.load(Ordering::SeqCst)
    }
}

/// Guards one `BEGIN TRANSACTION READ ONLY` on a reader connection.
/// Dropping without committing rolls back, so an early `?` return inside
/// [`WorkspaceDb::read_only`] never leaves the transaction open.
struct ReadOnlyGuard<'a> {
    conn: &'a duckdb::Connection,
    committed: bool,
}

impl<'a> ReadOnlyGuard<'a> {
    fn begin(conn: &'a duckdb::Connection) -> Result<Self> {
        conn.execute_batch("BEGIN TRANSACTION READ ONLY")?;
        Ok(Self {
            conn,
            committed: false,
        })
    }

    fn commit(&mut self) -> Result<()> {
        self.conn.execute_batch("COMMIT")?;
        self.committed = true;
        Ok(())
    }
}

impl Drop for ReadOnlyGuard<'_> {
    fn drop(&mut self) {
        if !self.committed
            && let Err(e) = self.conn.execute_batch("ROLLBACK")
        {
            tracing::warn!(error = %e, "failed to roll back a read-only transaction");
        }
    }
}

/// True when `sql` is one statement that starts with a read-only keyword
/// `DuckDB` cannot serialize. A semicolon anywhere but the very end disqualifies
/// it, so `DESCRIBE t; DROP TABLE t` is not read-only. `EXPLAIN` alone only
/// prints a plan, but `EXPLAIN ANALYZE` executes it — `EXPLAIN ANALYZE
/// INSERT INTO t VALUES (1)` really inserts — so that combination is never
/// read-only regardless of what it explains.
fn is_single_read_only_statement(sql: &str) -> bool {
    let trimmed = sql.trim().trim_end_matches(';').trim_end();
    if trimmed.contains(';') {
        return false;
    }
    let mut words = trimmed.split_whitespace();
    let Some(first) = words.next() else {
        return false;
    };
    if first.eq_ignore_ascii_case("EXPLAIN")
        && words
            .next()
            .is_some_and(|w| w.eq_ignore_ascii_case("ANALYZE"))
    {
        return false;
    }
    READ_ONLY_KEYWORDS
        .iter()
        .any(|k| k.eq_ignore_ascii_case(first))
}

/// Walk a serialized statement tree collecting `table_name` values from
/// base-table references.
/// Null out the `value` of every `CONSTANT` node and drop every
/// `query_location`, which shifts with a literal's length.
fn blank_constants(node: &mut serde_json::Value) {
    match node {
        serde_json::Value::Object(map) => {
            map.remove("query_location");
            if map.get("class").and_then(serde_json::Value::as_str) == Some("CONSTANT") {
                map.insert(String::from("value"), serde_json::Value::Null);
            }
            for child in map.values_mut() {
                blank_constants(child);
            }
        }
        serde_json::Value::Array(items) => {
            for child in items {
                blank_constants(child);
            }
        }
        _ => {}
    }
}

fn collect_table_names(node: &serde_json::Value, out: &mut Vec<String>) {
    match node {
        serde_json::Value::Object(map) => {
            let is_base_table = map
                .get("type")
                .and_then(serde_json::Value::as_str)
                .is_some_and(|t| t == "BASE_TABLE");
            if is_base_table
                && let Some(name) = map.get("table_name").and_then(serde_json::Value::as_str)
            {
                out.push(name.to_owned());
            }
            for child in map.values() {
                collect_table_names(child, out);
            }
        }
        serde_json::Value::Array(items) => {
            for item in items {
                collect_table_names(item, out);
            }
        }
        serde_json::Value::Null
        | serde_json::Value::Bool(_)
        | serde_json::Value::Number(_)
        | serde_json::Value::String(_) => {}
    }
}

/// Conservative token scan used for statements the parser will not serialize.
fn is_internal_name(name: &str) -> bool {
    name.to_ascii_lowercase().starts_with(INTERNAL_PREFIX)
}

/// A `DuckDB` error as quack shows it. Workspace connections report errors
/// as JSON (`errors_as_json`, set as they open), which keeps the names
/// `DuckDB` suggests ("Did you mean ...?") apart from the message, so the
/// message is rebuilt here without quack's internal tables: otherwise every
/// misspelled table name would offer `_quack_meta` to the person and the
/// model. An error that is not that JSON is shown as `DuckDB` wrote it.
pub struct DuckDbMessage<'a>(pub &'a duckdb::Error);

/// The fields of a JSON error report that quack shows.
#[derive(serde::Deserialize)]
struct ErrorReport {
    exception_type: String,
    exception_message: String,
    /// The names `DuckDB` suggests, comma-separated.
    #[serde(default)]
    candidates: Option<String>,
}

impl fmt::Display for DuckDbMessage<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let text = self.0.to_string();
        // A parser error comes as `Parser Error: {...}`, any other as the
        // bare report.
        let report = serde_json::from_str::<ErrorReport>(&text).ok().or_else(|| {
            text.split_once(": ")
                .and_then(|(_, rest)| serde_json::from_str(rest).ok())
        });
        match report {
            Some(report) => report.fmt(f),
            None => f.write_str(&text),
        }
    }
}

impl fmt::Display for ErrorReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let message = match &self.candidates {
            // `DuckDB` appends its suggestions below the message's first line.
            Some(_) => self
                .exception_message
                .split_once('\n')
                .map_or(self.exception_message.as_str(), |(first, _)| first),
            None => self.exception_message.as_str(),
        };
        write!(f, "{} Error: {message}", self.exception_type)?;
        let shown: Vec<String> = self
            .candidates
            .iter()
            .flat_map(|names| names.split(','))
            .map(str::trim)
            .filter(|name| {
                !name.is_empty() && !name.rsplit('.').next().is_some_and(is_internal_name)
            })
            .map(|name| format!("\"{name}\""))
            .collect();
        if shown.is_empty() {
            return Ok(());
        }
        if self.exception_type == "Binder" {
            write!(f, "\nCandidate bindings: {}", shown.join(", "))
        } else {
            write!(f, "\nDid you mean {}?", shown.join(" or "))
        }
    }
}

fn mentions_internal_table_token(sql: &str) -> bool {
    sql.split(|c: char| !(c.is_alphanumeric() || c == '_'))
        .any(is_internal_name)
}

/// A `DuckDB` list literal of text values, bound as `?::VARCHAR[]`.
fn sql_text_list(items: &[String]) -> String {
    let quoted: Vec<String> = items
        .iter()
        .map(|item| format!("'{}'", item.replace('\'', "''")))
        .collect();
    format!("[{}]", quoted.join(","))
}

/// Quote a SQL identifier for `DuckDB`: wrap in double quotes and double
/// any embedded double quote. This is the only way identifiers enter SQL.
#[must_use]
pub fn quote_ident(name: &str) -> String {
    format!("\"{}\"", name.replace('"', "\"\""))
}

fn extract_value(row: &duckdb::Row<'_>, idx: usize) -> serde_json::Value {
    match row.get_ref(idx) {
        Ok(value) => json_of(duckdb::types::Value::from(value)),
        Err(_) => serde_json::Value::Null,
    }
}

/// A decimal's digit text without the trailing zeros of its declared scale:
/// `12.50` is `12.5`, `100.00` is `100`, and `-0.00` is `0`.
fn decimal_digits(text: &str) -> &str {
    let trimmed = if text.contains('.') {
        text.trim_end_matches('0').trim_end_matches('.')
    } else {
        text
    };
    if trimmed == "-0" { "0" } else { trimmed }
}

/// A `DuckDB` value as JSON: numbers stay numbers (integers beyond i64, and
/// decimals whose normalized digits would not survive an `f64`, keep their
/// digits as strings),
/// dates and times are ISO 8601 text, blobs are base64, and lists, structs,
/// and maps nest.
fn json_of(value: duckdb::types::Value) -> serde_json::Value {
    use duckdb::types::Value;
    use serde_json::Value as Json;
    match value {
        Value::Null => Json::Null,
        Value::Boolean(b) => Json::Bool(b),
        Value::TinyInt(n) => Json::from(n),
        Value::SmallInt(n) => Json::from(n),
        Value::Int(n) => Json::from(n),
        Value::BigInt(n) => Json::from(n),
        Value::UTinyInt(n) => Json::from(n),
        Value::USmallInt(n) => Json::from(n),
        Value::UInt(n) => Json::from(n),
        Value::UBigInt(n) => Json::from(n),
        Value::HugeInt(n) => match i64::try_from(n) {
            Ok(n) => Json::from(n),
            Err(_) => Json::String(n.to_string()),
        },
        Value::UHugeInt(n) => match u64::try_from(n) {
            Ok(n) => Json::from(n),
            Err(_) => Json::String(n.to_string()),
        },
        Value::Float(f) => {
            serde_json::Number::from_f64(f64::from(f)).map_or(Json::Null, Json::Number)
        }
        Value::Double(f) => serde_json::Number::from_f64(f).map_or(Json::Null, Json::Number),
        Value::Decimal(d) => {
            // `serde_json` is built without `arbitrary_precision`, so a
            // fractional decimal parses to an `f64`-backed `Number`. Compare
            // against the normalized value (no trailing zeros of the declared
            // scale, so `12.50` is `12.5` and `100.00` is `100`): a value whose
            // digits survive the `f64` stays a number, so one money column is
            // all numbers; one that would lose digits keeps its exact text.
            let text = d.to_string();
            let normalized = decimal_digits(&text);
            match normalized.parse::<serde_json::Number>() {
                Ok(n) if n.to_string() == normalized => Json::Number(n),
                _ => Json::String(text),
            }
        }
        Value::Timestamp(unit, n) => Json::String(timestamp_text(unit, n)),
        Value::Date32(days) => Json::String(date_text(days)),
        Value::Time64(unit, n) => Json::String(time_text(unit, n)),
        Value::Interval {
            months,
            days,
            nanos,
        } => Json::String(format!("{months} months {days} days {nanos} ns")),
        Value::Text(s) | Value::Enum(s) => Json::String(s),
        Value::Blob(bytes) | Value::Geometry(bytes) => {
            use base64::Engine as _;
            Json::String(base64::engine::general_purpose::STANDARD.encode(bytes))
        }
        Value::List(items) | Value::Array(items) => {
            Json::Array(items.into_iter().map(json_of).collect())
        }
        Value::Struct(fields) => {
            let mut object = serde_json::Map::new();
            for (key, value) in fields.iter() {
                object.insert(key.clone(), json_of(value.clone()));
            }
            Json::Object(object)
        }
        Value::Map(entries) => {
            let mut object = serde_json::Map::new();
            for (key, value) in entries.iter() {
                let key = match json_of(key.clone()) {
                    Json::String(s) => s,
                    other => other.to_string(),
                };
                object.insert(key, json_of(value.clone()));
            }
            Json::Object(object)
        }
        Value::Union(inner) => json_of(*inner),
        // The enum is non-exhaustive; a type this build does not know
        // renders through Debug rather than silently as null.
        other => Json::String(format!("{other:?}")),
    }
}

fn unit_to_nanos(unit: duckdb::types::TimeUnit, n: i64) -> Option<i128> {
    let n = i128::from(n);
    match unit {
        duckdb::types::TimeUnit::Second => n.checked_mul(1_000_000_000),
        duckdb::types::TimeUnit::Millisecond => n.checked_mul(1_000_000),
        duckdb::types::TimeUnit::Microsecond => n.checked_mul(1_000),
        duckdb::types::TimeUnit::Nanosecond => Some(n),
    }
}

/// `YYYY-MM-DD HH:MM:SS[.ffffff]`, as `DuckDB` prints a naive timestamp.
fn timestamp_text(unit: duckdb::types::TimeUnit, n: i64) -> String {
    let Some(nanos) = unit_to_nanos(unit, n) else {
        return n.to_string();
    };
    let Ok(ts) = jiff::Timestamp::from_nanosecond(nanos) else {
        return n.to_string();
    };
    let civil = ts.to_zoned(jiff::tz::TimeZone::UTC).datetime();
    if civil.subsec_nanosecond() == 0 {
        civil.strftime("%Y-%m-%d %H:%M:%S").to_string()
    } else {
        civil.strftime("%Y-%m-%d %H:%M:%S%.6f").to_string()
    }
}

fn date_text(days: i32) -> String {
    jiff::civil::date(1970, 1, 1)
        .checked_add(jiff::Span::new().days(days))
        .map_or_else(|_| days.to_string(), |d| d.to_string())
}

fn time_text(unit: duckdb::types::TimeUnit, n: i64) -> String {
    let Some(nanos) = unit_to_nanos(unit, n) else {
        return n.to_string();
    };
    let Ok(nanos) = i64::try_from(nanos) else {
        return n.to_string();
    };
    jiff::civil::time(0, 0, 0, 0)
        .checked_add(jiff::Span::new().nanoseconds(nanos))
        .map_or_else(
            |_| n.to_string(),
            |t| {
                if t.subsec_nanosecond() == 0 {
                    t.strftime("%H:%M:%S").to_string()
                } else {
                    t.strftime("%H:%M:%S%.6f").to_string()
                }
            },
        )
}

/// One result cell, as each consumer shows it: a string bare, anything
/// else as JSON, and NULL as the word (tables, Markdown, chart labels) or
/// as nothing (CSV, the web console's grid).
#[derive(Debug, Clone, Copy)]
pub struct Cell<'a>(pub &'a serde_json::Value);

impl<'a> Cell<'a> {
    /// The cell at `index` of `row`; a short row reads as NULL.
    #[must_use]
    pub fn at(row: &'a [serde_json::Value], index: usize) -> Self {
        const NULL: &serde_json::Value = &serde_json::Value::Null;
        Self(row.get(index).unwrap_or(NULL))
    }

    /// NULL spelled out.
    #[must_use]
    pub fn label(self) -> String {
        match self.0 {
            serde_json::Value::Null => String::from("NULL"),
            _ => self.text(),
        }
    }

    /// NULL as nothing.
    #[must_use]
    pub fn text(self) -> String {
        match self.0 {
            serde_json::Value::Null => String::new(),
            serde_json::Value::String(s) => s.clone(),
            other => other.to_string(),
        }
    }

    /// The cell as a number: a numeric string parses, NULL is 0, anything
    /// else is none.
    #[must_use]
    pub fn number(self) -> Option<f64> {
        match self.0 {
            serde_json::Value::Number(n) => n.as_f64(),
            serde_json::Value::String(s) => s.parse::<f64>().ok(),
            serde_json::Value::Null => Some(0.0),
            _ => None,
        }
    }
}

impl QueryResults {
    /// The rows with every cell whose printed text is longer than
    /// `max_chars` cut, with an ellipsis. A nested value (a `STRUCT`, a
    /// `LIST`) is cut by the text `write_table` prints for it, so one JSON
    /// column cannot make a row any wider than a long string can.
    #[must_use]
    pub fn with_cells_cut(&self, max_chars: usize) -> Self {
        let mut out = self.clone();
        for row in &mut out.rows {
            for cell in row.iter_mut() {
                let text = Cell(cell).label();
                if text.chars().count() > max_chars {
                    let mut cut: String = text.chars().take(max_chars).collect();
                    cut.push('\u{2026}');
                    *cell = serde_json::Value::String(cut);
                }
            }
        }
        out
    }

    /// Write results as a human-readable aligned table.
    ///
    /// # Errors
    ///
    /// Returns an error if writing to `out` fails.
    pub fn write_table(&self, out: &mut impl Write) -> Result<()> {
        if self.columns.is_empty() {
            writeln!(out, "OK")?;
            return Ok(());
        }

        let display_rows: Vec<Vec<String>> = self
            .rows
            .iter()
            .map(|row| row.iter().map(|v| Cell(v).label()).collect())
            .collect();

        let mut widths: Vec<usize> = self.columns.iter().map(String::len).collect();
        for row in &display_rows {
            for (w, val) in widths.iter_mut().zip(row.iter()) {
                *w = (*w).max(val.len());
            }
        }

        for (i, (col, width)) in self.columns.iter().zip(widths.iter()).enumerate() {
            if i > 0 {
                write!(out, " | ")?;
            }
            write!(out, "{col:<width$}")?;
        }
        writeln!(out)?;

        for (i, width) in widths.iter().enumerate() {
            if i > 0 {
                write!(out, "-+-")?;
            }
            for _ in 0..*width {
                write!(out, "-")?;
            }
        }
        writeln!(out)?;

        for row in &display_rows {
            for (i, (val, width)) in row.iter().zip(widths.iter()).enumerate() {
                if i > 0 {
                    write!(out, " | ")?;
                }
                write!(out, "{val:<width$}")?;
            }
            writeln!(out)?;
        }

        let row_count = self.rows.len();
        writeln!(out, "({row_count} rows)")?;
        Ok(())
    }

    /// One JSON object per line.
    ///
    /// # Errors
    ///
    /// Returns an error if serialization or writing fails.
    pub fn write_ndjson(&self, out: &mut impl Write) -> Result<()> {
        let keys = self.json_keys();
        for row in &self.rows {
            let row = JsonRow {
                keys: &keys,
                values: row,
            };
            writeln!(out, "{}", row.render()?)?;
        }
        Ok(())
    }

    /// RFC 4180 CSV with a header row; fields containing a comma, quote, or
    /// newline are quoted and embedded quotes doubled.
    ///
    /// # Errors
    ///
    /// Returns an error if writing fails.
    pub fn write_csv(&self, out: &mut impl Write) -> Result<()> {
        let mut writer = csv::Writer::from_writer(out);
        writer.write_record(&self.columns)?;
        for row in &self.rows {
            let cells: Vec<String> = row.iter().map(|v| Cell(v).text()).collect();
            writer.write_record(&cells)?;
        }
        writer.flush()?;
        Ok(())
    }

    /// A GitHub-flavored Markdown table.
    ///
    /// # Errors
    ///
    /// Returns an error if writing fails.
    pub fn write_markdown(&self, out: &mut impl Write) -> Result<()> {
        fn cell(value: &str) -> String {
            value.replace('|', "\\|").replace('\n', " ")
        }
        if self.columns.is_empty() {
            writeln!(out, "OK")?;
            return Ok(());
        }
        let header: Vec<String> = self.columns.iter().map(|c| cell(c)).collect();
        writeln!(out, "| {} |", header.join(" | "))?;
        let rule: Vec<&str> = self.columns.iter().map(|_| "---").collect();
        writeln!(out, "| {} |", rule.join(" | "))?;
        for row in &self.rows {
            let cells: Vec<String> = row.iter().map(|v| cell(&Cell(v).label())).collect();
            writeln!(out, "| {} |", cells.join(" | "))?;
        }
        Ok(())
    }

    /// Write results as a JSON array of objects.
    ///
    /// # Errors
    ///
    /// Returns an error if writing to `out` or JSON serialization fails.
    pub fn write_json(&self, out: &mut impl Write) -> Result<()> {
        let keys = self.json_keys();
        let json_rows: Vec<serde_json::Map<String, serde_json::Value>> = self
            .rows
            .iter()
            .map(|row| {
                keys.iter()
                    .zip(row.iter())
                    .map(|(col, val)| (col.clone(), val.clone()))
                    .collect()
            })
            .collect();

        serde_json::to_writer_pretty(&mut *out, &json_rows)?;
        writeln!(out)?;
        Ok(())
    }

    /// The column names as object keys: a repeated name gets a numeric
    /// suffix (`a`, `a_1`, `a_2`) the way `DuckDB` itself writes JSON,
    /// so a self-join keeps every column (issue #65).
    #[must_use]
    pub fn json_keys(&self) -> Vec<String> {
        let mut taken: std::collections::HashSet<String> = std::collections::HashSet::new();
        let mut keys = Vec::with_capacity(self.columns.len());
        for column in &self.columns {
            let mut key = column.clone();
            let mut n: u32 = 0;
            while !taken.insert(key.clone()) {
                n = n.saturating_add(1);
                key = format!("{column}_{n}");
            }
            keys.push(key);
        }
        keys
    }
}

#[cfg(test)]
mod tests;
