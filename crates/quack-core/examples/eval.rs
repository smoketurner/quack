//! Evaluation harness (design doc section 16, issue #74): retrieval quality,
//! ontology induction, graph extraction, and citation validity, all over a
//! small in-tree storms-like fixture (`crates/quack-core/eval/`) with no
//! model required. Run with `make eval` or `cargo run -p quack-core --example
//! eval`; set `QUACK_EVAL_OUT=path.json` to also write the numbers as JSON
//! for a CI job or a before/after diff.
//!
//! The embedding backend is [`HashEmbedder`], a deterministic stand-in for a
//! real provider: it hashes each token into one of a fixed number of
//! buckets (a bag-of-words projection), not a semantic embedding, so vector
//! search is reproducible without Ollama. Its numbers are only meaningful
//! relative to another run of this harness, never to a production model.

#![expect(
    clippy::cast_precision_loss,
    reason = "ratios and ranks over small fixture-sized counts; exactness is not needed"
)]

use std::collections::{BTreeMap, BTreeSet};
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use quack_core::analysis::citations::{self, CitationRegistry};
use quack_core::config::{
    AnalysisConfig, AuthMode, Config, ContextConfig, EmbeddingConfig, GeneralConfig, GraphConfig,
    ImportConfig, IngestionConfig, JobsConfig, OntologyConfig, ProviderConfig, ProviderType,
    RetrievalConfig, ServerConfig,
};
use quack_core::embedding::{Dimension, Embedder, Input, Profile, Prompts};
use quack_core::error::{Error, Result};
use quack_core::graph::extract::{ChunkText, ExtractFuture, Extraction, GraphExtractor};
use quack_core::graph::{self, store};
use quack_core::ingestion::{self, NewFile};
use quack_core::ontology::Ontology;
use quack_core::ontology::induction::{self, Proposal, TableEvidenceOptions};
use quack_core::storage::workspace::{ChunkScope, ChunkSearchResult, WorkspaceDb};
use quack_core::storage::writer::Writer;
use rig::embeddings::{Embedding, EmbeddingError, EmbeddingModel};
use serde::{Deserialize, Serialize};

const HASH_DIM: usize = 64;
const HASH_DIM_U32: u32 = 64;

#[tokio::main]
async fn main() -> ExitCode {
    match run().await {
        Ok(report) => match report.print() {
            Ok(()) => match maybe_write_json(&report) {
                Ok(()) => ExitCode::SUCCESS,
                Err(e) => {
                    print_error(&format!("failed to write QUACK_EVAL_OUT: {e}"));
                    ExitCode::FAILURE
                }
            },
            Err(e) => {
                print_error(&format!("failed to print the report: {e}"));
                ExitCode::FAILURE
            }
        },
        Err(e) => {
            print_error(&format!("eval failed: {e}"));
            ExitCode::FAILURE
        }
    }
}

fn maybe_write_json(report: &Report) -> Result<()> {
    if let Ok(path) = std::env::var("QUACK_EVAL_OUT") {
        report.write_json(Path::new(&path))?;
    }
    Ok(())
}

fn print_error(message: &str) {
    use std::io::Write as _;
    if writeln!(std::io::stderr(), "{message}").is_err() {
        // Nothing more can be done if stderr itself is gone.
    }
}

async fn run() -> Result<Report> {
    let fixture_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("eval");
    let data_dir = tempfile::tempdir()?;
    let config = eval_config(data_dir.path());
    let workspace_id = "eval";
    let db = WorkspaceDb::open(&config, workspace_id)?;
    // No prefixes: the hashing embedder counts tokens, and a prefix would
    // add the same tokens to every input.
    let embedder = Embedder::new(
        HashEmbedder,
        Profile::new(
            "hash-embedder",
            Dimension::new(HASH_DIM_U32),
            Prompts::default(),
        ),
    );

    ingest_documents(
        &config,
        &db,
        workspace_id,
        &fixture_dir.join("documents"),
        &embedder,
    )
    .await?;
    ingest_tables(&config, &db, workspace_id, &fixture_dir.join("tables")).await?;

    let chunks = load_chunk_index(&db)?;

    let retrieval = evaluate_retrieval(
        &db,
        &fixture_dir.join("gold_questions.json"),
        &chunks,
        &embedder,
        config.retrieval.top_k,
        config.retrieval.rrf_k,
    )
    .await?;
    let ontology_induction = evaluate_induction(&db, &fixture_dir.join("expected_ontology.json"))?;
    let graph_extraction =
        evaluate_graph(&db, &chunks, &fixture_dir.join("graph_fixture.json")).await?;
    let citation_validity = evaluate_citations(&fixture_dir.join("citation_fixture.json"))?;

    Ok(Report {
        retrieval,
        ontology_induction,
        graph_extraction,
        citation_validity,
    })
}

fn eval_config(data_dir: &Path) -> Config {
    let mut providers = BTreeMap::new();
    providers.insert(
        String::from("eval"),
        ProviderConfig {
            provider_type: ProviderType::Ollama,
            auth: AuthMode::None,
            base_url: None,
            api_key_env: None,
            embedding_dimension: Some(HASH_DIM_U32),
            max_concurrent_requests: None,
            oauth: None,
        },
    );
    Config {
        general: GeneralConfig {
            data_dir: data_dir.to_path_buf(),
            default_workspace: String::from("eval"),
            chat_model: None,
            embedding_model: Some(String::from("eval/hash-embedder")),
        },
        providers,
        ingestion: IngestionConfig::default(),
        embedding: EmbeddingConfig::default(),
        retrieval: RetrievalConfig::default(),
        context: ContextConfig::default(),
        analysis: AnalysisConfig::default(),
        server: ServerConfig::default(),
        ontology: OntologyConfig::default(),
        graph: GraphConfig::default(),
        import: ImportConfig::default(),
        jobs: JobsConfig::default(),
    }
}

// ---------------------------------------------------------------------------
// Deterministic hashing embedder (stand-in for a real provider)
// ---------------------------------------------------------------------------

/// A bag-of-words hashing "embedding": lowercased alphanumeric tokens are
/// hashed into one of [`HASH_DIM`] buckets and the resulting counts are L2
/// normalized. Not a semantic embedding; see the module doc comment.
struct HashEmbedder;

impl EmbeddingModel for HashEmbedder {
    const MAX_DOCUMENTS: usize = 1024;
    type Client = ();

    fn make(_client: &Self::Client, _model: impl Into<String>, _dims: Option<usize>) -> Self {
        Self
    }

    fn ndims(&self) -> usize {
        HASH_DIM
    }

    fn embed_texts(
        &self,
        texts: impl IntoIterator<Item = String> + Send,
    ) -> impl std::future::Future<Output = std::result::Result<Vec<Embedding>, EmbeddingError>> + Send
    {
        let embeddings = texts
            .into_iter()
            .map(|text| {
                let vec = hash_embed(&text);
                Embedding {
                    document: text,
                    vec,
                }
            })
            .collect();
        std::future::ready(Ok(embeddings))
    }
}

fn tokenize(text: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut current = String::new();
    for c in text.chars() {
        if c.is_ascii_alphanumeric() {
            current.push(c.to_ascii_lowercase());
        } else if !current.is_empty() {
            tokens.push(std::mem::take(&mut current));
        }
    }
    if !current.is_empty() {
        tokens.push(current);
    }
    tokens
}

fn hash_bucket(token: &str) -> usize {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    token.hash(&mut hasher);
    usize::try_from(hasher.finish())
        .unwrap_or(usize::MAX)
        .checked_rem(HASH_DIM)
        .unwrap_or(0)
}

fn hash_embed(text: &str) -> Vec<f64> {
    let mut buckets = vec![0.0_f64; HASH_DIM];
    for token in tokenize(text) {
        if let Some(slot) = buckets.get_mut(hash_bucket(&token)) {
            *slot += 1.0;
        }
    }
    let norm = buckets.iter().map(|v| v * v).sum::<f64>().sqrt();
    if norm > 0.0 {
        for slot in &mut buckets {
            *slot /= norm;
        }
    }
    buckets
}

// ---------------------------------------------------------------------------
// Ingestion
// ---------------------------------------------------------------------------

/// A writer over a second connection to `db`'s database, for the
/// pipeline steps that take one, while the eval reads through `db`.
fn writer_of(db: &WorkspaceDb) -> Result<Writer> {
    Writer::spawn(db.try_clone_reader()?)
}

async fn ingest_documents(
    config: &Config,
    db: &WorkspaceDb,
    workspace_id: &str,
    dir: &Path,
    embedder: &Embedder<HashEmbedder>,
) -> Result<()> {
    let writer = writer_of(db)?;
    for path in list_files(dir, "md")? {
        let data = std::fs::read(&path)?;
        let filename = file_name(&path)?;
        ingestion::ingest_file(
            config,
            &writer,
            workspace_id,
            &NewFile::new(filename, &data),
            Some(embedder),
        )
        .await?;
    }
    Ok(())
}

async fn ingest_tables(
    config: &Config,
    db: &WorkspaceDb,
    workspace_id: &str,
    dir: &Path,
) -> Result<()> {
    let writer = writer_of(db)?;
    for path in list_files(dir, "csv")? {
        let data = std::fs::read(&path)?;
        let filename = file_name(&path)?;
        ingestion::ingest_file::<HashEmbedder>(
            config,
            &writer,
            workspace_id,
            &NewFile::new(filename, &data),
            None,
        )
        .await?;
    }
    Ok(())
}

fn list_files(dir: &Path, extension: &str) -> Result<Vec<PathBuf>> {
    let mut entries: Vec<PathBuf> = std::fs::read_dir(dir)?
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|ext| ext == extension))
        .collect();
    entries.sort();
    Ok(entries)
}

fn file_name(path: &Path) -> Result<&str> {
    path.file_name()
        .and_then(|n| n.to_str())
        .ok_or_else(|| Error::Ingestion(format!("'{}' is not a plain file name", path.display())))
}

// ---------------------------------------------------------------------------
// Chunk index: resolves gold-set (filename, substring) pairs to chunk ids,
// since chunk ids are minted as UUID v7 at ingest and cannot be fixed in a
// fixture ahead of time.
// ---------------------------------------------------------------------------

struct ChunkRow {
    id: String,
    document_id: String,
    filename: String,
    content: String,
}

fn load_chunk_index(db: &WorkspaceDb) -> Result<Vec<ChunkRow>> {
    let mut stmt = db.connection().prepare(
        "SELECT c.id, c.document_id, d.filename, c.content FROM _quack_chunks c \
         JOIN _quack_documents d ON d.id = c.document_id \
         ORDER BY d.filename, c.chunk_index",
    )?;
    let mut rows = stmt.query([])?;
    let mut out = Vec::new();
    while let Some(row) = rows.next()? {
        out.push(ChunkRow {
            id: row.get(0)?,
            document_id: row.get(1)?,
            filename: row.get(2)?,
            content: row.get(3)?,
        });
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Retrieval quality
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
struct GoldExpected {
    filename: String,
    contains: String,
}

#[derive(Debug, Deserialize)]
struct GoldQuestion {
    question: String,
    #[serde(default)]
    kind: String,
    expected: Vec<GoldExpected>,
    /// This question's correct answer is that nothing should come back: the
    /// individual words appear in the corpus (so a bag-of-words ranking
    /// returns something), but the exact phrase does not, so a phrase-aware
    /// search should filter every candidate out.
    #[serde(default)]
    expect_empty: bool,
}

/// What a gold question expects back: specific chunk ids, or nothing at
/// all (an `expect_empty` question).
#[derive(Debug, Clone)]
enum Expectation {
    Chunks(BTreeSet<String>),
    Empty,
}

#[derive(Debug, Clone, Serialize)]
struct RetrievalMetrics {
    recall_at_1: f64,
    recall_at_5: f64,
    recall_at_8: f64,
    mrr: f64,
    n: usize,
}

/// One kind's (`identifier`, `phrase`, `semantic`) metrics for every
/// backend, so a tokenization or phrase-filtering change shows up on its
/// own row instead of averaged away across question kinds.
#[derive(Debug, Serialize)]
struct BackendMetrics {
    keyword: RetrievalMetrics,
    similar: RetrievalMetrics,
    hybrid: RetrievalMetrics,
}

#[derive(Debug, Serialize)]
struct RetrievalReport {
    keyword: RetrievalMetrics,
    similar: RetrievalMetrics,
    hybrid: RetrievalMetrics,
    by_kind: BTreeMap<String, BackendMetrics>,
    n: usize,
}

fn ids_of(results: Vec<ChunkSearchResult>) -> Vec<String> {
    results.into_iter().map(|r| r.id).collect()
}

fn recall_at(expectation: &Expectation, ranked: &[String], k: usize) -> f64 {
    match expectation {
        Expectation::Chunks(expected) => {
            if expected.is_empty() {
                return 0.0;
            }
            let hit = ranked
                .iter()
                .take(k)
                .filter(|id| expected.contains(*id))
                .count();
            hit as f64 / expected.len() as f64
        }
        // Nothing should have been returned at all; a k-cutoff is
        // meaningless when the correct answer is an empty list.
        Expectation::Empty => f64::from(u8::from(ranked.is_empty())),
    }
}

fn reciprocal_rank(expectation: &Expectation, ranked: &[String]) -> f64 {
    match expectation {
        Expectation::Chunks(expected) => {
            for (i, id) in ranked.iter().enumerate() {
                if expected.contains(id) {
                    return 1.0 / (i as f64 + 1.0);
                }
            }
            0.0
        }
        Expectation::Empty => f64::from(u8::from(ranked.is_empty())),
    }
}

/// A gold question's expectation paired with what a search backend
/// actually ranked, for one question.
type QuestionPairs = Vec<(Expectation, Vec<String>)>;

fn aggregate(per_question: &QuestionPairs) -> RetrievalMetrics {
    let n = per_question.len();
    if n == 0 {
        return RetrievalMetrics {
            recall_at_1: 0.0,
            recall_at_5: 0.0,
            recall_at_8: 0.0,
            mrr: 0.0,
            n: 0,
        };
    }
    let mut sum1 = 0.0;
    let mut sum5 = 0.0;
    let mut sum8 = 0.0;
    let mut summrr = 0.0;
    for (expectation, ranked) in per_question {
        sum1 += recall_at(expectation, ranked, 1);
        sum5 += recall_at(expectation, ranked, 5);
        sum8 += recall_at(expectation, ranked, 8);
        summrr += reciprocal_rank(expectation, ranked);
    }
    let n_f = n as f64;
    RetrievalMetrics {
        recall_at_1: sum1 / n_f,
        recall_at_5: sum5 / n_f,
        recall_at_8: sum8 / n_f,
        mrr: summrr / n_f,
        n,
    }
}

async fn evaluate_retrieval(
    db: &WorkspaceDb,
    gold_path: &Path,
    chunks: &[ChunkRow],
    embedder: &Embedder<HashEmbedder>,
    top_k: u32,
    rrf_k: u32,
) -> Result<RetrievalReport> {
    let gold: Vec<GoldQuestion> = serde_json::from_str(&std::fs::read_to_string(gold_path)?)?;
    let mut keyword_pairs = Vec::with_capacity(gold.len());
    let mut similar_pairs = Vec::with_capacity(gold.len());
    let mut hybrid_pairs = Vec::with_capacity(gold.len());
    let mut by_kind: BTreeMap<String, (QuestionPairs, QuestionPairs, QuestionPairs)> =
        BTreeMap::new();

    for q in &gold {
        let expectation = if q.expect_empty {
            Expectation::Empty
        } else {
            let expected: BTreeSet<String> = chunks
                .iter()
                .filter(|c| {
                    q.expected.iter().any(|e| {
                        e.filename == c.filename && c.content.contains(e.contains.as_str())
                    })
                })
                .map(|c| c.id.clone())
                .collect();
            if expected.is_empty() {
                return Err(Error::Ingestion(format!(
                    "gold question '{}' matched no ingested chunk; fix the fixture",
                    q.question
                )));
            }
            Expectation::Chunks(expected)
        };

        let keyword_ids =
            ids_of(db.search_keyword_chunks(&q.question, top_k, &ChunkScope::all())?);
        let query_vec = embedder
            .embed_one(&Input::Query(q.question.clone()))
            .await?;
        let similar_ids =
            ids_of(db.search_similar_chunks(&query_vec, top_k, &ChunkScope::all())?);
        let hybrid_ids = ids_of(db.search_hybrid_chunks(
            &q.question,
            &query_vec,
            top_k,
            rrf_k,
            &ChunkScope::all(),
        )?);

        let kind_entry = by_kind.entry(q.kind.clone()).or_default();
        kind_entry
            .0
            .push((expectation.clone(), keyword_ids.clone()));
        kind_entry
            .1
            .push((expectation.clone(), similar_ids.clone()));
        kind_entry.2.push((expectation.clone(), hybrid_ids.clone()));

        keyword_pairs.push((expectation.clone(), keyword_ids));
        similar_pairs.push((expectation.clone(), similar_ids));
        hybrid_pairs.push((expectation, hybrid_ids));
    }

    Ok(RetrievalReport {
        keyword: aggregate(&keyword_pairs),
        similar: aggregate(&similar_pairs),
        hybrid: aggregate(&hybrid_pairs),
        by_kind: by_kind
            .into_iter()
            .map(|(kind, (kw, sim, hy))| {
                (
                    kind,
                    BackendMetrics {
                        keyword: aggregate(&kw),
                        similar: aggregate(&sim),
                        hybrid: aggregate(&hy),
                    },
                )
            })
            .collect(),
        n: gold.len(),
    })
}

// ---------------------------------------------------------------------------
// Ontology induction
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
struct ExpectedClass {
    id: String,
    properties: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct ExpectedRelation {
    id: String,
    domain: String,
    range: String,
}

#[derive(Debug, Deserialize)]
struct ExpectedOntology {
    classes: Vec<ExpectedClass>,
    relations: Vec<ExpectedRelation>,
}

#[derive(Debug, Clone, Serialize)]
struct PrecisionRecall {
    precision: f64,
    recall: f64,
    n_expected: usize,
    n_actual: usize,
}

fn precision_recall<T: Ord>(actual: &BTreeSet<T>, expected: &BTreeSet<T>) -> PrecisionRecall {
    let tp = actual.intersection(expected).count();
    let precision = if actual.is_empty() {
        0.0
    } else {
        tp as f64 / actual.len() as f64
    };
    let recall = if expected.is_empty() {
        0.0
    } else {
        tp as f64 / expected.len() as f64
    };
    PrecisionRecall {
        precision,
        recall,
        n_expected: expected.len(),
        n_actual: actual.len(),
    }
}

#[derive(Debug, Serialize)]
struct InductionReport {
    classes: PrecisionRecall,
    properties: PrecisionRecall,
    relations: PrecisionRecall,
}

fn evaluate_induction(db: &WorkspaceDb, expected_path: &Path) -> Result<InductionReport> {
    let expected: ExpectedOntology =
        serde_json::from_str(&std::fs::read_to_string(expected_path)?)?;
    let candidates = induction::propose_from_tables(db, None, &TableEvidenceOptions::default())?;

    let mut actual_classes = BTreeSet::new();
    let mut actual_properties = BTreeSet::new();
    let mut actual_relations = BTreeSet::new();
    for candidate in &candidates {
        match &candidate.proposal {
            Proposal::Class(class) => {
                actual_classes.insert(class.id.clone());
                for property in &class.properties {
                    actual_properties.insert((class.id.clone(), property.clone()));
                }
            }
            Proposal::Relation(relation) => {
                actual_relations.insert((
                    relation.id.clone(),
                    relation.domain.clone(),
                    relation.range.clone(),
                ));
            }
            Proposal::Property { .. } | Proposal::Mapping(_) => {}
        }
    }

    let expected_classes: BTreeSet<String> =
        expected.classes.iter().map(|c| c.id.clone()).collect();
    let expected_properties: BTreeSet<(String, String)> = expected
        .classes
        .iter()
        .flat_map(|c| c.properties.iter().map(move |p| (c.id.clone(), p.clone())))
        .collect();
    let expected_relations: BTreeSet<(String, String, String)> = expected
        .relations
        .iter()
        .map(|r| (r.id.clone(), r.domain.clone(), r.range.clone()))
        .collect();

    Ok(InductionReport {
        classes: precision_recall(&actual_classes, &expected_classes),
        properties: precision_recall(&actual_properties, &expected_properties),
        relations: precision_recall(&actual_relations, &expected_relations),
    })
}

// ---------------------------------------------------------------------------
// Graph extraction: a canned `GraphExtractor` so the harness measures
// validation, resolution, and storage, not the model.
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
struct FixtureChunk {
    anchor: String,
    canned_answer: Extraction,
}

#[derive(Debug, Deserialize)]
struct FixtureNode {
    label: String,
    class: String,
}

#[derive(Debug, Deserialize)]
struct FixtureEdge {
    source: String,
    target: String,
    relation: String,
}

#[derive(Debug, Deserialize)]
struct GraphFixture {
    ontology: serde_json::Value,
    chunks: Vec<FixtureChunk>,
    expected_nodes: Vec<FixtureNode>,
    expected_edges: Vec<FixtureEdge>,
}

/// Answers a chunk by scanning its text for a fixed anchor substring (a
/// report identifier), so the harness needs no model.
struct FixtureExtractor {
    answers: BTreeMap<String, Extraction>,
}

impl GraphExtractor for FixtureExtractor {
    fn extract<'a>(&'a self, text: &'a str) -> ExtractFuture<'a> {
        Box::pin(async move {
            for (anchor, extraction) in &self.answers {
                if text.contains(anchor.as_str()) {
                    return Ok(extraction.clone());
                }
            }
            Ok(Extraction::default())
        })
    }
}

#[derive(Debug, Serialize)]
struct GraphReport {
    nodes: PrecisionRecall,
    edges: PrecisionRecall,
    chunks_processed: u32,
    /// Distinct out-of-ontology classes and relations the canned answers
    /// tried to write, correctly dropped rather than stored.
    drift_categories: usize,
}

async fn evaluate_graph(
    db: &WorkspaceDb,
    chunks: &[ChunkRow],
    fixture_path: &Path,
) -> Result<GraphReport> {
    let fixture: GraphFixture = serde_json::from_str(&std::fs::read_to_string(fixture_path)?)?;
    let ontology = Ontology::from_json(&fixture.ontology.to_string())?;

    let mut answers = BTreeMap::new();
    let mut chunk_texts = Vec::with_capacity(fixture.chunks.len());
    for fixture_chunk in &fixture.chunks {
        let row = chunks
            .iter()
            .find(|c| c.content.contains(fixture_chunk.anchor.as_str()))
            .ok_or_else(|| {
                Error::Ontology(format!(
                    "no ingested chunk contains anchor '{}'; fix the fixture",
                    fixture_chunk.anchor
                ))
            })?;
        answers.insert(
            fixture_chunk.anchor.clone(),
            fixture_chunk.canned_answer.clone(),
        );
        chunk_texts.push(ChunkText {
            chunk_id: row.id.clone(),
            document_id: row.document_id.clone(),
            text: row.content.clone(),
        });
    }
    let extractor = FixtureExtractor { answers };

    let writer = writer_of(db)?;
    let summary = graph::extract::run(
        &writer,
        chunk_texts,
        &extractor,
        &ontology,
        false,
        4,
        &|_| {},
    )
    .await?;

    let node_ids = store::all_node_ids(db)?;
    let nodes = store::nodes(db, &node_ids)?;
    let edges = store::edges_among(db, &node_ids)?;
    let label_of: BTreeMap<String, String> = nodes
        .iter()
        .map(|n| (n.id.clone(), n.label.clone()))
        .collect();

    let actual_nodes: BTreeSet<(String, String)> = nodes
        .iter()
        .map(|n| (n.label.clone(), n.class_id.clone()))
        .collect();
    let expected_nodes: BTreeSet<(String, String)> = fixture
        .expected_nodes
        .iter()
        .map(|n| (n.label.clone(), n.class.clone()))
        .collect();

    let actual_edges: BTreeSet<(String, String, String)> = edges
        .iter()
        .filter_map(|e| {
            let source = label_of.get(&e.source_node_id)?;
            let target = label_of.get(&e.target_node_id)?;
            Some((source.clone(), target.clone(), e.relation_id.clone()))
        })
        .collect();
    let expected_edges: BTreeSet<(String, String, String)> = fixture
        .expected_edges
        .iter()
        .map(|e| (e.source.clone(), e.target.clone(), e.relation.clone()))
        .collect();

    Ok(GraphReport {
        nodes: precision_recall(&actual_nodes, &expected_nodes),
        edges: precision_recall(&actual_edges, &expected_edges),
        chunks_processed: summary.chunks,
        drift_categories: summary.drift.total(),
    })
}

// ---------------------------------------------------------------------------
// Citation validity
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
struct CitationCase {
    registered_chunks: u32,
    answer: String,
    expected_surviving_markers: usize,
}

#[derive(Debug, Serialize)]
struct CitationReport {
    /// Recorded answers whose surviving `[n]` marker count matched what
    /// the fixture recorded, out of `total`.
    matches: usize,
    total: usize,
    original_markers: usize,
    survived_markers: usize,
}

fn count_numeric_markers(answer: &str) -> usize {
    let mut count = 0usize;
    let mut pieces = answer.split('[');
    pieces.next();
    for piece in pieces {
        if let Some((inside, _)) = piece.split_once(']') {
            let digits = inside.trim();
            if !digits.is_empty() && digits.chars().all(|c| c.is_ascii_digit()) {
                count = count.saturating_add(1);
            }
        }
    }
    count
}

fn evaluate_citations(path: &Path) -> Result<CitationReport> {
    let cases: Vec<CitationCase> = serde_json::from_str(&std::fs::read_to_string(path)?)?;
    let mut matches = 0usize;
    let mut original_markers = 0usize;
    let mut survived_markers = 0usize;
    for case in &cases {
        let registry = CitationRegistry::default();
        let registered: Vec<ChunkSearchResult> = (0..case.registered_chunks)
            .map(|i| ChunkSearchResult {
                id: format!("chunk-{i}"),
                content: String::new(),
                document_id: String::from("doc"),
                chunk_index: i,
                filename: String::from("doc.md"),
                heading: None,
                page: None,
                score: 1.0,
            })
            .collect();
        let _first_marker = registry.register(&registered);
        let (_, cited) = citations::validate(&case.answer, &registry.all());
        original_markers = original_markers.saturating_add(count_numeric_markers(&case.answer));
        survived_markers = survived_markers.saturating_add(cited.len());
        if cited.len() == case.expected_surviving_markers {
            matches = matches.saturating_add(1);
        }
    }
    Ok(CitationReport {
        matches,
        total: cases.len(),
        original_markers,
        survived_markers,
    })
}

// ---------------------------------------------------------------------------
// Report
// ---------------------------------------------------------------------------

#[derive(Debug, Serialize)]
struct Report {
    retrieval: RetrievalReport,
    ontology_induction: InductionReport,
    graph_extraction: GraphReport,
    citation_validity: CitationReport,
}

impl Report {
    fn print(&self) -> Result<()> {
        use std::io::Write as _;
        let mut out = std::io::stdout().lock();

        writeln!(out, "Retrieval quality ({} questions)", self.retrieval.n)?;
        writeln!(
            out,
            "{:<10} {:>10} {:>10} {:>10} {:>8}",
            "method", "recall@1", "recall@5", "recall@8", "mrr"
        )?;
        for (name, m) in [
            ("keyword", &self.retrieval.keyword),
            ("similar", &self.retrieval.similar),
            ("hybrid", &self.retrieval.hybrid),
        ] {
            writeln!(
                out,
                "{name:<10} {:>10.3} {:>10.3} {:>10.3} {:>8.3}",
                m.recall_at_1, m.recall_at_5, m.recall_at_8, m.mrr
            )?;
        }
        writeln!(
            out,
            "\nRetrieval by question kind (identifier/phrase questions carry the decoys that make #77's fix visible):"
        )?;
        writeln!(
            out,
            "{:<10} {:<10} {:>10} {:>10} {:>10} {:>8} {:>6}",
            "kind", "method", "recall@1", "recall@5", "recall@8", "mrr", "n"
        )?;
        for (kind, backends) in &self.retrieval.by_kind {
            for (name, m) in [
                ("keyword", &backends.keyword),
                ("similar", &backends.similar),
                ("hybrid", &backends.hybrid),
            ] {
                writeln!(
                    out,
                    "{kind:<10} {name:<10} {:>10.3} {:>10.3} {:>10.3} {:>8.3} {:>6}",
                    m.recall_at_1, m.recall_at_5, m.recall_at_8, m.mrr, m.n
                )?;
            }
        }

        writeln!(
            out,
            "\nOntology induction (propose_from_tables vs. hand-written expectation)"
        )?;
        writeln!(
            out,
            "{:<12} {:>10} {:>10} {:>10} {:>10}",
            "item", "precision", "recall", "expected", "actual"
        )?;
        for (name, pr) in [
            ("classes", &self.ontology_induction.classes),
            ("properties", &self.ontology_induction.properties),
            ("relations", &self.ontology_induction.relations),
        ] {
            writeln!(
                out,
                "{name:<12} {:>10.3} {:>10.3} {:>10} {:>10}",
                pr.precision, pr.recall, pr.n_expected, pr.n_actual
            )?;
        }

        writeln!(
            out,
            "\nGraph extraction ({} chunks processed, {} drift categories dropped)",
            self.graph_extraction.chunks_processed, self.graph_extraction.drift_categories
        )?;
        writeln!(
            out,
            "{:<8} {:>10} {:>10} {:>10} {:>10}",
            "item", "precision", "recall", "expected", "actual"
        )?;
        for (name, pr) in [
            ("nodes", &self.graph_extraction.nodes),
            ("edges", &self.graph_extraction.edges),
        ] {
            writeln!(
                out,
                "{name:<8} {:>10.3} {:>10.3} {:>10} {:>10}",
                pr.precision, pr.recall, pr.n_expected, pr.n_actual
            )?;
        }

        writeln!(
            out,
            "\nCitation validity: {}/{} recorded answers validated as expected ({}/{} `[n]` markers survived)",
            self.citation_validity.matches,
            self.citation_validity.total,
            self.citation_validity.survived_markers,
            self.citation_validity.original_markers
        )?;
        Ok(())
    }

    fn write_json(&self, path: &Path) -> Result<()> {
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write(path, json)?;
        Ok(())
    }
}
