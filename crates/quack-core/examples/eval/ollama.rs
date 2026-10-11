//! The model half of the evaluation (`make eval`, `--ollama`): the fixture
//! questions put to the agent over Ollama, and the vision model reading an
//! image, a scanned PDF page, and a picture inside a DOCX. Nothing here
//! fails the run on a wrong answer: every case is scored and reported, so a
//! change to a prompt or a model shows up as a number to compare between
//! runs. What a model writes varies between runs, so one run's score is a
//! sample, not a verdict.
//!
//! The models come from the environment, each with a default small enough
//! for a CPU runner:
//!
//! - `QUACK_EVAL_OLLAMA_URL` (`http://127.0.0.1:11434`)
//! - `QUACK_EVAL_CHAT_MODEL` (`qwen2.5:3b`)
//! - `QUACK_EVAL_EMBEDDING_MODEL` (`embeddinggemma`) and
//!   `QUACK_EVAL_EMBEDDING_DIMENSION` (768)
//! - `QUACK_EVAL_VISION_MODEL` (`qwen2.5vl:3b`; `none` skips the vision
//!   section)
//! - `QUACK_EVAL_TURN_SECONDS` (300): how long one question may take. A
//!   small model can loop on a tool or generate until Ollama's context
//!   shifts, so a turn past it is stopped and scored as timed out, which
//!   keeps a nightly run bounded.

use std::collections::BTreeMap;
use std::io::Write;
use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};

use office_oxide::create::create_from_ir_to_writer;
use office_oxide::format::DocumentFormat;
use office_oxide::ir::{DocumentIR, Element, Image, ImageFormat as IrImageFormat};
use pdf_oxide::geometry::Rect;
use pdf_oxide::writer::DocumentBuilder;
use quack_core::analysis::events::{self, ToolName};
use quack_core::analysis::policy::WritePolicy;
use quack_core::analysis::tools::{ReaderDb, SharedDb};
use quack_core::classify::LabelJobs;
use quack_core::config::Config;
use quack_core::embedding::Input;
use quack_core::error::{Error, Result};
use quack_core::ingestion::{self, NewFile};
use quack_core::jobs::JobQueue;
use quack_core::llm::{CancellationToken, Embeddings, TurnRequest};
use quack_core::storage::sessions::{self, ChatMode};
use quack_core::storage::workspace::WorkspaceDb;
use quack_core::storage::writer::Writer;
use serde::{Deserialize, Serialize};

/// What the vision fixture says; each word is looked for, ignoring case, in
/// what the model wrote. No file name, title, or alt text the cases give
/// holds one of them, so only the model's reading can find them.
const NOTICE_WORDS: &[&str] = &["notice", "220", "october", "delgado"];

/// The longest stretch of an answer kept in the report.
const ANSWER_EXCERPT: usize = 400;

/// The models the run uses, from the environment.
#[derive(Debug, Clone, Serialize)]
pub(crate) struct Models {
    url: String,
    chat: String,
    embedding: String,
    dimension: u32,
    /// `None` skips the vision section.
    vision: Option<String>,
    turn_seconds: u64,
}

impl Models {
    pub(crate) fn from_env() -> Self {
        let var = |name: &str, default: &str| {
            std::env::var(name)
                .ok()
                .map(|v| v.trim().to_owned())
                .filter(|v| !v.is_empty())
                .unwrap_or_else(|| default.to_owned())
        };
        let vision = var("QUACK_EVAL_VISION_MODEL", "qwen2.5vl:3b");
        Self {
            url: var("QUACK_EVAL_OLLAMA_URL", "http://127.0.0.1:11434"),
            chat: var("QUACK_EVAL_CHAT_MODEL", "qwen2.5:3b"),
            embedding: var("QUACK_EVAL_EMBEDDING_MODEL", "embeddinggemma"),
            dimension: var("QUACK_EVAL_EMBEDDING_DIMENSION", "768")
                .parse()
                .unwrap_or(768),
            vision: (!vision.eq_ignore_ascii_case("none")).then_some(vision),
            turn_seconds: var("QUACK_EVAL_TURN_SECONDS", "300").parse().unwrap_or(300),
        }
    }

    /// A configuration naming these models on one Ollama provider, its data
    /// under `data_dir`.
    pub(crate) fn config(&self, data_dir: &Path) -> Result<Config> {
        let vision = self.vision.as_ref().map_or_else(String::new, |model| {
            format!(
                "[ingestion]\nvision_model = {:?}\n",
                format!("ollama/{model}")
            )
        });
        let toml = format!(
            "[general]\nchat_model = {chat:?}\n\
             [providers.ollama]\ntype = \"ollama\"\nbase_url = {url:?}\n\
             [embedding]\nmodel = {embedding:?}\ndimension = {dimension}\n{vision}",
            chat = format!("ollama/{}", self.chat),
            url = self.url,
            embedding = format!("ollama/{}", self.embedding),
            dimension = self.dimension,
        );
        let mut config = Config::parse(&toml)?;
        config.general.data_dir = data_dir.to_path_buf();
        Ok(config)
    }

    /// Fail early, saying what to do, when the embedding model cannot be
    /// reached: every later section needs Ollama.
    pub(crate) async fn reachable(&self, embedder: &Embeddings) -> Result<()> {
        embedder
            .embed_one(&Input::Query(String::from("reachable")))
            .await
            .map(|_| ())
            .map_err(|e| {
                Error::Embedding(format!(
                    "{e}\n\nmake eval needs Ollama at {} with {}, {}{} pulled \
                     (see crates/quack-core/eval/README.md); make eval-offline runs without a model",
                    self.url,
                    self.embedding,
                    self.chat,
                    self.vision
                        .as_ref()
                        .map_or_else(String::new, |v| format!(", and {v}")),
                ))
            })
    }
}

/// The model sections of the report.
#[derive(Debug, Serialize)]
pub(crate) struct ModelsReport {
    models: Models,
    agent: AgentReport,
    #[serde(skip_serializing_if = "Option::is_none")]
    vision: Option<VisionReport>,
}

impl ModelsReport {
    pub(crate) fn print(&self, out: &mut impl Write) -> Result<()> {
        let agent = &self.agent;
        writeln!(
            out,
            "\nAgent answers ({} questions, chat model {})",
            agent.n, self.models.chat
        )?;
        writeln!(
            out,
            "{:<10} {:>8} {:>10} {:>10} {:>8} {:>8} {:>10}",
            "kind", "correct", "right tool", "cited", "<think>", "errors", "seconds"
        )?;
        for (kind, tally) in agent
            .by_kind
            .iter()
            .chain([(&String::from("all"), &agent.all)])
        {
            writeln!(
                out,
                "{kind:<10} {:>8} {:>10} {:>10} {:>8} {:>8} {:>10.1}",
                format!("{}/{}", tally.correct, tally.n),
                format!("{}/{}", tally.right_tool, tally.n),
                format!("{}/{}", tally.cited, tally.citing),
                tally.think_leaks,
                tally.errors,
                tally.seconds,
            )?;
        }
        for case in agent
            .cases
            .iter()
            .filter(|c| !c.correct || c.error.is_some())
        {
            writeln!(
                out,
                "  missed: {} -> {}",
                case.question,
                case.error
                    .as_deref()
                    .unwrap_or(&case.answer)
                    .replace('\n', " ")
            )?;
        }
        if let Some(vision) = &self.vision {
            writeln!(
                out,
                "\nVision ({} words to find, model {})",
                NOTICE_WORDS.len(),
                vision.model
            )?;
            writeln!(out, "{:<16} {:>8} {:>10}", "case", "found", "seconds")?;
            for case in &vision.cases {
                writeln!(
                    out,
                    "{:<16} {:>8} {:>10.1}{}",
                    case.name,
                    format!("{}/{}", case.found.len(), NOTICE_WORDS.len()),
                    case.seconds,
                    case.error
                        .as_deref()
                        .map_or_else(String::new, |e| format!("  error: {e}")),
                )?;
            }
        }
        Ok(())
    }
}

/// The workspace the agent and the vision model work in: the fixture's
/// documents and tables, and nothing the other sections add (the graph
/// section's nodes would register the graph tools, which a person's fresh
/// workspace does not have).
const WORKSPACE: &str = "agent";

/// Put the agent questions to the chat model over a workspace of the
/// fixture's documents and tables, then have the vision model read the
/// notice three ways.
pub(crate) async fn evaluate(
    models: &Models,
    config: &Config,
    embedder: &Embeddings,
    fixture_dir: &Path,
) -> Result<ModelsReport> {
    let db = WorkspaceDb::open(config, WORKSPACE)?;
    super::ingest_documents(
        config,
        &db,
        WORKSPACE,
        &fixture_dir.join("documents"),
        embedder,
    )
    .await?;
    super::ingest_tables(config, &db, WORKSPACE, &fixture_dir.join("tables")).await?;
    let db = &db;
    let shared: SharedDb = Arc::new(Writer::spawn(db.try_clone_reader()?)?);
    let agent = evaluate_agent(
        config,
        &shared,
        models,
        &fixture_dir.join("agent_questions.json"),
    )
    .await?;
    let vision = match &models.vision {
        Some(model) => {
            Some(evaluate_vision(config, db, &shared, model, &fixture_dir.join("vision")).await?)
        }
        None => None,
    };
    Ok(ModelsReport {
        models: models.clone(),
        agent,
        vision,
    })
}

// ---------------------------------------------------------------------------
// Agent answers
// ---------------------------------------------------------------------------

/// One question of `agent_questions.json`.
#[derive(Debug, Deserialize)]
struct AgentCase {
    question: String,
    /// `table` (answered from the tables) or `document` (from the text).
    kind: String,
    /// Every group must be matched by one of its strings, ignoring case.
    expect: Vec<Vec<String>>,
    /// The tools an answer should come from; any one of them will do.
    tools: Vec<ToolName>,
    /// Whether the answer should cite a passage.
    #[serde(default)]
    cite: bool,
}

#[derive(Debug, Serialize)]
struct AgentOutcome {
    question: String,
    kind: String,
    correct: bool,
    right_tool: bool,
    /// `None` when the question does not call for a citation.
    cited: Option<bool>,
    think_leak: bool,
    tools: Vec<ToolName>,
    seconds: f64,
    answer: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
}

#[derive(Debug, Default, Serialize)]
struct Tally {
    n: usize,
    correct: usize,
    right_tool: usize,
    /// Of `citing`, the questions whose answer cited a passage.
    cited: usize,
    citing: usize,
    think_leaks: usize,
    errors: usize,
    seconds: f64,
}

impl Tally {
    fn add(&mut self, outcome: &AgentOutcome) {
        let count = |flag: bool| usize::from(flag);
        self.n = self.n.saturating_add(1);
        self.correct = self.correct.saturating_add(count(outcome.correct));
        self.right_tool = self.right_tool.saturating_add(count(outcome.right_tool));
        self.citing = self.citing.saturating_add(count(outcome.cited.is_some()));
        self.cited = self
            .cited
            .saturating_add(count(outcome.cited == Some(true)));
        self.think_leaks = self.think_leaks.saturating_add(count(outcome.think_leak));
        self.errors = self.errors.saturating_add(count(outcome.error.is_some()));
        self.seconds += outcome.seconds;
    }
}

#[derive(Debug, Serialize)]
struct AgentReport {
    n: usize,
    all: Tally,
    by_kind: BTreeMap<String, Tally>,
    cases: Vec<AgentOutcome>,
}

async fn evaluate_agent(
    config: &Config,
    shared: &SharedDb,
    models: &Models,
    path: &Path,
) -> Result<AgentReport> {
    let limit = Duration::from_secs(models.turn_seconds);
    let cases: Vec<AgentCase> = serde_json::from_str(&std::fs::read_to_string(path)?)?;
    let reader = ReaderDb::open(shared, config.analysis.reader_pool_size).await;
    let labelling = LabelJobs::new(JobQueue::new(1));
    let mut outcomes = Vec::with_capacity(cases.len());
    for case in cases {
        let model = format!("ollama/{}", models.chat);
        let session = shared
            .run(move |db| sessions::create_session(db, &model, ChatMode::Chat, None))
            .await?;
        let (sink, _events) = events::channel();
        let started = Instant::now();
        let cancel = CancellationToken::new();
        let turn = TurnRequest {
            db: Arc::clone(shared),
            reader_db: reader.clone(),
            session_id: &session.id,
            policy: WritePolicy::Deny,
            message: &case.question,
            documents: &[],
            user: None,
            labelling: labelling.clone(),
            sink,
            cancel: cancel.clone(),
        }
        .run(config);
        let turn = tokio::time::timeout(limit, turn).await.unwrap_or_else(|_| {
            cancel.cancel();
            Err(Error::Ingestion(format!(
                "timed out after {}s (QUACK_EVAL_TURN_SECONDS)",
                limit.as_secs()
            )))
        });
        let seconds = started.elapsed().as_secs_f64();
        outcomes.push(match turn {
            Ok(response) => {
                let tools: Vec<ToolName> = response.steps.iter().map(|s| s.tool).collect();
                let answer = response.content.to_lowercase();
                AgentOutcome {
                    correct: case.expect.iter().all(|group| {
                        group
                            .iter()
                            .any(|want| answer.contains(&want.to_lowercase()))
                    }),
                    right_tool: tools.iter().any(|t| case.tools.contains(t)),
                    cited: case.cite.then_some(!response.citations.is_empty()),
                    think_leak: answer.contains("<think>") || answer.contains("</think>"),
                    tools,
                    seconds,
                    answer: excerpt(&response.content),
                    error: None,
                    question: case.question,
                    kind: case.kind,
                }
            }
            Err(e) => AgentOutcome {
                correct: false,
                right_tool: false,
                cited: case.cite.then_some(false),
                think_leak: false,
                tools: Vec::new(),
                seconds,
                answer: String::new(),
                error: Some(e.to_string()),
                question: case.question,
                kind: case.kind,
            },
        });
    }
    let mut all = Tally::default();
    let mut by_kind: BTreeMap<String, Tally> = BTreeMap::new();
    for outcome in &outcomes {
        all.add(outcome);
        by_kind
            .entry(outcome.kind.clone())
            .or_default()
            .add(outcome);
    }
    Ok(AgentReport {
        n: outcomes.len(),
        all,
        by_kind,
        cases: outcomes,
    })
}

/// The start of `text`, cut at a character boundary.
fn excerpt(text: &str) -> String {
    text.chars().take(ANSWER_EXCERPT).collect()
}

// ---------------------------------------------------------------------------
// Vision
// ---------------------------------------------------------------------------

#[derive(Debug, Serialize)]
struct VisionCase {
    name: &'static str,
    /// The words of the notice the model's text holds.
    found: Vec<&'static str>,
    seconds: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
}

#[derive(Debug, Serialize)]
struct VisionReport {
    model: String,
    cases: Vec<VisionCase>,
}

/// The notice ingested as an image, as the only picture on a PDF page, and
/// as a figure inside a DOCX; each scored by the notice's words found in
/// the document's chunks.
async fn evaluate_vision(
    config: &Config,
    db: &WorkspaceDb,
    shared: &SharedDb,
    model: &str,
    dir: &Path,
) -> Result<VisionReport> {
    let notice = std::fs::read(dir.join("shelter-notice.jpg"))?;
    let files: [(&'static str, &'static str, Result<Vec<u8>>); 3] = [
        ("image", "posted-sign.jpg", Ok(notice.clone())),
        ("scanned pdf", "records-scan.pdf", scanned_pdf(&notice)),
        (
            "docx figure",
            "shelter-memo.docx",
            docx_with_figure(&notice),
        ),
    ];
    let mut cases = Vec::with_capacity(files.len());
    for (name, filename, bytes) in files {
        let started = Instant::now();
        let ingested = match bytes {
            Ok(bytes) => ingestion::ingest_file(
                config,
                shared,
                WORKSPACE,
                &NewFile::new(filename, &bytes),
                None::<&Embeddings>,
            )
            .await
            .map(|_| ()),
            Err(e) => Err(e),
        };
        let seconds = started.elapsed().as_secs_f64();
        let (found, error) = match ingested.and_then(|()| document_text(db, filename)) {
            Ok(text) => {
                let text = text.to_lowercase();
                (
                    NOTICE_WORDS
                        .iter()
                        .copied()
                        .filter(|word| text.contains(word))
                        .collect(),
                    None,
                )
            }
            Err(e) => (Vec::new(), Some(e.to_string())),
        };
        cases.push(VisionCase {
            name,
            found,
            seconds,
            error,
        });
    }
    Ok(VisionReport {
        model: model.to_owned(),
        cases,
    })
}

/// Every chunk of the document ingested as `filename`, joined.
fn document_text(db: &WorkspaceDb, filename: &str) -> Result<String> {
    let mut stmt = db.connection().prepare(
        "SELECT c.content FROM _quack_chunks c JOIN _quack_documents d ON d.id = c.document_id \
         WHERE d.filename = ? ORDER BY c.chunk_index",
    )?;
    let chunks = stmt
        .query_map(duckdb::params![filename], |row| row.get::<_, String>(0))?
        .collect::<duckdb::Result<Vec<_>>>()?;
    Ok(chunks.join("\n"))
}

/// A typed cover page, then a page that is only the notice's picture.
fn scanned_pdf(notice: &[u8]) -> Result<Vec<u8>> {
    let failed = |e: &dyn std::fmt::Display| Error::Ingestion(format!("building the PDF: {e}"));
    let mut pdf = DocumentBuilder::new().title("Shelter records");
    pdf.letter_page()
        .at(72.0, 720.0)
        .text("Shelter records, scanned for the archive")
        .done();
    pdf.letter_page()
        .image_from_bytes(notice, Rect::new(36.0, 300.0, 540.0, 312.0))
        .map_err(|e| failed(&e))?
        .done();
    pdf.build().map_err(|e| failed(&e))
}

/// A short memo with the notice as a two-by-three-inch figure.
fn docx_with_figure(notice: &[u8]) -> Result<Vec<u8>> {
    let mut ir = DocumentIR::from_markdown(
        "# Shelter memo\n\nThe posted sign is below.\n",
        DocumentFormat::Docx,
    );
    let Some(body) = ir.sections.first_mut() else {
        return Err(Error::Ingestion(String::from("the memo has no section")));
    };
    body.elements.push(Element::Image(Image {
        alt_text: Some(String::from("Posted sign")),
        data: Some(notice.to_vec()),
        format: Some(IrImageFormat::Jpeg),
        display_width_emu: Some(2_743_200),
        display_height_emu: Some(1_584_960),
        ..Image::default()
    }));
    let mut out = std::io::Cursor::new(Vec::new());
    create_from_ir_to_writer(&ir, DocumentFormat::Docx, &mut out)
        .map_err(|e| Error::Ingestion(format!("building the DOCX: {e}")))?;
    Ok(out.into_inner())
}
