//! `quack ontology`: show, install the default, export and import JSON,
//! list versions, diff, restore, and propose and review candidates. The
//! ontology lives in the workspace file; a file on disk is only ever a copy.

use std::io::Write;

use anyhow::{Context, Result};
use clap::Subcommand;
use quack_core::config::Config;
use quack_core::extraction::ExtractionRun;

use crate::confirm::Confirm;
use crate::graph_cli::RenderOnWriter;
use crate::stdio::StdioPath;
use crate::text_or_json::TextOrJson;
use quack_core::llm::{self, Embeddings};
use quack_core::ontology::OntologyVersion;
use quack_core::ontology::candidates::Queue;
use quack_core::ontology::documents::{self, DocumentProposal};
use quack_core::ontology::induction::{Candidate, Decision, ItemKind, propose_from_tables};
use quack_core::ontology::store::Revision;
use quack_core::ontology::{Ontology, candidates, store};
use quack_core::progress::RunControl;
use quack_core::storage::workspace::WorkspaceDb;
use quack_core::storage::writer::Writer;

#[derive(Subcommand)]
pub(crate) enum OntologyAction {
    /// Print the current ontology: classes, relations, properties, mappings
    Show {
        /// `json` prints the JSON interchange form
        #[arg(long, value_enum, default_value_t = TextOrJson::Text)]
        format: TextOrJson,
    },
    /// Install the built-in general ontology as version 1
    Init,
    /// Write the current ontology as JSON to a file (- for stdout)
    Export { file: StdioPath },
    /// Validate a JSON ontology and store it as a new version (- for stdin)
    Import { file: StdioPath },
    /// List versions, newest first
    Versions {
        #[arg(long, default_value_t = 20)]
        limit: u32,
    },
    /// What changed between two versions (defaults: previous and current)
    Diff {
        from: Option<OntologyVersion>,
        to: Option<OntologyVersion>,
    },
    /// Store an earlier version as the newest one
    Restore { version: OntologyVersion },
    /// Propose the classes, properties, keys, relations, and mappings the
    /// current ontology lacks (a full draft when there is none) from the
    /// tables (no model calls) and, with --documents, from a sample of the
    /// documents (one model call per chunk) into the review queue
    Propose(ProposeArgs),
    /// List pending candidates with their evidence
    Review {
        /// Show the candidates kept aside for low document support instead
        #[arg(long)]
        low_support: bool,
    },
    /// Accept candidates by id (prefixes accepted), as proposed or changed
    Accept(AcceptArgs),
    /// Reject candidates by id (prefixes accepted)
    Reject { ids: Vec<String> },
}

#[derive(clap::Args)]
pub(crate) struct ProposeArgs {
    /// Accept every proposal at once and write the version
    #[arg(long)]
    auto_accept: bool,
    /// Also run open extraction over a sample of the documents
    #[arg(long)]
    documents: bool,
    /// Chunks to sample with --documents (default from config)
    #[arg(long)]
    sample: Option<u32>,
    /// Do not ask before spending the model calls
    #[arg(long, short = 'y')]
    pub(crate) yes: bool,
    /// Seed from a JSON ontology first (stored as a new version), then
    /// propose only what it lacks
    #[arg(long, value_name = "FILE")]
    from: Option<String>,
}

#[derive(clap::Args)]
pub(crate) struct AcceptArgs {
    ids: Vec<String>,
    /// Accept under this id (one candidate only)
    #[arg(long, conflicts_with_all = ["merge_into", "reparent"])]
    rename: Option<String>,
    /// Treat the candidate as this existing class, relation, or property
    #[arg(long, conflicts_with = "reparent")]
    merge_into: Option<String>,
    /// Accept a class under this parent
    #[arg(long)]
    reparent: Option<String>,
}

impl AcceptArgs {
    /// Each id with the one decision the flags name.
    fn decisions(self) -> Result<Vec<(String, Decision)>> {
        if self.ids.is_empty() {
            anyhow::bail!("give at least one candidate id");
        }
        if self.ids.len() > 1
            && (self.rename.is_some() || self.merge_into.is_some() || self.reparent.is_some())
        {
            anyhow::bail!(
                "--rename, --merge-into, and --reparent apply to one candidate at a time"
            );
        }
        let decision = if let Some(new_id) = self.rename {
            Decision::Rename(new_id)
        } else if let Some(target) = self.merge_into {
            Decision::MergeInto(target)
        } else if let Some(parent) = self.reparent {
            Decision::Reparent(parent)
        } else {
            Decision::Accept
        };
        Ok(self
            .ids
            .into_iter()
            .map(|id| (id, decision.clone()))
            .collect())
    }
}

/// Run one ontology action, writing what the user should see to `out`
/// (stdout for the CLI, the transcript for the terminal session).
///
/// Each database step goes to the workspace writer on its own, never
/// spanning a model call; `control` hears about every chunk of a document pass and can stop it.
pub(crate) async fn run(
    config: &Config,
    db: &Writer,
    action: OntologyAction,
    confirm: Confirm,
    out: &mut impl Write,
    control: RunControl<'_>,
) -> Result<()> {
    match action {
        OntologyAction::Propose(args) => {
            run_propose(config, db, args, confirm, out, control).await?;
        }
        OntologyAction::Show { format } => {
            db.render(out, move |db, out| show(db, format, out)).await?;
        }
        OntologyAction::Init => db.render(out, init).await?,
        OntologyAction::Export { file } => {
            db.render(out, move |db, out| export(db, &file, out))
                .await?;
        }
        OntologyAction::Import { file } => {
            db.render(out, move |db, out| import(db, &file, out))
                .await?;
        }
        OntologyAction::Versions { limit } => {
            db.render(out, move |db, out| versions(db, limit, out))
                .await?;
        }
        OntologyAction::Diff { from, to } => {
            db.render(out, move |db, out| diff(db, from, to, out))
                .await?;
        }
        OntologyAction::Restore { version } => {
            db.render(out, move |db, out| {
                let stored = store::restore(db, version, None)?;
                writeln!(
                    out,
                    "restored version {version} as version {}",
                    stored.saved_version()?
                )?;
                Ok(())
            })
            .await?;
        }
        OntologyAction::Review { low_support } => {
            db.render(out, move |db, out| review(db, low_support, out))
                .await?;
        }
        OntologyAction::Accept(args) => {
            let decisions = args.decisions()?;
            db.render(out, move |db, out| {
                let stored = candidates::accept(db, &decisions, None)?;
                writeln!(
                    out,
                    "accepted {} candidate(s); ontology is now version {}",
                    decisions.len(),
                    stored.saved_version()?
                )?;
                Ok(())
            })
            .await?;
        }
        OntologyAction::Reject { ids } => {
            db.render(out, move |db, out| {
                let count = candidates::reject(db, &ids, None)?;
                writeln!(out, "rejected {count} candidate(s)")?;
                Ok(())
            })
            .await?;
        }
    }
    out.flush()?;
    Ok(())
}

fn show(db: &WorkspaceDb, format: TextOrJson, out: &mut impl Write) -> Result<()> {
    match (store::current(db)?, format) {
        (None, TextOrJson::Text | TextOrJson::Json) => writeln!(
            out,
            "No ontology yet. Run `quack ontology init` for the built-in one or `quack ontology import FILE`."
        )?,
        (Some(ontology), TextOrJson::Json) => writeln!(out, "{}", ontology.to_json()?)?,
        (Some(ontology), TextOrJson::Text) => write!(out, "{}", ontology.render_summary())?,
    }
    Ok(())
}

fn init(db: &WorkspaceDb, out: &mut Vec<u8>) -> Result<()> {
    if store::latest_version(db)?.is_some() {
        anyhow::bail!("an ontology already exists; import a file or restore a version instead");
    }
    let stored = store::save(
        db,
        &Ontology::builtin_default(),
        Revision::reviewed(None, Some("built-in default")),
    )?;
    writeln!(
        out,
        "installed the built-in ontology as version {}",
        stored.saved_version()?
    )?;
    Ok(())
}

fn export(db: &WorkspaceDb, file: &StdioPath, out: &mut impl Write) -> Result<()> {
    let ontology = store::current(db)?.context("no ontology to export")?;
    let json = ontology.to_json()?;
    match file {
        StdioPath::Stdio => writeln!(out, "{json}")?,
        StdioPath::Path(path) => {
            std::fs::write(path, format!("{json}\n"))
                .with_context(|| format!("failed to write {file}"))?;
            writeln!(out, "wrote version {} to {file}", ontology.saved_version()?)?;
        }
    }
    Ok(())
}

fn import(db: &WorkspaceDb, file: &StdioPath, out: &mut impl Write) -> Result<()> {
    let text = file.read_to_string()?;
    let ontology = Ontology::from_json(&text)?;
    let stored = store::save(
        db,
        &ontology,
        Revision::reviewed(None, Some(&format!("imported from {file}"))),
    )?;
    writeln!(out, "ontology is now version {}", stored.saved_version()?)?;
    Ok(())
}

fn versions(db: &WorkspaceDb, limit: u32, out: &mut impl Write) -> Result<()> {
    let versions = store::versions(db, limit)?;
    if versions.is_empty() {
        writeln!(out, "No versions yet.")?;
    }
    for v in versions {
        writeln!(
            out,
            "v{:<4} {}  {:<12} {}",
            v.version,
            v.created_at,
            v.author.as_deref().unwrap_or("-"),
            v.note.as_deref().unwrap_or("")
        )?;
    }
    Ok(())
}

/// What changed between two versions: the previous and the current by
/// default.
fn diff(
    db: &WorkspaceDb,
    from: Option<OntologyVersion>,
    to: Option<OntologyVersion>,
    out: &mut impl Write,
) -> Result<()> {
    let to = match to {
        Some(to) => to,
        None => store::latest_version(db)?.context("no ontology yet")?,
    };
    let from = from
        .or_else(|| to.previous())
        .with_context(|| format!("version {to} is the first; name one to compare it with"))?;
    let older =
        store::version(db, from)?.with_context(|| format!("version {from} does not exist"))?;
    let newer = store::version(db, to)?.with_context(|| format!("version {to} does not exist"))?;
    write!(out, "{}", newer.diff(&older))?;
    Ok(())
}

/// `propose`, with optional seeding from a file.
async fn run_propose(
    config: &Config,
    db: &Writer,
    args: ProposeArgs,
    confirm: Confirm,
    out: &mut impl Write,
    control: RunControl<'_>,
) -> Result<()> {
    let ProposeArgs {
        auto_accept,
        documents,
        sample,
        yes,
        from,
    } = args;
    db.render(out, move |db, out| seed(db, from.as_deref(), out))
        .await?;
    let pass = documents.then_some(DocumentPass {
        sample,
        confirm: confirm.or_yes(yes),
    });
    propose(config, db, auto_accept, pass, out, control).await
}

/// The candidates in a queue, with their evidence.
fn review(db: &WorkspaceDb, low_support: bool, out: &mut impl Write) -> Result<()> {
    let queue = if low_support {
        Queue::LowSupport
    } else {
        Queue::Pending
    };
    let pending = candidates::queue(db, queue)?;
    if pending.is_empty() && low_support {
        writeln!(out, "No low-support candidates.")?;
    } else if pending.is_empty() {
        let aside = candidates::queue(db, Queue::LowSupport)?.len();
        writeln!(
            out,
            "No pending candidates. Run `quack ontology propose`.{}",
            if aside > 0 {
                format!(" {aside} low-support candidates: `quack ontology review --low-support`.")
            } else {
                String::new()
            }
        )?;
    }
    for c in &pending {
        writeln!(
            out,
            "{}  {:<9} {:<28} {:.2}  {}",
            c.id,
            c.kind,
            c.proposal.id(),
            c.confidence,
            c.evidence_line()
        )?;
    }
    Ok(())
}

/// The document pass, when asked for.
struct DocumentPass {
    sample: Option<u32>,
    confirm: Confirm,
}

/// `--from FILE`: store the file as a new version so the proposal only
/// adds what it lacks.
fn seed(db: &WorkspaceDb, from: Option<&str>, out: &mut impl Write) -> Result<()> {
    let Some(file) = from else {
        return Ok(());
    };
    let text = std::fs::read_to_string(file).with_context(|| format!("failed to read {file}"))?;
    let ontology = Ontology::from_json(&text)?;
    let stored = store::save(
        db,
        &ontology,
        Revision::reviewed(None, Some(&format!("seeded from {file}"))),
    )?;
    writeln!(
        out,
        "seeded version {} from {file}",
        stored.saved_version()?
    )?;
    Ok(())
}

/// `quack ontology propose`: table evidence, plus document evidence when
/// asked, into the queue, or straight into a version with `--auto-accept`.
async fn propose(
    config: &Config,
    db: &Writer,
    auto_accept: bool,
    documents: Option<DocumentPass>,
    out: &mut impl Write,
    control: RunControl<'_>,
) -> Result<()> {
    let current = db.run(store::current).await?;
    let (known, evidence) = (current.clone(), config.ontology.table_evidence());
    let mut proposals = db
        .run(move |db| propose_from_tables(db, known.as_ref(), &evidence))
        .await?;
    if let Some(pass) = &documents {
        let mut options = config.ontology.document_evidence();
        if let Some(n) = pass.sample {
            options.sample_chunks = n;
        }
        let estimating = options;
        let cost = db
            .run(move |db| documents::estimate(db, &estimating))
            .await?;
        writeln!(
            out,
            "Document evidence: {} chunks sampled across {} documents, {} model calls to {}.",
            cost.chunks,
            cost.documents,
            cost.model_calls,
            config.chat_model_label()
        )?;
        // Shown before the model calls, ahead of the progress lines on
        // stderr.
        out.flush()?;
        if cost.chunks == 0 {
            writeln!(out, "No ready documents to sample.")?;
        } else if !pass.confirm.ask(out, "Proceed?", Some("--yes"))? {
            writeln!(out, "Skipped the document pass.")?;
        } else {
            let from_documents =
                run_documents(config, db, current.as_ref(), &options, out, control).await?;
            proposals.extend(from_documents);
        }
    }
    if proposals.is_empty() {
        writeln!(out, "Nothing to propose: the tables are already covered.")?;
    } else if auto_accept {
        let run = proposals.clone();
        let stored = db
            .run(move |db| {
                candidates::store_run(db, &run)?;
                candidates::accept_all(db, None)
            })
            .await?;
        writeln!(
            out,
            "accepted {} proposals; ontology is now version {}",
            proposals.len(),
            stored.saved_version()?
        )?;
    } else {
        let run = proposals.clone();
        db.run(move |db| candidates::store_run(db, &run)).await?;
        let by_kind = |kind: ItemKind| {
            proposals
                .iter()
                .filter(|c| c.proposal.kind() == kind)
                .count()
        };
        let low = proposals.iter().filter(|c| c.low_support).count();
        writeln!(
            out,
            "{} candidates queued: {} classes, {} properties, {} relations, {} mappings ({low} with low support, kept aside). Run `quack ontology review`.",
            proposals.len(),
            by_kind(ItemKind::Class),
            by_kind(ItemKind::Property),
            by_kind(ItemKind::Relation),
            by_kind(ItemKind::Mapping)
        )?;
    }
    Ok(())
}

/// Sample, extract with the chat model, and propose.
async fn run_documents(
    config: &Config,
    db: &Writer,
    current: Option<&Ontology>,
    options: &documents::DocumentEvidenceOptions,
    out: &mut impl Write,
    control: RunControl<'_>,
) -> Result<Vec<Candidate>> {
    let count = options.sample_chunks;
    let sample = db
        .run(move |db| documents::sample_chunks(db, count))
        .await?;
    let extractor = llm::chat_extractor(config).await?;
    let embeddings = Embeddings::from_config(config).await?;
    let DocumentProposal {
        candidates,
        summary,
    } = documents::run(
        sample,
        current,
        options,
        embeddings.as_ref(),
        ExtractionRun {
            extractor: extractor.as_ref(),
            concurrency: config.analysis.extraction_concurrency,
            control,
        },
    )
    .await?;
    writeln!(
        out,
        "extracted from {} chunks ({} failed): {} candidates, {} with low support",
        summary.sampled_chunks, summary.failed_chunks, summary.candidates, summary.low_support
    )?;
    Ok(candidates)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `accept` takes at most one change: two would leave one silently
    /// unapplied.
    #[test]
    fn accept_takes_one_change_at_a_time() {
        #[derive(clap::Parser)]
        #[command(no_binary_name = true)]
        struct Line {
            #[command(subcommand)]
            action: OntologyAction,
        }
        let parses = |args: &[&str]| <Line as clap::Parser>::try_parse_from(args).is_ok();
        assert!(parses(&["accept", "c1"]));
        assert!(parses(&["accept", "c1", "--reparent", "p"]));
        assert!(parses(&["accept", "c1", "--merge-into", "m"]));
        for pair in [
            ["--merge-into", "m", "--reparent", "p"],
            ["--rename", "r", "--reparent", "p"],
            ["--rename", "r", "--merge-into", "m"],
        ] {
            let mut args = vec!["accept", "c1"];
            args.extend(pair);
            assert!(!parses(&args), "{args:?}");
        }
    }
}
