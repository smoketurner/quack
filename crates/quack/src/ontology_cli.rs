//! `quack ontology`: show, install the default, export and import JSON,
//! list versions, diff, restore, and propose and review candidates. The
//! ontology lives in the workspace file; a file on disk is only ever a copy.

use std::io::Write;

use anyhow::{Context, Result};
use clap::Subcommand;
use quack_core::config::Config;

use crate::confirm::Confirm;
use crate::graph_cli::rendered;
use crate::stdio::StdioPath;
use quack_core::llm;
use quack_core::ontology::OntologyVersion;
use quack_core::ontology::candidates::{CandidateStatus, Queue};
use quack_core::ontology::induction::{Candidate, Decision, ItemKind, propose_from_tables};
use quack_core::ontology::store::Revision;
use quack_core::ontology::{Ontology, ROOT_CLASS, candidates, documents, store};
use quack_core::progress::{ChunkDone, Progress};
use quack_core::storage::workspace::WorkspaceDb;
use quack_core::storage::writer::Writer;

#[derive(Subcommand)]
pub(crate) enum OntologyAction {
    /// Print the current ontology: classes, relations, properties, mappings
    Show {
        /// Print the JSON interchange form instead
        #[arg(long)]
        json: bool,
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
    Propose {
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
        yes: bool,
        /// Seed from a JSON ontology first (stored as a new version), then
        /// propose only what it lacks
        #[arg(long, value_name = "FILE")]
        from: Option<String>,
    },
    /// List pending candidates with their evidence
    Review {
        /// Show the candidates kept aside for low document support instead
        #[arg(long)]
        low_support: bool,
    },
    /// Accept candidates by id (prefixes accepted), as proposed or changed
    Accept {
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
    },
    /// Reject candidates by id (prefixes accepted)
    Reject { ids: Vec<String> },
}

/// Run one ontology action, writing what the user should see to `out`
/// (stdout for the CLI, the transcript for the terminal session).
///
/// Each database step goes to the workspace writer on its own, never
/// spanning a model call; `progress` hears about every chunk of a document pass.
pub(crate) async fn run(
    config: &Config,
    db: &Writer,
    action: OntologyAction,
    out: &mut impl Write,
    progress: Progress<'_>,
) -> Result<()> {
    match action {
        propose_action @ OntologyAction::Propose { .. } => {
            run_propose(config, db, propose_action, out, progress).await?;
        }
        review @ (OntologyAction::Review { .. }
        | OntologyAction::Accept { .. }
        | OntologyAction::Reject { .. }) => {
            let text = db
                .run(move |db| Ok(rendered(|buf| run_review(db, review, buf))))
                .await?;
            out.write_all(&text.map_err(anyhow::Error::msg)?)?;
        }
        manage => {
            let text = db
                .run(move |db| Ok(rendered(|buf| run_manage(db, manage, buf))))
                .await?;
            out.write_all(&text.map_err(anyhow::Error::msg)?)?;
        }
    }
    out.flush()?;
    Ok(())
}

/// Show, init, export, import, versions, diff, restore.
fn run_manage(db: &WorkspaceDb, action: OntologyAction, out: &mut impl Write) -> Result<()> {
    match action {
        OntologyAction::Show { json } => match store::current(db)? {
            None => writeln!(
                out,
                "No ontology yet. Run `quack ontology init` for the built-in one or `quack ontology import FILE`."
            )?,
            Some(ontology) if json => writeln!(out, "{}", ontology.to_json()?)?,
            Some(ontology) => write!(out, "{}", summary(&ontology))?,
        },
        OntologyAction::Init => {
            if store::latest_version(db)?.is_some() {
                anyhow::bail!(
                    "an ontology already exists; import a file or restore a version instead"
                );
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
        }
        OntologyAction::Export { file } => {
            let ontology = store::current(db)?.context("no ontology to export")?;
            let json = ontology.to_json()?;
            match &file {
                StdioPath::Stdio => writeln!(out, "{json}")?,
                StdioPath::Path(path) => {
                    std::fs::write(path, format!("{json}\n"))
                        .with_context(|| format!("failed to write {file}"))?;
                    writeln!(out, "wrote version {} to {file}", ontology.saved_version()?)?;
                }
            }
        }
        OntologyAction::Import { file } => {
            let text = file.read_to_string()?;
            let ontology = Ontology::from_json(&text)?;
            let stored = store::save(
                db,
                &ontology,
                Revision::reviewed(None, Some(&format!("imported from {file}"))),
            )?;
            writeln!(out, "ontology is now version {}", stored.saved_version()?)?;
        }
        OntologyAction::Versions { limit } => {
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
        }
        OntologyAction::Diff { from, to } => {
            let to = match to {
                Some(to) => to,
                None => store::latest_version(db)?.context("no ontology yet")?,
            };
            let from = from.or_else(|| to.previous()).with_context(|| {
                format!("version {to} is the first; name one to compare it with")
            })?;
            let older = store::version(db, from)?
                .with_context(|| format!("version {from} does not exist"))?;
            let newer =
                store::version(db, to)?.with_context(|| format!("version {to} does not exist"))?;
            write!(out, "{}", newer.diff(&older))?;
        }
        OntologyAction::Restore { version } => {
            let stored = store::restore(db, version, None)?;
            writeln!(
                out,
                "restored version {version} as version {}",
                stored.saved_version()?
            )?;
        }
        OntologyAction::Propose { .. }
        | OntologyAction::Review { .. }
        | OntologyAction::Accept { .. }
        | OntologyAction::Reject { .. } => {}
    }
    Ok(())
}

/// `propose`, with optional seeding from a file.
async fn run_propose(
    config: &Config,
    db: &Writer,
    action: OntologyAction,
    out: &mut impl Write,
    progress: Progress<'_>,
) -> Result<()> {
    let OntologyAction::Propose {
        auto_accept,
        documents,
        sample,
        yes,
        from,
    } = action
    else {
        return Ok(());
    };
    let text = db
        .run(move |db| Ok(rendered(|buf| seed(db, from.as_deref(), buf))))
        .await?;
    out.write_all(&text.map_err(anyhow::Error::msg)?)?;
    let pass = documents.then_some(DocumentPass {
        sample,
        assume_yes: yes,
    });
    propose(
        config,
        db,
        ProposeArgs {
            auto_accept,
            documents: pass,
        },
        out,
        progress,
    )
    .await
}
/// The review commands.
fn run_review(db: &WorkspaceDb, action: OntologyAction, out: &mut impl Write) -> Result<()> {
    match action {
        OntologyAction::Review { low_support } => {
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
                        format!(
                            " {aside} low-support candidates: `quack ontology review --low-support`."
                        )
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
                    evidence_line(c)
                )?;
            }
        }
        OntologyAction::Accept {
            ids,
            rename,
            merge_into,
            reparent,
        } => {
            if ids.is_empty() {
                anyhow::bail!("give at least one candidate id");
            }
            if ids.len() > 1 && (rename.is_some() || merge_into.is_some() || reparent.is_some()) {
                anyhow::bail!(
                    "--rename, --merge-into, and --reparent apply to one candidate at a time"
                );
            }
            let decision = if let Some(new_id) = rename {
                Decision::Rename(new_id)
            } else if let Some(target) = merge_into {
                Decision::MergeInto(target)
            } else if let Some(parent) = reparent {
                Decision::Reparent(parent)
            } else {
                Decision::Accept
            };
            let decisions: Vec<(String, Decision)> =
                ids.into_iter().map(|id| (id, decision.clone())).collect();
            let stored = candidates::accept(db, &decisions, None)?;
            writeln!(
                out,
                "accepted {} candidate(s); ontology is now version {}",
                decisions.len(),
                stored.saved_version()?
            )?;
        }
        OntologyAction::Reject { ids } => {
            let count = candidates::reject(db, &ids, None)?;
            writeln!(out, "rejected {count} candidate(s)")?;
        }
        OntologyAction::Propose { .. }
        | OntologyAction::Show { .. }
        | OntologyAction::Init
        | OntologyAction::Export { .. }
        | OntologyAction::Import { .. }
        | OntologyAction::Versions { .. }
        | OntologyAction::Diff { .. }
        | OntologyAction::Restore { .. } => {}
    }
    Ok(())
}

/// The document pass, when asked for.
struct DocumentPass {
    sample: Option<u32>,
    assume_yes: bool,
}

struct ProposeArgs {
    auto_accept: bool,
    documents: Option<DocumentPass>,
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
    args: ProposeArgs,
    out: &mut impl Write,
    progress: Progress<'_>,
) -> Result<()> {
    let current = db.run(store::current).await?;
    let (known, evidence) = (current.clone(), config.ontology.table_evidence());
    let mut proposals = db
        .run(move |db| propose_from_tables(db, known.as_ref(), &evidence))
        .await?;
    if let Some(pass) = &args.documents {
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
            llm::chat_model_display(config)
        )?;
        // Shown before the model calls, ahead of the progress lines on
        // stderr.
        out.flush()?;
        if cost.chunks == 0 {
            writeln!(out, "No ready documents to sample.")?;
        } else if !Confirm::from_yes(pass.assume_yes).ask(out, "Proceed?", Some("--yes"))? {
            writeln!(out, "Skipped the document pass.")?;
        } else {
            let from_documents =
                run_documents(config, db, current.as_ref(), &options, out, progress).await?;
            proposals.extend(from_documents);
        }
    }
    if proposals.is_empty() {
        writeln!(out, "Nothing to propose: the tables are already covered.")?;
    } else if args.auto_accept {
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

/// One progress line per extracted chunk on stderr, for the ontology and
/// graph document passes (issue #67): stdout keeps the summary.
pub(crate) fn chunk_progress(done: ChunkDone) {
    let failed = if done.failed > 0 {
        format!(", {} failed", done.failed)
    } else {
        String::new()
    };
    drop(writeln!(
        std::io::stderr(),
        "chunk {}/{} done in {} s{failed}; {} s elapsed",
        done.done,
        done.total,
        done.took.as_secs(),
        done.elapsed.as_secs()
    ));
}

/// Sample, extract with the chat model, and propose.
async fn run_documents(
    config: &Config,
    db: &Writer,
    current: Option<&Ontology>,
    options: &documents::DocumentEvidenceOptions,
    out: &mut impl Write,
    progress: Progress<'_>,
) -> Result<Vec<Candidate>> {
    let count = options.sample_chunks;
    let sample = db
        .run(move |db| documents::sample_chunks(db, count))
        .await?;
    let extractor = llm::chat_extractor(config).await?;
    let embeddings = llm::optional_embedding_model(config).await?;
    let (candidates, summary) = documents::run(
        sample,
        extractor.as_ref(),
        current,
        options,
        embeddings.as_ref(),
        config.analysis.extraction_concurrency,
        progress,
    )
    .await?;
    writeln!(
        out,
        "extracted from {} chunks ({} failed): {} candidates, {} with low support",
        summary.sampled_chunks, summary.failed_chunks, summary.candidates, summary.low_support
    )?;
    Ok(candidates)
}

/// Ask on the terminal; a non-terminal stdin means no.
/// One line of evidence for the review listing.
fn evidence_line(c: &candidates::CandidateRow) -> String {
    let e = &c.evidence;
    let get = |k: &str| {
        e.get(k)
            .map(|v| match v {
                serde_json::Value::String(s) => s.clone(),
                other => other.to_string(),
            })
            .unwrap_or_default()
    };
    if e.get("source").and_then(|v| v.as_str()) == Some("okf") {
        let examples = e
            .get("examples")
            .and_then(|x| x.as_array())
            .map(|xs| {
                xs.iter()
                    .filter_map(|x| x.as_str())
                    .collect::<Vec<_>>()
                    .join(", ")
            })
            .unwrap_or_default();
        return format!("{} bundle files, e.g. {examples}", get("files"));
    }
    if e.get("source").and_then(|v| v.as_str()) == Some("documents") {
        let examples = e
            .get("examples")
            .and_then(|x| x.as_array())
            .map(|xs| {
                xs.iter()
                    .filter_map(|x| {
                        x.get("mention")
                            .or_else(|| x.get("subject"))
                            .or_else(|| x.get("value"))
                            .and_then(|v| v.as_str())
                    })
                    .collect::<Vec<_>>()
                    .join(", ")
            })
            .unwrap_or_default();
        return format!(
            "{} mentions in {} documents{} e.g. {examples}",
            get("occurrences"),
            get("documents"),
            if c.status == CandidateStatus::LowSupport {
                " (low support)"
            } else {
                ""
            }
        );
    }
    match c.kind {
        ItemKind::Class => format!(
            "table {} ({} rows, key {})",
            get("table"),
            get("rows"),
            get("key_column")
        ),
        ItemKind::Property => format!(
            "{}.{} {} distinct {} of {} e.g. {}",
            get("table"),
            get("column"),
            get("duckdb_type"),
            get("distinct"),
            get("rows"),
            get("samples")
        ),
        ItemKind::Relation => format!(
            "{}.{} matches {}.{} for {} of values",
            get("table"),
            get("column"),
            get("target_table"),
            get("target_key"),
            get("overlap")
        ),
        ItemKind::Mapping => format!("table {}", get("table")),
    }
}

/// A readable rendering: the class tree, then relations, properties, mappings.
fn summary(ontology: &Ontology) -> String {
    fn children(ontology: &Ontology, parent: &str, depth: usize, lines: &mut Vec<String>) {
        for class in ontology.classes.iter().filter(|c| c.parent == parent) {
            let key = class
                .key
                .as_deref()
                .map_or(String::new(), |k| format!(" [key {k}]"));
            let props = if class.properties.is_empty() {
                String::new()
            } else {
                format!(" {{{}}}", class.properties.join(", "))
            };
            lines.push(format!(
                "{}- {}{key}{props}",
                "  ".repeat(depth.saturating_add(1)),
                class.id
            ));
            children(ontology, &class.id, depth.saturating_add(1), lines);
        }
    }
    let mut lines = vec![
        match ontology.version {
            Some(version) => format!("Ontology version {version}"),
            None => String::from("Ontology (unsaved)"),
        },
        String::from("classes:"),
    ];
    children(ontology, ROOT_CLASS, 0, &mut lines);
    lines.push(String::from("relations:"));
    for r in &ontology.relations {
        lines.push(format!("  - {}: {} -> {}", r.id, r.domain, r.range));
    }
    lines.push(String::from("properties:"));
    for p in &ontology.properties {
        let values = if p.values.is_empty() {
            String::new()
        } else {
            format!(" [{}]", p.values.join(", "))
        };
        lines.push(format!("  - {}: {}{values}", p.id, p.kind.as_str()));
    }
    if !ontology.mappings.is_empty() {
        lines.push(String::from("mappings:"));
        for m in &ontology.mappings {
            lines.push(format!(
                "  - {} -> {} (key {}, {} properties, {} relations)",
                m.table,
                m.class,
                m.key,
                m.properties.len(),
                m.relations.len()
            ));
        }
    }
    let mut text = lines.join("\n");
    text.push('\n');
    text
}

#[cfg(test)]
mod tests {
    use super::*;
    use quack_core::ontology::Class;

    #[test]
    fn summary_nests_subclasses_under_parents() {
        let mut ontology = Ontology::builtin_default();
        ontology.classes.push(Class {
            id: String::from("vendor"),
            parent: String::from("organization"),
            label: None,
            description: None,
            key: None,
            properties: Vec::new(),
        });
        let text = summary(&ontology);
        assert!(
            text.contains("  - organization {industry, country}\n    - vendor\n"),
            "{text}"
        );
        assert!(text.contains("  - works_at: person -> organization"));
        assert!(text.contains("  - date: date"));
    }

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
