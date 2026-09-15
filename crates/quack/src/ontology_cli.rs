//! `quack ontology`: show, install the default, export and import JSON,
//! list versions, diff, restore, and propose and review candidates. The
//! ontology lives in the workspace file; a file on disk is only ever a copy.

use std::io::{Read, Write};

use anyhow::{Context, Result};
use clap::Subcommand;
use quack_core::config::Config;
use quack_core::ontology::Ontology;
use quack_core::ontology::induction::{Decision, propose_from_tables};
use quack_core::ontology::{candidates, documents, store};
use quack_core::storage::workspace::WorkspaceDb;

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
    Export { file: String },
    /// Validate a JSON ontology and store it as a new version (- for stdin)
    Import { file: String },
    /// List versions, newest first
    Versions {
        #[arg(long, default_value_t = 20)]
        limit: u32,
    },
    /// What changed between two versions (defaults: previous and current)
    Diff { from: Option<u32>, to: Option<u32> },
    /// Store an earlier version as the newest one
    Restore { version: u32 },
    /// Propose classes, properties, keys, relations, and mappings from the
    /// tables (no model calls) and, with --documents, from a sample of the
    /// documents (one model call per chunk) into the review queue
    Propose {
        /// Propose only what the current ontology lacks
        #[arg(long)]
        extend: bool,
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
        #[arg(long, conflicts_with_all = ["merge_into", "parent"])]
        rename: Option<String>,
        /// Treat the candidate as this existing class, relation, or property
        #[arg(long)]
        merge_into: Option<String>,
        /// Accept a class under this parent
        #[arg(long)]
        parent: Option<String>,
    },
    /// Reject candidates by id (prefixes accepted)
    Reject { ids: Vec<String> },
}

pub(crate) async fn run(config: &Config, db: &WorkspaceDb, action: OntologyAction) -> Result<()> {
    let stdout = std::io::stdout();
    let mut out = std::io::BufWriter::new(stdout.lock());
    match action {
        propose_action @ OntologyAction::Propose { .. } => {
            run_propose(config, db, propose_action, &mut out).await?;
        }
        review @ (OntologyAction::Review { .. }
        | OntologyAction::Accept { .. }
        | OntologyAction::Reject { .. }) => run_review(db, review, &mut out)?,
        manage => run_manage(db, manage, &mut out)?,
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
            if store::latest_version(db)? > 0 {
                anyhow::bail!(
                    "an ontology already exists; import a file or restore a version instead"
                );
            }
            let stored = store::save(
                db,
                &Ontology::builtin_default(),
                None,
                Some("built-in default"),
            )?;
            writeln!(
                out,
                "installed the built-in ontology as version {}",
                stored.version
            )?;
        }
        OntologyAction::Export { file } => {
            let ontology = store::current(db)?.context("no ontology to export")?;
            let json = ontology.to_json()?;
            if file == "-" {
                writeln!(out, "{json}")?;
            } else {
                std::fs::write(&file, format!("{json}\n"))
                    .with_context(|| format!("failed to write {file}"))?;
                writeln!(out, "wrote version {} to {file}", ontology.version)?;
            }
        }
        OntologyAction::Import { file } => {
            let text = if file == "-" {
                let mut buf = String::new();
                std::io::stdin().read_to_string(&mut buf)?;
                buf
            } else {
                std::fs::read_to_string(&file).with_context(|| format!("failed to read {file}"))?
            };
            let ontology = Ontology::from_json(&text)?;
            let stored = store::save(db, &ontology, None, Some(&format!("imported from {file}")))?;
            writeln!(out, "ontology is now version {}", stored.version)?;
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
            let latest = store::latest_version(db)?;
            let to = to.unwrap_or(latest);
            let from = from.unwrap_or(to.saturating_sub(1));
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
                stored.version
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
    db: &WorkspaceDb,
    action: OntologyAction,
    out: &mut impl Write,
) -> Result<()> {
    let OntologyAction::Propose {
        extend,
        auto_accept,
        documents,
        sample,
        yes,
        from,
    } = action
    else {
        return Ok(());
    };
    let seeded = seed(db, from.as_deref(), out)?;
    let pass = documents.then_some(DocumentPass {
        sample,
        assume_yes: yes,
    });
    propose(
        config,
        db,
        ProposeArgs {
            extend: extend || seeded,
            auto_accept,
            documents: pass,
        },
        out,
    )
    .await
}
/// The review commands.
fn run_review(db: &WorkspaceDb, action: OntologyAction, out: &mut impl Write) -> Result<()> {
    match action {
        OntologyAction::Review { low_support } => {
            let pending = if low_support {
                candidates::low_support(db)?
            } else {
                candidates::pending(db)?
            };
            if pending.is_empty() && low_support {
                writeln!(out, "No low-support candidates.")?;
            } else if pending.is_empty() {
                let aside = candidates::low_support(db)?.len();
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
            parent,
        } => {
            if ids.is_empty() {
                anyhow::bail!("give at least one candidate id");
            }
            if ids.len() > 1 && (rename.is_some() || merge_into.is_some() || parent.is_some()) {
                anyhow::bail!(
                    "--rename, --merge-into, and --parent apply to one candidate at a time"
                );
            }
            let decision = if let Some(new_id) = rename {
                Decision::Rename(new_id)
            } else if let Some(target) = merge_into {
                Decision::MergeInto(target)
            } else if let Some(parent) = parent {
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
                stored.version
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
    extend: bool,
    auto_accept: bool,
    documents: Option<DocumentPass>,
}

/// `--from FILE`: store the file as a new version so the proposal only
/// adds what it lacks. Returns whether a seed was applied.
fn seed(db: &WorkspaceDb, from: Option<&str>, out: &mut impl Write) -> Result<bool> {
    let Some(file) = from else {
        return Ok(false);
    };
    let text = std::fs::read_to_string(file).with_context(|| format!("failed to read {file}"))?;
    let ontology = Ontology::from_json(&text)?;
    let stored = store::save(db, &ontology, None, Some(&format!("seeded from {file}")))?;
    writeln!(out, "seeded version {} from {file}", stored.version)?;
    Ok(true)
}

/// `quack ontology propose`: table evidence, plus document evidence when
/// asked, into the queue, or straight into a version with `--auto-accept`.
async fn propose(
    config: &Config,
    db: &WorkspaceDb,
    args: ProposeArgs,
    out: &mut impl Write,
) -> Result<()> {
    let current = store::current(db)?;
    let base = if args.extend { current.as_ref() } else { None };
    let mut proposals = propose_from_tables(
        db,
        base.or(current.as_ref()),
        &config.ontology.table_evidence(),
    )?;
    if let Some(pass) = &args.documents {
        let mut options = config.ontology.document_evidence();
        if let Some(n) = pass.sample {
            options.sample_chunks = n;
        }
        let cost = documents::estimate(db, &options)?;
        writeln!(
            out,
            "Document evidence: {} chunks sampled across {} documents, {} model calls to {}.",
            cost.chunks,
            cost.documents,
            cost.model_calls,
            quack_core::llm::chat_model_display(config)
        )?;
        if cost.chunks == 0 {
            writeln!(out, "No ready documents to sample.")?;
        } else if !pass.assume_yes && !confirm(out)? {
            writeln!(out, "Skipped the document pass.")?;
        } else {
            let from_documents = run_documents(config, db, current.as_ref(), &options, out).await?;
            proposals.extend(from_documents);
        }
    }
    if proposals.is_empty() {
        writeln!(out, "Nothing to propose: the tables are already covered.")?;
    } else if args.auto_accept {
        candidates::store_run(db, &proposals)?;
        let stored = candidates::accept_all(db, None)?;
        writeln!(
            out,
            "accepted {} proposals; ontology is now version {}",
            proposals.len(),
            stored.version
        )?;
    } else {
        candidates::store_run(db, &proposals)?;
        let by_kind = |kind: &str| {
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
            by_kind("class"),
            by_kind("property"),
            by_kind("relation"),
            by_kind("mapping")
        )?;
    }
    Ok(())
}

/// Sample, extract with the chat model, and propose.
async fn run_documents(
    config: &Config,
    db: &WorkspaceDb,
    current: Option<&Ontology>,
    options: &documents::DocumentEvidenceOptions,
    out: &mut impl Write,
) -> Result<Vec<quack_core::ontology::induction::Candidate>> {
    let sample = documents::sample_chunks(db, options.sample_chunks)?;
    let extractor = quack_core::llm::chat_extractor(config).await?;
    let embeddings = quack_core::llm::optional_embedding_model(config).await?;
    let (candidates, summary) = documents::run(
        sample,
        extractor.as_ref(),
        current,
        options,
        embeddings.as_ref(),
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
fn confirm(out: &mut impl Write) -> Result<bool> {
    use std::io::{BufRead, IsTerminal};
    if !std::io::stdin().is_terminal() {
        writeln!(
            out,
            "Pass --yes to run the document pass without a terminal."
        )?;
        return Ok(false);
    }
    write!(out, "Proceed? [y/N] ")?;
    out.flush()?;
    let mut line = String::new();
    std::io::stdin().lock().read_line(&mut line)?;
    Ok(matches!(line.trim(), "y" | "Y" | "yes"))
}

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
            if c.status == "low_support" {
                " (low support)"
            } else {
                ""
            }
        );
    }
    match c.kind.as_str() {
        "class" => format!(
            "table {} ({} rows, key {})",
            get("table"),
            get("rows"),
            get("key_column")
        ),
        "property" => format!(
            "{}.{} {} distinct {} of {} e.g. {}",
            get("table"),
            get("column"),
            get("duckdb_type"),
            get("distinct"),
            get("rows"),
            get("samples")
        ),
        "relation" => format!(
            "{}.{} matches {}.{} for {} of values",
            get("table"),
            get("column"),
            get("target_table"),
            get("target_key"),
            get("overlap")
        ),
        _ => format!("table {}", get("table")),
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
        format!("Ontology version {}", ontology.version),
        String::from("classes:"),
    ];
    children(ontology, quack_core::ontology::ROOT_CLASS, 0, &mut lines);
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

    #[test]
    fn summary_nests_subclasses_under_parents() {
        let mut ontology = Ontology::builtin_default();
        ontology.classes.push(quack_core::ontology::Class {
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
}
