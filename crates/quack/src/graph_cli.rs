//! `quack graph`: explore the knowledge graph from the shell, build it from
//! the mapped tables and the documents, keep it in step with the ontology,
//! and review merge proposals. Design doc 6.4.

use std::io::Write;

use anyhow::{Context, Result};
use clap::Subcommand;
use quack_core::config::Config;
use quack_core::graph::{GraphResult, GraphStatus};
use quack_core::graph::{extract, resolve, store as graph_store, tables, traverse};
use quack_core::llm;
use quack_core::ontology::store as ontology_store;
use quack_core::storage::workspace::WorkspaceDb;

#[derive(Subcommand)]
pub(crate) enum GraphAction {
    /// Print an entity's neighbourhood as a tree, or every entity of a class
    Search {
        /// The entity's name as it appears in the data
        entity: Option<String>,
        /// Only this class (with its subclasses when listing)
        #[arg(long)]
        class: Option<String>,
        /// Follow only this relation
        #[arg(long)]
        relation: Option<String>,
        /// Hops out from the entity
        #[arg(long, default_value_t = 2)]
        hops: u32,
        /// Print the result as JSON (nodes, edges, provenance)
        #[arg(long)]
        json: bool,
    },
    /// The shortest chain of relations between two entities
    Path {
        from: String,
        to: String,
        #[arg(long, default_value_t = 4)]
        max_hops: u32,
        #[arg(long)]
        json: bool,
    },
    /// Node and edge counts, whether the graph is provisional or stale,
    /// pending merges, and what the corpus expressed that the ontology lacks
    Status {
        #[arg(long)]
        json: bool,
    },
    /// Build the graph: rows of mapped tables deterministically, then every
    /// chunk through the chat model (one call per chunk; asks first)
    Extract {
        /// Only the mapped tables, no model calls
        #[arg(long, conflicts_with = "documents_only")]
        tables_only: bool,
        /// Only the documents
        #[arg(long)]
        documents_only: bool,
        /// Chunks to send to the model at most (default: all)
        #[arg(long)]
        sample: Option<u32>,
        /// Start from an empty graph instead of adding to it
        #[arg(long)]
        reset: bool,
        /// Do not ask before spending the model calls
        #[arg(long, short = 'y')]
        yes: bool,
    },
    /// Drop nodes and edges the current ontology no longer allows (no
    /// model calls) and record the version the graph now matches
    Revalidate,
    /// Mark a provisional graph reviewed
    Review,
    /// List pending merge proposals
    Merges,
    /// Apply merge proposals by id (prefixes accepted)
    Merge { ids: Vec<String> },
    /// Decline merge proposals by id (prefixes accepted)
    Unmerge { ids: Vec<String> },
}

pub(crate) async fn run(config: &Config, db: &WorkspaceDb, action: GraphAction) -> Result<()> {
    let stdout = std::io::stdout();
    let mut out = std::io::BufWriter::new(stdout.lock());
    match action {
        search @ GraphAction::Search { .. } => run_search(config, db, &mut out, search).await?,
        path @ GraphAction::Path { .. } => run_path(config, db, &mut out, path).await?,
        GraphAction::Status { json } => {
            let status = graph_store::status(db)?;
            if json {
                writeln!(out, "{}", serde_json::to_string_pretty(&status)?)?;
            } else {
                write!(out, "{}", status_text(&status))?;
            }
        }
        GraphAction::Extract {
            tables_only,
            documents_only,
            sample,
            reset,
            yes,
        } => {
            let sources = if tables_only {
                Sources::Tables
            } else if documents_only {
                Sources::Documents
            } else {
                Sources::All
            };
            run_extract(
                config,
                db,
                &mut out,
                ExtractArgs {
                    sources,
                    sample,
                    reset,
                    yes,
                },
            )
            .await?;
        }
        GraphAction::Revalidate => {
            let outcome = graph_store::revalidate(db)?;
            writeln!(
                out,
                "Dropped {} nodes and {} edges; the graph now matches ontology version {}.",
                outcome.dropped_nodes, outcome.dropped_edges, outcome.version
            )?;
        }
        GraphAction::Review => {
            graph_store::mark_reviewed(db)?;
            writeln!(out, "The graph is no longer provisional.")?;
        }
        GraphAction::Merges => {
            let pending = resolve::pending(db)?;
            if pending.is_empty() {
                writeln!(out, "No pending merges.")?;
            }
            for m in &pending {
                writeln!(
                    out,
                    "{}  {:.3}  {} ({}) <- {}",
                    m.id, m.distance, m.keep.label, m.keep.class_id, m.drop.label
                )?;
            }
        }
        GraphAction::Merge { ids } => {
            for id in &ids {
                let m = resolve::accept(db, id, None)?;
                writeln!(out, "Merged {} into {}.", m.drop.label, m.keep.label)?;
            }
        }
        GraphAction::Unmerge { ids } => {
            for id in &ids {
                let m = resolve::reject(db, id, None)?;
                writeln!(out, "Kept {} and {} apart.", m.keep.label, m.drop.label)?;
            }
        }
    }
    out.flush()?;
    Ok(())
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Sources {
    All,
    Tables,
    Documents,
}

struct ExtractArgs {
    sources: Sources,
    sample: Option<u32>,
    reset: bool,
    yes: bool,
}

async fn run_search(
    config: &Config,
    db: &WorkspaceDb,
    out: &mut impl Write,
    action: GraphAction,
) -> Result<()> {
    let GraphAction::Search {
        entity,
        class,
        relation,
        hops,
        json,
    } = action
    else {
        return Ok(());
    };
    let options = config.graph.options();
    let entity = entity.as_deref().map(str::trim).filter(|e| !e.is_empty());
    let class = class.as_deref().map(str::trim).filter(|c| !c.is_empty());
    let result = match (entity, class) {
        (None, None) => anyhow::bail!("give an entity, --class CLASS, or both"),
        (Some(entity), class) => {
            let embedding = query_embedding(config, entity).await;
            let roots = traverse::resolve_entry(db, entity, class, embedding.as_deref())?;
            if roots.is_empty() {
                anyhow::bail!("no entity matches '{entity}'");
            }
            traverse::neighborhood(db, &roots, hops, relation.as_deref(), &options)?
        }
        (None, Some(class)) => {
            let ontology = ontology_store::current(db)?;
            traverse::by_class(db, ontology.as_ref(), class, options.max_nodes, &options)?
        }
    };
    print_result(out, &result, json)
}

async fn run_path(
    config: &Config,
    db: &WorkspaceDb,
    out: &mut impl Write,
    action: GraphAction,
) -> Result<()> {
    let GraphAction::Path {
        from,
        to,
        max_hops,
        json,
    } = action
    else {
        return Ok(());
    };
    let options = config.graph.options();
    let a = query_embedding(config, &from).await;
    let b = query_embedding(config, &to).await;
    let from_nodes = traverse::resolve_entry(db, &from, None, a.as_deref())?;
    let to_nodes = traverse::resolve_entry(db, &to, None, b.as_deref())?;
    let (Some(a), Some(b)) = (from_nodes.first(), to_nodes.first()) else {
        anyhow::bail!(
            "no entity matches '{}'",
            if from_nodes.is_empty() { &from } else { &to }
        );
    };
    let result = traverse::path(db, a, b, max_hops, &options)?;
    if result.is_empty() && !json {
        writeln!(out, "No path within {max_hops} hops.")?;
        return Ok(());
    }
    print_result(out, &result, json)
}

async fn run_extract(
    config: &Config,
    db: &WorkspaceDb,
    out: &mut impl Write,
    args: ExtractArgs,
) -> Result<()> {
    let ontology = ontology_store::current(db)?
        .context("no ontology yet: run `quack ontology init` or `quack ontology propose` first")?;
    let provisional = ontology_store::current_is_auto_accepted(db)?;
    if args.reset {
        graph_store::clear(db)?;
        writeln!(out, "Cleared the graph.")?;
    }
    if args.sources != Sources::Documents {
        if ontology.mappings.is_empty() {
            writeln!(
                out,
                "No mapped tables in the ontology; skipping table extraction."
            )?;
        } else {
            for summary in tables::extract(db, &ontology, provisional)? {
                match &summary.skipped {
                    Some(reason) => writeln!(out, "Table {}: skipped, {reason}", summary.table)?,
                    None => writeln!(
                        out,
                        "Table {}: {} rows -> {} nodes, {} edges",
                        summary.table, summary.rows, summary.nodes, summary.edges
                    )?,
                }
            }
        }
    }
    if args.sources != Sources::Tables {
        let chunks = extract::chunks(db, args.sample)?;
        if chunks.is_empty() {
            writeln!(out, "No ready documents to extract from.")?;
        } else {
            let chat = config.chat_model_ref()?;
            writeln!(
                out,
                "Document extraction: {} chunks, one model call each to {chat}.",
                chunks.len()
            )?;
            if args.yes || confirm(out)? {
                let extractor = llm::graph_extractor(config, &ontology).await?;
                let summary =
                    extract::run(db, chunks, extractor.as_ref(), &ontology, provisional).await?;
                writeln!(
                    out,
                    "Extracted {} nodes and {} edges from {} chunks ({} failed, {} edges did not fit).",
                    summary.nodes,
                    summary.edges,
                    summary.chunks,
                    summary.failed_chunks,
                    summary.invalid_edges
                )?;
                if summary.drift.total() > 0 {
                    writeln!(
                        out,
                        "The documents expressed {} things the ontology lacks; `quack graph status` lists them and `quack ontology propose --extend --documents` proposes them.",
                        summary.drift.total()
                    )?;
                }
            } else {
                writeln!(out, "Skipped the documents.")?;
            }
        }
    }
    let embeddings = llm::optional_embedding_model(config).await?;
    let resolved = resolve::resolve(db, embeddings.as_ref(), &config.graph.options()).await?;
    if resolved.auto_merged > 0 || resolved.proposed > 0 {
        writeln!(
            out,
            "Resolution: {} merged, {} proposed for review (`quack graph merges`).",
            resolved.auto_merged, resolved.proposed
        )?;
    }
    graph_store::set_built_with(db, ontology.version)?;
    write!(out, "{}", status_text(&graph_store::status(db)?))?;
    Ok(())
}

fn confirm(out: &mut impl Write) -> Result<bool> {
    write!(out, "Proceed? [y/N] ")?;
    out.flush()?;
    let mut answer = String::new();
    std::io::stdin().read_line(&mut answer)?;
    Ok(matches!(answer.trim(), "y" | "Y" | "yes"))
}

async fn query_embedding(config: &Config, text: &str) -> Option<Vec<f32>> {
    let model = llm::optional_embedding_model(config).await.ok()??;
    llm::embed_query(&model, text).await.ok()
}

fn print_result(out: &mut impl Write, result: &GraphResult, json: bool) -> Result<()> {
    if json {
        writeln!(out, "{}", serde_json::to_string_pretty(result)?)?;
    } else {
        write!(out, "{}", traverse::render_tree(result))?;
    }
    Ok(())
}

/// The status as `quack graph status` prints it.
pub(crate) fn status_text(status: &GraphStatus) -> String {
    let mut lines = vec![format!(
        "Graph: {} nodes, {} edges (ontology version {}, built with {}){}{}",
        status.nodes,
        status.edges,
        status.ontology_version,
        status.built_with_version,
        if status.stale {
            "; stale: run `quack graph revalidate` or `quack graph extract`"
        } else {
            ""
        },
        if status.provisional() {
            "; provisional: built from an unreviewed ontology, `quack graph review` clears it"
        } else {
            ""
        }
    )];
    if status.pending_merges > 0 {
        lines.push(format!(
            "{} merge proposals pending: `quack graph merges`",
            status.pending_merges
        ));
    }
    if !status.missing_tables.is_empty() {
        lines.push(format!(
            "Mapped tables no longer in the workspace (extraction skips them): {}",
            status.missing_tables.join(", ")
        ));
    }
    if status.drift.total() > 0 {
        let mut items: Vec<String> = status
            .drift
            .classes
            .iter()
            .map(|(k, v)| format!("class {k} ({v})"))
            .chain(
                status
                    .drift
                    .relations
                    .iter()
                    .map(|(k, v)| format!("relation {k} ({v})")),
            )
            .collect();
        items.sort();
        lines.push(format!(
            "The corpus expressed {} things the ontology lacks: {}",
            status.drift.total(),
            items.join(", ")
        ));
    }
    let mut text = lines.join("\n");
    text.push('\n');
    text
}
