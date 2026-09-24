//! `quack graph`: explore the knowledge graph from the shell, build it from
//! the mapped tables and the documents, keep it in step with the ontology,
//! and review merge proposals. Design doc 6.4.

use std::io::Write;

use anyhow::{Context, Result};
use clap::Subcommand;
use quack_core::config::Config;
use quack_core::extraction::ExtractionRun;
use quack_core::graph::extract::ChunkPlan;
use quack_core::graph::query::{GraphQuery, PathQuery, UnknownEntity};
use quack_core::graph::traverse::Hops;
use quack_core::graph::{ExtractSource, extract, resolve, store as graph_store, tables};
use quack_core::llm::{self, Embeddings};
use quack_core::ontology::store as ontology_store;
use quack_core::progress::RunControl;
use quack_core::storage::workspace::WorkspaceDb;
use quack_core::storage::writer::Writer;

use crate::confirm::Confirm;
use crate::text_or_json::TextOrJson;

#[derive(Subcommand)]
pub(crate) enum GraphAction {
    /// Print an entity's neighbourhood as a tree, or every entity of a class
    Search(SearchArgs),
    /// The shortest chain of relations between two entities
    Path(PathArgs),
    /// Node and edge counts, whether the graph is provisional or stale,
    /// pending merges, and what the corpus expressed that the ontology lacks
    Status {
        /// `json` prints the status as one JSON document
        #[arg(long, value_enum, default_value_t = TextOrJson::Text)]
        format: TextOrJson,
    },
    /// Build the graph: rows of mapped tables deterministically, then every
    /// chunk through the chat model (one call per chunk; asks first)
    Extract(ExtractArgs),
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
    Reject { ids: Vec<String> },
}

#[derive(clap::Args)]
pub(crate) struct SearchArgs {
    /// The entity's name as it appears in the data
    entity: Option<String>,
    /// Only this class (with its subclasses when listing)
    #[arg(long)]
    class: Option<String>,
    /// Follow only this relation
    #[arg(long)]
    relation: Option<String>,
    /// Hops out from the entity
    #[arg(long, default_value_t = Hops::NEIGHBORHOOD.get())]
    hops: u32,
    /// `json` prints the result (nodes, edges, provenance) as JSON
    #[arg(long, value_enum, default_value_t = TextOrJson::Text)]
    format: TextOrJson,
}

#[derive(clap::Args)]
pub(crate) struct PathArgs {
    from: String,
    to: String,
    #[arg(long, default_value_t = Hops::PATH.get())]
    max_hops: u32,
    /// `json` prints the path (nodes, edges, provenance) as JSON
    #[arg(long, value_enum, default_value_t = TextOrJson::Text)]
    format: TextOrJson,
}

#[derive(clap::Args)]
pub(crate) struct ExtractArgs {
    /// What to extract from: all, tables (no model calls), or documents
    #[arg(long, default_value = "all")]
    source: ExtractSource,
    /// Chunks to send to the model at most (default: all)
    #[arg(long)]
    sample: Option<u32>,
    /// Start from an empty graph instead of adding to it
    #[arg(long)]
    reset: bool,
    /// Do not ask before spending the model calls
    #[arg(long, short = 'y')]
    pub(crate) yes: bool,
}

/// A command step that runs on the workspace writer's thread: it renders
/// there into a buffer, and the bytes come back to be written to `out`
/// (which cannot cross to that thread).
pub(crate) trait RenderOnWriter {
    async fn render(
        &self,
        out: &mut impl Write,
        step: impl FnOnce(&WorkspaceDb, &mut Vec<u8>) -> Result<()> + Send + 'static,
    ) -> Result<()>;
}

impl RenderOnWriter for Writer {
    async fn render(
        &self,
        out: &mut impl Write,
        step: impl FnOnce(&WorkspaceDb, &mut Vec<u8>) -> Result<()> + Send + 'static,
    ) -> Result<()> {
        let rendered = self
            .run(move |db| {
                let mut buf = Vec::new();
                Ok(step(db, &mut buf).map(|()| buf))
            })
            .await??;
        out.write_all(&rendered)?;
        Ok(())
    }
}

/// Run one graph action, writing what the user should see to `out`
/// (stdout for the CLI, the transcript for the terminal session).
///
/// Each database step goes to the workspace writer on its own, never
/// spanning a model call, so the terminal's other work keeps going during
/// an extraction; `control` hears about every extracted chunk and can stop it.
pub(crate) async fn run(
    config: &Config,
    db: &Writer,
    action: GraphAction,
    confirm: Confirm,
    out: &mut impl Write,
    control: RunControl<'_>,
) -> Result<()> {
    match action {
        GraphAction::Search(args) => run_search(config, db, out, args).await?,
        GraphAction::Path(args) => run_path(config, db, out, args).await?,
        GraphAction::Extract(args) => {
            run_extract(config, db, out, &args, confirm.or_yes(args.yes), control).await?;
        }
        GraphAction::Status { format } => {
            db.render(out, move |db, out| {
                format.write(out, &graph_store::status(db)?)
            })
            .await?;
        }
        GraphAction::Revalidate => {
            db.render(out, |db, out| {
                let outcome = graph_store::revalidate(db)?;
                writeln!(
                    out,
                    "Dropped {} nodes and {} edges; the graph now matches ontology version {}.",
                    outcome.dropped_nodes, outcome.dropped_edges, outcome.version
                )?;
                Ok(())
            })
            .await?;
        }
        GraphAction::Review => {
            db.render(out, |db, out| {
                graph_store::mark_reviewed(db)?;
                writeln!(out, "The graph is no longer provisional.")?;
                Ok(())
            })
            .await?;
        }
        GraphAction::Merges => {
            db.render(out, |db, out| {
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
                Ok(())
            })
            .await?;
        }
        GraphAction::Merge { ids } => {
            db.render(out, move |db, out| {
                for id in &ids {
                    let m = resolve::decide(db, id, resolve::MergeDecision::Accept, None)?;
                    writeln!(out, "Merged {} into {}.", m.drop.label, m.keep.label)?;
                }
                Ok(())
            })
            .await?;
        }
        GraphAction::Reject { ids } => {
            db.render(out, move |db, out| {
                for id in &ids {
                    let m = resolve::decide(db, id, resolve::MergeDecision::Reject, None)?;
                    writeln!(out, "Kept {} and {} apart.", m.keep.label, m.drop.label)?;
                }
                Ok(())
            })
            .await?;
        }
    }
    out.flush()?;
    Ok(())
}

async fn run_search(
    config: &Config,
    db: &Writer,
    out: &mut impl Write,
    args: SearchArgs,
) -> Result<()> {
    let SearchArgs {
        entity,
        class,
        relation,
        hops,
        format,
    } = args;
    let options = config.graph.options();
    let query = GraphQuery::new(
        entity.as_deref(),
        class.as_deref(),
        relation.as_deref(),
        Some(hops),
    )?;
    let model = Embeddings::from_config(config).await?;
    let embedding = query.embedding(model.as_ref()).await?;
    let result = db
        .run(move |db| {
            let result = query.run(db, embedding.as_ref(), &options)?;
            // A walk from an entity always holds that entity, so an empty
            // one means the name resolved to nothing.
            match query.entity.as_deref() {
                Some(entity) if result.nodes.is_empty() => {
                    Err(UnknownEntity::find(db, entity, embedding.as_ref()).into())
                }
                Some(_) | None => Ok(result),
            }
        })
        .await?;
    format.write(out, &result)
}

async fn run_path(
    config: &Config,
    db: &Writer,
    out: &mut impl Write,
    args: PathArgs,
) -> Result<()> {
    let PathArgs {
        from,
        to,
        max_hops,
        format,
    } = args;
    let options = config.graph.options();
    let query = PathQuery::new(&from, &to, Some(max_hops))?;
    let model = Embeddings::from_config(config).await?;
    let ends = query.embeddings(model.as_ref()).await?;
    let max_hops = query.max_hops;
    let result = db.run(move |db| query.run(db, &ends, &options)).await?;
    if result.is_empty() && format == TextOrJson::Text {
        writeln!(out, "No path within {max_hops} hops.")?;
        return Ok(());
    }
    format.write(out, &result)
}

async fn run_extract(
    config: &Config,
    db: &Writer,
    out: &mut impl Write,
    args: &ExtractArgs,
    confirm: Confirm,
    control: RunControl<'_>,
) -> Result<()> {
    let ontology = db
        .run(ontology_store::current)
        .await?
        .context("no ontology yet: run `quack ontology init` or `quack ontology propose` first")?;
    let standing = db.run(ontology_store::current_standing).await?;
    if args.reset {
        db.run(graph_store::clear).await?;
        writeln!(out, "Cleared the graph.")?;
    }
    let sources = args.source;
    if sources.includes_tables() {
        if ontology.mappings.is_empty() {
            writeln!(
                out,
                "No mapped tables in the ontology; skipping table extraction."
            )?;
        } else {
            let mapped = ontology.clone();
            let summaries = db
                .run(move |db| tables::extract(db, &mapped, standing))
                .await?;
            for summary in summaries {
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
    if sources.includes_documents() {
        let sample = args.sample;
        let plan = db.run(move |db| ChunkPlan::new(db, sample)).await?;
        if plan.is_empty() {
            writeln!(
                out,
                "No chunks left to extract: every chunk of every ready document is on record (`--reset` starts over)."
            )?;
        } else {
            let chat = config.chat_model_ref()?;
            writeln!(
                out,
                "Document extraction: {} chunks, one model call each to {chat}.",
                plan.len()
            )?;
            out.flush()?;
            if confirm.ask(out, "Proceed?", Some("--yes"))? {
                let extractor = llm::graph_extractor(config, &ontology).await?;
                let summary = extract::run(
                    db,
                    &plan,
                    &ontology,
                    standing,
                    ExtractionRun {
                        extractor: extractor.as_ref(),
                        concurrency: config.analysis.extraction_concurrency,
                        control,
                    },
                )
                .await?;
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
                        "The documents expressed {} things the ontology lacks; `quack graph status` lists them and `quack ontology propose --documents` proposes them.",
                        summary.drift.total()
                    )?;
                }
            } else {
                writeln!(out, "Skipped the documents.")?;
            }
        }
    }
    let embeddings = Embeddings::from_config(config).await?;
    let resolved = resolve::resolve(db, embeddings.as_ref(), &config.graph.options()).await?;
    if resolved.auto_merged > 0 || resolved.proposed > 0 {
        writeln!(
            out,
            "Resolution: {} merged, {} proposed for review (`quack graph merges`).",
            resolved.auto_merged, resolved.proposed
        )?;
    }
    let version = ontology.saved_version()?;
    db.run(move |db| graph_store::set_built_with(db, version))
        .await?;
    write!(out, "{}", db.run(graph_store::status).await?)?;
    Ok(())
}
