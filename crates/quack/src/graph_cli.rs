//! `quack graph`: explore the knowledge graph from the shell, build it from
//! the mapped tables and the documents, keep it in step with the ontology,
//! and review merge proposals. Design doc 6.4.

use std::io::Write;

use anyhow::{Context, Result};
use clap::Subcommand;
use quack_core::analysis::tools::{FindPathArgs, NonBlank, SearchGraphArgs};
use quack_core::config::Config;
use quack_core::extraction::ExtractionRun;
use quack_core::graph::extract::ChunkPlan;
use quack_core::graph::query::UnknownEntity;
use quack_core::graph::store::{Assertion, Keep, NewEdge, NewNode, NodeEdit, Revalidation};
use quack_core::graph::traverse::Hops;
use quack_core::graph::{
    ExtractSource, Properties, Standing, extract, resolve, store as graph_store, tables,
};
use quack_core::ids::{ClassId, EdgeId, RelationId};
use quack_core::llm::{self, Embeddings};
use quack_core::ontology::{Ontology, store as ontology_store};
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
    /// model calls) and record the version the graph now matches; says
    /// what would go and asks first
    Revalidate {
        /// Do not ask before dropping
        // Not the id `yes`: the terminal session hides that flag as one it
        // supplies itself, and this one only the person may give.
        #[arg(id = "drop", long = "yes", short = 'y')]
        yes: bool,
    },
    /// Mark a provisional graph reviewed
    Review,
    /// List pending merge proposals
    Merges,
    /// Apply merge proposals by id (prefixes accepted)
    Merge { ids: Vec<String> },
    /// Decline merge proposals by id (prefixes accepted)
    Reject { ids: Vec<String> },
    /// Assert a node or an edge the documents and tables did not yield
    #[command(subcommand)]
    Add(AddWhat),
    /// Correct a node: its label, class, or properties
    Set(SetArgs),
    /// Delete a node (with its edges) or an edge
    #[command(subcommand)]
    Delete(DeleteWhat),
}

#[derive(Subcommand)]
pub(crate) enum AddWhat {
    /// A node of a class; one with the same label and class takes the
    /// properties instead
    Node {
        label: String,
        #[arg(long)]
        class: String,
        /// A property, as KEY=VALUE (repeatable)
        #[arg(long = "property", value_name = "KEY=VALUE")]
        properties: Vec<String>,
        /// Why: kept with the assertion
        #[arg(long)]
        note: Option<String>,
    },
    /// An edge between two nodes (by id, or by exact label)
    Edge {
        from: String,
        relation: String,
        to: String,
        /// A property, as KEY=VALUE (repeatable)
        #[arg(long = "property", value_name = "KEY=VALUE")]
        properties: Vec<String>,
        /// Why: kept with the assertion
        #[arg(long)]
        note: Option<String>,
    },
}

#[derive(clap::Args)]
pub(crate) struct SetArgs {
    /// The node, by id or exact label
    node: String,
    /// Only a node of this class, when the label names several
    #[arg(long)]
    class: Option<String>,
    #[arg(long)]
    label: Option<String>,
    /// Move the node to this class (its edges must still fit)
    #[arg(long = "to-class")]
    to_class: Option<String>,
    /// Set a property, as KEY=VALUE (repeatable)
    #[arg(long = "property", value_name = "KEY=VALUE")]
    properties: Vec<String>,
    /// Remove a property (repeatable)
    #[arg(long = "unset", value_name = "KEY")]
    unset: Vec<String>,
    /// Why: kept with the assertion
    #[arg(long)]
    note: Option<String>,
}

#[derive(Subcommand)]
pub(crate) enum DeleteWhat {
    /// A node by id or exact label, with every edge at it
    Node {
        node: String,
        #[arg(long)]
        class: Option<String>,
    },
    /// An edge by id
    Edge { id: String },
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
    /// Start from an empty graph instead of adding to it; nodes and edges
    /// a person asserted stay unless --all is given too
    #[arg(long)]
    reset: bool,
    /// With --reset: drop asserted nodes and edges too
    #[arg(long, requires = "reset")]
    all: bool,
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
        GraphAction::Revalidate { yes } => run_revalidate(db, out, yes, confirm).await?,
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
        GraphAction::Add(what) => run_add(db, out, what).await?,
        GraphAction::Set(args) => run_set(db, out, args).await?,
        GraphAction::Delete(what) => run_delete(db, out, what).await?,
    }
    out.flush()?;
    Ok(())
}

async fn run_add(db: &Writer, out: &mut impl Write, what: AddWhat) -> Result<()> {
    match what {
        AddWhat::Node {
            label,
            class,
            properties,
            note,
        } => {
            let properties = Properties::from(Properties::parse_pairs(&properties)?);
            let assertion = Assertion { author: None, note };
            db.render(out, move |db, out| {
                let added = graph_store::create_node(
                    db,
                    &NewNode {
                        label,
                        class_id: ClassId::from(class),
                        properties,
                        standing: Standing::Reviewed,
                    },
                    &assertion,
                )?;
                writeln!(
                    out,
                    "{} {} ({}).",
                    if added.created {
                        "Added"
                    } else {
                        "Already there; asserted"
                    },
                    added.subject,
                    added.subject.id
                )?;
                Ok(())
            })
            .await
        }
        AddWhat::Edge {
            from,
            relation,
            to,
            properties,
            note,
        } => {
            let properties = Properties::from(Properties::parse_pairs(&properties)?);
            let assertion = Assertion { author: None, note };
            db.render(out, move |db, out| {
                let source = graph_store::find_node(db, &from, None)?;
                let target = graph_store::find_node(db, &to, None)?;
                let added = graph_store::create_edge(
                    db,
                    &NewEdge {
                        source: source.id,
                        target: target.id,
                        relation: RelationId::from(relation),
                        properties,
                    },
                    &assertion,
                )?;
                writeln!(
                    out,
                    "{} {} -{}-> {} ({}).",
                    if added.created {
                        "Added"
                    } else {
                        "Already there; asserted"
                    },
                    source.label,
                    added.subject.relation_id,
                    target.label,
                    added.subject.id
                )?;
                Ok(())
            })
            .await
        }
    }
}

async fn run_set(db: &Writer, out: &mut impl Write, args: SetArgs) -> Result<()> {
    let SetArgs {
        node,
        class,
        label,
        to_class,
        properties,
        unset,
        note,
    } = args;
    let mut patch = Properties::parse_pairs(&properties)?;
    for key in unset {
        patch.insert(key, serde_json::Value::Null);
    }
    let edit = NodeEdit {
        label,
        class: to_class.map(ClassId::from),
        properties: (!patch.is_empty()).then_some(patch),
    };
    if edit.is_empty() {
        anyhow::bail!("nothing to change: give --label, --to-class, --property, or --unset");
    }
    let assertion = Assertion { author: None, note };
    db.render(out, move |db, out| {
        let found = graph_store::find_node(db, &node, class.as_deref())?;
        let updated = graph_store::update_node(db, &found.id, &edit, &assertion)?;
        writeln!(out, "Updated {updated} ({}).", updated.id)?;
        Ok(())
    })
    .await
}

async fn run_delete(db: &Writer, out: &mut impl Write, what: DeleteWhat) -> Result<()> {
    match what {
        DeleteWhat::Node { node, class } => {
            db.render(out, move |db, out| {
                let found = graph_store::find_node(db, &node, class.as_deref())?;
                let deleted = graph_store::delete_node(db, &found.id)?;
                writeln!(out, "Deleted {deleted} and its edges.")?;
                Ok(())
            })
            .await
        }
        DeleteWhat::Edge { id } => {
            db.render(out, move |db, out| {
                let deleted = graph_store::delete_edge(db, &EdgeId::from(id))?;
                writeln!(
                    out,
                    "Deleted edge {} -{}-> {}.",
                    deleted.source_node_id, deleted.relation_id, deleted.target_node_id
                )?;
                Ok(())
            })
            .await
        }
    }
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
    let options = config.graph;
    let query = SearchGraphArgs {
        entity: NonBlank::new(entity.as_deref()),
        class: NonBlank::new(class.as_deref()),
        relation: NonBlank::new(relation.as_deref()),
        hops: Some(hops),
    }
    .query()?;
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
    let options = config.graph;
    let query = FindPathArgs {
        from,
        to,
        max_hops: Some(max_hops),
    }
    .query()?;
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

/// `quack graph revalidate`: with something to drop, say what and ask.
async fn run_revalidate(
    db: &Writer,
    out: &mut impl Write,
    yes: bool,
    confirm: Confirm,
) -> Result<()> {
    let preview = db.run(Revalidation::preview).await?;
    if !preview.is_empty() {
        let question = format!(
            "{preview}If a class or relation was renamed, restore the earlier ontology version and \
             use `quack ontology rename`, which moves its nodes and edges. Dropped document \
             nodes come back only with `quack graph extract --reset`.\nDrop them?"
        );
        if !confirm.ask_to_drop(yes, out, &question)? {
            writeln!(out, "Nothing dropped; the graph is still stale.")?;
            return Ok(());
        }
    }
    db.render(out, |db, out| {
        let outcome = graph_store::revalidate(db)?;
        writeln!(
            out,
            "Dropped {} nodes and {} edges; the graph now matches ontology version {}.",
            outcome.dropped_nodes, outcome.dropped_edges, outcome.version
        )?;
        Ok(())
    })
    .await
}

/// Every mapping's rows into the graph, one line per table.
async fn extract_tables(
    db: &Writer,
    out: &mut impl Write,
    ontology: &Ontology,
    standing: Standing,
) -> Result<()> {
    if ontology.mappings.is_empty() {
        writeln!(
            out,
            "No mapped tables in the ontology; skipping table extraction."
        )?;
        return Ok(());
    }
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
    Ok(())
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
        let keep = if args.all {
            Keep::Nothing
        } else {
            Keep::Asserted
        };
        db.run(move |db| graph_store::clear(db, keep)).await?;
        writeln!(
            out,
            "{}",
            match keep {
                Keep::Asserted => "Cleared the graph, keeping what people asserted.",
                Keep::Nothing => "Cleared the graph.",
            }
        )?;
    }
    let sources = args.source;
    if sources.includes_tables() {
        extract_tables(db, out, &ontology, standing).await?;
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
    let resolved = resolve::resolve(db, embeddings.as_ref(), &config.graph).await?;
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

#[cfg(test)]
mod tests {
    use quack_core::embedding::Dimension;
    use quack_core::graph::store::NewNode;
    use quack_core::graph::{Properties, Standing};
    use quack_core::ids::ClassId;
    use quack_core::ontology::Ontology;
    use quack_core::ontology::store::Revision;

    use super::*;

    #[expect(clippy::panic, reason = "test failure path")]
    fn fail(msg: &str) -> ! {
        panic!("{msg}")
    }

    /// A graph of one person and one place, built with version 1, under
    /// an ontology whose version 2 no longer defines `place`.
    fn stale_graph() -> Writer {
        let db =
            WorkspaceDb::open_in_memory(Dimension::new(4)).unwrap_or_else(|e| fail(&e.to_string()));
        let mut ontology = Ontology::builtin_default();
        let built = ontology_store::save(&db, &ontology, Revision::reviewed(None, None))
            .and_then(|stored| stored.saved_version())
            .unwrap_or_else(|e| fail(&e.to_string()));
        for (label, class) in [("Ada", "person"), ("Nairobi", "place")] {
            graph_store::upsert_node(
                &db,
                &NewNode {
                    label: String::from(label),
                    class_id: ClassId::from(class),
                    properties: Properties::default(),
                    standing: Standing::Reviewed,
                },
            )
            .unwrap_or_else(|e| fail(&e.to_string()));
        }
        graph_store::set_built_with(&db, built).unwrap_or_else(|e| fail(&e.to_string()));
        ontology.classes.retain(|c| c.id != "place");
        ontology
            .relations
            .retain(|r| r.domain != "place" && r.range != "place");
        ontology_store::save(&db, &ontology, Revision::reviewed(None, None))
            .unwrap_or_else(|e| fail(&e.to_string()));
        Writer::spawn(db).unwrap_or_else(|e| fail(&e.to_string()))
    }

    async fn revalidate(db: &Writer, yes: bool) -> Result<String> {
        let mut out = Vec::new();
        run(
            &Config::default(),
            db,
            GraphAction::Revalidate { yes },
            Confirm::Assume,
            &mut out,
            RunControl::unobserved(),
        )
        .await?;
        Ok(String::from_utf8_lossy(&out).into_owned())
    }

    /// Where nobody can be asked, a revalidation that would drop
    /// something says what and stops; `--yes` drops it; and with nothing
    /// to drop it runs unasked.
    #[tokio::test]
    async fn revalidate_says_what_it_drops_and_needs_yes_to_drop_it() {
        let db = stale_graph();
        let refused = revalidate(&db, false)
            .await
            .err()
            .map(|e| e.to_string())
            .unwrap_or_default();
        for part in [
            "Revalidating against ontology version 2 drops 1 nodes and 0 edges:",
            "class place, which the ontology no longer defines: 1 nodes",
            "`quack ontology rename`",
            "Drop them? Nobody to ask here, so nothing was dropped; --yes goes ahead.",
        ] {
            assert!(refused.contains(part), "{part:?} missing from {refused}");
        }
        let status = db
            .run(graph_store::status)
            .await
            .unwrap_or_else(|e| fail(&e.to_string()));
        assert!(status.stale && status.nodes == 2, "{status}");

        let dropped = revalidate(&db, true)
            .await
            .unwrap_or_else(|e| fail(&e.to_string()));
        assert_eq!(
            dropped,
            "Dropped 1 nodes and 0 edges; the graph now matches ontology version 2.\n"
        );
        let again = revalidate(&db, false)
            .await
            .unwrap_or_else(|e| fail(&e.to_string()));
        assert_eq!(
            again,
            "Dropped 0 nodes and 0 edges; the graph now matches ontology version 2.\n"
        );
    }
}

#[cfg(test)]
mod edit_tests {
    use quack_core::embedding::Dimension;
    use quack_core::ontology::Ontology;
    use quack_core::ontology::store::Revision;

    use super::*;

    #[expect(clippy::panic, reason = "test failure path")]
    fn fail(msg: &str) -> ! {
        panic!("{msg}")
    }

    async fn graph(db: &Writer, action: GraphAction) -> Result<String> {
        let mut out = Vec::new();
        run(
            &Config::default(),
            db,
            action,
            Confirm::Assume,
            &mut out,
            RunControl::unobserved(),
        )
        .await?;
        Ok(String::from_utf8_lossy(&out).into_owned())
    }

    /// An empty graph under the built-in ontology.
    fn empty_graph() -> Writer {
        let db =
            WorkspaceDb::open_in_memory(Dimension::new(4)).unwrap_or_else(|e| fail(&e.to_string()));
        ontology_store::save(
            &db,
            &Ontology::builtin_default(),
            Revision::reviewed(None, None),
        )
        .unwrap_or_else(|e| fail(&e.to_string()));
        Writer::spawn(db).unwrap_or_else(|e| fail(&e.to_string()))
    }

    fn node(label: &str, class: &str, properties: &[&str], note: Option<&str>) -> GraphAction {
        GraphAction::Add(AddWhat::Node {
            label: String::from(label),
            class: String::from(class),
            properties: properties.iter().map(|p| String::from(*p)).collect(),
            note: note.map(String::from),
        })
    }

    /// `quack graph add` on the built-in ontology: a node by label (again
    /// finds it), a refusal from the ontology, and an edge by labels.
    #[tokio::test]
    async fn add_writes_asserted_nodes_and_edges() {
        let db = empty_graph();
        let added = graph(
            &db,
            node(
                "Ada Lovelace",
                "person",
                &["born=1815", "role=analyst"],
                Some("checked"),
            ),
        )
        .await
        .unwrap_or_else(|e| fail(&e.to_string()));
        assert!(
            added.starts_with("Added Ada Lovelace (person) ("),
            "{added}"
        );
        let again = graph(&db, node("ada lovelace", "person", &[], None))
            .await
            .unwrap_or_else(|e| fail(&e.to_string()));
        assert!(
            again.starts_with("Already there; asserted Ada Lovelace"),
            "{again}"
        );
        let refused = graph(&db, node("Babbage", "robot", &[], None))
            .await
            .err()
            .map(|e| e.to_string())
            .unwrap_or_default();
        assert!(refused.contains("no class 'robot'"), "{refused}");
        graph(&db, node("London", "place", &[], None))
            .await
            .unwrap_or_else(|e| fail(&e.to_string()));
        let edge = graph(
            &db,
            GraphAction::Add(AddWhat::Edge {
                from: String::from("Ada Lovelace"),
                relation: String::from("mentions"),
                to: String::from("London"),
                properties: Vec::new(),
                note: None,
            }),
        )
        .await
        .unwrap_or_else(|e| fail(&e.to_string()));
        assert!(
            edge.starts_with("Added Ada Lovelace -mentions-> London ("),
            "{edge}"
        );
        let bad_pair = Properties::parse_pairs(["novalue"])
            .err()
            .map(|e| e.to_string())
            .unwrap_or_default();
        assert!(bad_pair.contains("is not KEY=VALUE"), "{bad_pair}");
    }

    /// `quack graph set` corrects a node found by label; an empty edit is
    /// refused; `delete` takes a node with its edges.
    #[tokio::test]
    async fn set_and_delete_correct_and_remove_nodes() {
        let db = empty_graph();
        for action in [
            node(
                "Ada Lovelace",
                "person",
                &["born=1815", "role=analyst"],
                None,
            ),
            node("London", "place", &[], None),
            GraphAction::Add(AddWhat::Edge {
                from: String::from("Ada Lovelace"),
                relation: String::from("mentions"),
                to: String::from("London"),
                properties: Vec::new(),
                note: None,
            }),
        ] {
            graph(&db, action)
                .await
                .unwrap_or_else(|e| fail(&e.to_string()));
        }
        let set = graph(
            &db,
            GraphAction::Set(SetArgs {
                node: String::from("Ada Lovelace"),
                class: None,
                label: Some(String::from("Augusta Ada King")),
                to_class: None,
                properties: vec![String::from("born=1815-12-10")],
                unset: vec![String::from("role")],
                note: None,
            }),
        )
        .await
        .unwrap_or_else(|e| fail(&e.to_string()));
        assert!(
            set.starts_with("Updated Augusta Ada King (person)"),
            "{set}"
        );
        let node = db
            .run(|db| graph_store::find_node(db, "Augusta Ada King", None))
            .await
            .unwrap_or_else(|e| fail(&e.to_string()));
        assert_eq!(
            node.properties.get("born"),
            Some(&serde_json::json!("1815-12-10"))
        );
        assert_eq!(node.properties.get("role"), None);
        let nothing = graph(
            &db,
            GraphAction::Set(SetArgs {
                node: String::from("Augusta Ada King"),
                class: None,
                label: None,
                to_class: None,
                properties: Vec::new(),
                unset: Vec::new(),
                note: None,
            }),
        )
        .await
        .err()
        .map(|e| e.to_string())
        .unwrap_or_default();
        assert!(nothing.contains("nothing to change"), "{nothing}");
        let deleted = graph(
            &db,
            GraphAction::Delete(DeleteWhat::Node {
                node: String::from("London"),
                class: None,
            }),
        )
        .await
        .unwrap_or_else(|e| fail(&e.to_string()));
        assert_eq!(deleted, "Deleted London (place) and its edges.\n");
        let status = db
            .run(graph_store::status)
            .await
            .unwrap_or_else(|e| fail(&e.to_string()));
        assert_eq!((status.nodes, status.edges), (1, 0), "{status}");
    }
}
