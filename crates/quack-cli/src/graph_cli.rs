//! `quack graph`: explore the knowledge graph from the shell, build it from
//! the mapped tables and the documents, keep it in step with the ontology,
//! and review merge proposals. Design doc 6.4.

use std::io::Write;

use anyhow::{Context, Result};
use clap::Subcommand;
use quack_core::analysis::tools::{FindPathArgs, NonBlank, SearchGraphArgs};
use quack_core::config::Config;
use quack_core::extraction::ExtractionRun;
use quack_core::graph::export::{Destination, GraphExport, GraphFormat, ProvisionalExport};
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
use crate::stdio::StdioPath;
use crate::text_or_json::TextOrJson;

#[derive(Subcommand)]
pub enum GraphAction {
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
    /// Write the whole graph for Gephi, Neo4j, or `NetworkX`: a CSV bundle
    /// (nodes.csv, edges.csv, provenance.csv), `GraphML`, or JSON-LD, each
    /// node and edge with its provenance
    Export(ExportArgs),
}

#[derive(clap::Args)]
pub struct ExportArgs {
    /// Directory to write (created), or - for stdout: a tar of the CSV
    /// bundle, or the `GraphML` or JSON-LD document
    dir: StdioPath,
    /// csv, graphml, or jsonld
    #[arg(long, default_value = "csv")]
    format: GraphFormat,
    /// Include nodes and edges built from an unreviewed ontology
    #[arg(long)]
    include_provisional: bool,
}

impl ExportArgs {
    /// Whether the export goes to standard output, which only the command
    /// line writes (`run_graph_export`).
    #[must_use]
    pub fn to_stdout(&self) -> bool {
        self.dir == StdioPath::Stdio
    }

    #[must_use]
    pub fn export(&self) -> GraphExport {
        GraphExport {
            format: self.format,
            provisional: ProvisionalExport::from(self.include_provisional),
        }
    }

    /// Write the export to its directory and say what went.
    async fn run(self, db: &Writer, out: &mut impl Write) -> Result<()> {
        let export = self.export();
        let StdioPath::Path(dir) = self.dir else {
            anyhow::bail!("give a directory to write the graph to");
        };
        let target = dir.clone();
        let summary = db
            .run(move |db| {
                db.read_only(|db| export.write::<std::io::Sink>(db, Destination::Dir(&target)))
            })
            .await?;
        writeln!(
            out,
            "Wrote {} nodes, {} edges, and {} provenance rows as {} to {}.",
            summary.nodes,
            summary.edges,
            summary.provenance,
            summary.format,
            dir.display()
        )?;
        Ok(())
    }
}

#[derive(Subcommand)]
pub enum AddWhat {
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
pub struct SetArgs {
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
pub enum DeleteWhat {
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
pub struct SearchArgs {
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
pub struct PathArgs {
    from: String,
    to: String,
    #[arg(long, default_value_t = Hops::PATH.get())]
    max_hops: u32,
    /// `json` prints the path (nodes, edges, provenance) as JSON
    #[arg(long, value_enum, default_value_t = TextOrJson::Text)]
    format: TextOrJson,
}

#[derive(clap::Args)]
pub struct ExtractArgs {
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
    /// Do not ask before clearing the graph or spending the model calls
    #[arg(long, short = 'y')]
    pub yes: bool,
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

impl GraphAction {
    /// Run one graph action, writing what the user should see to `out`
    /// (stdout for the CLI, the transcript for the terminal session).
    ///
    /// Each database step goes to the workspace writer on its own, never
    /// spanning a model call, so the terminal's other work keeps going during
    /// an extraction; `control` hears about every extracted chunk and can stop it.
    ///
    /// # Errors
    ///
    /// Returns the database, model, or I/O error the command meets.
    pub async fn run(
        self,
        config: &Config,
        db: &Writer,
        confirm: Confirm,
        out: &mut impl Write,
        control: RunControl<'_>,
    ) -> Result<()> {
        match self {
            Self::Search(args) => args.run(config, db, out).await?,
            Self::Path(args) => args.run(config, db, out).await?,
            Self::Extract(args) => {
                args.run(config, db, out, confirm.or_yes(args.yes), control)
                    .await?;
            }
            Self::Status { format } => {
                db.render(out, move |db, out| {
                    format.write(out, &graph_store::status(db)?)
                })
                .await?;
            }
            Self::Revalidate { yes } => Self::revalidate(db, out, yes, confirm).await?,
            Self::Review => {
                db.render(out, |db, out| {
                    graph_store::mark_reviewed(db)?;
                    writeln!(out, "The graph is no longer provisional.")?;
                    Ok(())
                })
                .await?;
            }
            Self::Merges => {
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
            Self::Merge { ids } => {
                db.render(out, move |db, out| {
                    for id in &ids {
                        let m = resolve::decide(db, id, resolve::MergeDecision::Accept, None)?;
                        writeln!(out, "Merged {} into {}.", m.drop.label, m.keep.label)?;
                    }
                    Ok(())
                })
                .await?;
            }
            Self::Reject { ids } => {
                db.render(out, move |db, out| {
                    for id in &ids {
                        let m = resolve::decide(db, id, resolve::MergeDecision::Reject, None)?;
                        writeln!(out, "Kept {} and {} apart.", m.keep.label, m.drop.label)?;
                    }
                    Ok(())
                })
                .await?;
            }
            Self::Add(what) => what.run(db, out).await?,
            Self::Set(args) => args.run(db, out).await?,
            Self::Delete(what) => what.run(db, out).await?,
            Self::Export(args) => args.run(db, out).await?,
        }
        out.flush()?;
        Ok(())
    }

    /// `quack graph revalidate`: with something to drop, say what and ask.
    async fn revalidate(
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
}

impl AddWhat {
    /// Create the node or edge and report it on `out`.
    ///
    /// # Errors
    ///
    /// Returns the database, model, or I/O error the command meets.
    pub(crate) async fn run(self, db: &Writer, out: &mut impl Write) -> Result<()> {
        match self {
            Self::Node {
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
            Self::Edge {
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
}

impl SetArgs {
    /// Change a node and report it on `out`.
    ///
    /// # Errors
    ///
    /// Returns the database, model, or I/O error the command meets.
    pub(crate) async fn run(self, db: &Writer, out: &mut impl Write) -> Result<()> {
        let Self {
            node,
            class,
            label,
            to_class,
            properties,
            unset,
            note,
        } = self;
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
}

impl DeleteWhat {
    /// Delete a node or edge and report it on `out`.
    ///
    /// # Errors
    ///
    /// Returns the database, model, or I/O error the command meets.
    pub(crate) async fn run(self, db: &Writer, out: &mut impl Write) -> Result<()> {
        match self {
            Self::Node { node, class } => {
                db.render(out, move |db, out| {
                    let found = graph_store::find_node(db, &node, class.as_deref())?;
                    let deleted = graph_store::delete_node(db, &found.id)?;
                    writeln!(out, "Deleted {deleted} and its edges.")?;
                    Ok(())
                })
                .await
            }
            Self::Edge { id } => {
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
}

impl SearchArgs {
    /// Search the graph from an entity or a class and print the result on `out`.
    ///
    /// # Errors
    ///
    /// Returns the database, model, or I/O error the command meets.
    pub(crate) async fn run(
        self,
        config: &Config,
        db: &Writer,
        out: &mut impl Write,
    ) -> Result<()> {
        let Self {
            entity,
            class,
            relation,
            hops,
            format,
        } = self;
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
}

impl PathArgs {
    /// Print the shortest path between two entities on `out`.
    ///
    /// # Errors
    ///
    /// Returns the database, model, or I/O error the command meets.
    pub(crate) async fn run(
        self,
        config: &Config,
        db: &Writer,
        out: &mut impl Write,
    ) -> Result<()> {
        let Self {
            from,
            to,
            max_hops,
            format,
        } = self;
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
}

impl ExtractArgs {
    /// Clear the graph for `--reset`, after asking: `false` when nobody
    /// agreed, and nothing was cleared.
    async fn clear(&self, db: &Writer, out: &mut impl Write, confirm: Confirm) -> Result<bool> {
        let keep = if self.all {
            Keep::Nothing
        } else {
            Keep::Asserted
        };
        // Asked before anything goes: the question about model calls comes
        // later, once the graph is already cleared.
        let status = db.run(graph_store::status).await?;
        let question = match keep {
            Keep::Asserted => format!(
                "Clear the graph's {} nodes and {} edges, keeping what people asserted?",
                status.nodes, status.edges
            ),
            Keep::Nothing => format!(
                "Clear all {} nodes and {} edges, including what people asserted?",
                status.nodes, status.edges
            ),
        };
        if !confirm.ask_to_drop(self.yes, out, &question)? {
            writeln!(out, "Nothing cleared.")?;
            return Ok(false);
        }
        db.run(move |db| graph_store::clear(db, keep)).await?;
        writeln!(
            out,
            "{}",
            match keep {
                Keep::Asserted => "Cleared the graph, keeping what people asserted.",
                Keep::Nothing => "Cleared the graph.",
            }
        )?;
        Ok(true)
    }

    /// Extract the documents and tables into the graph, or reset the graph first.
    ///
    /// # Errors
    ///
    /// Returns the database, model, or I/O error the command meets.
    pub(crate) async fn run(
        &self,
        config: &Config,
        db: &Writer,
        out: &mut impl Write,
        confirm: Confirm,
        control: RunControl<'_>,
    ) -> Result<()> {
        let ontology = db.run(ontology_store::current).await?.context(
            "no ontology yet: run `quack ontology init` or `quack ontology propose` first",
        )?;
        let standing = db.run(ontology_store::current_standing).await?;
        if self.reset && !self.clear(db, out, confirm).await? {
            return Ok(());
        }
        let sources = self.source;
        if sources.includes_tables() {
            Self::tables(db, out, &ontology, standing).await?;
        }
        if sources.includes_documents() {
            let sample = self.sample;
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

    /// Every mapping's rows into the graph, one line per table.
    async fn tables(
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
}

#[cfg(test)]
mod tests;
