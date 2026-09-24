//! Open Knowledge Format bundles (OKF v0.1): a directory of Markdown files
//! with YAML front matter, one concept per file, `type` required,
//! Markdown links between files as the graph, an optional `index.md` and
//! `log.md`. A workspace exports as one (tables, ontology, documents,
//! context, log, and one concept per graph node with provenance), and a
//! bundle imports as documents whose front matter and links feed ontology
//! induction. Design doc section 17, issue #36.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::{self, Write as _};
use std::io::{Cursor, Read};
use std::path::Path;

use crate::error::{Error, Result};
use crate::graph::store::EdgeScope;
use crate::graph::{self, Origin, store as graph_store};
use crate::ids::{ClassId, DocumentId, NodeId, RelationId};
use crate::ontology::induction::{Candidate, Proposal};
use crate::ontology::store::Revision;
use crate::ontology::{
    self, Class, Ontology, OntologyVersion, Property, PropertyType, Relation, candidates,
    store as ontology_store,
};
use crate::storage::context;
use crate::storage::workspace::{DocumentInfo, WorkspaceDb};

/// One file of a bundle: a path relative to its root and the text.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BundleFile {
    pub path: String,
    pub content: String,
}

impl BundleFile {
    /// The document file name this concept file is ingested under: its path
    /// with the separators folded, so a bundle's files stay distinct.
    #[must_use]
    pub fn document_name(&self) -> String {
        self.path.replace('/', "__")
    }
}

/// A bundle in memory.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Bundle {
    pub files: Vec<BundleFile>,
}

impl Bundle {
    fn push(&mut self, path: impl Into<String>, content: String) {
        self.files.push(BundleFile {
            path: path.into(),
            content,
        });
    }

    /// The bundle as an uncompressed tar archive.
    ///
    /// # Errors
    ///
    /// Returns an error if writing the archive fails.
    pub fn to_tar(&self) -> Result<Vec<u8>> {
        let mut builder = tar::Builder::new(Vec::new());
        for file in &self.files {
            let mut header = tar::Header::new_gnu();
            header.set_size(u64::try_from(file.content.len()).unwrap_or(u64::MAX));
            header.set_mode(0o644);
            header.set_mtime(0);
            header.set_cksum();
            builder.append_data(&mut header, &file.path, file.content.as_bytes())?;
        }
        builder.into_inner().map_err(Error::Io)
    }

    /// Read a bundle from tar bytes; only `.md` entries count.
    ///
    /// # Errors
    ///
    /// Returns an error if the bytes are not a tar archive or an entry is
    /// not UTF-8.
    pub fn from_tar(bytes: &[u8]) -> Result<Self> {
        let mut archive = tar::Archive::new(Cursor::new(bytes));
        let mut bundle = Self::default();
        for entry in archive.entries()? {
            let mut entry = entry?;
            let path = entry.path()?.to_string_lossy().to_string();
            if !is_markdown(&path) || !entry.header().entry_type().is_file() {
                continue;
            }
            let mut content = String::new();
            entry
                .read_to_string(&mut content)
                .map_err(|e| Error::Ingestion(format!("{path} is not UTF-8: {e}")))?;
            bundle.push(path.trim_start_matches("./").to_owned(), content);
        }
        if bundle.files.is_empty() {
            return Err(Error::Ingestion(String::from(
                "the archive holds no Markdown files",
            )));
        }
        Ok(bundle)
    }

    /// Read every `.md` file under a directory, paths relative to it.
    ///
    /// # Errors
    ///
    /// Returns an error if the directory cannot be read or holds no
    /// Markdown.
    pub fn from_dir(root: &Path) -> Result<Self> {
        let mut bundle = Self::default();
        let mut stack = vec![root.to_path_buf()];
        while let Some(dir) = stack.pop() {
            let mut entries: Vec<_> = std::fs::read_dir(&dir)?.flatten().collect();
            entries.sort_by_key(std::fs::DirEntry::path);
            for entry in entries {
                let path = entry.path();
                if path.is_dir() {
                    stack.push(path);
                    continue;
                }
                if path
                    .extension()
                    .is_some_and(|e| e.eq_ignore_ascii_case("md"))
                {
                    let relative = path
                        .strip_prefix(root)
                        .map_err(|e| Error::Ingestion(e.to_string()))?
                        .to_string_lossy()
                        .replace('\\', "/");
                    bundle.push(relative, std::fs::read_to_string(&path)?);
                }
            }
        }
        if bundle.files.is_empty() {
            return Err(Error::Ingestion(format!(
                "{} holds no Markdown files",
                root.display()
            )));
        }
        bundle.files.sort_by(|a, b| a.path.cmp(&b.path));
        Ok(bundle)
    }

    /// Write the files under `root`, creating directories.
    ///
    /// # Errors
    ///
    /// Returns an error if a write fails.
    pub fn write_to(&self, root: &Path) -> Result<()> {
        for file in &self.files {
            let target = root.join(&file.path);
            if let Some(parent) = target.parent() {
                std::fs::create_dir_all(parent)?;
            }
            std::fs::write(&target, &file.content)?;
        }
        Ok(())
    }

    /// The `index.md` body, when the bundle has one.
    #[must_use]
    pub fn index(&self) -> Option<&BundleFile> {
        self.files.iter().find(|f| f.path == INDEX)
    }

    /// Every file but `index.md` and `log.md`: the concepts.
    pub fn concepts(&self) -> impl Iterator<Item = &BundleFile> {
        self.files
            .iter()
            .filter(|f| f.path != INDEX && f.path != LOG)
    }

    /// The concept files worth ingesting as documents: everything a foreign
    /// bundle holds, but not the stubs quack's own export writes (table
    /// schemas, document metadata, ontology and entity files), which carry
    /// no text a chunk should hold (issue #53).
    pub fn documents(&self) -> impl Iterator<Item = &BundleFile> {
        self.concepts()
            .filter(|f| parse_front_matter(&f.content).0.get("generator") != Some(GENERATOR))
    }

    /// Restore the bundle into a workspace: when the workspace has no
    /// ontology, the snapshot quack's export carries is saved under
    /// `revision`; then the bundle's types and links go to the review
    /// queue as candidates against the ontology now in force.
    ///
    /// # Errors
    ///
    /// Returns an error when the snapshot does not parse or a write fails.
    pub fn restore_into(&self, db: &WorkspaceDb, revision: Revision<'_>) -> Result<RestoreReport> {
        let mut current = ontology_store::current(db)?;
        let mut restored = None;
        if current.is_none()
            && let Some(snapshot) = self.ontology()?
        {
            let saved = ontology_store::save(db, &snapshot, revision)?;
            restored = Some(saved.saved_version()?);
            current = Some(saved);
        }
        let candidates = propose(self, current.as_ref());
        if !candidates.is_empty() {
            candidates::store_run(db, &candidates)?;
        }
        Ok(RestoreReport {
            restored,
            candidates: candidates.len(),
        })
    }

    /// The exact ontology snapshot quack's export carries, if any.
    ///
    /// # Errors
    ///
    /// Returns an error when the snapshot is present but does not parse.
    pub fn ontology(&self) -> Result<Option<Ontology>> {
        let Some(file) = self.files.iter().find(|f| f.path == ONTOLOGY_SNAPSHOT) else {
            return Ok(None);
        };
        let (_, body) = parse_front_matter(&file.content);
        let json = body
            .split("```json")
            .nth(1)
            .and_then(|rest| rest.split("```").next())
            .ok_or_else(|| Error::Ingestion(format!("{ONTOLOGY_SNAPSHOT} holds no JSON")))?;
        Ontology::from_json(json.trim()).map(Some)
    }
}

/// What restoring a bundle did.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RestoreReport {
    /// The version the bundle's ontology was saved as, when the workspace
    /// had none and the bundle carried one.
    pub restored: Option<OntologyVersion>,
    /// Candidates the bundle's types and links queued for review.
    pub candidates: usize,
}

/// The `generator` front-matter value on every stub quack exports.
pub const GENERATOR: &str = "quack";
/// The bundle's entry page, which the workspace context becomes.
pub const INDEX: &str = "index.md";
/// The bundle's history page.
pub const LOG: &str = "log.md";
/// Where the export keeps the ontology as JSON, inside a Markdown file so
/// bundle readers (which take `.md` only) carry it.
pub const ONTOLOGY_SNAPSHOT: &str = "ontology/ontology.md";

/// A concept file's front matter: the scalar keys in file order, and
/// `tags`. It reads with [`parse_front_matter`] and writes with `Display`.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct FrontMatter {
    pub fields: Vec<(String, String)>,
    pub tags: Vec<String>,
}

impl FrontMatter {
    /// The value of `key`; the last one when the file repeats it.
    #[must_use]
    pub fn get(&self, key: &str) -> Option<&str> {
        self.fields
            .iter()
            .rev()
            .find(|(k, _)| k == key)
            .map(|(_, v)| v.as_str())
    }

    /// Front matter of one `type`, the key every concept file carries.
    fn of_type(concept: impl fmt::Display) -> Self {
        Self::default().field("type", concept.to_string())
    }

    /// Front matter for one of quack's own stubs: its `type` and the
    /// `generator` that marks it as quack's.
    fn generated(concept: ConceptType) -> Self {
        Self::of_type(concept).field("generator", GENERATOR)
    }

    /// Add `key: value`; an empty value is left out when written.
    fn field(mut self, key: &str, value: impl Into<String>) -> Self {
        self.fields.push((key.to_owned(), value.into()));
        self
    }

    fn tag(mut self, tag: &str) -> Self {
        self.tags.push(tag.to_owned());
        self
    }

    fn yaml_scalar(value: &str) -> String {
        if value
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, ' ' | '_' | '-' | '.' | '/'))
            && !value.is_empty()
            && !value.starts_with(['-', ' '])
        {
            value.to_owned()
        } else {
            format!("\"{}\"", value.replace('\\', "\\\\").replace('"', "\\\""))
        }
    }
}

/// The `---` block a concept file starts with.
impl fmt::Display for FrontMatter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("---\n")?;
        for (key, value) in &self.fields {
            if !value.is_empty() {
                writeln!(f, "{key}: {}", Self::yaml_scalar(value))?;
            }
        }
        if !self.tags.is_empty() {
            let tags: Vec<String> = self.tags.iter().map(|t| Self::yaml_scalar(t)).collect();
            writeln!(f, "tags: [{}]", tags.join(", "))?;
        }
        f.write_str("---\n")
    }
}

/// Split a file into its front matter and body. The front matter is the
/// flat YAML OKF uses: `key: value` lines and a `tags` list (inline
/// `[a, b]` or `- a` items); anything else is kept as text.
#[must_use]
pub fn parse_front_matter(content: &str) -> (FrontMatter, &str) {
    let mut front = FrontMatter::default();
    let Some(rest) = content
        .strip_prefix("---\n")
        .or_else(|| content.strip_prefix("---\r\n"))
    else {
        return (front, content);
    };
    let Some(end) = rest.find("\n---") else {
        return (front, content);
    };
    let block = rest.get(..end).unwrap_or_default();
    let body = rest
        .get(end.saturating_add(4)..)
        .unwrap_or_default()
        .trim_start_matches(['\r', '\n']);
    let mut in_tags = false;
    for line in block.lines() {
        if in_tags && let Some(item) = line.trim().strip_prefix("- ") {
            front.tags.push(unquote(item));
            continue;
        }
        in_tags = false;
        let Some((key, value)) = line.split_once(':') else {
            continue;
        };
        let key = key.trim();
        let value = value.trim();
        if key.is_empty() || key.starts_with('#') {
            continue;
        }
        if key == "tags" {
            if value.is_empty() {
                in_tags = true;
            } else {
                front.tags = value
                    .trim_start_matches('[')
                    .trim_end_matches(']')
                    .split(',')
                    .map(unquote)
                    .filter(|t| !t.is_empty())
                    .collect();
            }
            continue;
        }
        front.fields.push((key.to_owned(), unquote(value)));
    }
    (front, body)
}

fn is_markdown(path: &str) -> bool {
    Path::new(path)
        .extension()
        .is_some_and(|e| e.eq_ignore_ascii_case("md"))
}

fn unquote(value: &str) -> String {
    let v = value.trim();
    v.strip_prefix('"')
        .and_then(|s| s.strip_suffix('"'))
        .or_else(|| v.strip_prefix('\'').and_then(|s| s.strip_suffix('\'')))
        .unwrap_or(v)
        .to_owned()
}

/// Relative Markdown link targets in a body (`[text](path.md)`), with
/// `..` segments resolved against `from`.
#[must_use]
pub fn links(from: &str, body: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut rest = body;
    while let Some(start) = rest.find("](") {
        let after = rest.get(start.saturating_add(2)..).unwrap_or_default();
        let Some(end) = after.find(')') else {
            break;
        };
        let target = after.get(..end).unwrap_or_default().trim();
        rest = after.get(end..).unwrap_or_default();
        if target.contains("://") || !is_markdown(target) || target.starts_with('#') {
            continue;
        }
        out.push(resolve(from, target));
    }
    out
}

fn resolve(from: &str, target: &str) -> String {
    let mut parts: Vec<&str> = from.rsplit_once('/').map_or_else(Vec::new, |(dir, _)| {
        dir.split('/').filter(|p| !p.is_empty()).collect()
    });
    for segment in target.split('/') {
        match segment {
            "." | "" => {}
            ".." => {
                parts.pop();
            }
            other => parts.push(other),
        }
    }
    parts.join("/")
}

/// A file-system-safe slug: lowercase, `[a-z0-9]` runs joined by `-`.
#[must_use]
pub fn slug(text: &str) -> String {
    let mut out = String::new();
    let mut dash = false;
    for c in text.chars() {
        if c.is_ascii_alphanumeric() {
            out.push(c.to_ascii_lowercase());
            dash = false;
        } else if !dash && !out.is_empty() {
            out.push('-');
            dash = true;
        }
    }
    let trimmed = out.trim_end_matches('-').to_owned();
    if trimmed.is_empty() {
        String::from("untitled")
    } else {
        trimmed
    }
}

/// The workspace as a bundle: `index.md` from the context, one file per
/// table (schema and sample rows), class, relation, document (metadata),
/// and graph node, the ontology as JSON, and `log.md` from the ontology
/// versions. A one-way knowledge export: the audit detail stays in the
/// workspace, table data and document text are not in it, and a
/// re-import restores the context and the ontology and proposes the rest.
///
/// # Errors
///
/// Returns an error if a read fails.
pub fn export(db: &WorkspaceDb, workspace_name: &str) -> Result<Bundle> {
    let ontology = ontology_store::current(db)?;
    let context_text = context::current(db)?.map(|c| c.content).unwrap_or_default();
    let mut index = FrontMatter::of_type(ConceptType::Index)
        .field("title", workspace_name)
        .to_string();
    writeln!(index, "# {workspace_name}\n")?;
    if !context_text.trim().is_empty() {
        index.push_str(context_text.trim());
        index.push_str("\n\n");
    }
    index.push_str("## Contents\n\n");
    let mut exporter = Exporter {
        db,
        bundle: Bundle::default(),
        index,
    };

    exporter.tables(ontology.as_ref())?;
    if let Some(ontology) = &ontology {
        exporter.ontology(ontology)?;
    }
    let documents = db.list_documents()?;
    exporter.documents(&documents)?;
    exporter.entities(&documents)?;

    let Exporter {
        mut bundle, index, ..
    } = exporter;
    let mut log = FrontMatter::of_type(ConceptType::Log).to_string();
    log.push_str("# Log\n\n## Ontology versions\n\n");
    for version in ontology_store::versions(db, 100)? {
        writeln!(
            log,
            "- {} version {}{}{}",
            version.created_at,
            version.version,
            version
                .author
                .as_deref()
                .map_or(String::new(), |a| format!(" by {a}")),
            version
                .note
                .as_deref()
                .map_or(String::new(), |n| format!(": {n}"))
        )?;
    }
    bundle.push(LOG, log);
    bundle.files.insert(
        0,
        BundleFile {
            path: String::from(INDEX),
            content: index,
        },
    );
    Ok(bundle)
}

/// A bundle being written: the files so far, and the index listing them.
struct Exporter<'a> {
    db: &'a WorkspaceDb,
    bundle: Bundle,
    index: String,
}

impl Exporter<'_> {
    /// Add a file, listing it in the index under `title`.
    fn listed(&mut self, path: String, title: &str, text: String) -> Result<()> {
        writeln!(self.index, "- [{title}]({path})")?;
        self.bundle.push(path, text);
        Ok(())
    }

    fn tables(&mut self, ontology: Option<&Ontology>) -> Result<()> {
        for table in &self.db.list_tables()? {
            let described = self.db.describe_table(table)?;
            let mapping = ontology.and_then(|o| o.mapping_for_table(table));
            let mut text = FrontMatter::generated(ConceptType::Table)
                .field("title", table.clone())
                .field(
                    "description",
                    format!(
                        "{} rows, {} columns",
                        described.row_count,
                        described.columns.len()
                    ),
                )
                .to_string();
            writeln!(
                text,
                "# {table}\n\n## Schema\n\n| column | type |\n|---|---|"
            )?;
            for column in &described.columns {
                writeln!(text, "| {} | {} |", column.name, column.column_type)?;
            }
            text.push_str("\n## Sample rows\n\n");
            if described.sample_rows.columns.is_empty() {
                text.push_str("(no rows)\n");
            } else {
                let mut rows = Vec::new();
                described.sample_rows.write_markdown(&mut rows)?;
                text.push_str(&String::from_utf8_lossy(&rows));
            }
            if let (Some(mapping), Some(ontology)) = (mapping, ontology) {
                writeln!(
                    text,
                    "\n## Ontology\n\nRows are [{}](../ontology/classes/{}.md) keyed by `{}`.",
                    mapping.class,
                    slug(mapping.class.as_str()),
                    mapping.key
                )?;
                for relation in &mapping.relations {
                    write!(
                        text,
                        "- `{}` [{}](../ontology/relations/{}.md) [{}](../ontology/classes/{}.md)",
                        relation.column,
                        relation.relation,
                        slug(relation.relation.as_str()),
                        relation.target_class,
                        slug(relation.target_class.as_str())
                    )?;
                    if let Some(other) = ontology
                        .mappings
                        .iter()
                        .find(|m| m.class == relation.target_class)
                    {
                        write!(
                            text,
                            " (see [{}](./{}.md))",
                            other.table,
                            slug(&other.table)
                        )?;
                    }
                    text.push('\n');
                }
            }
            self.listed(format!("tables/{}.md", slug(table)), table, text)?;
        }
        Ok(())
    }

    fn documents(&mut self, documents: &[DocumentInfo]) -> Result<()> {
        for document in documents {
            let mut text = FrontMatter::generated(ConceptType::Document)
                .field("title", document.display_name())
                .field("resource", document.filename.clone())
                .field("timestamp", document.ingested_at.clone())
                .to_string();
            writeln!(
                text,
                "# {}\n\n- status: {}\n- source: {}\n- chunks: {}\n- pinned: {}",
                document.display_name(),
                document.status,
                document.source,
                document.chunk_count.unwrap_or(0),
                document.pinned
            )?;
            if let Some(tables) = &document.tables {
                for table in tables {
                    writeln!(text, "- table: [{table}](../tables/{}.md)", slug(table))?;
                }
            }
            self.listed(
                format!("documents/{}.md", slug(&document.filename)),
                document.display_name(),
                text,
            )?;
        }
        Ok(())
    }

    fn ontology(&mut self, ontology: &Ontology) -> Result<()> {
        for class in &ontology.classes {
            let mut text = FrontMatter::generated(ConceptType::Class)
                .field(
                    "title",
                    class.label.clone().unwrap_or_else(|| class.id.to_string()),
                )
                .field("description", class.description.clone().unwrap_or_default())
                .to_string();
            writeln!(text, "# {}\n", class.id)?;
            if class.parent != ontology::ROOT_CLASS {
                writeln!(
                    text,
                    "- parent: [{}](./{}.md)",
                    class.parent,
                    slug(class.parent.as_str())
                )?;
            }
            if let Some(key) = &class.key {
                writeln!(text, "- key: [{key}](../properties/{}.md)", slug(key))?;
            }
            for property in &class.properties {
                writeln!(
                    text,
                    "- property: [{property}](../properties/{}.md)",
                    slug(property)
                )?;
            }
            for relation in ontology
                .relations
                .iter()
                .filter(|r| r.domain == class.id || r.range == class.id)
            {
                writeln!(
                    text,
                    "- relation: [{}](../relations/{}.md)",
                    relation.id,
                    slug(relation.id.as_str())
                )?;
            }
            self.listed(
                format!("ontology/classes/{}.md", slug(class.id.as_str())),
                class.id.as_str(),
                text,
            )?;
        }
        self.relations(ontology)?;
        self.properties(ontology)?;
        let version = ontology.saved_version()?;
        let mut snapshot = FrontMatter::generated(ConceptType::Ontology)
            .field("version", version.to_string())
            .to_string();
        writeln!(
            snapshot,
            "# Ontology version {}\n\nThe exact snapshot, as `quack ontology export` writes it; `quack ingest DIR` restores it into a workspace that has no ontology yet.\n\n```json\n{}\n```",
            version,
            ontology.to_json()?
        )?;
        self.bundle.push(ONTOLOGY_SNAPSHOT, snapshot);
        Ok(())
    }

    fn relations(&mut self, ontology: &Ontology) -> Result<()> {
        for relation in &ontology.relations {
            let mut text = FrontMatter::generated(ConceptType::Relation)
                .field(
                    "title",
                    relation
                        .label
                        .clone()
                        .unwrap_or_else(|| relation.id.to_string()),
                )
                .field(
                    "description",
                    relation.description.clone().unwrap_or_default(),
                )
                .to_string();
            writeln!(
                text,
                "# {}\n\n- domain: [{}](../classes/{}.md)\n- range: [{}](../classes/{}.md)",
                relation.id,
                relation.domain,
                slug(relation.domain.as_str()),
                relation.range,
                slug(relation.range.as_str())
            )?;
            self.bundle.push(
                format!("ontology/relations/{}.md", slug(relation.id.as_str())),
                text,
            );
        }
        Ok(())
    }

    fn properties(&mut self, ontology: &Ontology) -> Result<()> {
        for property in &ontology.properties {
            let mut text = FrontMatter::generated(ConceptType::Property)
                .field(
                    "title",
                    property
                        .label
                        .clone()
                        .unwrap_or_else(|| property.id.clone()),
                )
                .field("description", format!("{} property", property.kind))
                .to_string();
            writeln!(text, "# {}\n\n- type: {}", property.id, property.kind)?;
            if !property.values.is_empty() {
                writeln!(text, "- values: {}", property.values.join(", "))?;
            }
            self.bundle.push(
                format!("ontology/properties/{}.md", slug(&property.id)),
                text,
            );
        }
        Ok(())
    }

    fn entities(&mut self, documents: &[DocumentInfo]) -> Result<()> {
        let ids = graph_store::all_node_ids(self.db)?;
        if ids.is_empty() {
            return Ok(());
        }
        let nodes = graph_store::nodes(self.db, &ids)?;
        let edges = graph_store::edges(self.db, &ids, EdgeScope::Among)?;
        let subjects: Vec<String> = ids
            .iter()
            .map(NodeId::to_string)
            .chain(edges.iter().map(|e| e.id.to_string()))
            .collect();
        let provenance = graph_store::provenance_of(self.db, &subjects)?;
        // Every path is fixed before any file is written, so a link to a
        // node whose label collided points at the suffixed file it got.
        let mut seen: BTreeSet<String> = BTreeSet::new();
        let mut paths: BTreeMap<&str, String> = BTreeMap::new();
        for node in &nodes {
            let mut path = format!(
                "entities/{}/{}.md",
                slug(node.class_id.as_str()),
                slug(&node.label)
            );
            if !seen.insert(path.clone()) {
                path = format!(
                    "entities/{}/{}-{}.md",
                    slug(node.class_id.as_str()),
                    slug(&node.label),
                    node.id.as_str().chars().rev().take(6).collect::<String>()
                );
                seen.insert(path.clone());
            }
            paths.insert(node.id.as_str(), path);
        }
        let export = EntityExport {
            edges: &edges,
            by_id: nodes.iter().map(|n| (n.id.as_str(), n)).collect(),
            provenance: &provenance,
            documents,
        };
        for node in &nodes {
            let Some(path) = paths.get(node.id.as_str()) else {
                continue;
            };
            let text = export.file(node, &paths)?;
            self.listed(path.clone(), &node.label, text)?;
        }
        Ok(())
    }
}

/// What every entity file draws on: the graph's edges, nodes by id, the
/// provenance of both, and the documents it names.
struct EntityExport<'a> {
    edges: &'a [graph::Edge],
    by_id: BTreeMap<&'a str, &'a graph::Node>,
    provenance: &'a [graph::Provenance],
    documents: &'a [DocumentInfo],
}

impl EntityExport<'_> {
    /// The file name of a document by id, else the id.
    fn filename_of(&self, document_id: &DocumentId) -> String {
        self.documents
            .iter()
            .find(|d| &d.id == document_id)
            .map_or_else(|| document_id.to_string(), |d| d.filename.clone())
    }

    /// One node as a concept file: front matter, properties, links per
    /// edge, and provenance. `paths` holds every node's file.
    fn file(&self, node: &graph::Node, paths: &BTreeMap<&str, String>) -> Result<String> {
        let mut front = FrontMatter::of_type(&node.class_id)
            .field("generator", GENERATOR)
            .field("id", node.id.to_string())
            .field("title", node.label.clone());
        if node.provisional {
            front = front.tag("provisional");
        }
        let mut text = front.to_string();
        writeln!(text, "# {}\n", node.label)?;
        writeln!(
            text,
            "Class: [{}](../../ontology/classes/{}.md)\n",
            node.class_id,
            slug(node.class_id.as_str())
        )?;
        if !node.properties.is_empty() {
            text.push_str("## Properties\n\n");
            for (key, value) in node.properties.iter() {
                let shown = match value {
                    serde_json::Value::String(s) => s.clone(),
                    other => other.to_string(),
                };
                writeln!(text, "- {key}: {shown}")?;
            }
            text.push('\n');
        }
        // Outbound edges only: a link in both files would read as two relations.
        let mine: Vec<&graph::Edge> = self
            .edges
            .iter()
            .filter(|e| e.source_node_id == node.id)
            .collect();
        if !mine.is_empty() {
            text.push_str("## Links\n\n");
            for edge in &mine {
                let other = edge.target_node_id.as_str();
                if let (Some(target), Some(path)) = (self.by_id.get(other), paths.get(other)) {
                    writeln!(
                        text,
                        "- {}: [{}](../../{path})",
                        edge.relation_id, target.label,
                    )?;
                }
            }
            text.push('\n');
        }
        let sources: Vec<String> = self
            .provenance
            .iter()
            .filter(|p| p.subject_id == node.id.as_str())
            .map(|p| match &p.origin {
                Origin::Row {
                    table_name,
                    row_key,
                } => format!(
                    "- table [{table_name}](../../tables/{}.md) row `{row_key}`",
                    slug(table_name),
                ),
                Origin::Chunk {
                    document_id: Some(document),
                    chunk_id,
                } => format!(
                    "- document [{}](../../documents/{}.md) chunk `{chunk_id}` (confidence {:.2})",
                    self.filename_of(document),
                    slug(&self.filename_of(document)),
                    p.confidence
                ),
                Origin::Chunk {
                    document_id: None, ..
                } => String::from("- unknown"),
            })
            .collect();
        if !sources.is_empty() {
            text.push_str("## Provenance\n\n");
            text.push_str(&sources.join("\n"));
            text.push('\n');
        }
        Ok(text)
    }
}

/// Proposals from a bundle's front matter and links: every `type` a
/// class (`snake_case`), every link between two typed concepts a relation
/// from the source type to the target type, and `resource` a document
/// property. Types and relations the ontology already has are skipped.
#[must_use]
pub fn propose(bundle: &Bundle, current: Option<&Ontology>) -> Vec<Candidate> {
    let mut type_of: BTreeMap<&str, String> = BTreeMap::new();
    let mut examples: BTreeMap<String, Vec<String>> = BTreeMap::new();
    let mut bodies: Vec<(&str, &str)> = Vec::new();
    let mut resource_seen = false;
    for file in bundle.concepts() {
        let (front, body) = parse_front_matter(&file.content);
        // quack's own structural stubs (tables, documents, ontology files)
        // describe the workspace, not knowledge: only its entity files
        // carry types worth proposing.
        if front.get("generator") == Some(GENERATOR) && !file.path.starts_with("entities/") {
            continue;
        }
        let Some(kind) = front.get("type").map(type_id) else {
            continue;
        };
        if front.get("resource").is_some() {
            resource_seen = true;
        }
        if kind.is_empty() || ConceptType::is_structural(&kind) {
            continue;
        }
        examples
            .entry(kind.clone())
            .or_default()
            .push(front.get("title").unwrap_or(&file.path).to_owned());
        type_of.insert(file.path.as_str(), kind);
        bodies.push((file.path.as_str(), body));
    }
    let mut candidates = Vec::new();
    for (kind, titles) in &examples {
        if current.map_or(kind == ontology::ROOT_CLASS, |o| o.defines_class(kind)) {
            continue;
        }
        candidates.push(Candidate {
            proposal: Proposal::Class(Class {
                id: ClassId::from(kind.clone()),
                parent: ClassId::from(String::from(ontology::ROOT_CLASS)),
                label: None,
                description: None,
                key: None,
                properties: Vec::new(),
            }),
            evidence: serde_json::json!({
                "source": "okf",
                "files": titles.len(),
                "examples": titles.iter().take(5).collect::<Vec<_>>(),
            }),
            confidence: (f64::from(u32::try_from(titles.len()).unwrap_or(u32::MAX)) / 5.0).min(1.0),
            low_support: false,
        });
    }
    propose_links(&type_of, &bodies, current, &mut candidates);
    if resource_seen
        && current
            .is_some_and(|o| o.class("document").is_some() && o.property("resource").is_none())
    {
        candidates.push(Candidate {
            proposal: Proposal::Property {
                class: String::from("document"),
                property: Property {
                    id: String::from("resource"),
                    label: None,
                    kind: PropertyType::String,
                    values: Vec::new(),
                },
            },
            evidence: serde_json::json!({ "source": "okf", "files": 1, "examples": ["resource"] }),
            confidence: 0.6,
            low_support: false,
        });
    }
    candidates
}

/// Relation candidates from links between typed concepts. A link on a
/// `- name: [..](..)` line (the shape quack's own entity files write)
/// carries its relation id; a bare link is `<source>_links_<target>`.
/// Nothing is proposed when the ontology already has the id, or any
/// relation between the two classes or their ancestors (issue #70).
fn propose_links(
    type_of: &BTreeMap<&str, String>,
    bodies: &[(&str, &str)],
    current: Option<&Ontology>,
    candidates: &mut Vec<Candidate>,
) {
    let mut link_counts: BTreeMap<LinkKey, LinkEvidence> = BTreeMap::new();
    for (path, body) in bodies {
        let Some(source) = type_of.get(path) else {
            continue;
        };
        for line in body.lines() {
            let named = labelled_link(line);
            for target in links(path, line) {
                let Some(target_type) = type_of.get(target.as_str()) else {
                    continue;
                };
                if target_type == source {
                    continue;
                }
                let id =
                    named.map_or_else(|| format!("{source}_links_{target_type}"), str::to_owned);
                link_counts
                    .entry(LinkKey {
                        source: source.clone(),
                        target: target_type.clone(),
                        relation: id,
                    })
                    .or_default()
                    .add(format!("{path} -> {target}"));
            }
        }
    }
    for (
        LinkKey {
            source,
            target,
            relation: id,
        },
        LinkEvidence { count, samples },
    ) in &link_counts
    {
        if current.is_some_and(|o| {
            o.relation(id).is_some()
                || o.relations.iter().any(|r| {
                    o.is_subclass_of(source, r.domain.as_str())
                        && o.is_subclass_of(target, r.range.as_str())
                })
        }) {
            continue;
        }
        candidates.push(Candidate {
            proposal: Proposal::Relation(Relation {
                id: RelationId::from(id.clone()),
                label: None,
                description: Some(format!(
                    "OKF links from {source} concepts to {target} concepts"
                )),
                domain: ClassId::from(source.clone()),
                range: ClassId::from(target.clone()),
            }),
            evidence: serde_json::json!({
                "source": "okf",
                "files": count,
                "examples": samples,
            }),
            confidence: (f64::from(*count) / 5.0).min(1.0),
            low_support: false,
        });
    }
}

/// A relation links suggest: the source and target concept types and the
/// relation id.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct LinkKey {
    source: String,
    target: String,
    relation: String,
}

/// How many links suggest a relation, and the first few as examples.
#[derive(Debug, Clone, Default)]
struct LinkEvidence {
    count: u32,
    samples: Vec<String>,
}

impl LinkEvidence {
    const SAMPLES: usize = 5;

    fn add(&mut self, sample: String) {
        self.count = self.count.saturating_add(1);
        if self.samples.len() < Self::SAMPLES {
            self.samples.push(sample);
        }
    }
}

/// The relation id a `- <id>: [label](path)` line names, when the id is
/// an identifier (`snake_case`, as the ontology requires).
fn labelled_link(line: &str) -> Option<&str> {
    let (label, rest) = line.trim_start().strip_prefix("- ")?.split_once(':')?;
    if label.is_empty()
        || !label
            .chars()
            .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_')
        || !rest.trim_start().starts_with('[')
    {
        return None;
    }
    Some(label)
}

/// The `type`s a quack export writes for its own structure; they describe
/// the bundle, not the domain, so they never become class candidates.
/// Entity files carry their class id instead.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConceptType {
    Index,
    Log,
    Table,
    Document,
    Ontology,
    Class,
    Relation,
    Property,
}

text_enum!(ConceptType, "concept type", {
    Index => "index",
    Log => "log",
    Table => "DuckDB Table",
    Document => "document",
    Ontology => "ontology",
    Class => "class",
    Relation => "relation",
    Property => "property",
});

impl ConceptType {
    /// Whether `id`, an imported `type` through [`type_id`], is one of these.
    fn is_structural(id: &str) -> bool {
        Self::ALL.iter().any(|t| type_id(t.as_str()) == id)
    }
}

/// An OKF `type` as an ontology id: lowercase, words joined by `_`,
/// a trailing `s` dropped from the last word.
fn type_id(raw: &str) -> String {
    let mut words: Vec<String> = raw
        .split(|c: char| !c.is_ascii_alphanumeric())
        .filter(|w| !w.is_empty())
        .map(str::to_ascii_lowercase)
        .collect();
    if let Some(last) = words.last_mut()
        && last.len() > 3
        && last.ends_with('s')
        && !last.ends_with("ss")
    {
        last.pop();
    }
    let joined = words.join("_");
    if joined.chars().next().is_some_and(|c| c.is_ascii_digit()) {
        format!("t_{joined}")
    } else {
        joined
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{Properties, Standing};
    use crate::ontology::induction::ItemKind;
    use crate::ontology::store::Revision;
    use crate::storage::audit;

    #[test]
    fn front_matter_parses_scalars_and_both_tag_forms() {
        let (front, body) = parse_front_matter(
            "---\ntype: DuckDB Table\ntitle: \"Sales, Q1\"\ntags: [a, 'b c']\n---\n\n# Body\n",
        );
        assert_eq!(front.get("type"), Some("DuckDB Table"));
        assert_eq!(front.get("title"), Some("Sales, Q1"));
        assert_eq!(front.tags, ["a", "b c"]);
        assert_eq!(body, "# Body\n");
        let (front, body) =
            parse_front_matter("---\ntype: x\ntags:\n  - one\n  - two\nresource: r\n---\ntext");
        assert_eq!(front.tags, ["one", "two"]);
        assert_eq!(front.get("resource"), Some("r"));
        assert_eq!(body, "text");
        let (front, body) = parse_front_matter("no front matter");
        assert!(front.fields.is_empty());
        assert_eq!(body, "no front matter");
    }

    #[test]
    fn links_resolve_relative_paths_and_skip_urls() {
        let found = links(
            "entities/vendor/orgenics.md",
            "see [Kenya](../country/kenya.md) and [site](https://x.y/z.md) and [self](#top) and [t](../../tables/shipments.md)",
        );
        assert_eq!(found, ["entities/country/kenya.md", "tables/shipments.md"]);
    }

    #[test]
    fn exported_structural_types_are_never_domain_types() {
        for &concept in ConceptType::ALL {
            assert!(ConceptType::is_structural(&type_id(&concept.to_string())));
        }
        assert!(ConceptType::is_structural("duckdb_table"));
        assert!(!ConceptType::is_structural("organization"));
        assert!(!ConceptType::is_structural(""));
    }

    #[test]
    fn slugs_and_type_ids_normalize() {
        assert_eq!(slug("Orgenics Ltd."), "orgenics-ltd");
        assert_eq!(slug("  "), "untitled");
        assert_eq!(type_id("DuckDB Table"), "duckdb_table");
        assert_eq!(type_id("Organizations"), "organization");
        assert_eq!(type_id("class"), "class");
        assert_eq!(type_id("3d models"), "t_3d_model");
        assert_eq!(
            BundleFile {
                path: String::from("entities/vendor/x.md"),
                content: String::new(),
            }
            .document_name(),
            "entities__vendor__x.md"
        );
    }

    #[test]
    fn tar_round_trips_and_ignores_non_markdown() {
        let mut bundle = Bundle::default();
        bundle.push("index.md", String::from("---\ntype: index\n---\nhi"));
        bundle.push("a/b.md", String::from("---\ntype: thing\n---\nbody"));
        let bytes = bundle.to_tar().unwrap_or_default();
        let back = Bundle::from_tar(&bytes).unwrap_or_default();
        assert_eq!(back, bundle);
        assert!(Bundle::from_tar(b"not a tar").is_err());
    }

    /// The export is one-way knowledge (issue #53): no audit detail, the
    /// ontology as an exact snapshot, entity files with ids and links
    /// that resolve even when two labels share a slug, and quack's stubs
    /// neither ingested nor proposed on re-import.
    #[test]
    fn export_carries_ids_resolved_links_the_ontology_and_no_audit() {
        let db = WorkspaceDb::open_in_memory(4).unwrap_or_else(|e| unreachable_db(&e.to_string()));
        let mut ontology = Ontology::builtin_default();
        ontology.classes.push(ontology::Class {
            id: ClassId::from("harbour"),
            parent: ClassId::from(String::from(ontology::ROOT_CLASS)),
            label: None,
            description: None,
            key: None,
            properties: Vec::new(),
        });
        ontology.relations.push(ontology::Relation {
            id: RelationId::from("near"),
            label: None,
            description: None,
            domain: ClassId::from("harbour"),
            range: ClassId::from("harbour"),
        });
        let saved = ontology_store::save(&db, &ontology, Revision::reviewed(None, None))
            .unwrap_or_else(|e| unreachable_db(&e.to_string()));
        audit::AuditDetail {
            id: crate::ids::AuditId::from("a1"),
            user_id: Some(crate::ids::UserId::from("u")),
            action: String::from("sql"),
            detail: serde_json::json!({"sql": "SELECT secret"}),
        }
        .write(&db)
        .unwrap_or_else(|e| unreachable_db(&e.to_string()));
        let node = |label: &str| graph_store::NewNode {
            label: label.to_owned(),
            class_id: ClassId::from("harbour"),
            properties: Properties::default(),
            standing: Standing::Reviewed,
        };
        // Two labels, one slug: the second file gets a suffix and the
        // link from the first must point at it.
        let a = graph_store::upsert_node(&db, &node("Kenya-Coast"))
            .unwrap_or_else(|e| unreachable_db(&e.to_string()));
        let b = graph_store::upsert_node(&db, &node("Kenya Coast"))
            .unwrap_or_else(|e| unreachable_db(&e.to_string()));
        for id in [&a, &b] {
            graph_store::add_provenance(&db, id, &graph_store::Source::row("t", "k"))
                .unwrap_or_else(|e| unreachable_db(&e.to_string()));
        }
        let edge = graph_store::upsert_edge(
            &db,
            &a,
            &b,
            "near",
            &Properties::default(),
            Standing::Reviewed,
        )
        .unwrap_or_else(|e| unreachable_db(&e.to_string()));
        graph_store::add_provenance(&db, &edge, &graph_store::Source::row("t", "k"))
            .unwrap_or_else(|e| unreachable_db(&e.to_string()));

        let bundle = export(&db, "ws").unwrap_or_else(|e| unreachable_db(&e.to_string()));
        let log = bundle
            .files
            .iter()
            .find(|f| f.path == "log.md")
            .map(|f| f.content.clone())
            .unwrap_or_default();
        assert!(
            !log.contains("secret") && !log.contains("Activity"),
            "{log}"
        );
        let snapshot = bundle
            .ontology()
            .unwrap_or_else(|e| unreachable_db(&e.to_string()));
        assert_eq!(snapshot.map(|o| o.classes.len()), Some(saved.classes.len()));
        let entities: Vec<&BundleFile> = bundle
            .files
            .iter()
            .filter(|f| f.path.starts_with("entities/"))
            .collect();
        assert_eq!(
            entities.len(),
            2,
            "{:?}",
            bundle.files.iter().map(|f| &f.path).collect::<Vec<_>>()
        );
        let paths: Vec<&str> = entities.iter().map(|f| f.path.as_str()).collect();
        for file in &entities {
            assert!(file.content.contains("\nid: "), "{}", file.content);
            for link in links(&file.path, &file.content) {
                // Provenance links name table and document stubs, which a
                // real export writes alongside; entity links must resolve.
                assert!(
                    link.starts_with("ontology/")
                        || link.starts_with("tables/")
                        || link.starts_with("documents/")
                        || paths.contains(&link.as_str()),
                    "dangling link {link} in {}",
                    file.path
                );
            }
        }
        assert!(entities.iter().any(|f| f.content.contains("- near: [")));
        // Re-import: nothing to ingest as a document, and nothing to
        // propose beyond what the restored ontology already has.
        assert_eq!(bundle.documents().count(), 0);
        let snapshot = bundle
            .ontology()
            .unwrap_or_else(|e| unreachable_db(&e.to_string()));
        assert!(propose(&bundle, snapshot.as_ref()).is_empty());
    }

    #[expect(clippy::panic, reason = "test helper: the fixture must build")]
    fn unreachable_db<T>(msg: &str) -> T {
        panic!("fixture failed: {msg}")
    }

    #[test]
    fn proposals_come_from_types_links_and_resource() {
        let mut bundle = Bundle::default();
        bundle.push("index.md", String::from("---\ntype: index\n---\n"));
        bundle.push(
            "vendors/a.md",
            String::from("---\ntype: Vendors\ntitle: A\n---\nships to [K](../countries/k.md)"),
        );
        bundle.push(
            "vendors/b.md",
            String::from("---\ntype: vendors\ntitle: B\n---\n[K](../countries/k.md) again"),
        );
        bundle.push(
            "countries/k.md",
            String::from("---\ntype: country\ntitle: K\n---\n"),
        );
        bundle.push(
            "docs/d.md",
            String::from("---\ntype: document\nresource: d.pdf\n---\n"),
        );
        bundle.push("junk.md", String::from("no front matter"));
        let candidates = propose(&bundle, None);
        let ids: Vec<String> = candidates
            .iter()
            .map(|c| format!("{}:{}", c.proposal.kind(), c.proposal.id()))
            .collect();
        assert_eq!(
            ids,
            [
                "class:country",
                "class:vendor",
                "relation:vendor_links_country"
            ]
        );
        // With a document class in the ontology, `resource` is proposed on it.
        let with_documents = Ontology::builtin_default();
        let candidates = propose(&bundle, Some(&with_documents));
        assert!(
            candidates
                .iter()
                .any(|c| c.proposal.kind() == ItemKind::Property && c.proposal.id() == "resource")
        );
        let mut current = Ontology::default();
        current.classes.push(Class {
            id: ClassId::from("vendor"),
            parent: ClassId::from(String::from(ontology::ROOT_CLASS)),
            label: None,
            description: None,
            key: None,
            properties: Vec::new(),
        });
        let again = propose(&bundle, Some(&current));
        assert!(again.iter().all(|c| c.proposal.id() != "vendor"));
    }

    /// A `- id: [..](..)` link proposes that id; a relation the ontology
    /// already has between the two classes, under any id or on an
    /// ancestor, proposes nothing (issue #70: re-importing quack's own
    /// bundle queued every link again).
    #[test]
    fn named_links_keep_their_id_and_covered_relations_are_skipped() {
        let mut bundle = Bundle::default();
        bundle.push(
            "entities/a.md",
            String::from(
                "---\ntype: vendor\ntitle: A\ngenerator: quack\n---\n## Links\n\n- ships_to: [K](../entities/k.md)\n- Note: [K](../entities/k.md)\n",
            ),
        );
        bundle.push(
            "entities/k.md",
            String::from("---\ntype: country\ntitle: K\ngenerator: quack\n---\n"),
        );
        let ids = |candidates: &[Candidate]| -> Vec<String> {
            candidates
                .iter()
                .filter(|c| c.proposal.kind() == ItemKind::Relation)
                .map(|c| c.proposal.id().to_owned())
                .collect()
        };
        assert_eq!(
            ids(&propose(&bundle, None)),
            ["ships_to", "vendor_links_country"]
        );
        let relation = |id: &str, domain: &str, range: &str| Relation {
            id: RelationId::from(String::from(id)),
            label: None,
            description: None,
            domain: ClassId::from(String::from(domain)),
            range: ClassId::from(String::from(range)),
        };
        let class = |id: &str, parent: &str| Class {
            id: ClassId::from(String::from(id)),
            parent: ClassId::from(String::from(parent)),
            label: None,
            description: None,
            key: None,
            properties: Vec::new(),
        };
        let mut current = Ontology::default();
        current
            .classes
            .push(class("organisation", ontology::ROOT_CLASS));
        current.classes.push(class("vendor", "organisation"));
        current.classes.push(class("country", ontology::ROOT_CLASS));
        current
            .relations
            .push(relation("based_in", "organisation", "country"));
        assert!(ids(&propose(&bundle, Some(&current))).is_empty());
        current.relations.clear();
        current
            .relations
            .push(relation("ships_to", "vendor", "vendor"));
        assert_eq!(
            ids(&propose(&bundle, Some(&current))),
            ["vendor_links_country"]
        );
        assert_eq!(labelled_link("  - killed_in: [x](y.md)"), Some("killed_in"));
        assert_eq!(labelled_link("- Killed In: [x](y.md)"), None);
        assert_eq!(labelled_link("- table [t](../t.md)"), None);
    }
}
