//! Open Knowledge Format bundles (OKF v0.1): a directory of Markdown files
//! with YAML front matter, one concept per file, `type` required,
//! Markdown links between files as the graph, an optional `index.md` and
//! `log.md`. A workspace exports as one (tables, ontology, documents,
//! context, log, and one concept per graph node with provenance), and a
//! bundle imports as documents whose front matter and links feed ontology
//! induction. Design doc section 17, issue #36.

use crate::analysis::table_search;
use std::collections::{BTreeMap, HashMap};
use std::fmt::{self, Write as _};
use std::io::{Cursor, Read, Write};
use std::path::Path;

use serde::Deserialize;

use crate::error::{Error, Result};
use crate::graph;
use crate::ids::{ClassId, RelationId, RunId};
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

/// Where a bundle's files go, one at a time: an export writes each file as
/// it is made, so nothing holds the whole bundle.
pub trait BundleSink {
    /// Write one file at `path`, relative to the bundle's root.
    ///
    /// # Errors
    ///
    /// Returns an error if the write fails.
    fn file(&mut self, path: &str, content: &str) -> Result<()>;
}

/// A directory: each file lands under the root as it comes, directories
/// created as needed.
pub struct DirSink<'a> {
    root: &'a Path,
}

impl<'a> DirSink<'a> {
    #[must_use]
    pub fn new(root: &'a Path) -> Self {
        Self { root }
    }
}

impl BundleSink for DirSink<'_> {
    fn file(&mut self, path: &str, content: &str) -> Result<()> {
        let target = self.root.join(path);
        if let Some(parent) = target.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(&target, content)?;
        Ok(())
    }
}

/// An uncompressed tar archive written to `W` entry by entry: stdout, an
/// HTTP body, or a buffer.
pub struct TarSink<W: Write> {
    builder: tar::Builder<W>,
}

impl<W: Write> TarSink<W> {
    #[must_use]
    pub fn new(out: W) -> Self {
        Self {
            builder: tar::Builder::new(out),
        }
    }

    /// Write the archive's end and hand back the writer.
    ///
    /// # Errors
    ///
    /// Returns an error if the final write fails.
    pub fn finish(self) -> Result<W> {
        self.builder.into_inner().map_err(Error::Io)
    }
}

impl<W: Write> BundleSink for TarSink<W> {
    fn file(&mut self, path: &str, content: &str) -> Result<()> {
        let mut header = tar::Header::new_gnu();
        header.set_size(u64::try_from(content.len()).unwrap_or(u64::MAX));
        header.set_mode(0o644);
        header.set_mtime(0);
        header.set_cksum();
        self.builder
            .append_data(&mut header, path, content.as_bytes())?;
        Ok(())
    }
}

/// A bundle collected in memory, for tests and callers that already hold
/// one.
impl BundleSink for Bundle {
    fn file(&mut self, path: &str, content: &str) -> Result<()> {
        self.push(path, content.to_owned());
        Ok(())
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
    /// Whether `root` is a bundle rather than a folder of files: it has
    /// an `index.md` or a `log.md` at its top.
    #[must_use]
    pub fn is_dir(root: &Path) -> bool {
        root.join(INDEX).is_file() || root.join(LOG).is_file()
    }

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
            .filter(|f| parse_front_matter(&f.content).front.get("generator") != Some(GENERATOR))
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
        let run = if candidates.is_empty() {
            None
        } else {
            Some(candidates::store_run(db, &candidates)?)
        };
        Ok(RestoreReport {
            restored,
            candidates: candidates.len(),
            run,
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
        let body = parse_front_matter(&file.content).body;
        let json = body
            .split("```json")
            .nth(1)
            .and_then(|rest| rest.split("```").next())
            .ok_or_else(|| Error::Ingestion(format!("{ONTOLOGY_SNAPSHOT} holds no JSON")))?;
        Ontology::from_json(json.trim()).map(Some)
    }
}

/// What restoring a bundle did.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RestoreReport {
    /// The version the bundle's ontology was saved as, when the workspace
    /// had none and the bundle carried one.
    pub restored: Option<OntologyVersion>,
    /// Candidates the bundle's types and links queued for review.
    pub candidates: usize,
    /// The run id the candidates were stored under, when any were.
    pub run: Option<RunId>,
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

/// A Markdown file split into its front matter and its body.
#[derive(Debug, Clone)]
pub struct WithFrontMatter<'a> {
    pub front: FrontMatter,
    pub body: &'a str,
}

/// Split a file into its front matter and body. The front matter is the
/// flat YAML OKF uses: `key: value` lines and a `tags` list (inline
/// `[a, b]` or `- a` items); anything else is kept as text.
#[must_use]
pub fn parse_front_matter(content: &str) -> WithFrontMatter<'_> {
    let mut front = FrontMatter::default();
    let Some(rest) = content
        .strip_prefix("---\n")
        .or_else(|| content.strip_prefix("---\r\n"))
    else {
        return WithFrontMatter {
            front,
            body: content,
        };
    };
    let Some(end) = rest.find("\n---") else {
        return WithFrontMatter {
            front,
            body: content,
        };
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
    WithFrontMatter { front, body }
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
/// Files go to `sink` as they are made; graph nodes stream from one
/// query, so memory holds one node at a time plus the index, which lists
/// every file and is written last.
///
/// # Errors
///
/// Returns an error if a read or a write to `sink` fails.
pub fn export(
    db: &WorkspaceDb,
    workspace_name: &str,
    sink: &mut impl BundleSink,
) -> Result<ExportSummary> {
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
        sink,
        index,
        files: 0,
    };

    exporter.tables(ontology.as_ref())?;
    if let Some(ontology) = &ontology {
        exporter.ontology(ontology)?;
    }
    exporter.documents(&db.list_documents()?)?;
    exporter.entities()?;
    exporter.log()?;
    let index = std::mem::take(&mut exporter.index);
    exporter.write(INDEX, &index)?;
    Ok(ExportSummary {
        files: exporter.files,
    })
}

/// What an export wrote.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExportSummary {
    pub files: usize,
}

/// An export in progress: where its files go, and the index listing them.
struct Exporter<'a, S: BundleSink> {
    db: &'a WorkspaceDb,
    sink: &'a mut S,
    index: String,
    files: usize,
}

impl<S: BundleSink> Exporter<'_, S> {
    /// Write a file the index does not list.
    fn write(&mut self, path: &str, text: &str) -> Result<()> {
        self.sink.file(path, text)?;
        self.files = self.files.saturating_add(1);
        Ok(())
    }

    /// Write a file, listing it in the index under `title`.
    fn listed(&mut self, path: &str, title: &str, text: &str) -> Result<()> {
        writeln!(self.index, "- [{title}]({path})")?;
        self.write(path, text)
    }

    fn tables(&mut self, ontology: Option<&Ontology>) -> Result<()> {
        for table in &table_search::user_tables(self.db)? {
            let described = self.db.describe_table_under(table, ontology)?;
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
            self.listed(&format!("tables/{}.md", slug(table)), table, &text)?;
        }
        Ok(())
    }

    fn documents(&mut self, documents: &[DocumentInfo]) -> Result<()> {
        let paths: HashMap<String, String> = self
            .db
            .connection()
            .prepare(DOCUMENT_PATH_QUERY)?
            .query_map([], |row| Ok((row.get(0)?, row.get(1)?)))?
            .collect::<duckdb::Result<_>>()?;
        for document in documents {
            let Some(path) = paths.get(&document.id.to_string()) else {
                continue;
            };
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
                bool::from(document.pinning)
            )?;
            if let Some(tables) = &document.tables {
                for table in tables {
                    writeln!(text, "- table: [{table}](../tables/{}.md)", slug(table))?;
                }
            }
            self.listed(path, document.display_name(), &text)?;
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
                &format!("ontology/classes/{}.md", slug(class.id.as_str())),
                class.id.as_str(),
                &text,
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
        self.write(ONTOLOGY_SNAPSHOT, &snapshot)
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
            self.write(
                &format!("ontology/relations/{}.md", slug(relation.id.as_str())),
                &text,
            )?;
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
            self.write(
                &format!("ontology/properties/{}.md", slug(&property.id)),
                &text,
            )?;
        }
        Ok(())
    }

    /// One file per graph node, streamed from [`ENTITY_QUERY`] in node
    /// order: nothing about the graph is held but the row in hand.
    fn entities(&mut self) -> Result<()> {
        let mut stmt = self.db.connection().prepare(ENTITY_QUERY)?;
        let mut rows = stmt.query([])?;
        while let Some(row) = rows.next()? {
            let entity = EntityFile::try_from(row)?;
            let text = entity.render()?;
            self.listed(&entity.path, &entity.node.label, &text)?;
        }
        Ok(())
    }

    fn log(&mut self) -> Result<()> {
        let mut log = FrontMatter::of_type(ConceptType::Log).to_string();
        log.push_str("# Log\n\n## Ontology versions\n\n");
        for version in ontology_store::versions(self.db, 100)? {
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
        self.write(LOG, &log)
    }
}

/// [`slug`] in SQL, for a column: every run of characters outside
/// `[A-Za-z0-9]` becomes one `-`, the rest is lowercased, dashes at either
/// end are dropped, and nothing left is `untitled`. Replacing before
/// lowercasing keeps it ASCII-only, as `slug` is; a test holds the two to
/// the same answers.
macro_rules! sql_slug {
    ($column:literal) => {
        concat!(
            "coalesce(nullif(trim(lower(regexp_replace(",
            $column,
            ", '[^A-Za-z0-9]+', '-', 'g')), '-'), ''), 'untitled')"
        )
    };
}

/// The `docs` view: every live document with the file it is written to,
/// `documents/{filename}.md` by slug, suffixed as entities are when two
/// file names share a slug (`Report.pdf` and `report.pdf`, or one name in
/// two folders).
macro_rules! document_paths {
    () => {
        concat!(
            "docs AS ( \
               SELECT id, 'documents/' || s || \
                      CASE WHEN row_number() OVER (PARTITION BY s ORDER BY filename, id) = 1 \
                           THEN '' ELSE '-' || reverse(right(id, 6)) END || '.md' AS path \
               FROM (SELECT id, filename, ",
            sql_slug!("filename"),
            " AS s FROM _quack_documents WHERE status <> 'superseded'))"
        )
    };
}

/// Each live document's id and the file it is written to.
const DOCUMENT_PATH_QUERY: &str = concat!("WITH ", document_paths!(), " SELECT id, path FROM docs");

/// Every graph node with the file it is written to, its outbound links
/// (with each target's file), and its provenance, in node order.
///
/// A node's file is `entities/{class}/{label}.md` by slug; when labels
/// of one class share a slug, the first in label order keeps the plain
/// name and the rest add the last six characters of their id, reversed.
/// The window computes that once in `DuckDB`, which spills to disk as it
/// needs, instead of a path map over every node in memory.
const ENTITY_QUERY: &str = concat!(
    "WITH ",
    document_paths!(),
    ", n AS ( \
       SELECT id, label, class_id, properties, provisional, ",
    sql_slug!("class_id"),
    " AS class_slug, ",
    sql_slug!("label"),
    " AS label_slug FROM _quack_graph_nodes), \
     p AS ( \
       SELECT *, 'entities/' || class_slug || '/' || label_slug || \
              CASE WHEN row_number() OVER (PARTITION BY class_slug, label_slug ORDER BY label, id) = 1 \
                   THEN '' ELSE '-' || reverse(right(id, 6)) END || '.md' AS path \
       FROM n), \
     links AS ( \
       SELECT e.source_node_id AS id, \
              to_json(list({'relation': e.relation_id, 'label': t.label, 'path': t.path} \
                           ORDER BY e.relation_id, t.label, e.id))::VARCHAR AS links \
       FROM _quack_graph_edges e JOIN p t ON t.id = e.target_node_id \
       GROUP BY e.source_node_id), \
     sources AS ( \
       SELECT pr.subject_id AS id, \
              to_json(list({'table_name': pr.table_name, 'row_key': pr.row_key, \
                            'chunk_id': pr.chunk_id, \
                            'document': coalesce(d.filename, pr.document_id), \
                            'document_path': docs.path, \
                            'confidence': coalesce(pr.confidence, 1.0), \
                            'author': pr.author, 'note': pr.note, \
                            'asserted_at': CAST(pr.asserted_at AS VARCHAR)} \
                           ORDER BY pr.chunk_id, pr.table_name, pr.row_key))::VARCHAR AS sources \
       FROM _quack_provenance pr \
       JOIN _quack_graph_nodes g ON g.id = pr.subject_id \
       LEFT JOIN _quack_documents d ON d.id = pr.document_id \
       LEFT JOIN docs ON docs.id = pr.document_id \
       GROUP BY pr.subject_id) \
     SELECT p.id, p.label, p.class_id, CAST(p.properties AS VARCHAR), p.provisional, \
            p.path, links.links, sources.sources \
     FROM p LEFT JOIN links ON links.id = p.id LEFT JOIN sources ON sources.id = p.id \
     ORDER BY p.class_id, p.label, p.id"
);

/// One node's concept file, as [`ENTITY_QUERY`] returns it.
struct EntityFile {
    node: graph::Node,
    path: String,
    links: Vec<EntityLink>,
    sources: Vec<EntitySource>,
}

/// An outbound edge, with the file its target is written to.
#[derive(Deserialize)]
struct EntityLink {
    relation: String,
    label: String,
    path: String,
}

/// Where a node came from: a table row, a document chunk (`document` is
/// the file name, or the id when the document is gone; `document_path`
/// its file in the bundle, `None` when the bundle has none), or a
/// person's assertion.
#[derive(Deserialize)]
struct EntitySource {
    table_name: String,
    row_key: String,
    chunk_id: String,
    document: Option<String>,
    #[serde(default)]
    document_path: Option<String>,
    confidence: f64,
    #[serde(default)]
    author: Option<String>,
    #[serde(default)]
    note: Option<String>,
    #[serde(default)]
    asserted_at: Option<String>,
}

impl TryFrom<&duckdb::Row<'_>> for EntityFile {
    type Error = Error;

    fn try_from(row: &duckdb::Row<'_>) -> Result<Self> {
        let list = |idx: usize| -> Result<Option<String>> { Ok(row.get(idx)?) };
        Ok(Self {
            node: graph::Node::try_from(row)?,
            path: row.get(5)?,
            links: list(6)?
                .map(|text| serde_json::from_str(&text))
                .transpose()?
                .unwrap_or_default(),
            sources: list(7)?
                .map(|text| serde_json::from_str(&text))
                .transpose()?
                .unwrap_or_default(),
        })
    }
}

impl EntityFile {
    /// Front matter, properties, a link per outbound edge, and provenance.
    fn render(&self) -> Result<String> {
        let node = &self.node;
        let mut front = FrontMatter::of_type(&node.class_id)
            .field("generator", GENERATOR)
            .field("id", node.id.to_string())
            .field("title", node.label.clone());
        if node.standing == graph::Standing::Provisional {
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
        if !self.links.is_empty() {
            text.push_str("## Links\n\n");
            for link in &self.links {
                writeln!(
                    text,
                    "- {}: [{}](../../{})",
                    link.relation, link.label, link.path
                )?;
            }
            text.push('\n');
        }
        if !self.sources.is_empty() {
            text.push_str("## Provenance\n\n");
            let lines: Vec<String> = self.sources.iter().map(EntitySource::line).collect();
            text.push_str(&lines.join("\n"));
            text.push('\n');
        }
        Ok(text)
    }
}

impl EntitySource {
    /// The provenance line, linking the table or document stub.
    fn line(&self) -> String {
        if let Some(asserted_at) = &self.asserted_at {
            let assertion = graph::Origin::Manual {
                author: self.author.clone(),
                note: self.note.clone(),
                asserted_at: asserted_at.clone(),
            }
            .assertion()
            .unwrap_or_default();
            return format!("- {assertion} ({asserted_at})");
        }
        match (self.table_name.is_empty(), &self.document) {
            (false, _) => format!(
                "- table [{}](../../tables/{}.md) row `{}`",
                self.table_name,
                slug(&self.table_name),
                self.row_key
            ),
            (true, Some(document)) => match &self.document_path {
                Some(path) => format!(
                    "- document [{document}](../../{path}) chunk `{}` (confidence {:.2})",
                    self.chunk_id, self.confidence
                ),
                None => format!(
                    "- document {document} chunk `{}` (confidence {:.2})",
                    self.chunk_id, self.confidence
                ),
            },
            (true, None) => String::from("- unknown"),
        }
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
        let WithFrontMatter { front, body } = parse_front_matter(&file.content);
        // quack's own structural stubs (tables, documents, ontology files)
        // describe the workspace, not knowledge: only its entity files
        // carry types worth proposing.
        if front.get("generator") == Some(GENERATOR) && !file.path.starts_with("entities/") {
            continue;
        }
        let Some(raw) = front.get("type") else {
            continue;
        };
        // quack's own entity files carry the canonical class id verbatim, so
        // the foreign-bundle plural-folding `type_id` would mangle ids ending
        // in a single `s` (e.g. `series` -> `serie`): keep them as written.
        let kind =
            if front.get("generator") == Some(GENERATOR) && file.path.starts_with("entities/") {
                raw.to_owned()
            } else {
                type_id(raw)
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
                property: Property::new(String::from("resource"), PropertyType::String, Vec::new()),
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
mod tests;
