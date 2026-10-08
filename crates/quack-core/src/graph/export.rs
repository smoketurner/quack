//! The whole graph in forms other tools load (#411): a CSV bundle of
//! `nodes.csv`, `edges.csv`, and `provenance.csv` (Gephi, Neo4j's
//! `LOAD CSV`, pandas), `GraphML` (`NetworkX`, Gephi, yEd), or JSON-LD (any
//! linked-data store).
//!
//! Each part streams from one ordered prepared statement, so memory holds
//! one row at a time whatever the graph's size. A tar entry needs its size
//! before its bytes, so a CSV bundle written as a tar stages each part in
//! an anonymous file first ([`WorkspaceDb::spool_file`]). Nodes export
//! their id, label, class, properties, and standing, never their vector
//! or normalized label.

use std::borrow::Cow;
use std::fs::File;
use std::io::{BufWriter, Seek as _, Write};
use std::path::Path;

use quick_xml::Writer as XmlWriter;
use quick_xml::events::{BytesDecl, BytesEnd, BytesStart, BytesText, Event};
use serde::{Deserialize, Serialize};

use super::{Edge, Node, Origin, ProvenanceColumns, Standing};
use crate::error::Result;
use crate::ids::{ChunkId, DocumentId};
use crate::ontology::{MENTIONS_RELATION, Ontology, ROOT_CLASS, store as ontology_store};
use crate::storage::workspace::WorkspaceDb;

/// The form an export takes.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize, utoipa::ToSchema)]
#[serde(rename_all = "lowercase")]
pub enum GraphFormat {
    /// `nodes.csv`, `edges.csv`, and `provenance.csv`.
    #[default]
    Csv,
    GraphMl,
    JsonLd,
}

text_enum!(GraphFormat, "graph export format", {
    Csv => "csv",
    GraphMl => "graphml",
    JsonLd => "jsonld",
});

impl GraphFormat {
    /// The media type of the export written as one stream: a tar of the
    /// CSV bundle, or the document itself.
    #[must_use]
    pub const fn content_type(self) -> &'static str {
        match self {
            Self::Csv => "application/x-tar",
            Self::GraphMl => "application/graphml+xml",
            Self::JsonLd => "application/ld+json",
        }
    }

    /// The single document this format is, or `None` for the CSV bundle.
    const fn document(self) -> Option<DocumentKind> {
        match self {
            Self::Csv => None,
            Self::GraphMl => Some(DocumentKind::GraphMl),
            Self::JsonLd => Some(DocumentKind::JsonLd),
        }
    }

    /// The file extension of the export written as one stream.
    #[must_use]
    pub const fn extension(self) -> &'static str {
        match self {
            Self::Csv => "tar",
            Self::GraphMl => "graphml",
            Self::JsonLd => "jsonld",
        }
    }
}

/// Whether nodes and edges built from an unreviewed ontology go out too.
/// Without them, an edge goes only when it and both its ends are reviewed.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum ProvisionalExport {
    #[default]
    Exclude,
    Include,
}

flag_enum!(ProvisionalExport, false => Exclude, true => Include);

/// Where an export goes.
pub enum Destination<'a, W: Write> {
    /// A directory, created when missing: the CSV bundle's three files, or
    /// `graph.graphml` or `graph.jsonld`.
    Dir(&'a Path),
    /// One stream: a tar of the CSV bundle, or the document itself.
    Stream(W),
}

/// What an export wrote; the API's audit detail.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct ExportSummary {
    pub format: GraphFormat,
    pub nodes: u64,
    pub edges: u64,
    /// Provenance rows of the exported nodes and edges.
    pub provenance: u64,
}

/// An export of the whole graph.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct GraphExport {
    pub format: GraphFormat,
    pub provisional: ProvisionalExport,
}

impl GraphExport {
    /// Write the graph to `destination`. Run it inside one read
    /// ([`WorkspaceDb::read_only`]) so every part sees the same graph.
    ///
    /// # Errors
    ///
    /// Returns an error if a query or a write fails.
    pub fn write<W: Write>(
        self,
        db: &WorkspaceDb,
        destination: Destination<'_, W>,
    ) -> Result<ExportSummary> {
        let mut summary = ExportSummary {
            format: self.format,
            nodes: 0,
            edges: 0,
            provenance: 0,
        };
        match (self.format.document(), destination) {
            (None, Destination::Dir(dir)) => {
                std::fs::create_dir_all(dir)?;
                for part in Part::ALL {
                    let file = File::create(dir.join(part.file_name()))?;
                    let rows = part.write_csv(db, self.provisional, file)?;
                    summary.count(part, rows);
                }
            }
            (None, Destination::Stream(out)) => {
                let mut tar = tar::Builder::new(out);
                for part in Part::ALL {
                    let mut spool = db.spool_file()?;
                    let rows = part.write_csv(db, self.provisional, &mut spool)?;
                    summary.count(part, rows);
                    let mut header = tar::Header::new_gnu();
                    header.set_size(spool.stream_position()?);
                    header.set_mode(0o644);
                    header.set_mtime(0);
                    spool.rewind()?;
                    tar.append_data(&mut header, part.file_name(), spool)?;
                }
                tar.into_inner()?.flush()?;
            }
            (Some(kind), Destination::Dir(dir)) => {
                std::fs::create_dir_all(dir)?;
                let name = format!("graph.{}", self.format.extension());
                let mut file = BufWriter::new(File::create(dir.join(name))?);
                self.document(db, kind, &mut file, &mut summary)?;
                file.flush()?;
            }
            (Some(kind), Destination::Stream(mut out)) => {
                self.document(db, kind, &mut out, &mut summary)?;
                out.flush()?;
            }
        }
        Ok(summary)
    }

    /// `GraphML` or JSON-LD: nodes, then edges, each with its provenance.
    fn document(
        self,
        db: &WorkspaceDb,
        kind: DocumentKind,
        out: &mut impl Write,
        summary: &mut ExportSummary,
    ) -> Result<()> {
        let mut document = match kind {
            DocumentKind::GraphMl => Document::GraphMl(GraphMl::start(out)?),
            DocumentKind::JsonLd => {
                Document::JsonLd(JsonLd::start(out, ontology_store::current(db)?.as_ref())?)
            }
        };
        let mut stmt = db.connection().prepare(NODES_WITH_SOURCES)?;
        let mut rows = stmt.query(duckdb::params![self.provisional])?;
        while let Some(row) = rows.next()? {
            let node = Node::try_from(row)?;
            let sources = Sources::from_column(row.get(5)?)?;
            summary.add_sources(&sources);
            summary.nodes = summary.nodes.saturating_add(1);
            document.node(&node, &sources)?;
        }
        let mut stmt = db.connection().prepare(EDGES_WITH_SOURCES)?;
        let mut rows = stmt.query(duckdb::params![self.provisional])?;
        while let Some(row) = rows.next()? {
            let edge = Edge::try_from(row)?;
            let sources = Sources::from_column(row.get(7)?)?;
            summary.add_sources(&sources);
            summary.edges = summary.edges.saturating_add(1);
            document.edge(&edge, &sources)?;
        }
        document.finish()
    }
}

impl ExportSummary {
    fn count(&mut self, part: Part, rows: u64) {
        match part {
            Part::Nodes => self.nodes = rows,
            Part::Edges => self.edges = rows,
            Part::Provenance => self.provenance = rows,
        }
    }

    fn add_sources(&mut self, sources: &Sources) {
        let added = u64::try_from(sources.0.len()).unwrap_or(u64::MAX);
        self.provenance = self.provenance.saturating_add(added);
    }
}

// Without provisional rows, a node goes when it is reviewed, and an edge
// when it and both its ends are; an edge never goes without both ends. The
// one parameter of each is the `ProvisionalExport` flag.

const NODES: &str = "SELECT n.id, n.label, n.class_id, CAST(n.properties AS VARCHAR), \
     n.provisional FROM _quack_graph_nodes n \
     WHERE ? OR NOT n.provisional ORDER BY n.id";

const EDGES: &str = "SELECT e.id, e.source_node_id, e.target_node_id, e.relation_id, e.weight, \
     CAST(e.properties AS VARCHAR), e.provisional FROM _quack_graph_edges e \
     JOIN _quack_graph_nodes a ON a.id = e.source_node_id \
     JOIN _quack_graph_nodes b ON b.id = e.target_node_id \
     WHERE ? OR NOT (e.provisional OR a.provisional OR b.provisional) ORDER BY e.id";

/// Each subject's provenance as one JSON list, for the documents, which
/// carry it on the node or edge.
macro_rules! sources {
    () => {
        "(SELECT subject_id, to_json(list({'document_id': document_id, 'chunk_id': chunk_id, \
           'table_name': table_name, 'row_key': row_key, \
           'confidence': coalesce(confidence, 1.0), 'author': author, 'note': note, \
           'asserted_at': CAST(asserted_at AS VARCHAR)} \
           ORDER BY chunk_id, table_name, row_key))::VARCHAR AS sources \
         FROM _quack_provenance GROUP BY subject_id) s"
    };
}

const NODES_WITH_SOURCES: &str = concat!(
    "SELECT n.id, n.label, n.class_id, CAST(n.properties AS VARCHAR), n.provisional, s.sources \
     FROM _quack_graph_nodes n LEFT JOIN ",
    sources!(),
    " ON s.subject_id = n.id WHERE ? OR NOT n.provisional ORDER BY n.id"
);

const EDGES_WITH_SOURCES: &str = concat!(
    "SELECT e.id, e.source_node_id, e.target_node_id, e.relation_id, e.weight, \
     CAST(e.properties AS VARCHAR), e.provisional, s.sources FROM _quack_graph_edges e \
     JOIN _quack_graph_nodes a ON a.id = e.source_node_id \
     JOIN _quack_graph_nodes b ON b.id = e.target_node_id LEFT JOIN ",
    sources!(),
    " ON s.subject_id = e.id \
     WHERE ? OR NOT (e.provisional OR a.provisional OR b.provisional) ORDER BY e.id"
);

/// The provenance rows of the kept nodes and edges; the flag binds twice.
const PROVENANCE: &str = "SELECT p.subject_id, p.document_id, p.chunk_id, p.table_name, \
     p.row_key, coalesce(p.confidence, 1.0), p.author, p.note, CAST(p.asserted_at AS VARCHAR) \
     FROM _quack_provenance p \
     WHERE p.subject_id IN (SELECT n.id FROM _quack_graph_nodes n WHERE ? OR NOT n.provisional) \
        OR p.subject_id IN (SELECT e.id FROM _quack_graph_edges e \
             JOIN _quack_graph_nodes a ON a.id = e.source_node_id \
             JOIN _quack_graph_nodes b ON b.id = e.target_node_id \
             WHERE ? OR NOT (e.provisional OR a.provisional OR b.provisional)) \
     ORDER BY p.subject_id, p.chunk_id, p.table_name, p.row_key";

/// One file of the CSV bundle.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Part {
    Nodes,
    Edges,
    Provenance,
}

impl Part {
    const ALL: [Self; 3] = [Self::Nodes, Self::Edges, Self::Provenance];

    const fn file_name(self) -> &'static str {
        match self {
            Self::Nodes => "nodes.csv",
            Self::Edges => "edges.csv",
            Self::Provenance => "provenance.csv",
        }
    }

    /// `source` and `target` are the names Gephi's spreadsheet import
    /// reads an edge list by.
    const fn header(self) -> &'static [&'static str] {
        match self {
            Self::Nodes => &["id", "label", "class_id", "properties", "provisional"],
            Self::Edges => &[
                "id",
                "source",
                "target",
                "relation_id",
                "weight",
                "properties",
                "provisional",
            ],
            Self::Provenance => &[
                "subject_id",
                "document_id",
                "chunk_id",
                "table_name",
                "row_key",
                "confidence",
                "author",
                "note",
                "asserted_at",
            ],
        }
    }

    /// Write this part as CSV with a header row, and count its rows.
    fn write_csv(
        self,
        db: &WorkspaceDb,
        provisional: ProvisionalExport,
        out: impl Write,
    ) -> Result<u64> {
        let mut csv = csv::Writer::from_writer(out);
        csv.write_record(self.header())?;
        let mut written: u64 = 0;
        match self {
            Self::Nodes => {
                let mut stmt = db.connection().prepare(NODES)?;
                let mut rows = stmt.query(duckdb::params![provisional])?;
                while let Some(row) = rows.next()? {
                    let node = Node::try_from(row)?;
                    csv.write_record([
                        node.id.as_str(),
                        &node.label,
                        node.class_id.as_str(),
                        &node.properties.to_json(),
                        Flag(node.standing).as_str(),
                    ])?;
                    written = written.saturating_add(1);
                }
            }
            Self::Edges => {
                let mut stmt = db.connection().prepare(EDGES)?;
                let mut rows = stmt.query(duckdb::params![provisional])?;
                while let Some(row) = rows.next()? {
                    let edge = Edge::try_from(row)?;
                    csv.write_record([
                        edge.id.as_str(),
                        edge.source_node_id.as_str(),
                        edge.target_node_id.as_str(),
                        edge.relation_id.as_str(),
                        &edge.weight.to_string(),
                        &edge.properties.to_json(),
                        Flag(edge.standing).as_str(),
                    ])?;
                    written = written.saturating_add(1);
                }
            }
            Self::Provenance => {
                let mut stmt = db.connection().prepare(PROVENANCE)?;
                let mut rows = stmt.query(duckdb::params![provisional, provisional])?;
                while let Some(row) = rows.next()? {
                    let text = |idx: usize| -> Result<String> {
                        Ok(row.get::<_, Option<String>>(idx)?.unwrap_or_default())
                    };
                    let confidence: f64 = row.get(5)?;
                    csv.write_record([
                        text(0)?,
                        text(1)?,
                        text(2)?,
                        text(3)?,
                        text(4)?,
                        confidence.to_string(),
                        text(6)?,
                        text(7)?,
                        text(8)?,
                    ])?;
                    written = written.saturating_add(1);
                }
            }
        }
        csv.flush()?;
        Ok(written)
    }
}

/// A standing as the exports write it: `true` for provisional.
struct Flag(Standing);

impl Flag {
    const fn as_str(&self) -> &'static str {
        match self.0 {
            Standing::Reviewed => "false",
            Standing::Provisional => "true",
        }
    }
}

/// One provenance row as the aggregated JSON list holds it.
#[derive(Deserialize)]
struct SourceRow {
    document_id: Option<DocumentId>,
    chunk_id: ChunkId,
    table_name: String,
    row_key: String,
    confidence: f64,
    author: Option<String>,
    note: Option<String>,
    asserted_at: Option<String>,
}

/// One source of a node or edge in a document: where it came from (a
/// chunk, a table row, or a person's assertion), and how sure the
/// extraction was.
#[derive(Serialize)]
struct Source {
    #[serde(flatten)]
    origin: Origin,
    confidence: f64,
}

impl From<SourceRow> for Source {
    fn from(row: SourceRow) -> Self {
        Self {
            origin: Origin::from_columns(ProvenanceColumns {
                document_id: row.document_id,
                chunk_id: row.chunk_id,
                table_name: row.table_name,
                row_key: row.row_key,
                author: row.author,
                note: row.note,
                asserted_at: row.asserted_at,
            }),
            confidence: row.confidence,
        }
    }
}

/// A node's or edge's sources.
#[derive(Serialize)]
#[serde(transparent)]
struct Sources(Vec<Source>);

impl Sources {
    /// From the aggregated column: `NULL` when the subject has none.
    fn from_column(text: Option<String>) -> Result<Self> {
        let rows: Vec<SourceRow> = match text {
            Some(text) => serde_json::from_str(&text)?,
            None => Vec::new(),
        };
        Ok(Self(rows.into_iter().map(Source::from).collect()))
    }

    fn to_json(&self) -> Result<String> {
        Ok(serde_json::to_string(self)?)
    }
}

/// The formats that are one document rather than a bundle.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DocumentKind {
    GraphMl,
    JsonLd,
}

/// A document export in progress.
enum Document<W: Write> {
    GraphMl(GraphMl<W>),
    JsonLd(JsonLd<W>),
}

impl<W: Write> Document<W> {
    fn node(&mut self, node: &Node, sources: &Sources) -> Result<()> {
        match self {
            Self::GraphMl(graphml) => graphml.node(node, sources),
            Self::JsonLd(jsonld) => jsonld.node(node, sources),
        }
    }

    fn edge(&mut self, edge: &Edge, sources: &Sources) -> Result<()> {
        match self {
            Self::GraphMl(graphml) => graphml.edge(edge, sources),
            Self::JsonLd(jsonld) => jsonld.edge(edge, sources),
        }
    }

    fn finish(self) -> Result<()> {
        match self {
            Self::GraphMl(graphml) => graphml.finish(),
            Self::JsonLd(jsonld) => jsonld.finish(),
        }
    }
}

/// The `GraphML` namespace every reader expects on the root element.
const GRAPHML_NS: &str = "http://graphml.graphdrawing.org/xmlns";

/// The attribute keys a `GraphML` export declares: `(id, for, name, type)`.
/// Node and edge keys are declared apart, since some readers ignore
/// `for="all"`.
const GRAPHML_KEYS: [(&str, &str, &str, &str); 10] = [
    ("label", "node", "label", "string"),
    ("class_id", "node", "class_id", "string"),
    ("n_properties", "node", "properties", "string"),
    ("n_provisional", "node", "provisional", "boolean"),
    ("n_provenance", "node", "provenance", "string"),
    ("relation_id", "edge", "relation_id", "string"),
    ("weight", "edge", "weight", "double"),
    ("e_properties", "edge", "properties", "string"),
    ("e_provisional", "edge", "provisional", "boolean"),
    ("e_provenance", "edge", "provenance", "string"),
];

/// A `GraphML` document written element by element. Properties and
/// provenance are JSON text in a `data` element.
struct GraphMl<W: Write> {
    xml: XmlWriter<W>,
}

impl<W: Write> GraphMl<W> {
    fn start(out: W) -> Result<Self> {
        let mut xml = XmlWriter::new_with_indent(out, b' ', 2);
        xml.write_event(Event::Decl(BytesDecl::new("1.0", Some("UTF-8"), None)))?;
        xml.write_event(Event::Start(
            BytesStart::new("graphml").with_attributes([("xmlns", GRAPHML_NS)]),
        ))?;
        for (id, domain, name, kind) in GRAPHML_KEYS {
            xml.create_element("key")
                .with_attributes([
                    ("id", id),
                    ("for", domain),
                    ("attr.name", name),
                    ("attr.type", kind),
                ])
                .write_empty()?;
        }
        xml.write_event(Event::Start(
            BytesStart::new("graph").with_attributes([("id", "G"), ("edgedefault", "directed")]),
        ))?;
        Ok(Self { xml })
    }

    fn node(&mut self, node: &Node, sources: &Sources) -> Result<()> {
        let data = [
            ("label", node.label.clone()),
            ("class_id", node.class_id.to_string()),
            ("n_properties", node.properties.to_json()),
            ("n_provisional", Flag(node.standing).as_str().to_owned()),
            ("n_provenance", sources.to_json()?),
        ];
        let id = XmlText::new(node.id.as_str());
        self.xml
            .create_element("node")
            .with_attribute(("id", id.as_ref()))
            .write_inner_content(|xml| Self::data(xml, &data))?;
        Ok(())
    }

    fn edge(&mut self, edge: &Edge, sources: &Sources) -> Result<()> {
        let data = [
            ("relation_id", edge.relation_id.to_string()),
            ("weight", edge.weight.to_string()),
            ("e_properties", edge.properties.to_json()),
            ("e_provisional", Flag(edge.standing).as_str().to_owned()),
            ("e_provenance", sources.to_json()?),
        ];
        let id = XmlText::new(edge.id.as_str());
        let source = XmlText::new(edge.source_node_id.as_str());
        let target = XmlText::new(edge.target_node_id.as_str());
        self.xml
            .create_element("edge")
            .with_attributes([
                ("id", id.as_ref()),
                ("source", source.as_ref()),
                ("target", target.as_ref()),
            ])
            .write_inner_content(|xml| Self::data(xml, &data))?;
        Ok(())
    }

    fn data(xml: &mut XmlWriter<W>, data: &[(&str, String)]) -> std::io::Result<()> {
        for (key, value) in data {
            xml.create_element("data")
                .with_attribute(("key", *key))
                .write_text_content(BytesText::new(XmlText::new(value).as_ref()))?;
        }
        Ok(())
    }

    fn finish(mut self) -> Result<()> {
        self.xml.write_event(Event::End(BytesEnd::new("graph")))?;
        self.xml.write_event(Event::End(BytesEnd::new("graphml")))?;
        self.xml.get_mut().write_all(b"\n")?;
        Ok(())
    }
}

/// Text XML 1.0 can carry: a control character other than tab, line feed,
/// and carriage return, and U+FFFE and U+FFFF, have no representation,
/// escaped or not, so each becomes U+FFFD.
struct XmlText<'a>(Cow<'a, str>);

impl<'a> XmlText<'a> {
    fn new(text: &'a str) -> Self {
        if text.chars().all(Self::allowed) {
            return Self(Cow::Borrowed(text));
        }
        Self(Cow::Owned(
            text.chars()
                .map(|c| if Self::allowed(c) { c } else { '\u{fffd}' })
                .collect(),
        ))
    }

    fn allowed(c: char) -> bool {
        matches!(c, '\t' | '\n' | '\r') || (c >= ' ' && c != '\u{fffe}' && c != '\u{ffff}')
    }
}

impl AsRef<str> for XmlText<'_> {
    fn as_ref(&self) -> &str {
        &self.0
    }
}

/// A JSON-LD document written item by item into its `@graph`: first the
/// current ontology's classes (`rdfs:Class`) and relations (`rdf:Property`
/// with domain and range), then every node typed by its class, then every
/// edge as an `rdf:Statement` whose predicate is its relation. Class and
/// relation ids are IRIs under the context's `class:` and `relation:`
/// prefixes, so a class and a relation that share an id stay apart.
struct JsonLd<W: Write> {
    out: W,
    first: bool,
}

impl<W: Write> JsonLd<W> {
    fn start(out: W, ontology: Option<&Ontology>) -> Result<Self> {
        let context = serde_json::json!({
            "@version": 1.1,
            "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
            "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
            "prov": "http://www.w3.org/ns/prov#",
            "quack": "urn:quack:",
            "node": "urn:quack:node:",
            "edge": "urn:quack:edge:",
            "class": "urn:quack:class:",
            "relation": "urn:quack:relation:",
            "label": "rdfs:label",
            "comment": "rdfs:comment",
            "subclass_of": { "@id": "rdfs:subClassOf", "@type": "@id" },
            "domain": { "@id": "rdfs:domain", "@type": "@id" },
            "range": { "@id": "rdfs:range", "@type": "@id" },
            "source": { "@id": "rdf:subject", "@type": "@id" },
            "predicate": { "@id": "rdf:predicate", "@type": "@id" },
            "target": { "@id": "rdf:object", "@type": "@id" },
            "weight": "quack:weight",
            "provisional": "quack:provisional",
            "properties": { "@id": "quack:properties", "@type": "@json" },
            "provenance": { "@id": "prov:wasDerivedFrom", "@type": "@json" },
        });
        let mut doc = Self { out, first: true };
        doc.out.write_all(b"{\"@context\":")?;
        serde_json::to_writer(&mut doc.out, &context)?;
        doc.out.write_all(b",\"@graph\":[")?;
        doc.item(&serde_json::json!({
            "@id": format!("class:{ROOT_CLASS}"),
            "@type": "rdfs:Class",
            "label": ROOT_CLASS,
        }))?;
        doc.item(&serde_json::json!({
            "@id": format!("relation:{MENTIONS_RELATION}"),
            "@type": "rdf:Property",
            "label": MENTIONS_RELATION,
            "domain": format!("class:{ROOT_CLASS}"),
            "range": format!("class:{ROOT_CLASS}"),
        }))?;
        let (classes, relations) = ontology.map_or((&[][..], &[][..]), |o| {
            (o.classes.as_slice(), o.relations.as_slice())
        });
        for class in classes {
            doc.item(&serde_json::json!({
                "@id": format!("class:{}", class.id),
                "@type": "rdfs:Class",
                "label": class.label.as_deref().unwrap_or(class.id.as_str()),
                "comment": class.description,
                "subclass_of": format!("class:{}", class.parent),
            }))?;
        }
        for relation in relations {
            doc.item(&serde_json::json!({
                "@id": format!("relation:{}", relation.id),
                "@type": "rdf:Property",
                "label": relation.label.as_deref().unwrap_or(relation.id.as_str()),
                "comment": relation.description,
                "domain": format!("class:{}", relation.domain),
                "range": format!("class:{}", relation.range),
            }))?;
        }
        Ok(doc)
    }

    fn item(&mut self, item: &serde_json::Value) -> Result<()> {
        if !self.first {
            self.out.write_all(b",")?;
        }
        self.first = false;
        self.out.write_all(b"\n")?;
        serde_json::to_writer(&mut self.out, item)?;
        Ok(())
    }

    fn node(&mut self, node: &Node, sources: &Sources) -> Result<()> {
        self.item(&serde_json::json!({
            "@id": format!("node:{}", node.id),
            "@type": format!("class:{}", node.class_id),
            "label": node.label,
            "properties": node.properties,
            "provisional": bool::from(node.standing),
            "provenance": sources,
        }))
    }

    fn edge(&mut self, edge: &Edge, sources: &Sources) -> Result<()> {
        self.item(&serde_json::json!({
            "@id": format!("edge:{}", edge.id),
            "@type": "rdf:Statement",
            "source": format!("node:{}", edge.source_node_id),
            "predicate": format!("relation:{}", edge.relation_id),
            "target": format!("node:{}", edge.target_node_id),
            "weight": edge.weight,
            "properties": edge.properties,
            "provisional": bool::from(edge.standing),
            "provenance": sources,
        }))
    }

    fn finish(mut self) -> Result<()> {
        self.out.write_all(b"\n]}\n")?;
        Ok(())
    }
}

#[cfg(test)]
mod tests;
