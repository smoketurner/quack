#![expect(clippy::unwrap_used, reason = "test assertions use unwrap throughout")]

//! Regression tests for OKF bundle re-import of workspaces whose
//! user-defined class ids end in a single `s` (`series`, `status`, ...).
//!
//! `propose` once ran every concept file's `type` through `type_id`, which
//! drops a trailing single `s` from the last word. Quack's own entity files
//! carry the canonical class id verbatim (`EntityFile::render`), so ids like
//! `series` were singularized to `serie` and `Ontology::defines_class` (an
//! exact match) no longer recognized them, queueing a bogus class candidate
//! on every re-import. These guard the fix's two sides: quack-authored
//! entity files keep the raw id, and foreign `type` strings are still
//! normalized.

use quack_core::embedding::Dimension;
use quack_core::graph::{Properties, Standing, store as graph_store};
use quack_core::ids::{ClassId, RelationId};
use quack_core::okf::{Bundle, BundleFile, export, propose};
use quack_core::ontology::store::{self as ontology_store, Revision};
use quack_core::ontology::{self, Class, Ontology, Relation};
use quack_core::storage::workspace::WorkspaceDb;

/// A class with `id` under the root, no extra metadata.
fn class(id: &str) -> Class {
    Class {
        id: ClassId::from(id),
        parent: ClassId::from(String::from(ontology::ROOT_CLASS)),
        label: None,
        description: None,
        key: None,
        properties: Vec::new(),
    }
}

/// Save `ontology` into a fresh in-memory workspace.
fn save(ontology: &Ontology) -> WorkspaceDb {
    let db = WorkspaceDb::open_in_memory(Dimension::new(4)).unwrap();
    ontology_store::save(&db, ontology, Revision::reviewed(None, None)).unwrap();
    db
}

fn new_node(label: &str, class_id: &str) -> graph_store::NewNode {
    graph_store::NewNode {
        label: label.to_owned(),
        class_id: ClassId::from(class_id),
        properties: Properties::default(),
        standing: Standing::Reviewed,
    }
}

fn add_row_provenance(db: &WorkspaceDb, id: &(impl AsRef<str> + ?Sized)) {
    graph_store::add_provenance(db, id, &graph_store::Source::row("t", "k")).unwrap();
}

fn candidate_ids(candidates: &[quack_core::ontology::induction::Candidate]) -> Vec<String> {
    candidates
        .iter()
        .map(|c| format!("{}:{}", c.proposal.kind(), c.proposal.id()))
        .collect()
}

/// Re-importing quack's own bundle proposes nothing when the workspace has a
/// user-defined class id ending in a single `s` (`series`): the restored
/// snapshot already defines it, so `propose` must not singularize the
/// entity file's verbatim `type` into a fresh, mangled id (`serie`).
#[test]
fn series_roundtrip_proposes_bogus_class() {
    let db = WorkspaceDb::open_in_memory(Dimension::new(4)).unwrap();
    let mut ontology = Ontology::builtin_default();
    ontology.classes.push(class("series"));
    ontology_store::save(&db, &ontology, Revision::reviewed(None, None)).unwrap();
    let id = graph_store::upsert_node(&db, &new_node("The Expanse", "series")).unwrap();
    add_row_provenance(&db, &id);

    let mut bundle = Bundle::default();
    export(&db, "ws", &mut bundle).unwrap();

    let snapshot = bundle.ontology().unwrap();
    assert!(
        snapshot.as_ref().is_some_and(|o| o.defines_class("series")),
        "snapshot should define series"
    );
    let candidates = propose(&bundle, snapshot.as_ref());
    assert!(
        candidates.is_empty(),
        "re-import should propose nothing, but got {:?}",
        candidate_ids(&candidates)
    );
}

/// A named link between two `s`-ending classes is not re-proposed on
/// re-import: the covered-relation check needs the unmangled ids to match the
/// snapshot's domain/range. Singularizing `series`/`status` into
/// `serie`/`statu` would defeat that check and queue the relation again.
#[test]
fn named_link_between_s_ending_classes_is_not_reproposed() {
    let mut ontology = Ontology::builtin_default();
    ontology.classes.push(class("series"));
    ontology.classes.push(class("status"));
    ontology.relations.push(Relation {
        id: RelationId::from("includes"),
        label: None,
        description: None,
        domain: ClassId::from("series"),
        range: ClassId::from("status"),
    });
    let db = save(&ontology);
    let serie = graph_store::upsert_node(&db, &new_node("The Expanse", "series")).unwrap();
    let stat = graph_store::upsert_node(&db, &new_node("In Production", "status")).unwrap();
    add_row_provenance(&db, &serie);
    add_row_provenance(&db, &stat);
    let edge = graph_store::upsert_edge(
        &db,
        &serie,
        &stat,
        "includes",
        &Properties::default(),
        Standing::Reviewed,
    )
    .unwrap();
    add_row_provenance(&db, &edge);

    let mut bundle = Bundle::default();
    export(&db, "ws", &mut bundle).unwrap();

    let snapshot = bundle.ontology().unwrap();
    let candidates = propose(&bundle, snapshot.as_ref());
    assert!(
        candidates.is_empty(),
        "re-import should propose nothing, but got {:?}",
        candidate_ids(&candidates)
    );
}

/// The raw-id exemption is scoped to quack-authored entity files: a foreign
/// bundle (no `generator: quack`) under `entities/` still has its `type`
/// normalized, so human plurals fold to a canonical id
/// (`Vendors` -> `vendor`, `Organizations` -> `organization`).
#[test]
fn foreign_bundle_under_entities_still_singularizes_plural_types() {
    let mut bundle = Bundle::default();
    bundle.files.push(BundleFile {
        path: String::from("entities/vendors/a.md"),
        content: String::from("---\ntype: Vendors\ntitle: A\n---\n"),
    });
    bundle.files.push(BundleFile {
        path: String::from("entities/organizations/o.md"),
        content: String::from("---\ntype: Organizations\ntitle: O\n---\n"),
    });
    let candidates = propose(&bundle, None);
    let mut ids = candidate_ids(&candidates);
    ids.sort();
    assert_eq!(
        ids,
        ["class:organization", "class:vendor"],
        "foreign plurals normalize"
    );
}
