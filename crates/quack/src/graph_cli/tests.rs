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
    GraphAction::Revalidate { yes }
        .run(
            &Config::default(),
            db,
            Confirm::Assume,
            &mut out,
            RunControl::unobserved(),
        )
        .await?;
    Ok(String::from_utf8_lossy(&out).into_owned())
}

/// `extract --reset` asks before it clears, as revalidation does: where
/// nobody can be asked it stops with the graph as it was, and `--yes`
/// clears it. Tables-only extraction makes no model calls.
#[tokio::test]
async fn an_extract_reset_asks_before_it_clears_the_graph() {
    let db = stale_graph();
    let reset = |yes: bool| ExtractArgs {
        source: ExtractSource::Tables,
        sample: None,
        reset: true,
        all: true,
        yes,
    };
    let mut out = Vec::new();
    let refused = reset(false)
        .run(
            &Config::default(),
            &db,
            &mut out,
            Confirm::Assume,
            RunControl::unobserved(),
        )
        .await
        .err()
        .map(|e| e.to_string())
        .unwrap_or_default();
    assert!(
        refused.contains("Clear all 2 nodes and 0 edges, including what people asserted?")
            && refused.contains("nothing was dropped"),
        "{refused}"
    );
    let nodes = || async {
        db.run(graph_store::status)
            .await
            .unwrap_or_else(|e| fail(&e.to_string()))
            .nodes
    };
    assert_eq!(nodes().await, 2, "nothing was cleared");
    reset(true)
        .run(
            &Config::default(),
            &db,
            &mut out,
            Confirm::Assume,
            RunControl::unobserved(),
        )
        .await
        .unwrap_or_else(|e| fail(&e.to_string()));
    assert_eq!(nodes().await, 0, "--yes cleared the graph");
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
        action
            .run(
                &Config::default(),
                db,
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
