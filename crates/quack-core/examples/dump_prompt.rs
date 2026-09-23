//! Prints the system prompt for a small realistic workspace (a table, a
//! document, the built-in ontology, one graph node) as JSON, so a shell
//! script can feed it into a raw Ollama `/api/chat` call to measure
//! prefix-cache behavior across turns. Not a test; throwaway harness.

#![expect(clippy::print_stdout, reason = "a benchmark harness, not shipped code")]
#![expect(clippy::unwrap_used, reason = "a throwaway harness")]

use quack_core::analysis::policy::WritePolicy;
use quack_core::analysis::text_to_sql::{self, PromptOptions};
use quack_core::storage::sessions::ChatMode;
use quack_core::storage::workspace::{NewDocument, WorkspaceDb};

fn main() {
    let db = WorkspaceDb::open_in_memory(4).unwrap();
    db.execute_statement("CREATE TABLE claims(id INT, amount INT, status VARCHAR)")
        .unwrap();
    db.execute_statement("INSERT INTO claims VALUES (1, 100, 'paid'), (2, 200, 'denied')")
        .unwrap();
    db.insert_document(
        &NewDocument::new("d1", "policy.pdf", "application/pdf", 1)
            .with_status(quack_core::storage::workspace::DocumentStatus::Ready),
    )
    .unwrap();
    quack_core::ontology::store::save(
        &db,
        &quack_core::ontology::Ontology::builtin_default(),
        Some("tester"),
        None,
    )
    .unwrap();
    quack_core::graph::store::upsert_node(
        &db,
        &quack_core::graph::store::NewNode {
            label: String::from("Acme"),
            class_id: String::from("organization"),
            properties: serde_json::json!({}),
            provisional: false,
        },
    )
    .unwrap();

    let options = PromptOptions {
        mode: ChatMode::Chat,
        write_policy: WritePolicy::Deny,
        pinned_token_budget: 1000,
        context: Some(String::from("Amounts are in cents.")),
        context_max_tokens: 1000,
        ollama_context_cap: None,
    };
    let prompt = text_to_sql::build_system_prompt(&db, &options).unwrap();
    println!("{}", serde_json::to_string(&prompt).unwrap());
}
