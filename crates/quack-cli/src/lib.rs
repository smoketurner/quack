//! The command-line verbs the binary and the terminal share: print mode,
//! saved questions, the graph, ontology, tables, and embeddings commands,
//! and the arguments they take. Design doc section 11.

pub mod args;
pub mod confirm;
pub mod embeddings_cli;
pub mod graph_cli;
pub mod import_cli;
pub mod ontology_cli;
pub mod print;
pub mod saved_cli;
pub mod session;
pub mod stdio;
pub mod tables_cli;
pub mod text_or_json;
