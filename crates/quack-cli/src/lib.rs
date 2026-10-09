//! The command-line verbs the binary and the terminal share: print mode,
//! saved questions, the graph, ontology, tables, and embeddings commands,
//! and the arguments they take. Design doc section 11.

mod args;
mod confirm;
pub mod embeddings_cli;
pub mod graph_cli;
mod import_cli;
pub mod ontology_cli;
mod print;
pub mod saved_cli;
mod session;
mod stdio;
pub mod tables_cli;
mod text_or_json;

pub use args::{ExportFlags, ModeArg, QueryFormat};
pub use confirm::Confirm;
pub use import_cli::{ImportAction, ImportArgs, ImportContext};
pub use print::{AnswerTo, FOLLOW_UP_GRACE, PrintTurn, TurnOutcome};
pub use session::find_session;
pub use stdio::{NamedInput, StdioPath};
pub use text_or_json::TextOrJson;
