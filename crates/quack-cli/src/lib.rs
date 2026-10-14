//! The command-line verbs the binary and the terminal share: print mode,
//! saved questions, the graph, ontology, tables, and embeddings commands,
//! the arguments they take, and opening the workspace they run in
//! (`OpenedWorkspace`). Design doc section 11.

mod args;
pub mod classify_cli;
mod confirm;
pub mod embeddings_cli;
pub mod graph_cli;
mod import_cli;
pub mod ontology_cli;
mod print;
mod recorded;
pub mod saved_cli;
mod session;
mod stdio;
pub mod tables_cli;
mod text_or_json;
mod workspace;

pub use args::{ExportFlags, ModeArg, QueryFormat};
pub use confirm::Confirm;
pub use import_cli::{ImportAction, ImportArgs, ImportContext};
pub use print::{AnswerTo, FOLLOW_UP_GRACE, PrintTurn, TurnOutcome};
pub use recorded::RecordedRun;
pub use session::{SessionChoice, find_session, resolve_session};
pub use stdio::{NamedInput, StdioPath};
pub use text_or_json::TextOrJson;
pub use workspace::{HeldElsewhere, OpenedWorkspace};
