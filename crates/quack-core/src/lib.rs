#[macro_use]
mod text_enum;

pub mod analysis;
pub mod config;
pub mod crypto;
pub mod doctor;
pub mod embedding;
pub mod error;
pub mod extraction;
pub mod graph;
pub mod ids;
pub mod import;
pub mod ingestion;
pub mod jobs;
pub mod llm;
pub mod net;
pub mod ocsf;
pub mod oidc;
pub mod okf;
pub mod ontology;
pub mod prefix;
pub mod priority;
pub mod progress;
pub mod proxy;
pub mod saved;
pub mod storage;
pub mod text;
pub mod vault;
pub mod web_sessions;

/// The duck on `quack serve`'s startup banner and the browser login page.
pub const DUCK: [&str; 4] = ["  __", "<(o )___", " ( ._> /", "  `---'"];
