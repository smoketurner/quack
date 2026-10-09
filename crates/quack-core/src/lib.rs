//! The engine behind the `quack` binary: workspaces of documents, tables,
//! and a knowledge graph, and the agent that answers across them.
//!
//! This crate is internal to quack. It is not published, and it makes no
//! stability promise: the `quack` binary's subcommands (the web UI, the REST
//! API, the MCP server, the terminal, print mode) are its only clients, and
//! its public API changes whenever one of them needs it to.
//!
//! The entry points those clients use:
//!
//! - [`config::Config::load`]: the settings, from `config.toml` and the
//!   environment.
//! - [`llm::TurnRequest::run`]: one agent turn over a workspace, as a stream
//!   of [`analysis::events::AgentEvent`]s.
//! - [`ingestion::ingest_file`]: a file into a workspace, as tables or chunks.
//! - [`graph::query::GraphQuery::run`]: a traversal of the knowledge graph.
//! - [`storage::workspace::WorkspaceDb::execute_query`]: SQL against the
//!   workspace file.
//! - [`jobs::JobQueue`]: the background work every interface reports from.

#[macro_use]
mod text_enum;

pub mod analysis;
pub mod classify;
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
pub mod setup;
pub mod storage;
pub mod telemetry;
pub mod text;
pub mod vault;
pub mod web_sessions;

/// The duck on `quack serve`'s startup banner and the browser login page.
pub const DUCK: [&str; 4] = ["  __", "<(o )___", " ( ._> /", "  `---'"];
