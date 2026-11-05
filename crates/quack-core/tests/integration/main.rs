//! quack-core's integration tests, one binary: each file was its own,
//! and each linked the whole engine and `DuckDB` again, which made the
//! links most of a test run's time and disk.

mod agent_turn;
mod embedding_integration;
mod graph_integration;
mod ingestion_integration;
mod okf_series;
