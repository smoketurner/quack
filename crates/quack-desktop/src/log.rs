//! `tracing` to stderr and to a rolling daily file under the data
//! directory (`{data_dir}/logs/desktop.YYYY-MM-DD.log`, seven kept), in
//! `[server].log_format`'s text or JSON.

use std::path::Path;

use anyhow::{Context, Result};
use quack_core::config::{Config, LogFormat};
use quack_core::crypto::CryptoModule;
use tracing_appender::non_blocking::WorkerGuard;
use tracing_appender::rolling::{RollingFileAppender, Rotation};
use tracing_subscriber::EnvFilter;
use tracing_subscriber::fmt::writer::MakeWriterExt;

/// Log files kept before the oldest is deleted.
const KEPT: usize = 7;

/// The writer thread that must outlive every log line: drop it last.
pub(crate) struct Logging {
    _guard: WorkerGuard,
}

/// Install the subscriber. Called once, before anything logs.
///
/// # Errors
///
/// When the log directory cannot be created or the file appender built.
pub(crate) fn init(config: &Config) -> Result<Logging> {
    let dir = config.data_dir().join("logs");
    std::fs::create_dir_all(&dir).with_context(|| format!("failed to create {}", dir.display()))?;
    let file = appender(&dir)?;
    let (file, guard) = tracing_appender::non_blocking(file);
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
    let builder = tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_ansi(false)
        .with_writer(std::io::stderr.and(file));
    match config.server.log_format {
        LogFormat::Text => builder.init(),
        LogFormat::Json => builder.json().init(),
    }
    // The provider was installed before any subscriber existed; this is
    // the first point where saying so reaches a log.
    CryptoModule::linked().log();
    Ok(Logging { _guard: guard })
}

fn appender(dir: &Path) -> Result<RollingFileAppender> {
    RollingFileAppender::builder()
        .rotation(Rotation::DAILY)
        .max_log_files(KEPT)
        .filename_prefix("desktop")
        .filename_suffix("log")
        .build(dir)
        .with_context(|| format!("failed to open the log file in {}", dir.display()))
}
