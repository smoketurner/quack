//! `quack import`: the arguments of an import and of a saved one, run by the
//! command line and the terminal alike.

use std::fmt;
use std::io::Write;

use anyhow::{Context, Result};
use clap::Subcommand;
use quack_core::config::Config;
use quack_core::ids::WorkspaceId;
use quack_core::import::{
    self, ImportPolicy, ImportRequest, ImportSecrets, JsonPointer, KeepSecret, LoadStatus,
    RefreshWith, SavedImport, SourceHeader,
};
use quack_core::llm::Embeddings;
use quack_core::progress::RunControl;
use quack_core::storage::control::ControlPlane;
use quack_core::storage::profile::ColumnTypes;
use quack_core::storage::writer::Writer;
use quack_core::vault::Vault;

use crate::text_or_json::TextOrJson;

#[derive(Subcommand)]
pub enum ImportAction {
    /// List the imports saved with `--save`, with how each last ran
    List {
        /// `json` prints one JSON object per saved import
        #[arg(long, value_enum, default_value_t = TextOrJson::Text)]
        format: TextOrJson,
    },
    /// Run a saved import again; its table is replaced only when the
    /// source changed (cron schedules this)
    Refresh {
        /// The saved import's name or id
        name: String,
    },
    /// Remove a saved import and any secret sealed for it; its table stays
    Remove {
        /// The saved import's name or id
        name: String,
    },
}

#[derive(clap::Args)]
pub struct ImportArgs {
    /// A SQLite path as `sqlite:PATH`, an http(s) URL of a data file, or
    /// `s3://BUCKET/KEY` (the AWS CLI's credentials and region)
    #[arg(required = true)]
    url: Option<String>,
    /// The workspace table to create (replaced when it exists)
    #[arg(long, required = true)]
    table: Option<String>,
    /// A query to run on the source
    #[arg(long, conflicts_with = "from")]
    query: Option<String>,
    /// Pull a whole source table instead of a query
    #[arg(long, value_name = "SOURCE_TABLE")]
    from: Option<String>,
    /// Rows to pull at most (capped by `[import].max_rows`)
    #[arg(long)]
    limit: Option<u64>,
    /// Give columns of the table a type, as COLUMN=TYPE, comma-separated
    /// or repeated; every value must convert
    #[arg(long, value_name = "COLUMN=TYPE")]
    types: Vec<ColumnTypes>,
    /// Send a header with an http(s) download, as `NAME: VALUE`; repeat
    /// for more. Used once and never stored
    #[arg(long = "header", short = 'H', value_name = "NAME: VALUE")]
    headers: Vec<SourceHeader>,
    /// Send `Authorization: Bearer` with the token in this environment
    /// variable, read when the download starts
    #[arg(long, value_name = "VAR")]
    bearer_env: Option<String>,
    /// Load the array of rows at this RFC 6901 pointer inside a JSON
    /// download, as `/data/items`
    #[arg(long, value_name = "POINTER")]
    json_pointer: Option<JsonPointer>,
    /// Save the import under this name, so `quack import refresh NAME`
    /// runs it again
    #[arg(long, value_name = "NAME")]
    save: Option<String>,
    /// Keep the URL's password and the header values with the saved
    /// import, sealed under the vault key, so a refresh can send them
    #[arg(long, requires = "save")]
    store_credential: bool,
}

impl From<ImportArgs> for ImportRequest {
    fn from(args: ImportArgs) -> Self {
        let mut headers = args.headers;
        headers.extend(args.bearer_env.map(SourceHeader::BearerEnv));
        Self {
            query: args.query,
            source_table: args.from,
            limit: args.limit,
            types: ColumnTypes::joined(args.types),
            headers,
            json_pointer: args.json_pointer,
            ..Self::new(args.url.unwrap_or_default(), args.table.unwrap_or_default())
        }
    }
}

/// What every `quack import` form works with, in the CLI and the terminal:
/// the workspace, its writer, and where saved imports keep their sealed
/// secrets.
pub struct ImportContext<'a> {
    pub config: &'a Config,
    pub workspace: &'a WorkspaceId,
    pub control: &'a ControlPlane,
    pub vault: &'a Vault,
    pub db: &'a Writer,
}

impl ImportContext<'_> {
    fn secrets(&self) -> ImportSecrets<'_> {
        ImportSecrets {
            control: self.control,
            vault: self.vault,
            workspace: self.workspace,
        }
    }
}

impl ImportAction {
    /// What a job running it is called.
    #[must_use]
    pub fn label(&self) -> String {
        match self {
            Self::List { .. } => String::from("import list"),
            Self::Refresh { name } => format!("import refresh {name}"),
            Self::Remove { name } => format!("import remove {name}"),
        }
    }

    /// List, refresh, or remove a saved import, reporting to `out`.
    ///
    /// # Errors
    ///
    /// Returns the database, model, or I/O error the command meets.
    pub async fn run(self, context: &ImportContext<'_>, out: &mut impl Write) -> Result<()> {
        let db = context.db;
        match self {
            Self::List { format } => {
                let saved = db.run(SavedImport::list).await?;
                for import in &saved {
                    match format {
                        TextOrJson::Json => writeln!(out, "{}", serde_json::to_string(import)?)?,
                        TextOrJson::Text => writeln!(out, "{}", SavedLine(import))?,
                    }
                }
                if saved.is_empty() && format == TextOrJson::Text {
                    writeln!(
                        out,
                        "No saved imports; save one with `quack import ... --save NAME`."
                    )?;
                }
            }
            Self::Refresh { name } => {
                let saved = db.run(move |db| SavedImport::named(db, &name)).await?;
                let config = context.config;
                let embedder = Embeddings::from_config(config).await?;
                let summary = context
                    .secrets()
                    .refresh(
                        &saved,
                        RefreshWith {
                            config,
                            db,
                            policy: ImportPolicy::owner(),
                            embedder: embedder.as_ref(),
                            control: RunControl::unobserved(),
                        },
                    )
                    .await
                    .with_context(|| format!("refreshing '{}' failed", saved.name))?;
                match (summary.status, saved.last_rows) {
                    (LoadStatus::Unchanged, _) => {
                        writeln!(out, "{}: source unchanged", saved.name)?;
                    }
                    (LoadStatus::Loaded, Some(before)) => writeln!(
                        out,
                        "{}: {} rows (was {before}), replaced",
                        saved.name, summary.rows
                    )?,
                    (LoadStatus::Loaded, None) => {
                        writeln!(out, "{}: {} rows, loaded", saved.name, summary.rows)?;
                    }
                }
            }
            Self::Remove { name } => {
                let saved = db.run(move |db| SavedImport::named(db, &name)).await?;
                context.secrets().remove(db, &saved).await?;
                writeln!(
                    out,
                    "Removed saved import {}; table \"{}\" stays.",
                    saved.name, saved.table
                )?;
            }
        }
        Ok(())
    }
}

impl ImportArgs {
    /// Import once; with `--save`, keep it under a name for refreshing.
    ///
    /// # Errors
    ///
    /// Returns the import's error, a database error, or an I/O error from `out`.
    pub async fn run(self, context: &ImportContext<'_>) -> Result<()> {
        let save = self.save.clone();
        let keep = if self.store_credential {
            KeepSecret::Sealed
        } else {
            KeepSecret::No
        };
        let request = ImportRequest::from(self);
        if let Some(name) = &save {
            request.check_saveable(keep)?;
            let name = name.clone();
            context
                .db
                .run(move |db| SavedImport::check_name(db, &name))
                .await?;
        }
        let config = context.config;
        let embedder = Embeddings::from_config(config).await?;
        let summary = import::Importing {
            config,
            db: context.db,
            workspace_id: context.workspace.as_str(),
            request: &request,
            policy: ImportPolicy::owner(),
            embedder: embedder.as_ref(),
            control: RunControl::unobserved(),
        }
        .run()
        .await
        .context("import failed")?;
        let saved = match save {
            Some(name) => Some(
                context
                    .secrets()
                    .save(context.db, &name, &request, &summary, keep, None)
                    .await?,
            ),
            None => None,
        };
        let stdout = std::io::stdout();
        let mut out = stdout.lock();
        writeln!(
            out,
            "Imported {} rows from {} as table \"{}\" ({} columns: {}).",
            summary.rows,
            summary.source,
            summary.table,
            summary.columns.len(),
            summary.columns.join(", ")
        )?;
        if let Some(saved) = saved {
            writeln!(
                out,
                "Saved as \"{}\"; `quack import refresh {}` runs it again.",
                saved.name, saved.name
            )?;
        }
        Ok(())
    }
}

/// A saved import as `quack import list` prints it: name, table, source,
/// and how it last ran.
struct SavedLine<'a>(&'a SavedImport);

impl fmt::Display for SavedLine<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let import = self.0;
        write!(f, "{}  -> {}  {}", import.name, import.table, import.source)?;
        match (&import.last_error, import.last_rows, &import.last_run_at) {
            (Some(error), _, Some(at)) => write!(f, "  failed at {at}: {error}"),
            (None, Some(rows), Some(at)) => write!(f, "  {rows} rows at {at}"),
            _ => Ok(()),
        }
    }
}
