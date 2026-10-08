//! Imports saved under a name so they can run again (`quack import
//! refresh`, `POST .../imports/{id}/refresh`), kept in `_quack_imports` in
//! the workspace file. A refresh replaces the table through the usual
//! `--replace` path: the old rows keep serving until the new ones are ready,
//! and identical bytes leave everything in place.
//!
//! Secrets never reach the workspace file. A URL is stored redacted and a
//! header by name. A refresh gets its secret back in one of two ways: a
//! bearer token from the environment variable `--bearer-env` named (only the
//! name is saved), or the URL and header values the owner chose to keep with
//! `--store-credential`, sealed under the vault key in `control.db`
//! ([`ImportSecrets`]). An import with a secret and neither is refused
//! before it runs.

use http::{HeaderName, HeaderValue};
use serde::{Deserialize, Serialize};

use super::{
    ImportPolicy, ImportRequest, ImportSummary, Importing, JsonPointer, SourceHeader, SourceUrl,
};
use crate::config::Config;
use crate::embedding::{Embedder, EmbeddingModel};
use crate::error::{Error, Result};
use crate::ids::{DocumentId, ImportId, WorkspaceId};
use crate::progress::RunControl;
use crate::storage::control::{ControlPlane, ResourceKind, SealedOwner};
use crate::storage::profile::{ColumnRetype, ColumnTypes};
use crate::storage::workspace::WorkspaceDb;
use crate::storage::writer::Writer;
use crate::vault::{Opened, Purpose, Vault};

/// An import saved under a name, as `_quack_imports` keeps it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, utoipa::ToSchema)]
pub struct SavedImport {
    pub id: ImportId,
    pub name: String,
    /// The source, with any password removed.
    pub source: String,
    pub table: String,
    pub query: Option<String>,
    pub source_table: Option<String>,
    pub limit: Option<u64>,
    /// The columns retyped after each load.
    #[schema(value_type = Vec<ColumnRetype>)]
    pub types: ColumnTypes,
    pub json_pointer: Option<String>,
    /// The environment variable a refresh reads its bearer token from.
    pub bearer_env: Option<String>,
    /// The headers it sends, by name.
    pub header_names: Vec<String>,
    /// Whether its URL and header values are sealed in `control.db`.
    pub sealed_secret: bool,
    /// The document its rows loaded as last.
    pub document_id: Option<DocumentId>,
    pub created_by: Option<String>,
    pub created_at: String,
    pub last_run_at: Option<String>,
    pub last_rows: Option<u64>,
    pub last_error: Option<String>,
}

const IMPORT_COLUMNS: &str = "id, name, url, table_name, query, source_table, row_limit, types, \
     json_pointer, bearer_env, CAST(header_names AS VARCHAR), sealed_secret, document_id, \
     created_by, CAST(created_at AS VARCHAR), CAST(last_run_at AS VARCHAR), last_rows, last_error";

/// A row selected with `IMPORT_COLUMNS`.
impl TryFrom<&duckdb::Row<'_>> for SavedImport {
    type Error = Error;

    fn try_from(row: &duckdb::Row<'_>) -> Result<Self> {
        let header_names: Option<String> = row.get(10)?;
        Ok(Self {
            id: row.get(0)?,
            name: row.get(1)?,
            source: row.get(2)?,
            table: row.get(3)?,
            query: row.get(4)?,
            source_table: row.get(5)?,
            limit: row.get(6)?,
            types: row
                .get::<_, Option<String>>(7)?
                .map(|json| serde_json::from_str(&json))
                .transpose()?
                .unwrap_or_default(),
            json_pointer: row.get(8)?,
            bearer_env: row.get(9)?,
            header_names: header_names
                .map(|names| serde_json::from_str(&names))
                .transpose()?
                .unwrap_or_default(),
            sealed_secret: row.get(11)?,
            document_id: row.get(12)?,
            created_by: row.get(13)?,
            created_at: row.get(14)?,
            last_run_at: row.get(15)?,
            last_rows: row.get(16)?,
            last_error: row.get(17)?,
        })
    }
}

/// Whether to keep an import's secret (a URL password, header values) so
/// it can be refreshed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KeepSecret {
    /// Keep none: an import that has one cannot be saved.
    No,
    /// Seal it under the vault key in `control.db`.
    Sealed,
}

/// What a saved import cannot store in the workspace file: the URL with its
/// password, and the values of the headers it sends.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct Secret {
    url: String,
    headers: Vec<(String, String)>,
}

impl ImportRequest {
    /// The secret this request carries, if any: a password in its URL, or
    /// a header given with its value.
    fn secret(&self) -> Option<Secret> {
        let headers: Vec<(String, String)> = self
            .headers
            .iter()
            .filter_map(|header| match header {
                SourceHeader::Given { name, value } => value
                    .to_str()
                    .ok()
                    .map(|value| (name.as_str().to_owned(), value.to_owned())),
                SourceHeader::BearerEnv(_) => None,
            })
            .collect();
        let password = self.url.redacted() != self.url.expose();
        (password || !headers.is_empty()).then(|| Secret {
            url: self.url.expose().to_owned(),
            headers,
        })
    }

    /// Refuse to save an import whose secret would be lost: one with a URL
    /// password or a header value, unless it is kept sealed. Called before
    /// the import runs, so nothing loads that cannot be refreshed.
    ///
    /// # Errors
    ///
    /// Returns an ingestion error naming both ways to keep the secret.
    pub fn check_saveable(&self, keep: KeepSecret) -> Result<()> {
        if keep == KeepSecret::No && self.secret().is_some() {
            return Err(Error::Ingestion(String::from(
                "this import carries a password or a header value, which a saved import does \
                 not store; pass a token with --bearer-env, or keep it sealed with \
                 --store-credential",
            )));
        }
        Ok(())
    }
}

impl SavedImport {
    /// Every saved import, by name.
    ///
    /// # Errors
    ///
    /// Returns an error if the query fails.
    pub fn list(db: &WorkspaceDb) -> Result<Vec<Self>> {
        let sql = format!("SELECT {IMPORT_COLUMNS} FROM _quack_imports ORDER BY name");
        let mut stmt = db.connection().prepare(&sql)?;
        let mut rows = stmt.query([])?;
        let mut out = Vec::new();
        while let Some(row) = rows.next()? {
            out.push(Self::try_from(row)?);
        }
        Ok(out)
    }

    /// The saved import named `name`, or with that id.
    ///
    /// # Errors
    ///
    /// Returns `NotFound` when none matches, or a query error.
    pub fn named(db: &WorkspaceDb, name: &str) -> Result<Self> {
        let name = name.trim();
        let sql = format!("SELECT {IMPORT_COLUMNS} FROM _quack_imports WHERE name = ? OR id = ?");
        let mut stmt = db.connection().prepare(&sql)?;
        let mut rows = stmt.query(duckdb::params![name, name])?;
        rows.next()?
            .map(Self::try_from)
            .transpose()?
            .ok_or_else(|| ResourceKind::SavedImport.missing(name))
    }

    /// Refuse a name that is blank or already saved; checked before an
    /// import runs, so a taken name loads nothing.
    ///
    /// # Errors
    ///
    /// Returns `SavedImportExists` for a name in use, an ingestion error for
    /// a blank one, or a query error.
    pub fn check_name(db: &WorkspaceDb, name: &str) -> Result<()> {
        let name = name.trim();
        if name.is_empty() {
            return Err(Error::Ingestion(String::from(
                "a saved import needs a name",
            )));
        }
        match Self::named(db, name) {
            Ok(_) => Err(Error::SavedImportExists(name.to_owned())),
            Err(Error::NotFound { .. }) => Ok(()),
            Err(other) => Err(other),
        }
    }

    /// Save `request` under `name`, after a run that produced `summary`.
    /// The secret, when there is one, was sealed under `id` first.
    fn insert(
        db: &WorkspaceDb,
        id: &ImportId,
        name: &str,
        request: &ImportRequest,
        summary: &ImportSummary,
        sealed_secret: bool,
        created_by: Option<&str>,
    ) -> Result<Self> {
        let header_names: Vec<&str> = request.headers.iter().map(SourceHeader::name).collect();
        let bearer_env = request.headers.iter().find_map(|header| match header {
            SourceHeader::BearerEnv(name) => Some(name.as_str()),
            SourceHeader::Given { .. } => None,
        });
        let types = (!request.types.is_empty())
            .then(|| serde_json::to_string(&request.types))
            .transpose()?;
        db.connection().execute(
            "INSERT INTO _quack_imports (id, name, url, table_name, query, source_table, \
             row_limit, types, json_pointer, bearer_env, header_names, sealed_secret, \
             document_id, created_by, last_run_at, last_rows) \
             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, now(), ?)",
            duckdb::params![
                id,
                name,
                request.url.redacted(),
                summary.table,
                request.query,
                request.source_table,
                request.limit,
                types,
                request.json_pointer.as_ref().map(ToString::to_string),
                bearer_env,
                serde_json::to_string(&header_names)?,
                sealed_secret,
                summary.document_id,
                created_by,
                summary.rows,
            ],
        )?;
        Self::named(db, id.as_str())
    }

    /// Record how a refresh went: the document its rows now load as and
    /// how many there are, or why it failed (the rows before stay).
    ///
    /// # Errors
    ///
    /// Returns an error if the update fails.
    pub fn record_run(
        &self,
        db: &WorkspaceDb,
        outcome: std::result::Result<&ImportSummary, &str>,
    ) -> Result<()> {
        match outcome {
            Ok(summary) => db.connection().execute(
                "UPDATE _quack_imports SET document_id = ?, last_rows = ?, last_error = NULL, \
                 last_run_at = now() WHERE id = ?",
                duckdb::params![summary.document_id, summary.rows, self.id],
            )?,
            Err(error) => db.connection().execute(
                "UPDATE _quack_imports SET last_error = ?, last_run_at = now() WHERE id = ?",
                duckdb::params![error, self.id],
            )?,
        };
        Ok(())
    }

    /// The request a refresh runs: this import as saved, with its secret
    /// back when it has one, replacing the document it last loaded.
    fn request(&self, secret: Option<Secret>) -> Result<ImportRequest> {
        let (url, mut headers) = match secret {
            Some(secret) => (
                SourceUrl::from(secret.url),
                secret
                    .headers
                    .into_iter()
                    .map(|(name, value)| {
                        let name = HeaderName::try_from(name)
                            .map_err(|e| Error::Ingestion(format!("bad saved header: {e}")))?;
                        let mut value = HeaderValue::try_from(value).map_err(|_| {
                            Error::Ingestion(format!("bad saved value for header {name}"))
                        })?;
                        value.set_sensitive(true);
                        Ok(SourceHeader::Given { name, value })
                    })
                    .collect::<Result<Vec<_>>>()?,
            ),
            None => (SourceUrl::from(self.source.clone()), Vec::new()),
        };
        headers.extend(self.bearer_env.clone().map(SourceHeader::BearerEnv));
        Ok(ImportRequest {
            query: self.query.clone(),
            source_table: self.source_table.clone(),
            limit: self.limit,
            types: self.types.clone(),
            headers,
            json_pointer: self
                .json_pointer
                .as_deref()
                .map(str::parse::<JsonPointer>)
                .transpose()?,
            replaces: self.document_id.clone(),
            ..ImportRequest::new(url, self.table.clone())
        })
    }
}

/// What a refresh runs with besides the import itself.
pub struct RefreshWith<'a, M> {
    pub config: &'a Config,
    pub db: &'a Writer,
    pub policy: ImportPolicy,
    pub embedder: Option<&'a Embedder<M>>,
    pub control: RunControl<'a>,
}

/// Where a workspace's saved imports keep their sealed secrets: the vault
/// that seals them and the `control.db` rows that hold them.
pub struct ImportSecrets<'a> {
    pub control: &'a ControlPlane,
    pub vault: &'a Vault,
    pub workspace: &'a WorkspaceId,
}

impl ImportSecrets<'_> {
    /// Save `request`, which ran as `summary`, under `name`: its secret
    /// sealed first when `keep` says so, then the row in the workspace.
    ///
    /// # Errors
    ///
    /// Returns `SavedImportExists` for a name in use, the refusal of
    /// [`ImportRequest::check_saveable`], or a vault or storage error. A
    /// secret sealed for a row that then failed to write is removed again.
    pub async fn save(
        &self,
        db: &Writer,
        name: &str,
        request: &ImportRequest,
        summary: &ImportSummary,
        keep: KeepSecret,
        created_by: Option<&str>,
    ) -> Result<SavedImport> {
        let name = name.trim().to_owned();
        request.check_saveable(keep)?;
        {
            let name = name.clone();
            db.run(move |db| SavedImport::check_name(db, &name)).await?;
        }
        let id = ImportId::generate();
        let owner = SealedOwner::Import {
            workspace: self.workspace,
            import: &id,
        };
        let secret = request.secret().filter(|_| keep == KeepSecret::Sealed);
        if let Some(secret) = &secret {
            let sealed = self
                .vault
                .seal(
                    Purpose::ImportCredential,
                    id.as_str(),
                    &serde_json::to_vec(secret)?,
                )
                .await?;
            self.control.put_sealed(owner, &sealed).await?;
        }
        let saved = {
            let (id, request, summary) = (id.clone(), request.clone(), summary.clone());
            let created_by = created_by.map(str::to_owned);
            let sealed = secret.is_some();
            db.run(move |db| {
                SavedImport::insert(
                    db,
                    &id,
                    &name,
                    &request,
                    &summary,
                    sealed,
                    created_by.as_deref(),
                )
            })
            .await
        };
        // The save's own error is the one to report; a secret left behind
        // is logged, and a later save under the same id replaces it.
        if saved.is_err()
            && secret.is_some()
            && let Err(e) = self.control.delete_sealed(owner).await
        {
            tracing::warn!(error = %e, "the secret of an import that was not saved could not be removed");
        }
        saved
    }

    /// The request a refresh of `saved` runs, with its sealed secret opened.
    ///
    /// # Errors
    ///
    /// Returns an error when the sealed secret is missing or the vault key
    /// that sealed it is gone (save the import again), or a storage error.
    pub async fn request(&self, saved: &SavedImport) -> Result<ImportRequest> {
        if !saved.sealed_secret {
            return saved.request(None);
        }
        let lost = || {
            Error::Ingestion(format!(
                "the secret saved with import '{}' is gone (a restored or moved workspace, or a \
                 new vault key); remove it and save it again",
                saved.name
            ))
        };
        let owner = SealedOwner::Import {
            workspace: self.workspace,
            import: &saved.id,
        };
        let sealed = self.control.sealed(owner).await?.ok_or_else(lost)?;
        match self
            .vault
            .open(Purpose::ImportCredential, saved.id.as_str(), &sealed)
            .await?
        {
            Opened::Plaintext(bytes) => saved.request(Some(serde_json::from_slice(&bytes)?)),
            Opened::KeyGone => Err(lost()),
        }
    }

    /// Run `saved` again and record how it went: replaced when the source
    /// changed, left in place when it did not, and on a failure the rows
    /// before stay and the error is kept for `quack import list`.
    ///
    /// # Errors
    ///
    /// Returns the import's error, or a storage error recording the run.
    pub async fn refresh<M: EmbeddingModel>(
        &self,
        saved: &SavedImport,
        with: RefreshWith<'_, M>,
    ) -> Result<ImportSummary> {
        let outcome = match self.request(saved).await {
            Ok(request) => {
                Importing {
                    config: with.config,
                    db: with.db,
                    workspace_id: self.workspace.as_str(),
                    request: &request,
                    policy: with.policy,
                    embedder: with.embedder,
                    control: with.control,
                }
                .run()
                .await
            }
            Err(e) => Err(e),
        };
        let recorded = {
            let saved = saved.clone();
            let result = outcome
                .as_ref()
                .map(Clone::clone)
                .map_err(ToString::to_string);
            with.db
                .run(move |db| saved.record_run(db, result.as_ref().map_err(String::as_str)))
                .await
        };
        // The import's outcome is what happened to the table; a run that
        // could not be recorded is logged and does not change it.
        if let Err(e) = recorded {
            tracing::warn!(import = %saved.name, error = %e, "the refresh could not be recorded");
        }
        outcome
    }

    /// Remove `saved` and its sealed secret; its table stays.
    ///
    /// # Errors
    ///
    /// Returns an error if a delete fails.
    pub async fn remove(&self, db: &Writer, saved: &SavedImport) -> Result<()> {
        let id = saved.id.clone();
        db.run(move |db| {
            db.connection().execute(
                "DELETE FROM _quack_imports WHERE id = ?",
                duckdb::params![id],
            )?;
            Ok(())
        })
        .await?;
        self.control
            .delete_sealed(SealedOwner::Import {
                workspace: self.workspace,
                import: &saved.id,
            })
            .await
    }
}

#[cfg(test)]
mod tests;
