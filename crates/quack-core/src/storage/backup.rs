//! A workspace as one tar: `manifest.json` first, then `data.duckdb` and
//! everything under `files/`. The snapshot runs on the workspace's writer
//! thread after a `CHECKPOINT`, so the file it copies is complete and no
//! write lands between the checkpoint and the copy. A restore unpacks
//! into a new workspace's directory, which is then opened once through
//! `WorkspaceDb::open`, so schema upgrades run as for any older file.
//!
//! The manifest carries what `control.db` knows about the workspace that
//! the file does not: its name, classification, provider allow-list, and
//! members by username and role. It never carries tokens.

use std::io::{Read, Write};
use std::path::{Component, Path, PathBuf};
use std::sync::Arc;

use serde::{Deserialize, Serialize};

use super::audit::AuditDetail;
use super::control::{
    AuditAction, AuditEntry, ControlPlane, Outcome, ProviderAllowList, ResourceKind, Role,
    WorkspaceChanges, WorkspaceName, WorkspaceRow,
};
use super::workspace::{MetaKey, WorkspaceDb};
use crate::config::Config;
use crate::error::{Error, Result};
use crate::ids::UserId;

/// The tar layout this build writes and reads.
pub const FORMAT_VERSION: u32 = 1;
pub const MANIFEST: &str = "manifest.json";
pub const DATABASE: &str = "data.duckdb";
pub const FILES: &str = "files";

/// What a snapshot says about the workspace it holds.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Manifest {
    pub format: u32,
    pub quack_version: String,
    /// `_quack_meta.schema_version` of the file, when it had one.
    pub schema_version: Option<u32>,
    pub duckdb_version: String,
    /// The embedding profile's fingerprint the vectors were made under.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub embedding_profile: Option<String>,
    pub name: String,
    pub classification: String,
    /// Empty: every provider.
    #[serde(default)]
    pub allowed_providers: Vec<String>,
    #[serde(default)]
    pub members: Vec<ManifestMember>,
    pub taken_at: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ManifestMember {
    pub username: String,
    pub role: String,
}

/// What `control.db` knows about the workspace, for the manifest.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Described {
    pub name: String,
    pub classification: String,
    pub allowed_providers: Vec<String>,
    pub members: Vec<ManifestMember>,
}

impl Described {
    /// The workspace's row and its members, as the manifest carries them.
    ///
    /// # Errors
    ///
    /// Returns an error when the members cannot be read.
    pub async fn of(control: &ControlPlane, workspace: &WorkspaceRow) -> Result<Self> {
        let members = control
            .list_members(&workspace.id)
            .await?
            .into_iter()
            .map(|m| ManifestMember {
                username: m.username,
                role: m.role.to_string(),
            })
            .collect();
        Ok(Self {
            name: workspace.name.clone(),
            classification: workspace.classification.clone(),
            allowed_providers: workspace
                .allowed_providers
                .names()
                .map(|names| names.iter().cloned().collect())
                .unwrap_or_default(),
            members,
        })
    }
}

impl Manifest {
    /// This build's manifest for `db`, described as `control.db` has it.
    ///
    /// # Errors
    ///
    /// Returns an error if a read of the file's own record fails.
    pub fn of(db: &WorkspaceDb, described: Described) -> Result<Self> {
        Ok(Self {
            format: FORMAT_VERSION,
            quack_version: String::from(env!("CARGO_PKG_VERSION")),
            schema_version: db
                .meta(MetaKey::SchemaVersion)?
                .and_then(|v| v.parse().ok()),
            duckdb_version: db.duckdb_version()?,
            embedding_profile: db.embedding_fingerprint().map(|f| f.as_str().to_owned()),
            name: described.name,
            classification: described.classification,
            allowed_providers: described.allowed_providers,
            members: described.members,
            taken_at: jiff::Timestamp::now().to_string(),
        })
    }

    /// Whether this build can open the file the manifest describes.
    ///
    /// # Errors
    ///
    /// Returns an error naming the newer format or schema.
    pub fn check_readable(&self) -> Result<()> {
        if self.format > FORMAT_VERSION {
            return Err(Error::Snapshot(format!(
                "the snapshot is format {} (written by quack {}); this quack reads up to {FORMAT_VERSION}",
                self.format, self.quack_version
            )));
        }
        if let Some(schema) = self.schema_version
            && schema > WorkspaceDb::schema_version()
        {
            return Err(Error::Snapshot(format!(
                "the snapshot's workspace has schema version {schema} (written by quack {}); \
                 this quack writes {}; upgrade quack to restore it",
                self.quack_version,
                WorkspaceDb::schema_version()
            )));
        }
        Ok(())
    }
}

impl Manifest {
    /// Write this manifest, `workspace_dir`'s file, and its `files/` as a tar
    /// to `out`, after a checkpoint on `db`. Runs on the writer's thread.
    ///
    /// # Errors
    ///
    /// Returns an error if the checkpoint or a write fails.
    pub fn write<W: Write>(&self, db: &WorkspaceDb, workspace_dir: &Path, out: W) -> Result<W> {
        db.connection().execute_batch("CHECKPOINT")?;
        let mut builder = tar::Builder::new(out);
        let text = serde_json::to_vec_pretty(self)?;
        let mut header = tar::Header::new_gnu();
        header.set_size(u64::try_from(text.len()).unwrap_or(u64::MAX));
        header.set_mode(0o644);
        header.set_mtime(0);
        header.set_cksum();
        builder.append_data(&mut header, MANIFEST, text.as_slice())?;
        builder.append_path_with_name(workspace_dir.join(DATABASE), DATABASE)?;
        let files = workspace_dir.join(FILES);
        if files.is_dir() {
            builder.append_dir_all(FILES, &files)?;
        }
        builder.into_inner().map_err(Error::Io)
    }

    /// The manifest at the head of a snapshot.
    ///
    /// # Errors
    ///
    /// Returns an error when the bytes are not a snapshot.
    pub fn read<R: Read>(tar: R) -> Result<Self> {
        let mut archive = tar::Archive::new(tar);
        for entry in archive.entries().map_err(|e| Error::not_a_snapshot(&e))? {
            let mut entry = entry.map_err(|e| Error::not_a_snapshot(&e))?;
            if entry.path()?.to_string_lossy().trim_start_matches("./") == MANIFEST {
                let mut text = String::new();
                entry.read_to_string(&mut text)?;
                return Ok(serde_json::from_str(&text)?);
            }
        }
        Err(Error::Snapshot(String::from(
            "not a workspace snapshot: no manifest.json",
        )))
    }

    /// Unpack a snapshot into `workspace_dir`, which must not yet hold a
    /// database: `data.duckdb` and `files/`, nothing that escapes the
    /// directory. The manifest is returned.
    ///
    /// # Errors
    ///
    /// Returns an error when the bytes are not a snapshot, an entry's path
    /// leaves the directory, or a write fails.
    pub fn unpack<R: Read>(tar: R, workspace_dir: &Path) -> Result<Self> {
        if workspace_dir.join(DATABASE).exists() {
            return Err(Error::Snapshot(format!(
                "{} already holds a workspace file",
                workspace_dir.display()
            )));
        }
        std::fs::create_dir_all(workspace_dir)?;
        let mut archive = tar::Archive::new(tar);
        let mut manifest = None;
        for entry in archive.entries().map_err(|e| Error::not_a_snapshot(&e))? {
            let mut entry = entry.map_err(|e| Error::not_a_snapshot(&e))?;
            let path = entry.path()?.into_owned();
            let relative = EntryPath::try_from(path.as_path())?.0;
            if relative == Path::new(MANIFEST) {
                let mut text = String::new();
                entry.read_to_string(&mut text)?;
                manifest = Some(serde_json::from_str::<Self>(&text)?);
                continue;
            }
            let allowed = relative == Path::new(DATABASE) || relative.starts_with(FILES);
            if !allowed {
                tracing::warn!(entry = %relative.display(), "skipping an entry outside the snapshot layout");
                continue;
            }
            let target = workspace_dir.join(&relative);
            if entry.header().entry_type().is_dir() {
                std::fs::create_dir_all(&target)?;
                continue;
            }
            if !entry.header().entry_type().is_file() {
                continue;
            }
            if let Some(parent) = target.parent() {
                std::fs::create_dir_all(parent)?;
            }
            let mut file = std::fs::File::create(&target)?;
            std::io::copy(&mut entry, &mut file)?;
        }
        manifest.ok_or_else(|| {
            Error::Snapshot(String::from("not a workspace snapshot: no manifest.json"))
        })
    }
}

/// How a snapshot comes back as a workspace.
pub struct RestoreRequest<'a> {
    /// The new workspace's name; the manifest's when absent.
    pub name: Option<WorkspaceName>,
    /// Made an owner of the new workspace, beside the manifest's members.
    pub owner: Option<&'a UserId>,
    /// The entry each control-plane row of the restore is written as:
    /// who asked, and from where.
    pub audit: &'a (dyn Fn(AuditAction) -> AuditEntry + Sync),
}

/// What a restore made.
#[derive(Debug, Clone, Serialize)]
pub struct Restored {
    pub workspace: WorkspaceRow,
    pub manifest: Manifest,
    /// Manifest members given their role again.
    pub members_kept: usize,
    /// Manifest usernames this server has no user for.
    pub members_missing: Vec<String>,
    /// Allowed providers the manifest named that this server has not
    /// configured.
    pub providers_dropped: Vec<String>,
}

impl RestoreRequest<'_> {
    /// Restore a snapshot as a new workspace: the row, then the files
    /// unpacked into its directory, then its settings and members, then one
    /// open of the file, which runs any schema upgrade. A failure after the
    /// row exists deletes it and the directory again. `open` yields the tar
    /// bytes; it is read twice, for the manifest and for the files.
    ///
    /// # Errors
    ///
    /// Returns an error when the bytes are not a snapshot this build reads,
    /// the name is taken, or a write fails.
    pub async fn run<R, F>(
        self,
        control: &ControlPlane,
        config: &Config,
        open: F,
    ) -> Result<Restored>
    where
        R: Read,
        F: Fn() -> std::io::Result<R> + Send + Sync + 'static,
    {
        let manifest = Manifest::read(open()?)?;
        manifest.check_readable()?;
        let name = match &self.name {
            Some(name) => name.clone(),
            None => manifest.name.parse()?,
        };
        let workspace = control
            .create_workspace(&name, self.owner, (self.audit)(AuditAction::Workspace))
            .await?;
        let id = workspace.id.clone();
        match self
            .fill(control, config, &workspace, Arc::new(open), &manifest)
            .await
        {
            Ok(restored) => Ok(restored),
            Err(e) => {
                let undo = (self.audit)(AuditAction::Delete);
                let undo = AuditEntry {
                    outcome: Outcome::Error,
                    ..undo
                };
                if let Err(cleanup) = control.delete_workspace(&id, undo).await {
                    tracing::error!(workspace = %id, error = %cleanup, "a failed restore left its workspace row");
                }
                let dir = config.workspace_dir(id.as_str());
                if dir.exists()
                    && let Err(cleanup) = std::fs::remove_dir_all(&dir)
                {
                    tracing::error!(workspace = %id, error = %cleanup, "a failed restore left its directory");
                }
                Err(e)
            }
        }
    }

    /// Everything after the row exists; see [`Self::run`].
    async fn fill<R, F>(
        &self,
        control: &ControlPlane,
        config: &Config,
        workspace: &WorkspaceRow,
        open: Arc<F>,
        manifest: &Manifest,
    ) -> Result<Restored>
    where
        R: Read,
        F: Fn() -> std::io::Result<R> + Send + Sync + 'static,
    {
        let dir = config.workspace_dir(workspace.id.as_str());
        let unpack_dir = dir.clone();
        tokio::task::spawn_blocking(move || Manifest::unpack(open()?, &unpack_dir))
            .await
            .map_err(|e| {
                Error::Io(std::io::Error::other(format!(
                    "the restore's unpack task failed: {e}"
                )))
            })??;

        let (allowed, providers_dropped): (Vec<String>, Vec<String>) = manifest
            .allowed_providers
            .iter()
            .cloned()
            .partition(|name| config.providers.contains_key(name.as_str()));
        let allowed_providers = if manifest.allowed_providers.is_empty() {
            ProviderAllowList::All
        } else {
            ProviderAllowList::Only(allowed.into_iter().collect())
        };
        let workspace = control
            .update_workspace(
                &workspace.id,
                &WorkspaceChanges {
                    classification: Some(manifest.classification.clone()),
                    allowed_providers,
                },
            )
            .await?;

        let mut members_kept = 0_usize;
        let mut members_missing = Vec::new();
        for member in &manifest.members {
            let Some(user) = control.find_user_by_username(&member.username).await? else {
                members_missing.push(member.username.clone());
                continue;
            };
            let role: Role = member.role.parse()?;
            control
                .set_member(
                    &workspace.id,
                    &user.id,
                    role,
                    (self.audit)(AuditAction::Member),
                )
                .await?;
            members_kept = members_kept.saturating_add(1);
        }

        let entry = (self.audit)(AuditAction::Restore)
            .in_workspace(&workspace.id)
            .on(ResourceKind::Workspace.id(workspace.id.as_str()));
        let detail = AuditDetail {
            id: entry.id.clone(),
            user_id: entry.user_id.clone(),
            action: entry.action.to_string(),
            detail: serde_json::json!({
                "snapshot_taken_at": manifest.taken_at,
                "snapshot_quack_version": manifest.quack_version,
                "members_kept": members_kept,
                "members_missing": members_missing,
            }),
        };
        let open_config = config.clone();
        let open_id = workspace.id.clone();
        // One open, dropped before anyone else opens the file: the schema
        // upgrade runs here, and the detail row lands on the upgraded file.
        tokio::task::spawn_blocking(move || {
            let db = WorkspaceDb::open(&open_config, open_id.as_str())?;
            detail.write(&db)
        })
        .await
        .map_err(|e| {
            Error::Io(std::io::Error::other(format!(
                "the restore's open task failed: {e}"
            )))
        })??;
        control.record_audit(&entry).await?;
        Ok(Restored {
            workspace,
            manifest: manifest.clone(),
            members_kept,
            members_missing,
            providers_dropped,
        })
    }
}

/// A tar entry's path as a relative path with no `..` or root: the one
/// shape `Manifest::unpack` writes under the workspace directory.
struct EntryPath(PathBuf);

impl TryFrom<&Path> for EntryPath {
    type Error = Error;

    fn try_from(path: &Path) -> Result<Self> {
        let mut out = PathBuf::new();
        for component in path.components() {
            match component {
                Component::Normal(part) => out.push(part),
                Component::CurDir => {}
                Component::ParentDir | Component::RootDir | Component::Prefix(_) => {
                    return Err(Error::Snapshot(format!(
                        "the snapshot names a path outside the workspace: {}",
                        path.display()
                    )));
                }
            }
        }
        Ok(Self(out))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_manifest_refuses_newer_formats_and_schemas() {
        let mut manifest = Manifest {
            format: FORMAT_VERSION,
            quack_version: String::from("x"),
            schema_version: Some(WorkspaceDb::schema_version()),
            duckdb_version: String::from("v"),
            embedding_profile: None,
            name: String::from("ws"),
            classification: String::from("internal"),
            allowed_providers: Vec::new(),
            members: Vec::new(),
            taken_at: String::new(),
        };
        assert!(manifest.check_readable().is_ok());
        manifest.schema_version = Some(WorkspaceDb::schema_version().saturating_add(1));
        let newer = manifest
            .check_readable()
            .err()
            .map(|e| e.to_string())
            .unwrap_or_default();
        assert!(newer.contains("upgrade quack"), "{newer}");
        manifest.schema_version = None;
        manifest.format = FORMAT_VERSION.saturating_add(1);
        assert!(manifest.check_readable().is_err());
    }

    #[test]
    fn paths_that_escape_the_workspace_are_refused() {
        assert_eq!(
            EntryPath::try_from(Path::new("./files/a.csv"))
                .ok()
                .map(|p| p.0),
            Some(PathBuf::from("files/a.csv"))
        );
        assert!(EntryPath::try_from(Path::new("../etc/passwd")).is_err());
        assert!(EntryPath::try_from(Path::new("/etc/passwd")).is_err());
    }

    #[test]
    fn a_tar_without_a_manifest_is_not_a_snapshot() {
        let mut builder = tar::Builder::new(Vec::new());
        let mut header = tar::Header::new_gnu();
        header.set_size(2);
        header.set_cksum();
        builder
            .append_data(&mut header, "x.txt", b"hi".as_slice())
            .unwrap_or_else(|e| unreachable_tar(&e.to_string()));
        let bytes = builder
            .into_inner()
            .unwrap_or_else(|e| unreachable_tar(&e.to_string()));
        assert!(Manifest::read(bytes.as_slice()).is_err());
        let dir = tempfile::tempdir().unwrap_or_else(|e| unreachable_tar(&e.to_string()));
        assert!(Manifest::unpack(bytes.as_slice(), &dir.path().join("ws")).is_err());
    }

    /// The tar holds the manifest, the file, and the uploads; unpacked
    /// elsewhere, the file opens with its rows and the uploads in place.
    #[test]
    fn a_snapshot_unpacks_into_a_workspace_that_opens() {
        let dir = tempfile::tempdir().unwrap_or_else(|e| unreachable_tar(&e.to_string()));
        let mut config = Config::default();
        config.general.data_dir = dir.path().join("data");
        let db =
            WorkspaceDb::open(&config, "src").unwrap_or_else(|e| unreachable_tar(&e.to_string()));
        db.execute_query("CREATE TABLE t AS SELECT 42 AS a")
            .unwrap_or_else(|e| unreachable_tar(&e.to_string()));
        let files = config.workspace_files_dir("src");
        std::fs::create_dir_all(&files).unwrap_or_else(|e| unreachable_tar(&e.to_string()));
        std::fs::write(files.join("a.csv"), "a\n1\n")
            .unwrap_or_else(|e| unreachable_tar(&e.to_string()));
        let described = Described {
            name: String::from("src"),
            classification: String::from("internal"),
            allowed_providers: vec![String::from("local")],
            members: vec![ManifestMember {
                username: String::from("ann"),
                role: String::from("owner"),
            }],
        };
        let manifest =
            Manifest::of(&db, described).unwrap_or_else(|e| unreachable_tar(&e.to_string()));
        assert_eq!(manifest.schema_version, Some(WorkspaceDb::schema_version()));
        let tar = manifest
            .write(&db, &config.workspace_dir("src"), Vec::new())
            .unwrap_or_else(|e| unreachable_tar(&e.to_string()));
        drop(db);

        assert_eq!(
            Manifest::read(tar.as_slice()).ok().as_ref(),
            Some(&manifest)
        );
        let unpacked = Manifest::unpack(tar.as_slice(), &config.workspace_dir("dst"))
            .unwrap_or_else(|e| unreachable_tar(&e.to_string()));
        assert_eq!(unpacked, manifest);
        assert_eq!(
            std::fs::read_to_string(config.workspace_files_dir("dst").join("a.csv")).ok(),
            Some(String::from("a\n1\n"))
        );
        let restored =
            WorkspaceDb::open(&config, "dst").unwrap_or_else(|e| unreachable_tar(&e.to_string()));
        let rows = restored
            .execute_query("SELECT a FROM t")
            .unwrap_or_else(|e| unreachable_tar(&e.to_string()));
        assert_eq!(rows.rows, vec![vec![serde_json::json!(42)]]);
        // A second unpack into a directory that holds a file is refused.
        assert!(Manifest::unpack(tar.as_slice(), &config.workspace_dir("dst")).is_err());
    }

    #[expect(clippy::panic, reason = "test failure path")]
    fn unreachable_tar(msg: &str) -> ! {
        panic!("{msg}")
    }
}
