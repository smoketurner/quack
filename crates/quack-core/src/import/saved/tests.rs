use std::path::Path;

use super::*;
use crate::config::Config;
use crate::import::{ImportPolicy, Importing, LoadStatus};
use crate::llm::Embeddings;
use crate::llm::oauth::KeySource;
use crate::progress::RunControl;
use crate::storage::control::{AuditAction, AuditEntry, Channel, Outcome, WorkspaceName};

#[expect(clippy::panic, reason = "test failure path")]
fn fail(msg: &str) -> ! {
    panic!("{msg}")
}

/// A control plane with one workspace, its writer, and the vault, all under
/// `dir`.
struct Fixture {
    config: Config,
    control: ControlPlane,
    vault: Vault,
    workspace: WorkspaceId,
    db: Writer,
}

impl Fixture {
    async fn new(dir: &Path) -> Self {
        let mut config = Config::default();
        config.general.data_dir = dir.to_path_buf();
        let control = ControlPlane::open(&config)
            .await
            .unwrap_or_else(|e| fail(&e.to_string()));
        let name: WorkspaceName = "sales"
            .parse()
            .unwrap_or_else(|e: Error| fail(&e.to_string()));
        let row = control
            .create_workspace(
                &name,
                None,
                AuditEntry::new(AuditAction::Admin, Outcome::Allowed, Channel::Cli),
            )
            .await
            .unwrap_or_else(|e| fail(&e.to_string()));
        let db = WorkspaceDb::open(&config, row.id.as_str())
            .and_then(Writer::spawn)
            .unwrap_or_else(|e| fail(&e.to_string()));
        Self {
            vault: Vault::new(dir, KeySource::File),
            config,
            control,
            workspace: row.id,
            db,
        }
    }

    fn secrets(&self) -> ImportSecrets<'_> {
        ImportSecrets {
            control: &self.control,
            vault: &self.vault,
            workspace: &self.workspace,
        }
    }

    async fn import(&self, request: &ImportRequest) -> Result<ImportSummary> {
        Importing {
            config: &self.config,
            db: &self.db,
            workspace_id: self.workspace.as_str(),
            request,
            policy: ImportPolicy::owner(),
            embedder: None::<&Embeddings>,
            control: RunControl::unobserved(),
        }
        .run()
        .await
    }
}

/// A SQLite file outside the data directory with `rows` orders.
async fn source(dir: &Path, rows: &str) -> String {
    use sqlx::{Connection as _, Executor as _};
    let path = dir.join("shop.db");
    let mut conn =
        sqlx::SqliteConnection::connect(&format!("sqlite://{}?mode=rwc", path.display()))
            .await
            .unwrap_or_else(|e| fail(&e.to_string()));
    conn.execute("CREATE TABLE IF NOT EXISTS orders (id INTEGER, region TEXT)")
        .await
        .unwrap_or_else(|e| fail(&e.to_string()));
    conn.execute("DELETE FROM orders")
        .await
        .unwrap_or_else(|e| fail(&e.to_string()));
    let insert = format!("INSERT INTO orders VALUES {rows}");
    conn.execute(sqlx::query(sqlx::AssertSqlSafe(insert)))
        .await
        .unwrap_or_else(|e| fail(&e.to_string()));
    format!("sqlite:{}", path.display())
}

#[tokio::test]
async fn a_saved_import_refreshes_in_place_and_reports_an_unchanged_source() {
    let data = tempfile::tempdir().unwrap_or_else(|e| fail(&e.to_string()));
    let files = tempfile::tempdir().unwrap_or_else(|e| fail(&e.to_string()));
    let f = Fixture::new(data.path()).await;
    let url = source(files.path(), "(1, 'north'), (2, 'south')").await;
    let request = ImportRequest {
        source_table: Some(String::from("orders")),
        ..ImportRequest::new(url, "orders")
    };
    let first = f
        .import(&request)
        .await
        .unwrap_or_else(|e| fail(&e.to_string()));
    let saved = f
        .secrets()
        .save(
            &f.db,
            " nightly ",
            &request,
            &first,
            KeepSecret::No,
            Some("olive"),
        )
        .await
        .unwrap_or_else(|e| fail(&e.to_string()));
    assert_eq!(saved.name, "nightly");
    assert_eq!(saved.last_rows, Some(2));
    assert_eq!(saved.document_id.as_ref(), Some(&first.document_id));
    assert!(!saved.sealed_secret);
    let again = f
        .secrets()
        .save(&f.db, "nightly", &request, &first, KeepSecret::No, None)
        .await;
    assert!(matches!(again, Err(Error::SavedImportExists(name)) if name == "nightly"));

    // The same rows: nothing loads, and the document stays.
    let refresh = f
        .secrets()
        .request(&saved)
        .await
        .unwrap_or_else(|e| fail(&e.to_string()));
    assert_eq!(refresh.replaces.as_ref(), Some(&first.document_id));
    let unchanged = f
        .import(&refresh)
        .await
        .unwrap_or_else(|e| fail(&e.to_string()));
    assert_eq!(unchanged.status, LoadStatus::Unchanged);
    assert_eq!(unchanged.document_id, first.document_id);
    assert_eq!(unchanged.rows, 2);

    // New rows: a new document replaces the old, and the run is recorded.
    source(files.path(), "(1, 'north'), (2, 'south'), (3, 'west')").await;
    let loaded = f
        .import(&refresh)
        .await
        .unwrap_or_else(|e| fail(&e.to_string()));
    assert_eq!(loaded.status, LoadStatus::Loaded);
    assert_eq!(loaded.rows, 3);
    assert_ne!(loaded.document_id, first.document_id);
    let recorded = {
        let (saved, loaded) = (saved.clone(), loaded.clone());
        f.db.run(move |db| {
            saved.record_run(db, Ok(&loaded))?;
            SavedImport::named(db, "nightly")
        })
        .await
        .unwrap_or_else(|e| fail(&e.to_string()))
    };
    assert_eq!(recorded.last_rows, Some(3));
    assert_eq!(recorded.document_id.as_ref(), Some(&loaded.document_id));
    assert_eq!(recorded.last_error, None);
    let tables =
        f.db.run(WorkspaceDb::list_tables)
            .await
            .unwrap_or_else(|e| fail(&e.to_string()));
    assert_eq!(tables.iter().filter(|t| *t == "orders").count(), 1);

    // A failed run keeps the rows and says why.
    let failed = {
        let saved = recorded.clone();
        f.db.run(move |db| {
            saved.record_run(db, Err("the source did not answer"))?;
            SavedImport::named(db, saved.id.as_str())
        })
        .await
        .unwrap_or_else(|e| fail(&e.to_string()))
    };
    assert_eq!(
        failed.last_error.as_deref(),
        Some("the source did not answer")
    );
    assert_eq!(failed.last_rows, Some(3));

    f.secrets()
        .remove(&f.db, &failed)
        .await
        .unwrap_or_else(|e| fail(&e.to_string()));
    let gone = f.db.run(|db| SavedImport::named(db, "nightly")).await;
    assert!(gone.is_err());
    assert!(
        f.db.run(SavedImport::list)
            .await
            .unwrap_or_else(|e| fail(&e.to_string()))
            .is_empty()
    );
}

/// A request with a URL password, a header value, and a bearer variable.
fn with_secrets() -> ImportRequest {
    ImportRequest {
        headers: vec![
            "X-Api-Key: k-123"
                .parse()
                .unwrap_or_else(|e: Error| fail(&e.to_string())),
            SourceHeader::BearerEnv(String::from("SALES_TOKEN")),
        ],
        ..ImportRequest::new("https://alice:pa55@files.example.com/orders.csv", "orders")
    }
}

/// A password or header value cannot be stored in the workspace, so saving
/// one is refused unless it is sealed; a variable's name is not a secret.
#[test]
fn an_import_with_a_secret_is_saved_only_sealed() {
    let request = with_secrets();
    let refused = request
        .check_saveable(KeepSecret::No)
        .err()
        .map(|e| e.to_string())
        .unwrap_or_default();
    assert!(refused.contains("--store-credential"), "{refused}");
    assert!(refused.contains("--bearer-env"), "{refused}");
    assert!(
        ImportRequest {
            headers: vec![SourceHeader::BearerEnv(String::from("SALES_TOKEN"))],
            ..ImportRequest::new("https://files.example.com/orders.csv", "orders")
        }
        .check_saveable(KeepSecret::No)
        .is_ok(),
        "a variable's name is not a secret"
    );
    assert!(request.check_saveable(KeepSecret::Sealed).is_ok());
}

/// A sealed secret comes back whole for the refresh while the workspace
/// row holds only the redacted URL and the header's name, and it goes when
/// the import is removed.
#[tokio::test]
async fn a_sealed_secret_comes_back_for_a_refresh_and_goes_with_its_import() {
    let data = tempfile::tempdir().unwrap_or_else(|e| fail(&e.to_string()));
    let f = Fixture::new(data.path()).await;
    let request = with_secrets();
    let summary = ImportSummary {
        table: String::from("orders"),
        rows: 5,
        columns: vec![String::from("id")],
        source: request.url.redacted(),
        document_id: DocumentId::generate(),
        status: LoadStatus::Loaded,
    };
    let saved = f
        .secrets()
        .save(
            &f.db,
            "partner feed",
            &request,
            &summary,
            KeepSecret::Sealed,
            None,
        )
        .await
        .unwrap_or_else(|e| fail(&e.to_string()));
    assert!(saved.sealed_secret);
    assert_eq!(
        saved.source,
        "https://alice:***@files.example.com/orders.csv"
    );
    assert_eq!(saved.header_names, ["x-api-key", "authorization"]);
    assert_eq!(saved.bearer_env.as_deref(), Some("SALES_TOKEN"));
    let row =
        f.db.run(|db| {
            Ok(db.connection().query_row(
                "SELECT url || CAST(header_names AS VARCHAR) FROM _quack_imports",
                [],
                |row| row.get::<_, String>(0),
            )?)
        })
        .await
        .unwrap_or_else(|e| fail(&e.to_string()));
    assert!(!row.contains("pa55") && !row.contains("k-123"), "{row}");

    let refresh = f
        .secrets()
        .request(&saved)
        .await
        .unwrap_or_else(|e| fail(&e.to_string()));
    assert_eq!(
        refresh.url.expose(),
        "https://alice:pa55@files.example.com/orders.csv"
    );
    let sent: Vec<(String, String)> = refresh
        .headers
        .iter()
        .map(|header| match header {
            SourceHeader::Given { name, value } => (
                name.to_string(),
                value.to_str().unwrap_or_default().to_owned(),
            ),
            SourceHeader::BearerEnv(var) => (String::from("bearer"), var.clone()),
        })
        .collect();
    assert_eq!(
        sent,
        [
            (String::from("x-api-key"), String::from("k-123")),
            (String::from("bearer"), String::from("SALES_TOKEN"))
        ]
    );

    // Its sealed secret goes with it.
    f.secrets()
        .remove(&f.db, &saved)
        .await
        .unwrap_or_else(|e| fail(&e.to_string()));
    let owner = SealedOwner::Import {
        workspace: &f.workspace,
        import: &saved.id,
    };
    assert_eq!(
        f.control
            .sealed(owner)
            .await
            .unwrap_or_else(|e| fail(&e.to_string())),
        None
    );
    let lost = f.secrets().request(&saved).await;
    assert!(
        lost.err()
            .is_some_and(|e| e.to_string().contains("save it again"))
    );
}
