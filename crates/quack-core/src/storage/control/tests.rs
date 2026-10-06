use super::*;
use sqlx::AssertSqlSafe;

async fn open() -> (tempfile::TempDir, ControlPlane) {
    let dir = tempfile::tempdir().unwrap_or_else(|e| fail(&e.to_string()));
    let mut config = Config::default();
    config.general.data_dir = dir.path().to_path_buf();
    let cp = ControlPlane::open(&config)
        .await
        .unwrap_or_else(|e| fail(&e.to_string()));
    (dir, cp)
}

#[expect(clippy::panic, reason = "test failure path")]
fn fail(msg: &str) -> ! {
    panic!("{msg}")
}

/// The audit row a test's own setup writes.
fn workspace_name(name: &str) -> WorkspaceName {
    name.parse().unwrap_or_else(|e: Error| fail(&e.to_string()))
}

fn setup_audit() -> AuditEntry {
    AuditEntry::new(AuditAction::Admin, Outcome::Allowed, Channel::Cli)
}

/// A database from before sqlx migrations: the three sea-query versions
/// applied, progress recorded in `schema_version`, and rows in the tables
/// that replaying those versions would destroy.
async fn legacy_control_db(path: &std::path::Path) {
    let url = format!("sqlite:{}?mode=rwc", path.display());
    let options = SqliteConnectOptions::from_str(&url).unwrap_or_else(|e| fail(&e.to_string()));
    let pool = SqlitePoolOptions::new()
        .max_connections(1)
        .connect_with(options)
        .await
        .unwrap_or_else(|e| fail(&e.to_string()));

    let mut sql = String::from(
        "CREATE TABLE IF NOT EXISTS schema_version (\
             version INTEGER PRIMARY KEY, \
             applied_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP);",
    );
    for file in [
        "0001_access_control",
        "0002_audit_log_access_record",
        "0003_users_and_token_scopes",
    ] {
        sql.push_str(
            &std::fs::read_to_string(
                std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                    .join("migrations")
                    .join(format!("{file}.sql")),
            )
            .unwrap_or_else(|e| fail(&e.to_string())),
        );
    }
    sql.push_str("INSERT INTO schema_version (version) VALUES (1), (2), (3);");
    sql.push_str(
        "INSERT INTO audit_log (id, action, outcome, channel) \
         VALUES ('01890000-0000-7000-8000-000000000001', 'login', 'allowed', 'cli');",
    );
    sqlx::raw_sql(AssertSqlSafe(sql))
        .execute(&pool)
        .await
        .unwrap_or_else(|e| fail(&e.to_string()));
    pool.close().await;
}

/// Values are bound, not written into the SQL: a name full of quotes
/// and statement separators is stored and found as the text it is.
#[tokio::test]
async fn values_are_bound_not_spliced_into_the_sql() {
    let (_dir, cp) = open().await;
    let name = "o'brien\"; DROP TABLE users; --";
    let created = cp
        .create_workspace(&workspace_name(name), None, setup_audit())
        .await
        .unwrap_or_else(|e| fail(&e.to_string()));
    let found = cp
        .find_workspace_by_name(name)
        .await
        .unwrap_or_else(|e| fail(&e.to_string()));
    assert_eq!(found.map(|w| w.id), Some(created.id));
    assert!(cp.list_users().await.is_ok(), "users table still there");
}

#[test]
fn a_value_type_control_db_never_stores_is_refused() {
    let insert = Query::insert()
        .into_table(Users::Table)
        .columns([Users::Id])
        .values([1.5_f64.into()])
        .map_or_else(|e| fail(&e.to_string()), |q| q.to_owned());
    assert!(Bound::new(&insert).is_err());
}

#[tokio::test]
async fn legacy_schema_version_is_adopted_without_replaying_migrations() {
    let dir = tempfile::tempdir().unwrap_or_else(|e| fail(&e.to_string()));
    let mut config = Config::default();
    config.general.data_dir = dir.path().to_path_buf();
    config
        .ensure_dirs()
        .unwrap_or_else(|e| fail(&e.to_string()));
    legacy_control_db(&config.control_db_path()).await;

    let cp = ControlPlane::open(&config)
        .await
        .unwrap_or_else(|e| fail(&e.to_string()));

    assert!(
        cp.schema_version()
            .await
            .is_ok_and(|v| v == ControlPlane::latest_schema_version())
    );
    // Replaying v2 would have dropped and recreated audit_log.
    let rows: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM audit_log")
        .fetch_one(&cp.pool)
        .await
        .unwrap_or_else(|e| fail(&e.to_string()));
    assert_eq!(rows, 1, "the access record must survive adoption");
}

#[tokio::test]
async fn a_shipped_migration_may_not_change_under_an_existing_database() {
    let (dir, cp) = open().await;
    drop(cp);

    // Simulate an edit to a migration that has already been applied.
    sqlx::raw_sql("UPDATE _sqlx_migrations SET checksum = x'00' WHERE version = 1")
        .execute(
            &SqlitePoolOptions::new()
                .max_connections(1)
                .connect_with(
                    SqliteConnectOptions::from_str(&format!(
                        "sqlite:{}",
                        dir.path().join("control.db").display()
                    ))
                    .unwrap_or_else(|e| fail(&e.to_string())),
                )
                .await
                .unwrap_or_else(|e| fail(&e.to_string())),
        )
        .await
        .unwrap_or_else(|e| fail(&e.to_string()));

    let mut config = Config::default();
    config.general.data_dir = dir.path().to_path_buf();
    assert!(
        ControlPlane::open(&config).await.is_err(),
        "a changed checksum must refuse the database, not migrate it"
    );
}

#[test]
fn stored_allow_lists_decode_and_fail_closed() {
    assert_eq!(AllowedProviders::from_column(None), AllowedProviders::All);
    let only = AllowedProviders::from_column(Some("[\"ollama\"]"));
    assert!(only.permits("ollama") && !only.permits("openai"));
    let unreadable = AllowedProviders::from_column(Some("not json"));
    assert!(!unreadable.permits("ollama"));
    assert_eq!(
        serde_json::to_value(&only).ok(),
        Some(serde_json::json!(["ollama"]))
    );
    assert_eq!(
        serde_json::to_value(AllowedProviders::All).ok(),
        Some(serde_json::Value::Null)
    );
}

#[tokio::test]
async fn migrations_reach_the_latest_version_and_rerun_idempotently() {
    let (dir, cp) = open().await;
    assert!(
        cp.schema_version()
            .await
            .is_ok_and(|v| v == ControlPlane::latest_schema_version())
    );
    drop(cp);
    let mut config = Config::default();
    config.general.data_dir = dir.path().to_path_buf();
    let again = ControlPlane::open(&config).await;
    assert!(again.is_ok());
}

#[tokio::test]
async fn workspace_updates_keep_unset_fields() {
    let (_dir, cp) = open().await;
    let ws = cp
        .create_workspace(&workspace_name("w"), None, setup_audit())
        .await
        .unwrap_or_else(|e| fail(&e.to_string()));
    let changed = cp
        .update_workspace(
            &ws.id,
            &WorkspaceChanges {
                classification: Some(String::from("secret")),
                allowed_providers: ProviderAllowList::Only(BTreeSet::from([String::from(
                    "ollama",
                )])),
            },
        )
        .await;
    assert!(changed.is_ok_and(|w| {
        w.classification == "secret"
            && w.allowed_providers.permits("ollama")
            && !w.allowed_providers.permits("openai")
    }));
    let kept = cp
        .update_workspace(&ws.id, &WorkspaceChanges::default())
        .await;
    assert!(kept.is_ok_and(
        |w| w.classification == "secret" && w.allowed_providers != AllowedProviders::All
    ));
    let cleared = cp
        .update_workspace(
            &ws.id,
            &WorkspaceChanges {
                classification: None,
                allowed_providers: ProviderAllowList::All,
            },
        )
        .await;
    assert!(cleared.is_ok_and(|w| w.allowed_providers == AllowedProviders::All));
    assert!(
        cp.update_workspace(&WorkspaceId::from("missing"), &WorkspaceChanges::default())
            .await
            .is_err()
    );
    assert!(cp.get_workspace(&ws.id).await.is_ok_and(|w| w.is_some()));
    let taken = cp
        .create_workspace(&workspace_name("w"), None, setup_audit())
        .await;
    assert!(
        matches!(&taken, Err(Error::WorkspaceExists(name)) if name == "w"),
        "{taken:?}"
    );
    assert_eq!(
        taken.err().map(|e| e.to_string()).unwrap_or_default(),
        "workspace 'w' already exists"
    );
}

#[tokio::test]
async fn users_hash_verify_and_reject_duplicates() {
    let (_dir, cp) = open().await;
    let alice = cp
        .create_user("alice", "hunter42", UserKind::Admin, setup_audit())
        .await;
    assert!(
        alice
            .as_ref()
            .is_ok_and(|u| u.kind == UserKind::Admin && u.username == "alice")
    );
    assert!(
        cp.verify_password("alice", "hunter42")
            .await
            .is_ok_and(|u| u.is_some())
    );
    assert!(
        cp.verify_password("alice", "wrong")
            .await
            .is_ok_and(|u| u.is_none())
    );
    assert!(
        cp.verify_password("nobody", "hunter42")
            .await
            .is_ok_and(|u| u.is_none())
    );
    let dup = cp
        .create_user("alice", "x", UserKind::Standard, setup_audit())
        .await
        .err();
    assert!(dup.is_some_and(|e| e.to_string().contains("already exists")));
    assert!(
        cp.create_user("", "x", UserKind::Standard, setup_audit())
            .await
            .is_err()
    );
    assert!(
        cp.create_user("bob", "", UserKind::Standard, setup_audit())
            .await
            .is_err()
    );
    assert!(cp.list_users().await.is_ok_and(|u| u.len() == 1));
    assert!(
        cp.find_user_by_username(" alice ")
            .await
            .is_ok_and(|u| u.is_some())
    );
}

#[tokio::test]
async fn a_first_sign_in_creates_a_plain_user_and_never_takes_a_name() {
    let (_dir, cp) = open().await;
    let existing = cp
        .create_user("ada", "pw", UserKind::Admin, setup_audit())
        .await;
    assert!(existing.is_ok());
    let ada = OidcSubject::from("sub-ada");

    let first = cp.oidc_user(&ada, "ada").await;
    let Ok(first) = first else {
        fail(&format!("{:?}", first.err()));
    };
    assert!(
        first.username.starts_with("ada-") && first.username.len() == 12,
        "{}",
        first.username
    );
    assert_eq!(first.kind, UserKind::Standard);
    assert!(
        cp.workspaces_for_user(&first.id)
            .await
            .is_ok_and(|w| w.is_empty())
    );
    assert!(
        cp.verify_password(&first.username, "")
            .await
            .is_ok_and(|u| u.is_none())
    );

    let again = cp.oidc_user(&ada, "renamed-at-the-issuer").await;
    assert!(again.is_ok_and(|u| u.id == first.id && u.username == first.username));
    assert!(
        cp.find_user_by_oidc_subject(&ada)
            .await
            .is_ok_and(|u| u.is_some_and(|u| u.id == first.id))
    );

    let grace = cp
        .oidc_user(&OidcSubject::from("sub-grace"), " grace ")
        .await;
    assert!(grace.is_ok_and(|u| u.username == "grace"));
    assert!(
        cp.oidc_user(&OidcSubject::from("sub-x"), "  ")
            .await
            .is_err()
    );
    assert!(
        cp.find_user_by_oidc_subject(&OidcSubject::from("nobody"))
            .await
            .is_ok_and(|u| u.is_none())
    );
}

#[tokio::test]
async fn a_sealed_token_is_replaced_in_place_and_goes_with_its_user() {
    let (_dir, cp) = open().await;
    let user = cp
        .oidc_user(&OidcSubject::from("s"), "ada")
        .await
        .unwrap_or_else(|e| fail(&e.to_string()));
    let sealed = |id: &str| Sealed {
        key_id: id.to_owned(),
        enc: vec![4, 1, 2],
        ciphertext: vec![9, 8, 7],
    };
    assert!(
        cp.put_sealed(SealedOwner::User(&user.id), &sealed("k1"))
            .await
            .is_ok()
    );
    assert!(
        cp.put_sealed(SealedOwner::User(&user.id), &sealed("k2"))
            .await
            .is_ok()
    );
    assert!(
        cp.sealed(SealedOwner::User(&user.id))
            .await
            .is_ok_and(|t| t == Some(sealed("k2")))
    );
    assert!(
        cp.put_sealed(SealedOwner::User(&UserId::from("nobody")), &sealed("k1"))
            .await
            .is_err()
    );
    sqlx::query("DELETE FROM users")
        .execute(&cp.pool)
        .await
        .unwrap_or_else(|e| fail(&e.to_string()));
    assert!(
        cp.sealed(SealedOwner::User(&user.id))
            .await
            .is_ok_and(|t| t.is_none())
    );
    assert!(cp.delete_sealed(SealedOwner::User(&user.id)).await.is_ok());
}

#[tokio::test]
#[expect(clippy::too_many_lines, reason = "one registration's life, end to end")]
async fn a_registration_is_kept_with_its_key_and_forgotten_with_it() {
    let (_dir, cp) = open().await;
    let sealed = |id: &str| Sealed {
        key_id: id.to_owned(),
        enc: vec![1],
        ciphertext: vec![2, 3],
    };
    let key = sealed("key");
    let mut row = RegistrationRow {
        name: String::from("https://i"),
        client_id: String::from("c1"),
        registration_client_uri: Some(String::from("https://i/register/c1")),
        token: Some(sealed("t1")),
    };
    assert!(
        cp.registration("https://i")
            .await
            .is_ok_and(|r| r.is_none())
    );
    assert!(
        cp.add_sealed(SealedOwner::ClientKey("https://i"), &sealed("pending"))
            .await
            .is_ok()
    );
    // The pending key moves to the client's name with the registration.
    assert!(
        cp.save_registration(
            &row,
            Previous::Nothing,
            KeyChange {
                put: Some(("https://i c1", &key)),
                delete: &["https://i"],
            },
        )
        .await
        .is_ok_and(|saved| saved)
    );
    // A second new registration at the same name loses the race and
    // changes nothing, keys included.
    let rival = RegistrationRow {
        client_id: String::from("c9"),
        ..row.clone()
    };
    assert!(
        cp.save_registration(
            &rival,
            Previous::Nothing,
            KeyChange {
                put: Some(("https://i c9", &sealed("rival key"))),
                delete: &[],
            },
        )
        .await
        .is_ok_and(|saved| !saved)
    );
    assert!(
        cp.sealed(SealedOwner::ClientKey("https://i c9"))
            .await
            .is_ok_and(|k| k.is_none())
    );
    // An update names the client it expects, and misses when another
    // one is registered there.
    assert!(
        cp.save_registration(&rival, Previous::Client("c9"), KeyChange::default())
            .await
            .is_ok_and(|saved| !saved)
    );
    assert!(
        cp.registration("https://i")
            .await
            .is_ok_and(|r| r.as_ref() == Some(&row))
    );
    assert!(
        cp.sealed(SealedOwner::ClientKey("https://i c1"))
            .await
            .is_ok_and(|k| k == Some(key.clone()))
    );
    assert!(
        cp.sealed(SealedOwner::ClientKey("https://i"))
            .await
            .is_ok_and(|k| k.is_none())
    );
    // An issuer without RFC 7592 leaves nothing to manage it with.
    row.token = None;
    row.registration_client_uri = None;
    assert!(
        cp.save_registration(&row, Previous::Client("c1"), KeyChange::default())
            .await
            .is_ok_and(|saved| saved)
    );
    assert!(
        cp.registration("https://i")
            .await
            .is_ok_and(|r| r.as_ref() == Some(&row))
    );
    assert!(
        cp.delete_registration(
            "https://i",
            KeyChange {
                put: None,
                delete: &["https://i c1"],
            },
        )
        .await
        .is_ok()
    );
    assert!(
        cp.registration("https://i")
            .await
            .is_ok_and(|r| r.is_none())
    );
    assert!(
        cp.sealed(SealedOwner::ClientKey("https://i c1"))
            .await
            .is_ok_and(|k| k.is_none())
    );
}

#[tokio::test]
async fn a_client_key_row_is_added_once_replaced_in_place_and_deleted_by_name() {
    let (_dir, cp) = open().await;
    let sealed = |id: &str| Sealed {
        key_id: id.to_owned(),
        enc: vec![1],
        ciphertext: vec![2, 3],
    };
    let owner = SealedOwner::ClientKey("https://i c");
    assert!(
        cp.add_sealed(owner, &sealed("k1"))
            .await
            .is_ok_and(|kept| kept)
    );
    // A second process's key loses to the first.
    assert!(
        cp.add_sealed(owner, &sealed("k2"))
            .await
            .is_ok_and(|kept| !kept)
    );
    assert!(
        cp.sealed(owner)
            .await
            .is_ok_and(|t| t == Some(sealed("k1")))
    );
    assert!(cp.put_sealed(owner, &sealed("k3")).await.is_ok());
    assert!(
        cp.sealed(owner)
            .await
            .is_ok_and(|t| t == Some(sealed("k3")))
    );
    // Another client's key is another row.
    assert!(
        cp.sealed(SealedOwner::ClientKey("https://i other"))
            .await
            .is_ok_and(|t| t.is_none())
    );
    assert!(cp.delete_sealed(owner).await.is_ok());
    assert!(cp.sealed(owner).await.is_ok_and(|t| t.is_none()));
}

#[tokio::test]
async fn a_client_key_moves_to_another_name_in_one_step() {
    let (_dir, cp) = open().await;
    let sealed = |id: &str| Sealed {
        key_id: id.to_owned(),
        enc: vec![1],
        ciphertext: vec![2, 3],
    };
    let (current, next) = ("https://i c", "next https://i c");
    assert!(
        cp.add_sealed(SealedOwner::ClientKey(current), &sealed("old"))
            .await
            .is_ok()
    );
    assert!(
        cp.add_sealed(SealedOwner::ClientKey(next), &sealed("new"))
            .await
            .is_ok()
    );
    // The replacement, resealed for the client's name, takes its place.
    assert!(
        cp.change_client_keys(KeyChange {
            put: Some((current, &sealed("new for c"))),
            delete: &[next],
        })
        .await
        .is_ok()
    );
    assert!(
        cp.sealed(SealedOwner::ClientKey(current))
            .await
            .is_ok_and(|t| t == Some(sealed("new for c")))
    );
    assert!(
        cp.sealed(SealedOwner::ClientKey(next))
            .await
            .is_ok_and(|t| t.is_none())
    );
}

#[tokio::test]
async fn members_need_a_user_and_a_workspace() {
    let (_dir, cp) = open().await;
    let ws = cp
        .create_workspace(&workspace_name("w"), None, setup_audit())
        .await
        .unwrap_or_else(|e| fail(&e.to_string()));
    let bob = cp
        .create_user("bob", "pw", UserKind::Standard, setup_audit())
        .await
        .unwrap_or_else(|e| fail(&e.to_string()));
    assert!(
        cp.set_member(&ws.id, &UserId::from("ghost"), Role::Member, setup_audit())
            .await
            .is_err()
    );
    assert!(
        cp.set_member(&ws.id, &bob.id, Role::Viewer, setup_audit())
            .await
            .is_ok()
    );
    assert!(
        cp.member_role(&ws.id, &bob.id)
            .await
            .is_ok_and(|r| r == Some(Role::Viewer))
    );
    assert!(
        cp.set_member(&ws.id, &bob.id, Role::Owner, setup_audit())
            .await
            .is_ok()
    );
    assert!(
        cp.member_role(&ws.id, &bob.id)
            .await
            .is_ok_and(|r| r == Some(Role::Owner))
    );
    let listed = cp.list_members(&ws.id).await;
    assert!(listed.is_ok_and(|m| m.len() == 1 && m.first().is_some_and(|m| m.username == "bob")));
    let mine = cp.workspaces_for_user(&bob.id).await;
    assert!(mine.is_ok_and(|w| {
        w.len() == 1
            && w.first()
                .is_some_and(|m| m.workspace.name == "w" && m.role == Role::Owner)
    }));
    assert!(
        cp.remove_member(&ws.id, &bob.id, setup_audit())
            .await
            .is_ok_and(|removed| removed)
    );
    assert!(
        cp.remove_member(&ws.id, &bob.id, setup_audit())
            .await
            .is_ok_and(|removed| !removed)
    );
    assert!(Role::Viewer < Role::Member && Role::Member < Role::Owner);
    assert!("boss".parse::<Role>().is_err());
}

#[tokio::test]
async fn an_audited_change_names_what_it_changed_and_a_no_op_is_an_error() {
    let (_dir, cp) = open().await;
    let ws = cp
        .create_workspace(&workspace_name("w"), None, setup_audit())
        .await
        .unwrap_or_else(|e| fail(&e.to_string()));
    let bob = cp
        .create_user("bob", "pw", UserKind::Standard, setup_audit())
        .await
        .unwrap_or_else(|e| fail(&e.to_string()));
    let issued = cp
        .create_token(&ws.id, &bob.id, "t", &[], None, setup_audit())
        .await
        .unwrap_or_else(|e| fail(&e.to_string()));
    let removed = cp
        .remove_member(&ws.id, &bob.id, setup_audit())
        .await
        .unwrap_or_else(|e| fail(&e.to_string()));
    assert!(!removed);
    let rows = cp
        .query_audit(&AuditFilter {
            limit: 100,
            ..AuditFilter::default()
        })
        .await
        .unwrap_or_else(|e| fail(&e.to_string()))
        .rows;
    let named = |id: &str| rows.iter().find(|r| r.resource_id.as_deref() == Some(id));
    assert!(
        named(ws.id.as_str()).is_some_and(
            |r| r.workspace_id.as_ref() == Some(&ws.id) && r.outcome == Outcome::Allowed
        ),
        "{rows:?}"
    );
    assert!(
        named(&issued.row.token_hash).is_some_and(|r| r.workspace_id.as_ref() == Some(&ws.id)),
        "{rows:?}"
    );
    let bob_rows: Vec<_> = rows
        .iter()
        .filter(|r| r.resource_id.as_deref() == Some(bob.id.as_str()))
        .map(|r| r.outcome)
        .collect();
    assert_eq!(bob_rows, [Outcome::Error, Outcome::Allowed], "{rows:?}");
}

#[tokio::test]
async fn an_audited_change_does_not_stand_without_its_audit_row() {
    let (_dir, cp) = open().await;
    let ws = cp
        .create_workspace(&workspace_name("w"), None, setup_audit())
        .await
        .unwrap_or_else(|e| fail(&e.to_string()));
    let bob = cp
        .create_user("bob", "pw", UserKind::Standard, setup_audit())
        .await
        .unwrap_or_else(|e| fail(&e.to_string()));
    cp.set_member(&ws.id, &bob.id, Role::Viewer, setup_audit())
        .await
        .unwrap_or_else(|e| fail(&e.to_string()));
    let issued = cp
        .create_token(&ws.id, &bob.id, "t", &[], None, setup_audit())
        .await
        .unwrap_or_else(|e| fail(&e.to_string()));
    sqlx::query("DROP TABLE audit_log")
        .execute(&cp.pool)
        .await
        .unwrap_or_else(|e| fail(&e.to_string()));

    assert!(
        cp.create_workspace(&workspace_name("w2"), Some(&bob.id), setup_audit())
            .await
            .is_err()
    );
    assert!(
        cp.find_workspace_by_name("w2")
            .await
            .is_ok_and(|w| w.is_none())
    );
    assert!(
        cp.create_user("carol", "pw", UserKind::Standard, setup_audit())
            .await
            .is_err()
    );
    assert!(
        cp.find_user_by_username("carol")
            .await
            .is_ok_and(|u| u.is_none())
    );
    assert!(
        cp.set_member(&ws.id, &bob.id, Role::Owner, setup_audit())
            .await
            .is_err()
    );
    assert!(
        cp.remove_member(&ws.id, &bob.id, setup_audit())
            .await
            .is_err()
    );
    assert!(
        cp.member_role(&ws.id, &bob.id)
            .await
            .is_ok_and(|r| r == Some(Role::Viewer))
    );
    assert!(
        cp.create_token(&ws.id, &bob.id, "t2", &[], None, setup_audit())
            .await
            .is_err()
    );
    assert!(
        cp.delete_token(&issued.row.token_hash, setup_audit())
            .await
            .is_err()
    );
    assert!(
        cp.list_tokens(&ws.id)
            .await
            .is_ok_and(|t| t.len() == 1 && t.first().is_some_and(|t| t.name == "t"))
    );
}

#[tokio::test]
async fn tokens_are_stored_hashed_with_scopes_and_expiry() {
    let (_dir, cp) = open().await;
    let ws = cp
        .create_workspace(&workspace_name("w"), None, setup_audit())
        .await
        .unwrap_or_else(|e| fail(&e.to_string()));
    let bob = cp
        .create_user("bob", "pw", UserKind::Standard, setup_audit())
        .await
        .unwrap_or_else(|e| fail(&e.to_string()));
    let minted = cp
        .create_token(
            &ws.id,
            &bob.id,
            "ci",
            &[Scope::Read, Scope::Write],
            "2000-01-01 00:00:00".parse().ok(),
            setup_audit(),
        )
        .await;
    let Ok(IssuedToken { secret, row }) = minted else {
        fail("token creation failed");
    };
    assert!(secret.expose().starts_with("qk_"));
    assert!(!format!("{secret:?}").contains("qk_"));
    assert_eq!(row.token_hash, sha256_hex(secret.expose().as_bytes()));
    assert!(row.has_scope(Scope::Write) && !row.has_scope(Scope::Admin));
    let at = |text: &str| {
        text.parse::<Expiry>()
            .map_or_else(|e| fail(&e.to_string()), |at| at.0)
    };
    assert!(row.is_expired(at("2001-01-01 00:00:00")));
    assert!(!row.is_expired(at("1999-01-01 00:00:00")));
    assert_eq!(
        at("2000-01-01 00:00:00").to_string(),
        "2000-01-01T00:00:00Z",
        "stored expiries are UTC"
    );
    let unreadable = TokenRow {
        expires_at: Some(String::from("soon")),
        ..row.clone()
    };
    assert!(unreadable.is_expired(at("1999-01-01 00:00:00")));
    assert!(
        cp.find_token(&row.token_hash)
            .await
            .is_ok_and(|t| t.is_some())
    );
    assert!(cp.touch_token(&row.token_hash).await.is_ok());
    assert!(
        cp.find_token(&row.token_hash)
            .await
            .is_ok_and(|t| t.is_some_and(|t| t.last_used_at.is_some()))
    );
    assert!(cp.list_tokens(&ws.id).await.is_ok_and(|t| t.len() == 1));
    assert!(
        cp.create_token(
            &WorkspaceId::from("nope"),
            &bob.id,
            "x",
            &[],
            None,
            setup_audit()
        )
        .await
        .is_err()
    );
    assert!(
        cp.delete_token(&row.token_hash, setup_audit())
            .await
            .is_ok_and(|d| d)
    );
    assert!(
        cp.find_token(&row.token_hash)
            .await
            .is_ok_and(|t| t.is_none())
    );
    assert!("root".parse::<Scope>().is_err());
}

#[tokio::test]
async fn audit_rows_append_and_filter() {
    let (_dir, cp) = open().await;
    let mut allowed = AuditEntry::new(AuditAction::Open, Outcome::Allowed, Channel::Api)
        .in_workspace(&WorkspaceId::from("w1"));
    allowed.user_id = Some(UserId::from("u1"));
    let mut denied = AuditEntry::new(AuditAction::Open, Outcome::Denied, Channel::Web)
        .in_workspace(&WorkspaceId::from("w1"));
    denied.user_id = Some(UserId::from("u2"));
    let login = AuditEntry::new(AuditAction::Login, Outcome::Error, Channel::Web);
    for e in [&allowed, &denied, &login] {
        assert!(cp.record_audit(e).await.is_ok());
    }
    let all = cp
        .query_audit(&AuditFilter {
            limit: 10,
            ..AuditFilter::default()
        })
        .await
        .map(|page| page.rows);
    assert!(all.is_ok_and(|r| r.len() == 3 && r.first().is_some_and(|r| r.action == "login")));
    let denied_only = cp
        .query_audit(&AuditFilter {
            outcome: Some(Outcome::Denied),
            limit: 10,
            ..AuditFilter::default()
        })
        .await
        .map(|page| page.rows);
    assert!(denied_only.is_ok_and(|r| {
        r.len() == 1
            && r.first()
                .is_some_and(|r| r.user_id == Some(UserId::from("u2")))
    }));
    let for_ws = cp
        .query_audit(&AuditFilter {
            workspace_id: Some(WorkspaceId::from("w1")),
            user_id: Some(UserId::from("u1")),
            limit: 10,
            ..AuditFilter::default()
        })
        .await
        .map(|page| page.rows);
    assert!(for_ws.is_ok_and(|r| r.len() == 1 && r.first().is_some_and(|r| r.id == allowed.id)));
    let none = cp
        .query_audit(&AuditFilter {
            until: Some(String::from("1990-01-01")),
            limit: 10,
            ..AuditFilter::default()
        })
        .await
        .map(|page| page.rows);
    assert!(none.is_ok_and(|r| r.is_empty()));
}

/// A new workspace's name is trimmed, and refused when empty or when it
/// holds a slash, a backslash, or a dot.
#[test]
fn a_workspace_name_is_trimmed_and_refuses_empty_slashes_and_dots() {
    assert!(matches!(" sales ".parse::<WorkspaceName>(), Ok(name) if name.as_str() == "sales"));
    for refused in ["", "   ", "a/b", "a\\b", "a.b", ".."] {
        assert!(
            matches!(
                refused.parse::<WorkspaceName>(),
                Err(Error::InvalidWorkspaceName)
            ),
            "{refused:?}"
        );
    }
    assert!(WorkspaceName::try_from(String::from("a.b/c")).is_err());
}

/// Commands that use the default workspace first at the same moment,
/// from separate processes, all get the one row: whoever loses the
/// insert opens the winner's.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[expect(clippy::unwrap_used, reason = "test setup")]
async fn first_uses_of_the_default_workspace_that_race_share_one_row() {
    let default = WorkspaceName::default();
    let (dir, first) = open().await;
    let mut config = Config::default();
    config.general.data_dir = dir.path().to_path_buf();
    let mut users = vec![first];
    for _ in 0..7 {
        users.push(ControlPlane::open(&config).await.unwrap());
    }
    let mut racing = tokio::task::JoinSet::new();
    for cp in users {
        let default = default.clone();
        racing.spawn(async move { cp.workspace_or_default(None, &default).await });
    }
    let mut ids = BTreeSet::new();
    while let Some(opened) = racing.join_next().await {
        let opened = opened.unwrap();
        assert!(opened.is_ok(), "{opened:?}");
        ids.extend(opened.ok().map(|ws| ws.id.into_string()));
    }
    assert_eq!(ids.len(), 1, "{ids:?}");
}

/// A workspace a command names must exist, and nothing is created when
/// it does not; the default one is created, audited, by its first use,
/// whether or not the command names it.
#[tokio::test]
#[expect(clippy::unwrap_used, reason = "test setup")]
async fn a_named_workspace_must_exist_and_the_default_is_created_on_first_use() {
    let default = WorkspaceName::default();
    let entry = || AuditEntry::new(AuditAction::Workspace, Outcome::Allowed, Channel::Cli);

    let (_dir, cp) = open().await;
    let refused = cp
        .workspace_or_default(Some("slaes"), &default)
        .await
        .unwrap_err();
    assert_eq!(
        refused.to_string(),
        "no workspace named 'slaes'; create it with: quack workspace create slaes"
    );
    assert!(matches!(refused, Error::NoWorkspaceNamed(_)));
    assert!(cp.list_workspaces().await.unwrap().is_empty());
    assert!(matches!(
        cp.workspace_named("slaes").await,
        Err(Error::NoWorkspaceNamed(_))
    ));

    let created = cp.workspace_or_default(None, &default).await.unwrap();
    assert_eq!(created.name, "default");
    let audited = cp
        .query_audit(&AuditFilter {
            workspace_id: Some(created.id.clone()),
            limit: 10,
            ..AuditFilter::default()
        })
        .await
        .unwrap()
        .rows;
    assert!(
        matches!(audited.as_slice(), [row] if row.channel == Channel::Cli),
        "{audited:?}"
    );
    let again = cp
        .workspace_or_default(Some("default"), &default)
        .await
        .unwrap();
    assert_eq!(again.id, created.id);

    let sales = cp
        .create_workspace(&workspace_name("sales"), None, entry())
        .await
        .unwrap();
    let found = cp
        .workspace_or_default(Some("sales"), &default)
        .await
        .unwrap();
    assert_eq!(found.id, sales.id);
    assert_eq!(cp.list_workspaces().await.unwrap().len(), 2);

    // Naming the default explicitly creates it too.
    let (_dir, fresh) = open().await;
    let named = fresh
        .workspace_or_default(Some("default"), &default)
        .await
        .unwrap();
    assert_eq!(named.name, "default");
}

#[tokio::test]
#[expect(clippy::unwrap_used, reason = "test setup")]
async fn workspace_times_carry_the_newest_access_of_each_workspace() {
    let (_dir, cp) = open().await;
    let used = cp
        .create_workspace(&workspace_name("used"), None, setup_audit())
        .await
        .unwrap();
    // A workspace from before every creation was audited has no rows.
    let idle = WorkspaceId::generate();
    sqlx::query("INSERT INTO workspaces (id, name, classification) VALUES (?, 'idle', 'internal')")
        .bind(idle.as_str())
        .execute(&cp.pool)
        .await
        .unwrap();
    let denied =
        AuditEntry::new(AuditAction::Open, Outcome::Denied, Channel::Api).in_workspace(&used.id);
    cp.record_audit(&denied).await.unwrap();
    let mut times = cp.workspace_times().await.unwrap();
    assert_eq!(times.len(), 2);
    let newest = cp
        .query_audit(&AuditFilter {
            workspace_id: Some(used.id.clone()),
            limit: 1,
            ..AuditFilter::default()
        })
        .await
        .unwrap()
        .rows;
    let used = times.remove(&used.id).unwrap();
    assert_eq!(
        used.last_accessed_at.as_deref(),
        newest.first().map(|r| r.timestamp.as_str())
    );
    assert!(!used.created_at.is_empty());
    let idle = times.remove(&idle).unwrap();
    assert_eq!(idle.last_accessed_at, None);
    assert!(!idle.created_at.is_empty());
}

#[tokio::test]
async fn audit_pages_walk_the_log_once_in_order() {
    let (_dir, cp) = open().await;
    let mut written = Vec::new();
    for _ in 0..7 {
        let entry = AuditEntry::new(AuditAction::Open, Outcome::Allowed, Channel::Api)
            .in_workspace(&WorkspaceId::from("w1"));
        assert!(cp.record_audit(&entry).await.is_ok());
        written.push(entry.id);
    }
    let other = AuditEntry::new(AuditAction::Login, Outcome::Error, Channel::Web);
    assert!(cp.record_audit(&other).await.is_ok());

    let filter = AuditFilter {
        workspace_id: Some(WorkspaceId::from("w1")),
        limit: 3,
        ..AuditFilter::default()
    };
    let mut seen = Vec::new();
    let mut pages = 0;
    let mut after = None;
    loop {
        let page = cp
            .query_audit(&AuditFilter {
                after,
                ..filter.clone()
            })
            .await
            .unwrap_or_else(|e| fail(&e.to_string()));
        pages += 1;
        seen.extend(page.rows.into_iter().map(|r| r.id));
        match page.next {
            Some(next) => after = Some(next),
            None => break,
        }
    }
    written.reverse();
    assert_eq!(seen, written, "every row once, newest first");
    assert_eq!(pages, 3, "3 + 3 + 1");

    let first = cp
        .query_audit(&filter)
        .await
        .unwrap_or_else(|e| fail(&e.to_string()));
    let cursor = first.next.unwrap_or_else(|| fail("a second page"));
    let round_trip: AuditCursor = cursor
        .to_string()
        .parse()
        .unwrap_or_else(|e: Error| fail(&e.to_string()));
    assert_eq!(round_trip, cursor);
    let elsewhere = cp
        .query_audit(&AuditFilter {
            action: Some(String::from("open")),
            after: Some(cursor),
            ..filter.clone()
        })
        .await;
    assert!(elsewhere.is_err_and(|e| e.to_string().contains("different filter")));
    assert!("not a cursor".parse::<AuditCursor>().is_err());
    assert!("bm90IGEgY3Vyc29y".parse::<AuditCursor>().is_err());
}

#[test]
fn dummy_hash_parses_as_argon2id() {
    assert!(PasswordHash::new(StoredPasswordHash::DUMMY).is_ok());
    assert!(!StoredPasswordHash::stored_or_dummy(None).verifies("anything"));
}
