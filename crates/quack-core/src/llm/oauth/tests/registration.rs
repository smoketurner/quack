//! Dynamic client registration (RFC 7591) and its management (RFC 7592)
//! against the mock issuer's `/register` endpoints.

use super::*;
use crate::config::inspect::{Inspection, Origin};
use crate::doctor::{Area, Probing, Report, Status};
use crate::llm::oauth::client_key::ClientKey;
use crate::llm::oauth::registration::{
    ClientMetadata, ReadBack, Registered, Registrar, RegistrationName, Removal, metadata_for,
};
use crate::oidc::SignIn;
use crate::storage::control::ControlPlane;

/// quack's configuration for one Vouch-like issuer: sign-in and an
/// on-behalf-of provider without an actor, both without a `client_id`, and
/// `extra` appended.
fn registered_config(dir: &Path, issuer: &str, extra: &str) -> Config {
    let toml = format!(
        "[server.oidc]\nissuer_url = \"{issuer}\"\nclient_auth = \"private_key_jwt\"\n\
         redirect_uri = \"https://q.example.com/auth/oidc/callback\"\nscopes = [\"openid\", \"email\"]\n\
         [providers.gw]\ntype = \"openai\"\nbase_url = \"https://gw.example.com/v1\"\nauth = \"oauth\"\n\
         [providers.gw.oauth]\nissuer_url = \"{issuer}/\"\nclient_auth = \"private_key_jwt\"\n\
         grant = \"on-behalf-of\"\nactor = false\n{extra}"
    );
    let mut config = Config::parse(&toml).unwrap_or_else(|e| fail(&e.to_string()));
    config.general.data_dir = dir.to_path_buf();
    config
}

/// A provider at the issuer that obtains its own token with an assertion:
/// what shows which key the issuer accepts.
fn service_provider(issuer: &str) -> String {
    format!(
        "[providers.svc]\ntype = \"openai\"\nbase_url = \"https://svc.example.com/v1\"\nauth = \"oauth\"\n\
         [providers.svc.oauth]\nissuer_url = \"{issuer}\"\nclient_auth = \"private_key_jwt\"\n\
         grant = \"client-credentials\"\n"
    )
}

fn registrar(config: &Config) -> Registrar {
    Registrar::new(ClientKeys::new(config, KeySource::File))
        .unwrap_or_else(|e| fail(&e.to_string()))
}

async fn metadata(
    config: &Config,
    registrar: &Registrar,
    issuer: &RegistrationName,
) -> ClientMetadata {
    let key = registrar
        .pending_key(issuer)
        .await
        .unwrap_or_else(|e| fail(&e.to_string()));
    metadata_for(config, issuer, "quack", key.jwks()).unwrap_or_else(|e| fail(&e.to_string()))
}

async fn register(
    config: &Config,
    registrar: &Registrar,
    issuer: &RegistrationName,
    bearer: Option<&str>,
    replace: bool,
) -> Result<Registered> {
    let metadata = metadata(config, registrar, issuer).await;
    let bearer = bearer.map(|b| SecretString::from(b.to_owned()));
    registrar
        .register(issuer, &metadata, bearer.as_ref(), replace)
        .await
}

fn registration_requests(idp: &MockIdp) -> Vec<(String, serde_json::Value)> {
    idp.state
        .registration_requests
        .lock()
        .map(|r| r.clone())
        .unwrap_or_default()
}

fn management_requests(idp: &MockIdp) -> Vec<(String, String, String, String)> {
    idp.state
        .management_requests
        .lock()
        .map(|r| r.clone())
        .unwrap_or_default()
}

fn provider_manager(config: &Config, provider: &str) -> TokenManager {
    let Some(oauth) = config
        .providers
        .get(provider)
        .and_then(|p| p.auth.oauth())
        .cloned()
    else {
        fail("no such provider");
    };
    TokenManager::new(config, &name(provider), oauth, KeySource::File)
        .unwrap_or_else(|e| fail(&e.to_string()))
}

async fn stored_thumbprint(config: &Config, client_id: &str, issuer: &str) -> Option<String> {
    ClientKeys::new(config, KeySource::File)
        .existing(&ClientKeyName::new(issuer, client_id))
        .await
        .ok()
        .flatten()
        .map(|key| key.thumbprint().to_owned())
}

#[tokio::test]
async fn an_open_registration_sends_the_union_of_what_the_clients_need_and_no_bearer() {
    let idp = MockIdp::start().await;
    let dir = temp();
    let config = registered_config(dir.path(), &idp.issuer, "");
    let issuer = RegistrationName::new(&idp.issuer);
    let registrar = registrar(&config);
    let pending = registrar
        .pending_key(&issuer)
        .await
        .unwrap_or_else(|e| fail(&e.to_string()));

    let registered = register(&config, &registrar, &issuer, None, false).await;
    assert!(
        registered
            .as_ref()
            .is_ok_and(|r| r.client_id == "client-1" && r.manageable && r.replaced.is_none()),
        "{registered:?}"
    );

    let requests = registration_requests(&idp);
    let [(authorization, body)] = requests.as_slice() else {
        fail(&format!("one registration expected: {requests:?}"));
    };
    assert_eq!(authorization, "", "an open registration sends no bearer");
    assert_eq!(
        body.get("grant_types"),
        Some(&serde_json::json!([
            "authorization_code",
            "urn:ietf:params:oauth:grant-type:token-exchange"
        ])),
        "actor = false: no client_credentials, and no refresh_token without offline_access"
    );
    assert_eq!(
        body.get("response_types"),
        Some(&serde_json::json!(["code"]))
    );
    assert_eq!(
        body.get("redirect_uris"),
        Some(&serde_json::json!([
            "https://q.example.com/auth/oidc/callback"
        ]))
    );
    assert_eq!(
        body.get("application_type"),
        Some(&serde_json::json!("web"))
    );
    assert_eq!(
        body.get("token_endpoint_auth_method"),
        Some(&serde_json::json!("private_key_jwt"))
    );
    assert_eq!(
        body.get("token_endpoint_auth_signing_alg"),
        Some(&serde_json::json!("ES256"))
    );
    assert_eq!(body.get("scope"), Some(&serde_json::json!("openid email")));
    assert_eq!(body.get("client_name"), Some(&serde_json::json!("quack")));
    assert_eq!(
        body.get("jwks"),
        serde_json::to_value(pending.jwks()).ok().as_ref(),
        "the registration carries the key made for it"
    );
    for bound in [
        "dpop_bound_access_tokens",
        "tls_client_certificate_bound_access_tokens",
    ] {
        assert!(body.get(bound).is_none(), "{bound} makes a FAPI client");
    }

    // The key moved to the client's name; nothing is left pending.
    assert_eq!(
        stored_thumbprint(&config, "client-1", &idp.issuer).await,
        Some(pending.thumbprint().to_owned())
    );
    let control = ControlPlane::open(&config)
        .await
        .unwrap_or_else(|e| fail(&e.to_string()));
    assert!(
        control
            .sealed(crate::storage::control::SealedOwner::ClientKey(&idp.issuer))
            .await
            .is_ok_and(|k| k.is_none())
    );

    // Both sections read their client_id from the registration.
    let gw = provider_manager(&config, "gw");
    assert!(gw.client_id().await.is_ok_and(|id| id == "client-1"));
    let Some(oidc) = config.server.oidc.clone() else {
        fail("no [server.oidc]");
    };
    let sign_in = SignIn::new(oidc, ClientKeys::new(&config, KeySource::File))
        .unwrap_or_else(|e| fail(&e.to_string()));
    assert!(sign_in.client_id().await.is_ok_and(|id| id == "client-1"));

    // A second registration is refused without --replace.
    let again = register(&config, &registrar, &issuer, None, false).await;
    assert!(
        again
            .as_ref()
            .is_err_and(|e| e.to_string().contains("--replace")),
        "{again:?}"
    );
}

#[tokio::test]
async fn a_registration_with_an_access_token_sends_it_as_the_bearer() {
    let idp = MockIdp::start().await;
    let dir = temp();
    let config = registered_config(dir.path(), &idp.issuer, "");
    let issuer = RegistrationName::new(&idp.issuer);
    let registrar = registrar(&config);
    let registered = register(&config, &registrar, &issuer, Some("vouch-access"), false).await;
    assert!(registered.is_ok(), "{registered:?}");
    let requests = registration_requests(&idp);
    assert_eq!(
        requests.first().map(|(auth, _)| auth.as_str()),
        Some("Bearer vouch-access")
    );
}

#[tokio::test]
async fn the_metadata_follows_the_grants_the_sections_use() {
    let dir = temp();
    let issuer = "https://idp.example.com";
    let name = RegistrationName::new(issuer);
    let jwks =
        || ClientKey::generate().map_or_else(|e| fail(&e.to_string()), |(key, _)| key.jwks());
    let provider = |section: &str, extra: &str| {
        format!(
            "[providers.{section}]\ntype = \"openai\"\nbase_url = \"https://m.example.com/v1\"\nauth = \"oauth\"\n\
             [providers.{section}.oauth]\nissuer_url = \"{issuer}\"\nclient_auth = \"private_key_jwt\"\n{extra}"
        )
    };
    let config_of = |toml: String| {
        let mut config = Config::parse(&toml).unwrap_or_else(|e| fail(&e.to_string()));
        config.general.data_dir = dir.path().to_path_buf();
        config
    };

    // An actor token is quack's own client-credentials token; offline_access
    // asks for refresh tokens; a device-code login needs no redirect.
    let config = config_of(format!(
        "{}{}",
        provider(
            "a",
            "grant = \"on-behalf-of\"\nscopes = [\"model.use\", \"offline_access\"]\n"
        ),
        provider("b", "grant = \"device-code\"\n")
    ));
    let metadata = metadata_for(&config, &name, "q", jwks());
    assert!(
        metadata.as_ref().is_ok_and(|m| m.grant_types
            == [
                "urn:ietf:params:oauth:grant-type:token-exchange",
                "client_credentials",
                "urn:ietf:params:oauth:grant-type:device_code",
                "refresh_token"
            ]
            && m.redirect_uris.is_empty()
            && m.response_types.is_empty()
            && m.application_type.is_none()
            && m.scope.as_deref() == Some("model.use offline_access")
            && m.client_name == "q"),
        "{metadata:?}"
    );

    // A browser login alone registers its loopback redirect as a native app.
    let config = config_of(provider(
        "c",
        "redirect_uri = \"http://127.0.0.1:19876/callback\"\n",
    ));
    let metadata = metadata_for(&config, &name, "q", jwks());
    assert!(
        metadata
            .as_ref()
            .is_ok_and(|m| m.redirect_uris == ["http://127.0.0.1:19876/callback"]
                && m.application_type.as_deref() == Some("native")
                && m.response_types == ["code"]),
        "{metadata:?}"
    );

    // Beside the web sign-in it cannot share the registration.
    let sign_in = format!(
        "[server.oidc]\nissuer_url = \"{issuer}\"\nclient_auth = \"private_key_jwt\"\n\
         redirect_uri = \"https://q.example.com/auth/oidc/callback\"\n"
    );
    let config = config_of(format!("{sign_in}{}", provider("c", "")));
    let refused = metadata_for(&config, &name, "q", jwks());
    assert!(
        refused
            .as_ref()
            .is_err_and(|e| e.to_string().contains("[providers.c.oauth]")),
        "{refused:?}"
    );
    // Another issuer's sections are not this registration's.
    let elsewhere = metadata_for(
        &config,
        &RegistrationName::new("https://other"),
        "q",
        jwks(),
    );
    assert!(elsewhere.is_err());
}

#[tokio::test]
async fn a_client_without_a_registration_names_quack_auth_register() {
    let idp = MockIdp::start().await;
    let dir = temp();
    let config = registered_config(dir.path(), &idp.issuer, "");
    let gw = provider_manager(&config, "gw");
    let err = gw.client_id().await.map(str::to_owned);
    assert!(
        err.as_ref().is_err_and(|e| {
            let e = e.to_string();
            e.contains("[providers.gw.oauth]") && e.contains("quack auth register")
        }),
        "{err:?}"
    );
    let Some(oidc) = config.server.oidc.clone() else {
        fail("no [server.oidc]");
    };
    let sign_in = SignIn::new(oidc, ClientKeys::new(&config, KeySource::File))
        .unwrap_or_else(|e| fail(&e.to_string()));
    assert!(sign_in.begin().await.is_err_and(|e| {
        let e = e.to_string();
        e.contains("[server.oidc]") && e.contains("quack auth register")
    }));
}

#[tokio::test]
async fn replace_deletes_the_old_client_first_and_keeps_only_the_new_key() {
    let idp = MockIdp::start().await;
    let dir = temp();
    let config = registered_config(dir.path(), &idp.issuer, "");
    let issuer = RegistrationName::new(&idp.issuer);
    let registrar = registrar(&config);
    assert!(
        register(&config, &registrar, &issuer, None, false)
            .await
            .is_ok()
    );
    let first = stored_thumbprint(&config, "client-1", &idp.issuer).await;

    let replaced = register(&config, &registrar, &issuer, None, true).await;
    assert!(
        replaced.as_ref().is_ok_and(|r| r.client_id == "client-2"
            && r.replaced
                == Some(Removal::Deleted {
                    client_id: String::from("client-1")
                })),
        "{replaced:?}"
    );
    let management = management_requests(&idp);
    assert!(
        management
            .iter()
            .any(|(method, path, auth, _)| method == "DELETE"
                && path == "/register/client-1"
                && auth == "Bearer rat-1"),
        "{management:?}"
    );
    assert_eq!(
        stored_thumbprint(&config, "client-1", &idp.issuer).await,
        None
    );
    let second = stored_thumbprint(&config, "client-2", &idp.issuer).await;
    assert!(second.is_some() && second != first);
    assert!(
        provider_manager(&config, "gw")
            .client_id()
            .await
            .is_ok_and(|id| id == "client-2")
    );

    // A client the issuer already forgot is no reason to stop.
    if let Ok(mut clients) = idp.state.clients.lock() {
        clients.clear();
    }
    let again = register(&config, &registrar, &issuer, None, true).await;
    assert!(
        again.as_ref().is_ok_and(|r| r.replaced
            == Some(Removal::AlreadyGone {
                client_id: String::from("client-2"),
                status: 401
            })),
        "{again:?}"
    );
}

#[tokio::test]
async fn rotation_sends_the_whole_registration_and_switches_keys_only_once_accepted() {
    let idp = MockIdp::start().await;
    let dir = temp();
    let config = registered_config(dir.path(), &idp.issuer, &service_provider(&idp.issuer));
    let issuer = RegistrationName::new(&idp.issuer);
    let registrar = registrar(&config);
    assert!(
        register(&config, &registrar, &issuer, None, false)
            .await
            .is_ok()
    );
    let original = stored_thumbprint(&config, "client-1", &idp.issuer).await;
    let login = |config: Config| async move {
        provider_manager(&config, "svc")
            .login(LoginFlow::Configured, &|_| {})
            .await
    };
    assert!(login(config.clone()).await.is_ok(), "{:?}", refused(&idp));

    // A refused update leaves the key in use.
    idp.state.update_fails.store(true, Ordering::SeqCst);
    let refused_update = registrar.rotate(&issuer).await;
    assert!(
        refused_update
            .as_ref()
            .is_err_and(|e| e.to_string().contains("invalid_client_metadata")),
        "{refused_update:?}"
    );
    assert_eq!(
        stored_thumbprint(&config, "client-1", &idp.issuer).await,
        original
    );
    assert!(login(config.clone()).await.is_ok(), "{:?}", refused(&idp));

    idp.state.update_fails.store(false, Ordering::SeqCst);
    idp.state
        .rotate_registration_token
        .store(true, Ordering::SeqCst);
    let rotated = registrar.rotate(&issuer).await;
    let Ok(rotated) = rotated else {
        fail(&format!("{rotated:?}"));
    };
    assert_eq!(rotated.old_thumbprint, original);
    assert!(rotated.new_registration_token);
    assert_ne!(Some(rotated.new_thumbprint.clone()), original);

    // The update read the registration back and sent all of it, with the
    // client id and the new key, and none of the issuer's bookkeeping.
    let management = management_requests(&idp);
    // The last update, the one accepted.
    let Some((_, _, auth, body)) = management.iter().rev().find(|(m, ..)| m == "PUT") else {
        fail(&format!("no update: {management:?}"));
    };
    assert_eq!(auth, "Bearer rat-1");
    assert!(
        management
            .iter()
            .any(|(m, path, ..)| m == "GET" && path == "/register/client-1")
    );
    let sent: serde_json::Value = serde_json::from_str(body).unwrap_or_default();
    let registered = registration_requests(&idp)
        .first()
        .map(|(_, body)| body.clone())
        .unwrap_or_default();
    for field in [
        "grant_types",
        "response_types",
        "redirect_uris",
        "token_endpoint_auth_method",
        "token_endpoint_auth_signing_alg",
        "scope",
        "client_name",
        "application_type",
    ] {
        assert_eq!(sent.get(field), registered.get(field), "{field}");
    }
    assert_eq!(sent.get("client_id"), Some(&serde_json::json!("client-1")));
    for field in [
        "registration_access_token",
        "registration_client_uri",
        "client_id_issued_at",
        "client_secret_expires_at",
    ] {
        assert!(sent.get(field).is_none(), "{field} is the issuer's");
    }
    let new_kid = sent
        .get("jwks")
        .and_then(|j| j.get("keys"))
        .and_then(|k| k.get(0))
        .and_then(|k| k.get("kid"))
        .and_then(serde_json::Value::as_str)
        .map(str::to_owned);
    assert_eq!(new_kid.as_deref(), Some(rotated.new_thumbprint.as_str()));
    assert_eq!(
        stored_thumbprint(&config, "client-1", &idp.issuer).await,
        Some(rotated.new_thumbprint.clone())
    );

    // The next assertion is signed with the new key, which is the only one
    // the issuer now holds; the new registration token reads it back.
    assert!(login(config.clone()).await.is_ok(), "{:?}", refused(&idp));
    assert!(refused(&idp).is_empty(), "{:?}", refused(&idp));
    assert_eq!(
        registrar.read(&issuer).await.ok().flatten(),
        Some(ReadBack::Readable {
            client_id: String::from("client-1")
        })
    );
}

#[tokio::test]
async fn unregister_deletes_the_client_then_its_record_and_key() {
    let idp = MockIdp::start().await;
    let dir = temp();
    let config = registered_config(dir.path(), &idp.issuer, "");
    let issuer = RegistrationName::new(&idp.issuer);
    let registrar = registrar(&config);
    assert!(
        register(&config, &registrar, &issuer, None, false)
            .await
            .is_ok()
    );

    let removal = registrar.unregister(&issuer).await;
    assert_eq!(
        removal.ok(),
        Some(Removal::Deleted {
            client_id: String::from("client-1")
        })
    );
    assert!(
        management_requests(&idp)
            .iter()
            .any(|(m, path, auth, _)| m == "DELETE"
                && path == "/register/client-1"
                && auth == "Bearer rat-1")
    );
    assert!(
        registrar
            .keys()
            .registration(&issuer)
            .await
            .is_ok_and(|r| r.is_none())
    );
    assert_eq!(
        stored_thumbprint(&config, "client-1", &idp.issuer).await,
        None
    );
    assert!(provider_manager(&config, "gw").client_id().await.is_err());
    assert!(registrar.unregister(&issuer).await.is_err());
}

#[tokio::test]
async fn a_client_registered_by_hand_takes_over_the_key_its_registration_carried() {
    let idp = MockIdp::start().await;
    let dir = temp();
    let config = registered_config(dir.path(), &idp.issuer, "");
    let issuer = RegistrationName::new(&idp.issuer);
    let registrar = registrar(&config);
    // `quack auth register --print` made the key and printed the request.
    let printed = registrar
        .pending_key(&issuer)
        .await
        .unwrap_or_else(|e| fail(&e.to_string()));
    // The operator registered it by hand and wrote the client id down.
    let keys = ClientKeys::new(&config, KeySource::File);
    let adopted = keys
        .key(&ClientKeyName::new(&idp.issuer, "by-hand"))
        .await
        .unwrap_or_else(|e| fail(&e.to_string()));
    assert_eq!(adopted.thumbprint(), printed.thumbprint());
    assert!(
        keys.existing(&ClientKeyName::pending(&idp.issuer))
            .await
            .is_ok_and(|k| k.is_none())
    );
}

fn auth_checks(report: &Report) -> Vec<(Status, String, Option<String>)> {
    report
        .checks
        .iter()
        .filter(|c| c.area == Area::Auth)
        .map(|c| (c.status, c.summary.clone(), c.fix.clone()))
        .collect()
}

#[tokio::test]
async fn doctor_checks_the_registration_is_kept_and_still_readable() {
    let idp = MockIdp::start().await;
    let dir = temp();
    let config = registered_config(dir.path(), &idp.issuer, "");
    let control = ControlPlane::open(&config)
        .await
        .unwrap_or_else(|e| fail(&e.to_string()));
    let online = Probing::Online {
        timeout: std::time::Duration::from_secs(5),
    };
    let doctor = |probing: Probing| {
        let config = config.clone();
        let control = control.clone();
        async move {
            let mut report = Report::default();
            crate::doctor::check_registrations(
                &mut report,
                &config,
                Some(&control),
                probing,
                KeySource::File,
            )
            .await;
            auth_checks(&report)
        }
    };

    let missing = doctor(online).await;
    assert!(
        matches!(missing.as_slice(), [(Status::Fail, summary, Some(fix))]
            if summary.contains("[server.oidc] and [providers.gw.oauth]")
                && fix.contains("quack auth register")),
        "{missing:?}"
    );

    let issuer = RegistrationName::new(&idp.issuer);
    assert!(
        register(&config, &registrar(&config), &issuer, None, false)
            .await
            .is_ok()
    );
    let readable = doctor(online).await;
    assert!(
        matches!(readable.as_slice(), [(Status::Ok, summary, None)]
            if summary.contains("client-1") && summary.contains("still describes it")),
        "{readable:?}"
    );
    let offline = doctor(Probing::Offline).await;
    assert!(
        matches!(offline.as_slice(), [(Status::Ok, summary, None)] if summary.contains("--offline")),
        "{offline:?}"
    );

    // Deleted at the issuer: the registration token no longer reads it.
    if let Ok(mut clients) = idp.state.clients.lock() {
        clients.clear();
    }
    let gone = doctor(online).await;
    assert!(
        matches!(gone.as_slice(), [(Status::Fail, summary, Some(fix))]
            if summary.contains("HTTP 401") && fix.contains("--replace")),
        "{gone:?}"
    );
}

#[tokio::test]
async fn quack_config_shows_the_registered_client_id_and_where_it_came_from() {
    let idp = MockIdp::start().await;
    let dir = temp();
    let toml = format!(
        "[general]\ndata_dir = \"{}\"\n\
         [providers.gw]\ntype = \"openai\"\nbase_url = \"https://gw.example.com/v1\"\nauth = \"oauth\"\n\
         [providers.gw.oauth]\nissuer_url = \"{}\"\nclient_auth = \"private_key_jwt\"\n\
         grant = \"on-behalf-of\"\nactor = false\n",
        dir.path().display(),
        idp.issuer
    );
    let client_id = |inspection: &Inspection| {
        inspection
            .settings
            .iter()
            .find(|s| s.section == "providers.gw.oauth" && s.key == "client_id")
            .map(|s| (s.value.clone(), s.origin))
    };
    let mut before = Inspection::of(dir.path().join("config.toml"), Some(&toml));
    assert!(before.resolve_registered().await.is_ok());
    assert_eq!(client_id(&before), Some((None, Origin::Default)));

    let config = before.config.clone();
    let issuer = RegistrationName::new(&idp.issuer);
    assert!(
        register(&config, &registrar(&config), &issuer, None, false)
            .await
            .is_ok()
    );
    let mut after = Inspection::of(dir.path().join("config.toml"), Some(&toml));
    assert!(after.resolve_registered().await.is_ok());
    assert_eq!(
        client_id(&after),
        Some((Some(String::from("\"client-1\"")), Origin::Registration))
    );
    assert_eq!(Origin::Registration.to_string(), "registration");
}
