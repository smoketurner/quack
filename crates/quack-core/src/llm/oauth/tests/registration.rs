//! Dynamic client registration (RFC 7591) and its management (RFC 7592)
//! against the mock issuer's `/register` endpoints.

use super::*;
use crate::config::inspect::{Inspection, Origin};
use crate::doctor::{Area, Probing, Report, Status};
use crate::llm::oauth::client_key::ClientKey;
use crate::llm::oauth::registration::{
    ClientMetadata, ReadBack, Registered, Registrar, RegistrationName, Removal,
    SignIn as SignInWith, TemporaryClient, metadata_for,
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

/// `--replace` registers the new client and keeps it before it deletes
/// the old one, so a delete that fails leaves quack with the new client,
/// never with none.
#[tokio::test]
async fn replace_registers_the_new_client_first_then_deletes_the_old() {
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

    // A delete the issuer refuses after the new client is kept: quack uses
    // the new client and says the old one is left.
    idp.state.delete_fails.store(true, Ordering::SeqCst);
    let kept = register(&config, &registrar, &issuer, None, true).await;
    assert!(
        kept.as_ref().is_ok_and(|r| r.client_id == "client-3"
            && matches!(&r.replaced, Some(Removal::Left { client_id, .. }) if client_id == "client-2")),
        "{kept:?}"
    );
    assert!(
        registrar
            .keys()
            .registration(&issuer)
            .await
            .is_ok_and(|r| r.is_some_and(|r| r.client_id == "client-3"))
    );
    idp.state.delete_fails.store(false, Ordering::SeqCst);

    // A client the issuer already forgot is no reason to stop.
    if let Ok(mut clients) = idp.state.clients.lock() {
        clients.clear();
    }
    let again = register(&config, &registrar, &issuer, None, true).await;
    assert!(
        again.as_ref().is_ok_and(|r| r.replaced
            == Some(Removal::AlreadyGone {
                client_id: String::from("client-3"),
                status: 401
            })),
        "{again:?}"
    );
}

/// A registered client's key rotates as a client registered by hand does,
/// with quack updating the issuer (RFC 7592) itself: `--rotate` puts the
/// new key beside the one in use, and `--activate` signs with it and leaves
/// the issuer holding it alone. Every assertion along the way is accepted.
#[tokio::test]
#[expect(
    clippy::too_many_lines,
    reason = "three rotation steps checked against the issuer in one flow"
)]
async fn rotation_puts_both_keys_at_the_issuer_then_the_new_one_alone() {
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
    let client = ClientKeyName::new(&idp.issuer, "client-1");
    let original = stored_thumbprint(&config, "client-1", &idp.issuer).await;
    let login = |config: Config| async move {
        provider_manager(&config, "svc")
            .login(LoginFlow::Configured, &|_| {})
            .await
    };
    assert!(login(config.clone()).await.is_ok(), "{:?}", refused(&idp));

    // --rotate: the new key waits beside the one in use.
    let keys = registrar.keys();
    let (in_use, next) = keys
        .stage_replacement(&client)
        .await
        .unwrap_or_else(|e| fail(&e.to_string()));
    let mut both = next.jwks();
    if let Some(in_use) = &in_use {
        both.keys.insert(0, in_use.jwk().clone());
    }
    // A refused update leaves the issuer as it was.
    idp.state.update_fails.store(true, Ordering::SeqCst);
    let refused_update = registrar.publish_keys(&issuer, &both).await;
    assert!(
        refused_update
            .as_ref()
            .is_err_and(|e| e.to_string().contains("invalid_client_metadata")),
        "{refused_update:?}"
    );
    idp.state.update_fails.store(false, Ordering::SeqCst);
    idp.state
        .rotate_registration_token
        .store(true, Ordering::SeqCst);
    let published = registrar.publish_keys(&issuer, &both).await;
    assert!(
        published.as_ref().is_ok_and(|p| p.new_registration_token),
        "{published:?}"
    );
    assert_eq!(
        stored_thumbprint(&config, "client-1", &idp.issuer).await,
        original,
        "the key in use stays in use"
    );
    assert!(login(config.clone()).await.is_ok(), "{:?}", refused(&idp));

    // The update read the registration back and sent all of it, with the
    // client id and both keys, and none of the issuer's bookkeeping.
    let management = management_requests(&idp);
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
    assert_eq!(sent.get("jwks"), serde_json::to_value(&both).ok().as_ref());

    // --activate: quack signs with the new key, which the issuer accepts,
    let active = keys
        .activate_replacement(&client)
        .await
        .unwrap_or_else(|e| fail(&e.to_string()));
    assert_eq!(active.thumbprint(), next.thumbprint());
    let fresh = registered_config(dir.path(), &idp.issuer, &service_provider(&idp.issuer));
    assert!(login(fresh.clone()).await.is_ok(), "{:?}", refused(&idp));

    // and the issuer then holds it alone, still accepted; the rotated
    // registration token reads the registration back.
    let retired = registrar.publish_keys(&issuer, &active.jwks()).await;
    assert!(retired.is_ok(), "{retired:?}");
    assert!(login(fresh).await.is_ok(), "{:?}", refused(&idp));
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

/// A client named in the configuration (registered by hand) gets a key of
/// its own: it never takes over the key waiting for a registration, which
/// another client at the same issuer could otherwise claim.
#[tokio::test]
async fn a_client_named_in_the_configuration_never_takes_the_pending_key() {
    let idp = MockIdp::start().await;
    let dir = temp();
    let config = registered_config(dir.path(), &idp.issuer, "");
    let issuer = RegistrationName::new(&idp.issuer);
    let registrar = registrar(&config);
    let pending = registrar
        .pending_key(&issuer)
        .await
        .unwrap_or_else(|e| fail(&e.to_string()));
    let keys = ClientKeys::new(&config, KeySource::File);
    let by_hand = keys
        .key(&ClientKeyName::new(&idp.issuer, "by-hand"))
        .await
        .unwrap_or_else(|e| fail(&e.to_string()));
    assert_ne!(by_hand.thumbprint(), pending.thumbprint());
    assert!(
        keys.existing(&ClientKeyName::pending(&idp.issuer))
            .await
            .is_ok_and(|k| k.is_some_and(|k| k.thumbprint() == pending.thumbprint()))
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

/// Signing in for a registration: a temporary public client is registered
/// with no bearer, the person signs in through it (here the device-code
/// flow), quack's client is registered with that person's token, and the
/// temporary client is deleted last, since deleting a client can end the
/// sign-ins made through it.
#[tokio::test]
#[expect(
    clippy::too_many_lines,
    reason = "the whole signed-in registration, end to end"
)]
async fn a_signed_in_registration_uses_the_persons_token_then_deletes_the_sign_in_client() {
    let idp = MockIdp::start().await;
    let dir = temp();
    let config = registered_config(dir.path(), &idp.issuer, "");
    let issuer = RegistrationName::new(&idp.issuer);
    let registrar = registrar(&config);
    let metadata = metadata(&config, &registrar, &issuer).await;
    let prompts = StdMutex::new(Vec::new());
    let notify = |prompt: LoginPrompt| {
        if let Ok(mut seen) = prompts.lock() {
            seen.push(prompt);
        }
    };

    let done = registrar
        .register_signed_in(
            &issuer,
            &metadata,
            false,
            SignInWith {
                flow: LoginFlow::DeviceCode,
                key_source: KeySource::File,
                notify: &notify,
            },
        )
        .await
        .unwrap_or_else(|e| fail(&e.to_string()));

    assert_eq!(done.registered.client_id, "client-2");
    assert!(done.earlier.is_empty());
    assert!(
        registrar
            .sign_in_leftovers(&issuer)
            .await
            .is_ok_and(|l| l.is_empty()),
        "the temporary client's record is gone with it"
    );
    assert_eq!(
        done.temporary,
        TemporaryClient::Deleted {
            client_id: String::from("client-1")
        }
    );
    assert!(
        prompts
            .lock()
            .is_ok_and(|p| matches!(p.as_slice(), [LoginPrompt::DeviceCode { .. }])),
        "the person was shown the device code once"
    );

    let requests = registration_requests(&idp);
    let [(first_auth, sign_in_client), (second_auth, quacks_client)] = requests.as_slice() else {
        fail(&format!("two registrations expected: {requests:?}"));
    };
    // The temporary client: public, no bearer, a native app's grants.
    assert_eq!(first_auth, "");
    assert_eq!(sign_in_client["token_endpoint_auth_method"], "none");
    assert_eq!(
        sign_in_client["grant_types"],
        serde_json::json!([
            "authorization_code",
            "urn:ietf:params:oauth:grant-type:device_code"
        ])
    );
    assert!(sign_in_client.get("jwks").is_none());
    assert_eq!(sign_in_client["application_type"], "native");
    let redirect = sign_in_client
        .get("redirect_uris")
        .and_then(|uris| uris.get(0))
        .and_then(serde_json::Value::as_str)
        .unwrap_or_default();
    assert!(
        redirect.starts_with("http://127.0.0.1:") && redirect.ends_with("/callback"),
        "{redirect}"
    );
    assert!(
        !redirect.contains(":19876/"),
        "a free port, not the one `quack auth login` uses"
    );
    // quack's client: the person's token as the bearer, quack's key.
    assert_eq!(second_auth, "Bearer device-access");
    assert_eq!(
        quacks_client["token_endpoint_auth_method"],
        "private_key_jwt"
    );
    assert_eq!(
        quacks_client["jwks"],
        serde_json::to_value(&metadata.jwks).unwrap_or_default()
    );

    // The temporary client is deleted with its own registration token, and
    // after quack's client was registered.
    assert!(
        management_requests(&idp)
            .iter()
            .any(|(m, path, auth, _)| m == "DELETE"
                && path == "/register/client-1"
                && auth == "Bearer rat-1")
    );
    assert!(
        !management_requests(&idp)
            .iter()
            .any(|(_, path, _, _)| path == "/register/client-2"),
        "quack's own client is left alone"
    );
    // Only quack's client is kept, and nothing of the sign-in.
    assert!(
        registrar
            .keys()
            .registration(&issuer)
            .await
            .is_ok_and(|r| r.is_some_and(|r| r.client_id == "client-2"))
    );
    assert!(
        provider_manager(&config, "gw")
            .client_id()
            .await
            .is_ok_and(|id| id == "client-2")
    );
}

/// A registration that would be refused is refused before anyone is asked
/// to sign in, and no temporary client is left behind.
#[tokio::test]
async fn a_signed_in_registration_over_an_existing_client_is_refused_before_signing_in() {
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
    let metadata = metadata(&config, &registrar, &issuer).await;
    let notify = |prompt: LoginPrompt| fail(&format!("no sign-in expected: {prompt:?}"));

    let refused = registrar
        .register_signed_in(
            &issuer,
            &metadata,
            false,
            SignInWith {
                flow: LoginFlow::DeviceCode,
                key_source: KeySource::File,
                notify: &notify,
            },
        )
        .await;
    assert!(
        refused.is_err_and(|e| e.to_string().contains("--replace")),
        "an existing registration needs --replace"
    );
    assert_eq!(registration_requests(&idp).len(), 1, "no temporary client");
}

/// Sign in for a registration at `issuer` with the device-code flow, the
/// prompts dropped.
async fn sign_in_and_register(
    registrar: &Registrar,
    issuer: &RegistrationName,
    metadata: &ClientMetadata,
) -> Result<crate::llm::oauth::registration::SignedInRegistration> {
    registrar
        .register_signed_in(
            issuer,
            metadata,
            false,
            SignInWith {
                flow: LoginFlow::DeviceCode,
                key_source: KeySource::File,
                notify: &|_| {},
            },
        )
        .await
}

fn deleted(idp: &MockIdp, client_id: &str) -> bool {
    management_requests(idp)
        .iter()
        .any(|(m, path, ..)| m == "DELETE" && *path == format!("/register/{client_id}"))
}

/// A sign-in the person declines still deletes the temporary client, and
/// registers nothing else.
#[tokio::test]
async fn a_declined_sign_in_still_deletes_the_temporary_client() {
    let idp = MockIdp::start().await;
    let dir = temp();
    let config = registered_config(dir.path(), &idp.issuer, "");
    let issuer = RegistrationName::new(&idp.issuer);
    let registrar = registrar(&config);
    let metadata = metadata(&config, &registrar, &issuer).await;
    idp.state.device_denied.store(true, Ordering::SeqCst);

    let declined = sign_in_and_register(&registrar, &issuer, &metadata).await;
    assert!(declined.is_err(), "{declined:?}");
    assert_eq!(
        registration_requests(&idp).len(),
        1,
        "only the temporary client"
    );
    assert!(deleted(&idp, "client-1"));
    assert!(
        registrar
            .sign_in_leftovers(&issuer)
            .await
            .is_ok_and(|l| l.is_empty())
    );
    assert!(
        registrar
            .keys()
            .registration(&issuer)
            .await
            .is_ok_and(|r| r.is_none())
    );
}

/// A registration the issuer refuses after the sign-in still deletes the
/// temporary client.
#[tokio::test]
async fn a_refused_registration_still_deletes_the_temporary_client() {
    let idp = MockIdp::start().await;
    let dir = temp();
    let config = registered_config(dir.path(), &idp.issuer, "");
    let issuer = RegistrationName::new(&idp.issuer);
    let registrar = registrar(&config);
    let metadata = metadata(&config, &registrar, &issuer).await;
    idp.state
        .refuse_bearer_registrations
        .store(true, Ordering::SeqCst);

    let refused_registration = sign_in_and_register(&registrar, &issuer, &metadata).await;
    assert!(
        refused_registration
            .as_ref()
            .is_err_and(|e| e.to_string().contains("invalid_client_metadata")),
        "{refused_registration:?}"
    );
    assert_eq!(registration_requests(&idp).len(), 2);
    assert!(deleted(&idp, "client-1"));
    assert!(
        registrar
            .sign_in_leftovers(&issuer)
            .await
            .is_ok_and(|l| l.is_empty())
    );
}

/// A temporary client the issuer would not delete stays recorded, so
/// `quack doctor` names it and the next `quack auth register` deletes it
/// and reports it.
#[tokio::test]
async fn a_temporary_client_left_behind_is_recorded_and_deleted_later() {
    let idp = MockIdp::start().await;
    let dir = temp();
    let config = registered_config(dir.path(), &idp.issuer, "");
    let issuer = RegistrationName::new(&idp.issuer);
    let registrar = registrar(&config);
    let metadata = metadata(&config, &registrar, &issuer).await;
    idp.state.device_denied.store(true, Ordering::SeqCst);
    idp.state.delete_fails.store(true, Ordering::SeqCst);

    assert!(
        sign_in_and_register(&registrar, &issuer, &metadata)
            .await
            .is_err()
    );
    assert!(
        registrar
            .sign_in_leftovers(&issuer)
            .await
            .is_ok_and(|l| l == ["client-1"]),
        "the record survives a refused delete"
    );
    // `quack doctor` names it, with the command that deletes it.
    let control = ControlPlane::open(&config)
        .await
        .unwrap_or_else(|e| fail(&e.to_string()));
    let mut report = Report::default();
    crate::doctor::check_registrations(
        &mut report,
        &config,
        Some(&control),
        Probing::Offline,
        KeySource::File,
    )
    .await;
    assert!(
        auth_checks(&report)
            .iter()
            .any(|(status, summary, fix)| *status == Status::Warn
                && summary.contains("temporary sign-in client client-1")
                && fix
                    .as_deref()
                    .is_some_and(|f| f.contains("quack auth register --issuer"))),
        "{:?}",
        auth_checks(&report)
    );
    // Another left behind: each keeps a record of its own.
    assert!(
        sign_in_and_register(&registrar, &issuer, &metadata)
            .await
            .is_err()
    );
    assert!(
        registrar
            .sign_in_leftovers(&issuer)
            .await
            .is_ok_and(|l| l == ["client-1", "client-2"])
    );

    // The next `quack auth register` deletes them once the issuer deletes
    // again.
    idp.state.delete_fails.store(false, Ordering::SeqCst);
    let cleaned = registrar.clean_up_sign_in(&issuer).await;
    assert_eq!(
        cleaned.ok(),
        Some(vec![
            TemporaryClient::Deleted {
                client_id: String::from("client-1")
            },
            TemporaryClient::Deleted {
                client_id: String::from("client-2")
            },
        ])
    );
    assert!(
        registrar
            .sign_in_leftovers(&issuer)
            .await
            .is_ok_and(|l| l.is_empty())
    );

    // The next signed-in registration deletes one left before it, and says so.
    idp.state.delete_fails.store(true, Ordering::SeqCst);
    assert!(
        sign_in_and_register(&registrar, &issuer, &metadata)
            .await
            .is_err()
    );
    idp.state.delete_fails.store(false, Ordering::SeqCst);
    idp.state.device_denied.store(false, Ordering::SeqCst);
    let done = sign_in_and_register(&registrar, &issuer, &metadata).await;
    let Ok(done) = done else {
        fail(&format!("{done:?}"));
    };
    assert_eq!(
        done.earlier,
        vec![TemporaryClient::Deleted {
            client_id: String::from("client-3")
        }]
    );
    assert!(
        registrar
            .sign_in_leftovers(&issuer)
            .await
            .is_ok_and(|l| l.is_empty())
    );
}
