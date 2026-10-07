use std::sync::{Arc, Mutex, PoisonError};

use super::*;

#[expect(clippy::panic, reason = "test failure path")]
fn fail(msg: &str) -> ! {
    panic!("{msg}")
}

fn config(url: &str, kinds: Vec<JobKind>) -> WebhookConfig {
    WebhookConfig {
        url: url.to_owned(),
        secret_env: String::from("QUACK_TEST_WEBHOOK_SECRET"),
        kinds,
        timeout_seconds: 5,
    }
}

fn hook(url: &str, kinds: Vec<JobKind>) -> Webhook {
    Webhook::with_secret(&config(url, kinds), Some(String::from("Jefe")))
        .unwrap_or_else(|e| fail(&e.to_string()))
}

fn job(kind: JobKind, state: JobState) -> JobInfo {
    JobInfo {
        id: JobId::new(),
        number: JobNumber(7),
        kind,
        label: String::from("orders.csv from the finance share"),
        workspace_id: Some(WorkspaceId::from("ws-1")),
        owner: Some(UserId::from("ada")),
        lane: None,
        state,
        progress: Some(JobProgress { done: 3, total: 3 }),
        status: None,
        outcome: Some(String::from("1,204 rows from the confidential ledger")),
        queued_at: jiff::Timestamp::UNIX_EPOCH,
        started_at: Some(jiff::Timestamp::UNIX_EPOCH),
        finished_at: Some(jiff::Timestamp::UNIX_EPOCH),
        cancel_requested: false,
    }
}

#[test]
fn a_webhook_needs_its_secret_and_a_url_and_is_off_without_a_section() {
    assert!(matches!(Webhook::from_config(None), Ok(None)));
    let unset = Webhook::with_secret(&config("https://hooks.example.com/q", vec![]), None)
        .err()
        .map(|e| e.to_string())
        .unwrap_or_default();
    assert!(unset.contains("QUACK_TEST_WEBHOOK_SECRET"), "{unset}");
    assert!(
        Webhook::with_secret(
            &config("https://hooks.example.com/q", vec![]),
            Some(String::from("  "))
        )
        .is_err(),
        "a blank secret signs nothing"
    );
    assert!(Webhook::with_secret(&config("not a url", vec![]), Some(String::from("s"))).is_err());
}

#[test]
fn finished_jobs_of_the_named_kinds_are_reported() {
    let every = hook("https://hooks.example.com/q", vec![]);
    assert!(every.reports(&job(JobKind::Ingest, JobState::Succeeded)));
    assert!(every.reports(&job(JobKind::Import, JobState::Failed)));
    assert!(!every.reports(&job(JobKind::Ingest, JobState::Running)));
    assert!(
        !every.reports(&job(JobKind::Chat, JobState::Succeeded)),
        "turns are not reported unless named"
    );
    let imports = hook(
        "https://hooks.example.com/q",
        vec![JobKind::Import, JobKind::Chat],
    );
    assert!(imports.reports(&job(JobKind::Import, JobState::Cancelled)));
    assert!(imports.reports(&job(JobKind::Chat, JobState::Succeeded)));
    assert!(!imports.reports(&job(JobKind::Ingest, JobState::Succeeded)));
}

/// RFC 4231 test case 2: HMAC-SHA256 of "what do ya want for nothing?"
/// under "Jefe".
#[test]
fn the_signature_is_hmac_sha256_of_the_body() {
    assert_eq!(
        hook("https://hooks.example.com/q", vec![]).signature(b"what do ya want for nothing?"),
        "sha256=5bdcc146bf60754e6a042426089575c75a003f089d2739839dec58b964ec3843"
    );
}

/// What a test endpoint received: each request's signature and body.
type Received = Arc<Mutex<Vec<(String, Vec<u8>)>>>;

/// An endpoint that refuses the first POST and takes the rest.
async fn endpoint() -> (String, Received) {
    use axum::Router;
    use axum::body::Bytes;
    use axum::extract::State;
    use axum::http::{HeaderMap, StatusCode};
    use axum::routing::post;

    async fn take(State(seen): State<Received>, headers: HeaderMap, body: Bytes) -> StatusCode {
        let mut seen = seen.lock().unwrap_or_else(PoisonError::into_inner);
        let signature = headers
            .get(SIGNATURE_HEADER)
            .and_then(|v| v.to_str().ok())
            .unwrap_or_default()
            .to_owned();
        seen.push((signature, body.to_vec()));
        if seen.len() == 1 {
            StatusCode::INTERNAL_SERVER_ERROR
        } else {
            StatusCode::NO_CONTENT
        }
    }

    let seen: Received = Arc::default();
    let router = Router::new()
        .route("/hook", post(take))
        .with_state(Arc::clone(&seen));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .unwrap_or_else(|e| fail(&e.to_string()));
    let url = format!(
        "http://{}/hook",
        listener
            .local_addr()
            .map_or_else(|e| fail(&e.to_string()), |a| a.to_string())
    );
    tokio::spawn(async move { drop(axum::serve(listener, router).await) });
    (url, seen)
}

/// A finished job is sent signed, with no label or outcome text, and
/// sent once more when the endpoint fails the first time.
#[tokio::test]
async fn a_finished_job_is_posted_signed_and_retried_once() {
    let (url, seen) = endpoint().await;
    let (sender, receiver) = broadcast::channel(8);
    let stopping = CancellationToken::new();
    let task = hook(&url, vec![]).spawn(receiver, stopping.clone());
    drop(sender.send(job(JobKind::Ingest, JobState::Running)));
    let finished = job(JobKind::Ingest, JobState::Succeeded);
    drop(sender.send(finished.clone()));
    for _ in 0..300 {
        if seen.lock().unwrap_or_else(PoisonError::into_inner).len() >= 2 {
            break;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    stopping.cancel();
    drop(task.await);
    let seen = seen.lock().unwrap_or_else(PoisonError::into_inner).clone();
    assert_eq!(
        seen.len(),
        2,
        "the running job is not reported; the finished one is retried once"
    );
    let (signature, body) = seen.last().cloned().unwrap_or_default();
    assert_eq!(signature, hook(&url, vec![]).signature(&body));
    let sent: serde_json::Value =
        serde_json::from_slice(&body).unwrap_or_else(|e| fail(&e.to_string()));
    assert_eq!(
        sent.get("job_id").and_then(serde_json::Value::as_str),
        Some(finished.id.to_string().as_ref())
    );
    assert_eq!(
        sent.get("kind").and_then(serde_json::Value::as_str),
        Some("ingest")
    );
    assert_eq!(
        sent.get("state").and_then(serde_json::Value::as_str),
        Some("succeeded")
    );
    assert_eq!(
        sent.get("workspace_id").and_then(serde_json::Value::as_str),
        Some("ws-1")
    );
    let text = String::from_utf8_lossy(&body);
    assert!(
        !text.contains("finance share") && !text.contains("ledger"),
        "{text}"
    );
}
