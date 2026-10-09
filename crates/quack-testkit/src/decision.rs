//! A loopback Ollama for tests, shared by the crates that need a decision
//! model (`quack-core` includes this file by path): `/v1/systemone`, which refuses a state past
//! a character limit the way the real route does, plus the two listings the
//! decision code reads.
//!
//! A choice is answered with the first option whose name the state
//! contains (the last option otherwise), a score with the number of `!` in
//! the state, and a true-or-false question with 0.9 when the state contains
//! "cancel" and 0.1 when it does not.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, PoisonError};

use axum::extract::State;
use axum::http::StatusCode;
use axum::routing::{get, post};
use axum::{Json, Router};
use serde_json::{Value, json};
use tokio::net::TcpListener;

/// What a decision request asked, as the stub read it.
pub struct Seen {
    /// The characters across the state's values.
    pub chars: usize,
    /// The state's values joined by a space.
    pub text: String,
    /// The questions, as the request sent them.
    pub questions: Value,
}

/// A reply other than an answer.
pub struct Fault {
    /// The HTTP status.
    pub status: u16,
    /// The JSON body.
    pub body: Value,
}

impl Fault {
    /// Ollama's error body: `{"error": text}`.
    #[must_use]
    pub fn new(status: u16, error: &str) -> Self {
        Self {
            status,
            body: json!({ "error": error }),
        }
    }

    /// Any reply at all.
    #[must_use]
    pub fn raw(status: u16, body: Value) -> Self {
        Self { status, body }
    }
}

type Rule = dyn Fn(&Seen) -> Option<Fault> + Send + Sync;

impl Seen {
    /// The reply to a state longer than `limit` characters, as Ollama words it.
    #[must_use]
    pub fn too_long(&self, limit: usize) -> Option<Fault> {
        (self.chars > limit).then(|| {
            Fault::new(
                400,
                &format!(
                    "question 0: state has {} tokens; limit is {limit} with this question",
                    self.chars
                ),
            )
        })
    }
}

/// The server; it stops when dropped.
pub struct DecisionStub {
    base_url: String,
    shared: Shared,
    task: tokio::task::JoinHandle<()>,
}

impl Drop for DecisionStub {
    fn drop(&mut self) {
        self.task.abort();
    }
}

#[derive(Clone)]
struct Shared {
    rule: Arc<Rule>,
    requests: Arc<AtomicUsize>,
    bodies: Arc<Mutex<Vec<String>>>,
    digest: Arc<Mutex<String>>,
}

impl DecisionStub {
    /// A stub that accepts every state.
    pub async fn start() -> Self {
        Self::with_rule(|_| None).await
    }

    /// A stub that refuses a state longer than `limit` characters with a 400.
    pub async fn limited(limit: usize) -> Self {
        Self::with_rule(move |seen| seen.too_long(limit)).await
    }

    /// A stub that replies `rule`'s fault to a request, or answers it.
    pub async fn with_rule(rule: impl Fn(&Seen) -> Option<Fault> + Send + Sync + 'static) -> Self {
        let shared = Shared {
            rule: Arc::new(rule),
            requests: Arc::new(AtomicUsize::new(0)),
            bodies: Arc::new(Mutex::new(Vec::new())),
            digest: Arc::new(Mutex::new(String::from("sha256:aaaa"))),
        };
        let router = Router::new()
            .route("/api/tags", get(tags))
            .route("/api/show", post(show))
            .route("/v1/systemone", post(systemone))
            .with_state(shared.clone());
        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .unwrap_or_else(|e| unreachable_io(&e));
        let addr = listener.local_addr().unwrap_or_else(|e| unreachable_io(&e));
        let task = tokio::spawn(async move {
            drop(axum::serve(listener, router).await);
        });
        Self {
            base_url: format!("http://{addr}"),
            shared,
            task,
        }
    }

    /// Where the stub listens, as a provider's `base_url`.
    #[must_use]
    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    /// How many decision requests arrived.
    #[must_use]
    pub fn requests(&self) -> usize {
        self.shared.requests.load(Ordering::SeqCst)
    }

    /// Every decision request body, in arrival order.
    #[must_use]
    pub fn bodies(&self) -> Vec<String> {
        self.shared
            .bodies
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clone()
    }

    /// The digest `/api/tags` lists for the model from now on.
    pub fn set_digest(&self, digest: &str) {
        let mut held = self
            .shared
            .digest
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        digest.clone_into(&mut held);
    }
}

#[expect(clippy::panic, reason = "test support: a loopback socket must open")]
fn unreachable_io(e: &std::io::Error) -> ! {
    panic!("loopback stub: {e}")
}

async fn tags(State(shared): State<Shared>) -> Json<Value> {
    let digest = shared
        .digest
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .clone();
    Json(
        json!({"models": [{"name": "laya:latest", "model": "laya:latest",
                            "size": 846_000_000, "digest": digest}]}),
    )
}

async fn show() -> Json<Value> {
    Json(json!({"capabilities": ["decision"]}))
}

async fn systemone(State(shared): State<Shared>, body: String) -> (StatusCode, Json<Value>) {
    shared.requests.fetch_add(1, Ordering::SeqCst);
    shared
        .bodies
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .push(body.clone());
    let request: Value = serde_json::from_str(&body).unwrap_or(Value::Null);
    let values: Vec<&str> = request
        .get("state")
        .and_then(Value::as_object)
        .map(|state| state.values().filter_map(Value::as_str).collect())
        .unwrap_or_default();
    let seen = Seen {
        chars: values.iter().map(|v| v.chars().count()).sum(),
        text: values.join(" "),
        questions: request.get("questions").cloned().unwrap_or(Value::Null),
    };
    if let Some(fault) = (shared.rule)(&seen) {
        let status = StatusCode::from_u16(fault.status).unwrap_or(StatusCode::BAD_REQUEST);
        return (status, Json(fault.body));
    }
    (
        StatusCode::OK,
        Json(json!({"model": "laya", "answers": seen.answers()})),
    )
}

impl Seen {
    /// The stub's answer to every question: the option the text names (else
    /// the last), a level per exclamation mark, a yes when the text says
    /// cancel.
    fn answers(&self) -> Value {
        let mut out = serde_json::Map::new();
        let questions = self.questions.as_object().cloned().unwrap_or_default();
        let lower = self.text.to_lowercase();
        for (name, question) in questions {
            let criteria = question.get("criteria").cloned().unwrap_or(Value::Null);
            let answer = match question.get("type").and_then(Value::as_str) {
                Some("choice") => {
                    let options: Vec<String> = criteria
                        .as_object()
                        .map(|o| o.keys().cloned().collect())
                        .unwrap_or_default();
                    let chosen = options
                        .iter()
                        .find(|o| lower.contains(&o.to_lowercase()))
                        .or_else(|| options.last())
                        .cloned()
                        .unwrap_or_default();
                    let mut probabilities = serde_json::Map::new();
                    for option in &options {
                        let p = if *option == chosen { 0.9 } else { 0.1 };
                        probabilities.insert(option.clone(), json!(p));
                    }
                    json!({"type": "choice", "choice": chosen,
                       "probabilities": probabilities, "confidence": 0.8})
                }
                Some("score") => {
                    let levels = criteria.as_array().map_or(0, Vec::len);
                    let exclaimed = self.text.matches('!').count();
                    let level = exclaimed.min(levels.saturating_sub(1));
                    let mut probabilities = serde_json::Map::new();
                    for at in 0..levels {
                        let p = if at == level { 0.8 } else { 0.05 };
                        probabilities.insert(at.to_string(), json!(p));
                    }
                    json!({"type": "score", "score": f64::from(u32::try_from(level).unwrap_or(0)),
                       "probabilities": probabilities, "confidence": 0.7})
                }
                _ => {
                    let p = if lower.contains("cancel") { 0.9 } else { 0.1 };
                    json!({"type": "noul", "noul": p})
                }
            };
            out.insert(name, answer);
        }
        Value::Object(out)
    }
}
