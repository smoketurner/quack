//! What the decision tests share: a config that points at the stub, the
//! model it serves, and the scope `quack doctor` runs in.

use super::DecisionModel;
use super::stub::DecisionStub;
use crate::config::Config;
use crate::llm::egress::Egress;

/// A config whose `[decision].model` is `local/laya` on the Ollama at `base`.
pub(crate) fn config(base: &str) -> Config {
    Config::parse(&format!(
        "[providers.local]\ntype = \"ollama\"\nbase_url = \"{base}\"\nmax_retries = 0\n\
         [decision]\nmodel = \"local/laya\"\n"
    ))
    .unwrap_or_else(|e| failed(&e.to_string()))
}

/// The decision model the stub serves.
pub(crate) async fn model(stub: &DecisionStub) -> DecisionModel {
    DecisionModel::from_config(&config(stub.base_url()))
        .await
        .ok()
        .flatten()
        .unwrap_or_else(|| failed("no decision model"))
}

/// Run `work` as `quack doctor` does, with no workspace behind it.
pub(crate) async fn scoped<T>(work: impl Future<Output = T>) -> T {
    Egress::scope(Some(Egress::NoWorkspace), work).await
}

#[expect(
    clippy::panic,
    reason = "test support: the fixture config must be valid"
)]
fn failed(why: &str) -> ! {
    panic!("decision fixture: {why}")
}
