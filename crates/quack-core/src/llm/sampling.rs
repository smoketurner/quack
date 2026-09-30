//! Per-model request settings the agent and one-shot builders cannot know.
//!
//! The agent asks for temperature 0.1 and the one-shot calls for 0.0. Only
//! Ollama's own API gets it. Current Claude models (Opus 4.7 and later,
//! Sonnet 5) answer a non-default `temperature` with a 400, and so do
//! `OpenAI`'s reasoning models (GPT-5.x, GPT-6, the o-series) at any effort
//! but `none`; every API accepts a request without one, and a gateway's
//! model name need not say which model is behind it.
//!
//! Claude also needs `max_tokens` on every request: rig knows the Claude 4
//! families only and refuses to send a request for any other model without
//! one, and Bedrock's Converse default is sized for replies, not for
//! adaptive thinking, which counts against it.
//!
//! The same module turns `[analysis].effort` and `background_effort` into the
//! field each API takes, and refuses a level the model family lacks before
//! any request is sent. A model id quack does not recognize (a gateway
//! alias, a deployment name, an open-weight model on vLLM) gets the effort
//! field on Chat Completions and Responses, where the server judges the
//! level.
//!
//! `temperature`, `effort`, and `background_effort` on `[providers.NAME]`,
//! or on `[providers.NAME.models."ID"]` for one model, replace these
//! defaults and `[analysis]`'s efforts (`config::ModelSettings`).
//!
//! [`Sampled`] wraps every chat model and applies [`Sampling`] for the model
//! id and the API it is called through, so the rules hold on every provider
//! that hosts the model: the Anthropic API, Bedrock's Converse and
//! OpenAI-compatible APIs, `OpenAI` itself, and any OpenAI-compatible gateway.

use serde_json::{Map, Value, json};

use crate::config::{BedrockApi, Effort, ProviderConfig, ProviderType};
use crate::error::{Error, Result};

/// The output budget of a Claude model. Thinking counts against it, so it
/// is sized for a streamed turn rather than for the answer text alone.
pub const CLAUDE_MAX_TOKENS: u64 = 64_000;

/// The API a chat model is called through.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Wire {
    /// Ollama's own chat API.
    Ollama,
    /// OpenAI-compatible Chat Completions (`OpenAI`, Bedrock, gateways).
    ChatCompletions,
    /// OpenAI-compatible Responses (`OpenAI`, Bedrock).
    Responses,
    /// Anthropic's Messages API.
    Anthropic,
    /// Bedrock's Converse API.
    Converse,
}

impl Wire {
    /// The API `provider`'s chat model is called through.
    #[must_use]
    pub fn of(provider: &ProviderConfig) -> Self {
        let api = match provider.provider_type {
            ProviderType::Ollama => return Self::Ollama,
            ProviderType::Anthropic => return Self::Anthropic,
            ProviderType::Openai => provider.openai_chat_api(),
            ProviderType::Bedrock | ProviderType::BedrockMantle => match &provider.bedrock {
                Some(bedrock) => bedrock.api,
                None => return Self::Converse,
            },
        };
        match api {
            BedrockApi::Converse => Self::Converse,
            BedrockApi::ChatCompletions => Self::ChatCompletions,
            BedrockApi::Responses => Self::Responses,
        }
    }
}

/// The model families whose requests quack shapes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Family {
    Claude,
    /// GPT-5.x, GPT-6, and the o-series.
    OpenAiReasoning,
    /// `OpenAI`'s open-weight gpt-oss models.
    GptOss,
    Other,
}

impl Family {
    /// The family of `id`, a lowercase [`base_id`].
    fn of(id: &str) -> Self {
        if id.starts_with("claude-") {
            Self::Claude
        } else if id.starts_with("gpt-oss") {
            Self::GptOss
        } else if OPENAI_REASONING.iter().any(|p| id.starts_with(p)) {
            Self::OpenAiReasoning
        } else {
            Self::Other
        }
    }

    /// The effort levels the family takes on `wire`; empty where quack sends
    /// none.
    fn efforts(self, wire: Wire) -> &'static [Effort] {
        use Effort::{High, Low, Max, Medium, Minimal, None, Xhigh};
        match (self, wire) {
            (Self::Claude, Wire::Anthropic | Wire::Converse) => &[Low, Medium, High, Xhigh, Max],
            // An unrecognized model's levels vary by model and server, so
            // the server judges them.
            (Self::OpenAiReasoning, Wire::Responses)
            | (Self::Other, Wire::ChatCompletions | Wire::Responses) => {
                &[None, Minimal, Low, Medium, High, Xhigh, Max]
            }
            (Self::OpenAiReasoning, Wire::ChatCompletions) => {
                &[None, Minimal, Low, Medium, High, Xhigh]
            }
            (Self::GptOss, Wire::Ollama | Wire::ChatCompletions | Wire::Responses) => {
                &[Low, Medium, High]
            }
            _ => &[],
        }
    }
}

/// `OpenAI` reasoning model families.
const OPENAI_REASONING: [&str; 5] = ["gpt-5", "gpt-6", "o1", "o3", "o4"];

/// Bedrock's cross-region prefixes, taken off before the vendor prefix.
const REGION_PREFIXES: [&str; 8] = [
    "us.", "eu.", "au.", "jp.", "in.", "apac.", "global.", "us-gov.",
];

/// Vendor prefixes in Bedrock model ids.
const VENDOR_PREFIXES: [&str; 2] = ["anthropic.", "openai."];

/// `model` in lowercase, without a gateway's `vendor/` path, a Bedrock
/// region prefix, or a Bedrock vendor prefix.
fn base_id(model: &str) -> String {
    let model = model.to_ascii_lowercase();
    let mut id = model.rsplit('/').next().unwrap_or(&model);
    if let Some(rest) = REGION_PREFIXES.iter().find_map(|p| id.strip_prefix(p)) {
        id = rest;
    }
    if let Some(rest) = VENDOR_PREFIXES.iter().find_map(|p| id.strip_prefix(p)) {
        id = rest;
    }
    id.to_owned()
}

/// Refuse a chat turn that cannot work: GPT-5.6 models reject function tools
/// on Chat Completions unless reasoning effort is `none`, and every chat turn
/// sends tools. One-shot calls send none and are not checked.
///
/// # Errors
///
/// Returns [`Error::Config`] naming the fix for that combination.
pub fn check_tool_calls(model: &str, wire: Wire, effort: Option<Effort>) -> Result<()> {
    if wire == Wire::ChatCompletions
        && base_id(model).starts_with("gpt-5.6")
        && effort != Some(Effort::None)
    {
        return Err(Error::Config(format!(
            "{model} cannot call tools through Chat Completions unless its effort is \
             \"none\"; set api = \"responses\" on its provider, or effort = \"none\" on \
             the model, its provider, or [analysis] (the first one set wins)"
        )));
    }
    Ok(())
}

/// What a request to one model carries.
#[derive(Debug, Clone, PartialEq)]
pub struct Sampling {
    /// Whether the requested `temperature` is sent.
    temperature: bool,
    /// The `max_tokens` a request that sets none is given.
    max_tokens: Option<u64>,
    /// Fields merged into the request body for the configured effort.
    effort: Option<Map<String, Value>>,
    /// Why a configured effort is not sent, when it is not.
    unsent_effort: Option<String>,
}

impl Sampling {
    /// The rule for `model`, a model id as the provider takes it
    /// (`claude-opus-5-5`, `us.anthropic.claude-opus-5-5`, `gpt-5.6-sol`,
    /// `openai/gpt-5.6-sol`), called through `wire` at `effort`. `temperature`
    /// is what the provider's settings say; unset, only Ollama gets one.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Config`] when `effort` is a level the model does not
    /// take on `wire`.
    pub fn new(
        model: &str,
        wire: Wire,
        effort: Option<Effort>,
        temperature: Option<bool>,
    ) -> Result<Self> {
        let family = Family::of(&base_id(model));
        let accepted = family.efforts(wire);
        let mut unsent_effort = None;
        let effort = match effort {
            None => None,
            Some(effort) if accepted.is_empty() => {
                unsent_effort = Some(Self::unsent(model, family, wire, effort));
                None
            }
            Some(effort) => Some(effort_fields(model, family, wire, effort, accepted)?),
        };
        let temperature = temperature.unwrap_or(wire == Wire::Ollama);
        Ok(Self {
            temperature,
            max_tokens: (family == Family::Claude).then_some(CLAUDE_MAX_TOKENS),
            effort,
            unsent_effort,
        })
    }

    fn unsent(model: &str, family: Family, wire: Wire, effort: Effort) -> String {
        if family == Family::OpenAiReasoning && wire == Wire::Converse {
            format!(
                "effort \"{effort}\" is not sent to {model}: OpenAI models take none through \
                 Converse; use api = \"responses\""
            )
        } else {
            format!(
                "effort \"{effort}\" is not sent to {model}: quack knows no reasoning effort \
                 field for it on this API"
            )
        }
    }

    /// Why the configured effort is not sent, when it is not.
    #[must_use]
    pub fn unsent_effort(&self) -> Option<&str> {
        self.unsent_effort.as_deref()
    }

    fn apply(
        &self,
        mut request: rig::completion::CompletionRequest,
    ) -> rig::completion::CompletionRequest {
        if !self.temperature {
            request.temperature = None;
        }
        if request.max_tokens.is_none() {
            request.max_tokens = self.max_tokens;
        }
        if let Some(effort) = &self.effort {
            let mut params = match request.additional_params.take() {
                Some(Value::Object(map)) => map,
                _ => Map::new(),
            };
            merge(&mut params, effort);
            request.additional_params = Some(Value::Object(params));
        }
        request
    }
}

impl std::fmt::Display for Sampling {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(if self.temperature {
            "sends temperature"
        } else {
            "sends no temperature"
        })?;
        match &self.effort {
            Some(fields) => write!(f, ", reasoning effort as {}", Value::Object(fields.clone())),
            None => f.write_str(", no reasoning effort"),
        }?;
        if let Some(max_tokens) = self.max_tokens {
            write!(f, ", max_tokens {max_tokens}")?;
        }
        Ok(())
    }
}

/// The body fields for `effort` on `wire`, once `accepted` has it.
fn effort_fields(
    model: &str,
    family: Family,
    wire: Wire,
    effort: Effort,
    accepted: &[Effort],
) -> Result<Map<String, Value>> {
    if !accepted.contains(&effort) {
        let levels: Vec<&str> = accepted.iter().map(|e| e.as_str()).collect();
        return Err(Error::Config(format!(
            "effort \"{effort}\" is not a level {model} takes here; use one of: {}",
            levels.join(", ")
        )));
    }
    let level = effort.as_str();
    let mut fields = Map::new();
    match (family, wire) {
        (Family::Claude, _) => {
            fields.insert(String::from("output_config"), json!({ "effort": level }))
        }
        (_, Wire::Ollama) => fields.insert(String::from("think"), json!(level)),
        (_, Wire::Responses) => {
            fields.insert(String::from("reasoning"), json!({ "effort": level }))
        }
        _ => fields.insert(String::from("reasoning_effort"), json!(level)),
    };
    Ok(fields)
}

/// Merge `extra` into `params`, one level of objects deep, so an existing
/// `reasoning` or `output_config` object keeps its other fields.
fn merge(params: &mut Map<String, Value>, extra: &Map<String, Value>) {
    for (key, value) in extra {
        match (params.get_mut(key), value) {
            (Some(Value::Object(existing)), Value::Object(add)) => {
                for (k, v) in add {
                    existing.insert(k.clone(), v.clone());
                }
            }
            _ => {
                params.insert(key.clone(), value.clone());
            }
        }
    }
}

/// A chat model whose requests follow its [`Sampling`].
#[derive(Clone)]
pub struct Sampled<M> {
    model: M,
    sampling: Sampling,
}

impl<M> Sampled<M> {
    /// `model`, served as the model named `id` through `wire` at `effort`,
    /// sent `temperature` as [`Sampling::new`] decides. An effort that is
    /// not sent is logged as a warning, since the setting then does nothing.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Config`] when `effort` is a level the model does not
    /// take on `wire`.
    pub fn new(
        model: M,
        id: &str,
        wire: Wire,
        effort: Option<Effort>,
        temperature: Option<bool>,
    ) -> Result<Self> {
        let sampling = Sampling::new(id, wire, effort, temperature)?;
        if let Some(why) = sampling.unsent_effort() {
            tracing::warn!("{why}");
        }
        Ok(Self { model, sampling })
    }
}

impl<M: rig::completion::CompletionModel> rig::completion::CompletionModel for Sampled<M> {
    fn completion(
        &self,
        request: rig::completion::CompletionRequest,
    ) -> impl std::future::Future<
        Output = std::result::Result<
            rig::completion::CompletionResponse,
            rig::completion::CompletionError,
        >,
    > + Send {
        self.model.completion(self.sampling.apply(request))
    }

    fn stream(
        &self,
        request: rig::completion::CompletionRequest,
    ) -> impl std::future::Future<
        Output = std::result::Result<
            rig::streaming::StreamingCompletionResponse,
            rig::completion::CompletionError,
        >,
    > + Send {
        self.model.stream(self.sampling.apply(request))
    }

    fn capabilities(&self) -> rig::completion::ProviderCapabilities {
        self.model.capabilities()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[expect(clippy::panic, reason = "test failure path")]
    fn fail(msg: &str) -> ! {
        panic!("{msg}")
    }

    fn request() -> rig::completion::CompletionRequest {
        rig::completion::CompletionRequest {
            model: None,
            preamble: None,
            chat_history: Vec::new(),
            documents: Vec::new(),
            tools: Vec::new(),
            temperature: Some(0.1),
            max_tokens: None,
            tool_choice: None,
            additional_params: None,
            output_schema: None,
            record_telemetry_content: false,
        }
    }

    fn sampling(model: &str, wire: Wire, effort: Option<Effort>) -> Sampling {
        Sampling::new(model, wire, effort, None).unwrap_or_else(|e| fail(&format!("{model}: {e}")))
    }

    fn sent(model: &str, wire: Wire, effort: Option<Effort>) -> rig::completion::CompletionRequest {
        sampling(model, wire, effort).apply(request())
    }

    #[test]
    fn claude_gets_no_temperature_and_an_output_budget_on_every_host() {
        for (id, wire) in [
            ("claude-opus-5-5", Wire::Anthropic),
            ("us.anthropic.claude-opus-5-5", Wire::Converse),
            ("global.anthropic.claude-opus-5-5", Wire::Converse),
            ("anthropic.claude-sonnet-5", Wire::Converse),
            ("anthropic/claude-opus-5-5", Wire::ChatCompletions),
        ] {
            let sent = sent(id, wire, None);
            assert_eq!(sent.temperature, None, "{id}");
            assert_eq!(sent.max_tokens, Some(CLAUDE_MAX_TOKENS), "{id}");
            assert_eq!(sent.additional_params, None, "{id}");
        }
    }

    #[test]
    fn openai_reasoning_models_get_neither() {
        for id in [
            "gpt-5.6-sol",
            "us.openai.gpt-5.6-terra",
            "in.openai.gpt-5.6-luna",
            "openai.gpt-6-astra",
            "global.openai.gpt-6-sol",
            "openai/gpt-6-luna",
            "gpt-5.5",
            "o3",
        ] {
            let sent = sent(id, Wire::Responses, None);
            assert_eq!(sent.temperature, None, "{id}");
            assert_eq!(sent.max_tokens, None, "{id}");
        }
    }

    #[test]
    fn other_models_keep_the_requested_temperature() {
        for id in [
            "gpt-oss:20b",
            "openai.gpt-oss-120b",
            "llama3.1:8b",
            "qwen3:32b",
        ] {
            let sent = sent(id, Wire::Ollama, None);
            assert_eq!(sent.temperature, Some(0.1), "{id}");
            assert_eq!(sent.max_tokens, None, "{id}");
        }
    }

    #[test]
    fn a_request_that_sets_max_tokens_keeps_it() {
        let mut asked = request();
        asked.max_tokens = Some(1024);
        let sampling = sampling("claude-opus-5-5", Wire::Anthropic, None);
        assert_eq!(sampling.apply(asked).max_tokens, Some(1024));
    }

    #[test]
    fn effort_takes_each_api_s_field() {
        let cases = [
            (
                "claude-opus-5-5",
                Wire::Anthropic,
                Effort::Xhigh,
                json!({ "output_config": { "effort": "xhigh" } }),
            ),
            (
                "us.anthropic.claude-opus-5-5",
                Wire::Converse,
                Effort::Low,
                json!({ "output_config": { "effort": "low" } }),
            ),
            (
                "gpt-5.6-sol",
                Wire::Responses,
                Effort::Max,
                json!({ "reasoning": { "effort": "max" } }),
            ),
            (
                "openai.gpt-6-luna",
                Wire::ChatCompletions,
                Effort::None,
                json!({ "reasoning_effort": "none" }),
            ),
            (
                "gpt-oss:20b",
                Wire::Ollama,
                Effort::High,
                json!({ "think": "high" }),
            ),
            (
                "openai.gpt-oss-120b",
                Wire::Responses,
                Effort::Medium,
                json!({ "reasoning": { "effort": "medium" } }),
            ),
        ];
        for (id, wire, effort, fields) in cases {
            assert_eq!(
                sent(id, wire, Some(effort)).additional_params,
                Some(fields),
                "{id}"
            );
        }
    }

    #[test]
    fn effort_keeps_the_fields_already_set() {
        let mut asked = request();
        asked.additional_params =
            Some(json!({ "store": false, "reasoning": { "summary": "auto" } }));
        assert_eq!(
            sampling("gpt-5.6-sol", Wire::Responses, Some(Effort::High))
                .apply(asked)
                .additional_params,
            Some(json!({ "store": false, "reasoning": { "summary": "auto", "effort": "high" } }))
        );
        let mut asked = request();
        asked.additional_params = Some(json!({ "num_ctx": 8192, "keep_alive": "30m" }));
        assert_eq!(
            sampling("gpt-oss:20b", Wire::Ollama, Some(Effort::Low))
                .apply(asked)
                .additional_params,
            Some(json!({ "num_ctx": 8192, "keep_alive": "30m", "think": "low" }))
        );
    }

    #[test]
    fn a_level_the_model_lacks_is_refused() {
        for (id, wire, effort) in [
            ("claude-opus-5-5", Wire::Anthropic, Effort::None),
            (
                "us.anthropic.claude-opus-5-5",
                Wire::Converse,
                Effort::Minimal,
            ),
            ("gpt-5.6-sol", Wire::ChatCompletions, Effort::Max),
            ("gpt-oss:20b", Wire::Ollama, Effort::Xhigh),
        ] {
            let Err(e) = Sampling::new(id, wire, Some(effort), None) else {
                fail(&format!("{id} took {effort}"))
            };
            let message = e.to_string();
            assert!(message.contains(id), "{message}");
            assert!(message.contains(effort.as_str()), "{message}");
        }
    }

    #[test]
    fn models_quack_sends_no_effort_to_take_any_level() {
        for (id, wire) in [
            ("llama3.1:8b", Wire::Ollama),
            ("us.openai.gpt-5.6-sol", Wire::Converse),
        ] {
            assert_eq!(
                sent(id, wire, Some(Effort::Max)).additional_params,
                None,
                "{id}"
            );
        }
    }

    #[test]
    fn gpt_5_6_tool_calls_need_responses_or_no_reasoning() {
        for id in [
            "gpt-5.6-sol",
            "us.openai.gpt-5.6-terra",
            "openai.gpt-5.6-luna",
        ] {
            let Err(e) = check_tool_calls(id, Wire::ChatCompletions, None) else {
                fail(&format!("{id} on chat completions"))
            };
            assert!(e.to_string().contains("api = \"responses\""), "{e}");
            assert!(check_tool_calls(id, Wire::ChatCompletions, Some(Effort::None)).is_ok());
            assert!(check_tool_calls(id, Wire::Responses, Some(Effort::High)).is_ok());
        }
        assert!(check_tool_calls("gpt-6-sol", Wire::ChatCompletions, None).is_ok());
    }

    #[test]
    fn model_ids_match_in_any_case() {
        for id in [
            "GPT-5.6-Sol",
            "Azure/GPT-5.6",
            "OpenAI.GPT-6-Luna",
            "O3-Mini",
        ] {
            let family = Sampling::new(id, Wire::ChatCompletions, Some(Effort::Max), None);
            assert!(family.is_err(), "{id} is an OpenAI reasoning model");
        }
        let sent = sent("Claude-Opus-5-5", Wire::Anthropic, None);
        assert_eq!(sent.max_tokens, Some(CLAUDE_MAX_TOKENS));
        assert!(check_tool_calls("Azure/GPT-5.6-Sol", Wire::ChatCompletions, None).is_err());
    }

    #[test]
    fn only_ollama_gets_a_temperature_by_default() {
        for wire in [
            Wire::ChatCompletions,
            Wire::Responses,
            Wire::Anthropic,
            Wire::Converse,
        ] {
            for id in ["corp-model", "gpt-oss:20b", "qwen3:32b"] {
                assert_eq!(sent(id, wire, None).temperature, None, "{id} {wire:?}");
            }
        }
        assert_eq!(
            sent("corp-model", Wire::Ollama, None).temperature,
            Some(0.1)
        );
    }

    #[test]
    fn an_unknown_model_given_an_effort_gets_it() {
        for (wire, fields) in [
            (
                Wire::Responses,
                json!({ "reasoning": { "effort": "medium" } }),
            ),
            (
                Wire::ChatCompletions,
                json!({ "reasoning_effort": "medium" }),
            ),
        ] {
            let sampling = sampling("corp-reasoner", wire, Some(Effort::Medium));
            assert_eq!(sampling.unsent_effort(), None);
            let asked = sampling.apply(request());
            assert_eq!(asked.temperature, None, "{wire:?}");
            assert_eq!(asked.additional_params, Some(fields), "{wire:?}");
            let plain = sent("corp-reasoner", wire, None);
            assert_eq!(plain.additional_params, None, "{wire:?}");
        }
        let max = sent("deepseek-v4-pro", Wire::ChatCompletions, Some(Effort::Max));
        assert_eq!(
            max.additional_params,
            Some(json!({ "reasoning_effort": "max" }))
        );
    }

    #[test]
    fn an_effort_that_is_not_sent_says_why() {
        let note = |model: &str, wire: Wire| {
            sampling(model, wire, Some(Effort::Low))
                .unsent_effort()
                .map_or_else(|| fail(&format!("{model}: no note")), str::to_owned)
        };
        let llama = note("llama3.1:8b", Wire::Ollama);
        assert!(llama.contains("no reasoning effort field"), "{llama}");
        let converse = note("us.openai.gpt-5.6-sol", Wire::Converse);
        assert!(converse.contains("api = \"responses\""), "{converse}");
        let claude = note("anthropic/claude-opus-5-5", Wire::ChatCompletions);
        assert!(claude.contains("claude-opus-5-5"), "{claude}");
        let unsent = sampling("llama3.1:8b", Wire::Ollama, Some(Effort::High));
        assert_eq!(unsent.apply(request()).temperature, Some(0.1));
        assert_eq!(
            sampling("llama3.1:8b", Wire::Ollama, None).unsent_effort(),
            None
        );
    }

    #[test]
    fn a_declared_temperature_overrides_the_default() {
        let declared = |model: &str, wire: Wire, effort: Option<Effort>, temperature: bool| {
            Sampling::new(model, wire, effort, Some(temperature))
                .unwrap_or_else(|e| fail(&format!("{model}: {e}")))
                .apply(request())
        };
        assert_eq!(
            declared("qwen3:32b", Wire::Ollama, None, false).temperature,
            None
        );
        let asked = declared("gpt-5.2", Wire::Responses, Some(Effort::None), true);
        assert_eq!(asked.temperature, Some(0.1));
        assert_eq!(
            asked.additional_params,
            Some(json!({ "reasoning": { "effort": "none" } }))
        );
    }

    #[test]
    fn display_names_what_is_sent() {
        assert_eq!(
            sampling("claude-opus-5-5", Wire::Anthropic, Some(Effort::High)).to_string(),
            "sends no temperature, reasoning effort as {\"output_config\":{\"effort\":\"high\"}}, \
             max_tokens 64000"
        );
        assert_eq!(
            sampling("llama3.1:8b", Wire::Ollama, None).to_string(),
            "sends temperature, no reasoning effort"
        );
    }

    #[test]
    fn the_wire_follows_the_provider_s_type_and_api() {
        use crate::config::BedrockConfig;
        let openai = |api| ProviderConfig {
            openai_api: api,
            ..ProviderConfig::new(ProviderType::Openai)
        };
        assert_eq!(Wire::of(&openai(None)), Wire::Responses);
        assert_eq!(
            Wire::of(&openai(Some(BedrockApi::ChatCompletions))),
            Wire::ChatCompletions
        );
        assert_eq!(
            Wire::of(&ProviderConfig::new(ProviderType::Ollama)),
            Wire::Ollama
        );
        assert_eq!(
            Wire::of(&ProviderConfig::new(ProviderType::Anthropic)),
            Wire::Anthropic
        );
        assert_eq!(
            Wire::of(&ProviderConfig::new(ProviderType::Bedrock)),
            Wire::Converse
        );
        let mantle = ProviderConfig {
            bedrock: Some(BedrockConfig {
                api: BedrockApi::Responses,
                region: None,
            }),
            ..ProviderConfig::new(ProviderType::BedrockMantle)
        };
        assert_eq!(Wire::of(&mantle), Wire::Responses);
    }
}
