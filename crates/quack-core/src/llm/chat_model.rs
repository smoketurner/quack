//! Per-model request settings the agent and schema-call builders cannot know,
//! applied to rig's `AgentBuilder` by [`ChatModel::agent`].
//!
//! The agent asks for temperature 0.1 and the schema calls for 0.0. Only
//! Ollama's own API gets it. Current Claude models (Opus 4.7 and later,
//! Sonnet 5) answer a non-default `temperature` with a 400, and so do
//! `OpenAI`'s reasoning models (GPT-5.x, GPT-6, the o-series) at any effort
//! but `none`; every API accepts a request without one, and a gateway's
//! model name need not say which model is behind it.
//!
//! Claude also needs `max_tokens` on every request: rig's Anthropic client
//! knows the Claude 4 and 5 families by id, but refuses to send a request for
//! any other model without one (a gateway alias, a deployment name), and
//! Bedrock's Converse default is sized for replies, not for adaptive
//! thinking, which counts against it.
//!
//! `[analysis].effort` and `background_effort` go out as rig's typed
//! reasoning option: rig writes the field each API takes, and a level the
//! model lacks is refused when the model is built ([`ChatModel::new`]).
//!
//! `temperature`, `effort`, and `background_effort` on `[providers.NAME]`,
//! or on `[providers.NAME.models."ID"]` for one model, replace these
//! defaults and `[analysis]`'s efforts (`config::ModelSettings`).

use rig::AgentBuilder;
use rig::DynModel;
use rig::completion::{CompletionRequest, Message, options};
use rig::driver::Transport;
use rig::operation::Completion;
use rig::providers::openai::extension::OpenAiOptions;

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
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChatSettings {
    /// The API the model is called through.
    wire: Wire,
    /// Whether the requested `temperature` is sent.
    temperature: bool,
    /// The `max_tokens` every request is given.
    max_tokens: Option<u64>,
    /// The configured effort, sent as rig's typed reasoning option.
    effort: Option<Effort>,
}

impl ChatSettings {
    /// The settings for `model`, a model id as the provider takes it
    /// (`claude-opus-5-5`, `us.anthropic.claude-opus-5-5`, `gpt-5.6-sol`,
    /// `openai/gpt-5.6-sol`), called through `wire` at `effort`. `temperature`
    /// is what the provider's settings say; unset, only Ollama gets one.
    #[must_use]
    pub fn new(model: &str, wire: Wire, effort: Option<Effort>, temperature: Option<bool>) -> Self {
        Self {
            wire,
            temperature: temperature.unwrap_or(wire == Wire::Ollama),
            max_tokens: base_id(model)
                .starts_with("claude-")
                .then_some(CLAUDE_MAX_TOKENS),
            effort,
        }
    }
}

impl std::fmt::Display for ChatSettings {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(if self.temperature {
            "sends temperature"
        } else {
            "sends no temperature"
        })?;
        match self.effort {
            Some(effort) => write!(f, ", reasoning effort {effort}"),
            None => f.write_str(", no reasoning effort"),
        }?;
        if let Some(max_tokens) = self.max_tokens {
            write!(f, ", max_tokens {max_tokens}")?;
        }
        Ok(())
    }
}

/// A chat model with its wire and transport erased, and what every request
/// to it carries.
pub struct ChatModel {
    model: DynModel<Completion>,
    settings: ChatSettings,
}

impl ChatModel {
    /// `model`, the model named `id`, sent what `settings` says.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Config`] when rig refuses the effort for the model on
    /// its API: rig decides from its model catalog which levels the model
    /// takes, and the setting fails here rather than on the first request.
    pub fn new<W, T>(model: rig::Model<W, T>, id: &str, settings: ChatSettings) -> Result<Self>
    where
        W: rig::wire::Wire<Op = Completion>,
        T: Transport<W>,
    {
        if let (Some(effort), Some(target)) = (settings.effort, model.wire.describe().replay) {
            let mut probe = CompletionRequest::new(Message::user("")).reasoning(effort);
            options::check(target, &mut probe).map_err(|e| {
                let why = e
                    .unsupported_option()
                    .map_or_else(|| e.to_string(), |refused| refused.reason.clone());
                Error::Config(format!("effort \"{effort}\" for {id}: {why}"))
            })?;
        }
        Ok(Self {
            model: model.erase(),
            settings,
        })
    }

    /// A rig agent on the model with its settings applied; `temperature` is
    /// sent only where [`ChatSettings`] says the model takes one.
    #[must_use]
    pub fn agent(self, temperature: f64) -> AgentBuilder {
        let ChatSettings {
            wire,
            temperature: sends_temperature,
            max_tokens,
            effort,
        } = self.settings;
        let mut agent = AgentBuilder::new(self.model);
        if sends_temperature {
            agent = agent.temperature(temperature);
        }
        if let Some(max_tokens) = max_tokens {
            agent = agent.max_tokens(max_tokens);
        }
        if let Some(effort) = effort {
            agent = agent.reasoning(effort);
        }
        if wire == Wire::Responses {
            // Neither Bedrock nor `OpenAI` keeps a copy of the conversation
            // (Bedrock keeps one for 30 days by default), so no workspace
            // content is stored outside the workspace file (design doc
            // section 5). quack replays history itself and never uses
            // `previous_response_id`.
            agent = agent.provider_option(OpenAiOptions::default().store(false));
        }
        agent
    }
}

/// A model with no per-model rules, such as a scripted test model: the
/// requested temperature is sent, and nothing else is added.
impl From<DynModel<Completion>> for ChatModel {
    fn from(model: DynModel<Completion>) -> Self {
        Self {
            model,
            settings: ChatSettings {
                wire: Wire::ChatCompletions,
                temperature: true,
                max_tokens: None,
                effort: None,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[expect(clippy::panic, reason = "test failure path")]
    fn fail(msg: &str) -> ! {
        panic!("{msg}")
    }

    fn settings(model: &str, wire: Wire) -> ChatSettings {
        ChatSettings::new(model, wire, None, None)
    }

    #[test]
    fn claude_gets_no_temperature_and_an_output_budget_on_every_host() {
        for (id, wire) in [
            ("claude-opus-5-5", Wire::Anthropic),
            ("us.anthropic.claude-opus-5-5", Wire::Converse),
            ("global.anthropic.claude-opus-5-5", Wire::Converse),
            ("anthropic.claude-sonnet-5", Wire::Converse),
            ("anthropic/claude-opus-5-5", Wire::ChatCompletions),
            ("Claude-Opus-5-5", Wire::Anthropic),
        ] {
            let chat = settings(id, wire);
            assert!(!chat.temperature, "{id}");
            assert_eq!(chat.max_tokens, Some(CLAUDE_MAX_TOKENS), "{id}");
        }
    }

    #[test]
    fn only_ollama_gets_a_temperature_by_default() {
        for wire in [
            Wire::ChatCompletions,
            Wire::Responses,
            Wire::Anthropic,
            Wire::Converse,
        ] {
            for id in ["corp-model", "gpt-oss:20b", "gpt-5.6-sol", "o3"] {
                let chat = settings(id, wire);
                assert!(!chat.temperature, "{id} {wire:?}");
                assert_eq!(chat.max_tokens, None, "{id} {wire:?}");
            }
        }
        for id in ["corp-model", "gpt-oss:20b", "llama3.1:8b"] {
            assert!(settings(id, Wire::Ollama).temperature, "{id}");
        }
    }

    #[test]
    fn a_declared_temperature_overrides_the_default() {
        assert!(!ChatSettings::new("qwen3:32b", Wire::Ollama, None, Some(false)).temperature);
        assert!(ChatSettings::new("gpt-5.2", Wire::Responses, None, Some(true)).temperature);
    }

    #[test]
    fn display_names_what_is_sent() {
        assert_eq!(
            ChatSettings::new("claude-opus-5-5", Wire::Anthropic, Some(Effort::High), None)
                .to_string(),
            "sends no temperature, reasoning effort high, max_tokens 64000"
        );
        assert_eq!(
            settings("llama3.1:8b", Wire::Ollama).to_string(),
            "sends temperature, no reasoning effort"
        );
    }

    #[test]
    fn each_effort_is_rig_s_reasoning_level() {
        use rig::completion::{Effort as Level, Reasoning};
        for (effort, reasoning) in [
            (Effort::None, Reasoning::Off),
            (Effort::Minimal, Reasoning::Effort(Level::Minimal)),
            (Effort::Low, Reasoning::Effort(Level::Low)),
            (Effort::Medium, Reasoning::Effort(Level::Medium)),
            (Effort::High, Reasoning::Effort(Level::High)),
            (Effort::Xhigh, Reasoning::Effort(Level::XHigh)),
            (Effort::Max, Reasoning::Effort(Level::Max)),
        ] {
            assert_eq!(Reasoning::from(effort), reasoning, "{effort}");
        }
    }

    #[test]
    fn gpt_5_6_tool_calls_need_responses_or_no_reasoning() {
        for id in [
            "gpt-5.6-sol",
            "us.openai.gpt-5.6-terra",
            "openai.gpt-5.6-luna",
            "Azure/GPT-5.6-Sol",
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
