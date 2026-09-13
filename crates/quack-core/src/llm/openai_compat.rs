use reqwest::header::{AUTHORIZATION, HeaderMap, HeaderValue};
use serde::{Deserialize, Serialize};

use crate::config::ProviderConfig;
use crate::error::{Error, Result};
use crate::llm::{
    CompletionRequest, CompletionResponse, EmbeddingProvider, FinishReason, FunctionCall,
    LlmProvider, TokenUsage, ToolCall,
};

/// Client for OpenAI-compatible APIs (`/v1/chat/completions` + `/v1/embeddings`).
///
/// Covers `OpenAI`, Ollama, vLLM, `LiteLLM`, and most API gateways.
pub struct OpenAiCompatClient {
    http: reqwest::Client,
    base_url: String,
    chat_model: String,
    embedding_model: String,
    dimension: u32,
}

// --- Embedding API types ---

#[derive(Serialize)]
struct EmbeddingRequest<'a> {
    model: &'a str,
    input: &'a [&'a str],
}

#[derive(Deserialize)]
struct EmbeddingResponse {
    data: Vec<EmbeddingData>,
}

#[derive(Deserialize)]
struct EmbeddingData {
    embedding: Vec<f32>,
}

// --- Chat Completions API types ---

#[derive(Serialize)]
struct ChatRequest<'a> {
    model: &'a str,
    messages: &'a [ChatMessage],
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<&'a [crate::llm::ToolDefinition]>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
}

#[derive(Serialize)]
struct ChatMessage {
    role: String,
    content: serde_json::Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_call_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<ApiToolCall>>,
}

#[derive(Serialize, Deserialize)]
struct ApiToolCall {
    id: String,
    #[serde(rename = "type")]
    call_type: String,
    function: ApiFunction,
}

#[derive(Serialize, Deserialize)]
struct ApiFunction {
    name: String,
    arguments: String,
}

#[derive(Deserialize)]
struct ChatResponse {
    choices: Vec<ChatChoice>,
    #[serde(default)]
    usage: Option<ApiUsage>,
}

#[derive(Deserialize)]
struct ChatChoice {
    message: ChatChoiceMessage,
    finish_reason: Option<String>,
}

#[derive(Deserialize)]
struct ChatChoiceMessage {
    content: Option<String>,
    #[serde(default)]
    tool_calls: Option<Vec<ApiToolCall>>,
}

#[derive(Deserialize)]
#[expect(clippy::struct_field_names, reason = "mirrors OpenAI API field names")]
struct ApiUsage {
    #[serde(default)]
    prompt_tokens: u32,
    #[serde(default)]
    completion_tokens: u32,
    #[serde(default)]
    total_tokens: u32,
}

impl OpenAiCompatClient {
    /// Build a client from a provider config entry.
    ///
    /// # Errors
    ///
    /// Returns an error if required fields are missing or the HTTP client
    /// cannot be constructed.
    pub fn from_config(config: &ProviderConfig) -> Result<Self> {
        let base_url = config
            .base_url
            .as_deref()
            .ok_or_else(|| Error::Config("provider missing base_url".into()))?
            .trim_end_matches('/')
            .to_owned();

        let chat_model = config.model.as_deref().unwrap_or("gpt-4o").to_owned();

        let embedding_model = config
            .embedding_model
            .as_deref()
            .unwrap_or("nomic-embed-text")
            .to_owned();

        let dimension = config.embedding_dimension.unwrap_or(768);

        let mut headers = HeaderMap::new();

        if let Some(key) = config
            .api_key_env
            .as_deref()
            .and_then(|env_var| std::env::var(env_var).ok())
        {
            let mut auth = HeaderValue::from_str(&format!("Bearer {key}"))
                .map_err(|e| Error::Config(format!("invalid API key header: {e}")))?;
            auth.set_sensitive(true);
            headers.insert(AUTHORIZATION, auth);
        }

        let http = reqwest::Client::builder()
            .default_headers(headers)
            .build()
            .map_err(|e| Error::Config(format!("failed to build HTTP client: {e}")))?;

        Ok(Self {
            http,
            base_url,
            chat_model,
            embedding_model,
            dimension,
        })
    }

    fn convert_messages(messages: &[crate::llm::Message]) -> Vec<ChatMessage> {
        messages
            .iter()
            .map(|m| {
                let role = match m.role {
                    crate::llm::Role::System => "system",
                    crate::llm::Role::User => "user",
                    crate::llm::Role::Assistant => "assistant",
                    crate::llm::Role::Tool => "tool",
                };

                let tool_calls = m.tool_calls.as_ref().map(|tcs| {
                    tcs.iter()
                        .map(|tc| ApiToolCall {
                            id: tc.id.clone(),
                            call_type: String::from("function"),
                            function: ApiFunction {
                                name: tc.function.name.clone(),
                                arguments: tc.function.arguments.clone(),
                            },
                        })
                        .collect()
                });

                // For tool messages, content must be a string (not null)
                let content = if m.content.is_empty() {
                    serde_json::Value::Null
                } else {
                    serde_json::Value::String(m.content.clone())
                };

                ChatMessage {
                    role: role.to_owned(),
                    content,
                    tool_call_id: m.tool_call_id.clone(),
                    tool_calls,
                }
            })
            .collect()
    }
}

impl LlmProvider for OpenAiCompatClient {
    fn name(&self) -> &str {
        &self.chat_model
    }

    async fn complete(&self, request: CompletionRequest) -> Result<CompletionResponse> {
        let url = format!("{}/chat/completions", self.base_url);
        let messages = Self::convert_messages(&request.messages);

        let body = ChatRequest {
            model: &self.chat_model,
            messages: &messages,
            tools: request.tools.as_deref(),
            temperature: request.temperature,
            max_tokens: request.max_tokens,
        };

        let response = self
            .http
            .post(&url)
            .json(&body)
            .send()
            .await
            .map_err(|e| Error::Llm(format!("request failed: {e}")))?;

        if !response.status().is_success() {
            let status = response.status();
            let body_text = response
                .text()
                .await
                .unwrap_or_else(|_| String::from("<unreadable>"));
            return Err(Error::Llm(format!(
                "chat API returned {status}: {body_text}"
            )));
        }

        let result: ChatResponse = response
            .json()
            .await
            .map_err(|e| Error::Llm(format!("failed to parse chat response: {e}")))?;

        let choice = result
            .choices
            .into_iter()
            .next()
            .ok_or_else(|| Error::Llm("no choices in response".into()))?;

        let content = choice.message.content.unwrap_or_default();

        let tool_calls = choice
            .message
            .tool_calls
            .unwrap_or_default()
            .into_iter()
            .map(|tc| ToolCall {
                id: tc.id,
                function: FunctionCall {
                    name: tc.function.name,
                    arguments: tc.function.arguments,
                },
            })
            .collect::<Vec<_>>();

        let finish_reason = match choice.finish_reason.as_deref() {
            Some("stop") => FinishReason::Stop,
            Some("tool_calls") => FinishReason::ToolCalls,
            Some("length") => FinishReason::Length,
            Some(other) => FinishReason::Other(other.to_owned()),
            None => {
                if tool_calls.is_empty() {
                    FinishReason::Stop
                } else {
                    FinishReason::ToolCalls
                }
            }
        };

        let usage = result
            .usage
            .map_or_else(TokenUsage::default, |u| TokenUsage {
                prompt_tokens: u.prompt_tokens,
                completion_tokens: u.completion_tokens,
                total_tokens: u.total_tokens,
            });

        Ok(CompletionResponse {
            content,
            tool_calls,
            usage,
            finish_reason,
        })
    }

    async fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        let url = format!("{}/embeddings", self.base_url);
        let body = EmbeddingRequest {
            model: &self.embedding_model,
            input: texts,
        };

        let response = self
            .http
            .post(&url)
            .json(&body)
            .send()
            .await
            .map_err(|e| Error::Embedding(format!("request failed: {e}")))?;

        if !response.status().is_success() {
            let status = response.status();
            let body_text = response
                .text()
                .await
                .unwrap_or_else(|_| String::from("<unreadable>"));
            return Err(Error::Embedding(format!(
                "embedding API returned {status}: {body_text}"
            )));
        }

        let result: EmbeddingResponse = response
            .json()
            .await
            .map_err(|e| Error::Embedding(format!("failed to parse response: {e}")))?;

        Ok(result.data.into_iter().map(|d| d.embedding).collect())
    }

    fn embedding_dimension(&self) -> u32 {
        self.dimension
    }

    fn supports_tools(&self) -> bool {
        true
    }
}

impl EmbeddingProvider for OpenAiCompatClient {
    async fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        LlmProvider::embed(self, texts).await
    }

    fn embedding_dimension(&self) -> u32 {
        self.dimension
    }

    fn model_name(&self) -> &str {
        &self.embedding_model
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn base_provider_config() -> ProviderConfig {
        ProviderConfig {
            provider_type: "openai-compat".into(),
            base_url: Some("http://localhost:11434/v1".into()),
            api_key_env: None,
            model: Some("llama3.1:8b".into()),
            embedding_model: Some("nomic-embed-text".into()),
            embedding_dimension: Some(128),
        }
    }

    #[test]
    fn missing_base_url_returns_error() {
        let mut cfg = base_provider_config();
        cfg.base_url = None;
        assert!(OpenAiCompatClient::from_config(&cfg).is_err());
    }

    #[test]
    #[expect(clippy::unwrap_used, reason = "test asserts Ok")]
    fn valid_config_succeeds() {
        let client = OpenAiCompatClient::from_config(&base_provider_config()).unwrap();
        assert_eq!(client.chat_model, "llama3.1:8b");
        assert_eq!(client.embedding_model, "nomic-embed-text");
    }

    #[test]
    #[expect(clippy::unwrap_used, reason = "test asserts Ok")]
    fn strips_trailing_slash_from_base_url() {
        let mut cfg = base_provider_config();
        cfg.base_url = Some("http://localhost:11434/v1/".into());
        let client = OpenAiCompatClient::from_config(&cfg).unwrap();
        assert_eq!(client.base_url, "http://localhost:11434/v1");
    }

    #[test]
    #[expect(clippy::unwrap_used, reason = "test asserts Ok")]
    fn defaults_when_optional_fields_missing() {
        let cfg = ProviderConfig {
            provider_type: "openai-compat".into(),
            base_url: Some("http://localhost:11434/v1".into()),
            api_key_env: None,
            model: None,
            embedding_model: None,
            embedding_dimension: None,
        };
        let client = OpenAiCompatClient::from_config(&cfg).unwrap();
        assert_eq!(client.chat_model, "gpt-4o");
        assert_eq!(client.embedding_model, "nomic-embed-text");
        assert_eq!(client.dimension, 768);
    }

    #[test]
    #[expect(clippy::unwrap_used, reason = "test asserts Ok")]
    fn supports_tools_returns_true() {
        let client = OpenAiCompatClient::from_config(&base_provider_config()).unwrap();
        assert!(LlmProvider::supports_tools(&client));
    }

    #[test]
    fn convert_messages_handles_all_roles() {
        let messages = vec![
            crate::llm::Message::system("system prompt"),
            crate::llm::Message::user("hello"),
            crate::llm::Message::assistant("hi there"),
            crate::llm::Message::tool_result("call-1", "result data"),
        ];
        let converted = OpenAiCompatClient::convert_messages(&messages);
        assert_eq!(converted.len(), 4);
        assert_eq!(converted.first().map(|m| m.role.as_str()), Some("system"));
        assert_eq!(converted.get(1).map(|m| m.role.as_str()), Some("user"));
        assert_eq!(converted.get(2).map(|m| m.role.as_str()), Some("assistant"));
        assert_eq!(converted.get(3).map(|m| m.role.as_str()), Some("tool"));
        assert_eq!(
            converted.get(3).and_then(|m| m.tool_call_id.as_deref()),
            Some("call-1")
        );
    }

    #[test]
    fn convert_messages_with_tool_calls() {
        let messages = vec![crate::llm::Message::assistant_with_tool_calls(
            String::new(),
            vec![ToolCall {
                id: "tc-1".into(),
                function: FunctionCall {
                    name: "run_sql".into(),
                    arguments: r#"{"query":"SELECT 1"}"#.into(),
                },
            }],
        )];
        let converted = OpenAiCompatClient::convert_messages(&messages);
        let first = converted.first();
        assert!(first.is_some_and(|m| m.tool_calls.is_some()));
    }
}
