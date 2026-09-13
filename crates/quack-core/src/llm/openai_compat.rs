use reqwest::header::{AUTHORIZATION, HeaderMap, HeaderValue};
use serde::{Deserialize, Serialize};

use crate::config::ProviderConfig;
use crate::error::{Error, Result};
use crate::llm::EmbeddingProvider;

/// Client for OpenAI-compatible `/v1/embeddings` endpoints.
///
/// Covers `OpenAI`, Ollama, `vLLM`, `LiteLLM`, and most API gateways.
pub struct OpenAiCompatClient {
    http: reqwest::Client,
    base_url: String,
    model: String,
    dimension: u32,
}

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

        let model = config
            .embedding_model
            .as_deref()
            .ok_or_else(|| Error::Config("provider missing embedding_model".into()))?
            .to_owned();

        let dimension = config
            .embedding_dimension
            .ok_or_else(|| Error::Config("provider missing embedding_dimension".into()))?;

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
            model,
            dimension,
        })
    }
}

impl EmbeddingProvider for OpenAiCompatClient {
    async fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        let url = format!("{}/embeddings", self.base_url);
        let body = EmbeddingRequest {
            model: &self.model,
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

    fn model_name(&self) -> &str {
        &self.model
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
            model: None,
            embedding_model: Some("test-model".into()),
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
    fn missing_embedding_model_returns_error() {
        let mut cfg = base_provider_config();
        cfg.embedding_model = None;
        assert!(OpenAiCompatClient::from_config(&cfg).is_err());
    }

    #[test]
    fn missing_dimension_returns_error() {
        let mut cfg = base_provider_config();
        cfg.embedding_dimension = None;
        assert!(OpenAiCompatClient::from_config(&cfg).is_err());
    }

    #[test]
    fn valid_config_succeeds() {
        assert!(OpenAiCompatClient::from_config(&base_provider_config()).is_ok());
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
    fn exposes_dimension_and_model() {
        let client = OpenAiCompatClient::from_config(&base_provider_config()).unwrap();
        assert_eq!(client.embedding_dimension(), 128);
        assert_eq!(client.model_name(), "test-model");
    }
}
