//! What quack embeds, and how.
//!
//! Every embedding call names its role: a search [`Embedder::query`], the
//! [`Embedder::documents`] a query retrieves, or text compared with text of
//! its own kind ([`Embedder::similar`]: entity labels, ontology names). Most
//! embedding models were trained with a different input prefix for each
//! role and lose quality without it; [`presets`] carries the prefixes each
//! family's authors specify, and `[embedding]` in the config overrides them.
//!
//! The model, its vector width, and those prefixes together are a
//! [`Profile`]. Vectors made under one profile are not comparable with
//! vectors made under another, so the workspace records the profile of
//! every stored vector and searches only those made under the current one
//! (`storage::workspace`); `reembed` brings the rest up to date.

pub mod presets;
pub mod reembed;

use std::sync::Arc;

use rig::embeddings::EmbeddingModel;
use serde::{Deserialize, Serialize};

use crate::config::Config;
use crate::error::{Error, Result};
use crate::storage::control::sha256_hex;

/// The placeholder a document prefix may carry for the chunk's title.
pub const TITLE_PLACEHOLDER: &str = "{title}";

/// What fills [`TITLE_PLACEHOLDER`] when a chunk has no heading, as
/// `EmbeddingGemma`'s card specifies.
const NO_TITLE: &str = "none";

/// The prefix put before each role's input. Empty means none.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Prompts {
    pub query: String,
    /// May contain [`TITLE_PLACEHOLDER`].
    pub document: String,
    pub similarity: String,
}

impl Prompts {
    /// Whether every role goes to the model unprefixed.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.query.is_empty() && self.document.is_empty() && self.similarity.is_empty()
    }

    fn query_input(&self, text: &str) -> String {
        format!("{}{text}", self.query)
    }

    fn similarity_input(&self, text: &str) -> String {
        format!("{}{text}", self.similarity)
    }

    /// A document's input. A prefix with a title slot gets the title
    /// there; otherwise the title leads the text, so retrieval still sees
    /// which section a chunk came from.
    fn document_input(&self, document: &DocumentInput) -> String {
        if self.document.contains(TITLE_PLACEHOLDER) {
            let title = document
                .title
                .as_deref()
                .map(str::trim)
                .filter(|t| !t.is_empty())
                .unwrap_or(NO_TITLE);
            format!(
                "{}{}",
                self.document.replace(TITLE_PLACEHOLDER, title),
                document.text
            )
        } else {
            match &document.title {
                Some(title) => format!("{}{title}\n\n{}", self.document, document.text),
                None => format!("{}{}", self.document, document.text),
            }
        }
    }
}

/// What a text is embedded as.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Role {
    /// A search query, retrieving documents.
    Query,
    /// A document a query retrieves.
    Document,
    /// Text compared with text of its own kind: labels, names.
    Similarity,
}

/// A document chunk to embed: its text and the heading it sits under.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DocumentInput {
    pub title: Option<String>,
    pub text: String,
}

/// Where a profile's prompts came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PromptSource {
    /// The built-in family's, with no override.
    Family(&'static presets::Family),
    /// At least one role set in `[embedding]`.
    Config,
    /// No family is known for the model and nothing is configured.
    Unknown,
}

/// Everything that decides what vector a text becomes. Two vectors are
/// comparable only when their profiles are equal.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Profile {
    /// The model as configured, with Ollama's implicit `:latest` removed.
    pub model: String,
    pub dimension: u32,
    pub prompts: Prompts,
}

impl Profile {
    /// A profile over `model`, its name normalized.
    #[must_use]
    pub fn new(model: &str, dimension: u32, prompts: Prompts) -> Self {
        Self {
            model: model.strip_suffix(":latest").unwrap_or(model).to_owned(),
            dimension,
            prompts,
        }
    }

    /// The profile the configured embedding model runs under, or `None`
    /// when no embedding model is configured.
    ///
    /// # Errors
    ///
    /// Returns an error when the model reference is invalid.
    pub fn from_config(config: &Config) -> Result<Option<Self>> {
        let Some(model) = config.embedding_model_ref()? else {
            return Ok(None);
        };
        let Some(dimension) = model.provider.embedding_dimension else {
            return Err(Error::Config(format!(
                "provider '{}' is used for embeddings but has no embedding_dimension",
                model.provider_name
            )));
        };
        let (prompts, _) = prompts_for(config, model.model);
        Ok(Some(Self::new(model.model, dimension, prompts)))
    }

    /// The identity recorded beside each stored vector: the SHA-256 of
    /// the profile's JSON, whose field order is fixed.
    #[must_use]
    pub fn fingerprint(&self) -> String {
        let json = serde_json::to_vec(self).unwrap_or_default();
        sha256_hex(&json)
    }

    /// One line for notes: `embeddinggemma (768 dimensions, with prefixes)`.
    #[must_use]
    pub fn describe(&self) -> String {
        let prompts = if self.prompts.is_empty() {
            "no prefixes"
        } else {
            "with prefixes"
        };
        format!("{} ({} dimensions, {prompts})", self.model, self.dimension)
    }
}

/// The prompts `model` runs with, and where they came from: its family's,
/// with each role set in `[embedding]` taking precedence.
#[must_use]
pub fn prompts_for(config: &Config, model: &str) -> (Prompts, PromptSource) {
    let family = presets::family(model);
    let mut prompts = family.map(presets::Family::prompts).unwrap_or_default();
    let overrides = &config.embedding;
    let mut overridden = false;
    for (value, slot) in [
        (&overrides.query_prefix, &mut prompts.query),
        (&overrides.document_prefix, &mut prompts.document),
        (&overrides.similarity_prefix, &mut prompts.similarity),
    ] {
        if let Some(value) = value {
            value.clone_into(slot);
            overridden = true;
        }
    }
    let source = match (overridden, family) {
        (true, _) => PromptSource::Config,
        (false, Some(family)) => PromptSource::Family(family),
        (false, None) => PromptSource::Unknown,
    };
    (prompts, source)
}

/// An embedding model under a [`Profile`]: every call names its role, and
/// every vector that comes back is checked against the profile's width.
#[derive(Clone)]
pub struct Embedder<M> {
    model: M,
    profile: Arc<Profile>,
}

impl<M: EmbeddingModel> Embedder<M> {
    #[must_use]
    pub fn new(model: M, profile: Profile) -> Self {
        Self {
            model,
            profile: Arc::new(profile),
        }
    }

    #[must_use]
    pub fn profile(&self) -> &Profile {
        &self.profile
    }

    /// The model underneath, for provider-specific inspection.
    #[must_use]
    pub fn model(&self) -> &M {
        &self.model
    }

    /// One text's vector in `role` (a document without a title).
    ///
    /// # Errors
    ///
    /// Returns an error when the model fails or answers with the wrong
    /// width.
    pub async fn one(&self, role: Role, text: &str) -> Result<Vec<f32>> {
        let prompts = &self.profile.prompts;
        let input = match role {
            Role::Query => prompts.query_input(text),
            Role::Document => prompts.document_input(&DocumentInput {
                title: None,
                text: text.to_owned(),
            }),
            Role::Similarity => prompts.similarity_input(text),
        };
        let mut vectors = self.embed(vec![input]).await?;
        vectors
            .pop()
            .ok_or_else(|| Error::Embedding("the model returned no embedding".into()))
    }

    /// A search query's vector.
    ///
    /// # Errors
    ///
    /// Returns an error when the model fails or answers with the wrong
    /// width.
    pub async fn query(&self, text: &str) -> Result<Vec<f32>> {
        self.one(Role::Query, text).await
    }

    /// Document chunks' vectors, in order.
    ///
    /// # Errors
    ///
    /// Returns an error when the model fails or answers with the wrong
    /// width or count.
    pub async fn documents(&self, documents: &[DocumentInput]) -> Result<Vec<Vec<f32>>> {
        let inputs = documents
            .iter()
            .map(|d| self.profile.prompts.document_input(d))
            .collect();
        self.embed(inputs).await
    }

    /// Vectors for texts compared with each other rather than retrieved:
    /// entity labels, a name looked up among them, ontology names.
    ///
    /// # Errors
    ///
    /// Returns an error when the model fails or answers with the wrong
    /// width or count.
    pub async fn similar(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let inputs = texts
            .iter()
            .map(|t| self.profile.prompts.similarity_input(t))
            .collect();
        self.embed(inputs).await
    }

    async fn embed(&self, inputs: Vec<String>) -> Result<Vec<Vec<f32>>> {
        if inputs.is_empty() {
            return Ok(Vec::new());
        }
        let expected = inputs.len();
        #[expect(
            clippy::disallowed_methods,
            reason = "the one place a raw embedding call is made: every caller goes through a role"
        )]
        let embeddings = self
            .model
            .embed_texts(inputs)
            .await
            .map_err(|e| Error::Embedding(e.to_string()))?;
        if embeddings.len() != expected {
            return Err(Error::Embedding(format!(
                "{} returned {} embeddings for {expected} inputs",
                self.profile.model,
                embeddings.len()
            )));
        }
        let width = usize::try_from(self.profile.dimension).unwrap_or(usize::MAX);
        embeddings
            .into_iter()
            .map(|embedding| {
                if embedding.vec.len() != width {
                    return Err(width_mismatch(
                        &self.profile.model,
                        embedding.vec.len(),
                        self.profile.dimension,
                    ));
                }
                #[expect(clippy::cast_possible_truncation, reason = "vectors are stored as f32")]
                Ok(embedding.vec.into_iter().map(|v| v as f32).collect())
            })
            .collect()
    }
}

/// The error for a model whose vectors are not the configured width.
#[must_use]
pub fn width_mismatch(model: &str, returned: usize, configured: u32) -> Error {
    Error::Config(format!(
        "{model} returned {returned}-dimensional vectors but its provider's \
         embedding_dimension is {configured}; set embedding_dimension = {returned}"
    ))
}

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "tests")]
mod tests {
    use super::*;
    use rig::embeddings::{Embedding, EmbeddingError};
    use std::sync::Mutex;

    /// Records every input and answers with vectors of `width`.
    #[derive(Clone)]
    struct Recording {
        width: usize,
        inputs: Arc<Mutex<Vec<String>>>,
    }

    impl EmbeddingModel for Recording {
        const MAX_DOCUMENTS: usize = 64;
        type Client = ();

        fn make(_client: &Self::Client, _model: impl Into<String>, _dims: Option<usize>) -> Self {
            Self {
                width: 4,
                inputs: Arc::default(),
            }
        }

        fn ndims(&self) -> usize {
            self.width
        }

        fn embed_texts(
            &self,
            texts: impl IntoIterator<Item = String> + Send,
        ) -> impl Future<Output = std::result::Result<Vec<Embedding>, EmbeddingError>> + Send
        {
            let texts: Vec<String> = texts.into_iter().collect();
            self.inputs.lock().unwrap().extend(texts.iter().cloned());
            std::future::ready(Ok(texts
                .into_iter()
                .map(|document| Embedding {
                    document,
                    vec: vec![0.5; self.width],
                })
                .collect()))
        }
    }

    fn recording(
        width: usize,
        dimension: u32,
        prompts: Prompts,
    ) -> (Embedder<Recording>, Arc<Mutex<Vec<String>>>) {
        let inputs = Arc::default();
        let model = Recording {
            width,
            inputs: Arc::clone(&inputs),
        };
        (
            Embedder::new(model, Profile::new("m", dimension, prompts)),
            inputs,
        )
    }

    fn gemma() -> Prompts {
        presets::family("embeddinggemma").unwrap().prompts()
    }

    #[tokio::test]
    async fn each_role_gets_its_prefix() {
        let (embedder, inputs) = recording(4, 4, gemma());
        embedder.query("storm damage").await.unwrap();
        embedder
            .documents(&[
                DocumentInput {
                    title: Some("Scales".into()),
                    text: "EF0 to EF5".into(),
                },
                DocumentInput {
                    title: None,
                    text: "untitled".into(),
                },
                DocumentInput {
                    title: Some("  ".into()),
                    text: "blank heading".into(),
                },
            ])
            .await
            .unwrap();
        embedder.similar(&["Acme (company)".into()]).await.unwrap();
        assert_eq!(
            *inputs.lock().unwrap(),
            [
                "task: search result | query: storm damage",
                "title: Scales | text: EF0 to EF5",
                "title: none | text: untitled",
                "title: none | text: blank heading",
                "task: sentence similarity | query: Acme (company)",
            ]
        );
    }

    #[tokio::test]
    async fn without_a_title_slot_the_heading_leads_the_text() {
        let prompts = presets::family("nomic-embed-text").unwrap().prompts();
        let (embedder, inputs) = recording(4, 4, prompts);
        embedder
            .documents(&[DocumentInput {
                title: Some("Scales".into()),
                text: "EF0".into(),
            }])
            .await
            .unwrap();
        let (plain, plain_inputs) = recording(4, 4, Prompts::default());
        plain
            .documents(&[DocumentInput {
                title: Some("Scales".into()),
                text: "EF0".into(),
            }])
            .await
            .unwrap();
        plain.query("q").await.unwrap();
        assert_eq!(*inputs.lock().unwrap(), ["search_document: Scales\n\nEF0"]);
        assert_eq!(*plain_inputs.lock().unwrap(), ["Scales\n\nEF0", "q"]);
    }

    #[tokio::test]
    async fn a_vector_of_the_wrong_width_is_a_config_error_naming_both() {
        let (embedder, _) = recording(768, 1024, Prompts::default());
        let err = embedder.query("q").await.unwrap_err().to_string();
        assert!(err.contains("768-dimensional"), "{err}");
        assert!(err.contains("embedding_dimension is 1024"), "{err}");
        assert!(err.contains("embedding_dimension = 768"), "{err}");
    }

    #[tokio::test]
    async fn no_inputs_make_no_call() {
        let (embedder, inputs) = recording(4, 4, Prompts::default());
        assert!(embedder.similar(&[]).await.unwrap().is_empty());
        assert!(inputs.lock().unwrap().is_empty());
    }

    #[test]
    fn profiles_differ_by_any_part_and_ignore_latest() {
        let base = Profile::new("embeddinggemma:latest", 768, gemma());
        assert_eq!(base.model, "embeddinggemma");
        assert_eq!(
            base.fingerprint(),
            Profile::new("embeddinggemma", 768, gemma()).fingerprint()
        );
        for other in [
            Profile::new("embeddinggemma", 768, Prompts::default()),
            Profile::new("embeddinggemma", 512, gemma()),
            Profile::new("embeddinggemma:300m-qat-q4_0", 768, gemma()),
        ] {
            assert_ne!(base.fingerprint(), other.fingerprint(), "{other:?}");
        }
        assert_eq!(base.fingerprint().len(), 64);
    }

    #[test]
    fn describe_says_whether_prefixes_apply() {
        assert_eq!(
            Profile::new("m", 4, Prompts::default()).describe(),
            "m (4 dimensions, no prefixes)"
        );
        assert_eq!(
            Profile::new("m", 4, gemma()).describe(),
            "m (4 dimensions, with prefixes)"
        );
    }

    fn config(toml: &str) -> Config {
        toml::from_str(toml).unwrap()
    }

    #[test]
    fn config_overrides_one_role_and_keeps_the_family_for_the_rest() {
        let base = "[general]\nembedding_model = \"o/embeddinggemma\"\n[providers.o]\ntype = \"ollama\"\nembedding_dimension = 768\n";
        let (prompts, source) = prompts_for(&config(base), "embeddinggemma");
        assert_eq!(prompts, gemma());
        assert!(matches!(source, PromptSource::Family(f) if f.name == "EmbeddingGemma"));

        let with =
            format!("{base}[embedding]\nquery_prefix = \"task: question answering | query: \"\n");
        let (prompts, source) = prompts_for(&config(&with), "embeddinggemma");
        assert_eq!(prompts.query, "task: question answering | query: ");
        assert_eq!(prompts.document, gemma().document);
        assert_eq!(source, PromptSource::Config);

        let off = format!(
            "{base}[embedding]\nquery_prefix = \"\"\ndocument_prefix = \"\"\nsimilarity_prefix = \"\"\n"
        );
        let (prompts, _) = prompts_for(&config(&off), "embeddinggemma");
        assert!(prompts.is_empty());

        let (prompts, source) = prompts_for(&config(base), "my-embedder");
        assert!(prompts.is_empty());
        assert_eq!(source, PromptSource::Unknown);
    }

    #[test]
    fn from_config_builds_the_profile_in_force() {
        let c = config(
            "[general]\nembedding_model = \"o/embeddinggemma:latest\"\n[providers.o]\ntype = \"ollama\"\nembedding_dimension = 768\n",
        );
        let profile = Profile::from_config(&c).unwrap().unwrap();
        assert_eq!(profile, Profile::new("embeddinggemma", 768, gemma()));
        assert!(Profile::from_config(&Config::default()).unwrap().is_none());
    }
}
