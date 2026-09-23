//! What quack embeds, and how.
//!
//! Every text is embedded as an [`Input`], which names its role: a search
//! query, a document chunk a query retrieves, or text compared with text of
//! its own kind (entity labels, ontology names). Most embedding models were
//! trained with a different input prefix for each role and lose quality
//! without it; [`presets`] carries the prefixes each family's authors
//! specify, and `[embedding]` in the config overrides them.
//!
//! The model, its vector width, and those prefixes together are a
//! [`Profile`]. Vectors made under one profile are not comparable with
//! vectors made under another, so the workspace records the profile of
//! every stored vector and searches only those made under the current one
//! (`storage::workspace`); `refresh` brings the rest up to date.

pub mod presets;
pub mod refresh;
mod status;
mod vector;

use std::slice;
use std::sync::Arc;

use rig::embeddings::EmbeddingModel;
use serde::{Deserialize, Serialize};

pub use status::{EmbeddingStatus, StaleVectors};
pub use vector::{Dimension, Fingerprint, Vector, WidthMismatch};

use crate::config::Config;
use crate::error::{Error, Result};
use crate::priority::{Priority, with_priority};
use crate::storage::control::sha256_hex;

/// The placeholder a document prefix may carry for the chunk's title.
pub const TITLE_PLACEHOLDER: &str = "{title}";

/// What fills [`TITLE_PLACEHOLDER`] when a chunk has no heading, as
/// `EmbeddingGemma`'s card specifies.
const NO_TITLE: &str = "none";

/// A text to embed, in the role it is embedded for.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Input {
    /// A search query, retrieving documents.
    Query(String),
    /// A document chunk a query retrieves, under its heading.
    Document { title: Option<String>, text: String },
    /// Text compared with text of its own kind: entity labels, a name
    /// looked up among them, ontology names.
    Similarity(String),
}

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

    /// The text the model receives for `input`. A document prefix with a
    /// title slot gets the title there; otherwise the title leads the text,
    /// so retrieval still sees which section a chunk came from.
    fn render(&self, input: &Input) -> String {
        match input {
            Input::Query(text) => format!("{}{text}", self.query),
            Input::Similarity(text) => format!("{}{text}", self.similarity),
            Input::Document { title, text } if self.document.contains(TITLE_PLACEHOLDER) => {
                let title = title
                    .as_deref()
                    .map(str::trim)
                    .filter(|t| !t.is_empty())
                    .unwrap_or(NO_TITLE);
                format!("{}{text}", self.document.replace(TITLE_PLACEHOLDER, title))
            }
            Input::Document {
                title: Some(title),
                text,
            } => format!("{}{title}\n\n{text}", self.document),
            Input::Document { title: None, text } => format!("{}{text}", self.document),
        }
    }
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

/// The prompts a model runs with, and where they came from.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedPrompts {
    pub prompts: Prompts,
    pub source: PromptSource,
}

impl ResolvedPrompts {
    /// The prompts `model` runs with: its family's, with each role set in
    /// `[embedding]` taking precedence.
    #[must_use]
    pub fn for_model(config: &Config, model: &str) -> Self {
        let family = presets::Family::of(model);
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
        Self { prompts, source }
    }
}

/// Everything that decides what vector a text becomes. Two vectors are
/// comparable only when their profiles are equal.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Profile {
    /// The model as configured, with Ollama's implicit `:latest` removed.
    pub model: String,
    pub dimension: Dimension,
    pub prompts: Prompts,
}

impl Profile {
    /// A profile over `model`, its name normalized.
    #[must_use]
    pub fn new(model: &str, dimension: Dimension, prompts: Prompts) -> Self {
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
        let resolved = ResolvedPrompts::for_model(config, model.model);
        Ok(Some(Self::new(
            model.model,
            Dimension::new(dimension),
            resolved.prompts,
        )))
    }

    /// The identity recorded beside each stored vector: the SHA-256 of
    /// the profile's JSON, whose field order is fixed.
    #[must_use]
    pub fn fingerprint(&self) -> Fingerprint {
        let json = serde_json::to_vec(self).unwrap_or_default();
        Fingerprint::new(sha256_hex(&json))
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

/// An embedding model under a [`Profile`]: every input names its role, and
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

    /// One input's vector for a lookup someone is waiting on (a search, an
    /// entity name): its model request goes ahead of background work.
    ///
    /// # Errors
    ///
    /// Returns an error when the model fails or answers with the wrong
    /// width.
    pub async fn embed_interactive(&self, input: &Input) -> Result<Vector> {
        with_priority(Priority::Interactive, self.embed_one(input)).await
    }

    /// One input's vector.
    ///
    /// # Errors
    ///
    /// Returns an error when the model fails or answers with the wrong
    /// width.
    pub async fn embed_one(&self, input: &Input) -> Result<Vector> {
        let mut vectors = self.embed(slice::from_ref(input)).await?;
        vectors
            .pop()
            .ok_or_else(|| Error::Embedding("the model returned no embedding".into()))
    }

    /// The inputs' vectors, in order.
    ///
    /// # Errors
    ///
    /// Returns an error when the model fails or answers with the wrong
    /// width or count.
    pub async fn embed(&self, inputs: &[Input]) -> Result<Vec<Vector>> {
        if inputs.is_empty() {
            return Ok(Vec::new());
        }
        let texts: Vec<String> = inputs
            .iter()
            .map(|input| self.profile.prompts.render(input))
            .collect();
        #[expect(
            clippy::disallowed_methods,
            reason = "the one place a raw embedding call is made: every caller goes through an Input"
        )]
        let embeddings = self
            .model
            .embed_texts(texts)
            .await
            .map_err(|e| Error::Embedding(e.to_string()))?;
        if embeddings.len() != inputs.len() {
            return Err(Error::Embedding(format!(
                "{} returned {} embeddings for {} inputs",
                self.profile.model,
                embeddings.len(),
                inputs.len()
            )));
        }
        embeddings
            .into_iter()
            .map(|embedding| {
                #[expect(clippy::cast_possible_truncation, reason = "vectors are stored as f32")]
                let values = embedding.vec.into_iter().map(|v| v as f32).collect();
                Vector::new(values, self.profile.dimension)
                    .map_err(|mismatch| mismatch.for_model(&self.profile.model))
            })
            .collect()
    }
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
            Embedder::new(model, Profile::new("m", Dimension::new(dimension), prompts)),
            inputs,
        )
    }

    fn gemma() -> Prompts {
        presets::Family::of("embeddinggemma").unwrap().prompts()
    }

    fn document(title: Option<&str>, text: &str) -> Input {
        Input::Document {
            title: title.map(str::to_owned),
            text: text.to_owned(),
        }
    }

    #[tokio::test]
    async fn each_role_gets_its_prefix() {
        let (embedder, inputs) = recording(4, 4, gemma());
        embedder
            .embed(&[
                Input::Query("storm damage".into()),
                document(Some("Scales"), "EF0 to EF5"),
                document(None, "untitled"),
                document(Some("  "), "blank heading"),
                Input::Similarity("Acme (company)".into()),
            ])
            .await
            .unwrap();
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
        let prompts = presets::Family::of("nomic-embed-text").unwrap().prompts();
        let (embedder, inputs) = recording(4, 4, prompts);
        embedder
            .embed_one(&document(Some("Scales"), "EF0"))
            .await
            .unwrap();
        let (plain, plain_inputs) = recording(4, 4, Prompts::default());
        plain
            .embed(&[document(Some("Scales"), "EF0"), Input::Query("q".into())])
            .await
            .unwrap();
        assert_eq!(*inputs.lock().unwrap(), ["search_document: Scales\n\nEF0"]);
        assert_eq!(*plain_inputs.lock().unwrap(), ["Scales\n\nEF0", "q"]);
    }

    #[tokio::test]
    async fn a_vector_of_the_wrong_width_is_a_config_error_naming_both() {
        let (embedder, _) = recording(768, 1024, Prompts::default());
        let err = embedder
            .embed_one(&Input::Query("q".into()))
            .await
            .unwrap_err()
            .to_string();
        assert!(err.contains("768-dimensional"), "{err}");
        assert!(err.contains("embedding_dimension is 1024"), "{err}");
        assert!(err.contains("embedding_dimension = 768"), "{err}");
    }

    #[tokio::test]
    async fn vectors_come_back_at_the_profile_width() {
        let (embedder, inputs) = recording(4, 4, Prompts::default());
        assert!(embedder.embed(&[]).await.unwrap().is_empty());
        assert!(inputs.lock().unwrap().is_empty(), "no inputs, no call");
        let vector = embedder
            .embed_one(&Input::Similarity("a".into()))
            .await
            .unwrap();
        assert_eq!(vector.dimension(), Dimension::new(4));
    }

    #[test]
    fn profiles_differ_by_any_part_and_ignore_latest() {
        let d768 = Dimension::new(768);
        let base = Profile::new("embeddinggemma:latest", d768, gemma());
        assert_eq!(base.model, "embeddinggemma");
        assert_eq!(
            base.fingerprint(),
            Profile::new("embeddinggemma", d768, gemma()).fingerprint()
        );
        for other in [
            Profile::new("embeddinggemma", d768, Prompts::default()),
            Profile::new("embeddinggemma", Dimension::new(512), gemma()),
            Profile::new("embeddinggemma:300m-qat-q4_0", d768, gemma()),
        ] {
            assert_ne!(base.fingerprint(), other.fingerprint(), "{other:?}");
        }
        assert_eq!(base.fingerprint().as_str().len(), 64);
    }

    #[test]
    fn describe_says_whether_prefixes_apply() {
        let four = Dimension::new(4);
        assert_eq!(
            Profile::new("m", four, Prompts::default()).describe(),
            "m (4 dimensions, no prefixes)"
        );
        assert_eq!(
            Profile::new("m", four, gemma()).describe(),
            "m (4 dimensions, with prefixes)"
        );
    }

    fn config(toml: &str) -> Config {
        toml::from_str(toml).unwrap()
    }

    #[test]
    fn config_overrides_one_role_and_keeps_the_family_for_the_rest() {
        let base = "[general]\nembedding_model = \"o/embeddinggemma\"\n[providers.o]\ntype = \"ollama\"\nembedding_dimension = 768\n";
        let resolved = ResolvedPrompts::for_model(&config(base), "embeddinggemma");
        assert_eq!(resolved.prompts, gemma());
        assert!(matches!(resolved.source, PromptSource::Family(f) if f.name == "EmbeddingGemma"));

        let with =
            format!("{base}[embedding]\nquery_prefix = \"task: question answering | query: \"\n");
        let resolved = ResolvedPrompts::for_model(&config(&with), "embeddinggemma");
        assert_eq!(resolved.prompts.query, "task: question answering | query: ");
        assert_eq!(resolved.prompts.document, gemma().document);
        assert_eq!(resolved.source, PromptSource::Config);

        let off = format!(
            "{base}[embedding]\nquery_prefix = \"\"\ndocument_prefix = \"\"\nsimilarity_prefix = \"\"\n"
        );
        assert!(
            ResolvedPrompts::for_model(&config(&off), "embeddinggemma")
                .prompts
                .is_empty()
        );

        let resolved = ResolvedPrompts::for_model(&config(base), "my-embedder");
        assert!(resolved.prompts.is_empty());
        assert_eq!(resolved.source, PromptSource::Unknown);
    }

    #[test]
    fn from_config_builds_the_profile_in_force() {
        let c = config(
            "[general]\nembedding_model = \"o/embeddinggemma:latest\"\n[providers.o]\ntype = \"ollama\"\nembedding_dimension = 768\n",
        );
        let profile = Profile::from_config(&c).unwrap().unwrap();
        assert_eq!(
            profile,
            Profile::new("embeddinggemma", Dimension::new(768), gemma())
        );
        assert!(Profile::from_config(&Config::default()).unwrap().is_none());
    }
}
