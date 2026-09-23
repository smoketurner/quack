//! The input prefixes each embedding model family was trained with, taken
//! from its authors' model card or sentence-transformers configuration.
//! Ollama applies none of them (`/api/embed` passes the input through
//! unchanged), so quack adds them itself.
//!
//! A family is matched on the model name's last path segment with the tag
//! removed, lowercased: `hf.co/Qwen/Qwen3-Embedding-0.6B-GGUF:Q8_0` is
//! matched as `qwen3-embedding-0.6b-gguf`. Longer, more specific patterns
//! come first (`nomic-embed-text-v2` before `nomic-embed-text`).

use super::Prompts;

/// A model family and the prefixes its authors specify.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Family {
    /// A short name for notes and `quack doctor`.
    pub name: &'static str,
    pub query: &'static str,
    /// May contain `{title}`, replaced by the chunk's heading or `none`.
    pub document: &'static str,
    pub similarity: &'static str,
    /// Where the strings come from.
    pub source: &'static str,
}

impl Family {
    /// The prompts this family specifies.
    #[must_use]
    pub fn prompts(&self) -> Prompts {
        Prompts {
            query: self.query.to_owned(),
            document: self.document.to_owned(),
            similarity: self.similarity.to_owned(),
        }
    }

    /// A family whose authors specify no prefix for any role: known, so
    /// `quack doctor` does not suggest configuring one.
    const fn unprefixed(name: &'static str, source: &'static str) -> Self {
        Self {
            name,
            query: "",
            document: "",
            similarity: "",
            source,
        }
    }

    /// The family `model` belongs to, or `None` when quack does not know
    /// it. The match is on the model name's last path segment, tag
    /// removed, lowercased.
    #[must_use]
    pub fn of(model: &str) -> Option<&'static Self> {
        let segment = model.rsplit('/').next().unwrap_or(model);
        let name = segment
            .split_once(':')
            .map_or(segment, |(name, _)| name)
            .to_lowercase();
        let has = |pattern: &str| name.contains(pattern);
        // BGE's English models: Ollama names them without a language
        // (`bge-large`), Hugging Face with one (`bge-base-en-v1.5`). The
        // Chinese ones take a Chinese instruction, which quack does not carry.
        let bge_english = ["bge-large", "bge-base", "bge-small"]
            .iter()
            .any(|size| name == *size || name.starts_with(&format!("{size}-en")));
        let family = if has("embeddinggemma") {
            &EMBEDDING_GEMMA
        } else if has("qwen3-embedding") {
            &QWEN3_EMBEDDING
        } else if has("nomic-embed-text-v2") {
            &NOMIC_V2
        } else if has("nomic-embed-text") {
            &NOMIC_V1
        } else if has("mxbai-embed-large") {
            &MXBAI_EMBED_LARGE
        } else if has("arctic-embed2") || has("arctic-embed-l-v2") || has("arctic-embed-m-v2") {
            &ARCTIC_EMBED_V2
        } else if has("arctic-embed") {
            &ARCTIC_EMBED
        } else if name.starts_with("bge-m3") {
            &BGE_M3
        } else if bge_english {
            &BGE_EN
        } else if (name.starts_with("e5-") || has("multilingual-e5-")) && !has("instruct") {
            &E5
        } else if name.starts_with("all-minilm")
            || name.starts_with("all-mpnet")
            || name.starts_with("paraphrase-multilingual")
        {
            &SENTENCE_TRANSFORMERS
        } else if name.starts_with("granite-embedding") {
            &GRANITE
        } else if name.starts_with("text-embedding-3") || name.starts_with("text-embedding-ada") {
            &OPENAI
        } else {
            return None;
        };
        Some(family)
    }
}

const RETRIEVAL_QUERY: &str = "Represent this sentence for searching relevant passages: ";

/// Google's `EmbeddingGemma`: every role prefixed, and the document prompt
/// takes the title. The sentence-similarity prompt is the one the card
/// gives for duplicate detection, used on both sides.
const EMBEDDING_GEMMA: Family = Family {
    name: "EmbeddingGemma",
    query: "task: search result | query: ",
    document: "title: {title} | text: ",
    similarity: "task: sentence similarity | query: ",
    source: "https://huggingface.co/google/embeddinggemma-300m",
};

/// Qwen3-Embedding: an instruction on queries only (no space after
/// `Query:`, as the authors' code writes it); documents take nothing. The
/// similarity instruction is the one the authors' MTEB evaluation uses for
/// STS and pair classification.
const QWEN3_EMBEDDING: Family = Family {
    name: "Qwen3-Embedding",
    query: "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:",
    document: "",
    similarity: "Instruct: Retrieve semantically similar text\nQuery:",
    source: "https://huggingface.co/Qwen/Qwen3-Embedding-0.6B",
};

/// nomic-embed-text v2 (MoE): its configuration maps STS and pair
/// classification to `classification: `.
const NOMIC_V2: Family = Family {
    name: "nomic-embed-text v2",
    query: "search_query: ",
    document: "search_document: ",
    similarity: "classification: ",
    source: "https://huggingface.co/nomic-ai/nomic-embed-text-v2-moe",
};

/// nomic-embed-text v1 and v1.5: the card requires a prefix on every
/// input, and gives `clustering: ` for removing semantic duplicates.
const NOMIC_V1: Family = Family {
    name: "nomic-embed-text",
    query: "search_query: ",
    document: "search_document: ",
    similarity: "clustering: ",
    source: "https://huggingface.co/nomic-ai/nomic-embed-text-v1.5",
};

const MXBAI_EMBED_LARGE: Family = Family {
    name: "mxbai-embed-large",
    query: RETRIEVAL_QUERY,
    document: "",
    similarity: "",
    source: "https://huggingface.co/mixedbread-ai/mxbai-embed-large-v1",
};

const ARCTIC_EMBED_V2: Family = Family {
    name: "snowflake-arctic-embed v2",
    query: "query: ",
    document: "",
    similarity: "",
    source: "https://huggingface.co/Snowflake/snowflake-arctic-embed-l-v2.0",
};

const ARCTIC_EMBED: Family = Family {
    name: "snowflake-arctic-embed",
    query: RETRIEVAL_QUERY,
    document: "",
    similarity: "",
    source: "https://huggingface.co/Snowflake/snowflake-arctic-embed-l",
};

/// BGE v1.5 English: the query instruction for short-query retrieval only;
/// "no instruction needs to be added to passages".
const BGE_EN: Family = Family {
    name: "BGE (English)",
    query: RETRIEVAL_QUERY,
    document: "",
    similarity: "",
    source: "https://huggingface.co/BAAI/bge-large-en-v1.5",
};

/// E5: `query: ` on queries and on both sides of symmetric tasks,
/// `passage: ` on documents.
const E5: Family = Family {
    name: "E5",
    query: "query: ",
    document: "passage: ",
    similarity: "query: ",
    source: "https://huggingface.co/intfloat/e5-large-v2",
};

const BGE_M3: Family = Family::unprefixed("BGE-M3", "https://huggingface.co/BAAI/bge-m3");
const SENTENCE_TRANSFORMERS: Family = Family::unprefixed(
    "sentence-transformers",
    "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2",
);
const GRANITE: Family = Family::unprefixed(
    "granite-embedding",
    "https://huggingface.co/ibm-granite/granite-embedding-278m-multilingual",
);
const OPENAI: Family = Family::unprefixed(
    "OpenAI embeddings",
    "https://platform.openai.com/docs/guides/embeddings",
);

#[cfg(test)]
mod tests {
    use super::*;

    fn family_name(model: &str) -> Option<&'static str> {
        Family::of(model).map(|f| f.name)
    }

    #[test]
    fn ollama_names_match_their_family() {
        for (model, expected) in [
            ("embeddinggemma", "EmbeddingGemma"),
            ("embeddinggemma:latest", "EmbeddingGemma"),
            ("embeddinggemma:300m-qat-q4_0", "EmbeddingGemma"),
            ("qwen3-embedding:0.6b", "Qwen3-Embedding"),
            ("nomic-embed-text", "nomic-embed-text"),
            ("nomic-embed-text:v1.5", "nomic-embed-text"),
            ("nomic-embed-text-v2-moe", "nomic-embed-text v2"),
            ("mxbai-embed-large:335m", "mxbai-embed-large"),
            ("snowflake-arctic-embed:m", "snowflake-arctic-embed"),
            ("snowflake-arctic-embed2:568m", "snowflake-arctic-embed v2"),
            ("bge-m3:567m", "BGE-M3"),
            ("bge-large:335m-en-v1.5-fp16", "BGE (English)"),
            ("all-minilm:l6-v2", "sentence-transformers"),
            ("granite-embedding:278m", "granite-embedding"),
            ("text-embedding-3-small", "OpenAI embeddings"),
        ] {
            assert_eq!(family_name(model), Some(expected), "{model}");
        }
    }

    #[test]
    fn hugging_face_paths_match_on_their_last_segment() {
        for (model, expected) in [
            (
                "hf.co/Qwen/Qwen3-Embedding-0.6B-GGUF:Q8_0",
                "Qwen3-Embedding",
            ),
            (
                "hf.co/nomic-ai/nomic-embed-text-v1.5-GGUF",
                "nomic-embed-text",
            ),
            ("intfloat/e5-large-v2", "E5"),
            ("intfloat/multilingual-e5-large", "E5"),
            ("BAAI/bge-base-en-v1.5", "BGE (English)"),
            (
                "Snowflake/snowflake-arctic-embed-l-v2.0",
                "snowflake-arctic-embed v2",
            ),
        ] {
            assert_eq!(family_name(model), Some(expected), "{model}");
        }
    }

    #[test]
    fn unknown_and_ambiguous_models_have_no_family() {
        for model in [
            "my-embedder",
            "bge-small-zh-v1.5",
            "intfloat/multilingual-e5-large-instruct",
            "",
        ] {
            assert_eq!(family_name(model), None, "{model}");
        }
    }

    #[test]
    fn families_carry_their_cards_strings_exactly() {
        let gemma = Family::of("embeddinggemma").map(Family::prompts);
        assert_eq!(
            gemma,
            Some(Prompts {
                query: "task: search result | query: ".into(),
                document: "title: {title} | text: ".into(),
                similarity: "task: sentence similarity | query: ".into(),
            })
        );
        let qwen = Family::of("qwen3-embedding").map(Family::prompts);
        assert_eq!(
            qwen.as_ref().map(|p| p.query.as_str()),
            Some(
                "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:"
            )
        );
        assert_eq!(qwen.map(|p| p.document), Some(String::new()));
    }
}
