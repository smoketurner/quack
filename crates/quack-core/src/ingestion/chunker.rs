use crate::error::{Error, Result};
use crate::ingestion::parser::Section;

/// Split text into overlapping chunks by BPE token count.
///
/// Uses tiktoken's BPE tokenizer for accurate token boundaries. The encoding
/// name must match one of tiktoken's supported encodings (e.g., `cl100k_base`
/// for `OpenAI` embedding models).
///
/// # Errors
///
/// Returns an error if the encoding name is not recognized.
pub fn chunk_text(
    text: &str,
    chunk_size_tokens: u32,
    overlap_tokens: u32,
    encoding_name: &str,
) -> Result<Vec<String>> {
    let chunk_size = chunk_size_tokens as usize;

    if text.is_empty() || chunk_size == 0 {
        return Ok(Vec::new());
    }

    let enc = tiktoken::get_encoding(encoding_name)
        .ok_or_else(|| Error::Config(format!("unknown tiktoken encoding: {encoding_name}")))?;

    let tokens = enc.encode(text);
    if tokens.is_empty() {
        return Ok(Vec::new());
    }

    if tokens.len() <= chunk_size {
        let decoded = enc
            .decode_to_string(&tokens)
            .map_err(|e| Error::Ingestion(format!("token decode failed: {e}")))?;
        return Ok(vec![decoded]);
    }

    let overlap = (overlap_tokens as usize).min(chunk_size.saturating_sub(1));
    let step = chunk_size.saturating_sub(overlap).max(1);
    let mut chunks = Vec::new();
    let mut start = 0;

    while start < tokens.len() {
        let end = tokens.len().min(start.saturating_add(chunk_size));
        let chunk_tokens = tokens
            .get(start..end)
            .ok_or_else(|| Error::Ingestion("chunk slice out of bounds".into()))?;

        let decoded = enc
            .decode_to_string(chunk_tokens)
            .map_err(|e| Error::Ingestion(format!("token decode failed: {e}")))?;
        chunks.push(decoded);

        let next = start.saturating_add(step);
        if next <= start || next >= tokens.len() {
            break;
        }
        start = next;
    }

    Ok(chunks)
}

#[cfg(test)]
mod tests {
    use super::*;

    const ENC: &str = "cl100k_base";

    #[test]
    fn empty_text_returns_empty() {
        let result = chunk_text("", 100, 10, ENC);
        assert!(result.is_ok_and(|v| v.is_empty()));
    }

    #[test]
    #[expect(clippy::unwrap_used, reason = "test asserts Ok")]
    fn short_text_single_chunk() {
        let text = "one two three";
        let chunks = chunk_text(text, 100, 10, ENC).unwrap();
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks.first().map(String::as_str), Some("one two three"));
    }

    #[test]
    #[expect(clippy::unwrap_used, reason = "test asserts Ok")]
    fn splits_into_multiple_chunks() {
        let words: Vec<String> = (0..200).map(|i| format!("word{i}")).collect();
        let text = words.join(" ");
        let chunks = chunk_text(&text, 50, 10, ENC).unwrap();
        assert!(chunks.len() > 1, "expected multiple chunks");
    }

    #[test]
    #[expect(clippy::unwrap_used, reason = "test asserts Ok")]
    fn chunks_respect_token_limit() {
        let words: Vec<String> = (0..200).map(|i| format!("word{i}")).collect();
        let text = words.join(" ");
        let chunk_size: u32 = 50;
        let chunks = chunk_text(&text, chunk_size, 10, ENC).unwrap();
        let enc = tiktoken::get_encoding(ENC).unwrap();
        for chunk in &chunks {
            let count = enc.count(chunk);
            assert!(
                count <= chunk_size as usize,
                "chunk has {count} tokens, limit is {chunk_size}"
            );
        }
    }

    #[test]
    fn zero_chunk_size_returns_empty() {
        assert!(chunk_text("hello world", 0, 0, ENC).is_ok_and(|v| v.is_empty()));
    }

    #[test]
    fn overlap_larger_than_chunk_still_works() {
        let text = "a b c d e f g h i j k l m n o p q r s t u v w x y z";
        assert!(chunk_text(text, 3, 10, ENC).is_ok_and(|v| !v.is_empty()));
    }

    #[test]
    #[expect(clippy::unwrap_used, reason = "test asserts Ok")]
    fn whitespace_only_returns_single_chunk() {
        let chunks = chunk_text("   \n\t  ", 10, 2, ENC).unwrap();
        assert_eq!(chunks.len(), 1);
    }

    #[test]
    fn invalid_encoding_returns_error() {
        let result = chunk_text("hello", 10, 2, "nonexistent_encoding");
        assert!(result.is_err());
    }
}

/// A chunk ready to store: its text, where it came from, and the text to
/// embed (the heading prepended so retrieval sees the context).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Chunk {
    pub content: String,
    pub heading: Option<String>,
    pub page: Option<u32>,
}

impl Chunk {
    /// Text handed to the embedding model.
    #[must_use]
    pub fn embedding_input(&self) -> String {
        match &self.heading {
            Some(h) => format!("{h}\n\n{}", self.content),
            None => self.content.clone(),
        }
    }
}

/// Chunk every section, carrying its heading and page onto each chunk.
///
/// # Errors
///
/// Returns an error if the encoding name is not recognized.
pub fn chunk_sections(
    sections: &[Section],
    chunk_size_tokens: u32,
    overlap_tokens: u32,
    encoding_name: &str,
) -> Result<Vec<Chunk>> {
    let mut out = Vec::new();
    for section in sections {
        for content in chunk_text(
            &section.text,
            chunk_size_tokens,
            overlap_tokens,
            encoding_name,
        )? {
            out.push(Chunk {
                content,
                heading: section.heading.clone(),
                page: section.page,
            });
        }
    }
    Ok(out)
}

#[cfg(test)]
mod section_tests {
    use super::*;

    #[test]
    #[expect(clippy::unwrap_used, reason = "test asserts Ok")]
    fn sections_keep_heading_and_page_on_every_chunk() {
        let words: Vec<String> = (0..120).map(|i| format!("w{i}")).collect();
        let sections = vec![
            Section {
                heading: Some(String::from("Exclusions")),
                page: Some(3),
                text: words.join(" "),
            },
            Section {
                heading: None,
                page: Some(4),
                text: String::from("short"),
            },
        ];
        let chunks = chunk_sections(&sections, 40, 5, "cl100k_base").unwrap();
        assert!(chunks.len() > 2);
        let last = chunks.last().unwrap();
        assert_eq!(last.content, "short");
        assert_eq!(last.page, Some(4));
        assert!(last.heading.is_none());
        for chunk in chunks.iter().take(chunks.len() - 1) {
            assert_eq!(chunk.heading.as_deref(), Some("Exclusions"));
            assert_eq!(chunk.page, Some(3));
            assert!(chunk.embedding_input().starts_with("Exclusions\n\n"));
        }
        assert_eq!(last.embedding_input(), "short");
    }
}
