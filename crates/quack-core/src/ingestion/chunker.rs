use std::ops::Range;

use crate::embedding::Input;
use crate::error::{Error, Result};
use crate::ingestion::parser::{Extracted, Flow, Section};

/// Splits text into overlapping windows of BPE tokens.
///
/// Uses tiktoken's BPE tokenizer for accurate token boundaries. The encoding
/// name must match one of tiktoken's supported encodings (e.g., `cl100k_base`
/// for `OpenAI` embedding models).
#[derive(Clone, Copy)]
pub struct Chunker {
    bpe: &'static tiktoken::CoreBpe,
    /// Tokens per chunk at most; zero makes no chunks.
    size: usize,
    /// Tokens each chunk shares with the one before it.
    overlap: usize,
}

impl Chunker {
    /// Chunks of `size_tokens`, each overlapping the last by
    /// `overlap_tokens`, counted in `encoding`.
    ///
    /// # Errors
    ///
    /// Returns an error if the encoding name is not recognized.
    pub fn new(size_tokens: u32, overlap_tokens: u32, encoding: &str) -> Result<Self> {
        let bpe = tiktoken::get_encoding(encoding)
            .ok_or_else(|| Error::Config(format!("unknown tiktoken encoding: {encoding}")))?;
        Ok(Self {
            bpe,
            size: size_tokens as usize,
            overlap: overlap_tokens as usize,
        })
    }

    /// Split one text into chunks.
    ///
    /// # Errors
    ///
    /// Returns an error if a window falls outside the text's tokens.
    pub fn text(&self, text: &str) -> Result<Vec<String>> {
        if text.is_empty() || self.size == 0 {
            return Ok(Vec::new());
        }
        let tokens = self.bpe.encode(text);
        let mut chunks = Vec::new();
        for window in self.windows(tokens.len()) {
            chunks.push(self.decode(&tokens, window)?);
        }
        Ok(chunks)
    }

    /// The token ranges of a text `len` tokens long: windows of at most
    /// `size`, each starting `size - overlap` after the last, the final one
    /// ending at `len`.
    fn windows(&self, len: usize) -> Vec<Range<usize>> {
        if len == 0 || self.size == 0 {
            return Vec::new();
        }
        if len <= self.size {
            return std::iter::once(0..len).collect();
        }
        let overlap = self.overlap.min(self.size.saturating_sub(1));
        let step = self.size.saturating_sub(overlap).max(1);
        let mut ranges = Vec::new();
        let mut start = 0;
        while start < len {
            ranges.push(start..len.min(start.saturating_add(self.size)));
            let next = start.saturating_add(step);
            if next <= start || next >= len {
                break;
            }
            start = next;
        }
        ranges
    }

    /// The text of `tokens[window]`. A BPE token can hold part of a
    /// multi-byte character, so a window edge may cut one; the cut bytes
    /// become U+FFFD, and the overlap carries the whole character in the
    /// neighbouring chunk.
    fn decode(&self, tokens: &[u32], window: Range<usize>) -> Result<String> {
        let slice = tokens
            .get(window)
            .ok_or_else(|| Error::Ingestion("chunk slice out of bounds".into()))?;
        Ok(String::from_utf8_lossy(&self.bpe.decode(slice)).into_owned())
    }

    /// Chunk a parsed document by its flow: sectioned sources chunk
    /// section by section; a continuous source is windowed across its
    /// pages under its title, else `fallback_heading`, each chunk carrying
    /// the page it starts on.
    ///
    /// # Errors
    ///
    /// Returns an error if a window falls outside the text's tokens.
    pub fn document(
        &self,
        extracted: &Extracted,
        fallback_heading: Option<&str>,
    ) -> Result<Vec<Chunk>> {
        match extracted.flow {
            Flow::Sectioned => self.sections(&extracted.sections),
            Flow::Continuous => {
                self.pages(&extracted.sections, extracted.title().or(fallback_heading))
            }
        }
    }

    /// Window one running text across `pages`, with the pages' tokens
    /// joined by a blank line, so a window may span a page break. Each
    /// chunk records the page its first token lies on and carries
    /// `heading`.
    ///
    /// # Errors
    ///
    /// Returns an error if a window falls outside the text's tokens.
    pub fn pages(&self, pages: &[Section], heading: Option<&str>) -> Result<Vec<Chunk>> {
        let separator = self.bpe.encode("\n\n");
        let mut tokens: Vec<u32> = Vec::new();
        let mut page_starts: Vec<(usize, Option<u32>)> = Vec::new();
        for page in pages {
            let page_tokens = self.bpe.encode(&page.text);
            if page_tokens.is_empty() {
                continue;
            }
            if !tokens.is_empty() {
                tokens.extend_from_slice(&separator);
            }
            page_starts.push((tokens.len(), page.page));
            tokens.extend_from_slice(&page_tokens);
        }
        let mut chunks = Vec::new();
        for window in self.windows(tokens.len()) {
            let page = page_starts
                .iter()
                .rev()
                .find(|(first_token, _)| *first_token <= window.start)
                .and_then(|(_, page)| *page);
            let content = self.decode(&tokens, window)?.trim().to_owned();
            if content.is_empty() {
                continue;
            }
            chunks.push(Chunk {
                content,
                heading: heading.map(str::to_owned),
                page,
            });
        }
        Ok(chunks)
    }

    /// Chunk every section, carrying its heading and page onto each chunk.
    ///
    /// # Errors
    ///
    /// Returns an error if a window falls outside the text's tokens.
    pub fn sections(&self, sections: &[Section]) -> Result<Vec<Chunk>> {
        let mut out = Vec::new();
        for section in sections {
            for content in self.text(&section.text)? {
                out.push(Chunk {
                    content,
                    heading: section.heading.clone(),
                    page: section.page,
                });
            }
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const ENC: &str = "cl100k_base";

    #[test]
    fn empty_text_returns_empty() {
        let result = Chunker::new(100, 10, ENC).and_then(|c| c.text(""));
        assert!(result.is_ok_and(|v| v.is_empty()));
    }

    #[test]
    #[expect(clippy::unwrap_used, reason = "test asserts Ok")]
    fn short_text_single_chunk() {
        let text = "one two three";
        let chunks = Chunker::new(100, 10, ENC)
            .and_then(|c| c.text(text))
            .unwrap();
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks.first().map(String::as_str), Some("one two three"));
    }

    #[test]
    #[expect(clippy::unwrap_used, reason = "test asserts Ok")]
    fn splits_into_multiple_chunks() {
        let words: Vec<String> = (0..200).map(|i| format!("word{i}")).collect();
        let text = words.join(" ");
        let chunks = Chunker::new(50, 10, ENC)
            .and_then(|c| c.text(&text))
            .unwrap();
        assert!(chunks.len() > 1, "expected multiple chunks");
    }

    #[test]
    #[expect(clippy::unwrap_used, reason = "test asserts Ok")]
    fn chunks_respect_token_limit() {
        let words: Vec<String> = (0..200).map(|i| format!("word{i}")).collect();
        let text = words.join(" ");
        let chunk_size: u32 = 50;
        let chunks = Chunker::new(chunk_size, 10, ENC)
            .and_then(|c| c.text(&text))
            .unwrap();
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
        assert!(
            Chunker::new(0, 0, ENC)
                .and_then(|c| c.text("hello world"))
                .is_ok_and(|v| v.is_empty())
        );
    }

    #[test]
    fn overlap_larger_than_chunk_still_works() {
        let text = "a b c d e f g h i j k l m n o p q r s t u v w x y z";
        assert!(
            Chunker::new(3, 10, ENC)
                .and_then(|c| c.text(text))
                .is_ok_and(|v| !v.is_empty())
        );
    }

    #[test]
    #[expect(clippy::unwrap_used, reason = "test asserts Ok")]
    fn whitespace_only_returns_single_chunk() {
        let chunks = Chunker::new(10, 2, ENC)
            .and_then(|c| c.text("   \n\t  "))
            .unwrap();
        assert_eq!(chunks.len(), 1);
    }

    #[test]
    #[expect(clippy::unwrap_used, reason = "test asserts Ok")]
    fn a_window_edge_inside_a_multibyte_character_does_not_fail() {
        // Em dashes and accented letters span several BPE byte tokens;
        // a one-token window must land inside some of them.
        let text = "café — naïve — résumé — coöperate — façade — jalapeño";
        let chunks = Chunker::new(1, 0, ENC).and_then(|c| c.text(text)).unwrap();
        assert!(chunks.len() > 5);
        let joined: String = chunks.concat();
        assert!(joined.contains("caf"));
    }

    #[test]
    fn invalid_encoding_returns_error() {
        let result = Chunker::new(10, 2, "nonexistent_encoding").and_then(|c| c.text("hello"));
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
    /// What the embedding model is given: the text, under its heading.
    #[must_use]
    pub fn embedding_input(&self) -> Input {
        Input::Document {
            title: self.heading.clone(),
            text: self.content.clone(),
        }
    }
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
        let chunks = Chunker::new(40, 5, "cl100k_base")
            .and_then(|c| c.sections(&sections))
            .unwrap();
        assert!(chunks.len() > 2);
        let last = chunks.last().unwrap();
        assert_eq!(last.content, "short");
        assert_eq!(last.page, Some(4));
        assert!(last.heading.is_none());
        for chunk in chunks.iter().take(chunks.len() - 1) {
            assert_eq!(chunk.heading.as_deref(), Some("Exclusions"));
            assert_eq!(chunk.page, Some(3));
            assert_eq!(
                chunk.embedding_input(),
                Input::Document {
                    title: Some("Exclusions".into()),
                    text: chunk.content.clone(),
                }
            );
        }
        assert_eq!(
            last.embedding_input(),
            Input::Document {
                title: None,
                text: "short".into(),
            }
        );
    }

    fn page(number: u32, words: usize) -> Section {
        let text: Vec<String> = (0..words).map(|i| format!("p{number}w{i}")).collect();
        Section {
            heading: None,
            page: Some(number),
            text: text.join(" "),
        }
    }

    #[test]
    #[expect(clippy::unwrap_used, reason = "test asserts Ok")]
    fn continuous_pages_are_windowed_across_page_breaks() {
        let pages = vec![page(1, 8), page(2, 8), page(3, 8)];
        let chunks = Chunker::new(50, 10, "cl100k_base")
            .and_then(|c| c.pages(&pages, Some("Report")))
            .unwrap();
        assert!(chunks.len() > 1, "{chunks:?}");
        let enc = tiktoken::get_encoding("cl100k_base").unwrap();
        let first = chunks.first().unwrap();
        assert_eq!(first.page, Some(1));
        assert!(
            first.content.contains("p1w") && first.content.contains("p2w"),
            "the first window should cross from page 1 into page 2: {:?}",
            first.content
        );
        let pages_seen: Vec<Option<u32>> = chunks.iter().map(|c| c.page).collect();
        assert!(
            pages_seen.windows(2).all(|w| w.first() <= w.get(1)),
            "{pages_seen:?}"
        );
        assert!(chunks.iter().any(|c| c.page == Some(3)));
        for chunk in &chunks {
            assert!(enc.count(&chunk.content) <= 50);
            assert_eq!(chunk.heading.as_deref(), Some("Report"));
            assert!(
                matches!(chunk.embedding_input(), Input::Document { title: Some(t), .. } if t == "Report")
            );
            assert!(!chunk.content.starts_with('\n'));
        }
    }

    #[test]
    fn continuous_pages_with_no_text_give_no_chunks() {
        let pages = vec![Section {
            heading: None,
            page: Some(1),
            text: String::new(),
        }];
        assert!(
            Chunker::new(50, 10, "cl100k_base")
                .and_then(|c| c.pages(&pages, None))
                .is_ok_and(|c| c.is_empty())
        );
    }

    #[test]
    #[expect(clippy::unwrap_used, reason = "test asserts Ok")]
    fn chunk_document_dispatches_on_flow() {
        let sections = vec![page(1, 8), page(2, 8)];
        let sectioned = Chunker::new(50, 10, "cl100k_base")
            .and_then(|c| {
                c.document(
                    &Extracted {
                        title: None,
                        sections: sections.clone(),
                        flow: Flow::Sectioned,
                        pages_skipped: 0,
                    },
                    Some("file"),
                )
            })
            .unwrap();
        assert_eq!(sectioned.len(), 2);
        assert!(sectioned.iter().all(|c| c.heading.is_none()));

        let continuous = Chunker::new(50, 10, "cl100k_base")
            .and_then(|c| {
                c.document(
                    &Extracted {
                        title: None,
                        sections,
                        flow: Flow::Continuous,
                        pages_skipped: 0,
                    },
                    Some("file"),
                )
            })
            .unwrap();
        assert!(
            continuous
                .iter()
                .all(|c| c.heading.as_deref() == Some("file"))
        );
        assert!(
            continuous
                .first()
                .is_some_and(|c| c.content.contains("p2w0"))
        );
    }
}
