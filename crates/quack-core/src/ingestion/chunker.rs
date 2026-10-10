use std::ops::Range;

use crate::embedding::Input;
use crate::error::{Error, Result};
use crate::ingestion::parser::{Extracted, Flow, Section, SectionKind};
use crate::ingestion::table::Table;

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
    /// body pages under each page's heading, else the document's title or
    /// `fallback_heading`, each chunk carrying the page it starts on.
    /// Tables, notes, and code are chunked their own way in either flow.
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
                let fallback = extracted.title().or(fallback_heading);
                let mut out = Vec::new();
                let mut run: Vec<Section> = Vec::new();
                for section in &extracted.sections {
                    if section.kind == SectionKind::Body {
                        run.push(section.clone());
                        continue;
                    }
                    out.extend(self.pages(&run, fallback)?);
                    run.clear();
                    out.extend(self.sections(std::slice::from_ref(section))?);
                }
                out.extend(self.pages(&run, fallback)?);
                Ok(out)
            }
        }
    }

    /// Window one running text across `pages`, with the pages' tokens
    /// joined by a blank line, so a window may span a page break. Each
    /// chunk records the page its first token lies on and the heading in
    /// force there (a page's own, carried on from the one before, else
    /// `fallback`); a chunk that runs onto later pages names the last one
    /// in its locator (`through page 4`), so a citation of it covers every
    /// page its text came from.
    ///
    /// # Errors
    ///
    /// Returns an error if a window falls outside the text's tokens.
    pub fn pages(&self, pages: &[Section], fallback: Option<&str>) -> Result<Vec<Chunk>> {
        let separator = self.bpe.encode("\n\n");
        let mut tokens: Vec<u32> = Vec::new();
        let mut starts: Vec<(usize, Option<u32>, Option<String>)> = Vec::new();
        // The first token of each page's own text, separators left out.
        let mut texts: Vec<(usize, Option<u32>)> = Vec::new();
        for page in pages {
            let page_tokens = self.bpe.encode(&page.text);
            if page_tokens.is_empty() {
                continue;
            }
            let heading = page.heading.clone().or_else(|| fallback.map(str::to_owned));
            if !tokens.is_empty() {
                starts.push((tokens.len(), page.page, heading.clone()));
                tokens.extend_from_slice(&separator);
            }
            starts.push((tokens.len(), page.page, heading));
            texts.push((tokens.len(), page.page));
            tokens.extend_from_slice(&page_tokens);
        }
        let mut chunks = Vec::new();
        for window in self.windows(tokens.len()) {
            let at = starts
                .iter()
                .rev()
                .find(|(first_token, _, _)| *first_token <= window.start);
            let last = texts
                .iter()
                .rev()
                .find(|(first_token, _)| *first_token < window.end)
                .and_then(|(_, page)| *page);
            let content = self.decode(&tokens, window)?.trim().to_owned();
            if content.is_empty() {
                continue;
            }
            let page = at.and_then(|(_, page, _)| *page);
            chunks.push(Chunk {
                content,
                heading: at.and_then(|(_, _, heading)| heading.clone()),
                page,
                kind: SectionKind::Body,
                locator: last
                    .filter(|last| page.is_some_and(|first| *last > first))
                    .map(|last| format!("through page {last}")),
            });
        }
        Ok(chunks)
    }

    /// Chunk every section by its kind, carrying its heading, page, kind,
    /// and locator onto each chunk: body and note text in token windows,
    /// a table by rows with its header on every piece, code by lines with
    /// the line each piece starts on.
    ///
    /// # Errors
    ///
    /// Returns an error if a window falls outside the text's tokens.
    pub fn sections(&self, sections: &[Section]) -> Result<Vec<Chunk>> {
        let mut out = Vec::new();
        for section in sections {
            let pieces = match section.kind {
                SectionKind::Body | SectionKind::Note => self
                    .text(&section.text)?
                    .into_iter()
                    .map(|content| (content, section.locator.clone()))
                    .collect(),
                SectionKind::Table => self
                    .table(&section.text)
                    .into_iter()
                    .map(|content| (content, section.locator.clone()))
                    .collect(),
                SectionKind::Code => self.lines(&section.text),
            };
            for (content, locator) in pieces {
                out.push(Chunk {
                    content,
                    heading: section.heading.clone(),
                    page: section.page,
                    kind: section.kind,
                    locator,
                });
            }
        }
        Ok(out)
    }

    /// A rendered table in pieces of at most `size` tokens, the header
    /// and separator lines repeated on each; text that is not a rendered
    /// table is one piece.
    #[must_use]
    pub fn table(&self, markdown: &str) -> Vec<String> {
        let Some((header, rows)) = Table::split_rendered(markdown) else {
            return vec![markdown.to_owned()];
        };
        let header_tokens = header
            .iter()
            .map(|l| self.bpe.count(l).saturating_add(1))
            .sum::<usize>();
        let mut pieces = Vec::new();
        let mut piece: Vec<&str> = Vec::new();
        let mut used = header_tokens;
        for row in rows {
            let cost = self.bpe.count(row).saturating_add(1);
            if !piece.is_empty() && used.saturating_add(cost) > self.size {
                pieces.push([header.as_slice(), piece.as_slice()].concat().join("\n"));
                piece.clear();
                used = header_tokens;
            }
            piece.push(row);
            used = used.saturating_add(cost);
        }
        if !piece.is_empty() || pieces.is_empty() {
            pieces.push([header.as_slice(), piece.as_slice()].concat().join("\n"));
        }
        pieces
    }

    /// Code in pieces of whole lines of at most `size` tokens, each with
    /// `line N` for the 1-based line it starts on. A line longer than the
    /// window is a piece of its own.
    #[must_use]
    pub fn lines(&self, text: &str) -> Vec<(String, Option<String>)> {
        let mut pieces = Vec::new();
        let mut piece: Vec<&str> = Vec::new();
        let mut used = 0usize;
        let mut first_line = 1usize;
        for (index, line) in text.lines().enumerate() {
            let cost = self.bpe.count(line).saturating_add(1);
            if !piece.is_empty() && used.saturating_add(cost) > self.size {
                pieces.push((piece.join("\n"), Some(format!("line {first_line}"))));
                piece.clear();
                used = 0;
                first_line = index.saturating_add(1);
            }
            piece.push(line);
            used = used.saturating_add(cost);
        }
        if !piece.is_empty() {
            pieces.push((piece.join("\n"), Some(format!("line {first_line}"))));
        }
        pieces
            .into_iter()
            .filter(|(content, _)| !content.trim().is_empty())
            .collect()
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

/// A chunk ready to store: its text, where it came from, what kind of
/// text it is, and the text to embed (the heading prepended so retrieval
/// sees the context).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Chunk {
    pub content: String,
    pub heading: Option<String>,
    pub page: Option<u32>,
    pub kind: SectionKind,
    /// Where the chunk sits in a source without pages: `line 40`, `12:04`,
    /// `chapter 3`, `message 2`.
    pub locator: Option<String>,
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
            Section::body(Some(String::from("Exclusions")), words.join(" ")).on_page(Some(3)),
            Section::body(None, "short").on_page(Some(4)),
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
        Section::body(None, text.join(" ")).on_page(Some(number))
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

    /// A short report fits one chunk: it starts on page 1 and says it runs
    /// through page 6, so a fact from page 4 is not cited as page 1 alone.
    #[test]
    #[expect(clippy::unwrap_used, reason = "test asserts Ok")]
    fn a_chunk_over_several_pages_names_the_last_one() {
        let pages: Vec<Section> = (1..=6).map(|n| page(n, 8)).collect();
        let chunks = Chunker::new(512, 64, "cl100k_base")
            .and_then(|c| c.pages(&pages, None))
            .unwrap();
        assert_eq!(chunks.len(), 1, "{chunks:?}");
        let only = chunks.first().unwrap();
        assert_eq!(only.page, Some(1));
        assert_eq!(only.locator.as_deref(), Some("through page 6"));
    }

    /// Windows that span a break name their last page; one that stays on
    /// its page has no locator.
    #[test]
    #[expect(clippy::unwrap_used, reason = "test asserts Ok")]
    fn only_a_chunk_that_crosses_a_page_break_has_a_range() {
        let pages = vec![page(1, 8), page(2, 8), page(3, 8)];
        let chunks = Chunker::new(50, 10, "cl100k_base")
            .and_then(|c| c.pages(&pages, None))
            .unwrap();
        for chunk in &chunks {
            let first = chunk.page.unwrap();
            let last = (first..=3)
                .rev()
                .find(|n| chunk.content.contains(&format!("p{n}w")))
                .unwrap();
            let expected = (last > first).then(|| format!("through page {last}"));
            assert_eq!(chunk.locator, expected, "{chunk:?}");
        }
        assert!(chunks.iter().any(|c| c.locator.is_some()), "{chunks:?}");
    }

    #[test]
    #[expect(clippy::unwrap_used, reason = "test asserts Ok")]
    fn page_lookup_skips_separator_to_following_page() {
        let enc = tiktoken::get_encoding("cl100k_base").unwrap();
        let chunk_size: u32 = 512;
        let overlap: u32 = 64;
        let step: usize = 448; // chunk_size - overlap, the shipped config step
        let mut words1 = 1_usize;
        while !enc.encode(&page(1, words1).text).len().is_multiple_of(step) {
            words1 = words1.saturating_add(1);
        }
        let pages = vec![page(1, words1), page(2, 512)];
        let chunks = Chunker::new(chunk_size, overlap, "cl100k_base")
            .and_then(|c| c.pages(&pages, Some("Doc")))
            .unwrap();
        let boundary = chunks
            .iter()
            .find(|c| c.content.starts_with("p2w0"))
            .unwrap();
        assert_eq!(
            boundary.page,
            Some(2),
            "boundary chunk labelled {:?}, expected Some(2): {:?}",
            boundary.page,
            boundary
                .content
                .get(..40)
                .unwrap_or(boundary.content.as_str())
        );
    }

    #[test]
    #[expect(clippy::unwrap_used, reason = "test asserts Ok")]
    fn page_lookup_across_skipped_pages_points_to_kept_page() {
        let enc = tiktoken::get_encoding("cl100k_base").unwrap();
        let chunk_size: u32 = 512;
        let overlap: u32 = 64;
        let step: usize = 448; // chunk_size - overlap, the shipped config step
        let mut words1 = 1_usize;
        while !enc.encode(&page(1, words1).text).len().is_multiple_of(step) {
            words1 = words1.saturating_add(1);
        }
        // Sections exactly as `extract_pdf_pages` emits after skipping pages
        // 2-4: `page` keeps the original PDF number, skipped pages produce no
        // section, so the page gap carries through to the citation locator.
        let pages = vec![page(1, words1), page(5, 512)];
        let chunks = Chunker::new(chunk_size, overlap, "cl100k_base")
            .and_then(|c| c.pages(&pages, Some("Doc")))
            .unwrap();
        let boundary = chunks
            .iter()
            .find(|c| c.content.starts_with("p5w0"))
            .unwrap();
        assert_eq!(
            boundary.page,
            Some(5),
            "skipped-page boundary chunk labelled {:?}, expected Some(5): {:?}",
            boundary.page,
            boundary
                .content
                .get(..40)
                .unwrap_or(boundary.content.as_str())
        );
    }

    #[test]
    fn continuous_pages_with_no_text_give_no_chunks() {
        let pages = vec![Section::body(None, "").on_page(Some(1))];
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
                        sections: sections.clone(),
                        flow: Flow::Sectioned,
                        ..Extracted::default()
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
                        sections,
                        flow: Flow::Continuous,
                        ..Extracted::default()
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

#[cfg(test)]
mod kind_tests {
    use super::*;

    /// A chunk's kind, heading, page, and locator.
    type Placed<'a> = (&'a str, Option<&'a str>, Option<u32>, Option<&'a str>);

    fn chunker() -> Chunker {
        Chunker::new(40, 5, "cl100k_base").unwrap_or_else(|_| unreachable_chunker())
    }

    #[expect(clippy::panic, reason = "test failure path")]
    fn unreachable_chunker() -> ! {
        panic!("the encoding exists")
    }

    #[test]
    fn a_table_is_split_by_rows_with_its_header_on_every_piece() {
        let rows: Vec<Vec<String>> =
            std::iter::once(vec![String::from("Item"), String::from("Amount")])
                .chain((0..30).map(|i| vec![format!("item number {i}"), format!("{i}00")]))
                .collect();
        let table = Table::from_rows(rows)
            .map(|t| t.render())
            .unwrap_or_default();
        let pieces = chunker().table(&table);
        assert!(pieces.len() > 1, "{pieces:?}");
        for piece in &pieces {
            assert!(
                piece.starts_with("| Item | Amount |\n| --- | --- |\n| item"),
                "{piece}"
            );
        }
        let joined = pieces.join("\n");
        assert!(joined.contains("| item number 29 | 2900 |"));
        assert_eq!(
            chunker().table("not a table"),
            vec![String::from("not a table")]
        );
    }

    #[test]
    fn code_is_split_by_lines_with_the_starting_line_as_locator() {
        let code: String = (1..=60)
            .map(|i| format!("let v{i} = {i};"))
            .collect::<Vec<_>>()
            .join("\n");
        let pieces = chunker().lines(&code);
        assert!(pieces.len() > 1);
        assert_eq!(
            pieces.first().and_then(|(_, l)| l.as_deref()),
            Some("line 1")
        );
        let second_start = pieces
            .get(1)
            .and_then(|(content, _)| content.lines().next())
            .unwrap_or_default();
        let second_locator = pieces
            .get(1)
            .and_then(|(_, l)| l.clone())
            .unwrap_or_default();
        let n: usize = second_locator
            .trim_start_matches("line ")
            .parse()
            .unwrap_or(0);
        assert_eq!(second_start, format!("let v{n} = {n};"));
        assert!(chunker().lines("\n\n").is_empty());
    }

    #[test]
    fn sections_carry_kind_and_locator_and_continuous_flow_keeps_tables_apart() {
        let sections = vec![
            Section::body(Some(String::from("A")), "body text").on_page(Some(1)),
            Section::table(
                Some(String::from("A")),
                String::from("| x | y |\n| --- | --- |\n| 1 | 2 |"),
            )
            .on_page(Some(1)),
            Section::body(None, "more body").on_page(Some(2)),
            Section::note(Some(String::from("B")), "a note").at("message 2"),
        ];
        let chunks = chunker()
            .document(
                &Extracted {
                    sections: sections.clone(),
                    flow: Flow::Continuous,
                    ..Extracted::default()
                },
                Some("file"),
            )
            .unwrap_or_default();
        let summary: Vec<Placed<'_>> = chunks
            .iter()
            .map(|c| {
                (
                    c.kind.as_str(),
                    c.heading.as_deref(),
                    c.page,
                    c.locator.as_deref(),
                )
            })
            .collect();
        assert_eq!(
            summary,
            vec![
                ("body", Some("A"), Some(1), None),
                ("table", Some("A"), Some(1), None),
                ("body", Some("A"), Some(2), None),
                ("note", Some("B"), None, Some("message 2")),
            ]
        );
        let sectioned = chunker().sections(&sections).unwrap_or_default();
        assert_eq!(sectioned.len(), 4);
        assert_eq!(sectioned.get(1).map(|c| c.kind), Some(SectionKind::Table));
    }
}
