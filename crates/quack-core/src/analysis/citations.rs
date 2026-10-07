//! Citations: the chunks a turn retrieved, numbered `[n]`, and the check
//! that the answer only cites chunks it actually saw.

use std::fmt;
use std::sync::{Arc, Mutex};

use jiff::civil::DateTime;

use crate::ids::{ChunkId, DocumentId};
use crate::storage::workspace::ChunkSearchResult;
use crate::text::OneLine;

/// One retrievable source the model may cite by its marker.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize, utoipa::ToSchema)]
pub struct Citation {
    pub n: u32,
    pub chunk_id: ChunkId,
    pub document_id: DocumentId,
    pub filename: String,
    pub chunk_index: u32,
    pub page: Option<u32>,
    pub heading: Option<String>,
    /// Where the chunk sits in a source without pages (`line 40`,
    /// `12:04`, `chapter 3`, `message 2`); `None` for a paged source and
    /// on answers recorded before it was kept.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub locator: Option<String>,
    /// When the document was ingested (UTC); `None` on answers recorded
    /// before it was kept.
    #[serde(default)]
    #[schema(value_type = Option<String>)]
    pub ingested_at: Option<DateTime>,
    /// The start of the cited chunk's text, at most [`Self::EXCERPT_CHARS`]
    /// characters with an ellipsis when cut; empty on answers recorded
    /// before it was kept. The passage page has the whole chunk.
    #[serde(default)]
    pub excerpt: String,
}

impl Citation {
    /// Characters of chunk text an excerpt keeps.
    pub const EXCERPT_CHARS: usize = 500;

    /// The retrieved chunk `hit`, cited as `[n]`.
    #[must_use]
    pub fn new(n: u32, hit: &ChunkSearchResult) -> Self {
        let text = hit.content.trim();
        let mut excerpt: String = text.chars().take(Self::EXCERPT_CHARS).collect();
        if text.chars().count() > Self::EXCERPT_CHARS {
            excerpt.push('\u{2026}');
        }
        Self {
            n,
            chunk_id: hit.id.clone(),
            document_id: hit.document_id.clone(),
            filename: hit.filename.clone(),
            chunk_index: hit.chunk_index,
            page: hit.page,
            heading: hit.heading.clone(),
            locator: hit.locator.clone(),
            ingested_at: Some(hit.ingested_at),
            excerpt,
        }
    }

    /// `filename, page 12, under "Exclusions", ingested 2026-10-05` for
    /// footers and status lines.
    #[must_use]
    pub fn label(&self) -> String {
        let location = ChunkLocation {
            filename: &self.filename,
            page: self.page,
            locator: self.locator.as_deref(),
            heading: self.heading.as_deref(),
        };
        match self.ingested_at {
            Some(at) => format!("{location}, ingested {}", at.date()),
            None => location.to_string(),
        }
    }
}

/// Where a chunk sits: `policy.pdf, page 12, under "Exclusions"`.
pub struct ChunkLocation<'a> {
    pub filename: &'a str,
    pub page: Option<u32>,
    /// `line 40`, `12:04`, `chapter 3`, `message 2`: the place in a source
    /// that has no pages.
    pub locator: Option<&'a str>,
    pub heading: Option<&'a str>,
}

impl<'a> From<&'a ChunkSearchResult> for ChunkLocation<'a> {
    fn from(chunk: &'a ChunkSearchResult) -> Self {
        Self {
            filename: &chunk.filename,
            page: chunk.page,
            locator: chunk.locator.as_deref(),
            heading: chunk.heading.as_deref(),
        }
    }
}

impl fmt::Display for ChunkLocation<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", OneLine(self.filename))?;
        if let Some(page) = self.page {
            write!(f, ", page {page}")?;
        }
        if let Some(locator) = self.locator {
            write!(f, ", {}", OneLine(locator))?;
        }
        if let Some(heading) = self.heading {
            write!(f, ", under \"{}\"", OneLine(heading))?;
        }
        Ok(())
    }
}

/// The footer an answer's citations are listed in: `Sources:`, then one
/// `  [n] location` line per citation.
pub struct Sources<'a>(pub &'a [Citation]);

impl fmt::Display for Sources<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("Sources:")?;
        for citation in self.0 {
            write!(f, "\n  [{}] {}", citation.n, citation.label())?;
        }
        Ok(())
    }
}

/// The markers one registration assigned: `[first]`, `[first + 1]`, and so
/// on, one per chunk in the order registered.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Markers {
    first: u32,
}

impl Markers {
    /// Markers counting up from `first`.
    #[must_use]
    pub const fn starting_at(first: u32) -> Self {
        Self { first }
    }

    #[must_use]
    pub const fn first(self) -> u32 {
        self.first
    }

    /// The marker of the `index`th chunk registered.
    #[must_use]
    pub fn nth(self, index: usize) -> u32 {
        self.first
            .saturating_add(u32::try_from(index).unwrap_or(u32::MAX))
    }
}

/// An answer with its citation markers checked: the text, and the chunks
/// it cites in order of first use.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CitedAnswer {
    pub text: String,
    pub citations: Vec<Citation>,
}

/// Chunks retrieved during one turn, numbered in the order they were shown
/// to the model. Shared between the search tool and the agent loop.
#[derive(Debug, Clone, Default)]
pub struct CitationRegistry {
    inner: Arc<Mutex<Vec<Citation>>>,
}

impl CitationRegistry {
    /// Assign markers to `hits`, continuing from the last one.
    #[must_use]
    pub fn register(&self, hits: &[ChunkSearchResult]) -> Markers {
        let Ok(mut all) = self.inner.lock() else {
            return Markers { first: 1 };
        };
        let markers = Markers {
            first: u32::try_from(all.len())
                .unwrap_or(u32::MAX)
                .saturating_add(1),
        };
        for (i, hit) in hits.iter().enumerate() {
            all.push(Citation::new(markers.nth(i), hit));
        }
        markers
    }

    #[must_use]
    pub fn all(&self) -> Vec<Citation> {
        self.inner.lock().map(|c| c.clone()).unwrap_or_default()
    }

    /// Keep the `[n]` markers in `answer` that name a registered chunk,
    /// strip the rest, and renumber the kept ones from 1 in order of first
    /// use so the footer reads naturally.
    #[must_use]
    pub fn validate(&self, answer: &str) -> CitedAnswer {
        let registered = self.all();
        // Some models emit fullwidth brackets (【1】) or superscript-style
        // `[^1]`; normalize to `[1]` before scanning.
        let answer = answer
            .replace('【', "[")
            .replace('】', "]")
            .replace("[^", "[");
        let mut out = String::with_capacity(answer.len());
        let mut cited: Vec<Citation> = Vec::new();

        // Walk the text as `text[marker]text[marker]...`. Every '[' starts a
        // candidate; only `[digits]` naming a registered chunk is a marker.
        let mut pieces = answer.split('[');
        if let Some(first) = pieces.next() {
            out.push_str(first);
        }
        for piece in pieces {
            let Some((inside, after)) = piece.split_once(']') else {
                out.push('[');
                out.push_str(piece);
                continue;
            };
            // `[3]`, or a group the model wrote as `[2, 3]`: each number
            // that names a registered chunk becomes its own marker.
            let numbers: Option<Vec<u32>> = inside
                .split(',')
                .map(|part| {
                    let digits = part.trim();
                    if digits.is_empty() || !digits.chars().all(|c| c.is_ascii_digit()) {
                        None
                    } else {
                        digits.parse::<u32>().ok()
                    }
                })
                .collect();
            let Some(numbers) = numbers else {
                if !is_channel_marker(inside) {
                    // Not a marker; a provider's channel token is dropped.
                    out.push('[');
                    out.push_str(inside);
                    out.push(']');
                }
                out.push_str(after);
                continue;
            };
            for n in numbers {
                // A number the model invented is dropped.
                let Some(source) = registered.iter().find(|c| c.n == n) else {
                    continue;
                };
                let renumbered =
                    if let Some(pos) = cited.iter().position(|c| c.chunk_id == source.chunk_id) {
                        u32::try_from(pos).unwrap_or(u32::MAX).saturating_add(1)
                    } else {
                        let next = u32::try_from(cited.len())
                            .unwrap_or(u32::MAX)
                            .saturating_add(1);
                        let mut c = source.clone();
                        c.n = next;
                        cited.push(c);
                        next
                    };
                out.push('[');
                out.push_str(&renumbered.to_string());
                out.push(']');
            }
            out.push_str(after);
        }
        CitedAnswer {
            text: out,
            citations: cited,
        }
    }
}

/// A harmony-style channel token some providers leak into the answer,
/// such as `[commentary:functions.run_sql]` or `[analysis]`.
fn is_channel_marker(inside: &str) -> bool {
    let inside = inside.trim();
    ["commentary", "analysis", "final"].iter().any(|word| {
        inside
            .strip_prefix(word)
            .is_some_and(|rest| rest.is_empty() || rest.starts_with([':', ' ']))
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ingestion::parser::SectionKind;
    use crate::storage::workspace::Ranks;

    fn hit(id: &str, file: &str, idx: u32) -> ChunkSearchResult {
        ChunkSearchResult {
            id: ChunkId::from(id.to_owned()),
            content: String::new(),
            document_id: DocumentId::from(format!("doc-{file}")),
            chunk_index: idx,
            filename: file.to_owned(),
            heading: None,
            page: Some(idx.saturating_add(1)),
            score: 1.0,
            kind: SectionKind::Body,
            locator: None,
            ingested_at: DateTime::constant(2026, 10, 5, 14, 3, 0, 0),
            ranks: Ranks::default(),
        }
    }

    #[test]
    fn markers_continue_across_searches() {
        let registry = CitationRegistry::default();
        assert_eq!(
            registry
                .register(&[hit("a", "p.pdf", 0), hit("b", "p.pdf", 1)])
                .first(),
            1
        );
        assert_eq!(registry.register(&[hit("c", "q.md", 0)]).nth(0), 3);
        let all = registry.all();
        assert_eq!(all.iter().map(|c| c.n).collect::<Vec<_>>(), vec![1, 2, 3]);
        assert_eq!(all.last().map(|c| c.chunk_id.as_str()), Some("c"));
    }

    #[test]
    fn validate_keeps_known_markers_renumbered_and_drops_unknown() {
        let registry = CitationRegistry::default();
        let first = registry.register(&[
            hit("a", "p.pdf", 0),
            hit("b", "p.pdf", 1),
            hit("c", "q.md", 0),
        ]);
        assert_eq!(first.first(), 1);
        let CitedAnswer {
            text,
            citations: cited,
        } = registry.validate(
            "Flood is excluded [3]. Claims close in 30 days [1][3]. See also [7] and [x] and [ 2 ].",
        );
        assert_eq!(
            text,
            "Flood is excluded [1]. Claims close in 30 days [2][1]. See also  and [x] and [3]."
        );
        assert_eq!(
            cited
                .iter()
                .map(|c| (c.n, c.chunk_id.as_str()))
                .collect::<Vec<_>>(),
            vec![(1, "c"), (2, "a"), (3, "b")]
        );
    }

    #[test]
    fn a_group_of_markers_is_checked_number_by_number() {
        let registry = CitationRegistry::default();
        assert_eq!(
            registry
                .register(&[hit("a", "p.pdf", 0), hit("b", "p.pdf", 1)])
                .first(),
            1
        );
        let CitedAnswer { text, citations } = registry.validate(
            "Covered [2, 1]. Partly made up [1, 7]. All made up [3, 4]. A list [a, b] and [1,].",
        );
        assert_eq!(
            text,
            "Covered [1][2]. Partly made up [2]. All made up . A list [a, b] and [1,]."
        );
        assert_eq!(
            citations
                .iter()
                .map(|c| c.chunk_id.as_str())
                .collect::<Vec<_>>(),
            ["b", "a"]
        );
    }

    #[test]
    fn validate_accepts_fullwidth_and_footnote_brackets() {
        let registry = CitationRegistry::default();
        assert_eq!(registry.register(&[hit("a", "p.pdf", 0)]).first(), 1);
        let CitedAnswer {
            text,
            citations: cited,
        } = registry.validate("Renews in March【1】 and again[^1].");
        assert_eq!(text, "Renews in March[1] and again[1].");
        assert_eq!(cited.len(), 1);
    }

    #[test]
    fn validate_strips_leaked_channel_markers() {
        let registry = CitationRegistry::default();
        assert_eq!(registry.register(&[hit("a", "p.pdf", 0)]).first(), 1);
        let CitedAnswer {
            text,
            citations: cited,
        } = registry.validate(
            "[commentary:functions.run_sql] There were 12 storms [1].[analysis] [final] [finally]",
        );
        assert_eq!(text, " There were 12 storms [1].  [finally]");
        assert_eq!(cited.len(), 1);
    }

    /// The excerpt is the chunk's trimmed text up to the bound, with an
    /// ellipsis when cut on a character, never a byte, boundary.
    #[test]
    fn excerpt_is_the_chunk_text_cut_at_the_bound() {
        let short = Citation::new(
            1,
            &ChunkSearchResult {
                content: String::from("  Flood is excluded.  "),
                ..hit("a", "p.pdf", 0)
            },
        );
        assert_eq!(short.excerpt, "Flood is excluded.");
        let long = Citation::new(
            1,
            &ChunkSearchResult {
                content: "é".repeat(Citation::EXCERPT_CHARS.saturating_add(1)),
                ..hit("a", "p.pdf", 0)
            },
        );
        assert_eq!(
            long.excerpt.chars().count(),
            Citation::EXCERPT_CHARS.saturating_add(1)
        );
        assert!(long.excerpt.ends_with('\u{2026}'));
        let exact = Citation::new(
            1,
            &ChunkSearchResult {
                content: "x".repeat(Citation::EXCERPT_CHARS),
                ..hit("a", "p.pdf", 0)
            },
        );
        assert!(!exact.excerpt.ends_with('\u{2026}'));
    }

    #[test]
    fn validate_leaves_text_without_markers_alone() {
        let CitedAnswer {
            text,
            citations: cited,
        } = CitationRegistry::default().validate("no citations [here");
        assert_eq!(text, "no citations [here");
        assert!(cited.is_empty());
    }

    #[test]
    fn label_includes_page_heading_and_ingestion_date_when_present() {
        let mut c = Citation {
            n: 1,
            chunk_id: ChunkId::from("a"),
            document_id: DocumentId::from("d"),
            filename: String::from("policy.pdf"),
            chunk_index: 0,
            page: Some(12),
            heading: Some(String::from("Exclusions")),
            locator: None,
            ingested_at: Some(DateTime::constant(2026, 10, 5, 14, 3, 0, 0)),
            excerpt: String::new(),
        };
        assert_eq!(
            c.label(),
            "policy.pdf, page 12, under \"Exclusions\", ingested 2026-10-05"
        );
        c.ingested_at = None;
        assert_eq!(c.label(), "policy.pdf, page 12, under \"Exclusions\"");
        // A source without pages cites its locator instead.
        c.page = None;
        c.locator = Some(String::from("line 40"));
        c.filename = String::from("main.rs");
        assert_eq!(c.label(), "main.rs, line 40, under \"Exclusions\"");
    }

    /// An answer recorded before the ingestion time was kept still reads.
    #[test]
    #[expect(
        clippy::unwrap_used,
        clippy::indexing_slicing,
        reason = "test asserts Ok and the field exists"
    )]
    fn a_stored_citation_without_an_ingestion_time_still_decodes() {
        let stored = r#"{"n":1,"chunk_id":"a","document_id":"d","filename":"p.pdf","chunk_index":0,"page":null,"heading":null}"#;
        let citation: Citation = serde_json::from_str(stored).unwrap();
        assert_eq!(citation.ingested_at, None);
        let fresh = Citation::new(2, &hit("b", "q.md", 0));
        let json = serde_json::to_value(&fresh).unwrap();
        assert_eq!(json["ingested_at"], "2026-10-05T14:03:00");
    }
}
