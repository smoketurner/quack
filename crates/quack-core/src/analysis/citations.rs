//! Citations: the chunks a turn retrieved, numbered `[n]`, and the check
//! that the answer only cites chunks it actually saw.

use std::fmt;
use std::sync::{Arc, Mutex};

use crate::ids::{ChunkId, DocumentId};
use crate::storage::workspace::ChunkSearchResult;

/// One retrievable source the model may cite by its marker.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Citation {
    pub n: u32,
    pub chunk_id: ChunkId,
    pub document_id: DocumentId,
    pub filename: String,
    pub chunk_index: u32,
    pub page: Option<u32>,
    pub heading: Option<String>,
}

impl Citation {
    /// The retrieved chunk `hit`, cited as `[n]`.
    #[must_use]
    pub fn new(n: u32, hit: &ChunkSearchResult) -> Self {
        Self {
            n,
            chunk_id: hit.id.clone(),
            document_id: hit.document_id.clone(),
            filename: hit.filename.clone(),
            chunk_index: hit.chunk_index,
            page: hit.page,
            heading: hit.heading.clone(),
        }
    }

    /// `filename, page 12, under "Exclusions"` for footers and status lines.
    #[must_use]
    pub fn label(&self) -> String {
        ChunkLocation {
            filename: &self.filename,
            page: self.page,
            heading: self.heading.as_deref(),
        }
        .to_string()
    }
}

/// Where a chunk sits: `policy.pdf, page 12, under "Exclusions"`.
pub struct ChunkLocation<'a> {
    pub filename: &'a str,
    pub page: Option<u32>,
    pub heading: Option<&'a str>,
}

impl<'a> From<&'a ChunkSearchResult> for ChunkLocation<'a> {
    fn from(chunk: &'a ChunkSearchResult) -> Self {
        Self {
            filename: &chunk.filename,
            page: chunk.page,
            heading: chunk.heading.as_deref(),
        }
    }
}

impl fmt::Display for ChunkLocation<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.filename)?;
        if let Some(page) = self.page {
            write!(f, ", page {page}")?;
        }
        if let Some(heading) = self.heading {
            write!(f, ", under \"{heading}\"")?;
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
            let digits = inside.trim();
            let parsed = if digits.is_empty() || !digits.chars().all(|c| c.is_ascii_digit()) {
                None
            } else {
                digits.parse::<u32>().ok()
            };
            match parsed.and_then(|n| registered.iter().find(|c| c.n == n)) {
                Some(source) => {
                    let renumbered = if let Some(pos) =
                        cited.iter().position(|c| c.chunk_id == source.chunk_id)
                    {
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
                None if parsed.is_some() || is_channel_marker(inside) => {
                    // A marker the model invented, or a provider's channel
                    // token that leaked into the text: dropped.
                }
                None => {
                    out.push('[');
                    out.push_str(inside);
                    out.push(']');
                }
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
    fn label_includes_page_and_heading_when_present() {
        let c = Citation {
            n: 1,
            chunk_id: ChunkId::from("a"),
            document_id: DocumentId::from("d"),
            filename: String::from("policy.pdf"),
            chunk_index: 0,
            page: Some(12),
            heading: Some(String::from("Exclusions")),
        };
        assert_eq!(c.label(), "policy.pdf, page 12, under \"Exclusions\"");
    }
}
