//! Reasoning a model writes into its answer text.
//!
//! Some models (Qwen3 and DeepSeek-R1 on Ollama among them) put their
//! reasoning in the answer itself, inside `<think>…</think>`, even with
//! reasoning turned off, when it is often an empty block. Only a block that
//! opens a model call's text is reasoning: the same tag later in an answer
//! is kept, since the answer may be about it.

use std::mem;

const OPEN: &str = "<think>";
const CLOSE: &str = "</think>";

/// What [`ThinkFilter::push`] lets through.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct Filtered {
    /// Answer text.
    pub text: String,
    /// Some of the input was reasoning.
    pub reasoning: bool,
}

/// Drops the `<think>` block that opens one model call's streamed text,
/// whatever the deltas it arrives in.
#[derive(Debug, Default)]
pub struct ThinkFilter {
    state: State,
    /// Text held until it is known whether it opens a block or closes one:
    /// leading whitespace and the start of a tag.
    held: String,
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
enum State {
    /// Only whitespace so far.
    #[default]
    Start,
    /// Inside the opening block.
    Thinking,
    /// The block has closed; whitespace after it is not answer text.
    Closed,
    /// Answer text, passed on as it comes.
    Answer,
}

impl ThinkFilter {
    /// The answer text in `delta`.
    pub fn push(&mut self, delta: &str) -> Filtered {
        match self.state {
            State::Answer => Filtered {
                text: delta.to_owned(),
                reasoning: false,
            },
            State::Start => {
                self.held.push_str(delta);
                let trimmed = self.held.trim_start();
                if let Some(rest) = trimmed.strip_prefix(OPEN) {
                    let rest = rest.to_owned();
                    self.held.clear();
                    self.state = State::Thinking;
                    let text = self.push(&rest).text;
                    Filtered {
                        text,
                        reasoning: true,
                    }
                } else if OPEN.starts_with(trimmed) {
                    // Whitespace, or the tag's first characters: wait.
                    Filtered::default()
                } else {
                    self.state = State::Answer;
                    Filtered {
                        text: mem::take(&mut self.held),
                        reasoning: false,
                    }
                }
            }
            State::Thinking => {
                self.held.push_str(delta);
                if let Some(at) = self.held.find(CLOSE) {
                    let rest = self
                        .held
                        .get(at.saturating_add(CLOSE.len())..)
                        .unwrap_or_default()
                        .to_owned();
                    self.held.clear();
                    self.state = State::Closed;
                    let text = self.push(&rest).text;
                    Filtered {
                        text,
                        reasoning: true,
                    }
                } else {
                    // Keep only what could be the start of the closing tag.
                    let cut = self
                        .held
                        .len()
                        .saturating_sub(partial_suffix(&self.held, CLOSE));
                    self.held.drain(..cut);
                    Filtered {
                        text: String::new(),
                        reasoning: true,
                    }
                }
            }
            State::Closed => {
                let text = delta.trim_start();
                if !text.is_empty() {
                    self.state = State::Answer;
                }
                Filtered {
                    text: text.to_owned(),
                    reasoning: false,
                }
            }
        }
    }

    /// The call is over: text still held at its start is answer text after
    /// all; an unclosed block is dropped. The filter is ready for the next
    /// call.
    pub fn finish(&mut self) -> String {
        let held = mem::take(&mut self.held);
        match mem::take(&mut self.state) {
            State::Start => held,
            State::Thinking | State::Closed | State::Answer => String::new(),
        }
    }
}

/// `text` without the `<think>` block that opens it, for an answer that
/// arrived whole rather than streamed.
#[must_use]
pub fn strip(text: &str) -> String {
    let mut filter = ThinkFilter::default();
    let mut out = filter.push(text).text;
    out.push_str(&filter.finish());
    out
}

/// The length of the longest suffix of `text` that is a proper prefix of
/// `tag`.
fn partial_suffix(text: &str, tag: &str) -> usize {
    (1..tag.len())
        .rev()
        .find(|&n| tag.get(..n).is_some_and(|prefix| text.ends_with(prefix)))
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The text `deltas` stream as, and whether any was reasoning.
    fn stream(deltas: &[&str]) -> (String, bool) {
        let mut filter = ThinkFilter::default();
        let mut text = String::new();
        let mut reasoning = false;
        for delta in deltas {
            let out = filter.push(delta);
            text.push_str(&out.text);
            reasoning |= out.reasoning;
        }
        text.push_str(&filter.finish());
        (text, reasoning)
    }

    #[test]
    fn an_opening_block_is_dropped() {
        assert_eq!(
            stream(&["<think>\nWest fell.\n</think>\n\nWest fell by $118k."]),
            (String::from("West fell by $118k."), true)
        );
    }

    /// Qwen3 with reasoning off still writes an empty block.
    #[test]
    fn an_empty_block_is_dropped() {
        assert_eq!(
            stream(&["<think>\n\n</think>\n\n", "Hello."]),
            (String::from("Hello."), true)
        );
    }

    #[test]
    fn tags_split_across_deltas_are_found() {
        assert_eq!(
            stream(&[
                "  <th",
                "ink>plan ",
                "the query</thi",
                "nk>",
                "\n",
                "Answer."
            ]),
            (String::from("Answer."), true)
        );
    }

    #[test]
    fn text_without_a_block_passes_untouched() {
        assert_eq!(
            stream(&["  The ", "answer."]),
            (String::from("  The answer."), false)
        );
        assert_eq!(stream(&["<"]), (String::from("<"), false));
        assert_eq!(
            stream(&["<b>bold</b>"]),
            (String::from("<b>bold</b>"), false)
        );
    }

    /// The tag in the middle of an answer is about the tag.
    #[test]
    fn a_later_tag_is_kept() {
        let text = "Qwen writes <think> before reasoning.";
        assert_eq!(stream(&[text]), (String::from(text), false));
    }

    #[test]
    fn an_unclosed_block_is_dropped() {
        assert_eq!(stream(&["<think>still going"]), (String::new(), true));
    }

    #[test]
    fn finish_readies_the_filter_for_the_next_call() {
        let mut filter = ThinkFilter::default();
        assert_eq!(filter.push("<think>a</think>b").text, "b");
        assert!(filter.finish().is_empty());
        assert_eq!(filter.push("<think>c</think> d").text, "d");
    }

    #[test]
    fn strip_handles_a_whole_answer() {
        assert_eq!(strip("<think>x</think>\n\nThe answer."), "The answer.");
        assert_eq!(strip("The answer."), "The answer.");
    }
}
