//! Source code: one section of kind `code`, which the chunker cuts at
//! line boundaries so every chunk's locator names the line it starts on.
//! Nothing is parsed; a definition-aware split needs a grammar per
//! language, which is a dependency decision left open.

use super::parser::{Extracted, Flow, Section, SectionKind};

/// A source file as one code section, its file name's stem for a title
/// is left to the caller.
#[must_use]
pub fn extract(text: &str) -> Extracted {
    Extracted {
        sections: vec![Section {
            text: text.to_owned(),
            kind: SectionKind::Code,
            ..Section::default()
        }],
        flow: Flow::Sectioned,
        ..Extracted::default()
    }
}
