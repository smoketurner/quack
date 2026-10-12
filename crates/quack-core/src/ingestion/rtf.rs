//! Rich Text Format through `rclip-rtf`: the document's text as one
//! section, paragraphs kept.

use rclip_rtf::Document;

use super::parser::{Extracted, Flow, Section};
use crate::error::{Error, Result};

/// An RTF file's text.
///
/// # Errors
///
/// Returns an error when the bytes are not RTF or hold no words.
pub fn extract(data: &[u8]) -> Result<Extracted> {
    let document = Document::parse(data).map_err(|e| Error::Ingestion(format!("not RTF: {e}")))?;
    let text = document.text.trim();
    if text.is_empty() {
        return Err(Error::Ingestion(String::from(
            "no extractable text: the RTF file has no words",
        )));
    }
    Ok(Extracted {
        sections: vec![Section::body(None, text)],
        flow: Flow::Sectioned,
        ..Extracted::default()
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn text_of(rtf: &[u8]) -> Option<String> {
        extract(rtf)
            .ok()
            .and_then(|e| e.sections.first().map(|s| s.text.clone()))
    }

    #[test]
    fn rtf_text_is_read_and_junk_is_refused() {
        let rtf =
            br"{\rtf1\ansi{\fonttbl\f0\fswiss Helvetica;}\f0\pard Flood is {\b excluded}.\par }";
        assert_eq!(text_of(rtf).as_deref(), Some("Flood is excluded."));
        assert!(extract(br"{\rtf1 }").is_err());
        assert!(extract(b"plain text").is_err());
    }

    #[test]
    fn escapes_decode_under_the_declared_code_page() {
        let rtf = br"{\rtf1\ansi\ansicpg1252 caf\'e9 \u-10179?\u-8704? end\par next}";
        assert_eq!(text_of(rtf).as_deref(), Some("café 😀 end\nnext"));
    }

    #[test]
    fn a_lone_surrogate_becomes_a_replacement_character() {
        assert_eq!(
            text_of(br"{\rtf1 a\u55556 b\u-10179}").as_deref(),
            Some("a\u{FFFD}\u{FFFD}")
        );
        assert!(extract(b"{\\}5\0\0bt\\u55556}").is_err());
    }
}
