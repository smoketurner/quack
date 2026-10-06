//! Rich Text Format through `rtf-parser`: the document's text as one
//! section, paragraphs kept.

use rtf_parser::document::RtfDocument;

use super::parser::{Extracted, Flow, Section};
use crate::error::{Error, Result};

/// An RTF file's text.
///
/// # Errors
///
/// Returns an error when the text is not RTF or holds no words.
pub fn extract(text: &str) -> Result<Extracted> {
    let document =
        RtfDocument::try_from(text).map_err(|e| Error::Ingestion(format!("not RTF: {e}")))?;
    let text = document.get_text();
    let text = text.trim();
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

    #[test]
    fn rtf_text_is_read_and_junk_is_refused() {
        let rtf =
            r"{\rtf1\ansi{\fonttbl\f0\fswiss Helvetica;}\f0\pard Flood is {\b excluded}.\par }";
        let extracted = extract(rtf).unwrap_or_default();
        assert_eq!(
            extracted.sections.first().map(|s| s.text.as_str()),
            Some("Flood is excluded.")
        );
        assert!(extract(r"{\rtf1 }").is_err());
    }
}
