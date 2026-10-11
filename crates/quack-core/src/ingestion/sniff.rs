//! A file's type from its bytes, for a name whose extension says none: a
//! piped file, a download saved without one. The answer is an extension,
//! so the file is recorded under a name that says its type and every later
//! step reads the type from the name as usual.
//!
//! `file_format` reads the signatures and looks inside containers: the
//! parts of a zip tell DOCX, XLSX, PPTX, ODT, ODS, and EPUB apart, and the
//! streams of an OLE2 file tell `.doc`, `.xls`, and `.ppt` apart. Text it
//! can only call plain is told apart here: JSON, HTML, captions.

use std::io::{Read, Seek, SeekFrom};

use file_format::FileFormat;

use super::parser::FileType;

/// Bytes read from the start of a file to tell kinds of text apart.
const HEAD: usize = 8 * 1024;

/// The extension the bytes `data` show, when they are a type quack reads;
/// `None` otherwise, a `.doc` among them until there is a parser for it.
#[must_use]
pub fn extension(mut data: impl Read + Seek) -> Option<String> {
    let mut head = Vec::with_capacity(HEAD);
    data.by_ref()
        .take(u64::try_from(HEAD).unwrap_or(u64::MAX))
        .read_to_end(&mut head)
        .ok()?;
    data.seek(SeekFrom::Start(0)).ok()?;
    let format = FileFormat::from_reader(data).ok()?;
    match format {
        FileFormat::Empty | FileFormat::PlainText => text(&head).map(str::to_owned),
        _ => {
            let ext = format.extension();
            FileType::of(&format!("file.{ext}")).map(|_| ext.to_owned())
        }
    }
}

/// UTF-8 text, by how it starts; `None` for bytes that are not text.
fn text(head: &[u8]) -> Option<&'static str> {
    let text = match std::str::from_utf8(head) {
        Ok(text) => text,
        // Cut mid-character at the end of the head: still text.
        Err(e) if e.error_len().is_none() => head
            .get(..e.valid_up_to())
            .and_then(|valid| std::str::from_utf8(valid).ok())?,
        Err(_) => return None,
    };
    if text.contains('\0') {
        return None;
    }
    let start = text.trim_start_matches('\u{feff}').trim_start();
    let lower = start
        .get(..start.len().min(16))
        .unwrap_or_default()
        .to_ascii_lowercase();
    Some(if start.starts_with('{') || start.starts_with('[') {
        "json"
    } else if lower.starts_with("<!doctype html") || lower.starts_with("<html") {
        "html"
    } else if start.starts_with("WEBVTT") {
        "vtt"
    } else {
        "txt"
    })
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use super::super::zipped::tests::package;
    use super::*;

    fn of(bytes: &[u8]) -> Option<String> {
        extension(Cursor::new(bytes))
    }

    /// An EPUB or `OpenDocument` package: `mimetype` first and stored
    /// uncompressed, as both specifications require, then `parts`.
    fn package_of(mimetype: &str, parts: &[(&str, &str)]) -> Vec<u8> {
        use std::io::Write;
        use zip::write::SimpleFileOptions;
        let mut out = zip::ZipWriter::new(Cursor::new(Vec::new()));
        let stored =
            SimpleFileOptions::default().compression_method(zip::CompressionMethod::Stored);
        let wrote = out.start_file("mimetype", stored).is_ok()
            && out.write_all(mimetype.as_bytes()).is_ok()
            && parts.iter().all(|(name, body)| {
                out.start_file(*name, SimpleFileOptions::default()).is_ok()
                    && out.write_all(body.as_bytes()).is_ok()
            });
        if !wrote {
            return Vec::new();
        }
        out.finish().map(Cursor::into_inner).unwrap_or_default()
    }

    fn is(bytes: &[u8], ext: &str) {
        assert_eq!(of(bytes).as_deref(), Some(ext), "{:?}", bytes.get(..16));
    }

    #[test]
    fn signatures_name_binary_formats() {
        is(b"%PDF-1.7\n%\xE2\xE3\xCF\xD3\n1 0 obj\n", "pdf");
        is(b"\x89PNG\r\n\x1a\n\0\0\0\rIHDR", "png");
        is(b"\xFF\xD8\xFF\xE0\0\x10JFIF\0", "jpg");
        is(b"GIF89a\x01\0\x01\0", "gif");
        is(b"{\\rtf1\\ansi\\deff0 hello}", "rtf");
    }

    #[test]
    fn packages_are_told_apart_by_their_parts() {
        is(
            &package(&[
                ("[Content_Types].xml", "<Types/>"),
                ("word/document.xml", "<w:document/>"),
            ]),
            "docx",
        );
        is(
            &package(&[
                ("[Content_Types].xml", "<Types/>"),
                ("xl/workbook.xml", "<workbook/>"),
            ]),
            "xlsx",
        );
        is(
            &package(&[
                ("[Content_Types].xml", "<Types/>"),
                ("ppt/presentation.xml", "<p:presentation/>"),
            ]),
            "pptx",
        );
        is(
            &package_of(
                "application/epub+zip",
                &[("META-INF/container.xml", "<container/>")],
            ),
            "epub",
        );
        is(
            &package_of(
                "application/vnd.oasis.opendocument.text",
                &[("content.xml", "<office:document-content/>")],
            ),
            "odt",
        );
        is(
            &package_of(
                "application/vnd.oasis.opendocument.spreadsheet",
                &[("content.xml", "<office:document-content/>")],
            ),
            "ods",
        );
        // A zip that is no document type quack reads.
        assert_eq!(of(&package(&[("readme.txt", "hello")])), None);
    }

    #[test]
    fn text_is_json_html_captions_or_plain() {
        is(b"  [{\"a\": 1}]", "json");
        is(b"{\"a\": 1}\n{\"a\": 2}\n", "json");
        is(b"<!DOCTYPE html><html><body>Hi</body></html>", "html");
        is(b"WEBVTT\n\n00:00.000 --> 00:01.000\nHi", "vtt");
        is("# Notes\n\nCaf\u{e9} au lait.".as_bytes(), "txt");
    }

    #[test]
    fn other_bytes_are_no_type() {
        // An OLE2 compound file with no Word, Excel, or PowerPoint stream.
        assert_eq!(of(b"\xD0\xCF\x11\xE0\xA1\xB1\x1A\xE1\0\0\0\0"), None);
        assert_eq!(of(&[0, 1, 2, 3, 0xFE, 0xFF, 0, 0]), None);
    }

    /// A head cut in the middle of a character is still text.
    #[test]
    fn a_character_cut_by_the_head_is_still_text() {
        let mut bytes = vec![b'a'; HEAD - 1];
        bytes.extend_from_slice("\u{e9}".as_bytes());
        is(&bytes, "txt");
    }
}
