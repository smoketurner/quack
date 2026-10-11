//! A file's type from its bytes, for a name whose extension says none: a
//! piped file, a download saved without one. The answer is an extension,
//! so the file is recorded under a name that says its type and every later
//! step reads the type from the name as usual.

use std::io::{Read, Seek, SeekFrom};

/// Bytes read from the start of a file to decide.
const HEAD: usize = 8 * 1024;

/// The extension the bytes `data` show, or `None` when they are no type
/// quack reads. Office and EPUB files are zip archives, told apart by the
/// parts they hold; text that is not JSON, HTML, or captions is plain text.
#[must_use]
pub fn extension(mut data: impl Read + Seek) -> Option<&'static str> {
    let mut head = Vec::with_capacity(HEAD);
    data.by_ref()
        .take(u64::try_from(HEAD).unwrap_or(u64::MAX))
        .read_to_end(&mut head)
        .ok()?;
    if let Some(ext) = binary(&head) {
        return Some(ext);
    }
    if head.starts_with(b"PK\x03\x04") {
        data.seek(SeekFrom::Start(0)).ok()?;
        return archive(data);
    }
    text(&head)
}

/// A format with a signature at its start.
fn binary(head: &[u8]) -> Option<&'static str> {
    const SIGNATURES: &[(&[u8], &str)] = &[
        (b"%PDF-", "pdf"),
        (b"PAR1", "parquet"),
        (b"\x89PNG\r\n\x1a\n", "png"),
        (b"\xFF\xD8\xFF", "jpg"),
        (b"GIF87a", "gif"),
        (b"GIF89a", "gif"),
        (b"{\\rtf", "rtf"),
    ];
    if let Some((_, ext)) = SIGNATURES.iter().find(|(sig, _)| head.starts_with(sig)) {
        return Some(ext);
    }
    (head.starts_with(b"RIFF") && head.get(8..12) == Some(b"WEBP")).then_some("webp")
}

/// An Office, `OpenDocument`, or EPUB package, by the parts it holds.
fn archive(data: impl Read + Seek) -> Option<&'static str> {
    let mut zip = zip::ZipArchive::new(data).ok()?;
    let holds = |name: &str| zip.file_names().any(|n| n == name);
    if holds("word/document.xml") {
        return Some("docx");
    }
    if holds("ppt/presentation.xml") {
        return Some("pptx");
    }
    if holds("xl/workbook.xml") {
        return Some("xlsx");
    }
    let mut mimetype = String::new();
    zip.by_name("mimetype")
        .ok()?
        .take(128)
        .read_to_string(&mut mimetype)
        .ok()?;
    match mimetype.trim() {
        "application/epub+zip" => Some("epub"),
        "application/vnd.oasis.opendocument.text" => Some("odt"),
        "application/vnd.oasis.opendocument.spreadsheet" => Some("ods"),
        _ => None,
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

    use super::super::zipped::tests::package as zip;
    use super::*;

    fn of(bytes: &[u8]) -> Option<&'static str> {
        extension(Cursor::new(bytes))
    }

    #[test]
    fn signatures_name_binary_formats() {
        assert_eq!(of(b"%PDF-1.7\n..."), Some("pdf"));
        assert_eq!(of(b"PAR1\x15\x04"), Some("parquet"));
        assert_eq!(of(b"\x89PNG\r\n\x1a\n\0\0"), Some("png"));
        assert_eq!(of(b"\xFF\xD8\xFF\xE0"), Some("jpg"));
        assert_eq!(of(b"GIF89a"), Some("gif"));
        assert_eq!(of(b"RIFF\0\0\0\0WEBPVP8 "), Some("webp"));
        assert_eq!(of(b"{\\rtf1\\ansi"), Some("rtf"));
    }

    #[test]
    fn packages_are_told_apart_by_their_parts() {
        assert_eq!(of(&zip(&[("word/document.xml", "<w/>")])), Some("docx"));
        assert_eq!(of(&zip(&[("ppt/presentation.xml", "<p/>")])), Some("pptx"));
        assert_eq!(of(&zip(&[("xl/workbook.xml", "<x/>")])), Some("xlsx"));
        assert_eq!(
            of(&zip(&[("mimetype", "application/epub+zip")])),
            Some("epub")
        );
        assert_eq!(
            of(&zip(&[(
                "mimetype",
                "application/vnd.oasis.opendocument.spreadsheet"
            )])),
            Some("ods")
        );
        assert_eq!(of(&zip(&[("readme.txt", "hello")])), None);
    }

    #[test]
    fn text_is_json_html_captions_or_plain() {
        assert_eq!(of(b"  [{\"a\": 1}]"), Some("json"));
        assert_eq!(of(b"\xEF\xBB\xBF{\"a\": 1}\n{\"a\": 2}\n"), Some("json"));
        assert_eq!(of(b"<!DOCTYPE html><html></html>"), Some("html"));
        assert_eq!(of(b"WEBVTT\n\n00:00.000 --> 00:01.000\nHi"), Some("vtt"));
        assert_eq!(of("# Notes\n\nCafé au lait.".as_bytes()), Some("txt"));
    }

    #[test]
    fn other_bytes_are_no_type() {
        // An OLE2 compound file (.doc, .xls, .ppt) cannot be told apart
        // without parsing it.
        assert_eq!(of(b"\xD0\xCF\x11\xE0\xA1\xB1\x1A\xE1\0\0"), None);
        assert_eq!(of(b"\x00\x01\x02binary"), None);
        assert_eq!(of(b""), Some("txt"));
    }

    /// A head cut in the middle of a character is still text.
    #[test]
    fn a_character_cut_by_the_head_is_still_text() {
        let mut bytes = vec![b'a'; HEAD - 1];
        bytes.extend_from_slice("é".as_bytes());
        assert_eq!(of(&bytes), Some("txt"));
    }
}
