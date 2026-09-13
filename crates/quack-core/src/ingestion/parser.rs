use std::path::Path;

use crate::error::{Error, Result};

/// Recognized file types for ingestion.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FileType {
    Csv,
    Parquet,
    Json,
    Pdf,
    Text,
    Markdown,
    Unknown,
}

impl FileType {
    #[must_use]
    pub fn mime_type(&self) -> &'static str {
        match self {
            Self::Csv => "text/csv",
            Self::Parquet => "application/vnd.apache.parquet",
            Self::Json => "application/json",
            Self::Pdf => "application/pdf",
            Self::Text => "text/plain",
            Self::Markdown => "text/markdown",
            Self::Unknown => "application/octet-stream",
        }
    }

    #[must_use]
    pub fn is_structured(&self) -> bool {
        matches!(self, Self::Csv | Self::Parquet | Self::Json)
    }
}

impl std::fmt::Display for FileType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let label = match self {
            Self::Csv => "CSV",
            Self::Parquet => "Parquet",
            Self::Json => "JSON",
            Self::Pdf => "PDF",
            Self::Text => "Text",
            Self::Markdown => "Markdown",
            Self::Unknown => "Unknown",
        };
        f.write_str(label)
    }
}

/// Detect file type from the filename extension.
#[must_use]
pub fn detect_file_type(filename: &str) -> FileType {
    let ext = Path::new(filename)
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();

    match ext.as_str() {
        "csv" => FileType::Csv,
        "parquet" | "pq" => FileType::Parquet,
        "json" | "jsonl" | "ndjson" => FileType::Json,
        "pdf" => FileType::Pdf,
        "md" | "markdown" => FileType::Markdown,
        "txt" | "text" | "log" => FileType::Text,
        _ => FileType::Unknown,
    }
}

/// Extract text content from an unstructured file.
///
/// # Errors
///
/// Returns an error if the file cannot be parsed.
pub fn extract_text(file_type: &FileType, data: &[u8]) -> Result<String> {
    match file_type {
        FileType::Pdf => extract_pdf_text(data),
        FileType::Text | FileType::Markdown => String::from_utf8(data.to_vec())
            .map_err(|e| Error::Ingestion(format!("invalid UTF-8: {e}"))),
        other => Err(Error::Ingestion(format!(
            "cannot extract text from {other} files"
        ))),
    }
}

fn extract_pdf_text(data: &[u8]) -> Result<String> {
    pdf_extract::extract_text_from_mem(data)
        .map_err(|e| Error::Ingestion(format!("PDF extraction failed: {e}")))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_csv() {
        assert_eq!(detect_file_type("sales.csv"), FileType::Csv);
        assert_eq!(detect_file_type("DATA.CSV"), FileType::Csv);
    }

    #[test]
    fn detects_parquet() {
        assert_eq!(detect_file_type("data.parquet"), FileType::Parquet);
        assert_eq!(detect_file_type("data.pq"), FileType::Parquet);
    }

    #[test]
    fn detects_json() {
        assert_eq!(detect_file_type("config.json"), FileType::Json);
        assert_eq!(detect_file_type("events.jsonl"), FileType::Json);
        assert_eq!(detect_file_type("stream.ndjson"), FileType::Json);
    }

    #[test]
    fn detects_pdf() {
        assert_eq!(detect_file_type("report.pdf"), FileType::Pdf);
    }

    #[test]
    fn detects_text() {
        assert_eq!(detect_file_type("notes.txt"), FileType::Text);
        assert_eq!(detect_file_type("readme.md"), FileType::Markdown);
    }

    #[test]
    fn unknown_extension() {
        assert_eq!(detect_file_type("image.png"), FileType::Unknown);
        assert_eq!(detect_file_type("noext"), FileType::Unknown);
    }

    #[test]
    fn extracts_plain_text() {
        let data = b"Hello, world!";
        let text = extract_text(&FileType::Text, data);
        assert!(text.is_ok());
        assert_eq!(text.ok(), Some(String::from("Hello, world!")));
    }

    #[test]
    fn rejects_structured_extraction() {
        let result = extract_text(&FileType::Csv, b"a,b,c");
        assert!(result.is_err());
    }
}
