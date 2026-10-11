use super::*;

#[test]
fn sanitize_strips_extension() {
    assert_eq!(TableName::of_file("data.csv").as_str(), "data");
}

#[test]
fn sanitize_replaces_non_alphanumeric() {
    assert_eq!(
        TableName::of_file("my-data file.csv").as_str(),
        "my_data_file"
    );
}

#[test]
fn sanitize_preserves_underscores() {
    assert_eq!(TableName::of_file("my_data.json").as_str(), "my_data");
}

#[test]
fn sanitize_no_extension() {
    assert_eq!(TableName::of_file("readme").as_str(), "readme");
}

#[test]
fn sanitize_empty_uses_fallback() {
    assert_eq!(TableName::of_file("").as_str(), "imported");
}

#[test]
fn sanitize_dotfile_replaces_leading_dot() {
    assert_eq!(TableName::of_file(".hidden").as_str(), "_hidden");
}

#[test]
fn sanitize_multiple_extensions() {
    assert_eq!(TableName::of_file("data.2024.csv").as_str(), "data_2024");
}

#[test]
fn sanitize_preserves_alphanumeric() {
    assert_eq!(
        TableName::of_file("Sales2024.parquet").as_str(),
        "Sales2024"
    );
}

#[test]
fn given_names_are_trimmed_and_blank_ones_refused() {
    assert_eq!(
        TableName::given(" sales ").ok().map(TableName::into_string),
        Some(String::from("sales"))
    );
    assert!(TableName::given("   ").is_err());
    assert!(TableName::given("").is_err());
}

#[test]
fn sheet_tables_join_the_file_and_the_sheet() {
    assert_eq!(
        TableName::of_file("book.xlsx")
            .with_sheet("Q1 2024")
            .as_str(),
        "book_Q1_2024"
    );
}

#[test]
fn colliding_sheet_names_sanitize_to_one_table_name() {
    // Sanitization is not injective: spaces and hyphens both become `_`,
    // so two distinct sheet names target the same table. `WorkbookLoad`
    // must reject this rather than let `CREATE OR REPLACE` overwrite one.
    let stem = TableName::of_file("Region Sales.xlsx");
    assert_eq!(
        stem.with_sheet("Sales Q1").as_str(),
        stem.with_sheet("Sales-Q1").as_str()
    );
    assert_eq!(
        stem.with_sheet("Sales Q1").as_str(),
        "Region_Sales_Sales_Q1"
    );
    assert_eq!(
        stem.with_sheet("Sheet 1").as_str(),
        stem.with_sheet("Sheet_1").as_str()
    );
}

#[test]
fn reserved_table_names_are_refused() {
    for name in ["graph_suppliers.csv", "GRAPH_x.parquet", "_quack_meta.csv"] {
        assert!(
            TableName::of_file(name)
                .check_unreserved()
                .is_err_and(|e| e.to_string().contains("reserves")),
            "{name}"
        );
    }
    for name in ["graphs.csv", "my_graph_data.csv", "quack.csv"] {
        assert!(
            TableName::of_file(name).check_unreserved().is_ok(),
            "{name}"
        );
    }
}

/// A file streamed through the hash gives the size and digest its bytes
/// give in memory, across buffer boundaries and when it is empty.
#[test]
#[expect(clippy::unwrap_used, reason = "test")]
fn a_streamed_hash_matches_the_bytes() {
    let dir = tempfile::tempdir().unwrap();
    let past_buffer = (0..Measured::BUFFER * 2 + 7)
        .map(|i| u8::try_from(i % 251).unwrap())
        .collect::<Vec<u8>>();
    for bytes in [Vec::new(), b"a,b\n1,2\n".to_vec(), past_buffer] {
        let path = dir.path().join("f");
        std::fs::write(&path, &bytes).unwrap();
        assert_eq!(Measured::of_file(&path).unwrap(), Measured::of(&bytes));
    }
    assert!(Measured::of_file(&dir.path().join("missing")).is_err());
}

/// Piped table data and text still load as the `stdin` table; a document
/// piped in is ingested under the name its bytes give it.
#[test]
fn piped_bytes_are_table_data_or_a_document() {
    use super::Piped;
    use super::zipped::tests::package;
    assert_eq!(Piped::of(b"region,revenue\nWest,362\n"), Piped::Table);
    assert_eq!(Piped::of(b"[{\"region\": \"West\"}]"), Piped::Table);
    assert_eq!(Piped::of(b"just some notes"), Piped::Table);
    let document = |name: &str| Piped::Document {
        name: name.to_owned(),
    };
    assert_eq!(
        Piped::of(b"%PDF-1.7\n%\xE2\xE3\xCF\xD3\n1 0 obj\n"),
        document("stdin.pdf")
    );
    assert_eq!(
        Piped::of(b"<!DOCTYPE html><html><body>Hi</body></html>"),
        document("stdin.html")
    );
    assert_eq!(
        Piped::of(&package(&[
            ("[Content_Types].xml", "<Types/>"),
            ("word/document.xml", "<w:document/>"),
        ])),
        document("stdin.docx")
    );
    assert_eq!(
        Piped::of(b"\x89PNG\r\n\x1a\n\0\0\0\rIHDR"),
        document("stdin.png")
    );
}
