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
