//! YAML: a list of records loads as a table, the way a JSON array of
//! objects does; any other YAML (a configuration, a single object) is read
//! as text, as before.
//!
//! `serde_saphyr` parses under its default budget, which bounds the nodes,
//! depth, and alias expansion a file may cost: an upload of nested aliases
//! (a "billion laughs") fails to parse and is read as text, rather than
//! expanding in memory.

use serde_json::Value;

/// The records `text` holds, as the JSON array a table loads from: a
/// top-level list of mappings, or a mapping whose one key holds such a
/// list (`warehouses: [...]`). `None` for any other YAML, YAML past the
/// parser's budget, or text that is not YAML.
#[must_use]
pub fn records(text: &str) -> Option<Vec<u8>> {
    let document: Value = serde_saphyr::from_str(text).ok()?;
    let list = match document {
        Value::Array(items) => items,
        Value::Object(map) if map.len() == 1 => match map.into_iter().next()? {
            (_, Value::Array(items)) => items,
            _ => return None,
        },
        _ => return None,
    };
    if list.is_empty() || !list.iter().all(Value::is_object) {
        return None;
    }
    serde_json::to_vec(&list).ok()
}
#[cfg(test)]
mod tests {
    use super::*;

    fn rows(text: &str) -> Option<Value> {
        records(text).and_then(|bytes| serde_json::from_slice(&bytes).ok())
    }

    #[test]
    fn a_list_of_records_under_one_key_is_records() {
        let text = "warehouses:\n  - id: WH-OAK\n    reopen: 2026-10-18\n    serves: [West]\n    \
                    bays: 12\n  - id: WH-RNO\n    open: true\n";
        assert_eq!(
            rows(text),
            Some(serde_json::json!([
                {"id": "WH-OAK", "reopen": "2026-10-18", "serves": ["West"], "bays": 12},
                {"id": "WH-RNO", "open": true},
            ]))
        );
    }

    #[test]
    fn a_top_level_list_of_records_is_records() {
        assert_eq!(
            rows("- a: 1\n  b: 2.5\n- a: 2\n  b: ~\n"),
            Some(serde_json::json!([{"a": 1, "b": 2.5}, {"a": 2, "b": null}]))
        );
    }

    /// A configuration, a list of plain values, or not YAML at all is read
    /// as text.
    #[test]
    fn other_yaml_is_not_records() {
        assert_eq!(records("server:\n  port: 8080\nlog: info\n"), None);
        assert_eq!(records("- one\n- two\n"), None);
        assert_eq!(records("items: []\n"), None);
        assert_eq!(records("a: [unclosed\n"), None);
        assert_eq!(records("--- {a: 1}\n--- {a: 2}\n"), None);
    }

    /// Aliases nested to expand exponentially are refused by the parser's
    /// budget, not expanded.
    #[test]
    #[expect(clippy::format_push_string, reason = "a test building its input")]
    fn an_alias_bomb_is_not_expanded() {
        let mut text = String::from("a0: &a0 [x, x, x, x, x, x, x, x, x, x]\n");
        for n in 1..12 {
            let prev = n - 1;
            text.push_str(&format!(
                "a{n}: &a{n} [*a{prev}, *a{prev}, *a{prev}, *a{prev}, *a{prev}, \
                 *a{prev}, *a{prev}, *a{prev}, *a{prev}, *a{prev}]\n"
            ));
        }
        text.push_str("rows:\n  - {bomb: *a11}\n");
        assert_eq!(records(&text), None);
    }
}
