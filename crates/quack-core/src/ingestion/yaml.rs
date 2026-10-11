//! YAML: a list of records loads as a table, the way a JSON array of
//! objects does; any other YAML (a configuration, a single object) is read
//! as text, as before.

use serde_json::{Map, Number, Value};
use yaml_rust2::{Yaml, YamlLoader};

/// The records `text` holds, as the JSON array a table loads from: a
/// top-level list of mappings, or a mapping whose one key holds such a
/// list (`warehouses: [...]`). `None` for any other YAML, or text that is
/// not YAML.
#[must_use]
pub fn records(text: &str) -> Option<Vec<u8>> {
    let documents = YamlLoader::load_from_str(text).ok()?;
    let [document] = documents.as_slice() else {
        return None;
    };
    let list = match document {
        Yaml::Array(items) => items,
        Yaml::Hash(map) if map.len() == 1 => match map.values().next()? {
            Yaml::Array(items) => items,
            _ => return None,
        },
        _ => return None,
    };
    if list.is_empty() || !list.iter().all(|item| matches!(item, Yaml::Hash(_))) {
        return None;
    }
    serde_json::to_vec(&list.iter().map(json).collect::<Vec<_>>()).ok()
}

/// `value` as JSON. YAML dates and times stay strings, which the table
/// loader types as it does a JSON file's.
fn json(value: &Yaml) -> Value {
    match value {
        Yaml::Real(text) => text
            .parse::<f64>()
            .ok()
            .and_then(Number::from_f64)
            .map_or_else(|| Value::String(text.clone()), Value::Number),
        Yaml::Integer(n) => Value::from(*n),
        Yaml::String(text) => Value::String(text.clone()),
        Yaml::Boolean(b) => Value::Bool(*b),
        Yaml::Array(items) => Value::Array(items.iter().map(json).collect()),
        Yaml::Hash(map) => Value::Object(
            map.iter()
                .map(|(k, v)| (key(k), json(v)))
                .collect::<Map<_, _>>(),
        ),
        Yaml::Alias(_) | Yaml::Null | Yaml::BadValue => Value::Null,
    }
}

/// A mapping key as a column name.
fn key(value: &Yaml) -> String {
    match value {
        Yaml::String(text) | Yaml::Real(text) => text.clone(),
        Yaml::Integer(n) => n.to_string(),
        Yaml::Boolean(b) => b.to_string(),
        other => json(other).to_string(),
    }
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
}
