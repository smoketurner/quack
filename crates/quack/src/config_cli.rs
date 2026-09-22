//! `quack config`: what this binary makes of `config.toml`.
//!
//! Every section sets `deny_unknown_fields`, so one mistyped key makes
//! every command fail with a parser error and nothing else. This command
//! reads the file outside that gate ([`Inspection`]) and prints both
//! halves of the picture: the settings the binary recognizes, each with
//! the value in force and where it came from, and the keys in the file it
//! does not recognize. It never prints a credential — the file names the
//! environment variables that hold them, and the listing only says which
//! of those are set.

use std::io::Write;

use anyhow::Result;
use quack_core::config::inspect::{EnvVar, FileState, Inspection, Origin, Setting, UnknownKey};
use serde_json::{Value, json};

/// Print the report. Returns whether the configuration is one the binary
/// would start on, which the caller turns into the exit status.
pub(crate) fn run(
    out: &mut impl Write,
    inspection: &Inspection,
    json: bool,
    changed: bool,
) -> Result<bool> {
    if json {
        writeln!(
            out,
            "{}",
            serde_json::to_string_pretty(&as_json(inspection, changed))?
        )?;
    } else {
        write_text(out, inspection, changed)?;
    }
    Ok(inspection.is_usable())
}

fn write_text(out: &mut impl Write, inspection: &Inspection, changed: bool) -> Result<()> {
    let path = inspection.config_path.display();
    match &inspection.file_state {
        FileState::Missing => writeln!(
            out,
            "config file  {path}\n             no file: every value below is built in or from the environment"
        )?,
        FileState::Loaded => writeln!(out, "config file  {path}")?,
        FileState::Rejected(error) => writeln!(
            out,
            "config file  {path}\n             REJECTED, so every command fails until it is fixed:\n{}\n             the values below are the built-in ones, not the file's",
            indented(error.trim_end())
        )?,
    }
    writeln!(
        out,
        "data dir     {}",
        inspection.config.data_dir().display()
    )?;

    let rows: Vec<Row<'_>> = settings(inspection, changed)
        .into_iter()
        .map(Row::of)
        .collect();
    if rows.is_empty() {
        writeln!(
            out,
            "\nnothing in the file or the environment changes a default."
        )?;
    }
    let key_width = rows.iter().map(|r| r.setting.key.len()).max().unwrap_or(0);
    let value_width = rows
        .iter()
        .map(|r| r.setting.display_value().len())
        .max()
        .unwrap_or(0);

    let mut section = None;
    for row in &rows {
        if section != Some(&row.setting.section) {
            writeln!(out, "\n[{}]", row.setting.section)?;
            section = Some(&row.setting.section);
        }
        let origin = row.setting.origin.to_string();
        let line = format!(
            "  {:key_width$}  {:value_width$}  {origin:<16}{}",
            row.setting.key,
            row.setting.display_value(),
            row.note,
        );
        writeln!(out, "{}", line.trim_end())?;
    }

    if !inspection.unknown.is_empty() {
        writeln!(
            out,
            "\nkeys this binary does not recognize ({} — a section refuses any key it does not know):",
            plural(inspection.unknown.len(), "key", "keys")
        )?;
        let width = inspection
            .unknown
            .iter()
            .map(|u| u.path.len())
            .max()
            .unwrap_or(0);
        for unknown in &inspection.unknown {
            match &unknown.suggestion {
                Some(suggestion) => {
                    writeln!(out, "  {:width$}  did you mean {suggestion}?", unknown.path)?;
                }
                None => writeln!(out, "  {}", unknown.path)?,
            }
        }
    }

    writeln!(out, "\nenvironment")?;
    let width = inspection
        .environment
        .iter()
        .map(|v| v.name.len())
        .max()
        .unwrap_or(0);
    for var in &inspection.environment {
        let state = if var.set { "set" } else { "not set" };
        writeln!(out, "  {:width$}  {state:<9}{}", var.name, var.purpose)?;
    }
    Ok(())
}

/// The settings to report: every one, or only those the file or the
/// environment has a say in.
fn settings(inspection: &Inspection, changed: bool) -> Vec<&Setting> {
    if changed {
        inspection.changed().collect()
    } else {
        inspection.settings.iter().collect()
    }
}

/// One printed setting with the note that explains its origin.
struct Row<'a> {
    setting: &'a Setting,
    note: String,
}

impl<'a> Row<'a> {
    /// The note says what the value in force is not: the default it
    /// replaces, or the file value the environment or a rejected file
    /// keeps out.
    fn of(setting: &'a Setting) -> Self {
        let overridden_file = setting
            .file_value
            .as_ref()
            .filter(|_| setting.origin != Origin::File);
        let note = match (overridden_file, setting.is_default()) {
            (Some(file), _) => format!("file says {file}"),
            (None, false) => match &setting.default {
                Some(default) => format!("default {default}"),
                None => String::new(),
            },
            (None, true) => match setting.env {
                Some(var) => format!("override: {var}"),
                None => String::new(),
            },
        };
        Self { setting, note }
    }
}

/// A multi-line message under the `config file` heading, every line in
/// the same column.
fn indented(message: &str) -> String {
    message
        .lines()
        .map(|line| format!("             {line}"))
        .collect::<Vec<_>>()
        .join("\n")
}

fn plural(count: usize, one: &str, many: &str) -> String {
    let word = if count == 1 { one } else { many };
    format!("{count} {word}")
}

fn as_json(inspection: &Inspection, changed: bool) -> Value {
    let (state, error) = match &inspection.file_state {
        FileState::Missing => ("missing", None),
        FileState::Loaded => ("loaded", None),
        FileState::Rejected(error) => ("rejected", Some(error.clone())),
    };
    json!({
        "config_file": {
            "path": inspection.config_path.display().to_string(),
            "state": state,
            "error": error,
        },
        "data_dir": inspection.config.data_dir().display().to_string(),
        "settings": settings(inspection, changed).into_iter().map(setting_json).collect::<Vec<_>>(),
        "unrecognized": inspection.unknown.iter().map(unknown_json).collect::<Vec<_>>(),
        "environment": inspection.environment.iter().map(env_json).collect::<Vec<_>>(),
    })
}

/// Values are rendered as TOML, the form the config file writes them in,
/// so a string keeps its quotes.
fn setting_json(setting: &Setting) -> Value {
    json!({
        "section": setting.section,
        "key": setting.key,
        "value": setting.value,
        "default": setting.default,
        "origin": setting.origin.to_string(),
        "file_value": setting.file_value,
        "env": setting.env,
    })
}

fn unknown_json(unknown: &UnknownKey) -> Value {
    json!({ "path": unknown.path, "suggestion": unknown.suggestion })
}

fn env_json(var: &EnvVar) -> Value {
    json!({ "name": var.name, "set": var.set, "purpose": var.purpose })
}

#[cfg(test)]
#[expect(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    clippy::panic,
    reason = "tests assert on values they have just built"
)]
mod tests {
    use std::path::PathBuf;

    use super::*;

    const SAMPLE: &str = "[general]\nchat_model = \"ollama/llama3.1:8b\"\n\
                          [providers.ollama]\ntype = \"ollama\"\n[retrieval]\ntop_k = 3\n";

    fn report(contents: Option<&str>, json: bool, changed: bool) -> (String, bool) {
        let inspection = Inspection::of(PathBuf::from("/tmp/config.toml"), contents);
        let mut out = Vec::new();
        let usable = run(&mut out, &inspection, json, changed).unwrap();
        (String::from_utf8(out).unwrap(), usable)
    }

    /// The line a setting is printed on, whatever the column widths.
    fn line_of<'a>(text: &'a str, key: &str) -> &'a str {
        text.lines()
            .find(|line| line.split_whitespace().next() == Some(key))
            .unwrap_or_else(|| panic!("no line for {key} in\n{text}"))
    }

    #[test]
    fn text_report_lists_every_setting_with_its_origin() {
        let (text, usable) = report(Some(SAMPLE), false, false);
        assert!(usable);
        assert!(text.contains("[retrieval]"), "{text}");

        let top_k = line_of(&text, "top_k");
        assert!(top_k.contains(" 3 "), "{top_k}");
        assert!(top_k.contains("file"), "{top_k}");
        assert!(top_k.contains("default 8"), "{top_k}");

        // Untouched settings are listed too, with their built-in values.
        let rrf_k = line_of(&text, "rrf_k");
        assert!(rrf_k.contains("60"), "{rrf_k}");
        assert!(rrf_k.contains("default"), "{rrf_k}");

        assert!(text.contains("[providers.ollama]"), "{text}");
        assert!(line_of(&text, "api_key_env").contains("(unset)"), "{text}");
        assert!(text.contains("environment"), "{text}");
        assert!(line_of(&text, "QUACK_MODEL").contains("not set"), "{text}");
    }

    #[test]
    fn changed_report_leaves_out_untouched_defaults() {
        let (text, _) = report(Some(SAMPLE), false, true);
        assert!(text.contains("top_k"), "{text}");
        assert!(!text.contains("rrf_k"), "{text}");
    }

    #[test]
    fn an_unknown_key_is_named_with_what_it_resembles() {
        let (text, usable) = report(Some("[retrieval]\ntopk = 3\n"), false, false);
        assert!(!usable, "a key no section knows is refused: {text}");
        assert!(text.contains("REJECTED"), "{text}");
        assert!(text.contains("retrieval.topk"), "{text}");
        assert!(text.contains("did you mean top_k?"), "{text}");
    }

    #[test]
    fn a_rejected_file_shows_what_it_says_beside_the_value_in_force() {
        // Recognized keys, refused for a rule the parser cannot see: the
        // listing must not claim the file's values are the ones running.
        let (text, usable) = report(
            Some("[general]\nchat_model = \"missing/m\"\n"),
            false,
            false,
        );
        assert!(!usable);
        assert!(text.contains("REJECTED"), "{text}");
        let chat_model = line_of(&text, "chat_model");
        assert!(chat_model.contains("(unset)"), "{chat_model}");
        assert!(
            chat_model.contains("file says \"missing/m\""),
            "{chat_model}"
        );
    }

    #[test]
    fn a_missing_file_says_so_and_still_lists_the_settings() {
        let (text, usable) = report(None, false, false);
        assert!(usable);
        assert!(text.contains("no file"), "{text}");
        assert!(text.contains("top_k"), "{text}");
    }

    #[test]
    fn json_report_honors_changed() {
        let (text, _) = report(Some(SAMPLE), true, true);
        let value: Value = serde_json::from_str(&text).unwrap();
        let keys: Vec<&str> = value["settings"]
            .as_array()
            .unwrap()
            .iter()
            .map(|s| s["key"].as_str().unwrap())
            .collect();
        assert!(keys.contains(&"top_k"), "{keys:?}");
        assert!(!keys.contains(&"rrf_k"), "{keys:?}");
    }

    #[test]
    fn json_report_carries_the_same_facts() {
        let (text, _) = report(Some(SAMPLE), true, false);
        let value: Value = serde_json::from_str(&text).unwrap();
        assert_eq!(value["config_file"]["state"], "loaded");
        let settings = value["settings"].as_array().unwrap();
        let top_k = settings
            .iter()
            .find(|s| s["section"] == "retrieval" && s["key"] == "top_k")
            .unwrap();
        assert_eq!(top_k["value"], "3");
        assert_eq!(top_k["default"], "8");
        assert_eq!(top_k["origin"], "file");
        assert!(value["unrecognized"].as_array().unwrap().is_empty());
        assert!(!value["environment"].as_array().unwrap().is_empty());
    }
}
