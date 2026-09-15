//! `quack ontology`: show, install the default, export and import JSON,
//! list versions, diff, and restore. The ontology lives in the workspace
//! file; a file on disk is only ever a copy.

use std::io::{Read, Write};

use anyhow::{Context, Result};
use clap::Subcommand;
use quack_core::ontology::Ontology;
use quack_core::ontology::store;
use quack_core::storage::workspace::WorkspaceDb;

#[derive(Subcommand)]
pub(crate) enum OntologyAction {
    /// Print the current ontology: classes, relations, properties, mappings
    Show {
        /// Print the JSON interchange form instead
        #[arg(long)]
        json: bool,
    },
    /// Install the built-in general ontology as version 1
    Init,
    /// Write the current ontology as JSON to a file (- for stdout)
    Export { file: String },
    /// Validate a JSON ontology and store it as a new version (- for stdin)
    Import { file: String },
    /// List versions, newest first
    Versions {
        #[arg(long, default_value_t = 20)]
        limit: u32,
    },
    /// What changed between two versions (defaults: previous and current)
    Diff { from: Option<u32>, to: Option<u32> },
    /// Store an earlier version as the newest one
    Restore { version: u32 },
}

pub(crate) fn run(db: &WorkspaceDb, action: OntologyAction) -> Result<()> {
    let stdout = std::io::stdout();
    let mut out = std::io::BufWriter::new(stdout.lock());
    match action {
        OntologyAction::Show { json } => match store::current(db)? {
            None => writeln!(
                out,
                "No ontology yet. Run `quack ontology init` for the built-in one or `quack ontology import FILE`."
            )?,
            Some(ontology) if json => writeln!(out, "{}", ontology.to_json()?)?,
            Some(ontology) => write!(out, "{}", summary(&ontology))?,
        },
        OntologyAction::Init => {
            if store::latest_version(db)? > 0 {
                anyhow::bail!(
                    "an ontology already exists; import a file or restore a version instead"
                );
            }
            let stored = store::save(
                db,
                &Ontology::builtin_default(),
                None,
                Some("built-in default"),
            )?;
            writeln!(
                out,
                "installed the built-in ontology as version {}",
                stored.version
            )?;
        }
        OntologyAction::Export { file } => {
            let ontology = store::current(db)?.context("no ontology to export")?;
            let json = ontology.to_json()?;
            if file == "-" {
                writeln!(out, "{json}")?;
            } else {
                std::fs::write(&file, format!("{json}\n"))
                    .with_context(|| format!("failed to write {file}"))?;
                writeln!(out, "wrote version {} to {file}", ontology.version)?;
            }
        }
        OntologyAction::Import { file } => {
            let text = if file == "-" {
                let mut buf = String::new();
                std::io::stdin().read_to_string(&mut buf)?;
                buf
            } else {
                std::fs::read_to_string(&file).with_context(|| format!("failed to read {file}"))?
            };
            let ontology = Ontology::from_json(&text)?;
            let stored = store::save(db, &ontology, None, Some(&format!("imported from {file}")))?;
            writeln!(out, "ontology is now version {}", stored.version)?;
        }
        OntologyAction::Versions { limit } => {
            let versions = store::versions(db, limit)?;
            if versions.is_empty() {
                writeln!(out, "No versions yet.")?;
            }
            for v in versions {
                writeln!(
                    out,
                    "v{:<4} {}  {:<12} {}",
                    v.version,
                    v.created_at,
                    v.author.as_deref().unwrap_or("-"),
                    v.note.as_deref().unwrap_or("")
                )?;
            }
        }
        OntologyAction::Diff { from, to } => {
            let latest = store::latest_version(db)?;
            let to = to.unwrap_or(latest);
            let from = from.unwrap_or(to.saturating_sub(1));
            let older = store::version(db, from)?
                .with_context(|| format!("version {from} does not exist"))?;
            let newer =
                store::version(db, to)?.with_context(|| format!("version {to} does not exist"))?;
            write!(out, "{}", newer.diff(&older))?;
        }
        OntologyAction::Restore { version } => {
            let stored = store::restore(db, version, None)?;
            writeln!(
                out,
                "restored version {version} as version {}",
                stored.version
            )?;
        }
    }
    out.flush()?;
    Ok(())
}

/// A readable rendering: the class tree, then relations, properties, mappings.
fn summary(ontology: &Ontology) -> String {
    fn children(ontology: &Ontology, parent: &str, depth: usize, lines: &mut Vec<String>) {
        for class in ontology.classes.iter().filter(|c| c.parent == parent) {
            let key = class
                .key
                .as_deref()
                .map_or(String::new(), |k| format!(" [key {k}]"));
            let props = if class.properties.is_empty() {
                String::new()
            } else {
                format!(" {{{}}}", class.properties.join(", "))
            };
            lines.push(format!(
                "{}- {}{key}{props}",
                "  ".repeat(depth.saturating_add(1)),
                class.id
            ));
            children(ontology, &class.id, depth.saturating_add(1), lines);
        }
    }
    let mut lines = vec![
        format!("Ontology version {}", ontology.version),
        String::from("classes:"),
    ];
    children(ontology, quack_core::ontology::ROOT_CLASS, 0, &mut lines);
    lines.push(String::from("relations:"));
    for r in &ontology.relations {
        lines.push(format!("  - {}: {} -> {}", r.id, r.domain, r.range));
    }
    lines.push(String::from("properties:"));
    for p in &ontology.properties {
        let values = if p.values.is_empty() {
            String::new()
        } else {
            format!(" [{}]", p.values.join(", "))
        };
        lines.push(format!("  - {}: {}{values}", p.id, p.kind.as_str()));
    }
    if !ontology.mappings.is_empty() {
        lines.push(String::from("mappings:"));
        for m in &ontology.mappings {
            lines.push(format!(
                "  - {} -> {} (key {}, {} properties, {} relations)",
                m.table,
                m.class,
                m.key,
                m.properties.len(),
                m.relations.len()
            ));
        }
    }
    let mut text = lines.join("\n");
    text.push('\n');
    text
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn summary_nests_subclasses_under_parents() {
        let mut ontology = Ontology::builtin_default();
        ontology.classes.push(quack_core::ontology::Class {
            id: String::from("vendor"),
            parent: String::from("organization"),
            label: None,
            description: None,
            key: None,
            properties: Vec::new(),
        });
        let text = summary(&ontology);
        assert!(
            text.contains("  - organization {industry, country}\n    - vendor\n"),
            "{text}"
        );
        assert!(text.contains("  - works_at: person -> organization"));
        assert!(text.contains("  - date: date"));
    }
}
