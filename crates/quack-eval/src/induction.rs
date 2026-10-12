//! Ontology induction from the fixture's tables: `quack ontology propose
//! --auto-accept`, then what `quack ontology show` lists against the
//! hand-written expectation.

use std::collections::BTreeSet;
use std::io::Write;
use std::path::Path;

use anyhow::{Result, bail};
use serde::{Deserialize, Serialize};

use crate::quack::{Quack, progress};
use crate::retrieval::WORKSPACE;
use crate::share;

/// The root class every ontology has, which the tables do not propose.
const ROOT: &str = "entity";

#[derive(Debug, Deserialize)]
struct Class {
    id: String,
    #[serde(default)]
    properties: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct Relation {
    id: String,
    domain: String,
    range: String,
}

/// The shape `expected_ontology.json` and `quack ontology show --format
/// json` share.
#[derive(Debug, Deserialize)]
struct Shape {
    #[serde(default)]
    classes: Vec<Class>,
    #[serde(default)]
    relations: Vec<Relation>,
}

#[derive(Debug, Serialize)]
pub(crate) struct PrecisionRecall {
    pub(crate) precision: f64,
    pub(crate) recall: f64,
    pub(crate) expected: usize,
    pub(crate) actual: usize,
}

impl PrecisionRecall {
    fn of<T: Ord>(actual: &BTreeSet<T>, expected: &BTreeSet<T>) -> Self {
        let both = actual.intersection(expected).count();
        let ratio = |part: usize, whole: usize| {
            if whole == 0 { 1.0 } else { share(part, whole) }
        };
        Self {
            precision: ratio(both, actual.len()),
            recall: ratio(both, expected.len()),
            expected: expected.len(),
            actual: actual.len(),
        }
    }
}

#[derive(Debug, Serialize)]
pub(crate) struct InductionReport {
    pub(crate) classes: PrecisionRecall,
    pub(crate) properties: PrecisionRecall,
    pub(crate) relations: PrecisionRecall,
}

impl InductionReport {
    pub(crate) fn print(&self, out: &mut impl Write) -> Result<()> {
        writeln!(
            out,
            "\nOntology induction (quack ontology propose --auto-accept vs. hand-written expectation)"
        )?;
        writeln!(
            out,
            "{:<12} {:>10} {:>10} {:>10} {:>10}",
            "item", "precision", "recall", "expected", "actual"
        )?;
        for (name, pr) in [
            ("classes", &self.classes),
            ("properties", &self.properties),
            ("relations", &self.relations),
        ] {
            writeln!(
                out,
                "{name:<12} {:>10.3} {:>10.3} {:>10} {:>10}",
                pr.precision, pr.recall, pr.expected, pr.actual
            )?;
        }
        Ok(())
    }
}

pub(crate) fn evaluate(quack: &Quack, fixture: &Path) -> Result<InductionReport> {
    progress("Proposing an ontology from the tables...");
    let expected: Shape = serde_json::from_str(&std::fs::read_to_string(
        fixture.join("expected_ontology.json"),
    )?)?;
    let proposed = quack.run(&["ontology", "propose", "-w", WORKSPACE, "--auto-accept"])?;
    if !proposed.ok() {
        bail!("quack ontology propose: {}", proposed.failure());
    }
    let actual: Shape = quack.json(&["ontology", "show", "-w", WORKSPACE, "--format", "json"])?;
    let classes = |shape: &Shape| -> BTreeSet<String> {
        shape
            .classes
            .iter()
            .filter(|c| c.id != ROOT)
            .map(|c| c.id.clone())
            .collect()
    };
    let properties = |shape: &Shape| -> BTreeSet<(String, String)> {
        shape
            .classes
            .iter()
            .filter(|c| c.id != ROOT)
            .flat_map(|c| c.properties.iter().map(move |p| (c.id.clone(), p.clone())))
            .collect()
    };
    let relations = |shape: &Shape| -> BTreeSet<(String, String, String)> {
        shape
            .relations
            .iter()
            .map(|r| (r.id.clone(), r.domain.clone(), r.range.clone()))
            .collect()
    };
    Ok(InductionReport {
        classes: PrecisionRecall::of(&classes(&actual), &classes(&expected)),
        properties: PrecisionRecall::of(&properties(&actual), &properties(&expected)),
        relations: PrecisionRecall::of(&relations(&actual), &relations(&expected)),
    })
}
