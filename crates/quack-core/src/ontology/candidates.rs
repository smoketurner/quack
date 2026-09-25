//! The review queue: candidates stored per induction run, decided one by
//! one or all at once, and applied as a new ontology version.

use super::induction::{Candidate, Decision, ItemKind, Proposal, apply};
use super::store::{self, Acceptance, Revision};
use super::{Class, Ontology, ROOT_CLASS};
use crate::error::{Error, Record, Result};
use crate::ids::{CandidateId, ClassId, RunId};
use crate::prefix::PrefixMatch;
use crate::storage::workspace::WorkspaceDb;
use crate::text::NonBlankText;

/// A stored candidate.
#[derive(Debug, Clone, serde::Serialize)]
pub struct CandidateRow {
    pub id: CandidateId,
    pub kind: ItemKind,
    pub proposal: Proposal,
    pub evidence: serde_json::Value,
    pub confidence: f64,
    pub status: CandidateStatus,
    pub proposed_by: String,
    pub decided_by: Option<String>,
    pub decided_at: Option<String>,
}

impl CandidateRow {
    /// One line of evidence for the review listing.
    #[must_use]
    pub fn evidence_line(&self) -> String {
        let e = &self.evidence;
        let get = |k: &str| {
            e.get(k)
                .map(|v| match v {
                    serde_json::Value::String(s) => s.clone(),
                    other => other.to_string(),
                })
                .unwrap_or_default()
        };
        if e.get("source").and_then(|v| v.as_str()) == Some("okf") {
            let examples = e
                .get("examples")
                .and_then(|x| x.as_array())
                .map(|xs| {
                    xs.iter()
                        .filter_map(|x| x.as_str())
                        .collect::<Vec<_>>()
                        .join(", ")
                })
                .unwrap_or_default();
            return format!("{} bundle files, e.g. {examples}", get("files"));
        }
        if e.get("source").and_then(|v| v.as_str()) == Some("documents") {
            let examples = e
                .get("examples")
                .and_then(|x| x.as_array())
                .map(|xs| {
                    xs.iter()
                        .filter_map(|x| {
                            x.get("mention")
                                .or_else(|| x.get("subject"))
                                .or_else(|| x.get("value"))
                                .and_then(|v| v.as_str())
                        })
                        .collect::<Vec<_>>()
                        .join(", ")
                })
                .unwrap_or_default();
            return format!(
                "{} mentions in {} documents{} e.g. {examples}",
                get("occurrences"),
                get("documents"),
                if self.status == CandidateStatus::LowSupport {
                    " (low support)"
                } else {
                    ""
                }
            );
        }
        match self.kind {
            ItemKind::Class => format!(
                "table {} ({} rows, key {})",
                get("table"),
                get("rows"),
                get("key_column")
            ),
            ItemKind::Property => format!(
                "{}.{} {} distinct {} of {} e.g. {}",
                get("table"),
                get("column"),
                get("duckdb_type"),
                get("distinct"),
                get("rows"),
                get("samples")
            ),
            ItemKind::Relation => format!(
                "{}.{} matches {}.{} for {} of values",
                get("table"),
                get("column"),
                get("target_table"),
                get("target_key"),
                get("overlap")
            ),
            ItemKind::Mapping => format!("table {}", get("table")),
        }
    }
}

/// Where a candidate stands.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CandidateStatus {
    /// In the main review queue.
    Pending,
    /// Below the document support threshold: reviewable, kept aside.
    LowSupport,
    /// A later run proposed the same item again.
    Superseded,
    Rejected,
    Accepted,
}

text_enum!(CandidateStatus, "candidate status", {
    Pending => "pending",
    LowSupport => "low_support",
    Superseded => "superseded",
    Rejected => "rejected",
    Accepted => "accepted",
});
text_enum_sql!(CandidateStatus);

impl CandidateStatus {
    /// Still awaiting a decision.
    #[must_use]
    pub fn is_open(self) -> bool {
        match self {
            Self::Pending | Self::LowSupport => true,
            Self::Superseded | Self::Rejected | Self::Accepted => false,
        }
    }
}

/// The two review queues a reader pages through.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Queue {
    /// The main proposal.
    #[default]
    Pending,
    /// Candidates seen in too few documents.
    LowSupport,
}

text_enum!(Queue, "queue", {
    Pending => "pending",
    LowSupport => "low_support",
});

impl Queue {
    /// The status of the candidates in this queue.
    #[must_use]
    pub fn status(self) -> CandidateStatus {
        match self {
            Self::Pending => CandidateStatus::Pending,
            Self::LowSupport => CandidateStatus::LowSupport,
        }
    }
}

/// What a reviewer does with one candidate, from the API or the ontology
/// page; [`Self::decision`] turns it and its target into a [`Decision`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CandidateAction {
    Accept,
    Rename,
    MergeInto,
    Reparent,
    Reject,
}

text_enum!(CandidateAction, "candidate action", {
    Accept => "accept",
    Rename => "rename",
    MergeInto => "merge_into",
    Reparent => "reparent",
    Reject => "reject",
});

impl CandidateAction {
    /// The decision to apply, or `None` to reject. `target` is the new id
    /// for a rename, the existing item for a merge, or the parent for a
    /// reparent; those three need one.
    ///
    /// # Errors
    ///
    /// Returns an error when the action needs a target and none is given.
    pub fn decision(self, target: Option<&str>) -> Result<Option<Decision>> {
        let target = || {
            target
                .and_then(str::non_blank)
                .map(str::to_owned)
                .ok_or_else(|| Error::Ontology(format!("{self} needs a target")))
        };
        Ok(match self {
            Self::Accept => Some(Decision::Accept),
            Self::Rename => Some(Decision::Rename(target()?)),
            Self::MergeInto => Some(Decision::MergeInto(target()?)),
            Self::Reparent => Some(Decision::Reparent(target()?)),
            Self::Reject => None,
        })
    }
}

/// Store a run's candidates as pending, superseding earlier pending ones
/// of the same kind and id. Returns the run id.
///
/// # Errors
///
/// Returns an error if a write fails.
pub fn store_run(db: &WorkspaceDb, candidates: &[Candidate]) -> Result<RunId> {
    let run = RunId::generate();
    let conn = db.connection();
    for candidate in candidates {
        // The id sits at the top for classes and relations, under
        // `property` for properties (scoped by their class, since two
        // tables can share a column name), and is the table for mappings.
        let class = match &candidate.proposal {
            Proposal::Property { class, .. } => Some(class.as_str()),
            _ => None,
        };
        conn.execute(
            "UPDATE _quack_ontology_candidates SET status = ? \
             WHERE status IN (?, ?) AND kind = ? AND coalesce( \
                 json_extract_string(proposal, '$.id'), \
                 json_extract_string(proposal, '$.property.id'), \
                 json_extract_string(proposal, '$.table')) = ? \
             AND (kind != ? OR json_extract_string(proposal, '$.class') = ?)",
            duckdb::params![
                CandidateStatus::Superseded,
                CandidateStatus::Pending,
                CandidateStatus::LowSupport,
                candidate.proposal.kind(),
                candidate.proposal.id(),
                ItemKind::Property,
                class
            ],
        )?;
        let queue = if candidate.low_support {
            Queue::LowSupport
        } else {
            Queue::Pending
        };
        conn.execute(
            "INSERT INTO _quack_ontology_candidates (id, kind, proposal, evidence, confidence, status, proposed_by) \
             VALUES (?, ?, ?, ?, ?, ?, ?)",
            duckdb::params![
                CandidateId::generate(),
                candidate.proposal.kind(),
                serde_json::to_string(&candidate.proposal)?,
                serde_json::to_string(&candidate.evidence)?,
                candidate.confidence,
                queue.status(),
                run
            ],
        )?;
    }
    Ok(run)
}

fn row_from(row: &duckdb::Row<'_>) -> duckdb::Result<CandidateRow> {
    let proposal: String = row.get(2)?;
    let evidence: String = row.get(3)?;
    Ok(CandidateRow {
        id: row.get(0)?,
        kind: row.get(1)?,
        proposal: serde_json::from_str(&proposal).unwrap_or(Proposal::Class(Class {
            id: ClassId::from("unparseable"),
            parent: ClassId::from(String::from(ROOT_CLASS)),
            label: None,
            description: None,
            key: None,
            properties: Vec::new(),
        })),
        evidence: serde_json::from_str(&evidence).unwrap_or(serde_json::Value::Null),
        confidence: row.get(4)?,
        status: row.get(5)?,
        proposed_by: row.get(6)?,
        decided_by: row.get(7)?,
        decided_at: row.get(8)?,
    })
}

const COLUMNS: &str = "id, kind, CAST(proposal AS VARCHAR), CAST(evidence AS VARCHAR), confidence, status, \
     proposed_by, decided_by, CAST(decided_at AS VARCHAR)";

/// The pending-queue order: classes, properties, relations, then the
/// rest, so [`apply`] sees a class before the properties and relations
/// that reference it. Shared by the shared queue and the per-run view so
/// auto-accepting one run's candidates applies them in the same order.
const PENDING_ORDER: &str =
    "CASE kind WHEN 'class' THEN 0 WHEN 'property' THEN 1 WHEN 'relation' THEN 2 ELSE 3 END, id";

/// The candidates in a review queue: the main proposal with classes
/// first, then properties, relations, mappings; the low-support queue
/// most confident first.
///
/// # Errors
///
/// Returns an error if the query fails.
pub fn queue(db: &WorkspaceDb, queue: Queue) -> Result<Vec<CandidateRow>> {
    let order = match queue {
        Queue::Pending => PENDING_ORDER,
        Queue::LowSupport => "confidence DESC, id",
    };
    let sql = format!(
        "SELECT {COLUMNS} FROM _quack_ontology_candidates WHERE status = ? ORDER BY {order}"
    );
    let mut stmt = db.connection().prepare(&sql)?;
    let rows = stmt.query_map([queue.status()], row_from)?;
    Ok(rows.flatten().collect())
}

/// The pending candidates a single run queued: the rows [`store_run`]
/// inserted for `run` that are still awaiting a decision. A per-run
/// auto-accept ([`accept_run`]) reads this instead of the shared [`queue`]
/// so it does not drain other runs' undecided candidates.
fn pending_for_run(db: &WorkspaceDb, run: &RunId) -> Result<Vec<CandidateRow>> {
    let sql = format!(
        "SELECT {COLUMNS} FROM _quack_ontology_candidates WHERE status = ? AND proposed_by = ? ORDER BY {PENDING_ORDER}"
    );
    let mut stmt = db.connection().prepare(&sql)?;
    let rows = stmt.query_map(
        duckdb::params![Queue::Pending.status(), run.as_str()],
        row_from,
    )?;
    Ok(rows.flatten().collect())
}

/// One candidate by id or unique prefix.
///
/// # Errors
///
/// Returns an error when nothing or more than one candidate matches.
pub fn find(db: &WorkspaceDb, prefix: &str) -> Result<CandidateRow> {
    let sql = format!(
        "SELECT {COLUMNS} FROM _quack_ontology_candidates WHERE id = ? OR starts_with(id, ?)"
    );
    let mut stmt = db.connection().prepare(&sql)?;
    let rows: Vec<CandidateRow> = stmt
        .query_map(duckdb::params![prefix, prefix], row_from)?
        .flatten()
        .collect();
    PrefixMatch::of(rows, prefix, |r| r.id.as_str()).one(Record::Candidate, prefix)
}

/// A new ontology version built by auto-accepting candidates, and how
/// many candidates it applied to it. [`accept_run`] scopes the count to
/// one run's pending candidates; [`accept_all`] flushes the whole
/// pending queue. Callers report `accepted` rather than the number of
/// proposals they queued, so the published count matches the version's
/// contents.
#[derive(Debug, Clone)]
pub struct AutoAccepted {
    /// The stored ontology version.
    pub ontology: Ontology,
    /// How many candidates were accepted into it.
    pub accepted: usize,
}

/// Reject candidates: nothing changes in the ontology.
///
/// # Errors
///
/// Returns an error when an id does not match a pending candidate.
pub fn reject(db: &WorkspaceDb, ids: &[String], decided_by: Option<&str>) -> Result<usize> {
    let mut count: usize = 0;
    for id in ids {
        let row = find(db, id)?;
        if !row.status.is_open() {
            return Err(Error::Ontology(format!(
                "candidate {} is already {}",
                row.id, row.status
            )));
        }
        db.connection().execute(
            "UPDATE _quack_ontology_candidates SET status = ?, decided_by = ?, decided_at = now() WHERE id = ?",
            duckdb::params![CandidateStatus::Rejected, decided_by, row.id],
        )?;
        count = count.saturating_add(1);
    }
    Ok(count)
}

/// Apply decisions to pending candidates and save the result as a new
/// ontology version. Returns the stored ontology. Nothing is marked
/// decided unless the save succeeds.
///
/// # Errors
///
/// Returns an error when an id does not match a pending candidate or the
/// resulting ontology is invalid.
pub fn accept(
    db: &WorkspaceDb,
    decisions: &[(String, Decision)],
    decided_by: Option<&str>,
) -> Result<Ontology> {
    accept_as(db, decisions, decided_by, Acceptance::Reviewed)
}

fn accept_as(
    db: &WorkspaceDb,
    decisions: &[(String, Decision)],
    decided_by: Option<&str>,
    acceptance: Acceptance,
) -> Result<Ontology> {
    let mut resolved = Vec::with_capacity(decisions.len());
    for (id, decision) in decisions {
        let row = find(db, id)?;
        if !row.status.is_open() {
            return Err(Error::Ontology(format!(
                "candidate {} is already {}",
                row.id, row.status
            )));
        }
        resolved.push((row.id, row.proposal, decision.clone()));
    }
    let base = store::current(db)?;
    let proposals: Vec<(Proposal, Decision)> = resolved
        .iter()
        .map(|(_, p, d)| (p.clone(), d.clone()))
        .collect();
    let next = apply(base.as_ref(), &proposals)?;
    let verb = match acceptance {
        Acceptance::Reviewed => "accepted",
        Acceptance::Auto => "auto-accepted",
    };
    let note = format!("{verb} {} candidate(s)", resolved.len());
    let stored = store::save(
        db,
        &next,
        Revision {
            author: decided_by,
            note: Some(&note),
            acceptance,
        },
    )?;
    for (id, _, _) in &resolved {
        db.connection().execute(
            "UPDATE _quack_ontology_candidates SET status = ?, decided_by = ?, decided_at = now() WHERE id = ?",
            duckdb::params![CandidateStatus::Accepted, decided_by, id],
        )?;
    }
    Ok(stored)
}

/// Accept every pending candidate as proposed (`--auto-accept` over the
/// whole review queue). This is the queue-level flush: it accepts every
/// `Pending` row regardless of which run produced it, so the version may
/// contain candidates the caller did not propose this run. A per-run
/// auto-accept uses [`accept_run`] instead; this function remains the
/// deliberate "accept all pending" queue action.
///
/// # Errors
///
/// Returns an error when there is nothing pending or the result is invalid.
pub fn accept_all(db: &WorkspaceDb, decided_by: Option<&str>) -> Result<AutoAccepted> {
    let ids: Vec<(String, Decision)> = queue(db, Queue::Pending)?
        .into_iter()
        .map(|c| (c.id.into_string(), Decision::Accept))
        .collect();
    if ids.is_empty() {
        return Err(Error::Ontology(String::from("no pending candidates")));
    }
    let accepted = ids.len();
    let ontology = accept_as(db, &ids, decided_by, Acceptance::Auto)?;
    Ok(AutoAccepted { ontology, accepted })
}

/// Accept the pending candidates one run queued as proposed (`--auto-accept`
/// over a single run). Unlike [`accept_all`], this leaves other runs'
/// undecided candidates pending, so an auto-accepted version contains
/// only the candidates the caller proposed this run and the reported count
/// matches the version's contents.
///
/// # Errors
///
/// Returns an error when the run has nothing pending or the result is invalid.
pub fn accept_run(db: &WorkspaceDb, run: &RunId, decided_by: Option<&str>) -> Result<AutoAccepted> {
    let ids: Vec<(String, Decision)> = pending_for_run(db, run)?
        .into_iter()
        .map(|c| (c.id.into_string(), Decision::Accept))
        .collect();
    if ids.is_empty() {
        return Err(Error::Ontology(String::from(
            "no pending candidates for run",
        )));
    }
    let accepted = ids.len();
    let ontology = accept_as(db, &ids, decided_by, Acceptance::Auto)?;
    Ok(AutoAccepted { ontology, accepted })
}

/// Reject every pending candidate at once: nothing changes in the ontology.
/// The queue-level counterpart to [`accept_all`], reached from the
/// review-queue "reject all pending" action so clearing a backlog is a
/// deliberate, visible choice rather than a side effect of a per-run
/// auto-accept.
///
/// # Errors
///
/// Returns an error if a write fails.
pub fn reject_all(db: &WorkspaceDb, decided_by: Option<&str>) -> Result<usize> {
    let ids: Vec<String> = queue(db, Queue::Pending)?
        .into_iter()
        .map(|c| c.id.into_string())
        .collect();
    if ids.is_empty() {
        return Ok(0);
    }
    reject(db, &ids, decided_by)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embedding::Dimension;
    use crate::ontology::OntologyVersion;
    use crate::ontology::induction::{Candidate, TableEvidenceOptions, propose_from_tables};

    #[expect(clippy::panic, reason = "test failure path")]
    fn fail(msg: &str) -> ! {
        panic!("{msg}")
    }

    #[test]
    fn low_support_candidates_are_kept_aside_but_reviewable() {
        let db =
            WorkspaceDb::open_in_memory(Dimension::new(4)).unwrap_or_else(|e| fail(&e.to_string()));
        let thin = Candidate {
            proposal: Proposal::Class(Class {
                id: ClassId::from("rumor"),
                parent: ClassId::from(String::from(ROOT_CLASS)),
                label: None,
                description: None,
                key: None,
                properties: Vec::new(),
            }),
            evidence: serde_json::json!({ "source": "documents", "documents": 1 }),
            confidence: 0.2,
            low_support: true,
        };
        assert!(store_run(&db, std::slice::from_ref(&thin)).is_ok());
        assert!(queue(&db, Queue::Pending).is_ok_and(|p| p.is_empty()));
        let aside = queue(&db, Queue::LowSupport).unwrap_or_else(|e| fail(&e.to_string()));
        assert_eq!(aside.len(), 1);
        let id = aside.first().map(|c| c.id.to_string()).unwrap_or_default();
        let stored =
            accept(&db, &[(id, Decision::Accept)], None).unwrap_or_else(|e| fail(&e.to_string()));
        assert!(stored.class("rumor").is_some());
        assert!(queue(&db, Queue::LowSupport).is_ok_and(|p| p.is_empty()));
    }

    #[test]
    fn same_named_columns_on_different_tables_are_separate_candidates() {
        let db =
            WorkspaceDb::open_in_memory(Dimension::new(4)).unwrap_or_else(|e| fail(&e.to_string()));
        for sql in [
            "CREATE TABLE vendors (vendor_id INTEGER, name TEXT)",
            "CREATE TABLE sites (site_id INTEGER, name TEXT)",
            "INSERT INTO vendors VALUES (1, 'a')",
            "INSERT INTO sites VALUES (1, 'b')",
        ] {
            assert!(db.execute_statement(sql).is_ok());
        }
        let proposals = propose_from_tables(&db, None, &TableEvidenceOptions::default())
            .unwrap_or_else(|e| fail(&e.to_string()));
        assert!(store_run(&db, &proposals).is_ok());
        let names = queue(&db, Queue::Pending)
            .unwrap_or_else(|e| fail(&e.to_string()))
            .into_iter()
            .filter(|c| c.proposal.id() == "name")
            .count();
        assert_eq!(names, 2);
    }

    #[test]
    fn candidates_are_stored_reviewed_and_applied_as_a_version() {
        let db =
            WorkspaceDb::open_in_memory(Dimension::new(4)).unwrap_or_else(|e| fail(&e.to_string()));
        assert!(
            db.execute_statement(
                "CREATE TABLE vendors (vendor_id INTEGER, name TEXT, country TEXT)"
            )
            .is_ok()
        );
        for i in 0..5_u32 {
            assert!(
                db.execute_statement(&format!(
                    "INSERT INTO vendors VALUES ({i}, 'V{i}', 'C{}')",
                    i % 2
                ))
                .is_ok()
            );
        }
        let proposals = propose_from_tables(&db, None, &TableEvidenceOptions::default())
            .unwrap_or_else(|e| fail(&e.to_string()));
        let run = store_run(&db, &proposals).unwrap_or_else(|e| fail(&e.to_string()));
        assert!(!run.as_str().is_empty());
        let listed = queue(&db, Queue::Pending).unwrap_or_else(|e| fail(&e.to_string()));
        assert_eq!(listed.len(), 5, "class, three properties, mapping");
        let country = listed
            .iter()
            .find(|c| c.proposal.id() == "country")
            .unwrap_or_else(|| fail("no country"));
        assert_eq!(
            reject(&db, &[country.id.to_string()], Some("alice")).unwrap_or(0),
            1
        );
        assert!(
            reject(&db, &[country.id.to_string()], None).is_err(),
            "already decided"
        );
        let prefix = country.id.as_str().get(..30).unwrap_or(country.id.as_str());
        let found = find(&db, prefix).unwrap_or_else(|e| fail(&e.to_string()));
        assert_eq!(found.status, CandidateStatus::Rejected);
        assert_eq!(found.decided_by.as_deref(), Some("alice"));

        let rest: Vec<(String, Decision)> = queue(&db, Queue::Pending)
            .unwrap_or_else(|e| fail(&e.to_string()))
            .into_iter()
            .map(|c| {
                let d = if c.proposal.id() == "vendor" && c.kind == ItemKind::Class {
                    Decision::Rename(String::from("supplier"))
                } else {
                    Decision::Accept
                };
                (c.id.into_string(), d)
            })
            .collect();
        let stored = accept(&db, &rest, Some("bob")).unwrap_or_else(|e| fail(&e.to_string()));
        assert_eq!(stored.version, OntologyVersion::new(1));
        assert!(
            stored
                .class("supplier")
                .is_some_and(|c| c.key.as_deref() == Some("vendor_id"))
        );
        assert!(stored.property("country").is_none(), "rejected");
        assert!(
            stored
                .mappings
                .iter()
                .any(|m| m.class == "supplier" && !m.properties.contains_key("country"))
        );
        assert!(queue(&db, Queue::Pending).is_ok_and(|p| p.is_empty()));
        assert!(accept_all(&db, None).is_err(), "nothing pending");
        assert!(find(&db, "nope").is_err());
    }

    /// `accept_run` scopes auto-accept to the candidates one run queued, so
    /// a later run's `--auto-accept` does not sweep an earlier run's
    /// undecided candidates into the version and the reported count
    /// matches the version's contents — the bug `accept_all` had over the
    /// shared queue before a per-run path existed.
    #[test]
    fn accept_run_scopes_to_this_run_and_leaves_others_pending() {
        let db =
            WorkspaceDb::open_in_memory(Dimension::new(4)).unwrap_or_else(|e| fail(&e.to_string()));
        let doc = Candidate {
            proposal: Proposal::Class(Class {
                id: ClassId::from("organization"),
                parent: ClassId::from(String::from(ROOT_CLASS)),
                label: None,
                description: None,
                key: None,
                properties: Vec::new(),
            }),
            evidence: serde_json::json!({ "source": "documents", "documents": 4 }),
            confidence: 0.6,
            low_support: false,
        };
        let run_a =
            store_run(&db, std::slice::from_ref(&doc)).unwrap_or_else(|e| fail(&e.to_string()));
        let table = Candidate {
            proposal: Proposal::Class(Class {
                id: ClassId::from("customer"),
                parent: ClassId::from(String::from(ROOT_CLASS)),
                label: None,
                description: None,
                key: None,
                properties: Vec::new(),
            }),
            evidence: serde_json::json!({ "source": "tables" }),
            confidence: 0.9,
            low_support: false,
        };
        let run_b =
            store_run(&db, std::slice::from_ref(&table)).unwrap_or_else(|e| fail(&e.to_string()));
        assert_ne!(run_a.as_str(), run_b.as_str(), "two distinct runs");
        let before = queue(&db, Queue::Pending).unwrap_or_else(|e| fail(&e.to_string()));
        assert_eq!(before.len(), 2, "both runs' candidates are pending");

        let accepted = accept_run(&db, &run_b, None).unwrap_or_else(|e| fail(&e.to_string()));
        assert_eq!(
            accepted.accepted, 1,
            "only this run's candidate was accepted"
        );
        assert!(
            accepted.ontology.class("customer").is_some(),
            "this run's table candidate is accepted"
        );
        assert!(
            accepted.ontology.class("organization").is_none(),
            "the prior run's document candidate is not swept in"
        );

        let remaining = queue(&db, Queue::Pending).unwrap_or_else(|e| fail(&e.to_string()));
        assert_eq!(
            remaining.len(),
            1,
            "the prior run's candidate stays pending"
        );
        assert_eq!(
            remaining.first().map_or("", |c| c.proposed_by.as_str()),
            run_a.as_str(),
            "the leftover candidate is run A's"
        );

        // The reported count matches the version's additions exactly: no
        // misreport of the kind `propose_from_tables` had with `accept_all`.
        let in_version = ["customer", "organization"]
            .iter()
            .filter(|id| accepted.ontology.class(id).is_some())
            .count();
        assert_eq!(accepted.accepted, in_version, "count matches the version");
    }

    /// `accept_all` is the deliberate queue-level flush: it accepts every
    /// `Pending` row regardless of which run produced it. Auto-accepting a
    /// single run uses `accept_run`; `accept_all` remains how the "accept
    /// all pending" review-queue action drains a backlog, and now reports
    /// how many it actually accepted.
    #[test]
    fn accept_all_drains_the_whole_pending_queue() {
        let db =
            WorkspaceDb::open_in_memory(Dimension::new(4)).unwrap_or_else(|e| fail(&e.to_string()));
        let doc = Candidate {
            proposal: Proposal::Class(Class {
                id: ClassId::from("organization"),
                parent: ClassId::from(String::from(ROOT_CLASS)),
                label: None,
                description: None,
                key: None,
                properties: Vec::new(),
            }),
            evidence: serde_json::json!({ "source": "documents", "documents": 4 }),
            confidence: 0.6,
            low_support: false,
        };
        let run_a =
            store_run(&db, std::slice::from_ref(&doc)).unwrap_or_else(|e| fail(&e.to_string()));
        let table = Candidate {
            proposal: Proposal::Class(Class {
                id: ClassId::from("customer"),
                parent: ClassId::from(String::from(ROOT_CLASS)),
                label: None,
                description: None,
                key: None,
                properties: Vec::new(),
            }),
            evidence: serde_json::json!({ "source": "tables" }),
            confidence: 0.9,
            low_support: false,
        };
        let run_b =
            store_run(&db, std::slice::from_ref(&table)).unwrap_or_else(|e| fail(&e.to_string()));
        assert_ne!(run_a.as_str(), run_b.as_str(), "two distinct runs");

        let accepted = accept_all(&db, None).unwrap_or_else(|e| fail(&e.to_string()));
        assert_eq!(accepted.accepted, 2, "the whole pending queue is drained");
        assert!(accepted.ontology.class("customer").is_some());
        assert!(accepted.ontology.class("organization").is_some());
        assert!(queue(&db, Queue::Pending).is_ok_and(|p| p.is_empty()));
    }

    /// `reject_all` clears the pending queue without changing the ontology
    /// and leaves low-support candidates (kept aside) where they are.
    #[test]
    fn reject_all_clears_the_pending_queue() {
        let db =
            WorkspaceDb::open_in_memory(Dimension::new(4)).unwrap_or_else(|e| fail(&e.to_string()));
        let pending = Candidate {
            proposal: Proposal::Class(Class {
                id: ClassId::from("customer"),
                parent: ClassId::from(String::from(ROOT_CLASS)),
                label: None,
                description: None,
                key: None,
                properties: Vec::new(),
            }),
            evidence: serde_json::json!({ "source": "tables" }),
            confidence: 0.9,
            low_support: false,
        };
        let low = Candidate {
            proposal: Proposal::Class(Class {
                id: ClassId::from("rumor"),
                parent: ClassId::from(String::from(ROOT_CLASS)),
                label: None,
                description: None,
                key: None,
                properties: Vec::new(),
            }),
            evidence: serde_json::json!({ "source": "documents", "documents": 1 }),
            confidence: 0.2,
            low_support: true,
        };
        store_run(&db, &[pending, low]).unwrap_or_else(|e| fail(&e.to_string()));
        assert_eq!(
            queue(&db, Queue::Pending)
                .unwrap_or_else(|e| fail(&e.to_string()))
                .len(),
            1
        );

        let rejected = reject_all(&db, Some("alice")).unwrap_or_else(|e| fail(&e.to_string()));
        assert_eq!(rejected, 1, "only the pending candidate was rejected");
        assert!(
            queue(&db, Queue::Pending).is_ok_and(|p| p.is_empty()),
            "pending cleared"
        );
        assert_eq!(
            queue(&db, Queue::LowSupport)
                .unwrap_or_else(|e| fail(&e.to_string()))
                .len(),
            1,
            "low-support candidates are left aside"
        );
        assert_eq!(
            reject_all(&db, None).unwrap_or_else(|e| fail(&e.to_string())),
            0,
            "nothing pending rejects zero"
        );
    }

    /// `accept_run` errors when the run has no pending candidates, so an
    /// empty run never silently produces a version. A run whose only
    /// candidate was kept aside as low support has nothing pending, and a
    /// run that stored nothing has nothing pending either.
    #[test]
    fn accept_run_errors_when_the_run_has_nothing_pending() {
        let db =
            WorkspaceDb::open_in_memory(Dimension::new(4)).unwrap_or_else(|e| fail(&e.to_string()));
        let low = Candidate {
            proposal: Proposal::Class(Class {
                id: ClassId::from("rumor"),
                parent: ClassId::from(String::from(ROOT_CLASS)),
                label: None,
                description: None,
                key: None,
                properties: Vec::new(),
            }),
            evidence: serde_json::json!({ "source": "documents", "documents": 1 }),
            confidence: 0.2,
            low_support: true,
        };
        let run =
            store_run(&db, std::slice::from_ref(&low)).unwrap_or_else(|e| fail(&e.to_string()));
        assert!(queue(&db, Queue::Pending).is_ok_and(|p| p.is_empty()));
        assert!(
            accept_run(&db, &run, None).is_err(),
            "a run with only low-support candidates has nothing pending"
        );
        let other = RunId::generate();
        assert!(
            accept_run(&db, &other, None).is_err(),
            "a run that stored nothing has nothing pending"
        );
    }
}
