//! The review queue: candidates stored per induction run, decided one by
//! one or all at once, and applied as a new ontology version.

use super::induction::{Candidate, Decision, ItemKind, Proposal, apply};
use super::store::{self, Acceptance, Revision};
use super::{Class, Ontology, ROOT_CLASS};
use crate::error::{Error, Record, Result};
use crate::prefix::PrefixMatch;
use crate::storage::workspace::WorkspaceDb;
use crate::text::NonBlankText;

/// A stored candidate.
#[derive(Debug, Clone, serde::Serialize)]
pub struct CandidateRow {
    pub id: String,
    pub kind: ItemKind,
    pub proposal: Proposal,
    pub evidence: serde_json::Value,
    pub confidence: f64,
    pub status: CandidateStatus,
    pub proposed_by: String,
    pub decided_by: Option<String>,
    pub decided_at: Option<String>,
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
pub fn store_run(db: &WorkspaceDb, candidates: &[Candidate]) -> Result<String> {
    let run = uuid::Uuid::now_v7().to_string();
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
                uuid::Uuid::now_v7().to_string(),
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
            id: String::from("unparseable"),
            parent: String::from(ROOT_CLASS),
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

/// The candidates in a review queue: the main proposal with classes
/// first, then properties, relations, mappings; the low-support queue
/// most confident first.
///
/// # Errors
///
/// Returns an error if the query fails.
pub fn queue(db: &WorkspaceDb, queue: Queue) -> Result<Vec<CandidateRow>> {
    let order = match queue {
        Queue::Pending => {
            "CASE kind WHEN 'class' THEN 0 WHEN 'property' THEN 1 WHEN 'relation' THEN 2 ELSE 3 END, id"
        }
        Queue::LowSupport => "confidence DESC, id",
    };
    let sql = format!(
        "SELECT {COLUMNS} FROM _quack_ontology_candidates WHERE status = ? ORDER BY {order}"
    );
    let mut stmt = db.connection().prepare(&sql)?;
    let rows = stmt.query_map([queue.status()], row_from)?;
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

/// Accept every pending candidate as proposed (`--auto-accept`).
///
/// # Errors
///
/// Returns an error when there is nothing pending or the result is invalid.
pub fn accept_all(db: &WorkspaceDb, decided_by: Option<&str>) -> Result<Ontology> {
    let ids: Vec<(String, Decision)> = queue(db, Queue::Pending)?
        .into_iter()
        .map(|c| (c.id, Decision::Accept))
        .collect();
    if ids.is_empty() {
        return Err(Error::Ontology(String::from("no pending candidates")));
    }
    accept_as(db, &ids, decided_by, Acceptance::Auto)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ontology::induction::{Candidate, TableEvidenceOptions, propose_from_tables};

    #[expect(clippy::panic, reason = "test failure path")]
    fn fail(msg: &str) -> ! {
        panic!("{msg}")
    }

    #[test]
    fn low_support_candidates_are_kept_aside_but_reviewable() {
        let db = WorkspaceDb::open_in_memory(4).unwrap_or_else(|e| fail(&e.to_string()));
        let thin = Candidate {
            proposal: Proposal::Class(Class {
                id: String::from("rumor"),
                parent: String::from(ROOT_CLASS),
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
        let id = aside.first().map(|c| c.id.clone()).unwrap_or_default();
        let stored =
            accept(&db, &[(id, Decision::Accept)], None).unwrap_or_else(|e| fail(&e.to_string()));
        assert!(stored.class("rumor").is_some());
        assert!(queue(&db, Queue::LowSupport).is_ok_and(|p| p.is_empty()));
    }

    #[test]
    fn same_named_columns_on_different_tables_are_separate_candidates() {
        let db = WorkspaceDb::open_in_memory(4).unwrap_or_else(|e| fail(&e.to_string()));
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
        let db = WorkspaceDb::open_in_memory(4).unwrap_or_else(|e| fail(&e.to_string()));
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
        assert!(!run.is_empty());
        let listed = queue(&db, Queue::Pending).unwrap_or_else(|e| fail(&e.to_string()));
        assert_eq!(listed.len(), 5, "class, three properties, mapping");
        let country = listed
            .iter()
            .find(|c| c.proposal.id() == "country")
            .unwrap_or_else(|| fail("no country"));
        assert_eq!(
            reject(&db, std::slice::from_ref(&country.id), Some("alice")).unwrap_or(0),
            1
        );
        assert!(
            reject(&db, std::slice::from_ref(&country.id), None).is_err(),
            "already decided"
        );
        let prefix = country.id.get(..30).unwrap_or(&country.id);
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
                (c.id, d)
            })
            .collect();
        let stored = accept(&db, &rest, Some("bob")).unwrap_or_else(|e| fail(&e.to_string()));
        assert_eq!(stored.version, 1);
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
}
