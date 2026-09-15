//! The review queue: candidates stored per induction run, decided one by
//! one or all at once, and applied as a new ontology version.

use super::induction::{Candidate, Decision, Proposal, apply};
use super::{Ontology, store};
use crate::error::{Error, Result};
use crate::storage::workspace::WorkspaceDb;

/// A stored candidate.
#[derive(Debug, Clone, serde::Serialize)]
pub struct CandidateRow {
    pub id: String,
    pub kind: String,
    pub proposal: Proposal,
    pub evidence: serde_json::Value,
    pub confidence: f64,
    pub status: String,
    pub proposed_by: String,
    pub decided_by: Option<String>,
    pub decided_at: Option<String>,
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
            "UPDATE _quack_ontology_candidates SET status = 'superseded' \
             WHERE status = 'pending' AND kind = ? AND coalesce( \
                 json_extract_string(proposal, '$.id'), \
                 json_extract_string(proposal, '$.property.id'), \
                 json_extract_string(proposal, '$.table')) = ? \
             AND (kind != 'property' OR json_extract_string(proposal, '$.class') = ?)",
            duckdb::params![candidate.proposal.kind(), candidate.proposal.id(), class],
        )?;
        conn.execute(
            "INSERT INTO _quack_ontology_candidates (id, kind, proposal, evidence, confidence, status, proposed_by) \
             VALUES (?, ?, ?, ?, ?, 'pending', ?)",
            duckdb::params![
                uuid::Uuid::now_v7().to_string(),
                candidate.proposal.kind(),
                serde_json::to_string(&candidate.proposal)?,
                serde_json::to_string(&candidate.evidence)?,
                candidate.confidence,
                run
            ],
        )?;
    }
    Ok(run)
}

fn row_from(row: &duckdb::Row<'_>) -> duckdb::Result<(CandidateRow, String)> {
    let proposal: String = row.get(2)?;
    let evidence: String = row.get(3)?;
    Ok((
        CandidateRow {
            id: row.get(0)?,
            kind: row.get(1)?,
            proposal: serde_json::from_str(&proposal).unwrap_or(Proposal::Class(super::Class {
                id: String::from("unparseable"),
                parent: String::from(super::ROOT_CLASS),
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
        },
        proposal,
    ))
}

const COLUMNS: &str = "id, kind, CAST(proposal AS VARCHAR), CAST(evidence AS VARCHAR), confidence, status, \
     proposed_by, decided_by, CAST(decided_at AS VARCHAR)";

/// Pending candidates, classes first, then properties, relations, mappings.
///
/// # Errors
///
/// Returns an error if the query fails.
pub fn pending(db: &WorkspaceDb) -> Result<Vec<CandidateRow>> {
    let sql = format!(
        "SELECT {COLUMNS} FROM _quack_ontology_candidates WHERE status = 'pending' \
         ORDER BY CASE kind WHEN 'class' THEN 0 WHEN 'property' THEN 1 WHEN 'relation' THEN 2 ELSE 3 END, id"
    );
    let mut stmt = db.connection().prepare(&sql)?;
    let rows = stmt.query_map([], row_from)?;
    Ok(rows
        .filter_map(std::result::Result::ok)
        .map(|(r, _)| r)
        .collect())
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
        .filter_map(std::result::Result::ok)
        .map(|(r, _)| r)
        .collect();
    match rows.len() {
        1 => rows
            .into_iter()
            .next()
            .ok_or_else(|| Error::Ontology(String::from("candidate vanished"))),
        0 => Err(Error::Ontology(format!("no candidate matches '{prefix}'"))),
        n => Err(Error::Ontology(format!(
            "'{prefix}' matches {n} candidates; use more of the id"
        ))),
    }
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
        if row.status != "pending" {
            return Err(Error::Ontology(format!(
                "candidate {} is already {}",
                row.id, row.status
            )));
        }
        db.connection().execute(
            "UPDATE _quack_ontology_candidates SET status = 'rejected', decided_by = ?, decided_at = now() WHERE id = ?",
            duckdb::params![decided_by, row.id],
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
    let mut resolved = Vec::with_capacity(decisions.len());
    for (id, decision) in decisions {
        let row = find(db, id)?;
        if row.status != "pending" {
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
    let note = format!("accepted {} candidate(s)", resolved.len());
    let stored = store::save(db, &next, decided_by, Some(&note))?;
    for (id, _, _) in &resolved {
        db.connection().execute(
            "UPDATE _quack_ontology_candidates SET status = 'accepted', decided_by = ?, decided_at = now() WHERE id = ?",
            duckdb::params![decided_by, id],
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
    let ids: Vec<(String, Decision)> = pending(db)?
        .into_iter()
        .map(|c| (c.id, Decision::Accept))
        .collect();
    if ids.is_empty() {
        return Err(Error::Ontology(String::from("no pending candidates")));
    }
    accept(db, &ids, decided_by)
}

#[cfg(test)]
mod tests {
    use super::super::induction::{TableEvidenceOptions, propose_from_tables};
    use super::*;

    #[expect(clippy::panic, reason = "test failure path")]
    fn fail(msg: &str) -> ! {
        panic!("{msg}")
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
        let names = pending(&db)
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
        let queue = pending(&db).unwrap_or_else(|e| fail(&e.to_string()));
        assert_eq!(queue.len(), 5, "class, three properties, mapping");
        let country = queue
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
        assert_eq!(found.status, "rejected");
        assert_eq!(found.decided_by.as_deref(), Some("alice"));

        let rest: Vec<(String, Decision)> = pending(&db)
            .unwrap_or_else(|e| fail(&e.to_string()))
            .into_iter()
            .map(|c| {
                let d = if c.proposal.id() == "vendor" && c.kind == "class" {
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
        assert!(pending(&db).is_ok_and(|p| p.is_empty()));
        assert!(accept_all(&db, None).is_err(), "nothing pending");
        assert!(find(&db, "nope").is_err());
    }
}
