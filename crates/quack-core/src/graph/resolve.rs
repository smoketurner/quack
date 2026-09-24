//! Entity resolution beyond the exact merge on `(normalized_label, class)`:
//! nodes of one class whose label embeddings are close and whose labels
//! share a token are proposed as merges; the closest merge on their own,
//! the rest wait in `_quack_graph_merges` for review.

use std::collections::{BTreeMap, BTreeSet};
use std::time::Instant;

use rig::embeddings::EmbeddingModel;

use super::{GraphOptions, Node, store};
use crate::embedding::{Embedder, Input};
use crate::error::{Error, Record, Result};
use crate::prefix::PrefixMatch;
use crate::progress::{ChunkDone, RunControl};
use crate::storage::workspace::{WorkspaceDb, tokenize};
use crate::storage::writer::Writer;

/// A proposed merge: `drop` folds into `keep`.
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct MergeProposal {
    pub id: String,
    pub keep: Node,
    pub drop: Node,
    pub distance: f64,
    pub status: MergeStatus,
}

/// Where a merge proposal stands.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum MergeStatus {
    /// Waiting for review.
    Pending,
    /// Merged.
    Accepted,
    /// Kept apart; not proposed again.
    Rejected,
    /// One of its nodes merged into something else first.
    Superseded,
}

text_enum!(MergeStatus, "merge status", {
    Pending => "pending",
    Accepted => "accepted",
    Rejected => "rejected",
    Superseded => "superseded",
});
text_enum_sql!(MergeStatus);

/// A reviewer's answer to a merge proposal, from the CLI, the API, or the
/// graph page.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum MergeDecision {
    Accept,
    Reject,
}

text_enum!(MergeDecision, "merge decision", {
    Accept => "accept",
    Reject => "reject",
});

impl MergeDecision {
    /// The status a proposal takes when this is the answer.
    #[must_use]
    pub fn status(self) -> MergeStatus {
        match self {
            Self::Accept => MergeStatus::Accepted,
            Self::Reject => MergeStatus::Rejected,
        }
    }
}

/// What a resolution pass did.
#[derive(Debug, Clone, Default, PartialEq, Eq, serde::Serialize)]
pub struct ResolutionSummary {
    pub embedded: u32,
    pub auto_merged: u32,
    pub proposed: u32,
}

/// Embed nodes that lack an embedding, then compare each node with its
/// nearest neighbours of the same class. Without an embedding model only
/// the exact merge (already done at insert) applies.
///
/// # Errors
///
/// Returns an error when embedding or a write fails.
pub async fn resolve<M: EmbeddingModel>(
    db: &Writer,
    embedder: Option<&Embedder<M>>,
    options: &GraphOptions,
) -> Result<ResolutionSummary> {
    let mut summary = ResolutionSummary::default();
    let Some(embedder) = embedder else {
        return Ok(summary);
    };
    summary.embedded = embed_nodes(db, embedder, RunControl::unobserved()).await?;
    let options = *options;
    let (auto_merged, proposed) = db
        .run(move |db| {
            log_memory(db, "before merge proposals");
            propose_merges(db, &options)
        })
        .await?;
    summary.auto_merged = auto_merged;
    summary.proposed = proposed;
    Ok(summary)
}

/// Label vectors embedded per model call.
const NODE_BATCH: u32 = 64;

/// Give every node whose label vector is missing or stale one made under
/// the current profile, `NODE_BATCH` at a time; returns how many. Nothing
/// is embedded while the workspace's vectors are another width (`refresh`
/// retypes them first). Progress is reported after each batch; a cancel
/// stops between batches.
///
/// # Errors
///
/// Returns an error when embedding or a write fails, or
/// [`Error::Cancelled`].
pub async fn embed_nodes<M: EmbeddingModel>(
    db: &Writer,
    embedder: &Embedder<M>,
    control: RunControl<'_>,
) -> Result<u32> {
    let dimension = embedder.profile().dimension;
    let total = db
        .run(move |db| {
            if db.embedding_dimension() == dimension {
                store::count_nodes_needing_embedding(db).map(Some)
            } else {
                Ok(None)
            }
        })
        .await?;
    let Some(total) = total else {
        tracing::warn!(
            "node label vectors not embedded: run `quack embeddings refresh` after the width change"
        );
        return Ok(0);
    };
    let started = Instant::now();
    let mut embedded: u32 = 0;
    loop {
        control.check()?;
        let batch_started = Instant::now();
        let pending = db
            .run(|db| store::nodes_needing_embedding(db, NODE_BATCH))
            .await?;
        if pending.is_empty() {
            return Ok(embedded);
        }
        let inputs: Vec<Input> = pending.iter().map(Node::embedding_input).collect();
        let vectors = embedder.embed(&inputs).await?;
        let count = u32::try_from(pending.len()).unwrap_or(u32::MAX);
        db.run(move |db| {
            db.write_transaction(|db| {
                for (node, vector) in pending.iter().zip(&vectors) {
                    db.set_node_embedding(&node.id, vector)?;
                }
                Ok(())
            })
        })
        .await?;
        embedded = embedded.saturating_add(count);
        (control.progress)(ChunkDone {
            done: embedded,
            total: total.max(embedded),
            failed: 0,
            took: batch_started.elapsed(),
            elapsed: started.elapsed(),
        });
    }
}

/// Nearest same-class neighbours considered per node; bounds the work
/// and the proposals a big class can generate.
const NEIGHBOURS_PER_NODE: u32 = 5;

/// One node of a candidate pair.
struct Side {
    id: String,
    label: String,
    /// Whether the node comes from a keyed table row.
    keyed: bool,
}

/// A pair worth looking at: close embeddings, same class.
struct Candidate {
    a: Side,
    b: Side,
    distance: f64,
}

impl Candidate {
    /// Whether both sides came from the model, the only pairs that may
    /// merge without review.
    fn extracted_only(&self) -> bool {
        !self.a.keyed && !self.b.keyed
    }

    /// The node to keep and the one to fold into it: the keyed node, else
    /// the one with more provenance, else the earlier id.
    fn keep_and_drop(self, db: &WorkspaceDb) -> Result<(String, String)> {
        let prefer_b = match (self.a.keyed, self.b.keyed) {
            (false, true) => true,
            (true, false) => false,
            (true, true) | (false, false) => {
                provenance_count(db, &self.b.id)? > provenance_count(db, &self.a.id)?
            }
        };
        Ok(if prefer_b {
            (self.b.id, self.a.id)
        } else {
            (self.a.id, self.b.id)
        })
    }
}

/// Compare every embedded node with its nearest same-class neighbours;
/// returns (auto-merged, proposed).
///
/// Two nodes that both come from keyed table rows are distinct by
/// construction (different keys), so they are never candidates, however
/// alike their labels (issue #41: WEST VIRGINIA is not VIRGINIA). A pair
/// with one keyed side is only ever proposed; auto-merge is reserved for
/// two model-extracted nodes.
fn propose_merges(db: &WorkspaceDb, options: &GraphOptions) -> Result<(u32, u32)> {
    // The keyed flag is computed per node before the join: a correlated
    // EXISTS per pair row, or a window over the pairs, made DuckDB run
    // out of its 256 MiB on 3,667 nodes, while this streams in seconds.
    // The per-node cap is applied below, in candidate order.
    let mut stmt = db.connection().prepare(
        "WITH n AS ( \
           SELECT id, label, class_id, embedding, \
                  EXISTS (SELECT 1 FROM _quack_provenance p \
                          WHERE p.subject_id = _quack_graph_nodes.id AND p.table_name <> '') AS keyed \
           FROM _quack_graph_nodes \
           WHERE embedding IS NOT NULL AND embedding_profile IS NOT DISTINCT FROM ?) \
         SELECT a.id, a.label, b.id, b.label, \
                array_cosine_distance(a.embedding, b.embedding) AS d, a.keyed, b.keyed \
         FROM n a JOIN n b ON a.class_id = b.class_id AND a.id < b.id \
         WHERE NOT (a.keyed AND b.keyed) \
           AND array_cosine_distance(a.embedding, b.embedding) <= ? \
         ORDER BY d, a.id, b.id",
    )?;
    let mut rows = stmt.query(duckdb::params![
        db.embedding_fingerprint(),
        options.merge_threshold
    ])?;
    let mut candidates: Vec<Candidate> = Vec::new();
    let mut seen: BTreeMap<String, u32> = BTreeMap::new();
    while let Some(row) = rows.next()? {
        let a_id: String = row.get(0)?;
        let b_id: String = row.get(2)?;
        // Each node keeps its NEIGHBOURS_PER_NODE closest candidates.
        let a_seen = seen.get(&a_id).copied().unwrap_or(0);
        let b_seen = seen.get(&b_id).copied().unwrap_or(0);
        if a_seen >= NEIGHBOURS_PER_NODE || b_seen >= NEIGHBOURS_PER_NODE {
            continue;
        }
        seen.insert(a_id.clone(), a_seen.saturating_add(1));
        seen.insert(b_id.clone(), b_seen.saturating_add(1));
        candidates.push(Candidate {
            a: Side {
                id: a_id,
                label: row.get(1)?,
                keyed: row.get(5)?,
            },
            b: Side {
                id: b_id,
                label: row.get(3)?,
                keyed: row.get(6)?,
            },
            distance: row.get(4)?,
        });
    }
    drop(rows);
    drop(stmt);
    let mut auto = 0u32;
    let mut proposed = 0u32;
    let mut gone: BTreeSet<String> = BTreeSet::new();
    for candidate in candidates {
        if gone.contains(&candidate.a.id)
            || gone.contains(&candidate.b.id)
            || !share_token(&candidate.a.label, &candidate.b.label)
        {
            continue;
        }
        let distance = candidate.distance;
        let auto_merge = candidate.extracted_only() && distance <= options.auto_merge_threshold;
        let (keep, drop) = candidate.keep_and_drop(db)?;
        if auto_merge {
            merge_nodes(db, &keep, &drop)?;
            gone.insert(drop);
            auto = auto.saturating_add(1);
            continue;
        }
        let already: i64 = db.connection().query_row(
            "SELECT count(*) FROM _quack_graph_merges WHERE keep_node_id = ? AND drop_node_id = ?",
            duckdb::params![keep, drop],
            |r| r.get(0),
        )?;
        if already > 0 {
            continue;
        }
        db.connection().execute(
            "INSERT INTO _quack_graph_merges (id, keep_node_id, drop_node_id, distance) VALUES (?, ?, ?, ?)",
            duckdb::params![uuid::Uuid::now_v7().to_string(), keep, drop, distance],
        )?;
        proposed = proposed.saturating_add(1);
    }
    Ok((auto, proposed))
}

/// `DuckDB`'s own account of its memory, at debug level, for the moments
/// the pass gets near `[analysis].memory_limit_mb`.
fn log_memory(db: &WorkspaceDb, moment: &str) {
    if !tracing::enabled!(tracing::Level::DEBUG) {
        return;
    }
    let Ok(mut stmt) = db
        .connection()
        .prepare("SELECT tag, memory_usage_bytes, temporary_storage_bytes FROM duckdb_memory() WHERE memory_usage_bytes > 0 ORDER BY memory_usage_bytes DESC")
    else {
        return;
    };
    let Ok(mut rows) = stmt.query([]) else {
        return;
    };
    while let Ok(Some(row)) = rows.next() {
        let tag: String = row.get(0).unwrap_or_default();
        let used: i64 = row.get(1).unwrap_or_default();
        let temp: i64 = row.get(2).unwrap_or_default();
        tracing::debug!(moment, tag, used, temp, "duckdb memory");
    }
}

fn provenance_count(db: &WorkspaceDb, id: &str) -> Result<i64> {
    Ok(db.connection().query_row(
        "SELECT count(*) FROM _quack_provenance WHERE subject_id = ?",
        duckdb::params![id],
        |r| r.get(0),
    )?)
}

/// Whether two labels share a word of at least three characters.
#[must_use]
pub fn share_token(a: &str, b: &str) -> bool {
    let tokens = |s: &str| -> BTreeSet<String> {
        tokenize(s)
            .into_iter()
            .filter(|t| t.chars().count() >= 3)
            .collect()
    };
    !tokens(a).is_disjoint(&tokens(b))
}

/// Fold `drop` into `keep`: edges are repointed (duplicates removed),
/// provenance moves over, the dropped label joins `properties.aliases`.
/// The steps run in one transaction, so a failure part way leaves both
/// nodes as they were.
///
/// # Errors
///
/// Returns an error when either node is missing or a write fails.
pub fn merge_nodes(db: &WorkspaceDb, keep: &str, drop: &str) -> Result<()> {
    db.under_timeout(|db| {
        let tx = db.connection().unchecked_transaction()?;
        merge_nodes_in(db, keep, drop)?;
        tx.commit()?;
        Ok(())
    })
}

fn merge_nodes_in(db: &WorkspaceDb, keep: &str, drop: &str) -> Result<()> {
    let keep_node =
        store::node(db, keep)?.ok_or_else(|| Error::Analysis(format!("no node {keep}")))?;
    let drop_node =
        store::node(db, drop)?.ok_or_else(|| Error::Analysis(format!("no node {drop}")))?;
    let conn = db.connection();
    let mut properties = keep_node.properties;
    let mut aliases = properties.aliases();
    aliases.extend(drop_node.properties.aliases());
    if !aliases.contains(&drop_node.label) {
        aliases.push(drop_node.label.clone());
    }
    properties.fill_from(&drop_node.properties);
    properties.set_aliases(&aliases);
    conn.execute(
        "UPDATE _quack_graph_nodes SET properties = ?, provisional = provisional AND ? WHERE id = ?",
        duckdb::params![properties.to_json(), drop_node.provisional, keep],
    )?;
    // Repoint edges, dropping any that would duplicate an existing triple
    // or become a self-loop.
    conn.execute(
        "DELETE FROM _quack_provenance WHERE subject_id IN ( \
            SELECT e.id FROM _quack_graph_edges e WHERE (e.source_node_id = ? OR e.target_node_id = ?) \
            AND EXISTS (SELECT 1 FROM _quack_graph_edges k WHERE k.id <> e.id AND k.relation_id = e.relation_id \
                AND k.source_node_id = CASE WHEN e.source_node_id = ? THEN ? ELSE e.source_node_id END \
                AND k.target_node_id = CASE WHEN e.target_node_id = ? THEN ? ELSE e.target_node_id END))",
        duckdb::params![drop, drop, drop, keep, drop, keep],
    )?;
    conn.execute(
        "DELETE FROM _quack_graph_edges e WHERE (e.source_node_id = ? OR e.target_node_id = ?) \
            AND EXISTS (SELECT 1 FROM _quack_graph_edges k WHERE k.id <> e.id AND k.relation_id = e.relation_id \
                AND k.source_node_id = CASE WHEN e.source_node_id = ? THEN ? ELSE e.source_node_id END \
                AND k.target_node_id = CASE WHEN e.target_node_id = ? THEN ? ELSE e.target_node_id END)",
        duckdb::params![drop, drop, drop, keep, drop, keep],
    )?;
    conn.execute(
        "UPDATE _quack_graph_edges SET source_node_id = ? WHERE source_node_id = ?",
        duckdb::params![keep, drop],
    )?;
    conn.execute(
        "UPDATE _quack_graph_edges SET target_node_id = ? WHERE target_node_id = ?",
        duckdb::params![keep, drop],
    )?;
    conn.execute(
        "DELETE FROM _quack_provenance WHERE subject_id IN (SELECT id FROM _quack_graph_edges WHERE source_node_id = target_node_id)",
        [],
    )?;
    conn.execute(
        "DELETE FROM _quack_graph_edges WHERE source_node_id = target_node_id",
        [],
    )?;
    conn.execute(
        "INSERT OR IGNORE INTO _quack_provenance (subject_id, document_id, chunk_id, table_name, row_key, confidence) \
         SELECT ?, document_id, chunk_id, table_name, row_key, confidence FROM _quack_provenance WHERE subject_id = ?",
        duckdb::params![keep, drop],
    )?;
    conn.execute(
        "DELETE FROM _quack_provenance WHERE subject_id = ?",
        duckdb::params![drop],
    )?;
    conn.execute(
        "UPDATE _quack_graph_merges SET status = ? WHERE status = ? AND (keep_node_id = ? OR drop_node_id = ?)",
        duckdb::params![MergeStatus::Superseded, MergeStatus::Pending, drop, drop],
    )?;
    conn.execute(
        "DELETE FROM _quack_graph_nodes WHERE id = ?",
        duckdb::params![drop],
    )?;
    Ok(())
}

/// Pending merge proposals, closest first.
///
/// # Errors
///
/// Returns an error if the query fails.
pub fn pending(db: &WorkspaceDb) -> Result<Vec<MergeProposal>> {
    let mut stmt = db.connection().prepare(
        "SELECT id, keep_node_id, drop_node_id, distance, status FROM _quack_graph_merges \
         WHERE status = ? ORDER BY distance, id",
    )?;
    let mut rows = stmt.query([MergeStatus::Pending])?;
    let mut raw: Vec<(String, String, String, f64, MergeStatus)> = Vec::new();
    while let Some(row) = rows.next()? {
        raw.push((
            row.get(0)?,
            row.get(1)?,
            row.get(2)?,
            row.get(3)?,
            row.get(4)?,
        ));
    }
    drop(rows);
    drop(stmt);
    let mut out = Vec::new();
    for (id, keep_id, drop_id, distance, status) in raw {
        let (Some(keep), Some(drop)) = (store::node(db, &keep_id)?, store::node(db, &drop_id)?)
        else {
            continue;
        };
        out.push(MergeProposal {
            id,
            keep,
            drop,
            distance,
            status,
        });
    }
    Ok(out)
}

/// Resolve a full merge id or a unique prefix.
///
/// # Errors
///
/// Returns an error when nothing or more than one proposal matches.
pub fn find(db: &WorkspaceDb, prefix: &str) -> Result<MergeProposal> {
    PrefixMatch::of(pending(db)?, prefix, |m| m.id.as_str()).one(Record::MergeProposal, prefix)
}

/// Record a reviewer's answer to a pending proposal (full id or unique
/// prefix): an accepted one merges its nodes; a rejected one is not
/// proposed again.
///
/// # Errors
///
/// Returns an error when the proposal is not pending or the merge fails.
pub fn decide(
    db: &WorkspaceDb,
    id: &str,
    decision: MergeDecision,
    decided_by: Option<&str>,
) -> Result<MergeProposal> {
    let proposal = find(db, id)?;
    if decision == MergeDecision::Accept {
        merge_nodes(db, &proposal.keep.id, &proposal.drop.id)?;
    }
    db.connection().execute(
        "UPDATE _quack_graph_merges SET status = ?, decided_by = ?, decided_at = now() WHERE id = ?",
        duckdb::params![decision.status(), decided_by, proposal.id],
    )?;
    Ok(proposal)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tokens_of_three_letters_or_more_must_overlap() {
        assert!(share_token("Acme Corp", "ACME Corporation"));
        assert!(!share_token("Acme", "Apex"));
        assert!(!share_token("A B", "A C"));
    }
}
