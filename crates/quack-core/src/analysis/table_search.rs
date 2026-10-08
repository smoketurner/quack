//! Schema linking (issue #403): which tables a question is about, for a
//! workspace with more tables than the prompt describes. Each table has a
//! card (its name, columns with what the ontology says of them, the
//! owner's note, the class it is mapped to, and its common values); cards
//! are ranked by BM25 over the same tokens the document index uses and,
//! when an embedding model exists, by cosine over a stored vector of each
//! card, fused by reciprocal rank. A card's vector is kept in
//! `_quack_table_cards` with a digest of the card's text and the embedding
//! profile it was made under, and made again when either changes.

use std::collections::{BTreeMap, HashMap};
use std::fmt::Write as _;

use crate::crypto;
use crate::embedding::{Embedder, EmbeddingModel, Input, Vector};
use crate::error::Result;
use crate::graph::views;
use crate::ontology::{Ontology, store as ontology_store};
use crate::storage::profile::{TableNote, TableProfile};
use crate::storage::workspace::{Analyzer, INTERNAL_PREFIX, WorkspaceDb};
use crate::storage::writer::Writer;
use crate::text::OneLine;

use super::tools::ReaderDb;

/// The stored card vectors.
pub const DDL: &str = "CREATE TABLE IF NOT EXISTS _quack_table_cards (
    table_name TEXT PRIMARY KEY,
    digest TEXT NOT NULL,
    embedding FLOAT[],
    embedding_profile TEXT
);";

/// Tables the prompt describes in full; past this many the rest are
/// listed by name and `find_tables` ranks them.
pub const DETAILED_TABLES: usize = 25;
/// Columns a card names.
const CARD_COLUMNS: usize = 60;
/// Columns whose common values a card carries.
const CARD_SAMPLED_COLUMNS: usize = 20;
/// Characters of the owner's note a card carries.
const CARD_NOTE_CHARS: usize = 500;
/// Card vectors made at most per turn, so the first turn after many
/// tables arrive is bounded; the rest rank by keyword until a later turn.
const EMBEDDED_PER_TURN: usize = 256;
/// BM25 parameters, as for the document index.
const K1: f64 = 1.2;
const B: f64 = 0.75;

/// Whether the workspace has more tables than the prompt describes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TableLayout {
    /// The prompt describes every table.
    AllDescribed,
    /// Tables past [`DETAILED_TABLES`] are named only: `find_tables`
    /// registers and the prompt ranks the tables against the question.
    Ranked,
}

impl TableLayout {
    /// The layout for `user_tables` tables (graph views not counted).
    #[must_use]
    pub const fn of(user_tables: usize) -> Self {
        if user_tables > DETAILED_TABLES {
            Self::Ranked
        } else {
            Self::AllDescribed
        }
    }
}

/// The tables a person or the agent made, without quack's graph views,
/// in name order.
///
/// # Errors
///
/// Returns an error if the catalog cannot be read.
pub fn user_tables(db: &WorkspaceDb) -> Result<Vec<String>> {
    let views = views::names(db)?;
    Ok(db
        .list_tables()?
        .into_iter()
        .filter(|t| !views.contains(t))
        .collect())
}

/// One table's searchable text.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TableCard {
    pub table: String,
    pub text: String,
}

impl TableCard {
    /// The SHA-256 of the text: a card whose text changed needs a new
    /// vector.
    #[must_use]
    pub fn digest(&self) -> String {
        crypto::sha256_hex(self.text.as_bytes())
    }

    fn input(&self) -> Input {
        Input::Document {
            title: Some(self.table.clone()),
            text: self.text.clone(),
        }
    }
}

/// Every user table's card.
#[derive(Debug, Clone, Default)]
pub struct TableCards(Vec<TableCard>);

/// One table's place in a ranking.
#[derive(Debug, Clone, PartialEq)]
pub struct RankedTable {
    pub table: String,
    pub score: f64,
}

impl TableCards {
    /// The cards of every user table, built from the catalog, the notes,
    /// the stored profiles, and `ontology`: a handful of queries whatever
    /// the number of tables.
    ///
    /// # Errors
    ///
    /// Returns an error if a read fails.
    pub fn read(db: &WorkspaceDb, ontology: Option<&Ontology>) -> Result<Self> {
        let tables = user_tables(db)?;
        let notes = TableNote::all(db)?;
        let profiles = TableProfile::all(db)?;
        let mut columns: BTreeMap<String, Vec<(String, String)>> = BTreeMap::new();
        let mut stmt = db.connection().prepare(
            "SELECT table_name, column_name, data_type FROM information_schema.columns \
             WHERE table_schema = 'main' AND NOT starts_with(lower(table_name), ?) \
             ORDER BY table_name, ordinal_position",
        )?;
        let mut rows = stmt.query(duckdb::params![INTERNAL_PREFIX])?;
        while let Some(row) = rows.next()? {
            let table: String = row.get(0)?;
            columns
                .entry(table)
                .or_default()
                .push((row.get(1)?, row.get(2)?));
        }
        let mut cards = Vec::with_capacity(tables.len());
        for table in tables {
            let mut text = format!("table {}", table.replace('_', " "));
            let mapping = ontology.and_then(|o| o.mapping_for_table(&table));
            if let (Some(ontology), Some(mapping)) = (ontology, mapping) {
                write!(text, "\nclass {}", mapping.class)?;
                if let Some(description) = ontology
                    .class(mapping.class.as_str())
                    .and_then(|c| c.description.as_deref())
                {
                    write!(text, ": {}", OneLine(description))?;
                }
            }
            if let Some(note) = notes.get(&table) {
                let cut: String = note.note.chars().take(CARD_NOTE_CHARS).collect();
                write!(text, "\nnote: {}", OneLine(&cut))?;
            }
            let profile = profiles.get(&table);
            for (index, (name, kind)) in columns
                .get(&table)
                .map(Vec::as_slice)
                .unwrap_or_default()
                .iter()
                .take(CARD_COLUMNS)
                .enumerate()
            {
                write!(text, "\ncolumn {} ({kind})", name.replace('_', " "))?;
                let property = mapping
                    .and_then(|m| m.properties.get(name))
                    .and_then(|id| ontology.and_then(|o| o.property(id)));
                if let Some(property) = property {
                    for part in property
                        .description
                        .iter()
                        .chain(property.unit.iter())
                        .chain(property.synonyms.iter())
                    {
                        write!(text, " {}", OneLine(part))?;
                    }
                }
                if index < CARD_SAMPLED_COLUMNS
                    && let Some(samples) = profile.and_then(|p| p.column(name))
                    && !samples.samples.is_empty()
                {
                    write!(text, ": {}", OneLine(&samples.samples.join(", ")))?;
                }
            }
            cards.push(TableCard { table, text });
        }
        Ok(Self(cards))
    }

    #[must_use]
    pub fn cards(&self) -> &[TableCard] {
        &self.0
    }

    /// The tables ranked against `query`, best first, at most `top_k`:
    /// BM25 over the cards, fused with cosine over their stored vectors
    /// when `query_vector` is given. A table neither leg finds is left out.
    ///
    /// # Errors
    ///
    /// Returns an error if the stored vectors cannot be read.
    pub fn rank(
        &self,
        db: &WorkspaceDb,
        query: &str,
        query_vector: Option<&Vector>,
        top_k: usize,
        rrf_k: u32,
    ) -> Result<Vec<RankedTable>> {
        let mut legs = vec![self.keyword_ranking(query)];
        if let Some(vector) = query_vector {
            legs.push(self.vector_ranking(db, vector)?);
        }
        let mut fused: HashMap<&str, f64> = HashMap::new();
        for leg in &legs {
            for (rank, table) in leg.iter().enumerate() {
                let rank = f64::from(u32::try_from(rank).unwrap_or(u32::MAX));
                *fused.entry(table.as_str()).or_default() += 1.0 / (f64::from(rrf_k) + rank + 1.0);
            }
        }
        let mut ranked: Vec<RankedTable> = fused
            .into_iter()
            .map(|(table, score)| RankedTable {
                table: table.to_owned(),
                score,
            })
            .collect();
        ranked.sort_by(|a, b| {
            b.score
                .total_cmp(&a.score)
                .then_with(|| a.table.cmp(&b.table))
        });
        ranked.truncate(top_k);
        Ok(ranked)
    }

    /// Tables whose card shares a term with the query, by BM25.
    fn keyword_ranking(&self, query: &str) -> Vec<String> {
        // Table cards are names and notes, not documents: one analyzer for both sides.
        let analyzer = Analyzer::default();
        let terms = analyzer.terms(query);
        if terms.is_empty() || self.0.is_empty() {
            return Vec::new();
        }
        let docs: Vec<(&str, HashMap<String, u32>, usize)> = self
            .0
            .iter()
            .map(|card| {
                let tokens = analyzer.terms(&card.text);
                let mut tf: HashMap<String, u32> = HashMap::new();
                for token in &tokens {
                    let count = tf.entry(token.clone()).or_default();
                    *count = count.saturating_add(1);
                }
                (card.table.as_str(), tf, tokens.len())
            })
            .collect();
        #[expect(clippy::cast_precision_loss, reason = "counts of tables and tokens")]
        let n = docs.len() as f64;
        #[expect(clippy::cast_precision_loss, reason = "counts of tokens")]
        let average = docs.iter().map(|(_, _, len)| *len as f64).sum::<f64>() / n;
        // Each query term's inverse document frequency, counted once.
        let idf: HashMap<&String, f64> = terms
            .iter()
            .map(|term| {
                #[expect(clippy::cast_precision_loss, reason = "a count of tables")]
                let df = docs.iter().filter(|(_, d, _)| d.contains_key(term)).count() as f64;
                (term, ((n - df + 0.5) / (df + 0.5)).ln_1p())
            })
            .collect();
        let mut scored: Vec<(&str, f64)> = Vec::new();
        for (table, tf, len) in &docs {
            let mut score = 0.0;
            for term in &terms {
                let (Some(&count), Some(&idf)) = (tf.get(term), idf.get(term)) else {
                    continue;
                };
                let count = f64::from(count);
                #[expect(clippy::cast_precision_loss, reason = "a count of tokens")]
                let norm = 1.0 - B + B * (*len as f64) / average.max(1.0);
                score += idf * count * (K1 + 1.0) / K1.mul_add(norm, count);
            }
            if score > 0.0 {
                scored.push((table, score));
            }
        }
        scored.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(b.0)));
        scored.into_iter().map(|(t, _)| t.to_owned()).collect()
    }

    /// Every table with a current vector, by cosine to `query`.
    fn vector_ranking(&self, db: &WorkspaceDb, query: &Vector) -> Result<Vec<String>> {
        let stored = StoredVectors::read(db)?;
        let mut scored: Vec<(&str, f64)> = Vec::new();
        for card in &self.0 {
            if let Some(vector) = stored.current(db, card) {
                scored.push((card.table.as_str(), query.cosine(vector)));
            }
        }
        scored.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(b.0)));
        Ok(scored.into_iter().map(|(t, _)| t.to_owned()).collect())
    }

    /// Make the vectors of cards that have none under the current profile,
    /// or whose text changed, at most 256 at a time;
    /// forget the vectors of tables that are gone. Nothing happens unless
    /// the workspace has more tables than the prompt describes.
    ///
    /// # Errors
    ///
    /// Returns an error if a read, the model, or the write fails.
    pub async fn refresh_vectors<M: EmbeddingModel>(
        reader: &ReaderDb,
        writer: &Writer,
        embedder: &Embedder<M>,
    ) -> Result<usize> {
        let stale = reader
            .with_db(|db| {
                let tables = user_tables(db)?;
                if TableLayout::of(tables.len()) == TableLayout::AllDescribed {
                    return Ok(Vec::new());
                }
                let ontology = ontology_store::current(db)?;
                let cards = Self::read(db, ontology.as_ref())?;
                let stored = StoredVectors::read(db)?;
                Ok(cards
                    .0
                    .into_iter()
                    .filter(|card| stored.current(db, card).is_none())
                    .take(EMBEDDED_PER_TURN)
                    .collect::<Vec<_>>())
            })
            .await?;
        if stale.is_empty() {
            return Ok(0);
        }
        let inputs: Vec<Input> = stale.iter().map(TableCard::input).collect();
        let vectors = embedder.embed(&inputs).await?;
        let fingerprint = embedder.profile().fingerprint();
        let made = stale.len();
        writer
            .run(move |db| {
                db.write_transaction(|db| {
                    for (card, vector) in stale.iter().zip(&vectors) {
                        db.connection().execute(
                            "INSERT OR REPLACE INTO _quack_table_cards \
                             (table_name, digest, embedding, embedding_profile) \
                             VALUES (?, ?, ?::FLOAT[], ?)",
                            duckdb::params![
                                card.table,
                                card.digest(),
                                vector.sql_literal(),
                                fingerprint
                            ],
                        )?;
                    }
                    let tables = user_tables(db)?;
                    let mut stmt = db
                        .connection()
                        .prepare("SELECT table_name FROM _quack_table_cards")?;
                    let stored = stmt
                        .query_map([], |row| row.get(0))?
                        .collect::<duckdb::Result<Vec<String>>>()?;
                    for gone in stored.iter().filter(|t| !tables.contains(t)) {
                        db.connection().execute(
                            "DELETE FROM _quack_table_cards WHERE table_name = ?",
                            duckdb::params![gone],
                        )?;
                    }
                    Ok(())
                })
            })
            .await?;
        Ok(made)
    }
}

/// The stored card vectors, by table: digest, profile, vector.
struct StoredVectors(HashMap<String, (String, Option<String>, Vector)>);

impl StoredVectors {
    fn read(db: &WorkspaceDb) -> Result<Self> {
        let mut stmt = db.connection().prepare(
            "SELECT table_name, digest, embedding_profile, CAST(to_json(embedding) AS VARCHAR) \
             FROM _quack_table_cards WHERE embedding IS NOT NULL",
        )?;
        let mut rows = stmt.query([])?;
        let mut out = HashMap::new();
        while let Some(row) = rows.next()? {
            let json: String = row.get(3)?;
            let Ok(values) = serde_json::from_str::<Vec<f32>>(&json) else {
                continue;
            };
            out.insert(
                row.get(0)?,
                (row.get(1)?, row.get(2)?, Vector::from(values)),
            );
        }
        Ok(Self(out))
    }

    /// The card's vector, when it was made from this text under the
    /// workspace's current embedding profile.
    fn current(&self, db: &WorkspaceDb, card: &TableCard) -> Option<&Vector> {
        let fingerprint = db.embedding_fingerprint()?;
        let (digest, profile, vector) = self.0.get(&card.table)?;
        (*digest == card.digest() && profile.as_deref() == Some(fingerprint.as_str()))
            .then_some(vector)
    }
}

#[cfg(test)]
mod tests;
