//! The claim on a table name: the only permission to create a document's
//! table. A claim is taken in the writer step that checks the name is free,
//! and it is recorded on the document's row in that same step, so a second
//! ingest that checks next sees the name taken even though the first has not
//! finished loading. A failed document clears its claim with its other
//! stored state (`WorkspaceDb::mark_document_error`).

use super::TableName;
use crate::error::Result;
use crate::ids::DocumentId;
use crate::storage::workspace::WorkspaceDb;

/// Permission for one document to create the table `name`. Only
/// [`TableName::claim`] makes one, and the loaders take it by reference, so
/// no load path creates a table without having checked and recorded the name.
#[derive(Debug)]
pub(super) struct TableClaim {
    name: TableName,
}

impl TableClaim {
    pub(super) fn as_str(&self) -> &str {
        self.name.as_str()
    }

    pub(super) fn into_name(self) -> String {
        self.name.into_string()
    }
}

impl TableName {
    /// Claim this name for `owner`: refuse it when another live document
    /// holds it (`Error::TableTaken`) or quack reserves it, otherwise record
    /// it on `owner`'s row. Run in the same writer step as the load it
    /// permits; claiming again for the same document changes nothing.
    ///
    /// # Errors
    ///
    /// `Error::TableTaken` for a name another document holds, an
    /// `Ingestion` error for a reserved one, or the database's error.
    pub(super) fn claim(&self, db: &WorkspaceDb, owner: &DocumentId) -> Result<TableClaim> {
        self.check_free(db, Some(owner))?;
        db.add_document_table(owner, self.as_str())?;
        Ok(TableClaim { name: self.clone() })
    }
}
