//! A folder of files as documents: the sorted walk of what quack can load,
//! and the run that ingests each file, replaces the one a changed file
//! stands for, skips an unchanged one, follows a moved one to its new
//! path, and reports the files that are gone.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::time::Instant;

use crate::config::Config;
use crate::embedding::{Embedder, EmbeddingModel};
use crate::error::{Error, Result};
use crate::ids::DocumentId;
use crate::ingestion::parser::FileType;
use crate::ingestion::{IngestOutcome, NewFile, ingest_file, parse_off_runtime};
use crate::progress::{ChunkDone, RunControl};
use crate::storage::workspace::{DocumentInfo, DocumentSource};
use crate::storage::writer::Writer;

/// One file quack can load, where it is and its path under the root.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TreeFile {
    pub path: PathBuf,
    /// Root-relative, `/`-separated: what the document records as its
    /// `source_path`, and what a later run of the same root matches it by.
    pub relative: String,
}

impl TreeFile {
    /// The file's own name, which the document is named after.
    #[must_use]
    pub fn name(&self) -> &str {
        self.relative.rsplit('/').next().unwrap_or(&self.relative)
    }
}

/// A folder's files in path order: those quack loads, and those it does
/// not. Directories and files whose name starts with `.` are left out, and
/// symbolic links are not followed, as `find` does not: a link can point
/// outside the folder or back up into it.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct Tree {
    pub files: Vec<TreeFile>,
    /// Root-relative paths of the files no parser reads.
    pub unsupported: Vec<String>,
}

impl Tree {
    /// Walk `root`.
    ///
    /// # Errors
    ///
    /// Returns an error if a directory cannot be read.
    pub fn walk(root: &Path) -> Result<Self> {
        let mut tree = Self::default();
        let mut stack = vec![root.to_path_buf()];
        while let Some(dir) = stack.pop() {
            let mut entries: Vec<(PathBuf, std::fs::FileType)> = std::fs::read_dir(&dir)?
                .map(|entry| entry.and_then(|e| Ok((e.path(), e.file_type()?))))
                .collect::<std::io::Result<_>>()?;
            // Popped last-first, so push in reverse to read in path order.
            entries.sort_by(|a, b| a.0.cmp(&b.0));
            entries.reverse();
            for (path, kind) in entries {
                let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
                    continue;
                };
                if name.starts_with('.') || kind.is_symlink() {
                    continue;
                }
                if kind.is_dir() {
                    stack.push(path);
                    continue;
                }
                let relative = path
                    .strip_prefix(root)
                    .map_err(|e| Error::Ingestion(e.to_string()))?
                    .to_string_lossy()
                    .replace('\\', "/");
                if FileType::of(name).is_some() {
                    tree.files.push(TreeFile { path, relative });
                } else {
                    tree.unsupported.push(relative);
                }
            }
        }
        tree.files.sort_by(|a, b| a.relative.cmp(&b.relative));
        tree.unsupported.sort();
        Ok(tree)
    }
}

/// What the run did with one file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Outcome {
    Ingested(DocumentId),
    /// A changed file: the document at its path is superseded by the new one.
    Replaced {
        old: DocumentId,
        new: DocumentId,
    },
    /// Identical bytes are already this document.
    Skipped(DocumentId),
    /// Identical bytes are this folder's document from a path that no
    /// longer has a file: the file moved, and the document now records
    /// its new path.
    Moved {
        document: DocumentId,
        from: String,
    },
    /// Parsing or storing failed; the document row carries the message.
    Failed(String),
}

/// One file's outcome, by its root-relative path.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FileResult {
    pub relative: String,
    pub outcome: Outcome,
}

/// What became of a document whose file is no longer in the folder.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Prune {
    /// Report it and leave it.
    Keep,
    /// Delete it with its chunks and tables (`--prune`).
    Delete,
}

/// Everything one folder run found and did.
#[derive(Debug, Clone)]
pub struct FolderReport {
    pub results: Vec<FileResult>,
    pub unsupported: Vec<String>,
    /// Ready documents with a source path no file in the folder has now;
    /// deleted when `pruned`.
    pub gone: Vec<DocumentInfo>,
    pub pruned: Prune,
}

impl FolderReport {
    /// The results with this outcome's kind, for the summary lines.
    #[must_use]
    pub fn failed(&self) -> usize {
        self.results
            .iter()
            .filter(|r| matches!(r.outcome, Outcome::Failed(_)))
            .count()
    }
}

/// A folder to ingest into a workspace: every file quack can load becomes
/// a document named after it, recording the folder's canonical path and
/// the file's path under it. On a later run of the same folder an
/// unchanged file is skipped by its bytes, a changed file replaces the
/// ready document at its path, and a path with no file left is reported,
/// or deleted with `Prune::Delete`; documents from any other folder are
/// never touched. Progress is one unit per file, and a cancel stops
/// between files or inside one.
pub struct Folder<'a, M> {
    pub config: &'a Config,
    pub db: &'a Writer,
    pub workspace_id: &'a str,
    pub root: &'a Path,
    pub embedder: Option<&'a Embedder<M>>,
    pub control: RunControl<'a>,
    pub prune: Prune,
}

impl<M: EmbeddingModel> Folder<'_, M> {
    /// Walk the folder and ingest it.
    ///
    /// # Errors
    ///
    /// Returns an error when the folder cannot be walked, a workspace step
    /// fails, or the run is cancelled; a file that does not parse is a
    /// `Failed` outcome, not an error.
    pub async fn run(self) -> Result<FolderReport> {
        let mut tree = Tree::walk(self.root)?;
        let source_root = std::fs::canonicalize(self.root)?
            .to_string_lossy()
            .into_owned();
        let present: BTreeSet<String> = tree.files.iter().map(|f| f.relative.clone()).collect();
        // A path that already has a document goes first: a changed file
        // replaces its document before any other file is matched against
        // its old bytes, so content that moved into a new path while its
        // old path changed is ingested, not skipped as a duplicate.
        let root = source_root.clone();
        let known: BTreeSet<String> = self
            .db
            .run(move |db| db.documents_under(&root))
            .await?
            .into_iter()
            .filter_map(|d| d.source_path)
            .collect();
        tree.files
            .sort_by_key(|f| (!known.contains(&f.relative), f.relative.clone()));
        let total = u32::try_from(tree.files.len()).unwrap_or(u32::MAX);
        let started = Instant::now();
        let mut results = Vec::with_capacity(tree.files.len());
        let mut failed = 0u32;
        for (index, file) in tree.files.iter().enumerate() {
            self.control.check()?;
            let unit = Instant::now();
            let outcome = match self.one(&source_root, file, &present).await {
                Ok(outcome) => outcome,
                Err(Error::Cancelled) => return Err(Error::Cancelled),
                Err(e) => {
                    failed = failed.saturating_add(1);
                    Outcome::Failed(e.to_string())
                }
            };
            results.push(FileResult {
                relative: file.relative.clone(),
                outcome,
            });
            (self.control.progress)(ChunkDone {
                done: u32::try_from(index.saturating_add(1)).unwrap_or(u32::MAX),
                total,
                failed,
                took: unit.elapsed(),
                elapsed: started.elapsed(),
            });
        }
        results.sort_by(|a, b| a.relative.cmp(&b.relative));
        let root = source_root.clone();
        let gone: Vec<DocumentInfo> = self
            .db
            .run(move |db| db.documents_under(&root))
            .await?
            .into_iter()
            .filter(|d| {
                d.source_path
                    .as_deref()
                    .is_some_and(|path| !present.contains(path))
            })
            .collect();
        if self.prune == Prune::Delete {
            for document in &gone {
                let id = document.id.clone();
                self.db.run(move |db| db.delete_document(&id)).await?;
            }
        }
        Ok(FolderReport {
            results,
            unsupported: tree.unsupported,
            gone,
            pruned: self.prune,
        })
    }

    /// Ingest one file, replacing the ready document at its path under
    /// `source_root`, or moving this folder's document of the same bytes
    /// to it when that document's path has no file in `present`.
    async fn one(
        &self,
        source_root: &str,
        file: &TreeFile,
        present: &BTreeSet<String>,
    ) -> Result<Outcome> {
        let path = file.path.clone();
        let data = parse_off_runtime(move || Ok(std::fs::read(path)?)).await?;
        let (root, relative) = (source_root.to_owned(), file.relative.clone());
        let predecessor = self
            .db
            .run(move |db| db.newest_document_at_path(&root, &relative))
            .await?;
        // The run reports one unit per file; a file's own chunk progress
        // stays inside it, its cancel does not.
        let per_file = RunControl {
            progress: &|_| {},
            cancel: self.control.cancel,
        };
        let new_file = NewFile::new(file.name(), &data)
            .source(DocumentSource::Path)
            .in_folder(source_root, &file.relative)
            .replaces(predecessor.as_ref().map(|d| &d.id))
            .control(per_file);
        let outcome = ingest_file(
            self.config,
            self.db,
            self.workspace_id,
            &new_file,
            self.embedder,
        )
        .await?;
        Ok(match outcome {
            IngestOutcome::Ingested(result) => match result.replaced {
                Some(old) => Outcome::Replaced {
                    old,
                    new: result.document_id,
                },
                None => Outcome::Ingested(result.document_id),
            },
            IngestOutcome::Duplicate(existing) => {
                let moved_from = existing.source_path.clone().filter(|from| {
                    existing.source_root.as_deref() == Some(source_root)
                        && *from != file.relative
                        && !present.contains(from)
                });
                match moved_from {
                    Some(from) => {
                        let (id, to) = (existing.id.clone(), file.relative.clone());
                        self.db.run(move |db| db.move_document(&id, &to)).await?;
                        Outcome::Moved {
                            document: existing.id,
                            from,
                        }
                    }
                    None => Outcome::Skipped(existing.id),
                }
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[expect(clippy::panic, reason = "test failure path")]
    fn fail(msg: &str) -> ! {
        panic!("{msg}")
    }

    /// The walk lists what quack loads in path order with `/`-separated
    /// root-relative paths, names the rest, and skips dot entries.
    #[test]
    fn walk_sorts_files_and_names_the_unsupported_ones() {
        let dir = tempfile::tempdir().unwrap_or_else(|e| fail(&e.to_string()));
        let root = dir.path();
        for (path, text) in [
            ("b.md", "two"),
            ("sub/a.csv", "x,y\n1,2\n"),
            ("sub/deeper/c.PDF", "%PDF"),
            ("notes.xyz", "?"),
            (".hidden/d.md", "hidden"),
            (".DS_Store", ""),
        ] {
            let path = root.join(path);
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent).unwrap_or_else(|e| fail(&e.to_string()));
            }
            std::fs::write(&path, text).unwrap_or_else(|e| fail(&e.to_string()));
        }
        let tree = Tree::walk(root).unwrap_or_else(|e| fail(&e.to_string()));
        assert_eq!(
            tree.files
                .iter()
                .map(|f| f.relative.as_str())
                .collect::<Vec<_>>(),
            ["b.md", "sub/a.csv", "sub/deeper/c.PDF"]
        );
        assert_eq!(
            tree.files.iter().map(TreeFile::name).collect::<Vec<_>>(),
            ["b.md", "a.csv", "c.PDF"]
        );
        assert!(tree.files.iter().all(|f| f.path.starts_with(root)));
        assert_eq!(tree.unsupported, ["notes.xyz"]);
        let empty = tempfile::tempdir().unwrap_or_else(|e| fail(&e.to_string()));
        assert_eq!(
            Tree::walk(empty.path()).unwrap_or_else(|e| fail(&e.to_string())),
            Tree::default()
        );
        assert!(Tree::walk(&root.join("missing")).is_err());
    }
}
