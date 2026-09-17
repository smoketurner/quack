//! The upload queue: uploads return 202 with the document id and are
//! parsed, stored, and embedded by a bounded set of workers per workspace.
//! Clients poll the document until its status is `ready` or `error`.

use std::collections::HashMap;
use std::sync::Arc;

use quack_core::analysis::tools::SharedDb;
use quack_core::config::Config;
use quack_core::ingestion;
use quack_core::llm;
use tokio::sync::{Mutex, mpsc};

/// Queued uploads per workspace before `POST` blocks.
const QUEUE_DEPTH: usize = 64;

pub(crate) struct UploadJob {
    pub document_id: String,
    pub filename: String,
    pub data: Vec<u8>,
}

struct Lane {
    sender: mpsc::Sender<UploadJob>,
}

pub(crate) struct UploadQueue {
    workers: u32,
    lanes: Mutex<HashMap<String, Lane>>,
}

impl UploadQueue {
    pub(crate) fn new(workers_per_workspace: u32) -> Self {
        Self {
            workers: workers_per_workspace.max(1),
            lanes: Mutex::new(HashMap::new()),
        }
    }

    /// Queue a job for the workspace, starting its workers on first use.
    /// Waits when the lane is full.
    pub(crate) async fn submit(
        &self,
        config: &Config,
        workspace_id: &str,
        db: SharedDb,
        job: UploadJob,
    ) -> Result<(), String> {
        let sender = {
            let mut lanes = self.lanes.lock().await;
            if let Some(lane) = lanes.get(workspace_id) {
                lane.sender.clone()
            } else {
                let (sender, receiver) = mpsc::channel(QUEUE_DEPTH);
                let receiver = Arc::new(Mutex::new(receiver));
                for _ in 0..self.workers {
                    tokio::spawn(worker(
                        config.clone(),
                        workspace_id.to_owned(),
                        Arc::clone(&db),
                        Arc::clone(&receiver),
                    ));
                }
                lanes.insert(
                    workspace_id.to_owned(),
                    Lane {
                        sender: sender.clone(),
                    },
                );
                sender
            }
        };
        sender
            .send(job)
            .await
            .map_err(|_| String::from("upload workers are gone"))
    }
}

async fn worker(
    config: Config,
    workspace_id: String,
    db: SharedDb,
    receiver: Arc<Mutex<mpsc::Receiver<UploadJob>>>,
) {
    loop {
        let job = receiver.lock().await.recv().await;
        let Some(job) = job else {
            return;
        };
        let config = config.clone();
        let workspace_id = workspace_id.clone();
        let db = Arc::clone(&db);
        let handle = tokio::runtime::Handle::current();
        // Parsing and DuckDB writes are blocking work; the embedding calls
        // inside need the runtime, so block on it from a blocking thread.
        let outcome = tokio::task::spawn_blocking(move || {
            handle.block_on(process(&config, &workspace_id, &db, job));
        })
        .await;
        if let Err(e) = outcome {
            tracing::error!(error = %e, "upload worker task failed");
        }
    }
}

async fn process(config: &Config, workspace_id: &str, db: &SharedDb, job: UploadJob) {
    let model = match llm::optional_embedding_model(config).await {
        Ok(model) => model,
        Err(e) => {
            tracing::warn!(error = %e, document = %job.document_id, "upload fails: no embedding model");
            match db.lock() {
                Ok(guard) => {
                    if let Err(mark) = guard.mark_document_error(&job.document_id, &e.to_string()) {
                        tracing::error!(error = %mark, document = %job.document_id, "could not record the upload failure");
                    }
                }
                Err(poisoned) => {
                    tracing::error!(error = %poisoned, document = %job.document_id, "workspace lock poisoned; upload failure not recorded");
                }
            }
            return;
        }
    };
    let result = ingestion::process_document(
        config,
        db,
        workspace_id,
        &job.document_id,
        &job.filename,
        &job.data,
        model.as_ref(),
    )
    .await;
    match result {
        Ok(r) => {
            tracing::info!(document = %r.document_id, file = %r.filename, chunks = r.chunks_stored, "upload processed");
        }
        Err(e) => {
            tracing::warn!(document = %job.document_id, error = %e, "upload failed");
        }
    }
}
