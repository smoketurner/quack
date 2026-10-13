use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use super::Engine;
use crate::config::Config;
use crate::error::Error;
use crate::forward::Holder;
use crate::ids::WorkspaceId;
use crate::jobs::{JobKind, JobSpec, JobState};

fn config_in(dir: &Path) -> Config {
    let mut config = Config::default();
    config.general.data_dir = dir.join("data");
    config
}

#[tokio::test]
#[expect(clippy::unwrap_used, reason = "test")]
async fn a_workspace_opens_once_per_process() {
    let dir = tempfile::tempdir().unwrap();
    let engine = Engine::new(config_in(dir.path()), Holder::Serve);
    let id = WorkspaceId::from("ws");
    let first = engine.workspace(&id).await.unwrap();
    let second = engine.workspace(&id).await.unwrap();
    assert!(Arc::ptr_eq(&first.writer, &second.writer));
    first
        .writer
        .run(|db| {
            db.execute_query("CREATE TABLE t AS SELECT 1 AS n")
                .map(drop)
        })
        .await
        .unwrap();
    let rows = second
        .reader
        .with_db(|db| db.execute_query("SELECT n FROM t"))
        .await
        .unwrap();
    assert_eq!(rows.rows.len(), 1);
    engine.close().await;
}

#[tokio::test]
#[expect(clippy::unwrap_used, reason = "test")]
async fn a_holder_of_one_workspace_refuses_another() {
    let dir = tempfile::tempdir().unwrap();
    let held = WorkspaceId::from("mine");
    let engine = Engine::new(config_in(dir.path()), Holder::Terminal(held.clone()));
    assert!(engine.workspace(&held).await.is_ok());
    let refused = engine.workspace(&WorkspaceId::from("other")).await.err();
    assert!(
        matches!(&refused, Some(Error::WorkspaceNotHeld { holder }) if holder == "the quack terminal session"),
        "{refused:?}"
    );
    let serve = Engine::new(config_in(dir.path()), Holder::Serve);
    // Another file: the terminal engine above still holds `mine`.
    assert!(serve.workspace(&WorkspaceId::from("other")).await.is_ok());
    engine.close().await;
    serve.close().await;
}

#[tokio::test]
#[expect(clippy::unwrap_used, reason = "test")]
async fn close_workspace_refuses_the_held_one_and_a_busy_one() {
    let dir = tempfile::tempdir().unwrap();
    let held = WorkspaceId::from("mine");
    let engine = Engine::new(config_in(dir.path()), Holder::Terminal(held.clone()));
    assert!(matches!(
        engine.close_workspace(&held).await,
        Err(Error::WorkspaceHeld { .. })
    ));

    let serve = Engine::new(config_in(dir.path()), Holder::Serve);
    let busy = WorkspaceId::from("busy");
    serve.workspace(&busy).await.unwrap();
    let (release, released) = tokio::sync::oneshot::channel::<()>();
    // A queued job counts as active too, so no wait for it to start.
    let slow = serve.jobs.submit(
        JobSpec::new(JobKind::Ingest, "slow").workspace(busy.clone()),
        |_| async move {
            drop(released.await);
            Ok(String::new())
        },
    );
    assert!(matches!(
        serve.close_workspace(&busy).await,
        Err(Error::WorkspaceBusy { active: 1 })
    ));
    drop(release);
    serve.jobs.wait(slow.id).await;
    serve.close_workspace(&busy).await.unwrap();
    // Closed: the next use opens the file afresh.
    serve.workspace(&busy).await.unwrap();
    serve.close().await;
    engine.close().await;
}

#[tokio::test]
#[expect(clippy::unwrap_used, reason = "test")]
async fn concurrent_first_opens_share_one_handle() {
    let dir = tempfile::tempdir().unwrap();
    let engine = Engine::new(config_in(dir.path()), Holder::Serve);
    let id = WorkspaceId::from("ws");
    let (a, b, c) = tokio::join!(
        engine.workspace(&id),
        engine.workspace(&id),
        engine.workspace(&id)
    );
    let (a, b, c) = (a.unwrap(), b.unwrap(), c.unwrap());
    assert!(Arc::ptr_eq(&a.writer, &b.writer));
    assert!(Arc::ptr_eq(&b.writer, &c.writer));
    engine.close().await;
}

#[tokio::test]
#[expect(clippy::unwrap_used, reason = "test")]
async fn a_closed_engine_opens_nothing_again() {
    let dir = tempfile::tempdir().unwrap();
    let engine = Engine::new(config_in(dir.path()), Holder::Serve);
    let id = WorkspaceId::from("ws");
    // A straggler keeps its handle; a later open must not make a second
    // database instance beside it.
    let straggler = engine.workspace(&id).await.unwrap();
    engine.close().await;
    assert!(engine.is_stopping());
    assert!(matches!(
        engine.workspace(&id).await,
        Err(Error::EngineClosed)
    ));
    assert!(matches!(
        engine.with_workspace_closed(&id, || Ok(())).await,
        Err(Error::EngineClosed)
    ));
    drop(straggler);
}

/// A temp table loaded on the writer after the pool was built (the piped
/// `stdin` of `quack -p`) is what every read sees once the pool is told.
#[tokio::test]
#[expect(clippy::unwrap_used, reason = "test")]
async fn a_temp_table_loaded_after_the_open_reaches_the_readers() {
    let dir = tempfile::tempdir().unwrap();
    let engine = Engine::new(
        config_in(dir.path()),
        Holder::Command {
            workspace: WorkspaceId::from("ws"),
            verb: "-p",
        },
    );
    let handle = engine.workspace(&WorkspaceId::from("ws")).await.unwrap();
    handle
        .writer
        .run(|db| {
            db.execute_query("CREATE TEMP TABLE stdin AS SELECT 1 AS n")
                .map(drop)
        })
        .await
        .unwrap();
    handle.reader.observe_write().await;
    let rows = handle
        .reader
        .with_db(|db| db.execute_query("SELECT n FROM stdin"))
        .await
        .unwrap();
    assert_eq!(rows.rows.len(), 1);
    drop(handle);
    engine.close().await;
}

#[tokio::test]
#[expect(clippy::unwrap_used, reason = "test")]
async fn with_workspace_closed_reopens_the_file() {
    let dir = tempfile::tempdir().unwrap();
    let engine = Engine::new(config_in(dir.path()), Holder::Serve);
    let id = WorkspaceId::from("ws");
    let handle = engine.workspace(&id).await.unwrap();
    handle
        .writer
        .run(|db| {
            db.execute_query("CREATE TABLE t AS SELECT 1 AS n")
                .map(drop)
        })
        .await
        .unwrap();
    let path = engine.config.workspace_db_path("ws");
    let size = engine
        .with_workspace_closed(&id, move || Ok(std::fs::metadata(&path)?.len()))
        .await
        .unwrap();
    assert!(size > 0);
    let rows = handle
        .reader
        .with_db(|db| db.execute_query("SELECT n FROM t"))
        .await
        .unwrap();
    assert_eq!(rows.rows.len(), 1);
    engine.close().await;
}

#[tokio::test]
#[expect(clippy::unwrap_used, reason = "test")]
async fn stop_jobs_ends_what_watches_stopping() {
    let dir = tempfile::tempdir().unwrap();
    let engine = Engine::new(config_in(dir.path()), Holder::Serve);
    let watched = engine.stopping();
    assert!(!watched.is_cancelled());
    let left = engine.stop_jobs(Duration::from_millis(10)).await;
    assert!(left.is_empty());
    assert!(watched.is_cancelled());
    assert!(engine.is_stopping());
    let later = engine
        .jobs
        .submit(JobSpec::new(JobKind::Sql, "late"), |_| async {
            Ok(String::new())
        });
    assert_eq!(later.state, JobState::Cancelled);
}

/// A job's end is kept in its workspace's file: the next process to open
/// the file lists it, finished, with a fresh number; one an earlier
/// process left running is listed as failed.
#[tokio::test]
#[expect(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    reason = "test: the labels assertion fixes the length"
)]
async fn jobs_survive_the_process_in_the_workspace_file() {
    use crate::jobs::store;
    use crate::jobs::{JobId, JobInfo, JobNumber, JobProgress};
    use crate::storage::workspace::WorkspaceDb;

    let dir = tempfile::tempdir().unwrap();
    let config = config_in(dir.path());
    let id = WorkspaceId::from("ws");
    let first = Engine::new(config.clone(), Holder::Serve);
    first.workspace(&id).await.unwrap();
    let done = first.jobs.submit(
        JobSpec::new(JobKind::Ingest, "notes.md").workspace(id.clone()),
        |_| async { Ok(String::from("1 chunk")) },
    );
    let failed = first.jobs.submit(
        JobSpec::new(JobKind::Import, "orders.csv").workspace(id.clone()),
        |_| async { Err(String::from("404")) },
    );
    first.jobs.wait(done.id).await;
    first.jobs.wait(failed.id).await;
    // No workspace: never written anywhere.
    let homeless = first
        .jobs
        .submit(JobSpec::new(JobKind::Models, "models"), |_| async {
            Ok(String::new())
        });
    first.jobs.wait(homeless.id).await;
    first.close().await;

    // A job another process left running.
    {
        let db = WorkspaceDb::open(&config, "ws").unwrap();
        let running: JobId = "0199d1f0-0000-7000-8000-00000000abcd".parse().unwrap();
        store::upsert(
            &db,
            &JobInfo {
                id: running,
                number: JobNumber::default(),
                kind: JobKind::Graph,
                label: String::from("graph extraction"),
                workspace_id: Some(id.clone()),
                owner: None,
                lane: Some(String::from("graph:ws")),
                state: JobState::Running,
                progress: Some(JobProgress { done: 3, total: 9 }),
                status: Some(String::from("chunk 3")),
                outcome: None,
                queued_at: jiff::Timestamp::now(),
                started_at: Some(jiff::Timestamp::now()),
                finished_at: None,
                cancel_requested: false,
            },
            "a crashed process",
        )
        .unwrap();
    }

    let second = Engine::new(config, Holder::Serve);
    assert!(second.jobs.list().is_empty());
    second.workspace(&id).await.unwrap();
    let restored = second.jobs.list_workspace(&id);
    let labels: Vec<&str> = restored.iter().map(|j| j.label.as_str()).collect();
    assert_eq!(labels, vec!["notes.md", "orders.csv", "graph extraction"]);
    assert_eq!(restored[0].state, JobState::Succeeded);
    assert_eq!(restored[0].outcome.as_deref(), Some("1 chunk"));
    assert_eq!(restored[0].number.to_string(), "1");
    assert_eq!(restored[1].state, JobState::Failed);
    assert_eq!(restored[1].outcome.as_deref(), Some("404"));
    assert_eq!(restored[2].state, JobState::Failed);
    assert_eq!(restored[2].outcome.as_deref(), Some(store::INTERRUPTED));
    assert!(restored[2].finished_at.is_some());
    assert_eq!(
        restored[2].progress,
        Some(JobProgress { done: 3, total: 9 })
    );
    assert_eq!(second.jobs.list().len(), 3);
    // Opening again in the same process adds nothing twice.
    second.workspace(&id).await.unwrap();
    assert_eq!(second.jobs.list().len(), 3);
    second.close().await;
}
