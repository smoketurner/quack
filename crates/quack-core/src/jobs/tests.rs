use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use super::*;

#[expect(clippy::panic, reason = "test failure path")]
fn fail(msg: &str) -> ! {
    panic!("{msg}")
}

/// A job's `lane` shows its key as `kind:id`.
#[test]
fn lane_keys_read_as_kind_and_id() {
    let id = || WorkspaceId::from("w1");
    for (key, text) in [
        (LaneKey::Session(SessionId::from("w1")), "session:w1"),
        (LaneKey::Ingest(id()), "ingest:w1"),
        (LaneKey::Graph(id()), "graph:w1"),
        (LaneKey::Ontology(id()), "ontology:w1"),
        (LaneKey::Embeddings(id()), "embeddings:w1"),
    ] {
        assert_eq!(key.to_string(), text);
        assert_eq!(Lane::serial(&key).key(), text);
    }
}

#[test]
fn progress_ratio_stays_between_zero_and_one() {
    let ratio = |done, total| JobProgress { done, total }.ratio();
    assert!(ratio(0, 0).abs() < f64::EPSILON, "nothing to count");
    assert!((ratio(1, 4) - 0.25).abs() < f64::EPSILON);
    assert!((ratio(4, 4) - 1.0).abs() < f64::EPSILON);
    assert!(
        (ratio(9, 4) - 1.0).abs() < f64::EPSILON,
        "never past the end"
    );
}

#[test]
fn progress_shows_percent_rounded_down() {
    let shown = |done, total| JobProgress { done, total }.to_string();
    assert_eq!(shown(1576, 3835), "1576/3835 (41%)");
    assert_eq!(shown(0, 5), "0/5 (0%)");
    assert_eq!(shown(3834, 3835), "3834/3835 (99%)");
    assert_eq!(shown(3835, 3835), "3835/3835 (100%)");
    assert_eq!(
        shown(u32::MAX, u32::MAX),
        format!("{0}/{0} (100%)", u32::MAX)
    );
    assert_eq!(shown(0, 0), "0/0");
}

async fn finished(queue: &JobQueue, id: JobId) -> JobInfo {
    tokio::time::timeout(Duration::from_secs(5), queue.wait(id))
        .await
        .unwrap_or_else(|_| fail("job did not finish"))
        .unwrap_or_else(|| fail("job forgotten"))
}

/// Whether every lane is forgotten within a few seconds. A job reads as
/// finished just before it releases its lane slot (so the next job in the
/// lane cannot start first), and on a multi-threaded runtime the release
/// can land a moment after `finished` returns.
async fn lanes_drained(queue: &JobQueue) -> bool {
    let empty = || {
        queue
            .inner
            .lanes
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .is_empty()
    };
    tokio::time::timeout(Duration::from_secs(5), async {
        while !empty() {
            tokio::time::sleep(Duration::from_millis(1)).await;
        }
    })
    .await
    .is_ok()
}

#[tokio::test]
async fn a_job_acts_for_whoever_submitted_it() {
    use crate::llm::acting::Acting;
    let queue = JobQueue::new(10);
    let submit = |label: &str| {
        queue.submit(JobSpec::new(JobKind::Ingest, label), |_| async {
            Ok(Acting::current().map_or_else(|| String::from("nobody"), |a| a.user().to_string()))
        })
    };
    let ada = Acting::request(async {
        Acting::fixed(UserId::from("ada"), Err("unused")).enter();
        submit("ada's upload")
    })
    .await;
    let anonymous = submit("the CLI's upload");
    assert_eq!(
        finished(&queue, ada.id).await.outcome.as_deref(),
        Some("ada")
    );
    assert_eq!(
        finished(&queue, anonymous.id).await.outcome.as_deref(),
        Some("nobody")
    );
}

#[tokio::test]
async fn a_job_sends_where_its_submitter_may() {
    let queue = JobQueue::new(10);
    let submit = |label: &str| {
        queue.submit(JobSpec::new(JobKind::Ingest, label), |_| async {
            Ok(format!("{:?}", Egress::current()))
        })
    };
    let scoped = Egress::request(async {
        Egress::NoWorkspace.enter();
        submit("scoped")
    })
    .await;
    let unscoped = submit("unscoped");
    assert_eq!(
        finished(&queue, scoped.id).await.outcome.as_deref(),
        Some("Some(NoWorkspace)")
    );
    assert_eq!(
        finished(&queue, unscoped.id).await.outcome.as_deref(),
        Some("None")
    );
}

#[tokio::test]
async fn jobs_run_report_and_finish() {
    let queue = JobQueue::new(10);
    let mut events = queue.subscribe();
    let ok = queue
        .submit(
            JobSpec::new(JobKind::Sql, "select").workspace(WorkspaceId::from("ws")),
            |ctx| async move {
                ctx.progress(1, 2);
                ctx.status("halfway");
                Ok(String::from("2 rows"))
            },
        )
        .id;
    let err = queue
        .submit(JobSpec::new(JobKind::Ingest, "bad.pdf"), |_| async {
            Err(String::from("not a pdf"))
        })
        .id;
    let ok = finished(&queue, ok).await;
    assert_eq!(ok.state, JobState::Succeeded);
    assert_eq!(ok.outcome.as_deref(), Some("2 rows"));
    assert_eq!(ok.progress, Some(JobProgress { done: 1, total: 2 }));
    assert_eq!(ok.status.as_deref(), Some("halfway"));
    assert!(ok.started_at.is_some() && ok.finished_at.is_some());
    let err = finished(&queue, err).await;
    assert_eq!(err.state, JobState::Failed);
    assert_eq!(err.outcome.as_deref(), Some("not a pdf"));
    assert_eq!(err.number, JobNumber(2));

    // The first event is the queued snapshot.
    let first = events.recv().await.unwrap_or_else(|e| fail(&e.to_string()));
    assert_eq!(first.state, JobState::Queued);
    assert_eq!(queue.list_workspace(&WorkspaceId::from("ws")).len(), 1);
    assert_eq!(queue.counts(None).active(), 0);
    assert_eq!(queue.by_number(JobNumber(2)).map(|j| j.id), Some(err.id));
}

#[tokio::test]
async fn a_serial_lane_runs_in_order_and_other_work_is_not_held_up() {
    let queue = JobQueue::new(50);
    let log = Arc::new(Mutex::new(Vec::new()));
    let gate = Arc::new(tokio::sync::Notify::new());
    let mut ids = Vec::new();
    for n in 0..3_u32 {
        let log = Arc::clone(&log);
        let gate = Arc::clone(&gate);
        ids.push(
            queue
                .submit(
                    JobSpec::new(JobKind::Chat, format!("turn {n}"))
                        .lane(Lane::serial(&LaneKey::Session(SessionId::from("a")))),
                    move |_| async move {
                        if n == 0 {
                            gate.notified().await;
                        }
                        log.lock().unwrap_or_else(PoisonError::into_inner).push(n);
                        Ok(String::new())
                    },
                )
                .id,
        );
    }
    // Another lane is not held up by the first.
    let other = queue
        .submit(
            JobSpec::new(JobKind::Sql, "other")
                .lane(Lane::serial(&LaneKey::Session(SessionId::from("b")))),
            |_| async { Ok(String::from("done")) },
        )
        .id;
    assert_eq!(finished(&queue, other).await.state, JobState::Succeeded);
    let counts = queue.counts(None);
    assert_eq!((counts.running, counts.queued), (1, 2));
    gate.notify_one();
    let mut previous_end = None;
    for id in ids {
        let job = finished(&queue, id).await;
        // Each starts only after the one before it reads as finished.
        if let Some(end) = previous_end {
            assert!(job.started_at.is_some_and(|start| start >= end));
        }
        previous_end = job.finished_at;
    }
    assert_eq!(
        *log.lock().unwrap_or_else(PoisonError::into_inner),
        vec![0, 1, 2]
    );
    // The lane is forgotten once nobody holds it.
    assert!(lanes_drained(&queue).await);

    // A lane two wide runs two at once and queues the third; work
    // outside any lane is never held up by it.
    let wide = JobQueue::new(10);
    let gate = Arc::new(tokio::sync::Notify::new());
    let mut held = Vec::new();
    for n in 0..3 {
        let gate = Arc::clone(&gate);
        held.push(
            wide.submit(
                JobSpec::new(JobKind::Ingest, format!("{n}"))
                    .lane(Lane::new(&LaneKey::Ingest(WorkspaceId::from("w")), 2)),
                move |_| async move {
                    gate.notified().await;
                    Ok(String::new())
                },
            )
            .id,
        );
    }
    let free = wide
        .submit(JobSpec::new(JobKind::Sql, "select"), |_| async {
            Ok(String::new())
        })
        .id;
    assert_eq!(finished(&wide, free).await.state, JobState::Succeeded);
    tokio::time::sleep(Duration::from_millis(50)).await;
    let counts = wide.counts(None);
    assert_eq!((counts.running, counts.queued), (2, 1));
    assert_eq!(
        wide.lane_active(&LaneKey::Ingest(WorkspaceId::from("w"))),
        3
    );
    for _ in 0..3 {
        gate.notify_one();
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    for id in held {
        assert_eq!(finished(&wide, id).await.state, JobState::Succeeded);
    }
    assert_eq!(
        wide.lane_active(&LaneKey::Ingest(WorkspaceId::from("w"))),
        0
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_lane_keeps_submission_order_on_a_multi_threaded_runtime() {
    let queue = JobQueue::new(200);
    let log = Arc::new(Mutex::new(Vec::new()));
    let mut ids = Vec::new();
    for n in 0..50_u32 {
        let log = Arc::clone(&log);
        ids.push(
            queue
                .submit(
                    JobSpec::new(JobKind::Chat, format!("{n}"))
                        .lane(Lane::serial(&LaneKey::Session(SessionId::from("x")))),
                    move |_| async move {
                        log.lock().unwrap_or_else(PoisonError::into_inner).push(n);
                        Ok(String::new())
                    },
                )
                .id,
        );
    }
    // One cancelled while queued is skipped, not waited on.
    let skipped = ids.get(10).copied().unwrap_or_else(|| fail("no job 10"));
    queue.cancel(skipped);
    for id in ids {
        finished(&queue, id).await;
    }
    let ran = log.lock().unwrap_or_else(PoisonError::into_inner).clone();
    let mut sorted = ran.clone();
    sorted.sort_unstable();
    assert_eq!(ran, sorted, "the lane ran out of order");
    assert!(ran.len() >= 49);
    assert!(lanes_drained(&queue).await);
}

#[tokio::test]
async fn cancel_stops_queued_and_running_jobs_and_panics_fail() {
    let queue = JobQueue::new(10);
    let running = queue
        .submit(
            JobSpec::new(JobKind::Chat, "long")
                .lane(Lane::serial(&LaneKey::Session(SessionId::from("c")))),
            |ctx| async move {
                ctx.cancel_token().cancelled().await;
                Err(String::from("stopped"))
            },
        )
        .id;
    let queued = queue
        .submit(
            JobSpec::new(JobKind::Chat, "never")
                .lane(Lane::serial(&LaneKey::Session(SessionId::from("c")))),
            |_| async { Ok(String::from("ran")) },
        )
        .id;
    tokio::time::sleep(Duration::from_millis(20)).await;
    assert!(queue.cancel(queued));
    let queued = finished(&queue, queued).await;
    assert_eq!(queued.state, JobState::Cancelled);
    assert!(queued.started_at.is_none());
    assert!(queue.cancel(running));
    let running = finished(&queue, running).await;
    assert_eq!(running.state, JobState::Cancelled);
    assert!(running.cancel_requested);
    assert!(
        !queue.cancel(running.id),
        "a finished job cannot be cancelled"
    );

    #[expect(clippy::panic, reason = "the panic under test")]
    let panicked = queue
        .submit(JobSpec::new(JobKind::Graph, "boom"), |_| async {
            panic!("boom")
        })
        .id;
    let panicked = finished(&queue, panicked).await;
    assert_eq!(panicked.state, JobState::Failed);
    // The worker slot came back.
    let after = queue
        .submit(JobSpec::new(JobKind::Sql, "after"), |_| async {
            Ok(String::new())
        })
        .id;
    assert_eq!(finished(&queue, after).await.state, JobState::Succeeded);
}

#[tokio::test]
async fn history_keeps_the_newest_finished_jobs() {
    let queue = JobQueue::new(2);
    let mut last = None;
    for n in 0..5 {
        let id = queue
            .submit(JobSpec::new(JobKind::Sql, format!("{n}")), |_| async {
                Ok(String::new())
            })
            .id;
        finished(&queue, id).await;
        last = Some(id);
    }
    let listed = queue.list();
    assert_eq!(listed.len(), 2);
    assert_eq!(listed.last().map(|j| j.id), last);
    assert_eq!(
        listed
            .first()
            .map(|j| j.id.to_string())
            .unwrap_or_default()
            .parse::<JobId>()
            .ok(),
        listed.first().map(|j| j.id)
    );
}

/// `when_ended`'s cleanup must run even when the job is evicted from the
/// history ring in its own `finish` call. `a` queues behind a held job in
/// a serial lane, so it is submitted before 100 lane-less jobs that all
/// finish first; when `a` is then cancelled it is the oldest-by-submission
/// finished job and is evicted in the very `finish` that announced its end.
#[tokio::test(flavor = "current_thread")]
async fn when_ended_runs_for_a_job_evicted_in_its_own_finish() {
    let queue = JobQueue::new(100);
    let lane_key = LaneKey::Session(SessionId::from("z"));
    let gate = Arc::new(tokio::sync::Notify::new());
    let gate_for_work = Arc::clone(&gate);
    let held = queue
        .submit(
            JobSpec::new(JobKind::Chat, "held").lane(Lane::serial(&lane_key)),
            move |_| async move {
                gate_for_work.notified().await;
                Ok(String::new())
            },
        )
        .id;
    let a = queue
        .submit(
            JobSpec::new(JobKind::Chat, "a").lane(Lane::serial(&lane_key)),
            |_| async { Ok(String::new()) },
        )
        .id;
    for n in 0..100_u32 {
        let id = queue
            .submit(JobSpec::new(JobKind::Sql, format!("b{n}")), |_| async {
                Ok(String::new())
            })
            .id;
        finished(&queue, id).await;
    }

    let done = Arc::new(tokio::sync::Notify::new());
    let seen = Arc::new(Mutex::new(None::<JobInfo>));
    let done_for_record = Arc::clone(&done);
    let seen_for_record = Arc::clone(&seen);
    queue.when_ended(a, move |ended| {
        let done = Arc::clone(&done_for_record);
        let seen = Arc::clone(&seen_for_record);
        async move {
            *seen.lock().unwrap_or_else(PoisonError::into_inner) = Some(ended);
            done.notify_one();
        }
    });

    let wait_for_record = tokio::time::timeout(Duration::from_secs(5), done.notified());
    queue.cancel(a);
    match wait_for_record.await {
        Ok(()) => {}
        Err(_) => fail("when_ended record never ran at default history"),
    }

    gate.notify_one();
    assert!(
        tokio::time::timeout(Duration::from_secs(5), queue.wait(held))
            .await
            .is_ok(),
        "held did not settle"
    );

    let Some(ended) = seen.lock().unwrap_or_else(PoisonError::into_inner).clone() else {
        fail("no snapshot captured")
    };
    assert_eq!(ended.state, JobState::Cancelled);
    assert!(
        ended.never_started(),
        "evicted job's cleanup got its snapshot"
    );
    assert!(queue.get(a).is_none(), "a was evicted yet record still ran");
}

/// The same eviction shape as above, on the production multi-threaded
/// runtime. The evicting thread is strongly favored to re-lock before a
/// broadcast-woken watcher resumes, so the bug fires consistently here.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn when_ended_runs_for_a_job_evicted_on_a_multi_threaded_runtime() {
    let history = 100;
    let queue = JobQueue::new(history);
    let lane_key = LaneKey::Session(SessionId::from("m"));
    let gate = Arc::new(tokio::sync::Notify::new());
    let gate_for_work = Arc::clone(&gate);
    let holder = queue
        .submit(
            JobSpec::new(JobKind::Chat, "holder").lane(Lane::serial(&lane_key)),
            move |_| async move {
                gate_for_work.notified().await;
                Ok(String::new())
            },
        )
        .id;
    let a = queue
        .submit(
            JobSpec::new(JobKind::Chat, "a").lane(Lane::serial(&lane_key)),
            |_| async { Ok(String::new()) },
        )
        .id;
    for n in 0..history {
        let id = queue
            .submit(JobSpec::new(JobKind::Sql, format!("b{n}")), |_| async {
                Ok(String::new())
            })
            .id;
        finished(&queue, id).await;
    }

    let done = Arc::new(tokio::sync::Notify::new());
    let seen = Arc::new(Mutex::new(None::<JobInfo>));
    let done_for_record = Arc::clone(&done);
    let seen_for_record = Arc::clone(&seen);
    queue.when_ended(a, move |ended| {
        let done = Arc::clone(&done_for_record);
        let seen = Arc::clone(&seen_for_record);
        async move {
            *seen.lock().unwrap_or_else(PoisonError::into_inner) = Some(ended);
            done.notify_one();
        }
    });

    let wait_for_record = tokio::time::timeout(Duration::from_secs(5), done.notified());
    queue.cancel(a);
    match wait_for_record.await {
        Ok(()) => {}
        Err(_) => fail("when_ended record never ran on multi_thread"),
    }

    gate.notify_one();
    assert!(
        tokio::time::timeout(Duration::from_secs(5), queue.wait(holder))
            .await
            .is_ok(),
        "holder did not settle"
    );

    let Some(ended) = seen.lock().unwrap_or_else(PoisonError::into_inner).clone() else {
        fail("no snapshot captured")
    };
    assert_eq!(ended.state, JobState::Cancelled);
    assert!(ended.never_started());
    assert!(queue.get(a).is_none(), "a was evicted yet record still ran");
}

/// Many jobs sharing one serial lane, all queued behind a held job, each
/// with a `when_ended` watcher registered while queued. As they finish in
/// order the oldest are evicted past a small history bound; every watcher
/// must run its `record` exactly once.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn when_ended_runs_record_exactly_once_per_job_under_eviction() {
    let history = 8;
    let queue = JobQueue::new(history);
    let lane_key = LaneKey::Session(SessionId::from("race"));
    let gate = Arc::new(tokio::sync::Notify::new());
    let gate_for_work = Arc::clone(&gate);
    let _holder = queue
        .submit(
            JobSpec::new(JobKind::Chat, "holder").lane(Lane::serial(&lane_key)),
            move |_| async move {
                gate_for_work.notified().await;
                Ok(String::new())
            },
        )
        .id;
    let n = 64_u32;
    let called = Arc::new(AtomicUsize::new(0));
    let mut ids = Vec::new();
    for i in 0..n {
        let info = queue.submit(
            JobSpec::new(JobKind::Chat, format!("j{i}")).lane(Lane::serial(&lane_key)),
            |_| async { Ok(String::new()) },
        );
        let id = info.id;
        let called_for_record = Arc::clone(&called);
        queue.when_ended(id, move |_ended| {
            let called = Arc::clone(&called_for_record);
            async move {
                called.fetch_add(1, Ordering::SeqCst);
            }
        });
        ids.push(id);
    }
    gate.notify_one();
    for id in &ids {
        assert!(
            tokio::time::timeout(Duration::from_secs(5), queue.wait(*id))
                .await
                .is_ok(),
            "a watched job did not settle"
        );
    }
    let want = usize::try_from(n).unwrap_or(usize::MAX);
    tokio::time::timeout(Duration::from_secs(5), async {
        loop {
            if called.load(Ordering::SeqCst) == want {
                break;
            }
            tokio::time::sleep(Duration::from_millis(2)).await;
        }
    })
    .await
    .unwrap_or_else(|_| fail("not all when_ended records ran"));
    assert_eq!(
        called.load(Ordering::SeqCst),
        want,
        "each when_ended ran its record exactly once"
    );
    // Most watched jobs were evicted while their watcher was registered.
    let evicted = ids.iter().filter(|id| queue.get(**id).is_none()).count();
    let kept_bound = usize::try_from(history).unwrap_or(want);
    assert!(
        evicted >= want.saturating_sub(kept_bound),
        "test exercised eviction: only {evicted} of {want} were evicted"
    );
}

/// `when_ended` registered after the job has already finished but is still
/// in the history: the fast path delivers `record` from the current
/// snapshot.
#[tokio::test]
async fn when_ended_runs_when_registered_after_finish_while_still_in_history() {
    let queue = JobQueue::new(100);
    let job = queue
        .submit(JobSpec::new(JobKind::Sql, "ok"), |_| async {
            Ok(String::from("2 rows"))
        })
        .id;
    let finished_info = finished(&queue, job).await;
    assert_eq!(finished_info.state, JobState::Succeeded);

    let done = Arc::new(tokio::sync::Notify::new());
    let seen = Arc::new(Mutex::new(None::<JobInfo>));
    let done_for_record = Arc::clone(&done);
    let seen_for_record = Arc::clone(&seen);
    queue.when_ended(job, move |ended| {
        let done = Arc::clone(&done_for_record);
        let seen = Arc::clone(&seen_for_record);
        async move {
            *seen.lock().unwrap_or_else(PoisonError::into_inner) = Some(ended);
            done.notify_one();
        }
    });
    match tokio::time::timeout(Duration::from_secs(5), done.notified()).await {
        Ok(()) => {}
        Err(_) => fail("when_ended did not run for an already-finished job"),
    }
    let Some(ended) = seen.lock().unwrap_or_else(PoisonError::into_inner).clone() else {
        fail("no snapshot captured")
    };
    assert_eq!(ended.state, JobState::Succeeded);
    assert_eq!(ended.outcome.as_deref(), Some("2 rows"));
}

/// `when_ended` for a job that ended and was evicted before the call has
/// no snapshot left to deliver: `record` must not run, and the call must
/// not leave a task parked on a oneshot that is never sent to.
#[tokio::test]
async fn when_ended_does_not_run_or_hang_after_the_job_was_evicted() {
    let queue = JobQueue::new(1);
    let first = queue
        .submit(JobSpec::new(JobKind::Sql, "first"), |_| async {
            Ok(String::new())
        })
        .id;
    let second = queue
        .submit(JobSpec::new(JobKind::Sql, "second"), |_| async {
            Ok(String::new())
        })
        .id;
    finished(&queue, second).await;
    assert!(
        queue.get(first).is_none(),
        "first was evicted (history = 1)"
    );

    let queue_for_check = queue.clone();
    let called = Arc::new(AtomicBool::new(false));
    let called_for_record = Arc::clone(&called);
    // Wrap the call in a timeout to prove it returns promptly rather than
    // parking a task on a oneshot that nothing will ever send to.
    let spawn = tokio::spawn(async move {
        queue.when_ended(first, move |_ended| {
            let called = Arc::clone(&called_for_record);
            async move {
                called.store(true, Ordering::SeqCst);
            }
        });
    });
    match tokio::time::timeout(Duration::from_secs(5), spawn).await {
        Ok(Ok(())) => {}
        Ok(Err(_)) => fail("when_ended task panicked"),
        Err(_) => fail("when_ended hung for an evicted job"),
    }
    tokio::time::sleep(Duration::from_millis(50)).await;
    assert!(
        !called.load(Ordering::SeqCst),
        "record must not run for an already-evicted job with no snapshot"
    );
    assert!(
        queue_for_check.inner.registry().ended_waiters.is_empty(),
        "no waiter is left for an evicted job"
    );
}

/// The snapshot `when_ended` delivers is the actual finished snapshot
/// (state, outcome, `finished_at`), not a stale one, even when the job is
/// evicted by a later `finish` before the watcher resumes.
#[tokio::test]
async fn when_ended_delivers_the_actual_finished_snapshot_for_an_evicted_job() {
    let queue = JobQueue::new(1);
    let done = Arc::new(tokio::sync::Notify::new());
    let seen = Arc::new(Mutex::new(None::<JobInfo>));
    let watched = queue
        .submit(JobSpec::new(JobKind::Sql, "watched"), |_| async {
            Ok(String::from("summary-xyz"))
        })
        .id;
    let done_for_record = Arc::clone(&done);
    let seen_for_record = Arc::clone(&seen);
    queue.when_ended(watched, move |ended| {
        let done = Arc::clone(&done_for_record);
        let seen = Arc::clone(&seen_for_record);
        async move {
            *seen.lock().unwrap_or_else(PoisonError::into_inner) = Some(ended);
            done.notify_one();
        }
    });
    let later = queue
        .submit(JobSpec::new(JobKind::Sql, "later"), |_| async {
            Ok(String::new())
        })
        .id;
    drop(finished(&queue, later).await);
    match tokio::time::timeout(Duration::from_secs(5), done.notified()).await {
        Ok(()) => {}
        Err(_) => fail("when_ended did not run for an evicted job"),
    }
    let Some(ended) = seen.lock().unwrap_or_else(PoisonError::into_inner).clone() else {
        fail("no snapshot captured")
    };
    assert_eq!(ended.state, JobState::Succeeded);
    assert_eq!(ended.outcome.as_deref(), Some("summary-xyz"));
    assert!(ended.finished_at.is_some());
    assert!(
        queue.get(watched).is_none(),
        "watched was evicted but its snapshot was still delivered"
    );
}

/// Concurrent finishes past a history of one: each finish evicts another
/// job's entry, and every registered cleanup must still run.
#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn when_ended_runs_for_every_job_under_concurrent_finishes() {
    let queue = JobQueue::new(1);
    let n = 2000_usize;
    let called = Arc::new(AtomicUsize::new(0));
    for i in 0..n {
        let id = queue
            .submit(JobSpec::new(JobKind::Sql, format!("j{i}")), |_| async {
                for _ in 0..3 {
                    tokio::task::yield_now().await;
                }
                Ok(String::new())
            })
            .id;
        let called_for_record = Arc::clone(&called);
        queue.when_ended(id, move |_ended| async move {
            called_for_record.fetch_add(1, Ordering::SeqCst);
        });
    }
    let all_ran = tokio::time::timeout(Duration::from_secs(10), async {
        while called.load(Ordering::SeqCst) < n {
            tokio::time::sleep(Duration::from_millis(2)).await;
        }
    })
    .await
    .is_ok();
    let stranded = queue.inner.registry().ended_waiters.len();
    assert!(
        all_ran,
        "{} of {n} cleanups ran; {stranded} waiters stranded",
        called.load(Ordering::SeqCst)
    );
    assert_eq!(stranded, 0, "no waiter outlives its job");
}

/// Two cleanups registered on one job both run.
#[tokio::test]
async fn when_ended_runs_every_registration_on_one_job() {
    let queue = JobQueue::new(100);
    let gate = Arc::new(tokio::sync::Notify::new());
    let gate_for_work = Arc::clone(&gate);
    let id = queue
        .submit(JobSpec::new(JobKind::Sql, "held"), move |_| async move {
            gate_for_work.notified().await;
            Ok(String::new())
        })
        .id;
    let called = Arc::new(AtomicUsize::new(0));
    for _ in 0..2 {
        let called_for_record = Arc::clone(&called);
        queue.when_ended(id, move |_ended| async move {
            called_for_record.fetch_add(1, Ordering::SeqCst);
        });
    }
    gate.notify_one();
    finished(&queue, id).await;
    tokio::time::timeout(Duration::from_secs(5), async {
        while called.load(Ordering::SeqCst) < 2 {
            tokio::time::sleep(Duration::from_millis(1)).await;
        }
    })
    .await
    .unwrap_or_else(|_| fail("not every registration ran"));
    assert_eq!(called.load(Ordering::SeqCst), 2);
    assert!(queue.inner.registry().ended_waiters.is_empty());
}

/// Cancelling a job that already ended reports it inactive and announces
/// nothing after its finished snapshot.
#[tokio::test]
async fn cancel_after_finish_is_refused_without_an_event() {
    let queue = JobQueue::new(100);
    let id = queue
        .submit(JobSpec::new(JobKind::Sql, "done"), |_| async {
            Ok(String::new())
        })
        .id;
    finished(&queue, id).await;
    let mut events = queue.subscribe();
    assert!(!queue.cancel(id), "a finished job is not active");
    assert!(
        matches!(
            events.try_recv(),
            Err(broadcast::error::TryRecvError::Empty)
        ),
        "no event follows the finished one"
    );
    let Some(info) = queue.get(id) else {
        fail("job forgotten")
    };
    assert!(!info.cancel_requested);
}

/// Shutdown ends every job: a queued one without running, a running
/// one through its token, and it waits for what their ends record.
#[tokio::test]
async fn shutdown_cancels_queued_jobs_unrun_and_waits_for_their_records() {
    let queue = JobQueue::new(10);
    let lane = || Lane::serial(&LaneKey::Graph(WorkspaceId::from("ws")));
    let running = queue
        .submit(
            JobSpec::new(JobKind::Graph, "running").lane(lane()),
            |ctx| async move {
                ctx.cancel_token().cancelled().await;
                Err(String::from("stopped"))
            },
        )
        .id;
    let ran = Arc::new(AtomicBool::new(false));
    let queued: Vec<JobId> = (0..2)
        .map(|n| {
            let ran = Arc::clone(&ran);
            queue
                .submit(
                    JobSpec::new(JobKind::Graph, format!("queued {n}")).lane(lane()),
                    move |_| async move {
                        ran.store(true, Ordering::SeqCst);
                        Ok(String::new())
                    },
                )
                .id
        })
        .collect();
    let recorded = Arc::new(AtomicUsize::new(0));
    for id in &queued {
        let recorded = Arc::clone(&recorded);
        queue.when_ended(*id, move |ended| async move {
            tokio::time::sleep(Duration::from_millis(20)).await;
            if ended.never_started() {
                recorded.fetch_add(1, Ordering::SeqCst);
            }
        });
    }
    tokio::time::sleep(Duration::from_millis(20)).await;
    assert_eq!(queue.counts(None).running, 1);

    let left = queue.shutdown(Duration::from_secs(5)).await;
    assert!(left.is_empty(), "{left:?}");
    assert_eq!(recorded.load(Ordering::SeqCst), 2, "records were awaited");
    assert!(!ran.load(Ordering::SeqCst), "a queued job never ran");
    let state = |id| {
        queue
            .get(id)
            .map(|job| (job.state, job.started_at.is_some()))
    };
    assert_eq!(state(running), Some((JobState::Cancelled, true)));
    for id in queued {
        assert_eq!(state(id), Some((JobState::Cancelled, false)));
    }
}

#[tokio::test]
async fn a_job_submitted_after_shutdown_is_recorded_as_cancelled_and_never_runs() {
    let queue = JobQueue::new(10);
    assert!(queue.shutdown(Duration::ZERO).await.is_empty());
    let mut events = queue.subscribe();
    let ran = Arc::new(AtomicBool::new(false));
    let work_ran = Arc::clone(&ran);
    let refused = queue.submit(
        JobSpec::new(JobKind::Ingest, "late.pdf")
            .lane(Lane::serial(&LaneKey::Ingest(WorkspaceId::from("ws")))),
        move |_| async move {
            work_ran.store(true, Ordering::SeqCst);
            Ok(String::new())
        },
    );
    assert_eq!(refused.state, JobState::Cancelled);
    assert!(refused.never_started() && refused.finished_at.is_some());
    assert_eq!(refused.outcome.as_deref(), Some(REFUSED_BY_SHUTDOWN));
    assert_eq!(queue.get(refused.id), Some(refused.clone()));
    assert_eq!(events.try_recv().ok(), Some(refused.clone()));
    assert_eq!(queue.counts(None).active(), 0);
    assert!(lanes_drained(&queue).await, "a refused job takes no lane");

    // Its end is recorded like any other job cancelled while queued.
    let (told, record) = oneshot::channel();
    queue.when_ended(refused.id, move |ended| async move {
        assert!(told.send(ended.never_started()).is_ok());
    });
    assert_eq!(record.await.ok(), Some(true));
    assert!(queue.shutdown(Duration::from_secs(5)).await.is_empty());
    assert!(!ran.load(Ordering::SeqCst));
}

#[tokio::test]
async fn shutdown_reports_the_jobs_still_active_when_its_grace_runs_out() {
    let queue = JobQueue::new(10);
    let (release, released) = oneshot::channel::<()>();
    let deaf = queue
        .submit(JobSpec::new(JobKind::Ingest, "deaf"), |_| async move {
            drop(released.await);
            Ok(String::from("finished anyway"))
        })
        .id;
    tokio::time::sleep(Duration::from_millis(20)).await;
    let left = queue.shutdown(Duration::from_millis(50)).await;
    assert_eq!(left.len(), 1, "{left:?}");
    assert!(
        left.iter()
            .all(|job| job.id == deaf && job.state == JobState::Running && job.cancel_requested)
    );
    assert!(release.send(()).is_ok());
    assert_eq!(finished(&queue, deaf).await.state, JobState::Succeeded);
}

/// A job cancelled before its task first ran is still queued, so its
/// work never starts.
#[tokio::test]
async fn a_job_cancelled_before_its_task_first_runs_never_starts() {
    let queue = JobQueue::new(10);
    let ran = Arc::new(AtomicBool::new(false));
    let work_ran = Arc::clone(&ran);
    let id = queue
        .submit(JobSpec::new(JobKind::Sql, "select"), move |_| async move {
            work_ran.store(true, Ordering::SeqCst);
            Ok(String::new())
        })
        .id;
    assert!(queue.cancel(id));
    let ended = finished(&queue, id).await;
    assert!(ended.never_started(), "{ended:?}");
    assert!(!ran.load(Ordering::SeqCst));
}
