#![expect(
    clippy::unwrap_used,
    clippy::panic,
    reason = "tests assert on values they have just built"
)]

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use serde_json::{Value, json};
use tokio_util::sync::CancellationToken;

use super::*;
use crate::embedding::Dimension;
use crate::error::Error;
use crate::ids::DocumentId;
use crate::jobs::JobState;
use crate::llm::decision::fixture::{model, scoped};
use crate::llm::decision::stub::{DecisionStub, Fault};
use crate::llm::egress::Egress;
use crate::progress::ChunkDone;
use crate::storage::profile::{ColumnType, Retype, TableProfile};
use crate::storage::workspace::{DocumentSource, WorkspaceDb};
use crate::storage::writer::Claimed;
use plan::Plan;
use store::{Ended, PageWrite, Start};

fn triage() -> QuestionSet {
    named("triage")
}

fn named(name: &str) -> QuestionSet {
    serde_json::from_value(json!({
        "name": name,
        "questions": {
            "department": {
                "type": "choice",
                "instructions": "Which department should handle this ticket?",
                "criteria": {"billing": null, "technical": "Bugs, outages", "none": null}
            },
            "urgency": {
                "type": "score",
                "instructions": "How urgent is this ticket?",
                "criteria": ["Not urgent", "Soon", "Blocking"]
            },
            "churn": {"type": "noul", "instructions": "Does the customer threaten to cancel?"}
        }
    }))
    .unwrap()
}

fn request() -> Classification {
    Classification {
        table: String::from("tickets"),
        text_columns: vec![String::from("subject"), String::from("body")],
        key: None,
        question_set: triage(),
        rows: Rows::Missing,
    }
}

/// A workspace with a `tickets` table of 600 rows, a writer over it, and
/// the stub the decision model is served by.
struct Fixture {
    writer: Arc<Writer>,
    stub: DecisionStub,
}

impl Fixture {
    async fn new() -> Self {
        Self::with(DecisionStub::start().await)
    }

    fn with(stub: DecisionStub) -> Self {
        let db = WorkspaceDb::open_in_memory(Dimension::new(4)).unwrap();
        db.execute_statement(
            "CREATE TABLE tickets AS SELECT range AS id, \
             CASE WHEN range % 3 = 0 THEN 'technical t' ELSE 'billing t' END || range AS subject, \
             'please help' AS body FROM range(600)",
        )
        .unwrap();
        Self {
            writer: Arc::new(Writer::spawn(db).unwrap()),
            stub,
        }
    }

    async fn sql(&self, sql: &'static str) {
        self.writer
            .run(move |db| db.execute_statement(sql))
            .await
            .unwrap();
    }

    async fn count(&self, sql: &'static str) -> i64 {
        self.writer
            .run(move |db| Ok(db.connection().query_row(sql, [], |r| r.get(0))?))
            .await
            .unwrap()
    }

    async fn try_run(
        &self,
        request: &Classification,
        control: RunControl<'_>,
        waiting: Waiting,
    ) -> Result<ClassificationRun> {
        let decision = model(&self.stub).await;
        request
            .run(Labelling {
                db: &self.writer,
                decision: &decision,
                started_by: Some("user"),
                run_id: RunId::generate(),
                waiting,
                control,
            })
            .await
    }

    async fn run(&self, request: &Classification) -> ClassificationRun {
        self.try_run(request, RunControl::unobserved(), Waiting::Job)
            .await
            .unwrap()
    }

    async fn refusal(&self, request: &Classification) -> ClassifyError {
        match self
            .try_run(request, RunControl::unobserved(), Waiting::Job)
            .await
        {
            Err(Error::Classify(refusal)) => refusal,
            other => panic!("expected a classify refusal, got {other:?}"),
        }
    }

    async fn runs(&self) -> Vec<ClassificationRun> {
        self.writer
            .run(|db| ClassificationRun::list(db, 100))
            .await
            .unwrap()
            .runs
    }

    /// The `subject` of each decision request after the two probes, in
    /// order.
    fn asked_subjects(&self) -> Vec<String> {
        self.stub
            .bodies()
            .iter()
            .skip(2)
            .filter_map(|body| {
                let body: Value = serde_json::from_str(body).ok()?;
                Some(body.pointer("/state/subject")?.as_str()?.to_owned())
            })
            .collect()
    }
}

#[tokio::test]
async fn a_run_labels_every_row_in_key_order_one_page_at_a_time() {
    let fixture = Fixture::new().await;
    let reports = Mutex::new(Vec::new());
    let progress = |done: ChunkDone| reports.lock().unwrap().push((done.done, done.total));
    scoped(async {
        let run = fixture
            .try_run(
                &request(),
                RunControl {
                    progress: &progress,
                    cancel: None,
                },
                Waiting::Job,
            )
            .await
            .unwrap();
        assert_eq!(run.status, RunStatus::Completed);
        assert_eq!(
            (run.labelled, run.cut, run.empty, run.skipped),
            (600, 0, 0, 0)
        );
        assert_eq!(run.rows, Rows::Missing);
        assert_eq!(run.output_table, "tickets_triage");
        assert_eq!(run.key_column, "id");
    })
    .await;
    assert_eq!(
        *reports.lock().unwrap(),
        [(256, 600), (512, 600), (600, 600)]
    );
    assert_eq!(
        fixture.stub.requests(),
        602,
        "two probes and a request a row"
    );
    let order: Vec<u32> = fixture
        .asked_subjects()
        .iter()
        .filter_map(|s| s.rsplit('t').next()?.parse().ok())
        .collect();
    assert_eq!(
        order,
        (0..600).collect::<Vec<u32>>(),
        "numeric key order across pages"
    );
    assert_eq!(
        fixture.count("SELECT count(*) FROM tickets_triage").await,
        600
    );
    assert_eq!(
        fixture
            .count("SELECT count(*) FROM tickets_triage WHERE department = 'technical'")
            .await,
        200
    );
    assert_eq!(
        fixture
            .count("SELECT count(*) FROM tickets_triage WHERE urgency_level = 0 AND churn < 0.5")
            .await,
        600
    );
    assert_eq!(
        fixture
            .count("SELECT count(*) FROM tickets t JOIN tickets_triage l USING (id)")
            .await,
        600
    );
}

#[tokio::test]
async fn the_output_is_a_document_with_the_meaning_of_its_columns() {
    let fixture = Fixture::new().await;
    scoped(fixture.run(&request())).await;
    let (owner, described) = fixture
        .writer
        .run(|db| {
            Ok((
                db.table_owner("tickets_triage")?,
                db.describe_table("tickets_triage")?,
            ))
        })
        .await
        .unwrap();
    let owner = owner.unwrap();
    assert_eq!(owner.source, DocumentSource::Classify);
    assert_eq!(owner.ingested_by.as_deref(), Some("user"));
    assert_eq!(
        fixture
            .count("SELECT count(*) FROM _quack_classifications WHERE started_by = 'user'")
            .await,
        1
    );
    assert_eq!(owner.tables, Some(vec![String::from("tickets_triage")]));
    assert_eq!(owner.title.as_deref(), Some("tickets labelled by triage"));
    let meaning = |column: &str| {
        described
            .columns
            .iter()
            .find(|c| c.name == column)
            .and_then(|c| c.meaning.as_ref())
            .and_then(|m| m.description.clone())
            .unwrap_or_default()
    };
    assert!(meaning("department").starts_with("Which department should handle this ticket? One of billing, technical (Bugs, outages), none"));
    assert_eq!(
        meaning("department_p"),
        "probability of the chosen department"
    );
    assert!(meaning("urgency").ends_with("expected level, 0 Not urgent, 1 Soon, 2 Blocking"));
    assert_eq!(meaning("urgency_level"), "most probable level of urgency");
    assert_eq!(meaning("id"), "the key of tickets; join on it");
    assert_eq!(
        meaning("truncated"),
        "the row's text was cut to fit the model"
    );
    assert_eq!(
        described.labelled_by.map(|r| r.status),
        Some(RunStatus::Completed)
    );
}

#[tokio::test]
async fn a_rerun_labels_only_the_keys_the_output_lacks() {
    let fixture = Fixture::new().await;
    scoped(async {
        fixture.run(&request()).await;
        fixture
            .sql("INSERT INTO tickets SELECT 600 + range, 'billing t' || (600 + range), 'x' FROM range(10)")
            .await;
        let before = fixture.stub.requests();
        let run = fixture.run(&request()).await;
        assert_eq!(run.labelled, 10);
        assert_eq!(fixture.stub.requests() - before, 12, "two probes and ten rows");
    })
    .await;
    assert_eq!(
        fixture.count("SELECT count(*) FROM tickets_triage").await,
        610
    );
}

#[tokio::test]
async fn labelling_every_row_again_replaces_the_labels_when_it_completes() {
    let fixture = Fixture::new().await;
    scoped(async {
        let first = fixture.run(&request()).await;
        fixture
            .sql("UPDATE tickets SET subject = 'zzz' WHERE id = 0")
            .await;
        let before = fixture.stub.requests();
        let all = fixture
            .run(&Classification {
                rows: Rows::All,
                ..request()
            })
            .await;
        assert_eq!(all.rows, Rows::All);
        assert_eq!(all.status, RunStatus::Completed);
        assert_eq!(fixture.stub.requests() - before, 602);
        assert_eq!(all.document_id, first.document_id, "the document stays");
    })
    .await;
    assert_eq!(
        fixture
            .count("SELECT count(*) FROM tickets_triage WHERE id = 0 AND department = 'none'")
            .await,
        1
    );
    assert_eq!(
        fixture.count("SELECT count(*) FROM tickets_triage").await,
        600
    );
    assert_eq!(
        fixture
            .count("SELECT count(*) FROM duckdb_tables() WHERE starts_with(table_name, '_quack_stage_')")
            .await,
        0
    );
    assert_eq!(
        fixture
            .count("SELECT count(*) FROM duckdb_constraints() WHERE table_name = 'tickets_triage' AND constraint_type = 'PRIMARY KEY'")
            .await,
        1,
        "the swapped-in table keeps the key"
    );
}

#[tokio::test]
async fn a_cancelled_run_that_labels_everything_again_leaves_the_old_labels() {
    let fixture = Fixture::new().await;
    let token = CancellationToken::new();
    let firing = token.clone();
    let progress = move |_: ChunkDone| firing.cancel();
    scoped(async {
        let first = fixture.run(&request()).await;
        fixture
            .sql("UPDATE tickets SET subject = 'zzz' WHERE id = 0")
            .await;
        let stopped = fixture
            .try_run(
                &Classification {
                    rows: Rows::All,
                    ..request()
                },
                RunControl {
                    progress: &progress,
                    cancel: Some(&token),
                },
                Waiting::Job,
            )
            .await;
        assert!(matches!(stopped, Err(Error::Cancelled)), "{stopped:?}");
        assert_eq!(
            fixture
                .count(
                    "SELECT count(*) FROM tickets_triage WHERE id = 0 AND department = 'technical'"
                )
                .await,
            1,
            "the old label serves"
        );
        let runs = fixture.runs().await;
        assert_eq!(
            runs.first().map(|r| (r.status, r.rows)),
            Some((RunStatus::Cancelled, Rows::All))
        );
        assert_eq!(
            runs.first().map(|r| (r.labelled, r.cut, r.empty)),
            Some((0, 0, 0)),
            "the labels of a discarded stage are not counted"
        );
        let in_force = fixture
            .writer
            .run(|db| ClassificationRun::in_force(db, "tickets_triage"))
            .await
            .unwrap()
            .unwrap();
        assert_eq!(
            in_force.id, first.id,
            "a staged run that did not complete is not in force"
        );
    })
    .await;
    assert_eq!(
        fixture
            .count("SELECT count(*) FROM duckdb_tables() WHERE starts_with(table_name, '_quack_stage_')")
            .await,
        0
    );
}

#[tokio::test]
async fn a_cancel_keeps_the_rows_answered_and_a_rerun_finishes() {
    let fixture = Fixture::new().await;
    let token = CancellationToken::new();
    let firing = token.clone();
    let progress = move |_: ChunkDone| firing.cancel();
    scoped(async {
        let stopped = fixture
            .try_run(
                &request(),
                RunControl {
                    progress: &progress,
                    cancel: Some(&token),
                },
                Waiting::Job,
            )
            .await;
        assert!(matches!(stopped, Err(Error::Cancelled)), "{stopped:?}");
        assert_eq!(
            fixture.count("SELECT count(*) FROM tickets_triage").await,
            256
        );
        let runs = fixture.runs().await;
        assert_eq!(
            runs.first().map(|r| (r.status, r.labelled)),
            Some((RunStatus::Cancelled, 256))
        );
        let profiled = fixture
            .writer
            .run(|db| TableProfile::current(db, "tickets_triage", 256))
            .await
            .unwrap();
        assert!(profiled.is_some(), "a stopped run refreshes the profile");
        let rerun = fixture.run(&request()).await;
        assert_eq!(rerun.labelled, 344);
    })
    .await;
    assert_eq!(
        fixture.count("SELECT count(*) FROM tickets_triage").await,
        600
    );
}

#[tokio::test]
async fn other_questions_or_weights_need_everything_labelled_again() {
    let fixture = Fixture::new().await;
    scoped(Box::pin(async {
        let first = fixture.run(&request()).await;
        let mut changed = request();
        changed.question_set = serde_json::from_value(json!({
            "name": "triage",
            "questions": {"churn": {"type": "noul", "instructions": "Will they leave?"}}
        }))
        .unwrap();
        let ClassifyError::DefinitionChanged { differs, .. } = fixture.refusal(&changed).await
        else {
            panic!("expected a changed definition");
        };
        assert_eq!(differs, ["questions"]);

        fixture.stub.set_digest("sha256:bbbb");
        let ClassifyError::DefinitionChanged { differs, .. } = fixture.refusal(&request()).await
        else {
            panic!("expected a changed definition");
        };
        assert_eq!(differs, ["model weights"]);
        let refused = fixture.refusal(&request()).await.to_string();
        assert!(refused.contains("--all"), "{refused}");

        let swapped = fixture
            .run(&Classification {
                rows: Rows::All,
                ..changed.clone()
            })
            .await;
        assert_eq!(swapped.status, RunStatus::Completed);
        assert_eq!(
            fixture.count("SELECT count(*) FROM tickets_triage").await,
            600
        );

        // A deleted output is not compared against the definition it had.
        let document = swapped.document_id;
        fixture
            .writer
            .run(move |db| db.delete_document(&document))
            .await
            .unwrap();
        fixture.sql("UPDATE tickets SET body = 'again'").await;
        let fresh = fixture.run(&request()).await;
        assert_eq!(fresh.labelled, 600);
        assert_ne!(fresh.document_id, first.document_id);
    }))
    .await;
}

#[tokio::test]
async fn a_first_run_that_asked_for_everything_is_recorded_as_a_first_labelling() {
    let fixture = Fixture::new().await;
    let token = CancellationToken::new();
    let firing = token.clone();
    let progress = move |_: ChunkDone| firing.cancel();
    scoped(async {
        let stopped = fixture
            .try_run(
                &Classification {
                    rows: Rows::All,
                    ..request()
                },
                RunControl {
                    progress: &progress,
                    cancel: Some(&token),
                },
                Waiting::Job,
            )
            .await;
        assert!(matches!(stopped, Err(Error::Cancelled)));
        let runs = fixture.runs().await;
        assert_eq!(runs.first().map(|r| r.rows), Some(Rows::Missing));
        let mut changed = request();
        changed.question_set = serde_json::from_value(json!({
            "name": "triage",
            "questions": {"churn": {"type": "noul", "instructions": "Will they leave?"}}
        }))
        .unwrap();
        assert!(matches!(
            fixture.refusal(&changed).await,
            ClassifyError::DefinitionChanged { .. }
        ));
    })
    .await;
}

#[tokio::test]
async fn a_set_named_in_another_case_labels_into_the_same_table() {
    let fixture = Fixture::new().await;
    scoped(async {
        fixture.run(&request()).await;
        let before = fixture.stub.requests();
        let run = fixture
            .run(&Classification {
                question_set: named("Triage"),
                ..request()
            })
            .await;
        assert_eq!(run.output_table, "tickets_triage");
        assert_eq!(run.labelled, 0);
        assert_eq!(fixture.stub.requests() - before, 2, "only the probes");

        let claim = fixture
            .writer
            .claim(Claimed::Classify(String::from("tickets_triage")));
        assert!(claim.is_some());
        let refused = fixture
            .refusal(&Classification {
                question_set: named("TRIAGE"),
                ..request()
            })
            .await;
        assert!(
            matches!(refused, ClassifyError::Running { .. }),
            "{refused}"
        );
    })
    .await;
    assert_eq!(
        fixture
            .count("SELECT count(*) FROM duckdb_tables() WHERE table_name ILIKE 'tickets_triage'")
            .await,
        1
    );
}

#[tokio::test]
async fn an_output_that_lost_its_key_is_refused_until_everything_is_labelled_again() {
    let fixture = Fixture::new().await;
    scoped(async {
        fixture.run(&request()).await;
        fixture
            .sql("CREATE OR REPLACE TABLE tickets_triage AS SELECT * FROM tickets_triage")
            .await;
        assert!(matches!(
            fixture.refusal(&request()).await,
            ClassifyError::KeyLost { .. }
        ));
        let all = fixture
            .run(&Classification {
                rows: Rows::All,
                ..request()
            })
            .await;
        assert_eq!(all.status, RunStatus::Completed);
        let relabelled = fixture.run(&request()).await;
        assert_eq!(relabelled.labelled, 0);
    })
    .await;
}

#[tokio::test]
async fn a_table_without_an_id_column_names_the_columns_that_could_serve() {
    let fixture = Fixture::new().await;
    fixture
        .sql("CREATE TABLE notes AS SELECT 'row ' || range AS subject, 'x' AS body, range * 10 AS ref FROM range(5)")
        .await;
    scoped(async {
        let request = Classification {
            table: String::from("notes"),
            text_columns: vec![String::from("subject")],
            ..request()
        };
        let said = fixture.refusal(&request).await.to_string();
        assert!(said.contains("named id, or ending in _id or Id"), "{said}");
        let ClassifyError::NoKey {
            unique, closest, ..
        } = fixture.refusal(&request).await
        else {
            panic!("expected no key");
        };
        assert_eq!(unique, ["subject", "ref"]);
        assert_eq!(
            closest.first().map(|c| (c.column.as_str(), c.distinct)),
            Some(("body", 1))
        );

        let chosen = fixture
            .run(&Classification {
                key: Some(String::from("REF")),
                ..request.clone()
            })
            .await;
        assert_eq!((chosen.key_column.as_str(), chosen.labelled), ("ref", 5));

        let ClassifyError::KeyNotUnique { rows, distinct, .. } = fixture
            .refusal(&Classification {
                key: Some(String::from("body")),
                ..request
            })
            .await
        else {
            panic!("expected a key that is not unique");
        };
        assert_eq!((rows, distinct), (5, 1));
    })
    .await;
}

#[tokio::test]
async fn a_key_with_a_missing_value_is_not_a_key() {
    let fixture = Fixture::new().await;
    fixture
        .sql("CREATE TABLE gaps AS SELECT CASE WHEN range = 2 THEN NULL ELSE range END AS id, 'x' AS body FROM range(4)")
        .await;
    scoped(async {
        let request = Classification {
            table: String::from("gaps"),
            text_columns: vec![String::from("body")],
            key: Some(String::from("id")),
            ..request()
        };
        let ClassifyError::KeyNotUnique { missing, .. } = fixture.refusal(&request).await else {
            panic!("expected a key with a missing value");
        };
        assert_eq!(missing, 1);
    })
    .await;
}

#[tokio::test]
async fn names_quack_keeps_and_columns_that_do_not_exist_are_refused() {
    let fixture = Fixture::new().await;
    fixture
        .sql("CREATE TABLE graph_things AS SELECT 1 AS id, 'x' AS body")
        .await;
    scoped(async {
        for table in ["graph_things", "_quack_documents"] {
            let refused = fixture
                .try_run(
                    &Classification {
                        table: table.to_owned(),
                        ..request()
                    },
                    RunControl::unobserved(),
                    Waiting::Job,
                )
                .await;
            assert!(
                matches!(&refused, Err(Error::Ingestion(m)) if m.contains("reserves")),
                "{refused:?}"
            );
        }
        assert!(matches!(
            fixture
                .refusal(&Classification {
                    table: String::from("nothing"),
                    ..request()
                })
                .await,
            ClassifyError::NoTable(_)
        ));
        assert!(matches!(
            fixture
                .refusal(&Classification {
                    text_columns: vec![String::from("nope")],
                    ..request()
                })
                .await,
            ClassifyError::NoColumn { .. }
        ));
        assert!(matches!(
            fixture
                .refusal(&Classification {
                    text_columns: Vec::new(),
                    ..request()
                })
                .await,
            ClassifyError::TextColumns(0)
        ));
        let nine: Vec<String> = (0..9).map(|n| format!("c{n}")).collect();
        assert!(matches!(
            fixture
                .refusal(&Classification {
                    text_columns: nine,
                    ..request()
                })
                .await,
            ClassifyError::TextColumns(9)
        ));
        assert!(matches!(
            fixture
                .refusal(&Classification {
                    text_columns: vec![String::from("subject"), String::from("SUBJECT")],
                    ..request()
                })
                .await,
            ClassifyError::DuplicateColumn(_)
        ));
    })
    .await;
    assert!(fixture.runs().await.is_empty(), "a refusal records nothing");
}

#[tokio::test]
async fn a_source_named_with_a_dot_is_found_by_its_exact_name() {
    let fixture = Fixture::new().await;
    fixture
        .sql(r#"CREATE TABLE "orders.v2" AS SELECT 1 AS id, 'billing' AS body"#)
        .await;
    scoped(async {
        let run = fixture
            .run(&Classification {
                table: String::from("ORDERS.V2"),
                text_columns: vec![String::from("body")],
                ..request()
            })
            .await;
        assert_eq!(
            (run.source_table.as_str(), run.output_table.as_str()),
            ("orders.v2", "orders_v2_triage")
        );
    })
    .await;
}

#[tokio::test]
async fn a_preview_labels_rows_and_writes_nothing() {
    let fixture = Fixture::new().await;
    scoped(async {
        let decision = model(&fixture.stub).await;
        let preview = request()
            .preview(
                &fixture.writer,
                &decision,
                5,
                Waiting::Job,
                RunControl::unobserved(),
            )
            .await
            .unwrap();
        assert_eq!(preview.result.rows.len(), 5);
        assert_eq!(
            preview.result.columns,
            [
                "id",
                "department",
                "department_p",
                "department_confidence",
                "urgency",
                "urgency_level",
                "churn",
                "truncated"
            ]
        );
        assert_eq!(
            preview.result.rows.first().and_then(|r| r.first()),
            Some(&json!("0"))
        );
        assert_eq!(
            preview.result.rows.first().and_then(|r| r.get(1)),
            Some(&json!("technical"))
        );
        assert_eq!(
            (
                preview.remaining,
                preview.labelled,
                preview.key_column.as_str()
            ),
            (600, 5, "id")
        );
        assert!(
            preview
                .to_string()
                .contains("Run without --preview to label 600 rows.")
        );
        assert_eq!(fixture.stub.requests(), 7, "two probes and five rows");
    })
    .await;
    assert!(fixture.runs().await.is_empty());
    let tables = fixture.writer.run(WorkspaceDb::list_tables).await.unwrap();
    assert_eq!(tables, ["tickets"]);
    let documents = fixture.count("SELECT count(*) FROM _quack_documents").await;
    assert_eq!(documents, 0);
    let claim = fixture
        .writer
        .claim(Claimed::Classify(String::from("tickets_triage")));
    assert!(claim.is_some(), "a preview holds no claim");
}

#[tokio::test]
async fn a_second_run_into_the_same_output_is_refused_while_the_first_holds_it() {
    let fixture = Fixture::new().await;
    let _held = fixture
        .writer
        .claim(Claimed::Classify(String::from("tickets_triage")))
        .unwrap();
    scoped(async {
        let refused = fixture.refusal(&request()).await;
        assert!(
            matches!(&refused, ClassifyError::Running { table } if table == "tickets_triage"),
            "{refused}"
        );
    })
    .await;
    assert_eq!(fixture.stub.requests(), 0, "nothing was asked");
}

#[tokio::test]
async fn a_key_the_output_holds_already_is_left_alone_and_not_counted() {
    let fixture = Fixture::new().await;
    scoped(fixture.run(&Classification {
        rows: Rows::Missing,
        ..request()
    }))
    .await;
    let page = |key: &str| PageWrite {
        rows: vec![(
            key.to_owned(),
            vec![duckdb::types::Value::Null; 7],
            store::Written::Empty,
        )],
        skipped: 0,
    };
    let reader = fixture
        .writer
        .run(WorkspaceDb::try_clone_reader)
        .await
        .unwrap();
    let plan = Arc::new(Plan::resolve(&reader, &request()).unwrap());
    let run = fixture.runs().await.remove(0).id;
    let added = fixture
        .writer
        .run(move |db| {
            let twice = page("1").write(db, &plan, &run, "tickets_triage")?;
            let fresh = page("9999").write(db, &plan, &run, "tickets_triage")?;
            Ok((twice, fresh))
        })
        .await
        .unwrap();
    assert_eq!(added, (0, 1));
}

#[tokio::test]
async fn keys_of_every_supported_type_read_back_and_join() {
    let fixture = Fixture::new().await;
    let keys = [
        ("INTEGER", "n"),
        ("VARCHAR", "'k' || n"),
        ("DATE", "DATE '2026-01-01' + CAST(n AS INTEGER)"),
        (
            "TIMESTAMP",
            "TIMESTAMP '2026-01-01 00:00:00' + to_hours(CAST(n AS INTEGER))",
        ),
        (
            "TIMESTAMPTZ",
            "CASE n WHEN 0 THEN TIMESTAMPTZ '2026-01-01 00:00:00+00' \
             WHEN 1 THEN TIMESTAMPTZ '2026-01-01 01:30:00+00' \
             ELSE TIMESTAMPTZ '2026-06-01 12:00:00.123456+00' END",
        ),
        ("DECIMAL(18,4)", "n + 0.0001"),
        (
            "UUID",
            "CAST(printf('00000000-0000-0000-0000-%012d', n) AS UUID)",
        ),
    ];
    scoped(async {
        for (at, (ty, expr)) in keys.into_iter().enumerate() {
            let table = format!("keyed{at}");
            let create = format!(
                "CREATE TABLE {table} AS SELECT CAST({expr} AS {ty}) AS id, 'billing' AS body FROM range(3) t(n)"
            );
            fixture.writer.run(move |db| db.execute_statement(&create)).await.unwrap();
            let run = fixture
                .run(&Classification {
                    table: table.clone(),
                    text_columns: vec![String::from("body")],
                    ..request()
                })
                .await;
            assert_eq!(run.labelled, 3, "{ty}");
            let join = format!("SELECT count(*) FROM {table} s JOIN {table}_triage l USING (id)");
            let joined: i64 = fixture
                .writer
                .run(move |db| Ok(db.connection().query_row(&join, [], |r| r.get(0))?))
                .await
                .unwrap();
            assert_eq!(joined, 3, "{ty}");
            let rerun = fixture
                .run(&Classification {
                    table,
                    text_columns: vec![String::from("body")],
                    ..request()
                })
                .await;
            assert_eq!(rerun.labelled, 0, "{ty}: the labelled keys are found again");
        }
    })
    .await;
}

#[tokio::test]
async fn rows_without_text_are_written_with_null_labels_and_never_sent() {
    let fixture = Fixture::new().await;
    fixture
        .sql("INSERT INTO tickets VALUES (900, NULL, '  '), (901, 'a long billing text', NULL)")
        .await;
    scoped(async {
        let run = fixture.run(&request()).await;
        assert_eq!((run.labelled, run.empty), (601, 1));
    })
    .await;
    assert_eq!(fixture.stub.requests(), 2 + 601);
    assert_eq!(
        fixture
            .count("SELECT count(*) FROM tickets_triage WHERE id = 900 AND department IS NULL AND churn IS NULL AND NOT truncated")
            .await,
        1
    );
}

#[tokio::test]
async fn text_the_model_must_be_given_in_part_is_counted_as_cut() {
    let fixture = Fixture::with(
        DecisionStub::with_rule(|seen| {
            (seen.chars > 1000).then(|| Fault::new(400, "state is too long"))
        })
        .await,
    );
    fixture
        .sql("UPDATE tickets SET body = repeat('word ', 800) WHERE id < 3")
        .await;
    scoped(async {
        let run = fixture.run(&request()).await;
        assert_eq!((run.labelled, run.cut), (600, 3));
    })
    .await;
    assert_eq!(
        fixture
            .count("SELECT count(*) FROM tickets_triage WHERE truncated")
            .await,
        3
    );
}

#[tokio::test]
async fn rows_the_model_refuses_are_skipped_and_tried_again_by_the_next_run() {
    let allow = Arc::new(AtomicBool::new(false));
    let gate = Arc::clone(&allow);
    let fixture = Fixture::with(
        DecisionStub::with_rule(move |seen| {
            (!gate.load(Ordering::SeqCst) && seen.text.contains("zzz"))
                .then(|| Fault::new(400, "state has too many tokens"))
        })
        .await,
    );
    fixture
        .sql("UPDATE tickets SET subject = 'zzz', body = repeat('a', 3000) WHERE id = 7")
        .await;
    scoped(async {
        let run = fixture.run(&request()).await;
        assert_eq!((run.labelled, run.skipped), (599, 1));
        assert!(
            run.to_string()
                .contains("1 skipped (the model refused their text at every length")
        );
        assert_eq!(
            fixture
                .count("SELECT count(*) FROM tickets_triage WHERE id = 7")
                .await,
            0
        );
        allow.store(true, Ordering::SeqCst);
        let again = fixture.run(&request()).await;
        assert_eq!((again.labelled, again.skipped), (1, 0));
    })
    .await;
    assert_eq!(
        fixture.count("SELECT count(*) FROM tickets_triage").await,
        600
    );
}

#[tokio::test]
async fn a_refusal_in_the_middle_of_a_run_fails_it_and_keeps_the_rows_before() {
    let seen = Arc::new(AtomicUsize::new(0));
    let counter = Arc::clone(&seen);
    let fixture = Fixture::with(
        DecisionStub::with_rule(move |_| {
            (counter.fetch_add(1, Ordering::SeqCst) >= 102).then(|| Fault::new(403, "forbidden"))
        })
        .await,
    );
    scoped(async {
        let failed = fixture
            .try_run(&request(), RunControl::unobserved(), Waiting::Job)
            .await;
        assert!(
            matches!(&failed, Err(Error::DecisionRefused(m)) if m == "forbidden"),
            "{failed:?}"
        );
        let runs = fixture.runs().await;
        let run = runs.first().unwrap();
        assert_eq!((run.status, run.labelled), (RunStatus::Failed, 100));
        assert_eq!(
            run.error.as_deref(),
            Some("the decision model refused the request: forbidden")
        );
    })
    .await;
    assert_eq!(
        fixture.count("SELECT count(*) FROM tickets_triage").await,
        100
    );
}

#[tokio::test]
async fn a_question_set_the_model_refuses_creates_nothing() {
    let fixture = Fixture::with(
        DecisionStub::with_rule(|_| {
            Some(Fault::new(
                400,
                "decision options exceed the 176-token budget",
            ))
        })
        .await,
    );
    scoped(async {
        let refused = fixture
            .try_run(&request(), RunControl::unobserved(), Waiting::Job)
            .await;
        assert!(
            matches!(refused, Err(Error::DecisionRefused(_))),
            "{refused:?}"
        );
    })
    .await;
    assert!(fixture.runs().await.is_empty());
    assert_eq!(
        fixture.writer.run(WorkspaceDb::list_tables).await.unwrap(),
        ["tickets"]
    );
}

#[tokio::test]
async fn a_caller_that_waits_is_refused_a_run_past_its_budget() {
    let fixture = Fixture::new().await;
    scoped(async {
        let refused = fixture
            .try_run(
                &request(),
                RunControl::unobserved(),
                Waiting::Caller { budget: 1500 },
            )
            .await;
        let Err(Error::Classify(ClassifyError::TooLargeToWait {
            rows,
            questions,
            budget,
            ..
        })) = refused
        else {
            panic!("expected a run too large to wait for, got {refused:?}");
        };
        assert_eq!((rows, questions, budget), (600, 3, 1500));
        assert_eq!(
            fixture.stub.requests(),
            0,
            "refused before the model was asked"
        );
        let fits = fixture
            .try_run(
                &request(),
                RunControl::unobserved(),
                Waiting::Caller { budget: 1800 },
            )
            .await
            .unwrap();
        assert_eq!(fits.labelled, 600);
    })
    .await;
}

#[tokio::test]
async fn the_key_of_a_table_of_labels_keeps_the_type_of_the_sources_key() {
    let fixture = Fixture::new().await;
    scoped(fixture.run(&request())).await;
    let refused = fixture
        .writer
        .run(|db| {
            Retype {
                table: "tickets_triage",
                column: "id",
                to: ColumnType::Varchar,
            }
            .run(db)
        })
        .await;
    assert!(
        matches!(&refused, Err(Error::Analysis(m)) if m.contains("keeps the type of the key in 'tickets'")),
        "{refused:?}"
    );
}

#[tokio::test]
async fn a_table_of_labels_is_not_replaced_by_a_file() {
    let fixture = Fixture::new().await;
    let run = scoped(fixture.run(&request())).await;
    let refused = fixture
        .writer
        .run(move |db| db.begin_replacement(&run.document_id, &DocumentId::generate()))
        .await;
    assert!(
        matches!(refused, Err(Error::Classify(ClassifyError::ReplaceRefused))),
        "{refused:?}"
    );
}

#[tokio::test]
async fn a_table_someone_else_made_is_not_overwritten() {
    let fixture = Fixture::new().await;
    fixture.sql("CREATE TABLE tickets_triage (x INTEGER)").await;
    scoped(async {
        assert!(matches!(
            fixture.refusal(&request()).await,
            ClassifyError::OutputTaken { .. }
        ));
    })
    .await;
}

#[tokio::test]
async fn two_columns_with_one_name_are_refused() {
    let fixture = Fixture::new().await;
    scoped(async {
        let mut clashing = request();
        clashing.question_set = serde_json::from_value(json!({
            "name": "triage",
            "questions": {
                "dept": {"type": "choice", "instructions": "Which?", "criteria": {"a": null, "b": null}},
                "DEPT_P": {"type": "noul", "instructions": "Is it?"}
            }
        }))
        .unwrap();
        assert!(matches!(fixture.refusal(&clashing).await, ClassifyError::ColumnClash(name) if name == "DEPT_P"));
        clashing.question_set = serde_json::from_value(json!({
            "name": "triage",
            "questions": {"Truncated": {"type": "noul", "instructions": "Is it?"}}
        }))
        .unwrap();
        assert!(matches!(fixture.refusal(&clashing).await, ClassifyError::ColumnClash(_)));
    })
    .await;
}

#[test]
fn every_question_type_fills_the_columns_it_names() {
    let columns = columns::OutputColumns::new("tickets", "id", &triage().questions).unwrap();
    assert_eq!(columns.column_names().len(), 8);
    assert_eq!(columns.cells(None).len(), 7, "every column after the key");
    assert_eq!(columns.meanings().len(), 8);
}

#[test]
fn a_set_name_follows_the_rule_of_a_question_name() {
    assert!(SetName::try_from(String::from("triage_v2")).is_ok());
    assert!(SetName::try_from(String::from("2triage")).is_err());
    assert!(SetName::try_from(String::from("a b")).is_err());
}

#[test]
fn the_question_file_of_the_issue_reads_and_unknown_keys_are_refused() {
    let file = r#"{"name": "ticket_triage", "questions": {
        "department": {"type": "choice", "instructions": "Which department should handle this ticket?",
                       "criteria": {"billing": "Invoices, payments, refunds", "none": null}}}}"#;
    assert!(serde_json::from_str::<QuestionSet>(file).is_ok());
    let misspelled = file.replace("\"name\"", "\"title\"");
    assert!(serde_json::from_str::<QuestionSet>(&misspelled).is_err());
}

#[tokio::test]
async fn a_rerun_labels_every_key_the_output_lacks_wherever_it_sorts() {
    let fixture = Fixture::new().await;
    scoped(async {
        fixture.run(&request()).await;
        fixture
            .sql("DELETE FROM tickets_triage WHERE id IN (100, 150)")
            .await;
        fixture
            .sql("INSERT INTO tickets SELECT -1 - range, 'billing t', 'x' FROM range(10)")
            .await;
        let before = fixture.stub.requests();
        let run = fixture.run(&request()).await;
        assert_eq!(run.labelled, 12);
        assert_eq!(
            fixture.stub.requests() - before,
            14,
            "two probes and twelve rows"
        );
    })
    .await;
    assert_eq!(
        fixture.count("SELECT count(*) FROM tickets_triage").await,
        610
    );
}

#[tokio::test]
async fn keys_that_cross_pages_are_labelled_once_each() {
    let fixture = Fixture::new().await;
    let keys = [
        (
            "TIMESTAMPTZ",
            "CAST(strftime(TIMESTAMP '2026-01-01 00:00:00' + to_seconds(n), '%Y-%m-%d %H:%M:%S') \
             || '.123456+00' AS TIMESTAMPTZ)",
        ),
        ("DECIMAL(18,4)", "CAST(n AS DECIMAL(18,4)) + 0.0001"),
        (
            "UUID",
            "CAST(printf('00000000-0000-0000-0000-%012d', n) AS UUID)",
        ),
    ];
    scoped(async {
        for (at, (ty, expr)) in keys.into_iter().enumerate() {
            let table = format!("paged{at}");
            let create = format!(
                "CREATE TABLE {table} AS SELECT CAST({expr} AS {ty}) AS id, 'billing' AS body \
                 FROM range(600) t(n)"
            );
            fixture
                .writer
                .run(move |db| db.execute_statement(&create))
                .await
                .unwrap();
            let request = Classification {
                table: table.clone(),
                text_columns: vec![String::from("body")],
                ..request()
            };
            let before = fixture.stub.requests();
            let run = fixture.run(&request).await;
            assert_eq!(run.labelled, 600, "{ty}");
            assert_eq!(
                fixture.stub.requests() - before,
                602,
                "{ty}: every key once"
            );
            let join = format!("SELECT count(*) FROM {table} s JOIN {table}_triage l USING (id)");
            let joined: i64 = fixture
                .writer
                .run(move |db| Ok(db.connection().query_row(&join, [], |r| r.get(0))?))
                .await
                .unwrap();
            assert_eq!(joined, 600, "{ty}");
            assert_eq!(
                fixture.run(&request).await.labelled,
                0,
                "{ty}: a rerun finds them"
            );
        }
    })
    .await;
}

#[tokio::test]
async fn a_table_that_takes_a_deleted_outputs_name_gets_none_of_its_meanings() {
    let fixture = Fixture::new().await;
    scoped(async {
        let run = fixture.run(&request()).await;
        let document = run.document_id;
        fixture
            .writer
            .run(move |db| db.delete_document(&document))
            .await
            .unwrap();
        fixture
            .sql("CREATE TABLE tickets_triage AS SELECT 1 AS id, 'x' AS department")
            .await;
        let described = fixture
            .writer
            .run(|db| db.describe_table("tickets_triage"))
            .await
            .unwrap();
        assert!(described.labelled_by.is_none());
        assert!(described.columns.iter().all(|c| c.meaning.is_none()));
        let retyped = fixture
            .writer
            .run(|db| {
                Retype {
                    table: "tickets_triage",
                    column: "id",
                    to: ColumnType::Varchar,
                }
                .run(db)
            })
            .await;
        assert!(retyped.is_ok(), "{retyped:?}");
    })
    .await;
}

#[tokio::test]
async fn a_table_that_appears_before_the_run_starts_is_not_dropped() {
    let fixture = Fixture::new().await;
    let reader = fixture
        .writer
        .run(WorkspaceDb::try_clone_reader)
        .await
        .unwrap();
    let plan = Plan::resolve(&reader, &request()).unwrap();
    fixture
        .sql("CREATE TABLE tickets_triage AS SELECT 7 AS x")
        .await;
    let run = RunId::generate();
    let claims = fixture.writer.claims();
    let refused = fixture
        .writer
        .run(move |db| {
            Start {
                plan: &plan,
                run_id: &run,
                model: "m",
                digest: "d",
                started_by: None,
                claims: &claims,
            }
            .apply(db)
        })
        .await;
    assert!(
        matches!(
            &refused,
            Err(Error::Classify(ClassifyError::OutputTaken { .. }))
        ),
        "{refused:?}"
    );
    assert_eq!(fixture.count("SELECT x FROM tickets_triage").await, 7);
    assert_eq!(
        fixture.count("SELECT count(*) FROM _quack_documents").await,
        0
    );
    assert!(fixture.runs().await.is_empty());
}

/// A run that labels everything again, started and with its stage written,
/// ready to end.
async fn staged(fixture: &Fixture) -> (Plan, RunId) {
    scoped(fixture.run(&request())).await;
    let reader = fixture
        .writer
        .run(WorkspaceDb::try_clone_reader)
        .await
        .unwrap();
    let plan = Plan::resolve(
        &reader,
        &Classification {
            rows: Rows::All,
            ..request()
        },
    )
    .unwrap();
    let (run, started, owned) = (RunId::generate(), plan.clone(), plan.clone());
    let id = run.clone();
    let claims = fixture.writer.claims();
    fixture
        .writer
        .run(move |db| {
            Start {
                plan: &started,
                run_id: &id,
                model: "m",
                digest: "sha256:aaaa",
                started_by: None,
                claims: &claims,
            }
            .apply(db)
        })
        .await
        .unwrap();
    fixture
        .sql("INSERT INTO _quack_stage_tickets_triage (id) VALUES (-1)")
        .await;
    (owned, run)
}

#[tokio::test]
async fn the_swap_leaves_a_table_that_took_the_outputs_name_alone() {
    let fixture = Fixture::new().await;
    let (plan, run) = staged(&fixture).await;
    // Another document now owns the name.
    fixture
        .sql("UPDATE _quack_documents SET source = 'upload' WHERE filename = 'tickets_triage'")
        .await;
    let ended = fixture
        .writer
        .run(move |db| Ended::Completed.record(db, &plan, &run))
        .await;
    assert!(
        matches!(
            &ended,
            Err(Error::Classify(ClassifyError::OutputTaken { .. }))
        ),
        "{ended:?}"
    );
    assert_eq!(
        fixture.count("SELECT count(*) FROM tickets_triage").await,
        600
    );
    assert_eq!(
        fixture
            .count("SELECT count(*) FROM duckdb_tables() WHERE starts_with(table_name, '_quack_stage_')")
            .await,
        0
    );
    let runs = fixture.runs().await;
    assert_eq!(
        runs.first().map(|r| (r.status, r.labelled)),
        Some((RunStatus::Failed, 0))
    );
}

#[tokio::test]
async fn the_swap_installs_the_stage_when_the_output_is_gone() {
    let fixture = Fixture::new().await;
    let (plan, run) = staged(&fixture).await;
    fixture.sql("DROP TABLE tickets_triage").await;
    let record = fixture
        .writer
        .run(move |db| Ended::Completed.record(db, &plan, &run))
        .await
        .unwrap();
    assert_eq!(record.status, RunStatus::Completed);
    assert_eq!(
        fixture.count("SELECT count(*) FROM tickets_triage").await,
        1
    );
    assert_eq!(
        fixture
            .count("SELECT count(*) FROM duckdb_constraints() WHERE table_name = 'tickets_triage' AND constraint_type = 'PRIMARY KEY'")
            .await,
        1
    );
    assert_eq!(
        fixture
            .count("SELECT count(*) FROM duckdb_tables() WHERE starts_with(table_name, '_quack_stage_')")
            .await,
        0
    );
}

#[tokio::test]
async fn a_leftover_stage_goes_with_the_next_run_and_with_the_document() {
    let fixture = Fixture::new().await;
    scoped(async {
        let run = fixture.run(&request()).await;
        let stages =
            "SELECT count(*) FROM duckdb_tables() WHERE starts_with(table_name, '_quack_stage_')";
        fixture
            .sql("CREATE TABLE _quack_stage_tickets_triage AS SELECT 1 AS x")
            .await;
        assert_eq!(fixture.count(stages).await, 1);
        fixture.run(&request()).await;
        assert_eq!(fixture.count(stages).await, 0, "the next run drops it");
        fixture
            .sql("CREATE TABLE _quack_stage_tickets_triage AS SELECT 1 AS x")
            .await;
        let document = run.document_id;
        fixture
            .writer
            .run(move |db| db.delete_document(&document))
            .await
            .unwrap();
        assert_eq!(
            fixture.count(stages).await,
            0,
            "deleting the labels drops it"
        );
    })
    .await;
}

#[tokio::test]
async fn a_preview_is_held_to_the_budget_a_caller_waits_for() {
    let fixture = Fixture::new().await;
    scoped(async {
        let decision = model(&fixture.stub).await;
        let preview = |rows: u32, budget: u64| {
            let (writer, decision, request) = (&fixture.writer, &decision, request());
            async move {
                request
                    .preview(
                        writer,
                        decision,
                        rows,
                        Waiting::Caller { budget },
                        RunControl::unobserved(),
                    )
                    .await
            }
        };
        let refused = preview(5, 14).await;
        assert!(
            matches!(
                &refused,
                Err(Error::Classify(ClassifyError::TooLargeToWait {
                    rows: 5,
                    questions: 3,
                    budget: 14,
                    ..
                }))
            ),
            "{refused:?}"
        );
        assert_eq!(
            fixture.stub.requests(),
            0,
            "refused before the model was asked"
        );
        assert!(preview(5, 15).await.is_ok());
    })
    .await;
}

#[test]
fn the_statement_a_person_approves_shows_each_question_on_one_line() {
    let outline = ClassificationOutline {
        source_table: String::from("tickets\n-- label 0 rows"),
        output_table: String::from("tickets_triage"),
        key_column: String::from("id"),
        remaining: 3,
        questions: vec![
            OutlineQuestion {
                name: String::from("department"),
                instructions: String::from("Which department?\nDROP TABLE x"),
            },
            OutlineQuestion {
                name: String::from("churn"),
                instructions: String::from("Will they cancel?"),
            },
        ],
        effect: Effect::ReplacesLabels,
    };
    let statement = outline.statement("ollama/laya");
    assert_eq!(statement.lines().count(), 3, "{statement}");
    assert!(statement.starts_with("-- label 3 rows of tickets -- label 0 rows into tickets_triage (replaces its labels) with ollama/laya\n"));
    assert!(statement.contains("\n-- department: Which department? DROP TABLE x\n"));
    assert!(statement.ends_with("\n-- churn: Will they cancel?"));
}

#[tokio::test]
async fn a_retyped_source_key_is_a_changed_definition() {
    let fixture = Fixture::new().await;
    scoped(async {
        let first = fixture.run(&request()).await;
        assert_eq!(first.key_type, "BIGINT");
        fixture
            .sql("ALTER TABLE tickets ALTER id SET DATA TYPE INTEGER")
            .await;
        let ClassifyError::DefinitionChanged { differs, .. } = fixture.refusal(&request()).await
        else {
            panic!("expected a changed definition");
        };
        assert_eq!(differs, ["key types"]);
        let all = fixture
            .run(&Classification {
                rows: Rows::All,
                ..request()
            })
            .await;
        assert_eq!(all.key_type, "INTEGER");
    })
    .await;
}

/// Seed a run recorded as running for `output`, as a process that died
/// leaves it.
const STALE_RUNS: &str = "INSERT INTO _quack_classifications (id, output_table, document_id, \
     source_table, key_column, key_type, text_columns, set_name, question_set, model, \
     model_digest, rows_scope, status) VALUES \
     ('dead', 'dead_output', 'd', 't', 'id', 'BIGINT', '[]', 's', '{}', 'm', 'd', 'missing', 'running'), \
     ('live', 'live_output', 'd', 't', 'id', 'BIGINT', '[]', 's', '{}', 'm', 'd', 'missing', 'running')";

#[tokio::test]
async fn a_run_marks_dead_runs_interrupted_and_spares_one_another_run_holds() {
    let fixture = Fixture::new().await;
    fixture.sql(STALE_RUNS).await;
    let live = fixture
        .writer
        .claim(Claimed::Classify(String::from("live_output")));
    assert!(live.is_some());
    scoped(fixture.run(&request())).await;
    let status = |id: &'static str| {
        let fixture = &fixture;
        async move {
            fixture
                .writer
                .run(move |db| {
                    Ok(db.connection().query_row(
                        "SELECT status FROM _quack_classifications WHERE id = ?",
                        [id],
                        |row| row.get::<_, String>(0),
                    )?)
                })
                .await
                .unwrap()
        }
    };
    assert_eq!(status("dead").await, "interrupted");
    assert_eq!(status("live").await, "running");
}

#[tokio::test]
async fn an_output_another_table_took_since_planning_is_not_written_into() {
    let fixture = Fixture::new().await;
    let first = scoped(fixture.run(&request())).await;
    let reader = fixture
        .writer
        .run(WorkspaceDb::try_clone_reader)
        .await
        .unwrap();
    let plan = Plan::resolve(&reader, &request()).unwrap();
    assert!(plan.output_exists);
    let document = first.document_id;
    fixture
        .writer
        .run(move |db| db.delete_document(&document))
        .await
        .unwrap();
    fixture
        .sql("CREATE TABLE tickets_triage AS SELECT 7 AS id")
        .await;
    let run = RunId::generate();
    let claims = fixture.writer.claims();
    let refused = fixture
        .writer
        .run(move |db| {
            Start {
                plan: &plan,
                run_id: &run,
                model: "m",
                digest: "sha256:aaaa",
                started_by: None,
                claims: &claims,
            }
            .apply(db)
        })
        .await;
    assert!(
        matches!(
            &refused,
            Err(Error::Classify(ClassifyError::OutputTaken { .. }))
        ),
        "{refused:?}"
    );
    assert_eq!(
        fixture.count("SELECT count(*) FROM tickets_triage").await,
        1,
        "the other table is untouched"
    );
}

#[test]
fn a_question_set_is_read_from_text_within_the_cap() {
    let text = serde_json::to_string(&triage()).unwrap();
    assert_eq!(QuestionSet::parse(&text).unwrap(), triage());
    assert!(matches!(
        QuestionSet::parse("{"),
        Err(QuestionSetError::Unreadable(_))
    ));
    assert!(matches!(
        QuestionSet::parse(r#"{"name": "x", "questions": {}, "extra": 1}"#),
        Err(QuestionSetError::Unreadable(_))
    ));
    let padded = format!("{text}{}", " ".repeat(MAX_QUESTION_SET_BYTES));
    assert!(matches!(
        QuestionSet::parse(&padded),
        Err(QuestionSetError::SetTooLarge { bytes, max })
            if bytes == padded.len() && max == MAX_QUESTION_SET_BYTES
    ));
}

/// The runs a hook was told about.
type Reported = Arc<Mutex<Vec<RunEnded>>>;

fn recording() -> (OnEnd, Reported) {
    let ended: Reported = Arc::default();
    let seen = Arc::clone(&ended);
    let hook = OnEnd::new(move |run| {
        let seen = Arc::clone(&seen);
        async move { seen.lock().unwrap().push(run) }
    });
    (hook, ended)
}

/// A run for the user `user-1` of the labels the stub serves.
async fn tracked_run(fixture: &Fixture, jobs: LabelJobs) -> (Tracked, RunId) {
    let run_id = RunId::generate();
    let tracked = Tracked {
        db: Arc::clone(&fixture.writer),
        decision: scoped(model(&fixture.stub)).await,
        started_by: Some(UserId::from("user-1")),
        run_id: run_id.clone(),
        waiting: Waiting::Job,
        cancel: CancellationToken::new(),
        jobs,
    };
    (tracked, run_id)
}

/// The only element of `items`.
fn only<T: Clone>(items: &[T]) -> T {
    let [item] = items else {
        panic!("expected one, got {}", items.len());
    };
    item.clone()
}

#[tokio::test]
async fn a_run_as_a_job_is_listed_returned_and_reported() {
    let fixture = Fixture::new().await;
    let queue = JobQueue::new(10);
    let (hook, ended) = recording();
    let jobs = LabelJobs::new(queue.clone())
        .workspace(WorkspaceId::from("ws"))
        .on_end(hook);
    let (tracked, run_id) = tracked_run(&fixture, jobs).await;
    let run = scoped(request().run_as_job(tracked, |_| {})).await.unwrap();
    assert_eq!(run.id, run_id);
    assert_eq!((run.status, run.labelled), (RunStatus::Completed, 600));
    let job = only(&queue.list());
    assert_eq!(job.kind, JobKind::Classify);
    assert_eq!(job.state, JobState::Succeeded);
    assert_eq!(job.workspace_id, Some(WorkspaceId::from("ws")));
    assert_eq!(job.owner, Some(UserId::from("user-1")));
    let reported = only(&ended.lock().unwrap());
    assert_eq!(reported.outcome, Outcome::Allowed);
    assert_eq!(reported.run.id, run.id);
}

#[tokio::test]
async fn a_run_whose_caller_is_dropped_goes_on_to_its_end_and_is_reported() {
    let fixture = Fixture::new().await;
    let queue = JobQueue::new(10);
    let (hook, ended) = recording();
    let (tracked, _) = tracked_run(&fixture, LabelJobs::new(queue.clone()).on_end(hook)).await;
    scoped(async {
        let waiting = request().run_as_job(tracked, |_| {});
        tokio::select! {
            biased;
            _ = waiting => panic!("the run cannot have ended yet"),
            () = tokio::task::yield_now() => {}
        }
    })
    .await;
    let job = only(&queue.list());
    let finished = queue.wait(job.id).await.unwrap();
    assert_eq!(finished.state, JobState::Succeeded, "{finished:?}");
    let reported = only(&ended.lock().unwrap());
    assert_eq!(
        (reported.run.status, reported.run.labelled),
        (RunStatus::Completed, 600)
    );
}

#[tokio::test]
async fn the_queues_shutdown_cancels_a_run_and_waits_for_it_to_record() {
    let fixture = Fixture::new().await;
    let queue = JobQueue::new(10);
    let (hook, ended) = recording();
    let (tracked, _) = tracked_run(&fixture, LabelJobs::new(queue.clone()).on_end(hook)).await;
    let page = Arc::new(tokio::sync::Notify::new());
    let first_page = Arc::clone(&page);
    let caller = tokio::spawn(Egress::scope(
        Some(Egress::NoWorkspace),
        request().run_as_job(tracked, move |_| first_page.notify_one()),
    ));
    page.notified().await;
    let left = queue.shutdown(Duration::from_secs(30)).await;
    assert!(left.is_empty(), "{left:?}");
    let result = caller.await.unwrap();
    assert!(matches!(result, Err(Error::Cancelled)), "{result:?}");
    let reported = only(&ended.lock().unwrap());
    assert_eq!(reported.run.status, RunStatus::Cancelled);
    assert_eq!(reported.outcome, Outcome::Error);
}

#[tokio::test]
async fn a_failed_run_is_reported_with_its_rows_and_a_refused_request_is_not() {
    let seen = Arc::new(AtomicUsize::new(0));
    let counter = Arc::clone(&seen);
    let fixture = Fixture::with(
        DecisionStub::with_rule(move |_| {
            (counter.fetch_add(1, Ordering::SeqCst) >= 102).then(|| Fault::new(403, "forbidden"))
        })
        .await,
    );
    let queue = JobQueue::new(10);
    let (hook, ended) = recording();
    let jobs = LabelJobs::new(queue.clone()).on_end(hook);

    let (tracked, _) = tracked_run(&fixture, jobs.clone()).await;
    let failed = scoped(request().run_as_job(tracked, |_| {})).await;
    assert!(
        matches!(failed, Err(Error::DecisionRefused(_))),
        "{failed:?}"
    );

    let (tracked, never_begun) = tracked_run(&fixture, jobs).await;
    let missing = Classification {
        table: String::from("nothing"),
        ..request()
    };
    let refused = scoped(missing.run_as_job(tracked, |_| {})).await;
    assert!(
        matches!(refused, Err(Error::Classify(ClassifyError::NoTable(_)))),
        "{refused:?}"
    );
    let recorded = fixture
        .writer
        .run(move |db| ClassificationRun::get(db, &never_begun))
        .await;
    assert!(recorded.is_err(), "a refused request records no run");

    let reported = only(&ended.lock().unwrap());
    assert_eq!(
        (reported.run.status, reported.run.labelled, reported.outcome),
        (RunStatus::Failed, 100, Outcome::Error),
        "only the run that began is reported"
    );
}

/// Start a run for `plan` in one writer step, as a run does once it holds
/// its claim.
async fn start(fixture: &Fixture, plan: Plan) {
    let (id, claims) = (RunId::generate(), fixture.writer.claims());
    fixture
        .writer
        .run(move |db| {
            Start {
                plan: &plan,
                run_id: &id,
                model: "m",
                digest: "sha256:aaaa",
                started_by: None,
                claims: &claims,
            }
            .apply(db)
        })
        .await
        .unwrap();
}

#[tokio::test]
async fn a_run_that_started_and_claimed_before_another_starts_is_not_marked_interrupted() {
    let fixture = Fixture::new().await;
    let reader = fixture
        .writer
        .run(WorkspaceDb::try_clone_reader)
        .await
        .unwrap();
    let first = Plan::resolve(&reader, &request()).unwrap();
    let second = Plan::resolve(
        &reader,
        &Classification {
            question_set: named("other"),
            ..request()
        },
    )
    .unwrap();
    let held_first = fixture.writer.claim(first.claimed());
    let held_second = fixture.writer.claim(second.claimed());
    assert!(held_first.is_some() && held_second.is_some());
    // The second run claims and starts; then the first run's start reads
    // the claims in its own writer step and finds the second there.
    start(&fixture, second).await;
    start(&fixture, first).await;
    assert_eq!(
        fixture
            .count("SELECT count(*) FROM _quack_classifications WHERE status = 'running'")
            .await,
        2
    );
}
