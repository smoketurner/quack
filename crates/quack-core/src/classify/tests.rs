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
use crate::error::Error as CoreError;
use crate::extraction::ExtractFuture;
use crate::ids::DocumentId;
use crate::ingestion::TableName;
use crate::jobs::JobState;
use crate::llm::decision::QuestionSetError;
use crate::llm::decision::fixture::{model, scoped};
use crate::llm::decision::stub::{DecisionStub, Fault};
use crate::llm::egress::Egress;
use crate::progress::ChunkDone;
use crate::storage::profile::{ColumnType, Retype, TableProfile};
use crate::storage::workspace::{DocumentSource, WorkspaceDb};
use crate::storage::writer::Claimed;
use plan::Plan;
use store::{Ended, PageWrite, Start};

fn questions() -> Questions {
    serde_json::from_value(json!({
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
    }))
    .unwrap()
}

fn other_questions() -> Questions {
    serde_json::from_value(json!({
        "churn": {"type": "noul", "instructions": "Will they leave?"}
    }))
    .unwrap()
}

/// What a test labels: a table, the columns the model reads, a key (the
/// `id` column unless named), and the questions.
#[derive(Clone)]
struct Classification {
    table: String,
    text_columns: Vec<String>,
    key: Option<String>,
    questions: Questions,
    rows: Rows,
}

impl Classification {
    fn set(&self) -> LabelSet {
        let key = self.key.clone().unwrap_or_else(|| String::from("id"));
        LabelSet {
            key_reason: KeyReason::of(&key),
            key_column: key,
            text_columns: self.text_columns.clone(),
            questions: self.questions.clone(),
            sentence: Some(String::from("which department, and how urgent")),
        }
    }

    fn draft(&self) -> Draft {
        Draft {
            table: self.table.clone(),
            output_table: String::new(),
            set: self.set(),
            rows: 0,
            sample_rows: 0,
            origin: DraftOrigin::Given,
            drafted_by_model: Some(String::from("test/chat")),
            approved_at: None,
            columns_gone: Vec::new(),
        }
    }
}

fn request() -> Classification {
    Classification {
        table: String::from("tickets"),
        text_columns: vec![String::from("subject"), String::from("body")],
        key: None,
        questions: questions(),
        rows: Rows::Missing,
    }
}

/// `request` checked against the workspace, labelling the rows it asks for.
fn plan_of(db: &WorkspaceDb, request: &Classification) -> Plan {
    let plan = Plan::resolve(db, &request.table, &request.set()).unwrap();
    let because = (request.rows == Rows::All).then_some(RelabelReason::Asked);
    plan.labelling(db, request.rows, because).unwrap()
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
    ) -> Result<Run> {
        let decision = model(&self.stub).await;
        request
            .draft()
            .run(
                Labelling {
                    db: &self.writer,
                    decision: &decision,
                    started_by: Some("user"),
                    run_id: RunId::generate(),
                    waiting,
                    control,
                },
                request.rows,
            )
            .await
    }

    async fn outline(&self, request: &Classification) -> Outline {
        let decision = model(&self.stub).await;
        request
            .draft()
            .outline(&self.writer, &decision, request.rows)
            .await
            .unwrap()
    }

    async fn run(&self, request: &Classification) -> Run {
        self.try_run(request, RunControl::unobserved(), Waiting::Job)
            .await
            .unwrap()
    }

    async fn refusal(&self, request: &Classification) -> Error {
        match self
            .try_run(request, RunControl::unobserved(), Waiting::Job)
            .await
        {
            Err(CoreError::Classify(refusal)) => refusal,
            other => panic!("expected a classify refusal, got {other:?}"),
        }
    }

    async fn runs(&self) -> Vec<Run> {
        self.writer.run(|db| Run::list(db, 100)).await.unwrap().runs
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
        assert_eq!(run.output_table, "tickets_labels");
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
        fixture.count("SELECT count(*) FROM tickets_labels").await,
        600
    );
    assert_eq!(
        fixture
            .count("SELECT count(*) FROM tickets_labels WHERE department = 'technical'")
            .await,
        200
    );
    assert_eq!(
        fixture
            .count("SELECT count(*) FROM tickets_labels WHERE urgency_level = 0 AND churn < 0.5")
            .await,
        600
    );
    assert_eq!(
        fixture
            .count("SELECT count(*) FROM tickets t JOIN tickets_labels l USING (id)")
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
                db.table_owner("tickets_labels")?,
                db.describe_table("tickets_labels")?,
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
    assert_eq!(owner.tables, Some(vec![String::from("tickets_labels")]));
    assert_eq!(owner.title.as_deref(), Some("tickets labels"));
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
        fixture.count("SELECT count(*) FROM tickets_labels").await,
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
            .count("SELECT count(*) FROM tickets_labels WHERE id = 0 AND department = 'none'")
            .await,
        1
    );
    assert_eq!(
        fixture.count("SELECT count(*) FROM tickets_labels").await,
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
            .count("SELECT count(*) FROM duckdb_constraints() WHERE table_name = 'tickets_labels' AND constraint_type = 'PRIMARY KEY'")
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
        assert!(matches!(stopped, Err(CoreError::Cancelled)), "{stopped:?}");
        assert_eq!(
            fixture
                .count(
                    "SELECT count(*) FROM tickets_labels WHERE id = 0 AND department = 'technical'"
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
            .run(|db| Run::in_force(db, "tickets_labels"))
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
        assert!(matches!(stopped, Err(CoreError::Cancelled)), "{stopped:?}");
        assert_eq!(
            fixture.count("SELECT count(*) FROM tickets_labels").await,
            256
        );
        let runs = fixture.runs().await;
        assert_eq!(
            runs.first().map(|r| (r.status, r.labelled)),
            Some((RunStatus::Cancelled, 256))
        );
        let profiled = fixture
            .writer
            .run(|db| TableProfile::current(db, "tickets_labels", 256))
            .await
            .unwrap();
        assert!(profiled.is_some(), "a stopped run refreshes the profile");
        let rerun = fixture.run(&request()).await;
        assert_eq!(rerun.labelled, 344);
    })
    .await;
    assert_eq!(
        fixture.count("SELECT count(*) FROM tickets_labels").await,
        600
    );
}

#[tokio::test]
async fn other_questions_or_weights_label_every_row_again() {
    let fixture = Fixture::new().await;
    scoped(Box::pin(async {
        let first = fixture.run(&request()).await;
        assert_eq!(
            fixture.outline(&request()).await.effect,
            Effect::AddsRows,
            "the same questions add rows"
        );
        let changed = Classification {
            questions: other_questions(),
            ..request()
        };
        let outline = fixture.outline(&changed).await;
        assert_eq!(
            outline.effect,
            Effect::ReplacesLabels {
                because: RelabelReason::QuestionsChanged
            }
        );
        assert_eq!(outline.remaining, 600);

        fixture.stub.set_digest("sha256:bbbb");
        assert_eq!(
            fixture.outline(&request()).await.effect,
            Effect::ReplacesLabels {
                because: RelabelReason::ModelChanged
            }
        );

        let swapped = fixture.run(&changed).await;
        assert_eq!(swapped.status, RunStatus::Completed);
        assert_eq!(swapped.rows, Rows::All, "the run labels every row again");
        assert_eq!(
            fixture.count("SELECT count(*) FROM tickets_labels").await,
            600
        );
        assert_eq!(
            fixture
                .count("SELECT count(*) FROM duckdb_columns() WHERE table_name = 'tickets_labels' AND column_name = 'department'")
                .await,
            0,
            "the labels of the old questions are gone"
        );

        // A deleted output is not compared against the set it had.
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
        assert!(matches!(stopped, Err(CoreError::Cancelled)));
        let runs = fixture.runs().await;
        assert_eq!(runs.first().map(|r| r.rows), Some(Rows::Missing));
        let changed = Classification {
            questions: other_questions(),
            ..request()
        };
        assert_eq!(
            fixture.outline(&changed).await.effect,
            Effect::ReplacesLabels {
                because: RelabelReason::QuestionsChanged
            },
            "later questions are compared against the first run's"
        );
    })
    .await;
}

#[tokio::test]
async fn an_output_that_lost_its_key_is_refused_until_everything_is_labelled_again() {
    let fixture = Fixture::new().await;
    scoped(async {
        fixture.run(&request()).await;
        fixture
            .sql("CREATE OR REPLACE TABLE tickets_labels AS SELECT * FROM tickets_labels")
            .await;
        assert!(matches!(
            fixture.refusal(&request()).await,
            Error::KeyLost { .. }
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
async fn a_table_without_an_id_column_falls_back_to_a_unique_number_then_short_text() {
    let fixture = Fixture::new().await;
    fixture
        .sql("CREATE TABLE notes AS SELECT 'row ' || range AS subject, 'x' AS body, range * 10 AS ref FROM range(5)")
        .await;
    fixture
        .sql("CREATE TABLE both_ids AS SELECT range AS ref, range + 100 AS ticket_id FROM range(5)")
        .await;
    fixture
        .sql("CREATE TABLE words AS SELECT 'row ' || range AS subject, 'x' AS body FROM range(5)")
        .await;
    fixture
        .sql("CREATE TABLE floaty AS SELECT range * 1.5 AS score, 'x' AS body FROM range(5)")
        .await;
    fixture
        .sql("CREATE TABLE essays AS SELECT repeat('long ', 40) || range AS text, 'x' AS body FROM range(5)")
        .await;
    let key_of = |table: &'static str| {
        let fixture = &fixture;
        async move {
            fixture
                .writer
                .run(move |db| {
                    let source = TableName::exact(&db.list_tables()?, table)?;
                    let columns = db.describe_columns(source.as_str())?;
                    KeyColumn::resolve(db, &source, &columns).map(|key| key.name)
                })
                .await
        }
    };
    assert_eq!(
        key_of("notes").await.unwrap(),
        "ref",
        "a number before text"
    );
    assert_eq!(
        key_of("both_ids").await.unwrap(),
        "ticket_id",
        "an id-like name before a number"
    );
    assert_eq!(key_of("words").await.unwrap(), "subject", "short text");
    for table in ["floaty", "essays"] {
        let refused = key_of(table).await;
        let Err(CoreError::Classify(Error::NoKey {
            table: named,
            unique,
            ..
        })) = refused
        else {
            panic!("{table}: expected no key, got {refused:?}");
        };
        assert_eq!(named, table);
        assert_eq!(unique.len(), 1, "{table}: the unique column is named");
    }
    let said = key_of("floaty").await.unwrap_err().to_string();
    assert!(said.contains("row_number()"), "{said}");
    assert!(
        said.contains("Unique columns of a type that cannot be a key: score"),
        "{said}"
    );
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
        let Error::KeyNotUnique { missing, .. } = fixture.refusal(&request).await else {
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
    fixture
        .sql("CREATE TABLE wide AS SELECT range AS id, 'a' AS c0, 'a' AS c1, 'a' AS c2, 'a' AS c3, 'a' AS c4, 'a' AS c5, 'a' AS c6, 'a' AS c7, 'a' AS c8 FROM range(3)")
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
                matches!(&refused, Err(CoreError::Ingestion(m)) if m.contains("reserves")),
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
            Error::NoTable(_)
        ));
        let Error::SetColumnsGone { columns, .. } = fixture
            .refusal(&Classification {
                text_columns: vec![String::from("nope")],
                ..request()
            })
            .await
        else {
            panic!("expected columns that are gone");
        };
        assert_eq!(columns, ["nope"]);
        let Error::SetColumnsGone { columns, .. } = fixture
            .refusal(&Classification {
                key: Some(String::from("missing_key")),
                ..request()
            })
            .await
        else {
            panic!("expected a key that is gone");
        };
        assert_eq!(columns, ["missing_key"]);
        assert!(matches!(
            fixture
                .refusal(&Classification {
                    text_columns: Vec::new(),
                    ..request()
                })
                .await,
            Error::NoText { .. }
        ));
        let nine: Vec<String> = (0..9).map(|n| format!("c{n}")).collect();
        assert!(matches!(
            fixture
                .refusal(&Classification {
                    table: String::from("wide"),
                    text_columns: nine,
                    ..request()
                })
                .await,
            Error::TextColumns(9)
        ));
        let twice = fixture
            .outline(&Classification {
                text_columns: vec![String::from("subject"), String::from("SUBJECT")],
                ..request()
            })
            .await;
        assert_eq!(
            twice.text_columns,
            ["subject"],
            "a repeated column is read once"
        );
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
            ("orders.v2", "orders_v2_labels")
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
            .draft()
            .preview(
                &fixture.writer,
                &decision,
                5,
                Rows::Missing,
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
        assert!(preview.to_string().starts_with("5 rows in "), "{preview}");
        assert_eq!(
            preview.compact.columns,
            ["id", "department (p)", "urgency", "churn"]
        );
        assert_eq!(
            preview.compact.rows.first(),
            Some(&vec![
                json!("0"),
                json!("technical (0.90)"),
                json!("0.00"),
                json!("0.10")
            ])
        );
        assert!(preview.seconds_for(600).is_some());
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
        .claim(Claimed(String::from("tickets_labels")));
    assert!(claim.is_some(), "a preview holds no claim");
}

#[tokio::test]
async fn a_second_run_into_the_same_output_is_refused_while_the_first_holds_it() {
    let fixture = Fixture::new().await;
    let _held = fixture
        .writer
        .claim(Claimed(String::from("tickets_labels")))
        .unwrap();
    scoped(async {
        let refused = fixture.refusal(&request()).await;
        assert!(
            matches!(&refused, Error::Running { table } if table == "tickets_labels"),
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
        ask_ms: 0,
    };
    let reader = fixture
        .writer
        .run(WorkspaceDb::try_clone_reader)
        .await
        .unwrap();
    let plan = Arc::new(plan_of(&reader, &request()));
    let run = fixture.runs().await.remove(0).id;
    let added = fixture
        .writer
        .run(move |db| {
            let twice = page("1").write(db, &plan, &run, "tickets_labels")?;
            let fresh = page("9999").write(db, &plan, &run, "tickets_labels")?;
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
            let join = format!("SELECT count(*) FROM {table} s JOIN {table}_labels l USING (id)");
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
            .count("SELECT count(*) FROM tickets_labels WHERE id = 900 AND department IS NULL AND churn IS NULL AND NOT truncated")
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
            .count("SELECT count(*) FROM tickets_labels WHERE truncated")
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
                .count("SELECT count(*) FROM tickets_labels WHERE id = 7")
                .await,
            0
        );
        allow.store(true, Ordering::SeqCst);
        let again = fixture.run(&request()).await;
        assert_eq!((again.labelled, again.skipped), (1, 0));
    })
    .await;
    assert_eq!(
        fixture.count("SELECT count(*) FROM tickets_labels").await,
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
            matches!(&failed, Err(CoreError::DecisionRefused(m)) if m == "forbidden"),
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
        fixture.count("SELECT count(*) FROM tickets_labels").await,
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
            matches!(refused, Err(CoreError::DecisionRefused(_))),
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
        let Err(CoreError::Classify(Error::TooLargeToWait {
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
                table: "tickets_labels",
                column: "id",
                to: ColumnType::Varchar,
            }
            .run(db)
        })
        .await;
    assert!(
        matches!(&refused, Err(CoreError::Analysis(m)) if m.contains("keeps the type of the key in 'tickets'")),
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
        matches!(refused, Err(CoreError::Classify(Error::ReplaceRefused))),
        "{refused:?}"
    );
}

#[tokio::test]
async fn a_table_someone_else_made_is_not_overwritten() {
    let fixture = Fixture::new().await;
    fixture.sql("CREATE TABLE tickets_labels (x INTEGER)").await;
    scoped(async {
        assert!(matches!(
            fixture.refusal(&request()).await,
            Error::OutputTaken { .. }
        ));
    })
    .await;
}

#[tokio::test]
async fn two_columns_with_one_name_are_refused() {
    let fixture = Fixture::new().await;
    scoped(async {
        let clashing = |questions: Value| Classification {
            questions: serde_json::from_value(questions).unwrap(),
            ..request()
        };
        let refused = fixture
            .refusal(&clashing(json!({
                "dept": {"type": "choice", "instructions": "Which?", "criteria": {"a": null, "b": null}},
                "DEPT_P": {"type": "noul", "instructions": "Is it?"}
            })))
            .await;
        assert!(matches!(refused, Error::ColumnClash(name) if name == "DEPT_P"));
        let refused = fixture
            .refusal(&clashing(json!({"Truncated": {"type": "noul", "instructions": "Is it?"}})))
            .await;
        assert!(matches!(refused, Error::ColumnClash(_)));
    })
    .await;
}

#[test]
fn every_question_type_fills_the_columns_it_names() {
    let columns = columns::OutputColumns::new("tickets", "id", &questions()).unwrap();
    assert_eq!(columns.column_names().len(), 8);
    assert_eq!(columns.cells(None).len(), 7, "every column after the key");
    assert_eq!(columns.meanings().len(), 8);
}

#[tokio::test]
async fn a_rerun_labels_every_key_the_output_lacks_wherever_it_sorts() {
    let fixture = Fixture::new().await;
    scoped(async {
        fixture.run(&request()).await;
        fixture
            .sql("DELETE FROM tickets_labels WHERE id IN (100, 150)")
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
        fixture.count("SELECT count(*) FROM tickets_labels").await,
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
            let join = format!("SELECT count(*) FROM {table} s JOIN {table}_labels l USING (id)");
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
            .sql("CREATE TABLE tickets_labels AS SELECT 1 AS id, 'x' AS department")
            .await;
        let described = fixture
            .writer
            .run(|db| db.describe_table("tickets_labels"))
            .await
            .unwrap();
        assert!(described.labelled_by.is_none());
        assert!(described.columns.iter().all(|c| c.meaning.is_none()));
        let retyped = fixture
            .writer
            .run(|db| {
                Retype {
                    table: "tickets_labels",
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
    let plan = plan_of(&reader, &request());
    fixture
        .sql("CREATE TABLE tickets_labels AS SELECT 7 AS x")
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
                drafted_by_model: None,
                claims: &claims,
            }
            .apply(db)
        })
        .await;
    assert!(
        matches!(
            &refused,
            Err(CoreError::Classify(Error::OutputTaken { .. }))
        ),
        "{refused:?}"
    );
    assert_eq!(fixture.count("SELECT x FROM tickets_labels").await, 7);
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
    let plan = plan_of(
        &reader,
        &Classification {
            rows: Rows::All,
            ..request()
        },
    );
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
                drafted_by_model: None,
                claims: &claims,
            }
            .apply(db)
        })
        .await
        .unwrap();
    fixture
        .sql("INSERT INTO _quack_stage_tickets_labels (id) VALUES (-1)")
        .await;
    (owned, run)
}

#[tokio::test]
async fn the_swap_leaves_a_table_that_took_the_outputs_name_alone() {
    let fixture = Fixture::new().await;
    let (plan, run) = staged(&fixture).await;
    // Another document now owns the name.
    fixture
        .sql("UPDATE _quack_documents SET source = 'upload' WHERE filename = 'tickets_labels'")
        .await;
    let ended = fixture
        .writer
        .run(move |db| Ended::Completed.record(db, &plan, &run))
        .await;
    assert!(
        matches!(&ended, Err(CoreError::Classify(Error::OutputTaken { .. }))),
        "{ended:?}"
    );
    assert_eq!(
        fixture.count("SELECT count(*) FROM tickets_labels").await,
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
    fixture.sql("DROP TABLE tickets_labels").await;
    let record = fixture
        .writer
        .run(move |db| Ended::Completed.record(db, &plan, &run))
        .await
        .unwrap();
    assert_eq!(record.status, RunStatus::Completed);
    assert_eq!(
        fixture.count("SELECT count(*) FROM tickets_labels").await,
        1
    );
    assert_eq!(
        fixture
            .count("SELECT count(*) FROM duckdb_constraints() WHERE table_name = 'tickets_labels' AND constraint_type = 'PRIMARY KEY'")
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
            .sql("CREATE TABLE _quack_stage_tickets_labels AS SELECT 1 AS x")
            .await;
        assert_eq!(fixture.count(stages).await, 1);
        fixture.run(&request()).await;
        assert_eq!(fixture.count(stages).await, 0, "the next run drops it");
        fixture
            .sql("CREATE TABLE _quack_stage_tickets_labels AS SELECT 1 AS x")
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
                    .draft()
                    .preview(
                        writer,
                        decision,
                        rows,
                        Rows::Missing,
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
                Err(CoreError::Classify(Error::TooLargeToWait {
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
    let outline = Outline {
        source_table: String::from("tickets\n-- label 0 rows"),
        output_table: String::from("tickets_labels"),
        key_column: String::from("id"),
        key_reason: KeyReason::Unique,
        text_columns: vec![String::from("subject"), String::from("body\nx")],
        remaining: 3,
        questions: vec![
            DraftQuestion {
                name: String::from("department"),
                kind: QuestionKind::Choice,
                options: vec![
                    DraftOption {
                        label: String::from("billing"),
                        description: String::new(),
                    },
                    DraftOption {
                        label: String::from("tech"),
                        description: String::new(),
                    },
                ],
                levels: Vec::new(),
                instructions: String::from("Which department?\nDROP TABLE x"),
            },
            DraftQuestion {
                name: String::from("churn"),
                kind: QuestionKind::Noul,
                options: Vec::new(),
                levels: Vec::new(),
                instructions: String::from("Will they cancel?"),
            },
        ],
        effect: Effect::ReplacesLabels {
            because: RelabelReason::QuestionsChanged,
        },
        estimate_seconds: Some(7300),
    };
    let statement = outline.statement("ollama/laya");
    assert_eq!(statement.lines().count(), 4, "{statement}");
    assert_eq!(
        statement,
        "-- label 3 rows of tickets -- label 0 rows into tickets_labels (replaces its labels: the \
         questions changed) with ollama/laya, 2 questions, about 2 hours\n\
         -- key id (all different; not named like an id); reads subject, body x\n\
         -- department (choice: billing, tech): Which department? DROP TABLE x\n\
         -- churn (yes/no): Will they cancel?"
    );
}

#[tokio::test]
async fn a_retyped_source_key_labels_every_row_again() {
    let fixture = Fixture::new().await;
    scoped(async {
        let first = fixture.run(&request()).await;
        assert_eq!(first.key_type, "BIGINT");
        fixture
            .sql("ALTER TABLE tickets ALTER id SET DATA TYPE INTEGER")
            .await;
        assert_eq!(
            fixture.outline(&request()).await.effect,
            Effect::ReplacesLabels {
                because: RelabelReason::KeyChanged
            }
        );
        let again = fixture.run(&request()).await;
        assert_eq!(
            (again.rows, again.key_type.as_str()),
            (Rows::All, "INTEGER")
        );
    })
    .await;
}

/// Seed a run recorded as running for `output`, as a process that died
/// leaves it.
const STALE_RUNS: &str = "INSERT INTO _quack_classifications (id, output_table, document_id, \
     source_table, key_column, key_type, key_reason, text_columns, questions, model, \
     model_digest, rows_scope, status) VALUES \
     ('dead', 'dead_output', 'd', 't', 'id', 'BIGINT', 'unique', '[]', '{}', 'm', 'd', 'missing', 'running'), \
     ('live', 'live_output', 'd', 't', 'id', 'BIGINT', 'unique', '[]', '{}', 'm', 'd', 'missing', 'running')";

#[tokio::test]
async fn a_run_marks_dead_runs_interrupted_and_spares_one_another_run_holds() {
    let fixture = Fixture::new().await;
    fixture.sql(STALE_RUNS).await;
    let live = fixture.writer.claim(Claimed(String::from("live_output")));
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
    let plan = plan_of(&reader, &request());
    assert!(plan.output_exists);
    let document = first.document_id;
    fixture
        .writer
        .run(move |db| db.delete_document(&document))
        .await
        .unwrap();
    fixture
        .sql("CREATE TABLE tickets_labels AS SELECT 7 AS id")
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
                drafted_by_model: None,
                claims: &claims,
            }
            .apply(db)
        })
        .await;
    assert!(
        matches!(
            &refused,
            Err(CoreError::Classify(Error::OutputTaken { .. }))
        ),
        "{refused:?}"
    );
    assert_eq!(
        fixture.count("SELECT count(*) FROM tickets_labels").await,
        1,
        "the other table is untouched"
    );
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
async fn tracked_run(fixture: &Fixture, jobs: LabelJobs) -> (LabellingJob, RunId) {
    let run_id = RunId::generate();
    let tracked = LabellingJob {
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
    let run = scoped(request().draft().run_as_job(tracked, Rows::Missing, |_| {}))
        .await
        .unwrap();
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
        let waiting = request().draft().run_as_job(tracked, Rows::Missing, |_| {});
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
        request()
            .draft()
            .run_as_job(tracked, Rows::Missing, move |_| first_page.notify_one()),
    ));
    page.notified().await;
    let left = queue.shutdown(Duration::from_secs(30)).await;
    assert!(left.is_empty(), "{left:?}");
    let result = caller.await.unwrap();
    assert!(matches!(result, Err(CoreError::Cancelled)), "{result:?}");
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
    let failed = scoped(request().draft().run_as_job(tracked, Rows::Missing, |_| {})).await;
    assert!(
        matches!(failed, Err(CoreError::DecisionRefused(_))),
        "{failed:?}"
    );

    let (tracked, never_begun) = tracked_run(&fixture, jobs).await;
    let missing = Classification {
        table: String::from("nothing"),
        ..request()
    };
    let refused = scoped(missing.draft().run_as_job(tracked, Rows::Missing, |_| {})).await;
    assert!(
        matches!(refused, Err(CoreError::Classify(Error::NoTable(_)))),
        "{refused:?}"
    );
    let recorded = fixture
        .writer
        .run(move |db| Run::get(db, &never_begun))
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
                drafted_by_model: None,
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
    fixture
        .sql("CREATE TABLE more AS SELECT * FROM tickets")
        .await;
    let reader = fixture
        .writer
        .run(WorkspaceDb::try_clone_reader)
        .await
        .unwrap();
    let first = plan_of(&reader, &request());
    let second = plan_of(
        &reader,
        &Classification {
            table: String::from("more"),
            ..request()
        },
    );
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

/// A table the chat model can draft questions about: a key, a subject, a
/// channel with three values, and a body.
const SUPPORT: &str = "CREATE TABLE support AS SELECT range AS id, 'ticket ' || range AS subject, \
     ['email', 'chat', 'phone'][1 + range % 3] AS channel, 'text ' || range || ' help' AS body \
     FROM range(50)";

/// A drafter that answers from a script and remembers what it was asked.
struct Canned {
    answers: Mutex<std::collections::VecDeque<DraftAnswer>>,
    asked: Mutex<Vec<(Vec<String>, String)>>,
}

impl Canned {
    fn new(answers: Vec<DraftAnswer>) -> Self {
        Self {
            answers: Mutex::new(answers.into()),
            asked: Mutex::default(),
        }
    }

    fn asked(&self) -> Vec<(Vec<String>, String)> {
        self.asked.lock().unwrap().clone()
    }

    /// The message of the `at`th call.
    fn message(&self, at: usize) -> String {
        self.asked().into_iter().nth(at).unwrap().1
    }
}

impl Drafter for Canned {
    fn answer<'a>(
        &'a self,
        candidates: &'a [String],
        message: &'a str,
    ) -> ExtractFuture<'a, DraftAnswer> {
        self.asked
            .lock()
            .unwrap()
            .push((candidates.to_vec(), message.to_owned()));
        let next = self.answers.lock().unwrap().pop_front();
        Box::pin(async move {
            next.ok_or_else(|| CoreError::Llm(String::from("no answer was scripted")))
        })
    }

    fn label(&self) -> String {
        String::from("test/chat")
    }
}

/// The chat model's answer: these text columns and these questions.
fn answer(columns: &[&str], questions: Value) -> DraftAnswer {
    DraftAnswer {
        text_columns: columns.iter().map(|c| (*c).to_owned()).collect(),
        questions: serde_json::from_value(questions).unwrap(),
    }
}

fn channel_question() -> Value {
    json!([
        {"name": "topic", "type": "choice", "instructions": "What is the ticket about?",
         "options": [{"label": "billing", "description": "Invoices"}, {"label": "other", "description": ""}],
         "levels": []},
        {"name": "angry", "type": "noul", "instructions": "Is the customer angry?",
         "options": [], "levels": []}
    ])
}

/// The draft of `sentence` for `table`.
async fn drafted(
    fixture: &Fixture,
    drafter: Option<&dyn Drafter>,
    table: &str,
    sentence: Option<&str>,
) -> Result<Draft> {
    let decision = scoped(model(&fixture.stub)).await;
    let ctx = DraftContext {
        db: &fixture.writer,
        decision: &decision,
        drafter,
    };
    scoped(Draft::prepare(&ctx, table, sentence)).await
}

/// Run `draft`, approving its set.
async fn approved(fixture: &Fixture, draft: &Draft) -> Run {
    let decision = scoped(model(&fixture.stub)).await;
    scoped(draft.run(
        Labelling {
            db: &fixture.writer,
            decision: &decision,
            started_by: Some("user"),
            run_id: RunId::generate(),
            waiting: Waiting::Job,
            control: RunControl::unobserved(),
        },
        Rows::Missing,
    ))
    .await
    .unwrap()
}

#[tokio::test]
async fn a_sentence_is_drafted_into_questions_and_nothing_is_stored() {
    let fixture = Fixture::new().await;
    fixture.sql(SUPPORT).await;
    let canned = Canned::new(vec![answer(&["subject", "body"], channel_question())]);
    let draft = drafted(
        &fixture,
        Some(&canned),
        "support",
        Some("what is each ticket about"),
    )
    .await
    .unwrap();
    assert_eq!(draft.origin, DraftOrigin::Drafted);
    assert_eq!(
        (
            draft.table.as_str(),
            draft.output_table.as_str(),
            draft.rows,
            draft.sample_rows
        ),
        ("support", "support_labels", 50, 20)
    );
    assert_eq!(draft.drafted_by_model.as_deref(), Some("test/chat"));
    assert_eq!(draft.set.key_column, "id");
    assert_eq!(draft.set.key_reason, KeyReason::IdLike);
    assert_eq!(draft.set.text_columns, ["subject", "body"]);
    assert_eq!(
        draft.set.sentence.as_deref(),
        Some("what is each ticket about")
    );
    assert_eq!(draft.set.questions.count(), 2);
    assert_eq!(
        draft.header(),
        "support: 50 rows, key id, text in subject and body."
    );
    assert_eq!(
        fixture.stub.requests(),
        2,
        "the decision model's two probes"
    );
    assert!(fixture.runs().await.is_empty());
    assert_eq!(
        fixture.writer.run(WorkspaceDb::list_tables).await.unwrap(),
        ["support", "tickets"]
    );
    assert_eq!(
        fixture.count("SELECT count(*) FROM _quack_documents").await,
        0
    );
}

#[tokio::test]
async fn the_same_sentence_reuses_the_approved_questions_and_another_revises_them() {
    let fixture = Fixture::new().await;
    fixture.sql(SUPPORT).await;
    let canned = Canned::new(vec![
        answer(&["subject"], channel_question()),
        answer(&["subject", "channel"], channel_question()),
    ]);
    let first = drafted(
        &fixture,
        Some(&canned),
        "support",
        Some("What is each ticket about?"),
    )
    .await
    .unwrap();
    let run = approved(&fixture, &first).await;
    assert_eq!(run.sentence.as_deref(), Some("What is each ticket about?"));
    assert_eq!(run.drafted_by_model.as_deref(), Some("test/chat"));

    for again in [Some("  what is each ticket ABOUT? "), None] {
        let reused = drafted(&fixture, Some(&canned), "support", again)
            .await
            .unwrap();
        assert_eq!(reused.origin, DraftOrigin::Reused);
        assert_eq!(reused.set, first.set);
        assert_eq!(reused.approved_at.as_deref(), Some(run.started_at.as_str()));
    }
    assert_eq!(canned.asked().len(), 1, "no chat call to reuse a set");

    let revised = drafted(
        &fixture,
        Some(&canned),
        "support",
        Some("and the channel too"),
    )
    .await
    .unwrap();
    assert_eq!(revised.origin, DraftOrigin::Revised);
    assert_eq!(revised.set.text_columns, ["subject", "channel"]);
    let asked = canned.asked();
    assert_eq!(asked.len(), 2);
    let message = &asked.last().unwrap().1;
    assert!(
        message.contains("Current questions, to revise: {\"text_columns\":[\"subject\"]"),
        "{message}"
    );
    assert!(message.contains("\"name\":\"topic\""), "{message}");
}

#[tokio::test]
async fn nothing_to_ask_and_nobody_to_ask_are_said() {
    let fixture = Fixture::new().await;
    fixture.sql(SUPPORT).await;
    let canned = Canned::new(Vec::new());
    let none = drafted(&fixture, Some(&canned), "support", None).await;
    assert!(
        matches!(&none, Err(CoreError::Classify(Error::NoQuestions { table })) if table == "support"),
        "{none:?}"
    );
    let blank = drafted(&fixture, Some(&canned), "support", Some("   ")).await;
    assert!(matches!(
        blank,
        Err(CoreError::Classify(Error::NoQuestions { .. }))
    ));
    let nobody = drafted(&fixture, None, "support", Some("what is it about")).await;
    assert!(
        matches!(&nobody, Err(CoreError::Classify(Error::NoDrafter { .. }))),
        "{nobody:?}"
    );
    assert!(canned.asked().is_empty());
    assert_eq!(fixture.stub.requests(), 0);
}

#[tokio::test]
async fn a_refused_answer_is_asked_again_once_with_the_reason() {
    let fixture = Fixture::new().await;
    fixture.sql(SUPPORT).await;
    let canned = Canned::new(vec![
        answer(&["nope"], channel_question()),
        answer(&["subject"], channel_question()),
    ]);
    let draft = drafted(&fixture, Some(&canned), "support", Some("what is it about"))
        .await
        .unwrap();
    assert_eq!(draft.set.text_columns, ["subject"]);
    let asked = canned.asked();
    assert_eq!(asked.len(), 2);
    assert!(!canned.message(0).contains("Your answer was refused"));
    assert!(
        canned.message(1).ends_with(
            "Your answer was refused: 'nope' is not one of the columns offered. Answer again \
             with that fixed."
        ),
        "{}",
        canned.message(1)
    );

    let one_option = json!([{"name": "x", "type": "choice", "instructions": "Which?",
        "options": [{"label": "only", "description": ""}], "levels": []}]);
    let canned = Canned::new(vec![
        answer(&["subject"], one_option.clone()),
        answer(&["subject"], one_option),
    ]);
    let refused = drafted(&fixture, Some(&canned), "support", Some("what is it about")).await;
    let Err(CoreError::Classify(Error::DraftRefused { table, reason })) = refused else {
        panic!("expected a draft refused twice, got {refused:?}");
    };
    assert_eq!(table, "support");
    assert!(reason.contains("has 1 options"), "{reason}");
    assert_eq!(canned.asked().len(), 2, "one more try, no more");
}

#[tokio::test]
async fn the_decision_models_refusal_is_fed_back_too() {
    let fixture = Fixture::with(
        DecisionStub::with_rule(|seen| {
            seen.questions
                .get("angry")
                .map(|_| Fault::new(400, "decision options exceed the token budget"))
        })
        .await,
    );
    fixture.sql(SUPPORT).await;
    let calm = json!([{"name": "topic", "type": "choice", "instructions": "About what?",
        "options": [{"label": "a", "description": ""}, {"label": "b", "description": ""}],
        "levels": []}]);
    let canned = Canned::new(vec![
        answer(&["subject"], channel_question()),
        answer(&["subject"], calm),
    ]);
    let draft = drafted(&fixture, Some(&canned), "support", Some("what is it about"))
        .await
        .unwrap();
    assert_eq!(draft.set.questions.count(), 1);
    assert!(
        canned
            .message(1)
            .contains("Your answer was refused: decision options exceed the token budget."),
        "{}",
        canned.message(1)
    );
}

#[tokio::test]
async fn the_chat_model_is_shown_the_columns_their_values_and_a_fenced_sample() {
    let fixture = Fixture::new().await;
    fixture.sql(SUPPORT).await;
    let canned = Canned::new(vec![answer(&["subject"], channel_question())]);
    drafted(
        &fixture,
        Some(&canned),
        "support",
        Some("what\nis it about"),
    )
    .await
    .unwrap();
    let (candidates, message) = canned.asked().remove(0);
    assert_eq!(
        candidates,
        ["subject", "channel", "body"],
        "table order, no key"
    );
    assert!(
        message.starts_with("Table: support, 50 rows. Key column: id.\n"),
        "{message}"
    );
    assert!(message.contains("\n- subject (50 distinct)\n"), "{message}");
    assert!(
        message.contains("\n- channel (3 distinct: chat, email, phone)\n"),
        "{message}"
    );
    assert!(
        message.contains("\nThe person wants to know: what is it about\n"),
        "{message}"
    );
    assert!(
        message.contains("\nSample rows (20 of 50), one JSON object per line:\n<<document "),
        "{message}"
    );
    let sample: Vec<&str> = message
        .lines()
        .filter(|line| line.starts_with('{') && line.ends_with('}'))
        .collect();
    assert_eq!(sample.len(), 20);
    assert!(sample.iter().map(|line| line.len()).sum::<usize>() <= 16_000);
    assert!(
        sample
            .iter()
            .all(|line| serde_json::from_str::<Value>(line).is_ok())
    );
    assert!(
        !message.contains("Current questions"),
        "a first draft has none"
    );
}

#[tokio::test]
async fn a_column_that_was_not_offered_is_refused_and_a_repeated_one_is_read_once() {
    let fixture = Fixture::new().await;
    fixture.sql(SUPPORT).await;
    let canned = Canned::new(vec![
        answer(&["id"], channel_question()),
        answer(&["Subject", "subject", "BODY"], channel_question()),
    ]);
    let draft = drafted(&fixture, Some(&canned), "support", Some("what is it about"))
        .await
        .unwrap();
    assert!(
        canned
            .message(1)
            .contains("'id' is not one of the columns offered")
    );
    assert_eq!(draft.set.text_columns, ["subject", "body"]);
}

#[tokio::test]
async fn a_table_with_no_text_to_read_is_said() {
    let fixture = Fixture::new().await;
    fixture
        .sql("CREATE TABLE numbers AS SELECT range AS id, range * 2 AS twice FROM range(10)")
        .await;
    let canned = Canned::new(Vec::new());
    let refused = drafted(&fixture, Some(&canned), "numbers", Some("what is it")).await;
    assert!(
        matches!(&refused, Err(CoreError::Classify(Error::NoText { table })) if table == "numbers"),
        "{refused:?}"
    );
    assert!(canned.asked().is_empty(), "the chat model was not asked");
}

#[tokio::test]
async fn a_set_sent_back_is_checked_and_its_key_reason_is_worked_out_again() {
    let fixture = Fixture::new().await;
    fixture.sql(SUPPORT).await;
    let decision = scoped(model(&fixture.stub)).await;
    let ctx = DraftContext {
        db: &fixture.writer,
        decision: &decision,
        drafter: None,
    };
    let set = LabelSet {
        key_column: String::from("SUBJECT"),
        key_reason: KeyReason::IdLike,
        text_columns: vec![String::from("body"), String::from("Channel")],
        questions: serde_json::from_value(json!({
            "topic": {"type": "noul", "instructions": "Is it about billing?"}
        }))
        .unwrap(),
        sentence: None,
    };
    let given = scoped(Draft::given(&ctx, "SUPPORT", set.clone()))
        .await
        .unwrap();
    assert_eq!(given.origin, DraftOrigin::Given);
    assert_eq!(given.table, "support");
    assert_eq!(
        given.set.key_column, "subject",
        "spelled as the table spells it"
    );
    assert_eq!(
        given.set.key_reason,
        KeyReason::Unique,
        "worked out, not believed"
    );
    assert_eq!(given.set.text_columns, ["body", "channel"]);
    assert_eq!(given.rows, 50);

    let refuse = |set: LabelSet| {
        let ctx = &ctx;
        async move {
            match scoped(Draft::given(ctx, "support", set)).await {
                Err(CoreError::Classify(refusal)) => refusal,
                other => panic!("expected a refusal, got {other:?}"),
            }
        }
    };
    let gone = refuse(LabelSet {
        text_columns: vec![String::from("body"), String::from("removed")],
        ..set.clone()
    })
    .await;
    assert!(
        matches!(&gone, Error::SetColumnsGone { columns, .. } if columns == &["removed"]),
        "{gone}"
    );
    let not_unique = refuse(LabelSet {
        key_column: String::from("channel"),
        ..set.clone()
    })
    .await;
    assert!(matches!(
        not_unique,
        Error::KeyNotUnique { distinct: 3, .. }
    ));
    let clash = refuse(LabelSet {
        questions: serde_json::from_value(json!({
            "dept": {"type": "choice", "instructions": "Which?", "criteria": {"a": null, "b": null}},
            "DEPT_P": {"type": "noul", "instructions": "Is it?"}
        }))
        .unwrap(),
        ..set.clone()
    })
    .await;
    assert!(matches!(clash, Error::ColumnClash(name) if name == "DEPT_P"));
}

#[tokio::test]
async fn approved_questions_whose_columns_are_gone_are_not_reused() {
    let fixture = Fixture::new().await;
    fixture.sql(SUPPORT).await;
    let canned = Canned::new(vec![answer(&["subject", "body"], channel_question())]);
    let draft = drafted(&fixture, Some(&canned), "support", Some("what is it about"))
        .await
        .unwrap();
    approved(&fixture, &draft).await;
    fixture.sql("ALTER TABLE support DROP COLUMN body").await;
    let refused = drafted(&fixture, Some(&canned), "support", None).await;
    let Err(CoreError::Classify(Error::SetColumnsGone { columns, .. })) = refused else {
        panic!("expected the columns to be gone, got {refused:?}");
    };
    assert_eq!(columns, ["body"]);
}

#[tokio::test]
async fn a_cancelled_relabel_leaves_its_questions_to_be_the_next_run() {
    let fixture = Fixture::new().await;
    let token = CancellationToken::new();
    let firing = token.clone();
    let progress = move |_: ChunkDone| firing.cancel();
    scoped(async {
        fixture.run(&request()).await;
        let changed = Classification {
            questions: other_questions(),
            rows: Rows::All,
            ..request()
        };
        let stopped = fixture
            .try_run(
                &changed,
                RunControl {
                    progress: &progress,
                    cancel: Some(&token),
                },
                Waiting::Job,
            )
            .await;
        assert!(matches!(stopped, Err(CoreError::Cancelled)), "{stopped:?}");
    })
    .await;
    let next = drafted(&fixture, None, "tickets", None).await.unwrap();
    assert_eq!(next.origin, DraftOrigin::Reused);
    assert_eq!(
        next.set.questions,
        other_questions(),
        "the newest run's set"
    );
    let decision = scoped(model(&fixture.stub)).await;
    let outline = scoped(next.outline(&fixture.writer, &decision, Rows::Missing))
        .await
        .unwrap();
    assert_eq!(
        outline.effect,
        Effect::ReplacesLabels {
            because: RelabelReason::QuestionsChanged
        },
        "the next run labels every row again from the start"
    );
}

#[tokio::test]
async fn an_outline_says_what_the_run_does_to_the_output() {
    let fixture = Fixture::new().await;
    scoped(async {
        let first = fixture.outline(&request()).await;
        assert_eq!(first.effect, Effect::NewTable);
        assert_eq!((first.remaining, first.estimate_seconds), (600, None));
        assert_eq!(first.output_table, "tickets_labels");
        fixture.run(&request()).await;
        fixture
            .sql("INSERT INTO tickets SELECT 600 + range, 'billing t', 'x' FROM range(5)")
            .await;
        let adds = fixture.outline(&request()).await;
        assert_eq!((adds.effect, adds.remaining), (Effect::AddsRows, 5));
        assert!(
            adds.estimate_seconds.is_some(),
            "the earlier run measured the speed"
        );
        let other = fixture
            .outline(&Classification {
                questions: other_questions(),
                ..request()
            })
            .await;
        assert_eq!(
            other.estimate_seconds, None,
            "other questions have no history"
        );
        let all = fixture
            .outline(&Classification {
                rows: Rows::All,
                ..request()
            })
            .await;
        assert_eq!(
            all.effect,
            Effect::ReplacesLabels {
                because: RelabelReason::Asked
            }
        );
        fixture.run(&request()).await;
        let nothing = fixture.outline(&request()).await;
        assert_eq!((nothing.effect, nothing.remaining), (Effect::AddsRows, 0));
    })
    .await;
}

#[test]
fn an_estimate_is_said_in_minutes_or_hours() {
    let said = |seconds: u64| Estimate(seconds).to_string();
    assert_eq!(said(0), "under a minute");
    assert_eq!(said(44), "under a minute");
    assert_eq!(said(45), "about 1 minute");
    assert_eq!(said(89), "about 1 minute");
    assert_eq!(said(90), "about 2 minutes");
    assert_eq!(said(3600), "about 60 minutes");
    assert_eq!(said(7200), "about 2 hours");
    assert_eq!(said(7300), "about 2 hours");
}

#[test]
fn a_preview_estimates_from_the_rows_it_asked() {
    let preview = Preview {
        key_column: String::from("id"),
        output_table: String::from("t_labels"),
        result: QueryResults {
            columns: Vec::new(),
            rows: Vec::new(),
        },
        compact: QueryResults {
            columns: Vec::new(),
            rows: Vec::new(),
        },
        labelled: 8,
        cut: 0,
        empty: 2,
        skipped: 2,
        took_ms: 9000,
        ask_ms: 5000,
        remaining: 1000,
    };
    assert_eq!(preview.seconds_for(1000), Some(500), "500 ms a row asked");
    let none = Preview {
        labelled: 0,
        skipped: 0,
        ..preview
    };
    assert_eq!(none.seconds_for(1000), None);
}

#[test]
fn a_label_set_travels_as_json_and_its_options_as_lines() {
    let set = request().set();
    let text = serde_json::to_string(&set).unwrap();
    assert_eq!(serde_json::from_str::<LabelSet>(&text).unwrap(), set);
    let without_reason = text.replace("\"key_reason\":\"id_like\",", "");
    assert!(serde_json::from_str::<LabelSet>(&without_reason).is_ok());
    assert!(serde_json::from_str::<LabelSet>(&text.replace("key_column", "keyColumn")).is_err());

    let choice = DraftQuestion::from_lines(
        " topic ",
        QuestionKind::Choice,
        " What? ",
        "billing: Invoices, refunds\nsales\n\n  other: Anything: else\nv2:1",
    );
    assert_eq!(choice.name, "topic");
    assert_eq!(choice.instructions, "What?");
    let labels: Vec<(&str, &str)> = choice
        .options
        .iter()
        .map(|o| (o.label.as_str(), o.description.as_str()))
        .collect();
    assert_eq!(
        labels,
        [
            ("billing", "Invoices, refunds"),
            ("sales", ""),
            ("other", "Anything: else"),
            ("v2:1", "")
        ],
        "split at the first colon and space only"
    );
    assert_eq!(
        choice.to_lines(),
        "billing: Invoices, refunds\nsales\nother: Anything: else\nv2:1"
    );
    let again =
        DraftQuestion::from_lines("topic", QuestionKind::Choice, "What?", &choice.to_lines());
    assert_eq!(again, choice);
    let score = DraftQuestion::from_lines(
        "urgent",
        QuestionKind::Score,
        "How?",
        "not\nsoon\n\nnow: blocking",
    );
    assert_eq!(
        score.levels,
        ["not", "soon", "now: blocking"],
        "a level is taken whole"
    );
    assert_eq!(score.to_lines(), "not\nsoon\nnow: blocking");
    let flag = DraftQuestion::from_lines("angry", QuestionKind::Noul, "Angry?", "ignored");
    assert!(flag.options.is_empty() && flag.levels.is_empty());
    assert_eq!(flag.to_lines(), "");
    assert!(choice.clone().into_question().is_ok());
    let single = DraftQuestion::from_lines("x", QuestionKind::Choice, "Which?", "a\na");
    assert!(matches!(
        single.into_question(),
        Err(QuestionSetError::Duplicate(_))
    ));
}
