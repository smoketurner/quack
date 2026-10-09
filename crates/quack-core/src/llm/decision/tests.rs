#![expect(
    clippy::unwrap_used,
    clippy::panic,
    reason = "tests assert on values they have just built"
)]

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use serde_json::{Value, json};

use super::fixture::{config, model, scoped};
use super::stub::{DecisionStub, Fault, too_long};
use super::*;
use crate::llm::egress::Egress;
use crate::storage::control::AllowedProviders;

fn triage() -> Questions {
    serde_json::from_value(json!({
        "department": {
            "type": "choice",
            "instructions": "Which department should handle this ticket?",
            "criteria": {
                "billing": "Invoices, payments, refunds",
                "technical": "Bugs, outages",
                "sales": "Pricing, contracts",
                "none": null
            }
        },
        "urgency": {
            "type": "score",
            "instructions": "How urgent is this ticket?",
            "criteria": ["Not urgent", "Soon", "Blocking or deadline"]
        },
        "churn_risk": {"type": "noul", "instructions": "Does the customer threaten to cancel?"}
    }))
    .unwrap()
}

fn state(text: &str) -> State {
    State::new([(String::from("text"), Some(text.to_owned()))])
}

fn choice(question: &str, options: usize) -> (QuestionName, Question) {
    let criteria = (0..options)
        .map(|n| (Label::try_from(format!("option{n}")).unwrap(), None))
        .collect();
    (
        QuestionName::try_from(question.to_owned()).unwrap(),
        Question::Choice {
            instructions: Instructions::try_from(String::from("Which?")).unwrap(),
            criteria: Unique::from_pairs(criteria).unwrap(),
        },
    )
}

#[test]
fn a_question_set_serializes_as_the_wire_wants_it() {
    let written = serde_json::to_value(triage()).unwrap();
    assert_eq!(
        written,
        json!({
            "department": {
                "type": "choice",
                "instructions": "Which department should handle this ticket?",
                "criteria": {
                    "billing": "Invoices, payments, refunds",
                    "technical": "Bugs, outages",
                    "sales": "Pricing, contracts",
                    "none": null
                }
            },
            "urgency": {
                "type": "score",
                "instructions": "How urgent is this ticket?",
                "criteria": ["Not urgent", "Soon", "Blocking or deadline"]
            },
            "churn_risk": {"type": "noul", "instructions": "Does the customer threaten to cancel?"}
        })
    );
    let order: Vec<_> = triage().iter().map(|(n, _)| n.to_string()).collect();
    assert_eq!(order, ["department", "urgency", "churn_risk"]);
}

#[test]
fn a_noul_question_may_describe_its_two_answers() {
    let set: Questions = serde_json::from_value(json!({
        "q": {"type": "noul", "instructions": "Is it?",
              "criteria": {"true": "yes it is", "false": "no it is not"}}
    }))
    .unwrap();
    let again = serde_json::to_value(&set).unwrap();
    assert_eq!(
        again.pointer("/q/criteria/true"),
        Some(&json!("yes it is")),
        "{again}"
    );
}

#[test]
fn a_question_set_refuses_what_the_server_would() {
    assert_eq!(
        Questions::new(Vec::new()),
        Err(QuestionSetError::NoQuestions)
    );
    let many = (0..65).map(|n| choice(&format!("q{n}"), 2)).collect();
    assert_eq!(
        Questions::new(many),
        Err(QuestionSetError::TooManyQuestions(65))
    );
    for options in [1, 27] {
        assert_eq!(
            Questions::new(vec![choice("q", options)]),
            Err(QuestionSetError::Options {
                question: String::from("q"),
                count: options
            })
        );
    }
    let rubric = Question::Score {
        instructions: Instructions::try_from(String::from("How?")).unwrap(),
        criteria: vec![String::from("only one")],
    };
    let name = QuestionName::try_from(String::from("level")).unwrap();
    assert_eq!(
        Questions::new(vec![(name, rubric)]),
        Err(QuestionSetError::Options {
            question: String::from("level"),
            count: 1
        })
    );
    assert!(Questions::new((0..64).map(|n| choice(&format!("q{n}"), 2)).collect()).is_ok());
    assert!(Questions::new(vec![choice("q", 26)]).is_ok());
}

#[test]
fn question_names_differing_only_in_case_are_the_same_name() {
    assert_eq!(
        Questions::new(vec![choice("Dept", 2), choice("dept", 2)]),
        Err(QuestionSetError::Duplicate(String::from("Dept")))
    );
}

#[test]
fn a_repeated_key_in_a_question_file_is_refused_not_overwritten() {
    let text = r#"{"q": {"type": "noul", "instructions": "a"}, "q": {"type": "noul", "instructions": "b"}}"#;
    let refused = serde_json::from_str::<Questions>(text).unwrap_err();
    assert!(
        refused.to_string().contains("'q' appears twice"),
        "{refused}"
    );
    let options =
        r#"{"q": {"type": "choice", "instructions": "a", "criteria": {"x": null, "x": "again"}}}"#;
    let refused = serde_json::from_str::<Questions>(options).unwrap_err();
    assert!(
        refused.to_string().contains("'x' appears twice"),
        "{refused}"
    );
}

#[test]
fn a_question_file_with_a_misspelled_key_is_refused() {
    let text = r#"{"q": {"type": "noul", "instructions": "a", "instruction": "b"}}"#;
    assert!(serde_json::from_str::<Questions>(text).is_err());
}

#[test]
fn a_question_set_over_16_kib_is_refused() {
    let heavy = Question::Choice {
        instructions: Instructions::try_from(String::from("Which?")).unwrap(),
        criteria: Unique::from_pairs(vec![
            (
                Label::try_from(String::from("a")).unwrap(),
                Some("x".repeat(9000)),
            ),
            (
                Label::try_from(String::from("b")).unwrap(),
                Some("y".repeat(9000)),
            ),
        ])
        .unwrap(),
    };
    let name = QuestionName::try_from(String::from("heavy")).unwrap();
    let refused = Questions::new(vec![(name, heavy)]).unwrap_err();
    assert!(
        matches!(refused, QuestionSetError::TooLarge(n) if n > 16 * 1024),
        "{refused}"
    );
}

#[test]
fn names_labels_and_instructions_are_checked_where_they_are_made() {
    for bad in ["", "1a", "a b", "a-b", "é", &"a".repeat(49)] {
        assert!(QuestionName::try_from(bad.to_owned()).is_err(), "{bad:?}");
    }
    for good in ["a", "dept_1", &"a".repeat(48)] {
        assert!(QuestionName::try_from(good.to_owned()).is_ok(), "{good:?}");
    }
    assert!(Label::try_from(String::from("  ")).is_err());
    assert!(Label::try_from("x".repeat(65)).is_err());
    assert!(Label::try_from("x".repeat(64)).is_ok());
    assert!(Instructions::try_from(String::new()).is_err());
    assert!(Instructions::try_from("x".repeat(1001)).is_err());
}

#[test]
fn a_state_drops_blank_fields_and_caps_its_total() {
    let state = State::new([
        (String::from("a"), Some(String::from("hello"))),
        (String::from("b"), None),
        (String::from("c"), Some(String::from("   "))),
        (String::from("d"), Some(String::from("world"))),
    ]);
    assert_eq!(
        serde_json::to_value(&state).unwrap(),
        json!({"a": "hello", "d": "world"})
    );
    assert!(!state.is_cut());
    assert_eq!(state.len(), 10);

    let long = State::new([
        (String::from("a"), Some("é".repeat(4000))),
        (String::from("b"), Some("ü".repeat(500))),
        (String::from("c"), Some(String::from("never"))),
    ]);
    assert_eq!(long.len(), STATE_CHARS);
    assert!(long.is_cut());
    let written = serde_json::to_value(&long).unwrap();
    assert_eq!(
        written
            .get("b")
            .and_then(Value::as_str)
            .map(|s| s.chars().count()),
        Some(96)
    );
    assert!(written.get("c").is_none());

    assert!(State::new([(String::from("a"), None)]).is_empty());
    let exact = State::new([(String::from("a"), Some("x".repeat(STATE_CHARS)))]);
    assert!(!exact.is_cut());
    let over = State::new([(String::from("a"), Some("x".repeat(STATE_CHARS + 1)))]);
    assert!(over.is_cut());
}

#[tokio::test]
async fn the_request_body_is_what_the_server_expects() {
    let stub = DecisionStub::start().await;
    scoped(async {
        let model = model(&stub).await;
        let questions = triage();
        let asker = model.asker(&questions).await.unwrap();
        let asked = asker
            .ask(&State::new([
                (
                    String::from("subject"),
                    Some(String::from("Duplicate billing")),
                ),
                (String::from("body"), Some(String::from("we will cancel"))),
            ]))
            .await
            .unwrap();
        assert!(matches!(asked, Asked::Answered(_)));
        let body: Value = serde_json::from_str(stub.bodies().last().unwrap()).unwrap();
        assert_eq!(body.get("model"), Some(&json!("laya")));
        assert_eq!(body.get("keep_alive"), Some(&json!(1800)));
        assert_eq!(
            body.get("state"),
            Some(&json!({"subject": "Duplicate billing", "body": "we will cancel"}))
        );
        assert_eq!(
            body.get("questions"),
            Some(&serde_json::to_value(&questions).unwrap())
        );
    })
    .await;
}

#[tokio::test]
async fn each_answer_type_is_read_against_its_question() {
    let stub = DecisionStub::start().await;
    scoped(async {
        let model = model(&stub).await;
        let questions = triage();
        let asker = model.asker(&questions).await.unwrap();
        let Asked::Answered(answers) = asker
            .ask(&state("a billing mistake! we will cancel"))
            .await
            .unwrap()
        else {
            panic!("expected answers");
        };
        assert!(!answers.truncated());
        let read: Vec<_> = answers.iter().collect();
        assert_eq!(read.len(), 3);
        assert_eq!(read.first().map(|(n, _)| n.as_str()), Some("department"));
        assert_eq!(
            read.first().map(|(_, a)| a),
            Some(&Answer::Choice {
                choice: Label::try_from(String::from("billing")).unwrap(),
                probability: 0.9,
                confidence: 0.8
            })
        );
        assert_eq!(
            read.get(1).map(|(_, a)| a),
            Some(&Answer::Score {
                score: 1.0,
                level: 1,
                confidence: 0.7
            })
        );
        assert_eq!(
            read.get(2).map(|(_, a)| a),
            Some(&Answer::Noul { probability: 0.9 })
        );
    })
    .await;
}

#[tokio::test]
async fn an_answer_missing_a_question_or_of_the_wrong_kind_is_an_error() {
    let missing = DecisionStub::with_rule(|seen| {
        (seen.text.contains("row")).then(|| {
            Fault::raw(
                200,
                json!({"answers": {"churn_risk": {"type": "noul", "noul": 0.5}}}),
            )
        })
    })
    .await;
    let wrong = DecisionStub::with_rule(|seen| {
        (seen.text.contains("row")).then(|| {
            Fault::raw(
                200,
                json!({"answers": {
                    "department": {"type": "choice", "choice": "legal",
                                   "probabilities": {"legal": 1.0}, "confidence": 1.0},
                    "urgency": {"type": "noul", "noul": 0.5},
                    "churn_risk": {"type": "noul", "noul": 0.5}}}),
            )
        })
    })
    .await;
    scoped(async {
        let questions = triage();
        for (stub, expected) in [
            (&missing, "lacks questions department, urgency"),
            (&wrong, "chose 'legal', which is not an option"),
        ] {
            let model = model(stub).await;
            let asker = model.asker(&questions).await.unwrap();
            let refused = asker.ask(&state("row")).await.unwrap_err();
            assert!(
                matches!(&refused, Error::Llm(m) if m.contains(expected)),
                "{refused}"
            );
        }
    })
    .await;
}

#[tokio::test]
async fn a_row_within_the_limit_goes_whole_and_is_not_cut() {
    let stub = DecisionStub::limited(1000).await;
    scoped(async {
        let model = model(&stub).await;
        let questions = triage();
        let asker = model.asker(&questions).await.unwrap();
        let before = stub.requests();
        let Asked::Answered(answers) = asker.ask(&state(&"a".repeat(1000))).await.unwrap() else {
            panic!("expected answers");
        };
        assert!(!answers.truncated());
        assert_eq!(stub.requests() - before, 1);
    })
    .await;
}

#[tokio::test]
async fn a_row_over_the_limit_is_cut_to_within_64_characters_of_it() {
    let stub = DecisionStub::limited(1000).await;
    scoped(async {
        let model = model(&stub).await;
        let questions = triage();
        let asker = model.asker(&questions).await.unwrap();
        let before = stub.requests();
        let Asked::Answered(answers) = asker.ask(&state(&"a".repeat(3000))).await.unwrap() else {
            panic!("expected answers");
        };
        assert!(answers.truncated());
        let spent = stub.requests() - before;
        assert_eq!(spent, 7, "the whole state, then six halvings");
        let accepted = stub
            .bodies()
            .iter()
            .skip(before)
            .filter_map(|b| {
                let body: Value = serde_json::from_str(b).ok()?;
                let text = body.pointer("/state/text")?.as_str()?.chars().count();
                (text <= 1000).then_some(text)
            })
            .max()
            .unwrap();
        assert_eq!(accepted, 984);
    })
    .await;
}

#[tokio::test]
async fn a_413_is_a_length_refusal_too() {
    let stub = DecisionStub::with_rule(|seen| {
        too_long(seen, 1000).map(|_| Fault::new(413, "text and schema must not exceed 64 KiB"))
    })
    .await;
    scoped(async {
        let model = model(&stub).await;
        let questions = triage();
        let asker = model.asker(&questions).await.unwrap();
        let asked = asker.ask(&state(&"a".repeat(3000))).await.unwrap();
        assert!(matches!(asked, Asked::Answered(a) if a.truncated()));
    })
    .await;
}

#[tokio::test]
async fn the_search_does_not_stop_before_some_prefix_was_accepted() {
    // Texts of 21 to 599 characters are refused, so the row's whole text
    // and its first middle are, and only a short prefix fits.
    let stub = DecisionStub::with_rule(|seen| {
        (21..600)
            .contains(&seen.chars)
            .then(|| Fault::new(400, "state is too long"))
    })
    .await;
    scoped(async {
        let model = model(&stub).await;
        let questions = triage();
        let asker = model.asker(&questions).await.unwrap();
        let asked = asker.ask(&state(&"a".repeat(50))).await.unwrap();
        assert!(
            matches!(asked, Asked::Answered(ref a) if a.truncated()),
            "{asked:?}"
        );
    })
    .await;
}

#[tokio::test]
async fn a_row_the_model_refuses_at_every_length_is_unfit_when_the_set_still_passes() {
    let stub = DecisionStub::with_rule(|seen| {
        seen.text
            .contains("zzz")
            .then(|| Fault::new(400, "state has too many tokens"))
    })
    .await;
    scoped(async {
        let model = model(&stub).await;
        let questions = triage();
        let asker = model.asker(&questions).await.unwrap();
        let before = stub.requests();
        let asked = asker
            .ask(&state(&format!("zzz{}", "a".repeat(2000))))
            .await
            .unwrap();
        assert_eq!(asked, Asked::Unfit);
        assert_eq!(stub.requests() - before, 9, "8 attempts and the re-probe");
    })
    .await;
}

#[tokio::test]
async fn a_set_the_model_stops_accepting_fails_the_run_instead_of_skipping_rows() {
    let seen = Arc::new(AtomicUsize::new(0));
    let counter = Arc::clone(&seen);
    let stub = DecisionStub::with_rule(move |_| {
        (counter.fetch_add(1, Ordering::SeqCst) >= 2)
            .then(|| Fault::new(400, "decision options exceed the 176-token budget"))
    })
    .await;
    scoped(async {
        let model = model(&stub).await;
        let questions = triage();
        let asker = model.asker(&questions).await.unwrap();
        let refused = asker.ask(&state("a row")).await.unwrap_err();
        assert!(
            matches!(&refused, Error::DecisionRefused(m) if m.contains("now refuses the question set")),
            "{refused}"
        );
    })
    .await;
}

#[tokio::test]
async fn a_refusal_that_is_not_about_length_fails_without_a_search() {
    let seen = Arc::new(AtomicUsize::new(0));
    let counter = Arc::clone(&seen);
    let stub = DecisionStub::with_rule(move |_| {
        (counter.fetch_add(1, Ordering::SeqCst) >= 2).then(|| Fault::new(403, "forbidden"))
    })
    .await;
    scoped(async {
        let model = model(&stub).await;
        let questions = triage();
        let asker = model.asker(&questions).await.unwrap();
        let before = stub.requests();
        let refused = asker.ask(&state(&"a".repeat(2000))).await.unwrap_err();
        assert!(
            matches!(&refused, Error::DecisionRefused(m) if m == "forbidden"),
            "{refused}"
        );
        assert_eq!(stub.requests() - before, 1);
    })
    .await;
}

#[tokio::test]
async fn a_server_failure_is_an_llm_error_not_a_refusal() {
    let stub =
        DecisionStub::with_rule(|seen| seen.text.contains("row").then(|| Fault::new(500, "boom")))
            .await;
    scoped(async {
        let model = model(&stub).await;
        let questions = triage();
        let asker = model.asker(&questions).await.unwrap();
        let failed = asker.ask(&state("row")).await.unwrap_err();
        assert!(
            matches!(&failed, Error::Llm(m) if m.contains("500") && m.contains("boom")),
            "{failed}"
        );
    })
    .await;
}

#[tokio::test]
async fn a_set_the_server_refuses_is_refused_with_its_reason() {
    let stub = DecisionStub::with_rule(|_| {
        Some(Fault::new(
            404,
            "model \"laya\" not found, try pulling it first",
        ))
    })
    .await;
    scoped(async {
        let model = model(&stub).await;
        let questions = triage();
        let refused = model.asker(&questions).await.unwrap_err();
        assert!(
            matches!(&refused, Error::DecisionRefused(m) if m.contains("try pulling it first")),
            "{refused}"
        );
    })
    .await;
}

#[tokio::test]
async fn a_set_that_leaves_too_little_room_for_text_is_refused() {
    let stub = DecisionStub::limited(500).await;
    scoped(async {
        let model = model(&stub).await;
        let questions = triage();
        let refused = model.asker(&questions).await.unwrap_err();
        assert!(
            matches!(&refused, Error::DecisionRefused(m) if m.contains("too little room")),
            "{refused}"
        );
    })
    .await;
}

#[tokio::test]
async fn the_digest_is_found_under_the_implicit_latest_tag() {
    let stub = DecisionStub::start().await;
    scoped(async {
        let model = model(&stub).await;
        assert_eq!(model.digest().await.unwrap(), "sha256:aaaa");
        stub.set_digest("sha256:bbbb");
        assert_eq!(model.digest().await.unwrap(), "sha256:bbbb");
        assert_eq!(
            model.capabilities().await.unwrap(),
            [OllamaCapability::Decision]
        );
    })
    .await;
    let other = Config::parse(&format!(
        "[providers.local]\ntype = \"ollama\"\nbase_url = \"{}\"\n[decision]\nmodel = \"local/other\"\n",
        stub.base_url()
    ))
    .unwrap();
    scoped(async {
        let model = DecisionModel::from_config(&other).await.unwrap().unwrap();
        let unlisted = model.digest().await.unwrap_err();
        assert!(
            unlisted.to_string().contains("ollama pull other"),
            "{unlisted}"
        );
    })
    .await;
}

#[tokio::test]
async fn a_workspace_that_does_not_allow_the_provider_gets_a_typed_refusal() {
    let stub = DecisionStub::start().await;
    let allowed = Egress::Workspace(AllowedProviders::Only(
        std::iter::once(String::from("other")).collect(),
    ));
    Egress::scope(Some(allowed), async {
        let refused = DecisionModel::from_config(&config(stub.base_url()))
            .await
            .unwrap_err();
        assert!(refused.is_provider_refusal(), "{refused}");
    })
    .await;
}

#[tokio::test]
async fn work_outside_any_scope_cannot_send() {
    let stub = DecisionStub::start().await;
    let refused = DecisionModel::from_config(&config(stub.base_url()))
        .await
        .unwrap_err();
    assert!(
        matches!(refused, Error::ModelRequestUnscoped { .. }),
        "{refused}"
    );
}

#[test]
fn the_setting_is_optional_and_names_an_ollama_provider() {
    assert!(Config::default().decision_model_ref().unwrap().is_none());
    let ollama =
        Config::parse("[providers.local]\ntype = \"ollama\"\n[decision]\nmodel = \"local/laya\"\n")
            .unwrap();
    assert_eq!(
        ollama.decision_model_ref().unwrap().unwrap().to_string(),
        "local/laya"
    );
    assert_eq!(ollama.decision.keep_alive_minutes, 30);
    assert_eq!(ollama.decision.interactive_budget, 1500);

    let openai = Config::parse(
        "[providers.cloud]\ntype = \"openai\"\nauth = \"api-key\"\napi_key_env = \"X\"\n\
         [decision]\nmodel = \"cloud/laya\"\n",
    )
    .unwrap_err();
    assert!(
        openai
            .to_string()
            .contains("openai has no decision endpoint"),
        "{openai}"
    );
    let unknown = Config::parse("[decision]\nmodel = \"nowhere/laya\"\n").unwrap_err();
    assert!(
        unknown.to_string().contains("[decision].model"),
        "{unknown}"
    );
    assert!(Config::parse("[decision]\nmodle = \"x/y\"\n").is_err());
}

#[test]
fn a_question_file_past_the_caps_is_refused_without_reading_on() {
    let questions: String = (0..65)
        .map(|n| format!(r#""q{n}": {{"type": "noul", "instructions": "a"}}"#))
        .collect::<Vec<_>>()
        .join(",");
    let refused = serde_json::from_str::<Questions>(&format!("{{{questions}}}")).unwrap_err();
    assert!(
        refused.to_string().contains("more than 64 members"),
        "{refused}"
    );

    let options: String = (0..27)
        .map(|n| format!(r#""o{n}": null"#))
        .collect::<Vec<_>>()
        .join(",");
    let choice =
        format!(r#"{{"q": {{"type": "choice", "instructions": "a", "criteria": {{{options}}}}}}}"#);
    let refused = serde_json::from_str::<Questions>(&choice).unwrap_err();
    assert!(
        refused.to_string().contains("more than 26 members"),
        "{refused}"
    );

    // A body of a hundred thousand options is refused at the 27th, not
    // collected and compared pairwise: the error comes back at once.
    let huge: String = (0..100_000)
        .map(|n| format!(r#""option{n}": null"#))
        .collect::<Vec<_>>()
        .join(",");
    let body =
        format!(r#"{{"q": {{"type": "choice", "instructions": "a", "criteria": {{{huge}}}}}}}"#);
    let started = std::time::Instant::now();
    assert!(serde_json::from_str::<Questions>(&body).is_err());
    assert!(started.elapsed() < std::time::Duration::from_secs(1));
}

#[test]
fn a_prefix_of_multibyte_text_never_splits_a_character() {
    let state = State::new([
        (String::from("a"), Some(String::from("é日😀x"))),
        (String::from("b"), Some(String::from("ünï😀日"))),
    ]);
    let all: Vec<char> = "é日😀xünï😀日".chars().collect();
    for chars in 0..=all.len() {
        let prefix = state.prefix(chars);
        let kept: String = prefix
            .fields
            .iter()
            .map(|(_, value)| value.as_str())
            .collect();
        assert_eq!(kept, all.iter().take(chars).collect::<String>(), "{chars}");
        assert!(prefix.is_cut());
    }
}
