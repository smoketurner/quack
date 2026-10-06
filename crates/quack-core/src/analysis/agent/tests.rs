use super::*;
use crate::analysis::citations::CitationRegistry;

#[test]
fn the_ollama_window_rounds_up_within_bounds() {
    let window = |tokens, cap| OllamaWindow::for_prompt(Tokens::new(tokens), Tokens::new(cap)).0;
    assert_eq!(window(0, 32_768), 8_192);
    assert_eq!(window(1_000, 32_768), 16_384);
    // 12,875 prompt tokens plus headroom rounds to 24,576.
    assert_eq!(window(12_875, 32_768), 24_576);
    assert_eq!(window(100_000, 32_768), 32_768);
    assert_eq!(window(100_000, 2_048), 8_192);
}
use crate::ids::{ChunkId, DocumentId};
use rig::completion::PromptError;

fn prompt_error(e: PromptError) -> rig::agent::StreamingError {
    rig::agent::StreamingError::Prompt(e)
}

#[test]
fn stream_errors_are_explained_for_the_user() {
    let config = AnalysisConfig::default();
    let unknown = prompt_error(PromptError::UnknownToolCall {
        tool_name: String::from("container.exec"),
        available_tools: vec![String::from("run_sql")],
        allowed_tools: vec![String::from("run_sql")],
        chat_history: Vec::new(),
    });
    assert!(StreamStop(&unknown).by_agent_loop());
    let text = StreamStop(&unknown).explain(config.max_turns, Window::Ollama(OllamaWindow(8_192)));
    assert!(text.contains("container.exec"), "{text}");
    assert!(text.contains("max_context_tokens"), "{text}");
    let text = StreamStop(&unknown).explain(config.max_turns, Window::Provider);
    assert!(!text.contains("Ollama"), "{text}");

    let limit = prompt_error(PromptError::MaxTurnsError {
        max_turns: 10,
        chat_history: Vec::new(),
        prompt: Message::user("q"),
    });
    let text = StreamStop(&limit).explain(config.max_turns, Window::Provider);
    assert!(
        text.contains(&format!("{} tool calls", config.max_turns)),
        "{text}"
    );

    let provider = rig::agent::StreamingError::Completion(ProviderError::Provider(String::from(
        "connection refused",
    )));
    assert!(!StreamStop(&provider).by_agent_loop());
    let text = StreamStop(&provider).explain(config.max_turns, Window::Provider);
    assert!(text.contains("connection refused"), "{text}");
}

#[test]
#[expect(clippy::indexing_slicing, reason = "test asserts fixed keys")]
fn to_json_carries_every_field_and_derives_queries() {
    let response = AgentResponse {
        content: String::from("12 storms [1]"),
        steps: vec![
            ToolStep {
                tool: ToolName::RunSql,
                detail: String::from("SELECT count(*) FROM events"),
                summary: String::from("the summary is not parsed"),
                rows: Some(1),
                duration_ms: 7,
            },
            ToolStep {
                tool: ToolName::SearchDocuments,
                detail: String::from("storms"),
                summary: String::from("3 chunks"),
                rows: None,
                duration_ms: 4,
            },
        ],
        citations: vec![Citation {
            n: 1,
            chunk_id: ChunkId::from("c"),
            document_id: DocumentId::from("d"),
            filename: String::from("noaa.pdf"),
            chunk_index: 2,
            page: Some(4),
            heading: None,
            ingested_at: Some(jiff::civil::DateTime::constant(2026, 10, 5, 14, 3, 0, 0)),
        }],
        ..AgentResponse::default()
    };
    let json = response.to_json(&SessionId::from("s1"));
    assert_eq!(json["answer"], "12 storms [1]");
    assert_eq!(json["queries"][0]["sql"], "SELECT count(*) FROM events");
    assert_eq!(json["queries"][0]["rows"], 1);
    assert_eq!(json["queries"].as_array().map(Vec::len), Some(1));
    assert_eq!(
        json["citations"][0]["label"],
        "noaa.pdf, page 4, ingested 2026-10-05"
    );
    assert_eq!(json["citations"][0]["ingested_at"], "2026-10-05T14:03:00");
    assert_eq!(json["citations"][0]["chunk_id"], "c");
    assert_eq!(json["session_id"], "s1");
    assert_eq!(json["write_refused"], false);
    assert_eq!(json["cancelled"], false);
    assert!(json["graph"].is_array() && json["chart"].is_null());
    // A provider that reported nothing leaves `usage` null rather than
    // claiming the turn was free.
    assert!(json["usage"].is_null());

    let counted = AgentResponse {
        usage: Some(TokenUsage {
            input_tokens: 980,
            output_tokens: 43,
            total_tokens: 1_023,
        }),
        ..AgentResponse::default()
    };
    let json = counted.to_json(&SessionId::from("s1"));
    assert_eq!(json["usage"]["input_tokens"], 980);
    assert_eq!(json["usage"]["output_tokens"], 43);
    assert_eq!(json["usage"]["total_tokens"], 1_023);
}

#[test]
fn per_call_usage_accumulates_across_a_turns_completion_requests() {
    let call = |input: u64, output: u64| rig::completion::Usage {
        input_tokens: Some(input),
        output_tokens: Some(output),
        total_tokens: Some(input.saturating_add(output)),
        ..rig::completion::Usage::default()
    };
    let mut usage = TokenUsage::default();
    usage.add(call(400, 20));
    usage.add(call(650, 35));
    assert_eq!(
        usage,
        TokenUsage {
            input_tokens: 1_050,
            output_tokens: 55,
            total_tokens: 1_105,
        }
    );
    assert_eq!(usage.reported(), Some(usage));
    // A provider that reports nothing leaves the accumulator at its
    // default, which `run_inner` reads as "no counts", not zero cost.
    let mut none = TokenUsage::default();
    none.add(rig::completion::Usage::default());
    assert_eq!(none, TokenUsage::default());
    assert_eq!(none.reported(), None);
}

#[test]
fn turn_text_keeps_streamed_text_and_notes_early_stops() {
    let as_is = |text: &str| CitedAnswer {
        text: text.to_owned(),
        citations: Vec::new(),
    };
    let text = |streamed: &str, final_text: Option<&str>, stopped: Option<&str>, window| {
        turn_text(
            streamed.to_owned(),
            final_text.map(str::to_owned),
            stopped.map(str::to_owned),
            window,
            as_is,
        )
        .text
    };
    assert_eq!(
        text("so far", None, Some("why"), Window::Provider),
        "so far\n\n(why)"
    );
    assert_eq!(text("", Some("final"), None, Window::Provider), "final");
    let empty = text("", Some(""), None, Window::Ollama(OllamaWindow(8_192)));
    assert!(empty.contains("[analysis].max_context_tokens"), "{empty}");
    let empty = text("", None, None, Window::Provider);
    assert!(!empty.contains("Ollama"), "{empty}");
}

/// The note goes on after the citation check: an answer the check
/// empties (a lone invented marker) gets the no-text note, and the
/// check, which drops leaked `[analysis]` channel tokens, never sees
/// quack's own `[analysis]` setting names.
#[test]
fn notes_are_added_after_the_citation_check() {
    let check = |text: &str| CitationRegistry::default().validate(text);
    let answer = turn_text(
        String::from("[7]"),
        None,
        None,
        Window::Ollama(OllamaWindow(8_192)),
        check,
    );
    assert!(
        answer.text.starts_with("(The model returned no text.")
            && answer.text.contains("raise [analysis].max_context_tokens"),
        "{}",
        answer.text
    );
    let answer = turn_text(
        String::from("Partly [analysis]answered"),
        None,
        Some(String::from("see [analysis].max_turns")),
        Window::Provider,
        check,
    );
    assert_eq!(answer.text, "Partly answered\n\n(see [analysis].max_turns)");
}
