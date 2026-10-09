use quack_core::storage::sessions::ExportFormat;

use super::*;

#[expect(clippy::panic, reason = "test failure path")]
fn fail(msg: &str) -> ! {
    panic!("{msg}")
}

fn words(line: &str) -> Vec<String> {
    Completion::for_line(line)
        .map(|c| c.items.into_iter().map(|s| s.word).collect())
        .unwrap_or_default()
}

fn parses(line: &str) -> bool {
    SlashCommand::parse(line).is_ok()
}

#[test]
fn the_parser_definition_is_valid() {
    SlashLine::command().debug_assert();
}

#[test]
fn command_names_complete_by_prefix() {
    assert_eq!(words("/").len(), visible_subcommands(&TREE).count());
    assert_eq!(words("/ta"), ["/tables"]);
    assert_eq!(
        words("/s"),
        [
            "/sql",
            "/search",
            "/scope",
            "/saved",
            "/sessions",
            "/share",
            "/steps"
        ]
    );
    assert_eq!(words("/se"), ["/search", "/sessions"]);
    assert!(words("/nothing").is_empty());
    assert!(words("hello").is_empty());
    assert!(words("").is_empty());
    assert!(words("/sql SELECT 1").is_empty(), "free text gets nothing");
    assert!(words("/exit ").is_empty());
}

/// `/search` takes the rest of the line as typed, quotes included, and
/// needs a query.
#[test]
fn search_takes_the_query_verbatim() {
    let parsed = SlashCommand::parse("/search \"flood exclusion\" --in policy");
    assert!(
        matches!(parsed, Ok(SlashCommand::Search { ref query }) if query == "\"flood exclusion\" --in policy"),
        "the line after the command is the query"
    );
    assert!(!parses("/search"));
}

#[test]
fn verbs_and_their_flags_come_from_the_cli() {
    assert_eq!(words("/ontology pr"), ["propose"]);
    assert_eq!(words("/graph st"), ["status"]);
    assert_eq!(words("/embeddings "), ["refresh"]);
    assert_eq!(
        words("/ontology propose --"),
        ["--auto-accept", "--documents", "--sample", "--from"],
        "--yes is implied in the terminal"
    );
    assert!(words("/ontology propose --documents --d").is_empty());
    assert_eq!(words("/graph extract --source documents --s"), ["--sample"]);
    assert!(
        words("/graph Alice").is_empty(),
        "an entity name gets nothing"
    );
    assert!(words("/graph Alice ").is_empty());
    assert!(
        words("/graph Alice status").is_empty(),
        "a verb only comes first"
    );
    assert!(
        words("/ontology propose ").is_empty(),
        "flags need a dash first"
    );
    assert!(words("/unknown ").is_empty());
}

#[test]
fn choices_and_nested_verbs_follow_their_command() {
    assert_eq!(words("/mode "), ["chat", "query"]);
    assert_eq!(words("/mode q"), ["query"]);
    assert!(words("/mode chat ").is_empty());
    assert_eq!(words("/export --"), ["--sql", "--markdown"]);
    assert!(
        words("/export --sql --s").is_empty(),
        "a typed flag is not offered again"
    );
    assert_eq!(words("/context e"), ["export"]);
    assert!(words("/context import ").is_empty());
}

#[test]
fn accepting_replaces_the_word_and_spaces_when_more_may_follow() {
    let apply = |line: &str| {
        let completion = Completion::for_line(line)?;
        let item = completion.get(0)?;
        Some((completion.apply(line, item).0, item.finishes()))
    };
    assert_eq!(apply("/ta"), Some((String::from("/tables "), false)));
    assert_eq!(apply("/he"), Some((String::from("/help"), true)));
    assert_eq!(
        apply("/graph sta"),
        Some((String::from("/graph status "), false))
    );
    assert_eq!(
        apply("/graph rev"),
        Some((String::from("/graph revalidate "), false))
    );
    assert_eq!(
        apply("/ontology propose --fr"),
        Some((String::from("/ontology propose --from "), false))
    );
    assert_eq!(apply("/mode c"), Some((String::from("/mode chat"), true)));
}

#[test]
fn selection_past_the_end_clamps_to_the_last() {
    let completion = Completion::for_line("/mode ");
    let last = completion
        .as_ref()
        .and_then(|c| c.get(9))
        .map(|s| s.word.as_str());
    assert_eq!(last, Some("query"));
}

#[test]
fn help_lists_every_command_alias_and_usage() {
    let help = SlashCommand::help();
    for command in visible_subcommands(&TREE) {
        assert!(
            help.contains(command.get_name()),
            "{} missing",
            command.get_name()
        );
        for alias in command.get_visible_aliases() {
            assert!(help.contains(alias), "{alias} missing");
        }
    }
    for line in [
        "/help, /?",
        "/quit, /exit, /q",
        "/tables [TABLE] [--note TEXT] [--retype COLUMN=TYPE]",
        "/mode [chat|query]",
        "/export [--sql] [--markdown] [FILE]",
        "/import URL TABLE [SOURCE_TABLE] [--query SQL]",
        "/ontology VERB ...",
        "/context [import FILE | export FILE]",
        "/embeddings refresh  ",
        "/ingest, /attach PATH  Load",
    ] {
        assert!(help.contains(line), "{line} missing from\n{help}");
    }
    assert!(help.contains("Shortcuts:"));
}

/// `/saved` is the CLI's verbs: bare, it lists; `add` takes the
/// session's last answer, so `--from-session` is optional here; the
/// command-line flags are neither offered nor accepted.
#[test]
fn saved_verbs_parse_and_refuse_the_command_line_only_flags() {
    assert!(matches!(
        SlashCommand::parse("/saved"),
        Ok(SlashCommand::Saved { action: None })
    ));
    assert!(matches!(
        SlashCommand::parse("/saved add overdue"),
        Ok(SlashCommand::Saved {
            action: Some(SavedAction::Add {
                from_session: None,
                message: None,
                ..
            })
        })
    ));
    assert!(matches!(
        SlashCommand::parse("/saved add overdue --message 5"),
        Ok(SlashCommand::Saved {
            action: Some(SavedAction::Add {
                message: Some(5),
                ..
            })
        })
    ));
    assert!(matches!(
        SlashCommand::parse("/saved run overdue"),
        Ok(SlashCommand::Saved {
            action: Some(SavedAction::Run { refresh: false, .. })
        })
    ));
    assert!(matches!(
        SlashCommand::parse("/saved remove overdue"),
        Ok(SlashCommand::Saved {
            action: Some(SavedAction::Remove { .. })
        })
    ));
    assert!(parses("/saved show overdue"));
    assert!(parses("/saved list --format json"));
    assert!(parses("/saved run overdue -f csv"));
    assert!(!parses("/saved run"));
    assert!(!parses("/saved forget overdue"));
    for flag in ["--refresh", "--exit-code"] {
        let refused = SlashCommand::parse(&format!("/saved run overdue {flag}"))
            .err()
            .map(|e| (e.kind(), e.to_string()));
        let Some((kind, text)) = refused else {
            fail(&format!("{flag} was accepted"));
        };
        assert_eq!(kind, ErrorKind::UnknownArgument);
        assert!(
            text.contains(&format!("quack saved run NAME {flag}")),
            "{text}"
        );
    }
    assert_eq!(words("/saved "), ["list", "add", "run", "show", "remove"]);
    assert_eq!(words("/saved run overdue --"), ["--format"]);
    assert_eq!(words("/saved add x --"), ["--from-session", "--message"]);
}

#[test]
fn free_text_parses_and_missing_arguments_are_refused() {
    assert!(parses("/sql"));
    assert!(parses("/sql SELECT -1 AS x"));
    assert!(parses("/path Alice -> Bob"));
    assert!(parses("/graph --class Person"));
    assert!(parses("/graph Alice 2"));
    assert!(parses("/graph status --format json"));
    assert!(parses("/q"));
    assert!(parses("/? "));
    assert!(parses("/sql SELECT '-h' --help"), "free text keeps -h");
    assert!(matches!(
        SlashCommand::parse("/ontology --help").map_err(|e| e.kind()),
        Err(ErrorKind::DisplayHelp)
    ));
    assert!(!parses("/schema"), "/tables TABLE describes a table");
    assert!(!parses("/tables --note x"), "a note needs a table");
    assert!(!parses("/graph"));
    assert!(!parses("/mode fast"));
    assert!(!parses("/nothing"));
    assert!(parses("/ontology propose --documents"));
    assert!(parses("/ontology rename class vendor supplier"));
    assert!(parses("/ontology rename relation ships_to delivers_to"));
    assert!(!parses("/ontology rename class vendor"));
    assert_eq!(
        Completion::for_line("/graph revalidate --").map(|c| c
            .items
            .into_iter()
            .map(|i| i.word)
            .collect::<Vec<_>>()),
        Some(vec![String::from("--yes")]),
        "the session offers the flag it does not supply itself"
    );
    assert!(matches!(
        SlashCommand::parse("/graph revalidate -y"),
        Ok(SlashCommand::Graph {
            action: Some(GraphAction::Revalidate { yes: true }),
            walk: None,
        })
    ));
    assert!(matches!(
        SlashCommand::parse("/graph revalidate"),
        Ok(SlashCommand::Graph {
            action: Some(GraphAction::Revalidate { yes: false }),
            walk: None,
        })
    ));
    assert!(
        !parses("/ontology propose --extend"),
        "propose has one behavior: what the ontology lacks"
    );
    assert!(matches!(
        SlashCommand::parse("/graph merges"),
        Ok(SlashCommand::Graph {
            action: Some(GraphAction::Merges),
            walk: None,
        })
    ));
}

#[test]
fn free_text_arrives_as_typed_and_the_rest_splits_like_a_shell_line() {
    assert!(matches!(
        SlashCommand::parse("/sql SELECT 'a  b' AS \"x\""),
        Ok(SlashCommand::Sql { statement: Some(s) }) if s == "SELECT 'a  b' AS \"x\""
    ));
    assert!(matches!(
        SlashCommand::parse("/sql"),
        Ok(SlashCommand::Sql { statement: None })
    ));
    assert!(matches!(
        SlashCommand::parse("/context import my notes.md"),
        Ok(SlashCommand::Context { action: Some(ContextAction::Import { file }) })
            if file == "my notes.md"
    ));
    assert!(matches!(
        SlashCommand::parse("/import sqlite:/tmp/a.db t --query \"SELECT * FROM x WHERE y = 'z'\""),
        Ok(SlashCommand::Import { action: None, url: Some(url), table: Some(table), source_table: None, query: Some(q), .. })
            if url == "sqlite:/tmp/a.db" && table == "t" && q == "SELECT * FROM x WHERE y = 'z'"
    ));
    assert!(matches!(
        SlashCommand::parse("/import refresh nightly"),
        Ok(SlashCommand::Import { action: Some(ImportAction::Refresh { name }), .. }) if name == "nightly"
    ));
    assert!(matches!(
        SlashCommand::parse("/import list"),
        Ok(SlashCommand::Import {
            action: Some(ImportAction::List { .. }),
            ..
        })
    ));
    assert!(
        SlashCommand::parse("/import sqlite:/tmp/a.db").is_err(),
        "a table is needed"
    );
    assert!(matches!(
        SlashCommand::parse(
            "/import https://api.example.com/sales.json sales -H 'X-Team: data' --bearer-env SALES_TOKEN --json-pointer /data/items"
        ),
        Ok(SlashCommand::Import { headers, bearer_env: Some(var), json_pointer: Some(p), .. })
            if headers.len() == 1
                && headers.first().is_some_and(|h| h.name() == "x-team")
                && var == "SALES_TOKEN"
                && p.to_string() == "/data/items"
    ));
    assert!(SlashCommand::parse("/import https://x/a.json t --json-pointer data").is_err());
    assert!(matches!(
        SlashCommand::parse("/rename  Q3 --freight review "),
        Ok(SlashCommand::Rename { title: Some(t) }) if t == "Q3 --freight review"
    ));
    assert!(matches!(
        SlashCommand::parse("/rename"),
        Ok(SlashCommand::Rename { title: None })
    ));
    assert!(matches!(
        SlashCommand::parse("/sessions"),
        Ok(SlashCommand::Sessions { query: None })
    ));
    assert!(matches!(
        SlashCommand::parse("/sessions late \"vendors\""),
        Ok(SlashCommand::Sessions { query: Some(q) }) if q == "late \"vendors\""
    ));
    assert!(matches!(
        SlashCommand::parse("/export --sql 'the session.sql'"),
        Ok(SlashCommand::Export { flags, file: Some(f) })
            if flags.format() == ExportFormat::Sql && f == "the session.sql"
    ));
    let unclosed = SlashCommand::parse("/export 'open");
    assert!(
        unclosed
            .as_ref()
            .is_err_and(|e| e.to_string().contains("quote is not closed")),
        "{:?}",
        unclosed.map(|_| ())
    );
    assert!(matches!(
        SlashCommand::parse("/cancel #3"),
        Ok(SlashCommand::Cancel { job }) if job.to_string() == "3"
    ));
    assert!(!parses("/cancel x"));
    assert!(matches!(
        SlashCommand::parse("/unknown 'quote"),
        Err(e) if e.kind() == ErrorKind::InvalidSubcommand
    ));
}

#[test]
fn a_graph_walk_is_an_entity_with_optional_hops_or_a_class() {
    let walk = |text: &str| text.parse::<GraphWalk>();
    assert_eq!(
        walk("O'Brien Ltd 3"),
        Ok(GraphWalk::Entity {
            name: String::from("O'Brien Ltd"),
            hops: Hops::new(3)
        })
    );
    assert_eq!(
        walk("Alice"),
        Ok(GraphWalk::Entity {
            name: String::from("Alice"),
            hops: Hops::NEIGHBORHOOD
        })
    );
    assert_eq!(
        walk("--class  Person"),
        Ok(GraphWalk::Class(String::from("Person")))
    );
    assert!(walk("--class").is_err());
    assert!(matches!(
        SlashCommand::parse("/graph O'Brien 2"),
        Ok(SlashCommand::Graph { action: None, walk: Some(GraphWalk::Entity { name, .. }) })
            if name == "O'Brien"
    ));
    assert_eq!(
        "Alice -> Bob Jones".parse::<Route>(),
        Ok(Route {
            from: String::from("Alice"),
            to: String::from("Bob Jones")
        })
    );
    assert!("Alice ->".parse::<Route>().is_err());
    assert!("Alice".parse::<Route>().is_err());
}

#[test]
fn a_line_is_a_command_a_file_sql_or_a_question() {
    let dir = tempfile::tempdir().unwrap_or_else(|e| fail(&e.to_string()));
    let file = dir.path().join("notes.md");
    std::fs::write(&file, "# hi").unwrap_or_else(|e| fail(&e.to_string()));
    let path = file.display().to_string();
    assert_eq!(
        Input::classify(String::from("/tables")),
        Input::Command(String::from("/tables"))
    );
    assert_eq!(
        Input::classify(format!("'{path}'")),
        Input::Files(FileLine::Files(vec![file.clone()]))
    );
    assert_eq!(
        Input::classify(String::from("SELECT 1")),
        Input::Sql(String::from("SELECT 1"))
    );
    assert_eq!(
        Input::classify(String::from("what is in notes.md")),
        Input::Question(String::from("what is in notes.md"))
    );
    assert!(FileLine::of("/nowhere/notes.md").is_none());
    assert!(FileLine::of("").is_none());

    // A dropped file arrives shell-quoted: spaces escaped or the path
    // in quotes, several files on one line.
    let spaced = dir.path().join("q3 review (final).md");
    std::fs::write(&spaced, "# hi").unwrap_or_else(|e| fail(&e.to_string()));
    let literal = spaced.display().to_string();
    // A Unix terminal escapes the spaces; a Windows one quotes the path.
    let escaped = if cfg!(windows) {
        format!("\"{literal}\"")
    } else {
        literal
            .replace(' ', "\\ ")
            .replace('(', "\\(")
            .replace(')', "\\)")
    };
    for line in [literal.clone(), escaped.clone(), format!("\"{literal}\"")] {
        assert_eq!(
            Input::classify(line.clone()),
            Input::Files(FileLine::Files(vec![spaced.clone()])),
            "{line}"
        );
    }
    assert_eq!(
        FileLine::of(&format!("{escaped} {}\n", quoted(&path))),
        Some(FileLine::Files(vec![spaced, file.clone()]))
    );
    // One name that is not a file makes the whole line something else.
    assert!(FileLine::of(&format!("{escaped} /nowhere/notes.md")).is_none());
    assert!(FileLine::of(&format!("summarize {escaped}")).is_none());

    // A `#` inside a name is part of it, alone or among several.
    let hashed = dir.path().join("#drafts.md");
    std::fs::write(&hashed, "# hi").unwrap_or_else(|e| fail(&e.to_string()));
    let hash_path = hashed.display().to_string();
    assert_eq!(
        FileLine::of(&format!("{path} {hash_path}")),
        Some(FileLine::Files(vec![file, hashed]))
    );
    // A word that starts with `#` ends the names the POSIX split returns:
    // the line is refused whole, whatever follows the `#`. Windows quoting
    // has no comments.
    #[cfg(not(windows))]
    for cut in [
        String::from("#drafts.md"),
        String::from("#drafts.md 'more.md"),
        String::from("#"),
        format!("'{hash_path}' #more.md"),
    ] {
        assert_eq!(
            FileLine::of(&format!("{path} {cut}")),
            Some(FileLine::Comment),
            "{cut}"
        );
    }
    // Quoted, it is a name again: no such file here, so not a file line.
    assert!(FileLine::of(&format!("{path} {}", quoted("#drafts.md"))).is_none());
    // A question that mentions a file and a `#` word stays a question.
    assert!(FileLine::of(&format!("summarize {path} #urgent")).is_none());
}

/// A path quoted as the platform's terminal quotes it.
fn quoted(path: &str) -> String {
    if cfg!(windows) {
        format!("\"{path}\"")
    } else {
        format!("'{path}'")
    }
}

/// `/classify` takes what `quack classify` does: a table, its text columns,
/// and a question file; or `list`.
#[test]
fn classify_takes_a_table_columns_and_a_question_file() {
    assert!(parses(
        "/classify tickets --text subject,body --questions q.json --preview 5 --all"
    ));
    assert!(parses("/classify list"));
    assert!(!parses("/classify"));
    assert!(!parses("/classify tickets"));
    assert!(!parses(
        "/classify tickets --text subject --questions q.json --preview 0"
    ));
    assert_eq!(words("/cla"), ["/classify"]);
}
