//! The slash commands as a clap parser: the terminal dispatches on it, and
//! `/help` and the input popup are rendered from the same definition.

use std::path::PathBuf;
use std::str::FromStr;
use std::sync::LazyLock;

use clap::error::ErrorKind;
use clap::{Arg, Command, CommandFactory, Parser, Subcommand};
use quack_core::graph::traverse::Hops;
use quack_core::ingestion::parser::FileType;
use quack_core::jobs::JobNumber;
use quack_core::storage::workspace::looks_like_direct_sql;

use crate::embeddings_cli::EmbeddingsAction;
use crate::graph_cli::GraphAction;
use crate::ontology_cli::OntologyAction;
use crate::{ExportFlags, ModeArg};

/// The argument id of a command that takes the rest of the line as typed
/// (a statement, a path, an entity name), so quotes and spacing survive.
const VERBATIM: &str = "verbatim";

/// One slash command line, parsed from its words.
#[derive(Parser)]
#[command(name = "quack", no_binary_name = true, disable_help_subcommand = true)]
struct SlashLine {
    #[command(subcommand)]
    command: SlashCommand,
}

/// The commands.
#[derive(Subcommand)]
pub(crate) enum SlashCommand {
    /// Show this help message
    #[command(name = "/help", visible_alias = "/?")]
    Help,
    /// Run SQL directly; with no argument, edit the last query
    #[command(name = "/sql", disable_help_flag = true)]
    Sql {
        #[arg(id = VERBATIM, allow_hyphen_values = true, value_name = "STATEMENT")]
        statement: Option<String>,
    },
    /// List tables in the workspace
    #[command(name = "/tables")]
    Tables,
    /// Columns, types, and sample rows of a table
    #[command(name = "/schema", disable_help_flag = true)]
    Schema {
        #[arg(id = VERBATIM, allow_hyphen_values = true, value_name = "TABLE")]
        table: String,
    },
    /// Load a file (a bare path typed at the prompt does the same)
    #[command(name = "/ingest", visible_alias = "/attach", disable_help_flag = true)]
    Ingest {
        #[arg(id = VERBATIM, allow_hyphen_values = true, value_name = "PATH")]
        path: String,
    },
    /// Pull rows from Postgres, SQLite, or a URL
    #[command(name = "/import", disable_help_flag = true)]
    Import {
        url: String,
        table: String,
        source_table: Option<String>,
        /// Run this query on the source instead of reading a table
        #[arg(long, value_name = "SQL")]
        query: Option<String>,
    },
    /// List ingested documents
    #[command(name = "/docs")]
    Docs,
    /// Pin a document's full text into every prompt
    #[command(name = "/pin", disable_help_flag = true)]
    Pin { id: String },
    /// Stop pinning a document's full text
    #[command(name = "/unpin", disable_help_flag = true)]
    Unpin { id: String },
    /// Delete a document with its chunks, table, and graph rows
    #[command(name = "/delete", disable_help_flag = true)]
    Delete { id: String },
    /// The ontology: show, init, propose, review, accept, reject, and versions
    #[command(name = "/ontology")]
    Ontology {
        #[command(subcommand)]
        action: OntologyAction,
    },
    /// Walk the knowledge graph from an entity or a class, or run a graph verb
    #[command(
        name = "/graph",
        args_conflicts_with_subcommands = true,
        subcommand_negates_reqs = true
    )]
    Graph {
        #[command(subcommand)]
        action: Option<GraphAction>,
        #[arg(
            id = VERBATIM,
            required = true,
            allow_hyphen_values = true,
            value_name = "ENTITY [HOPS] | --class CLASS"
        )]
        walk: Option<GraphWalk>,
    },
    /// Shortest relation chain between two entities
    #[command(name = "/path", disable_help_flag = true)]
    Path {
        #[arg(id = VERBATIM, allow_hyphen_values = true, value_name = "FROM -> TO")]
        route: Route,
    },
    /// Show, replace, or save the workspace context
    #[command(name = "/context")]
    Context {
        #[command(subcommand)]
        action: Option<ContextAction>,
    },
    /// Export the workspace as an Open Knowledge Format bundle
    #[command(name = "/okf", disable_help_flag = true)]
    Okf {
        #[arg(id = VERBATIM, allow_hyphen_values = true, value_name = "DIR")]
        dir: String,
    },
    /// Refresh what the embedding model, width, or prefixes left stale
    #[command(name = "/embeddings")]
    Embeddings {
        #[command(subcommand)]
        action: EmbeddingsAction,
    },
    /// List recent sessions
    #[command(name = "/sessions")]
    Sessions,
    /// Switch to a session (id prefix accepted) and replay it
    #[command(name = "/resume", disable_help_flag = true)]
    Resume { id: String },
    /// Start a fresh session
    #[command(name = "/new")]
    New,
    /// Show or set the answer mode
    #[command(name = "/mode")]
    Mode { mode: Option<ModeArg> },
    /// Share this session with every member
    #[command(name = "/share")]
    Share,
    /// Stop sharing this session
    #[command(name = "/unshare")]
    Unshare,
    /// Save this session
    #[command(name = "/export")]
    Export {
        #[command(flatten)]
        flags: ExportFlags,
        file: Option<String>,
    },
    /// List running, queued, and recent jobs
    #[command(name = "/jobs")]
    Jobs,
    /// Cancel job N from /jobs (queued or running)
    #[command(name = "/cancel", disable_help_flag = true)]
    Cancel {
        // As typed: a shell split would read `#3`, the way /jobs shows a
        // number, as a comment.
        #[arg(id = VERBATIM, value_name = "N")]
        job: JobNumber,
    },
    /// Show the chart of the Nth chart-bearing answer (default: the last)
    #[command(name = "/chart", disable_help_flag = true)]
    Chart {
        #[arg(value_name = "N")]
        n: Option<usize>,
    },
    /// Expand or collapse the tool call details
    #[command(name = "/steps")]
    Steps,
    /// Show the chat and embedding models in use
    #[command(name = "/model")]
    Model,
    /// Clear messages and chart
    #[command(name = "/clear")]
    Clear,
    /// Show current workspace and session
    #[command(name = "/workspace")]
    Workspace,
    /// Exit quack
    #[command(name = "/quit", visible_aliases = ["/exit", "/q"])]
    Quit,
}

#[derive(Subcommand)]
pub(crate) enum ContextAction {
    /// Replace the workspace context with a file
    Import {
        #[arg(id = VERBATIM, allow_hyphen_values = true, value_name = "FILE")]
        file: String,
    },
    /// Save the workspace context to a file
    Export {
        #[arg(id = VERBATIM, allow_hyphen_values = true, value_name = "FILE")]
        file: String,
    },
}

/// `/graph`'s walk: an entity's neighbourhood, or a class's entities.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum GraphWalk {
    /// `ENTITY [HOPS]`: a trailing number is the hop count.
    Entity { name: String, hops: Hops },
    /// `--class CLASS`.
    Class(String),
}

impl FromStr for GraphWalk {
    type Err = String;

    fn from_str(text: &str) -> Result<Self, Self::Err> {
        let text = text.trim();
        if let Some(class) = text.strip_prefix("--class") {
            let class = class.trim();
            return if class.is_empty() {
                Err(String::from("--class needs a class name"))
            } else {
                Ok(Self::Class(class.to_owned()))
            };
        }
        let (name, hops) = text
            .rsplit_once(char::is_whitespace)
            .and_then(|(name, hops)| Some((name.trim(), Hops::new(hops.parse().ok()?))))
            .unwrap_or((text, Hops::NEIGHBORHOOD));
        Ok(Self::Entity {
            name: name.to_owned(),
            hops,
        })
    }
}

/// `/path`'s `FROM -> TO`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Route {
    pub(crate) from: String,
    pub(crate) to: String,
}

impl FromStr for Route {
    type Err = String;

    fn from_str(text: &str) -> Result<Self, Self::Err> {
        text.split_once("->")
            .map(|(from, to)| (from.trim(), to.trim()))
            .filter(|(from, to)| !from.is_empty() && !to.is_empty())
            .map(|(from, to)| Self {
                from: from.to_owned(),
                to: to.to_owned(),
            })
            .ok_or_else(|| String::from("expected FROM -> TO"))
    }
}

impl SlashCommand {
    /// Parse a typed `/` line. Words naming a command or one of its verbs
    /// are read first; after them, a command taking free text gets the rest
    /// of the line as typed, and any other has it split like a shell line.
    pub(crate) fn parse(line: &str) -> Result<Self, clap::Error> {
        let mut words: Vec<String> = Vec::new();
        let mut command: &Command = &TREE;
        let mut rest = line.trim();
        while let Some((word, after)) = next_word(rest)
            && let Some(sub) = command.find_subcommand(word)
        {
            words.push(word.to_owned());
            command = sub;
            rest = after;
        }
        if words.is_empty() {
            // Not a command: clap names the first word as unknown.
            words.extend(rest.split_whitespace().map(str::to_owned));
        } else if takes_verbatim(command) {
            words.extend((!rest.is_empty()).then(|| rest.to_owned()));
        } else {
            let split = shlex::split(rest).ok_or_else(|| {
                SlashLine::command().error(ErrorKind::ValueValidation, "a quote is not closed")
            })?;
            words.extend(split);
        }
        SlashLine::try_parse_from(words).map(|line| line.command)
    }
}

/// The first word of `text` and what follows it, trimmed.
fn next_word(text: &str) -> Option<(&str, &str)> {
    if text.is_empty() {
        return None;
    }
    Some(
        text.split_once(char::is_whitespace)
            .map_or((text, ""), |(word, after)| (word, after.trim_start())),
    )
}

/// Whether `command` reads the rest of the line as typed.
fn takes_verbatim(command: &Command) -> bool {
    command.get_arguments().any(|arg| arg.get_id() == VERBATIM)
}

/// What a submitted line is.
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum Input {
    /// A `/` command.
    Command(String),
    /// A path to a file quack can load.
    File(PathBuf),
    /// A statement to run as typed.
    Sql(String),
    /// A question for the agent.
    Question(String),
}

impl Input {
    pub(crate) fn classify(line: String) -> Self {
        if line.starts_with('/') {
            Self::Command(line)
        } else if let Some(path) = Self::file(&line) {
            Self::File(path)
        } else if looks_like_direct_sql(&line) {
            Self::Sql(line)
        } else {
            Self::Question(line)
        }
    }

    /// The file `text` names, when it is one quack can load: quoted or
    /// not, `~/` for the home directory, relative to the working directory
    /// otherwise.
    pub(crate) fn file(text: &str) -> Option<PathBuf> {
        let cleaned = text.trim().trim_matches('\'').trim_matches('"');
        if cleaned.is_empty() || cleaned.contains('\n') {
            return None;
        }
        FileType::of(cleaned)?;
        let path = match cleaned.strip_prefix("~/") {
            Some(under_home) => dirs::home_dir()?.join(under_home),
            None => PathBuf::from(cleaned),
        };
        path.is_file().then_some(path)
    }
}

/// The parser's command tree, built once.
static TREE: LazyLock<Command> = LazyLock::new(SlashLine::command);

const HELP_TAIL: &str = "
Everything you send runs as a background job, so you can keep typing: ask
the next question, run SQL, or load a file while an answer streams. Questions
in one session are answered in order; other work runs alongside, up to
[jobs].workers at once. The strip above the input shows what is running.

Shortcuts:
  /                 List commands; Up/Down pick, Tab fills in, Enter runs, Esc hides
  Enter             Send message
  Up/Down           Browse input history (kept across sessions)
  PageUp/PageDown, mouse wheel   Scroll messages; Home/End jump
  Ctrl+U            Clear input line
  Ctrl+L            Clear screen
  Esc or Ctrl+C     Cancel this session's newest question (running or queued)
  Ctrl+C            Quit (twice while background jobs are still running)";

/// Where `/help` starts each description, when the command fits before it.
const HELP_COLUMN: usize = 18;

/// A command with at most this many verbs spells them out in its usage;
/// more read as `VERB ...` and the popup lists them.
const INLINE_VERBS: usize = 3;

/// Arguments the terminal supplies itself (`--yes`: it never asks), or
/// clap's own, so offering them would mislead.
const IMPLIED_ARGS: &[&str] = &["yes", "help"];

impl SlashCommand {
    /// The `/help` text: every command with its aliases and arguments,
    /// then how the session works.
    pub(crate) fn help() -> String {
        let mut lines = vec![String::from("Commands:")];
        for command in visible_subcommands(&TREE) {
            let names = std::iter::once(command.get_name())
                .chain(command.get_visible_aliases())
                .collect::<Vec<_>>()
                .join(", ");
            let usage = usage(command);
            let left = if usage.is_empty() {
                names
            } else {
                format!("{names} {usage}")
            };
            let pad = HELP_COLUMN.saturating_sub(left.chars().count()).max(2);
            lines.push(format!("  {left}{:pad$}{}", "", about(command)));
        }
        lines.push(String::from(HELP_TAIL));
        lines.join("\n")
    }
}

fn visible_subcommands(command: &Command) -> impl Iterator<Item = &Command> {
    command.get_subcommands().filter(|sub| !sub.is_hide_set())
}

fn visible_args(command: &Command) -> impl Iterator<Item = &Arg> {
    command
        .get_arguments()
        .filter(|arg| !arg.is_hide_set() && !IMPLIED_ARGS.contains(&arg.get_id().as_str()))
}

/// The first line of a command's description.
fn about(command: &Command) -> String {
    command
        .get_about()
        .map(ToString::to_string)
        .unwrap_or_default()
}

/// What may follow a command's name: its positionals by value name,
/// bracketed when optional, their choices when fixed, `[--flag]` for each
/// flag, and `VERB ...` when it has subcommands.
fn usage(command: &Command) -> String {
    let mut parts: Vec<String> = Vec::new();
    for arg in visible_args(command) {
        let part = if arg.is_positional() {
            let choices: Vec<String> = arg
                .get_possible_values()
                .iter()
                .map(|value| value.get_name().to_owned())
                .collect();
            if choices.is_empty() {
                value_name(arg)
            } else {
                choices.join("|")
            }
        } else {
            match arg.get_long() {
                Some(long) if arg.get_action().takes_values() => {
                    format!("--{long} {}", value_name(arg))
                }
                Some(long) => format!("--{long}"),
                None => continue,
            }
        };
        let optional = !arg.is_required_set() || !arg.is_positional();
        parts.push(if optional { format!("[{part}]") } else { part });
    }
    let mut usage = parts.join(" ");
    let verbs: Vec<&Command> = visible_subcommands(command).collect();
    if !verbs.is_empty() {
        let mut verb = if verbs.len() > INLINE_VERBS {
            String::from("VERB ...")
        } else {
            verbs
                .iter()
                .map(|sub| {
                    let rest = self::usage(sub);
                    if rest.is_empty() {
                        sub.get_name().to_owned()
                    } else {
                        format!("{} {rest}", sub.get_name())
                    }
                })
                .collect::<Vec<_>>()
                .join(" | ")
        };
        let required = command.is_subcommand_required_set()
            || command.is_args_conflicts_with_subcommands_set();
        if !required {
            verb = format!("[{verb}]");
        }
        usage = if usage.is_empty() {
            verb
        } else {
            format!("{usage} | {verb}")
        };
    }
    usage
}

fn value_name(arg: &Arg) -> String {
    arg.get_value_names()
        .and_then(|names| names.first())
        .map_or_else(|| arg.get_id().as_str().to_uppercase(), ToString::to_string)
}

/// Whether anything may follow a command, so accepting it adds a space.
fn takes_more(command: &Command) -> bool {
    visible_args(command).next().is_some() || visible_subcommands(command).next().is_some()
}

/// One entry of the popup.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Suggestion {
    /// What replaces the word being typed.
    pub(crate) word: String,
    /// How the popup shows it, with any arguments it takes.
    pub(crate) label: String,
    pub(crate) about: String,
    /// Whether anything may follow, so accepting it adds a space.
    more: bool,
}

impl Suggestion {
    fn command(command: &Command) -> Self {
        let usage = usage(command);
        let label = if usage.is_empty() {
            command.get_name().to_owned()
        } else {
            format!("{} {usage}", command.get_name())
        };
        Self {
            word: command.get_name().to_owned(),
            label,
            about: about(command),
            more: takes_more(command),
        }
    }

    fn verb(command: &Command) -> Self {
        let name = command.get_name().to_owned();
        Self {
            label: name.clone(),
            word: name,
            about: about(command),
            more: takes_more(command),
        }
    }

    fn flag(arg: &Arg, long: &str) -> Self {
        let flag = format!("--{long}");
        let label = if arg.get_action().takes_values() {
            format!("{flag} {}", value_name(arg))
        } else {
            flag.clone()
        };
        Self {
            word: flag,
            label,
            about: arg.get_help().map(ToString::to_string).unwrap_or_default(),
            more: true,
        }
    }

    /// Whether accepting it leaves nothing more to type.
    pub(crate) fn finishes(&self) -> bool {
        !self.more
    }
}

/// The suggestions for a line being typed, and where the word they
/// complete starts.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Completion {
    start: usize,
    pub(crate) items: Vec<Suggestion>,
}

impl Completion {
    /// What the popup offers for `line`, or `None` when it is not a slash
    /// command or nothing matches. The first word completes to a command;
    /// after it, each word that names a subcommand descends into it, a word
    /// starting with `-` completes to a flag not typed yet, and the first
    /// argument completes to a subcommand or a fixed choice.
    pub(crate) fn for_line(line: &str) -> Option<Self> {
        if !line.starts_with('/') {
            return None;
        }
        let start = line.rfind(' ').map_or(0, |space| space.saturating_add(1));
        let word = line.get(start..)?;
        let items: Vec<Suggestion> = if start == 0 {
            visible_subcommands(&TREE)
                .filter(|c| c.get_name().starts_with(word))
                .map(Suggestion::command)
                .collect()
        } else {
            let mut typed = line.get(..start)?.split_whitespace();
            let mut command = TREE.find_subcommand(typed.next()?)?;
            let mut arguments = 0_usize;
            let mut flags: Vec<&str> = Vec::new();
            for token in typed {
                if token.starts_with('-') {
                    flags.push(token);
                } else if let Some(sub) = command.find_subcommand(token).filter(|_| arguments == 0)
                {
                    command = sub;
                } else {
                    arguments = arguments.saturating_add(1);
                }
            }
            Self::after(command, arguments, &flags, word)
        };
        (!items.is_empty()).then_some(Self { start, items })
    }

    /// What may follow `command` once `arguments` positionals and `flags`
    /// are typed, starting with `word`.
    fn after(command: &Command, arguments: usize, flags: &[&str], word: &str) -> Vec<Suggestion> {
        let mut items = Vec::new();
        if word.starts_with('-') {
            for arg in visible_args(command) {
                if let Some(long) = arg.get_long()
                    && format!("--{long}").starts_with(word)
                    && !flags.iter().any(|f| f.strip_prefix("--") == Some(long))
                {
                    items.push(Suggestion::flag(arg, long));
                }
            }
            return items;
        }
        if arguments > 0 {
            return items;
        }
        for sub in visible_subcommands(command) {
            if sub.get_name().starts_with(word) {
                items.push(Suggestion::verb(sub));
            }
        }
        if let Some(first) = visible_args(command).find(|arg| arg.is_positional()) {
            for value in first.get_possible_values() {
                if !value.is_hide_set() && value.get_name().starts_with(word) {
                    items.push(Suggestion {
                        word: value.get_name().to_owned(),
                        label: value.get_name().to_owned(),
                        about: value
                            .get_help()
                            .map(ToString::to_string)
                            .unwrap_or_default(),
                        more: false,
                    });
                }
            }
        }
        items
    }

    /// The entry at `selected`, clamped to the last.
    pub(crate) fn get(&self, selected: usize) -> Option<&Suggestion> {
        self.items.get(selected).or_else(|| self.items.last())
    }

    /// `line` with the word being typed replaced by `item`, and a space
    /// after it when something may follow.
    pub(crate) fn apply(&self, line: &str, item: &Suggestion) -> String {
        let head = line.get(..self.start).unwrap_or_default();
        let space = if item.more { " " } else { "" };
        format!("{head}{}{space}", item.word)
    }
}

#[cfg(test)]
mod tests {
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
        assert_eq!(words("/sch"), ["/schema"]);
        assert_eq!(
            words("/s"),
            ["/sql", "/schema", "/sessions", "/share", "/steps"]
        );
        assert!(words("/nothing").is_empty());
        assert!(words("hello").is_empty());
        assert!(words("").is_empty());
        assert!(words("/sql SELECT 1").is_empty(), "free text gets nothing");
        assert!(words("/exit ").is_empty());
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
            Some((completion.apply(line, item), item.finishes()))
        };
        assert_eq!(apply("/sch"), Some((String::from("/schema "), false)));
        assert_eq!(apply("/he"), Some((String::from("/help"), true)));
        assert_eq!(
            apply("/graph sta"),
            Some((String::from("/graph status "), false))
        );
        assert_eq!(
            apply("/graph rev"),
            Some((String::from("/graph revalidate"), true))
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
            "/schema TABLE",
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
        assert!(!parses("/schema"));
        assert!(!parses("/graph"));
        assert!(!parses("/mode fast"));
        assert!(!parses("/nothing"));
        assert!(parses("/ontology propose --documents"));
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
            Ok(SlashCommand::Import { url, table, source_table: None, query: Some(q) })
                if url == "sqlite:/tmp/a.db" && table == "t" && q == "SELECT * FROM x WHERE y = 'z'"
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
            SlashCommand::parse("/chart 2"),
            Ok(SlashCommand::Chart { n: Some(2) })
        ));
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
        assert_eq!(Input::classify(format!("'{path}'")), Input::File(file));
        assert_eq!(
            Input::classify(String::from("SELECT 1")),
            Input::Sql(String::from("SELECT 1"))
        );
        assert_eq!(
            Input::classify(String::from("what is in notes.md")),
            Input::Question(String::from("what is in notes.md"))
        );
        assert!(Input::file("/nowhere/notes.md").is_none());
    }
}
