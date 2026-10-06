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
use quack_core::storage::workspace::{SqlName, looks_like_direct_sql};

use crate::embeddings_cli::EmbeddingsAction;
use crate::graph_cli::GraphAction;
use crate::ontology_cli::OntologyAction;
use crate::saved_cli::SavedAction;
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
    /// Load a file (a path typed at the prompt or a file dropped on the terminal does the same)
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
    /// The ontology: show, init, propose, review, accept, reject, rename, and versions
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
    /// Saved questions: list them, or add NAME (this session's last answer), run NAME, show NAME, remove NAME
    #[command(name = "/saved")]
    Saved {
        #[command(subcommand)]
        action: Option<SavedAction>,
    },
    /// Pick a recent session to resume
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
    /// Show running, queued, and recent jobs; c cancels one
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
    /// Expand or collapse the tool call details
    #[command(name = "/steps")]
    Steps,
    /// Show the chat and embedding models in use
    #[command(name = "/model")]
    Model,
    /// Clear messages
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
        let command = SlashLine::try_parse_from(words)?.command;
        if let Self::Saved {
            action: Some(SavedAction::Run {
                refresh, exit_code, ..
            }),
        } = &command
            && let Some(flag) = [("--refresh", *refresh), ("--exit-code", *exit_code)]
                .into_iter()
                .find_map(|(flag, given)| given.then_some(flag))
        {
            return Err(SlashLine::command().error(
                ErrorKind::UnknownArgument,
                format!("{flag} is a command-line flag: quack saved run NAME {flag}"),
            ));
        }
        Ok(command)
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
    /// A line of names of files quack can load.
    Files(FileLine),
    /// A statement to run as typed.
    Sql(String),
    /// A question for the agent.
    Question(String),
}

impl Input {
    pub(crate) fn classify(line: String) -> Self {
        // Files first: an absolute path starts with `/` like a command.
        if let Some(files) = FileLine::of(&line) {
            Self::Files(files)
        } else if line.starts_with('/') {
            Self::Command(line)
        } else if looks_like_direct_sql(&line) {
            Self::Sql(line)
        } else {
            Self::Question(line)
        }
    }
}

/// A line that names only files quack can load.
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum FileLine {
    /// Every name on it, as paths.
    Files(Vec<PathBuf>),
    /// Its names end at a word starting with `#`, which the shell split
    /// reads as a comment: loading the names before it would leave files
    /// out without saying so.
    Comment,
}

impl FileLine {
    pub(crate) const COMMENT: &str = "a name starting with # reads as a comment, so it and the                                       names after it would be left out; nothing was loaded.                                       Write it as ./#name or in quotes";

    /// `text` as files, when all of it is files quack can load: one path
    /// as typed, quoted or not, or the shell-quoted paths a terminal
    /// writes for files dropped on it.
    pub(crate) fn of(text: &str) -> Option<Self> {
        let text = text.trim();
        if let Some(path) = Self::file(text.trim_matches('\'').trim_matches('"')) {
            return Some(Self::Files(vec![path]));
        }
        let words = Self::words(text)?;
        let paths: Vec<PathBuf> = words
            .iter()
            .map(|word| Self::file(word))
            .collect::<Option<_>>()?;
        if paths.is_empty() {
            return None;
        }
        // `#` opens a comment only where a word starts, so the line with
        // every `#` made an ordinary character splits into more words
        // exactly when a comment was dropped from it.
        let whole = Self::words(&text.replace('#', "x"));
        Some(if whole.is_some_and(|all| all.len() == words.len()) {
            Self::Files(paths)
        } else {
            Self::Comment
        })
    }

    /// The words of a pasted line as the terminal quoted them: POSIX shell
    /// rules (backslash escapes, `#` comments) where terminals write them,
    /// Windows command-line rules on Windows, where a dropped path is
    /// double-quoted and its backslashes are separators.
    #[cfg(not(windows))]
    fn words(text: &str) -> Option<Vec<String>> {
        shlex::split(text)
    }

    #[cfg(windows)]
    fn words(text: &str) -> Option<Vec<String>> {
        Some(winsplit::split(text))
    }

    /// `name` as a path, when it is a file quack can load: `~/` for the
    /// home directory, relative to the working directory otherwise.
    fn file(name: &str) -> Option<PathBuf> {
        FileType::of(name)?;
        let path = match name.strip_prefix("~/") {
            Some(under_home) => dirs::home_dir()?.join(under_home),
            None => PathBuf::from(name),
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
  /jobs, /sessions  Open a list; Up/Down move, Enter picks, Esc closes
  Enter             Send message
  Up/Down           Browse input history (kept across sessions)
  PageUp/PageDown, mouse wheel   Scroll messages; Home/End jump
  Drag the mouse    Select messages; letting go copies them to the clipboard
  Ctrl+U            Clear input line
  Ctrl+L            Clear screen
  Esc or Ctrl+C     Cancel this session's newest question (running or queued)
  Ctrl+C            Quit (twice while background jobs are still running)";

/// Where `/help` starts each description, when the command fits before it.
const HELP_COLUMN: usize = 18;

/// A command with at most this many verbs spells them out in its usage;
/// more read as `VERB ...` and the popup lists them.
const INLINE_VERBS: usize = 3;

/// Arguments the terminal supplies itself (`--yes`: it never asks), refuses
/// (`/saved run`'s `--exit-code` and `--refresh` are command-line flags),
/// or clap's own, so offering them would mislead.
const IMPLIED_ARGS: &[&str] = &["yes", "help", "exit_code", "refresh"];

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
    follows: Follows,
}

/// What may come after an accepted entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Follows {
    /// More arguments: accepting it adds a space.
    Arguments,
    /// Nothing: Enter sends the line.
    Nothing,
    /// The rest of a statement: accepting it adds nothing and sends nothing.
    Statement,
}

impl Follows {
    const fn after(more: bool) -> Self {
        if more { Self::Arguments } else { Self::Nothing }
    }
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
            follows: Follows::after(takes_more(command)),
        }
    }

    fn verb(command: &Command) -> Self {
        let name = command.get_name().to_owned();
        Self {
            label: name.clone(),
            word: name,
            about: about(command),
            follows: Follows::after(takes_more(command)),
        }
    }

    /// A table or column name, written as a statement needs it.
    pub(crate) fn sql(name: &SqlName, about: &str) -> Self {
        Self {
            word: name.sql.clone(),
            label: name.name.clone(),
            about: about.to_owned(),
            follows: Follows::Statement,
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
            follows: Follows::Arguments,
        }
    }

    /// Whether accepting it leaves nothing more to type.
    pub(crate) fn finishes(&self) -> bool {
        self.follows == Follows::Nothing
    }
}

/// The suggestions for a line being typed, and the byte range of the word
/// they replace: from where it starts to the cursor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Completion {
    start: usize,
    end: usize,
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
        (!items.is_empty()).then(|| Self::at(start, line.len(), items))
    }

    pub(crate) const fn at(start: usize, end: usize, items: Vec<Suggestion>) -> Self {
        Self { start, end, items }
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
                        follows: Follows::Nothing,
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
    /// after it when arguments may follow; with the cursor's character
    /// position after what was filled in.
    pub(crate) fn apply(&self, line: &str, item: &Suggestion) -> (String, usize) {
        let head = line.get(..self.start).unwrap_or_default();
        let tail = line.get(self.end..).unwrap_or_default();
        let space = if item.follows == Follows::Arguments {
            " "
        } else {
            ""
        };
        let filled = format!("{head}{}{space}", item.word);
        let cursor = filled.chars().count();
        (format!("{filled}{tail}"), cursor)
    }
}

#[cfg(test)]
mod tests;
