//! Server administration from the command line: users, tokens, members,
//! and the access audit log. Every mutation is itself audited on the `cli`
//! channel with no user, because the operator at the shell is implicit.

use std::io::{IsTerminal, Write};

use anyhow::{Context, Result};
use clap::{Args, Subcommand, ValueEnum};
use quack_core::config::Config;
use quack_core::error::Record;
use quack_core::ids::WorkspaceId;
use quack_core::prefix::PrefixMatch;
use quack_core::storage::control::{
    AuditAction, AuditEntry, AuditFilter, AuditRow, Channel, ControlPlane, Expiry, IssuedToken,
    Outcome, ResourceKind, Role, Scope, UserKind, WorkspaceRow,
};

use crate::text_or_json::TextOrJson;

/// Server administration: users, tokens, membership, and the audit log.
#[derive(Subcommand)]
pub(crate) enum AdminCommand {
    /// Server users: create one or list them
    #[command(subcommand)]
    User(UserAction),

    /// API tokens scoped to a workspace: create, list, or revoke
    #[command(subcommand)]
    Token(TokenAction),

    /// Workspace membership: add, remove, or list members
    #[command(subcommand)]
    Member(MemberAction),

    /// Read the access audit log with filters
    Audit(AuditArgs),
}

impl AdminCommand {
    /// Run it against the control database; token and member commands
    /// act on `workspace`.
    pub(crate) async fn run(self, config: &Config, workspace: Option<&str>) -> Result<()> {
        match self {
            Self::User(action) => run_user(config, action).await,
            Self::Token(action) => run_token(config, workspace, action).await,
            Self::Member(action) => run_member(config, workspace, action).await,
            Self::Audit(args) => run_audit(config, args).await,
        }
    }
}

#[derive(Subcommand)]
pub(crate) enum UserAction {
    /// Create a user; the password is read from the terminal without echo,
    /// or from stdin when stdin is not a terminal
    Add {
        username: String,
        /// Grant the server-wide admin flag
        #[arg(long)]
        admin: bool,
    },
    /// List users
    List {
        #[arg(long)]
        json: bool,
    },
}

#[derive(Subcommand)]
pub(crate) enum TokenAction {
    /// Mint a token for a user in the workspace; the token is shown once
    Create {
        /// The user the token acts as
        #[arg(long)]
        user: String,
        /// A label for the token
        #[arg(long)]
        name: String,
        /// Comma-separated scopes: read, write, admin
        #[arg(long, default_value = "read", value_delimiter = ',')]
        scopes: Vec<Scope>,
        /// Expire after this many days
        #[arg(long, value_name = "DAYS")]
        expires: Option<u32>,
    },
    /// List the workspace's tokens
    List {
        #[arg(long)]
        json: bool,
    },
    /// Revoke a token by its hash (prefixes accepted)
    Revoke { token_hash: String },
}

#[derive(Subcommand)]
pub(crate) enum MemberAction {
    /// Add a user to the workspace, or change their role
    Add {
        username: String,
        #[arg(long, default_value = "member")]
        role: Role,
    },
    /// Remove a user from the workspace
    Remove { username: String },
    /// List members
    List {
        #[arg(long)]
        json: bool,
    },
}

#[derive(Args)]
pub(crate) struct AuditArgs {
    /// Filter by username
    #[arg(long)]
    user: Option<String>,
    /// Filter by workspace name
    #[arg(long, short = 'w')]
    workspace: Option<String>,
    /// Filter by action (open, query, sql, ingest, ...)
    #[arg(long)]
    action: Option<String>,
    /// Filter by outcome: allowed, denied, error
    #[arg(long)]
    outcome: Option<Outcome>,
    /// Rows at or after this time (YYYY-MM-DD HH:MM:SS, UTC)
    #[arg(long)]
    since: Option<String>,
    /// Rows before this time
    #[arg(long)]
    until: Option<String>,
    /// Rows to show, newest first; 0 for the whole log
    #[arg(long, default_value_t = 100)]
    limit: u32,
    #[arg(long, value_enum, default_value_t = AuditFormat::Text)]
    format: AuditFormat,
}

/// How `quack audit` prints.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum AuditFormat {
    /// One line per row
    Text,
    /// One JSON object per row
    Json,
    /// Comma-separated values with a header
    Csv,
    /// One OCSF 1.9.0 event per line, for a SIEM or an archive
    Ocsf,
}

pub(crate) async fn run_user(config: &Config, action: UserAction) -> Result<()> {
    let control = ControlPlane::open(config).await?;
    let stdout = std::io::stdout();
    match action {
        UserAction::Add { username, admin } => {
            let password = read_password(&format!("Password for {username}: "))?;
            let user = control
                .create_user(&username, &password, UserKind::from(admin))
                .await?;
            let entry = AuditEntry::new(AuditAction::Admin, Outcome::Allowed, Channel::Cli)
                .on(ResourceKind::User.id(user.id.as_str()));
            control.record_audit(&entry).await?;
            let mut out = stdout.lock();
            writeln!(
                out,
                "Created user '{}' ({}){}",
                user.username,
                user.id,
                if user.is_admin { ", admin" } else { "" }
            )?;
            out.flush()?;
        }
        UserAction::List { json } => {
            let users = control.list_users().await?;
            let mut out = std::io::BufWriter::new(stdout.lock());
            TextOrJson::of(json).write_rows(
                &mut out,
                &users,
                "No users yet. Run `quack user add NAME`.",
                |out, user| {
                    writeln!(
                        out,
                        "{}  {:<24} {}  {}",
                        user.id,
                        user.username,
                        if user.is_admin { "admin " } else { "      " },
                        user.created_at
                    )
                },
            )?;
            out.flush()?;
        }
    }
    Ok(())
}

pub(crate) async fn run_token(
    config: &Config,
    workspace: Option<&str>,
    action: TokenAction,
) -> Result<()> {
    let control = ControlPlane::open(config).await?;
    let ws = existing_workspace(&control, config, workspace).await?;
    match action {
        TokenAction::Create {
            user,
            name,
            scopes,
            expires,
        } => create_token(&control, &ws, &user, &name, &scopes, expires).await,
        TokenAction::List { json } => list_tokens(&control, &ws, TextOrJson::of(json)).await,
        TokenAction::Revoke { token_hash } => revoke_token(&control, &ws, &token_hash).await,
    }
}

fn scope_list(scopes: &[Scope]) -> String {
    scopes
        .iter()
        .map(|s| s.as_str())
        .collect::<Vec<_>>()
        .join(",")
}

async fn create_token(
    control: &ControlPlane,
    ws: &WorkspaceRow,
    user: &str,
    name: &str,
    scopes: &[Scope],
    expires: Option<u32>,
) -> Result<()> {
    let user_row = control
        .find_user_by_username(user)
        .await?
        .with_context(|| format!("no user named '{user}'"))?;
    let expires_at = expires.map(Expiry::after_days).transpose()?;
    let IssuedToken { secret, row } = control
        .create_token(&ws.id, &user_row.id, name, scopes, expires_at)
        .await?;
    let entry = AuditEntry::new(AuditAction::Token, Outcome::Allowed, Channel::Cli)
        .in_workspace(&ws.id)
        .on(ResourceKind::Token.id(&row.token_hash));
    control.record_audit(&entry).await?;
    let stdout = std::io::stdout();
    let mut out = stdout.lock();
    writeln!(out, "{}", secret.expose())?;
    writeln!(
        out,
        "Token '{}' for {} in workspace '{}' with scopes {}{}. It is shown only once.",
        row.name,
        user_row.username,
        ws.name,
        scope_list(&row.scopes),
        row.expires_at
            .as_deref()
            .map_or(String::new(), |e| format!(", expires {e}"))
    )?;
    out.flush()?;
    Ok(())
}

async fn list_tokens(control: &ControlPlane, ws: &WorkspaceRow, format: TextOrJson) -> Result<()> {
    let tokens = control.list_tokens(&ws.id).await?;
    let stdout = std::io::stdout();
    let mut out = std::io::BufWriter::new(stdout.lock());
    format.write_rows(
        &mut out,
        &tokens,
        &format!("No tokens in workspace '{}'.", ws.name),
        |out, token| {
            writeln!(
                out,
                "{}  {:<16} {:<16} {}  {}",
                token.token_hash,
                token.name,
                scope_list(&token.scopes),
                token.expires_at.as_deref().unwrap_or("no expiry"),
                token.last_used_at.as_deref().unwrap_or("never used")
            )
        },
    )?;
    out.flush()?;
    Ok(())
}

async fn revoke_token(control: &ControlPlane, ws: &WorkspaceRow, prefix: &str) -> Result<()> {
    let hash = PrefixMatch::of(control.list_tokens(&ws.id).await?, prefix, |t| {
        t.token_hash.as_str()
    })
    .one(Record::Token, prefix)?
    .token_hash;
    control.delete_token(&hash).await?;
    let entry = AuditEntry::new(AuditAction::Token, Outcome::Allowed, Channel::Cli)
        .in_workspace(&ws.id)
        .on(ResourceKind::Token.id(&hash));
    control.record_audit(&entry).await?;
    let stdout = std::io::stdout();
    let mut out = stdout.lock();
    writeln!(out, "Revoked token {hash}.")?;
    out.flush()?;
    Ok(())
}

pub(crate) async fn run_member(
    config: &Config,
    workspace: Option<&str>,
    action: MemberAction,
) -> Result<()> {
    let control = ControlPlane::open(config).await?;
    let ws = existing_workspace(&control, config, workspace).await?;
    let stdout = std::io::stdout();
    match action {
        MemberAction::Add { username, role } => {
            let user = control
                .find_user_by_username(&username)
                .await?
                .with_context(|| format!("no user named '{username}'"))?;
            control.set_member(&ws.id, &user.id, role).await?;
            control
                .record_audit(&member_entry(&ws, user.id.as_str()))
                .await?;
            let mut out = stdout.lock();
            writeln!(out, "{} is now {role} of '{}'.", user.username, ws.name)?;
            out.flush()?;
        }
        MemberAction::Remove { username } => {
            let user = control
                .find_user_by_username(&username)
                .await?
                .with_context(|| format!("no user named '{username}'"))?;
            let removed = control.remove_member(&ws.id, &user.id).await?;
            control
                .record_audit(&member_entry(&ws, user.id.as_str()))
                .await?;
            let mut out = stdout.lock();
            if removed {
                writeln!(out, "Removed {} from '{}'.", user.username, ws.name)?;
            } else {
                writeln!(out, "{} was not a member of '{}'.", user.username, ws.name)?;
            }
            out.flush()?;
        }
        MemberAction::List { json } => {
            let members = control.list_members(&ws.id).await?;
            let mut out = std::io::BufWriter::new(stdout.lock());
            TextOrJson::of(json).write_rows(
                &mut out,
                &members,
                &format!("No members in '{}'.", ws.name),
                |out, member| {
                    writeln!(
                        out,
                        "{:<24} {:<8} {}",
                        member.username, member.role, member.created_at
                    )
                },
            )?;
            out.flush()?;
        }
    }
    Ok(())
}

/// The audit row for a membership change to `user_id` in `ws`.
fn member_entry(ws: &WorkspaceRow, user_id: &str) -> AuditEntry {
    AuditEntry::new(AuditAction::Member, Outcome::Allowed, Channel::Cli)
        .in_workspace(&ws.id)
        .on(ResourceKind::User.id(user_id))
}

pub(crate) async fn run_audit(config: &Config, args: AuditArgs) -> Result<()> {
    let format = args.format;
    let control = ControlPlane::open(config).await?;
    let user_id = match args.user.as_deref() {
        Some(name) => Some(
            control
                .find_user_by_username(name)
                .await?
                .with_context(|| format!("no user named '{name}'"))?
                .id,
        ),
        None => None,
    };
    let workspace_id = match args.workspace.as_deref() {
        Some(name) => Some(
            control
                .find_workspace_by_name(name)
                .await?
                .with_context(|| format!("no workspace named '{name}'"))?
                .id,
        ),
        None => None,
    };
    let mut filter = AuditFilter {
        user_id,
        workspace_id,
        action: args.action,
        outcome: args.outcome,
        since: args.since,
        until: args.until,
        limit: AUDIT_PAGE,
        after: None,
    };
    let mut remaining = (args.limit != 0).then_some(args.limit);
    let stdout = std::io::stdout();
    let mut output = AuditOutput::start(format, std::io::BufWriter::new(stdout.lock()))?;
    loop {
        filter.limit = remaining.map_or(AUDIT_PAGE, |left| left.min(AUDIT_PAGE));
        let page = control.query_audit(&filter).await?;
        output.page(&page.rows)?;
        remaining = remaining
            .map(|left| left.saturating_sub(u32::try_from(page.rows.len()).unwrap_or(u32::MAX)));
        match (page.next, remaining) {
            (Some(next), None) => filter.after = Some(next),
            (Some(next), Some(left)) if left > 0 => filter.after = Some(next),
            (Some(_) | None, _) => break,
        }
    }
    output.finish()
}

const AUDIT_PAGE: u32 = 1_000;

/// `quack audit` output, written a page at a time.
enum AuditOutput<W: Write> {
    Csv(Box<csv::Writer<W>>),
    Rows(TextOrJson, W),
    Ocsf(W),
}

impl<W: Write> AuditOutput<W> {
    fn start(format: AuditFormat, out: W) -> Result<Self> {
        Ok(match format {
            AuditFormat::Csv => {
                let mut writer = csv::WriterBuilder::new()
                    .has_headers(false)
                    .from_writer(out);
                writer.write_record([
                    "id",
                    "timestamp",
                    "user_id",
                    "token_hash",
                    "workspace_id",
                    "action",
                    "resource_type",
                    "resource_id",
                    "outcome",
                    "channel",
                    "client_addr",
                    "request_id",
                ])?;
                Self::Csv(Box::new(writer))
            }
            AuditFormat::Text => Self::Rows(TextOrJson::Text, out),
            AuditFormat::Json => Self::Rows(TextOrJson::Json, out),
            AuditFormat::Ocsf => Self::Ocsf(out),
        })
    }

    fn page(&mut self, rows: &[AuditRow]) -> Result<()> {
        match self {
            Self::Csv(writer) => {
                for r in rows {
                    writer.serialize(r)?;
                }
            }
            Self::Ocsf(out) => {
                for r in rows {
                    serde_json::to_writer(&mut *out, &r.to_ocsf()?)?;
                    writeln!(out)?;
                }
            }
            Self::Rows(format, out) => {
                format.write_rows(out, rows, "No audit rows match.", |out, r| {
                    writeln!(
                        out,
                        "{}  {:<7} {:<5} {:<12} {:<10} {:<36} {}",
                        r.timestamp,
                        r.outcome,
                        r.channel,
                        r.action,
                        r.user_id
                            .as_ref()
                            .map_or("-", |u| u.as_str().get(..8).unwrap_or(u.as_str())),
                        r.workspace_id.as_ref().map_or("-", WorkspaceId::as_str),
                        r.resource_id.as_deref().unwrap_or("")
                    )
                })?;
            }
        }
        Ok(())
    }

    fn finish(self) -> Result<()> {
        match self {
            Self::Csv(mut writer) => writer.flush()?,
            Self::Rows(_, mut out) | Self::Ocsf(mut out) => out.flush()?,
        }
        Ok(())
    }
}

/// The named workspace, or the default one, which must already exist: an
/// admin command never creates one by mistyping it.
async fn existing_workspace(
    control: &ControlPlane,
    config: &Config,
    name: Option<&str>,
) -> Result<WorkspaceRow> {
    let name = name.unwrap_or(&config.general.default_workspace);
    control
        .find_workspace_by_name(name)
        .await?
        .with_context(|| format!("no workspace named '{name}'; create it by opening it once"))
}

/// Read a password: without echo from a terminal, else one line from stdin.
fn read_password(prompt: &str) -> Result<String> {
    let stdin = std::io::stdin();
    if !stdin.is_terminal() {
        let mut line = String::new();
        stdin.read_line(&mut line)?;
        return Ok(line.trim_end_matches(['\r', '\n']).to_owned());
    }
    let mut err = std::io::stderr();
    write!(err, "{prompt}")?;
    err.flush()?;
    crossterm::terminal::enable_raw_mode()?;
    let typed = read_hidden_line();
    drop(crossterm::terminal::disable_raw_mode());
    writeln!(err)?;
    typed
}

fn read_hidden_line() -> Result<String> {
    use crossterm::event::{Event, KeyCode, KeyModifiers, read};
    let mut password = String::new();
    loop {
        if let Event::Key(key) = read()? {
            match key.code {
                KeyCode::Enter => return Ok(password),
                KeyCode::Backspace => {
                    password.pop();
                }
                KeyCode::Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                    anyhow::bail!("cancelled");
                }
                KeyCode::Char(c) => password.push(c),
                _ => {}
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(clap::Parser)]
    #[command(no_binary_name = true)]
    enum Line {
        #[command(subcommand)]
        Token(TokenAction),
        #[command(subcommand)]
        Member(MemberAction),
        Audit(AuditArgs),
    }

    /// Roles, scopes, and outcomes are checked by clap as they are typed,
    /// with the values each accepts.
    #[test]
    fn roles_scopes_and_outcomes_parse_at_the_command_line() {
        let parse = |args: &[&str]| <Line as clap::Parser>::try_parse_from(args);
        assert!(matches!(
            parse(&["member", "add", "ann", "--role", "Owner"]),
            Ok(Line::Member(MemberAction::Add {
                role: Role::Owner,
                ..
            }))
        ));
        assert!(matches!(
            parse(&["token", "create", "--user", "u", "--name", "n", "--scopes", "read,write"]),
            Ok(Line::Token(TokenAction::Create { scopes, .. }))
                if scopes == [Scope::Read, Scope::Write]
        ));
        assert!(matches!(
            parse(&["audit", "--outcome", "denied"]),
            Ok(Line::Audit(AuditArgs {
                outcome: Some(Outcome::Denied),
                ..
            }))
        ));
        let refused = parse(&["member", "add", "ann", "--role", "boss"])
            .err()
            .map(|e| e.to_string())
            .unwrap_or_default();
        assert!(
            refused.contains("use one of: viewer, member, owner"),
            "{refused}"
        );
        assert!(
            parse(&[
                "token", "create", "--user", "u", "--name", "n", "--scopes", "root"
            ])
            .is_err()
        );
        assert!(parse(&["audit", "--outcome", "maybe"]).is_err());
    }
}
