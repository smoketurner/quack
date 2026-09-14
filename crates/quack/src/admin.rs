//! Server administration from the command line: users, tokens, members,
//! and the access audit log. Every mutation is itself audited on the `cli`
//! channel with no user, because the operator at the shell is implicit.

use std::io::{IsTerminal, Write};

use anyhow::{Context, Result};
use clap::{Args, Subcommand};
use quack_core::config::Config;
use quack_core::storage::control::{
    AuditEntry, AuditFilter, Channel, ControlPlane, Outcome, Role, Scope, WorkspaceRow,
};

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
        scopes: Vec<String>,
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
        role: String,
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
    outcome: Option<String>,
    /// Rows at or after this time (YYYY-MM-DD HH:MM:SS, UTC)
    #[arg(long)]
    since: Option<String>,
    /// Rows before this time
    #[arg(long)]
    until: Option<String>,
    #[arg(long, default_value_t = 100)]
    limit: u32,
    /// One JSON object per row
    #[arg(long, conflicts_with = "csv")]
    json: bool,
    /// Comma-separated values with a header
    #[arg(long)]
    csv: bool,
}

pub(crate) async fn run_user(config: &Config, action: UserAction) -> Result<()> {
    let control = ControlPlane::open(config).await?;
    let stdout = std::io::stdout();
    match action {
        UserAction::Add { username, admin } => {
            let password = read_password(&format!("Password for {username}: "))?;
            let user = control.create_user(&username, &password, admin).await?;
            let mut entry = AuditEntry::new("admin", Outcome::Allowed, Channel::Cli);
            entry.resource_type = Some(String::from("user"));
            entry.resource_id = Some(user.id.clone());
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
            if json {
                for user in &users {
                    serde_json::to_writer(&mut out, user)?;
                    writeln!(out)?;
                }
            } else if users.is_empty() {
                writeln!(out, "No users yet. Run `quack user add NAME`.")?;
            } else {
                for user in &users {
                    writeln!(
                        out,
                        "{}  {:<24} {}  {}",
                        user.id,
                        user.username,
                        if user.is_admin { "admin " } else { "      " },
                        user.created_at
                    )?;
                }
            }
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
    let ws = resolve_workspace(&control, config, workspace).await?;
    match action {
        TokenAction::Create {
            user,
            name,
            scopes,
            expires,
        } => create_token(&control, &ws, &user, &name, &scopes, expires).await,
        TokenAction::List { json } => list_tokens(&control, &ws, json).await,
        TokenAction::Revoke { token_hash } => revoke_token(&control, &ws, &token_hash).await,
    }
}

fn scope_list(scopes: &[Scope]) -> String {
    scopes
        .iter()
        .map(|s| format!("{s:?}").to_ascii_lowercase())
        .collect::<Vec<_>>()
        .join(",")
}

async fn create_token(
    control: &ControlPlane,
    ws: &WorkspaceRow,
    user: &str,
    name: &str,
    scopes: &[String],
    expires: Option<u32>,
) -> Result<()> {
    let user_row = control
        .find_user_by_username(user)
        .await?
        .with_context(|| format!("no user named '{user}'"))?;
    let scopes = scopes
        .iter()
        .map(|s| Scope::parse(s))
        .collect::<quack_core::error::Result<Vec<_>>>()?;
    let expires_at = expires
        .map(|days| {
            jiff::Timestamp::now()
                .checked_add(jiff::SignedDuration::from_hours(
                    i64::from(days).saturating_mul(24),
                ))
                .map(|t| t.strftime("%Y-%m-%d %H:%M:%S").to_string())
        })
        .transpose()
        .context("expiry is too far in the future")?;
    let (token, row) = control
        .create_token(&ws.id, &user_row.id, name, &scopes, expires_at.as_deref())
        .await?;
    let mut entry = AuditEntry::new("token", Outcome::Allowed, Channel::Cli);
    entry.workspace_id = Some(ws.id.clone());
    entry.resource_type = Some(String::from("token"));
    entry.resource_id = Some(row.token_hash.clone());
    control.record_audit(&entry).await?;
    let stdout = std::io::stdout();
    let mut out = stdout.lock();
    writeln!(out, "{token}")?;
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

async fn list_tokens(control: &ControlPlane, ws: &WorkspaceRow, json: bool) -> Result<()> {
    let tokens = control.list_tokens(&ws.id).await?;
    let stdout = std::io::stdout();
    let mut out = std::io::BufWriter::new(stdout.lock());
    if json {
        for token in &tokens {
            serde_json::to_writer(&mut out, token)?;
            writeln!(out)?;
        }
    } else if tokens.is_empty() {
        writeln!(out, "No tokens in workspace '{}'.", ws.name)?;
    } else {
        for token in &tokens {
            writeln!(
                out,
                "{}  {:<16} {:<16} {}  {}",
                token.token_hash,
                token.name,
                scope_list(&token.scopes),
                token.expires_at.as_deref().unwrap_or("no expiry"),
                token.last_used_at.as_deref().unwrap_or("never used")
            )?;
        }
    }
    out.flush()?;
    Ok(())
}

async fn revoke_token(control: &ControlPlane, ws: &WorkspaceRow, prefix: &str) -> Result<()> {
    let matches: Vec<String> = control
        .list_tokens(&ws.id)
        .await?
        .into_iter()
        .filter(|t| t.token_hash.starts_with(prefix))
        .map(|t| t.token_hash)
        .collect();
    let hash = match matches.as_slice() {
        [one] => one.clone(),
        [] => anyhow::bail!("no token in '{}' matches '{prefix}'", ws.name),
        many => anyhow::bail!(
            "'{prefix}' matches {} tokens; use more of the hash",
            many.len()
        ),
    };
    control.delete_token(&hash).await?;
    let mut entry = AuditEntry::new("token", Outcome::Allowed, Channel::Cli);
    entry.workspace_id = Some(ws.id.clone());
    entry.resource_type = Some(String::from("token"));
    entry.resource_id = Some(hash.clone());
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
    let ws = resolve_workspace(&control, config, workspace).await?;
    let stdout = std::io::stdout();
    match action {
        MemberAction::Add { username, role } => {
            let user = control
                .find_user_by_username(&username)
                .await?
                .with_context(|| format!("no user named '{username}'"))?;
            let role = Role::parse(&role)?;
            control.set_member(&ws.id, &user.id, role).await?;
            audit_member(&control, &ws, &user.id).await?;
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
            audit_member(&control, &ws, &user.id).await?;
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
            if json {
                for member in &members {
                    serde_json::to_writer(&mut out, member)?;
                    writeln!(out)?;
                }
            } else if members.is_empty() {
                writeln!(out, "No members in '{}'.", ws.name)?;
            } else {
                for member in &members {
                    writeln!(
                        out,
                        "{:<24} {:<8} {}",
                        member.username, member.role, member.created_at
                    )?;
                }
            }
            out.flush()?;
        }
    }
    Ok(())
}

async fn audit_member(control: &ControlPlane, ws: &WorkspaceRow, user_id: &str) -> Result<()> {
    let mut entry = AuditEntry::new("member", Outcome::Allowed, Channel::Cli);
    entry.workspace_id = Some(ws.id.clone());
    entry.resource_type = Some(String::from("user"));
    entry.resource_id = Some(user_id.to_owned());
    control.record_audit(&entry).await?;
    Ok(())
}

pub(crate) async fn run_audit(config: &Config, args: AuditArgs) -> Result<()> {
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
    let rows = control
        .query_audit(&AuditFilter {
            user_id,
            workspace_id,
            action: args.action,
            outcome: args.outcome,
            since: args.since,
            until: args.until,
            limit: args.limit,
        })
        .await?;
    let stdout = std::io::stdout();
    let mut out = std::io::BufWriter::new(stdout.lock());
    if args.json {
        for row in &rows {
            serde_json::to_writer(&mut out, row)?;
            writeln!(out)?;
        }
    } else if args.csv {
        writeln!(
            out,
            "id,timestamp,user_id,token_hash,workspace_id,action,resource_type,resource_id,outcome,channel,client_addr,request_id"
        )?;
        for r in &rows {
            let fields = [
                r.id.as_str(),
                r.timestamp.as_str(),
                r.user_id.as_deref().unwrap_or(""),
                r.token_hash.as_deref().unwrap_or(""),
                r.workspace_id.as_deref().unwrap_or(""),
                r.action.as_str(),
                r.resource_type.as_deref().unwrap_or(""),
                r.resource_id.as_deref().unwrap_or(""),
                r.outcome.as_str(),
                r.channel.as_str(),
                r.client_addr.as_deref().unwrap_or(""),
                r.request_id.as_deref().unwrap_or(""),
            ];
            let line: Vec<String> = fields.iter().map(|f| csv_field(f)).collect();
            writeln!(out, "{}", line.join(","))?;
        }
    } else if rows.is_empty() {
        writeln!(out, "No audit rows match.")?;
    } else {
        for r in &rows {
            writeln!(
                out,
                "{}  {:<7} {:<5} {:<12} {:<10} {:<36} {}",
                r.timestamp,
                r.outcome,
                r.channel,
                r.action,
                r.user_id
                    .as_deref()
                    .map_or("-", |u| u.get(..8).unwrap_or(u)),
                r.workspace_id.as_deref().unwrap_or("-"),
                r.resource_id.as_deref().unwrap_or("")
            )?;
        }
    }
    out.flush()?;
    Ok(())
}

fn csv_field(value: &str) -> String {
    if value.contains([',', '"', '\n']) {
        format!("\"{}\"", value.replace('"', "\"\""))
    } else {
        value.to_owned()
    }
}

async fn resolve_workspace(
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

    #[test]
    fn csv_fields_are_quoted_only_when_needed() {
        assert_eq!(csv_field("plain"), "plain");
        assert_eq!(csv_field("a,b"), "\"a,b\"");
        assert_eq!(csv_field("say \"hi\""), "\"say \"\"hi\"\"\"");
    }
}
