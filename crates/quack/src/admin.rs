//! Server administration from the command line: workspaces, users, tokens,
//! members, and the access audit log. Every mutation is itself audited on the `cli`
//! channel with no user, because the operator at the shell is implicit.

use std::io::{IsTerminal, Read, Write};
use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::{Args, Subcommand, ValueEnum};
use quack_core::config::Config;
use quack_core::error::Error as CoreError;

use quack_core::ids::{UserId, WorkspaceId};
use quack_core::ocsf::PromptText;
use quack_core::prefix::PrefixMatch;
use quack_core::storage::audit;
use quack_core::storage::backup::{Described, Manifest, RestoreRequest};
use quack_core::storage::control::{
    AuditAction, AuditEntry, AuditFilter, AuditRow, Channel, ControlPlane, Expiry, GrantedBy,
    IssuedToken, Outcome, ResourceKind, Role, Scope, UserKind, WorkspaceName, WorkspaceRow,
};
use quack_core::storage::workspace::WorkspaceDb;

use crate::confirm::Confirm;
use crate::text_or_json::TextOrJson;

/// Server administration: workspaces, users, tokens, membership, and the
/// audit log.
#[derive(Subcommand)]
pub(crate) enum AdminCommand {
    /// Workspaces: create, list, rename, delete, snapshot, or restore one
    #[command(subcommand)]
    Workspace(WorkspaceAction),

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
            Self::Workspace(action) => action.run(config).await,
            Self::User(action) => action.run(config).await,
            Self::Token(action) => action.run(config, workspace).await,
            Self::Member(action) => action.run(config, workspace).await,
            Self::Audit(args) => args.run(config).await,
        }
    }
}

#[derive(Subcommand)]
pub(crate) enum WorkspaceAction {
    /// Create a workspace; `-w NAME` then names it on every command
    Create {
        /// Non-empty, with no slashes or dots
        name: WorkspaceName,
    },
    /// List workspaces
    List {
        /// `json` prints one JSON object per row
        #[arg(long, value_enum, default_value_t = TextOrJson::Text)]
        format: TextOrJson,
    },
    /// Give a workspace a new name; `-w` and the URL bar then use it
    Rename {
        /// The workspace's current name
        name: String,
        /// Non-empty, with no slashes or dots
        new_name: WorkspaceName,
    },
    /// Delete a workspace: its file, its members, and its API tokens;
    /// the access audit log keeps its rows
    Delete {
        name: String,
        /// Delete without asking
        #[arg(short = 'y', long)]
        yes: bool,
    },
    /// Write a workspace as one tar: its file after a checkpoint, its
    /// uploaded files, and a manifest naming its members and settings
    Snapshot {
        name: String,
        /// Write the tar here instead of stdout
        #[arg(long, value_name = "FILE")]
        to: Option<PathBuf>,
    },
    /// A snapshot's tar as a new workspace: members this server has users
    /// for get their role again
    Restore {
        /// The tar, or `-` for stdin
        file: PathBuf,
        /// The new workspace's name; the snapshot's own when absent
        #[arg(long)]
        name: Option<WorkspaceName>,
    },
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
        /// `json` prints one JSON object per row
        #[arg(long, value_enum, default_value_t = TextOrJson::Text)]
        format: TextOrJson,
    },
    /// Refuse the user's logins and credentials from now on; `quack serve`
    /// ends their sessions at their next request
    Disable { username: String },
    /// Let a disabled user log in again, and clear any lockout
    Enable { username: String },
    /// Set a new password, read from the terminal without echo, or from
    /// stdin when stdin is not a terminal
    Passwd { username: String },
    /// Give the user the server-wide admin flag, or take it with `--off`
    Admin {
        username: String,
        #[arg(long)]
        off: bool,
    },
    /// Delete the user: their memberships, tokens, and stored sign-in go,
    /// and each workspace file replaces their name with "removed"; the
    /// access audit keeps its rows
    Remove {
        username: String,
        /// Delete without asking
        #[arg(short = 'y', long)]
        yes: bool,
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
        /// `json` prints one JSON object per row
        #[arg(long, value_enum, default_value_t = TextOrJson::Text)]
        format: TextOrJson,
    },
    /// Revoke a token by its hash (prefixes accepted)
    Revoke { token_hash: String },
}

#[derive(Subcommand)]
pub(crate) enum MemberAction {
    /// Add a user to the workspace, or change their role; with `--group`,
    /// give an identity-provider group the role instead
    Add {
        /// The user; absent with `--group`
        #[arg(required_unless_present = "group", conflicts_with = "group")]
        username: Option<String>,
        /// The identity provider's group (`[server.oidc].groups_claim`)
        #[arg(long)]
        group: Option<String>,
        #[arg(long, default_value = "member")]
        role: Role,
    },
    /// Remove a user from the workspace, or with `--group` a group's role
    Remove {
        /// The user; absent with `--group`
        #[arg(required_unless_present = "group", conflicts_with = "group")]
        username: Option<String>,
        /// The identity provider's group
        #[arg(long)]
        group: Option<String>,
    },
    /// List members
    List {
        /// `json` prints one JSON object per row
        #[arg(long, value_enum, default_value_t = TextOrJson::Text)]
        format: TextOrJson,
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
    /// With -w and --format ocsf: join each row to the workspace's own
    /// audit detail, so query events carry the model, the tools, and the
    /// documents cited (the OCSF `ai_operation` profile)
    #[arg(long, requires = "workspace")]
    detail: bool,
    /// With --detail: carry each question's text on its event
    #[arg(long, requires = "detail")]
    with_prompt: bool,
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

impl WorkspaceAction {
    pub(crate) async fn run(self, config: &Config) -> Result<()> {
        let control = ControlPlane::open(config).await?;
        let stdout = std::io::stdout();
        match self {
            Self::Create { name } => {
                let entry = AuditEntry::new(AuditAction::Workspace, Outcome::Allowed, Channel::Cli);
                let ws = control.create_workspace(&name, None, entry).await?;
                let mut out = stdout.lock();
                writeln!(out, "Created workspace '{}' ({})", ws.name, ws.id)?;
                out.flush()?;
            }
            Self::List { format } => {
                let workspaces = control.list_workspaces().await?;
                let mut out = std::io::BufWriter::new(stdout.lock());
                format.write_rows(
                    &mut out,
                    &workspaces,
                    "No workspaces yet. Run `quack workspace create NAME`.",
                    |out, ws| writeln!(out, "{}  {:<24} {}", ws.id, ws.name, ws.classification),
                )?;
                out.flush()?;
            }
            Self::Rename { name, new_name } => {
                let ws = control.workspace_named(&name).await?;
                let entry = AuditEntry::new(AuditAction::Workspace, Outcome::Allowed, Channel::Cli);
                let ws = control.rename_workspace(&ws.id, &new_name, entry).await?;
                let mut out = stdout.lock();
                writeln!(out, "Renamed '{name}' to '{}' ({})", ws.name, ws.id)?;
                out.flush()?;
            }
            Self::Delete { name, yes } => {
                let ws = control.workspace_named(&name).await?;
                let mut out = stdout.lock();
                let question = format!(
                    "Delete workspace '{name}' ({}), its file, its members, and its tokens?",
                    ws.id
                );
                if !Confirm::Ask.ask_to_drop(yes, &mut out, &question)? {
                    writeln!(out, "Nothing deleted.")?;
                    return Ok(());
                }
                let entry = AuditEntry::new(AuditAction::Delete, Outcome::Allowed, Channel::Cli);
                control.delete_workspace(&ws.id, entry).await?;
                let dir = config.workspace_dir(ws.id.as_str());
                if dir.exists() {
                    std::fs::remove_dir_all(&dir)
                        .with_context(|| format!("failed to remove {}", dir.display()))?;
                }
                writeln!(out, "Deleted workspace '{name}' ({})", ws.id)?;
                out.flush()?;
            }
            Self::Snapshot { name, to } => {
                Self::snapshot(config, &control, &name, to).await?;
            }
            Self::Restore { file, name } => {
                Self::restore(config, &control, file, name).await?;
            }
        }
        Ok(())
    }

    /// `quack workspace snapshot`: the tar to `to`, or to a piped stdout.
    async fn snapshot(
        config: &Config,
        control: &ControlPlane,
        name: &str,
        to: Option<PathBuf>,
    ) -> Result<()> {
        let stdout = std::io::stdout();
        let ws = control.workspace_named(name).await?;
        let described = Described::of(control, &ws).await?;
        let dir = config.workspace_dir(ws.id.as_str());
        if to.is_none() && stdout.is_terminal() {
            anyhow::bail!("stdout is a terminal; pipe the tar somewhere or use --to FILE");
        }
        let open_config = config.clone();
        let out_path = to.clone();
        // The file is copied closed: Windows lets no other handle open a
        // `DuckDB` file in use (#448).
        tokio::task::spawn_blocking(move || {
            let db = WorkspaceDb::open(&open_config, ws.id.as_str())?;
            let manifest = Manifest::of(&db, described)?;
            db.checkpoint()?;
            drop(db);
            match out_path {
                Some(path) => {
                    let file = std::fs::File::create(&path)?;
                    manifest.write(&dir, file)?.sync_all()?;
                }
                None => manifest.write(&dir, std::io::stdout().lock())?.flush()?,
            }
            Ok::<_, CoreError>(())
        })
        .await
        .context("the snapshot task failed")??;
        let Some(path) = to else {
            return Ok(());
        };
        let mut out = stdout.lock();
        writeln!(out, "Wrote '{name}' to {}", path.display())?;
        out.flush()?;
        Ok(())
    }

    /// `quack workspace restore`: a new workspace from `file` (`-` is stdin).
    async fn restore(
        config: &Config,
        control: &ControlPlane,
        file: PathBuf,
        name: Option<WorkspaceName>,
    ) -> Result<()> {
        let stdout = std::io::stdout();
        let audit = |action| AuditEntry::new(action, Outcome::Allowed, Channel::Cli);
        let source = if file.as_os_str() == "-" {
            let mut bytes = Vec::new();
            std::io::stdin().lock().read_to_end(&mut bytes)?;
            Source::Bytes(bytes)
        } else {
            Source::File(file)
        };
        let restored = RestoreRequest {
            name,
            owner: None,
            audit: &audit,
        }
        .run(control, config, move || source.open())
        .await?;
        let mut out = stdout.lock();
        writeln!(
            out,
            "Restored '{}' ({}) from a snapshot taken {} by quack {}",
            restored.workspace.name,
            restored.workspace.id,
            restored.manifest.taken_at,
            restored.manifest.quack_version
        )?;
        writeln!(out, "{} member(s) kept their role", restored.members_kept)?;
        if !restored.members_missing.is_empty() {
            writeln!(
                out,
                "no user here for: {} (`quack user add` them, then `quack member add`)",
                restored.members_missing.join(", ")
            )?;
        }
        if !restored.providers_dropped.is_empty() {
            writeln!(
                out,
                "allowed providers not configured here, dropped: {}",
                restored.providers_dropped.join(", ")
            )?;
        }
        out.flush()?;
        Ok(())
    }
}

/// Where a restore reads its tar from, twice: the manifest, then the files.
enum Source {
    File(PathBuf),
    Bytes(Vec<u8>),
}

impl Source {
    fn open(&self) -> std::io::Result<Box<dyn Read + Send>> {
        Ok(match self {
            Self::File(path) => Box::new(std::fs::File::open(path)?),
            Self::Bytes(bytes) => Box::new(std::io::Cursor::new(bytes.clone())),
        })
    }
}

impl UserAction {
    pub(crate) async fn run(self, config: &Config) -> Result<()> {
        let control = ControlPlane::open(config).await?;
        let stdout = std::io::stdout();
        match self {
            Self::Add { username, admin } => {
                let password = read_password(&format!("Password for {username}: "))?;
                let entry = AuditEntry::new(AuditAction::Admin, Outcome::Allowed, Channel::Cli);
                let user = control
                    .create_user(&username, &password, UserKind::from(admin), entry)
                    .await?;
                let mut out = stdout.lock();
                writeln!(
                    out,
                    "Created user '{}' ({}){}",
                    user.username,
                    user.id,
                    if user.kind == UserKind::Admin {
                        ", admin"
                    } else {
                        ""
                    }
                )?;
                out.flush()?;
            }
            Self::List { format } => {
                let users = control.list_users().await?;
                let mut out = std::io::BufWriter::new(stdout.lock());
                format.write_rows(
                    &mut out,
                    &users,
                    "No users yet. Run `quack user add NAME`.",
                    |out, user| {
                        writeln!(
                            out,
                            "{}  {:<24} {}  {}",
                            user.id,
                            user.username,
                            if user.kind == UserKind::Admin {
                                "admin "
                            } else {
                                "      "
                            },
                            user.created_at
                        )
                    },
                )?;
                out.flush()?;
            }
            Self::Disable { username } => {
                let user = control.user_named(&username).await?;
                let entry = AuditEntry::new(AuditAction::Admin, Outcome::Allowed, Channel::Cli);
                control.disable_user(&user.id, entry).await?;
                let mut out = stdout.lock();
                writeln!(out, "Disabled '{}' ({})", user.username, user.id)?;
                out.flush()?;
            }
            Self::Enable { username } => {
                let user = control.user_named(&username).await?;
                let entry = AuditEntry::new(AuditAction::Admin, Outcome::Allowed, Channel::Cli);
                control.enable_user(&user.id, entry).await?;
                let mut out = stdout.lock();
                writeln!(out, "Enabled '{}' ({})", user.username, user.id)?;
                out.flush()?;
            }
            Self::Passwd { username } => {
                let user = control.user_named(&username).await?;
                let password = read_password(&format!("New password for {username}: "))?;
                let entry = AuditEntry::new(AuditAction::Admin, Outcome::Allowed, Channel::Cli);
                control.set_password(&user.id, &password, entry).await?;
                let mut out = stdout.lock();
                writeln!(out, "Changed the password of '{}'", user.username)?;
                out.flush()?;
            }
            Self::Admin { username, off } => {
                let user = control.user_named(&username).await?;
                let kind = if off {
                    UserKind::Standard
                } else {
                    UserKind::Admin
                };
                let entry = AuditEntry::new(AuditAction::Admin, Outcome::Allowed, Channel::Cli);
                control.set_admin(&user.id, kind, entry).await?;
                let mut out = stdout.lock();
                writeln!(
                    out,
                    "'{}' is {} an admin",
                    user.username,
                    if off { "no longer" } else { "now" }
                )?;
                out.flush()?;
            }
            Self::Remove { username, yes } => {
                Self::remove(config, &control, &username, yes).await?;
            }
        }
        Ok(())
    }

    /// `quack user remove`: the row, then the user's name in every workspace
    /// file. A workspace file another process holds (a running `quack serve`)
    /// is reported; delete through the server's API or page while it runs.
    async fn remove(
        config: &Config,
        control: &ControlPlane,
        username: &str,
        yes: bool,
    ) -> Result<()> {
        let user = control.user_named(username).await?;
        let stdout = std::io::stdout();
        let mut out = stdout.lock();
        let question = format!(
            "Remove user '{}' ({}), their memberships, tokens, and stored sign-in?",
            user.username, user.id
        );
        if !Confirm::Ask.ask_to_drop(yes, &mut out, &question)? {
            writeln!(out, "Nothing removed.")?;
            return Ok(());
        }
        // Every workspace forgets the user before the account goes: one that
        // cannot be opened stops here with the user still there, and the
        // command can be run again.
        for ws in control.list_workspaces().await? {
            let db = WorkspaceDb::open(config, ws.id.as_str()).with_context(|| {
                format!(
                    "workspace '{}' could not be opened, so user '{}' was not removed; \
                     run the command again once it opens",
                    ws.name, user.username
                )
            })?;
            let changed = db.forget_user(&user.id, &user.username)?;
            if changed > 0 {
                writeln!(out, "'{}': {changed} row(s) now name \"removed\"", ws.name)?;
            }
        }
        let entry = AuditEntry::new(AuditAction::Admin, Outcome::Allowed, Channel::Cli);
        control.delete_user(&user.id, entry).await?;
        writeln!(out, "Removed user '{}' ({})", user.username, user.id)?;
        out.flush()?;
        Ok(())
    }
}

impl TokenAction {
    pub(crate) async fn run(self, config: &Config, workspace: Option<&str>) -> Result<()> {
        let control = ControlPlane::open(config).await?;
        let ws = control
            .workspace_or_default(workspace, &config.general.default_workspace)
            .await?;
        match self {
            Self::Create {
                user,
                name,
                scopes,
                expires,
            } => Self::create(&control, &ws, &user, &name, &scopes, expires).await,
            Self::List { format } => Self::list(&control, &ws, format).await,
            Self::Revoke { token_hash } => Self::revoke(&control, &ws, &token_hash).await,
        }
    }

    async fn create(
        control: &ControlPlane,
        ws: &WorkspaceRow,
        user: &str,
        name: &str,
        scopes: &[Scope],
        expires: Option<u32>,
    ) -> Result<()> {
        let user_row = control.user_named(user).await?;
        let expires_at = expires.map(Expiry::after_days).transpose()?;
        let entry = AuditEntry::new(AuditAction::Token, Outcome::Allowed, Channel::Cli);
        let IssuedToken { secret, row } = control
            .create_token(&ws.id, &user_row.id, name, scopes, expires_at, entry)
            .await?;
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

    async fn list(control: &ControlPlane, ws: &WorkspaceRow, format: TextOrJson) -> Result<()> {
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

    async fn revoke(control: &ControlPlane, ws: &WorkspaceRow, prefix: &str) -> Result<()> {
        let hash = PrefixMatch::of(control.list_tokens(&ws.id).await?, prefix, |t| {
            t.token_hash.as_str()
        })
        .one(ResourceKind::Token, prefix)?
        .token_hash;
        let entry = AuditEntry::new(AuditAction::Token, Outcome::Allowed, Channel::Cli)
            .in_workspace(&ws.id);
        control.delete_token(&hash, entry).await?;
        let stdout = std::io::stdout();
        let mut out = stdout.lock();
        writeln!(out, "Revoked token {hash}.")?;
        out.flush()?;
        Ok(())
    }
}

fn scope_list(scopes: &[Scope]) -> String {
    scopes
        .iter()
        .map(|s| s.as_str())
        .collect::<Vec<_>>()
        .join(",")
}

impl MemberAction {
    pub(crate) async fn run(self, config: &Config, workspace: Option<&str>) -> Result<()> {
        let control = ControlPlane::open(config).await?;
        let ws = control
            .workspace_or_default(workspace, &config.general.default_workspace)
            .await?;
        let stdout = std::io::stdout();
        match self {
            Self::Add {
                group: Some(group),
                role,
                ..
            } => Self::group_role(&control, &ws, &group, Some(role)).await?,
            Self::Add {
                username,
                group: None,
                role,
            } => {
                let username = username.unwrap_or_default();
                let user = control.user_named(&username).await?;
                control
                    .set_member(
                        &ws.id,
                        &user.id,
                        role,
                        AuditEntry::new(AuditAction::Member, Outcome::Allowed, Channel::Cli),
                    )
                    .await?;
                let mut out = stdout.lock();
                writeln!(out, "{} is now {role} of '{}'.", user.username, ws.name)?;
                out.flush()?;
            }
            Self::Remove {
                group: Some(group), ..
            } => Self::group_role(&control, &ws, &group, None).await?,
            Self::Remove {
                username,
                group: None,
            } => {
                let username = username.unwrap_or_default();
                let user = control.user_named(&username).await?;
                let removed = control
                    .remove_member(
                        &ws.id,
                        &user.id,
                        AuditEntry::new(AuditAction::Member, Outcome::Allowed, Channel::Cli),
                    )
                    .await?;
                let mut out = stdout.lock();
                if removed {
                    writeln!(out, "Removed {} from '{}'.", user.username, ws.name)?;
                } else {
                    writeln!(out, "{} was not a member of '{}'.", user.username, ws.name)?;
                }
                out.flush()?;
            }
            Self::List { format } => {
                let members = control.list_members(&ws.id).await?;
                let mut out = std::io::BufWriter::new(stdout.lock());
                format.write_rows(
                    &mut out,
                    &members,
                    &format!("No members in '{}'.", ws.name),
                    |out, member| {
                        writeln!(
                            out,
                            "{:<24} {:<8} {}{}",
                            member.username,
                            member.role,
                            member.created_at,
                            if member.granted_by == GrantedBy::Idp {
                                "  (via group)"
                            } else {
                                ""
                            }
                        )
                    },
                )?;
                let groups = control.list_group_roles(&ws.id).await?;
                if !groups.is_empty() && format == TextOrJson::Text {
                    writeln!(out, "\nGroups granting a role at sign-in:")?;
                    for g in &groups {
                        writeln!(out, "{:<24} {:<8} {}", g.group_name, g.role, g.created_at)?;
                    }
                }
                out.flush()?;
            }
        }
        Ok(())
    }

    /// `quack member add|remove --group`: give the group `role` here, or take
    /// its role away.
    async fn group_role(
        control: &ControlPlane,
        ws: &WorkspaceRow,
        group: &str,
        role: Option<Role>,
    ) -> Result<()> {
        let stdout = std::io::stdout();
        let mut out = stdout.lock();
        let entry = AuditEntry::new(AuditAction::Member, Outcome::Allowed, Channel::Cli);
        match role {
            Some(role) => {
                let row = control.set_group_role(&ws.id, group, role, entry).await?;
                writeln!(
                    out,
                    "group '{}' now grants {} in '{}' at sign-in.",
                    row.group_name, row.role, ws.name
                )?;
            }
            None => {
                if control.remove_group_role(&ws.id, group, entry).await? {
                    writeln!(
                        out,
                        "group '{group}' no longer grants a role in '{}'.",
                        ws.name
                    )?;
                } else {
                    writeln!(out, "group '{group}' had no role in '{}'.", ws.name)?;
                }
            }
        }
        out.flush()?;
        Ok(())
    }
}

impl AuditArgs {
    /// The audit row for a membership change to `user_id` in `ws`.
    pub(crate) async fn run(self, config: &Config) -> Result<()> {
        let format = self.format;
        let control = ControlPlane::open(config).await?;
        let user_id = match self.user.as_deref() {
            Some(name) => Some(control.user_named(name).await?.id),
            None => None,
        };
        let workspace_id = match self.workspace.as_deref() {
            Some(name) => Some(control.workspace_named(name).await?.id),
            None => None,
        };
        if self.detail {
            let Some(workspace_id) = workspace_id else {
                anyhow::bail!("--detail needs -w WORKSPACE");
            };
            if format != AuditFormat::Ocsf {
                anyhow::bail!("--detail prints OCSF events; add --format ocsf");
            }
            return self.with_detail(config, &control, &workspace_id).await;
        }
        let mut filter = AuditFilter {
            user_id,
            workspace_id,
            action: self.action,
            outcome: self.outcome,
            since: self.since,
            until: self.until,
            limit: AUDIT_PAGE,
            cursor: None,
        };
        let mut remaining = (self.limit != 0).then_some(self.limit);
        let stdout = std::io::stdout();
        let mut output = AuditOutput::start(format, std::io::BufWriter::new(stdout.lock()))?;
        loop {
            filter.limit = remaining.map_or(AUDIT_PAGE, |left| left.min(AUDIT_PAGE));
            let page = control.query_audit(&filter).await?;
            output.page(&page.rows)?;
            remaining = remaining.map(|left| {
                left.saturating_sub(u32::try_from(page.rows.len()).unwrap_or(u32::MAX))
            });
            match (page.next, remaining) {
                (Some(next), None) => filter.cursor = Some(next),
                (Some(next), Some(left)) if left > 0 => filter.cursor = Some(next),
                (Some(_) | None, _) => break,
            }
        }
        output.finish()
    }

    /// `quack audit -w ws --detail --format ocsf`: the workspace's detail rows
    /// (its own file, opened here) joined to their access rows, one OCSF event
    /// per line. `--limit 0` reads every detail row.
    async fn with_detail(
        &self,
        config: &Config,
        control: &ControlPlane,
        workspace_id: &WorkspaceId,
    ) -> Result<()> {
        let db = WorkspaceDb::open(config, workspace_id.as_str())?;
        let limit = if self.limit == 0 {
            u32::MAX
        } else {
            self.limit
        };
        let details = audit::list(&db, limit)?;
        drop(db);
        let events = control
            .ocsf_events(&details, PromptText::from(self.with_prompt))
            .await?;
        let stdout = std::io::stdout();
        let mut out = std::io::BufWriter::new(stdout.lock());
        for event in &events {
            writeln!(out, "{}", serde_json::to_string(event)?)?;
        }
        out.flush()?;
        Ok(())
    }
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
                    let e = &r.entry;
                    writer.write_record([
                        e.id.as_str(),
                        r.timestamp.as_str(),
                        e.user_id.as_ref().map_or("", UserId::as_str),
                        e.token_hash.as_deref().unwrap_or(""),
                        e.workspace_id.as_ref().map_or("", WorkspaceId::as_str),
                        e.action.as_str(),
                        e.resource_type.as_ref().map_or("", ResourceKind::as_str),
                        e.resource_id.as_deref().unwrap_or(""),
                        e.outcome.as_str(),
                        e.origin.channel.as_str(),
                        e.origin.client_addr.as_deref().unwrap_or(""),
                        e.origin.request_id.as_deref().unwrap_or(""),
                    ])?;
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
                        r.entry.outcome,
                        r.entry.origin.channel,
                        r.entry.action,
                        r.entry
                            .user_id
                            .as_ref()
                            .map_or("-", |u| u.as_str().get(..8).unwrap_or(u.as_str())),
                        r.entry
                            .workspace_id
                            .as_ref()
                            .map_or("-", WorkspaceId::as_str),
                        r.entry.resource_id.as_deref().unwrap_or("")
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
    read_hidden_line_from(crossterm::event::read)
}

/// Read one line of input without echo until Enter. `next` is the source of
/// keyboard events: production passes `crossterm::event::read`, and tests feed
/// scripted events so the control-key handling can be exercised without a real
/// terminal.
///
/// Control-letter combos (Ctrl-A, Ctrl-D, Ctrl-E, Ctrl-U, Ctrl-W, ...) arrive
/// from crossterm 0.29 on Unix as `KeyCode::Char(letter)` with
/// `KeyModifiers::CONTROL`, not as a distinct keycode. They are intentionally
/// ignored here rather than appended to the buffer, so an incidental Ctrl-combo
/// while typing a password can never silently corrupt the stored credential.
/// `Ctrl-C` still cancels, and `Shift` is unaffected so capital letters typed
/// with `Shift` are still appended.
///
/// On Windows, `AltGr` is reported as CONTROL and ALT together, and it is how
/// `@`, `{`, or `€` are typed on many layouts, so a character with both is
/// kept. Windows also reports key releases; only presses count, or every
/// character and every backspace would happen twice.
fn read_hidden_line_from<F>(mut next: F) -> Result<String>
where
    F: FnMut() -> std::io::Result<crossterm::event::Event>,
{
    use crossterm::event::{Event, KeyCode, KeyEventKind, KeyModifiers};
    let mut password = String::new();
    loop {
        let Event::Key(key) = next()? else { continue };
        if key.kind != KeyEventKind::Press {
            continue;
        }
        // Control without Alt: CONTROL|ALT is AltGr on Windows, a character.
        let control = key.modifiers.contains(KeyModifiers::CONTROL)
            && !key.modifiers.contains(KeyModifiers::ALT);
        match key.code {
            KeyCode::Enter => return Ok(password),
            KeyCode::Backspace => {
                password.pop();
            }
            KeyCode::Char('c') if control => anyhow::bail!("cancelled"),
            // A Ctrl combo is not part of the password.
            KeyCode::Char(_) if control => {}
            KeyCode::Char(c) => password.push(c),
            _ => {}
        }
    }
}

#[cfg(test)]
mod tests;
