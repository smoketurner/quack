//! The Saved page (`/w/{id}/saved`, design doc 8.1): the workspace's
//! saved questions, each run again without the model or removed, and the
//! chat's form that saves a session's last answer. The terminal's `/saved`
//! and `quack saved` do the same through `saved_cli`.

use askama::Template;
use axum::Form;
use axum::extract::{Path, State};
use axum::response::{IntoResponse, Response};
use quack_core::ids::{SavedId, SessionId, WorkspaceId};
use quack_core::saved::{SavedQuestion, SavedRun};
use quack_core::storage::control::UserKind;
use quack_core::storage::workspace::Cell;
use serde::Deserialize;

use super::flash::{Flash, Flashed};
use super::{Page, Tab, WebResult, WebUser, html};
use crate::server::api::saved::RunOf;
use crate::server::auth::{Access, Need};
use crate::server::state::App;

/// A saved question as the page lists it.
struct SavedView {
    question: SavedQuestion,
    /// Whether the person may remove it: its creator, or an owner.
    removable: bool,
}

/// One statement of a run, with its rows as text cells when they were
/// kept.
struct StatementView {
    sql: String,
    columns: Vec<String>,
    rows: Vec<Vec<String>>,
    outcome: String,
}

/// The run the page shows above the list, after Run.
struct RunView {
    name: String,
    verdict: &'static str,
    statements: Vec<StatementView>,
}

impl RunView {
    fn of(name: String, run: &SavedRun) -> Self {
        Self {
            name,
            verdict: run.verdict(),
            statements: run
                .statements
                .iter()
                .map(|statement| StatementView {
                    sql: statement.sql.trim().to_owned(),
                    columns: statement.columns.clone(),
                    rows: statement
                        .result
                        .iter()
                        .flatten()
                        .map(|row| row.iter().map(|v| Cell(v).text()).collect())
                        .collect(),
                    outcome: if statement.result.is_none() && statement.error.is_none() {
                        format!("{} (past the row cap, so not kept)", statement.outcome())
                    } else {
                        statement.outcome()
                    },
                })
                .collect(),
        }
    }
}

#[derive(Template)]
#[template(path = "saved.html")]
struct SavedPage {
    page: Page,
    saved: Vec<SavedView>,
    run: Option<RunView>,
    error: Option<String>,
    notice: Option<String>,
}

impl SavedPage {
    async fn render(
        app: &App,
        access: &Access,
        run: Option<RunView>,
        flash: &Flashed,
    ) -> WebResult<Response> {
        let saved = access
            .list_saved(app)
            .await?
            .into_iter()
            .map(|question| SavedView {
                removable: access.owns(question.created_by.as_ref()),
                question,
            })
            .collect();
        html(&Self {
            page: Page::in_workspace(app, Tab::Saved, access),
            saved,
            run,
            error: flash.error(),
            notice: flash.notice(),
        })
    }
}

pub(super) async fn saved_page(
    State(app): State<App>,
    WebUser(identity): WebUser,
    Path(id): Path<WorkspaceId>,
    flash: Flashed,
) -> WebResult<Response> {
    let access = Access::resolve(&app, identity, &id, Need::READ).await?;
    SavedPage::render(&app, &access, None, &flash).await
}

/// The chat's Save form: a name for the session's last answer.
#[derive(Deserialize)]
pub(super) struct SaveForm {
    name: String,
    session: SessionId,
}

pub(super) async fn save(
    State(app): State<App>,
    WebUser(identity): WebUser,
    Path(id): Path<WorkspaceId>,
    Form(form): Form<SaveForm>,
) -> WebResult<Response> {
    let access = Access::resolve(&app, identity, &id, Need::READ).await?;
    let saved = access
        .save_answer(&app, form.name, &form.session, None)
        .await;
    Ok(match saved {
        Ok(saved) => Flash::notice(
            format!("/w/{id}/saved"),
            format!(
                "Saved '{}' with {} statement{}.",
                saved.name,
                saved.statements.len(),
                if saved.statements.len() == 1 { "" } else { "s" }
            ),
        ),
        Err(e) => Flash::error(format!("/w/{id}/chat?session={}", form.session), e.message),
    }
    .into_response())
}

pub(super) async fn run(
    State(app): State<App>,
    WebUser(identity): WebUser,
    Path((id, saved)): Path<(WorkspaceId, SavedId)>,
    flash: Flashed,
) -> WebResult<Response> {
    let access = Access::resolve(&app, identity, &id, Need::READ).await?;
    let ran = match access.run_saved(&app, &saved).await {
        Ok(ran) => ran,
        Err(e) => return Ok(Flash::error(format!("/w/{id}/saved"), e.message).into_response()),
    };
    let RunOf { question, run } = ran;
    SavedPage::render(
        &app,
        &access,
        Some(RunView::of(question.name, &run)),
        &flash,
    )
    .await
}

pub(super) async fn remove(
    State(app): State<App>,
    WebUser(identity): WebUser,
    Path((id, saved)): Path<(WorkspaceId, SavedId)>,
) -> WebResult<Response> {
    let access = Access::resolve(&app, identity, &id, Need::READ).await?;
    let removed = access.remove_saved(&app, &saved).await;
    Ok(Flash::after(format!("/w/{id}/saved"), removed, |()| {
        Some(String::from("Removed the saved question."))
    })
    .into_response())
}
