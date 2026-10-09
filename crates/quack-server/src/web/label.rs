//! The Tables page's "Label rows" section and "Labelled by" block (issue
//! #472): a person says what they want to know about each row, the chat
//! model drafts the questions into an editor, a preview shows the labels
//! of the first rows, and Label rows starts the run with the questions as
//! edited. All of it goes through the same `Access` methods the API uses.

use std::collections::HashMap;

use askama::Template;
use axum::Form;
use axum::extract::{Path, State};
use axum::http::HeaderValue;
use axum::response::{IntoResponse, Response};
use quack_core::classify::{
    self, Draft, DraftOrigin, DraftQuestion, Effect, KeyReason, LabelSet, QuestionKind, Rows,
};
use quack_core::ids::WorkspaceId;
use quack_core::ingestion::TableName;
use quack_core::llm::decision::{DecisionModel, Questions};
use quack_core::storage::workspace::Cell;
use quack_core::text::Thousands;

use super::flash::Flash;
use super::{WebResult, WebUser};
use crate::auth::{Access, Need};
use crate::state::App;

/// Rows the Preview button labels.
const PREVIEW_ROWS: u32 = 10;
/// Question rows the editor reads back; more are ignored.
const MOST_ROWS: usize = 16;

/// The decision model's name when the form is offered: the workspace has a
/// decision model its provider list allows. A setting that cannot be used
/// hides the form and is logged, not raised on a page about something else.
pub(super) async fn offered(app: &App) -> Option<String> {
    match DecisionModel::offered(&app.config).await {
        Ok(model) => model.map(|model| model.label().to_owned()),
        Err(e) => {
            tracing::warn!(error = %e, "the Label rows form is not offered");
            None
        }
    }
}

/// The run a labels table was made by, as the page says it.
pub(super) struct LabelledView {
    pub sentence: Option<String>,
    pub source: String,
    pub key: String,
    pub model: String,
    pub status: String,
    pub counts: String,
    pub started_at: String,
}

impl LabelledView {
    pub(super) fn of(run: &classify::Run) -> Self {
        Self {
            sentence: run.sentence.clone(),
            source: run.source_table.clone(),
            key: run.key_column.clone(),
            model: run.model.clone(),
            status: run.status.to_string(),
            counts: format!(
                "{} labelled, {} cut to fit the model, {} empty, {} skipped",
                Thousands(run.labelled),
                Thousands(run.cut),
                Thousands(run.empty),
                Thousands(run.skipped)
            ),
            started_at: run.started_at.clone(),
        }
    }
}

/// One question as the editor shows it.
struct QuestionRow {
    name: String,
    /// `choice`, `noul`, or `score`: the value of the select.
    kind: &'static str,
    ask: String,
    lines: String,
}

impl QuestionRow {
    fn of(question: &DraftQuestion) -> Self {
        Self {
            name: question.name.clone(),
            kind: question.kind.as_str(),
            ask: question.instructions.clone(),
            lines: question.to_lines(),
        }
    }

    fn blank() -> Self {
        Self {
            name: String::new(),
            kind: QuestionKind::Choice.as_str(),
            ask: String::new(),
            lines: String::new(),
        }
    }
}

/// The "Label rows" section of a selected table.
#[derive(Template)]
#[template(path = "label_section.html")]
pub(super) struct LabelSection {
    ws_id: String,
    table: String,
    output: String,
    model: String,
    sentence: String,
    /// The editor, rendered.
    editor: Option<String>,
}

impl LabelSection {
    /// The section for `table`, with the editor filled from the questions
    /// last approved for it, when there are any.
    ///
    /// # Errors
    ///
    /// Returns the template's error.
    pub(super) fn of(
        ws_id: &WorkspaceId,
        table: &str,
        model: String,
        approved: Option<&Draft>,
    ) -> Result<Self, askama::Error> {
        let editor = approved
            .map(|draft| LabelEditor::of(ws_id, draft, true).render())
            .transpose()?;
        Ok(Self {
            ws_id: ws_id.to_string(),
            table: table.to_owned(),
            output: approved.map_or_else(
                || TableName::sanitized(&format!("{table}_labels")).to_string(),
                |draft| draft.output_table.clone(),
            ),
            model,
            sentence: approved
                .and_then(|draft| draft.set.sentence.clone())
                .unwrap_or_default(),
            editor,
        })
    }
}

/// The editor: the questions of a draft, one row each, and a blank row.
#[derive(Template)]
#[template(path = "label_editor.html")]
struct LabelEditor {
    ws_id: String,
    table: String,
    key: String,
    text: String,
    sentence: String,
    /// Where the questions came from.
    note: String,
    rows: Vec<QuestionRow>,
    /// The line under the rows: what is read, the key, the table's size.
    reads: String,
    /// Whether the table of labels exists, so every row can be labelled
    /// again.
    can_relabel: bool,
}

impl LabelEditor {
    fn of(ws_id: &WorkspaceId, draft: &Draft, approved: bool) -> Self {
        let mut rows: Vec<QuestionRow> = draft
            .set
            .questions
            .iter()
            .map(|(name, question)| QuestionRow::of(&DraftQuestion::of(name, question)))
            .collect();
        rows.push(QuestionRow::blank());
        let note = match draft.origin {
            DraftOrigin::Drafted => format!(
                "Questions drafted from {} sample rows. Edit them, then preview.",
                draft.sample_rows
            ),
            DraftOrigin::Revised => format!(
                "Questions revised from {} sample rows. Edit them, then preview.",
                draft.sample_rows
            ),
            DraftOrigin::Reused | DraftOrigin::Given => {
                String::from("The questions approved before. Edit them, then preview.")
            }
        };
        let gone = if draft.columns_gone.is_empty() {
            String::new()
        } else {
            format!(
                " The table no longer has {}; draft new questions.",
                draft.columns_gone.join(", ")
            )
        };
        let key_note = draft
            .set
            .key_reason
            .note()
            .map(|note| format!(" ({note})"))
            .unwrap_or_default();
        Self {
            ws_id: ws_id.to_string(),
            table: draft.table.clone(),
            key: draft.set.key_column.clone(),
            text: draft.set.text_columns.join(", "),
            sentence: draft.set.sentence.clone().unwrap_or_default(),
            note: format!("{note}{gone}"),
            reads: format!(
                "Reads {} · key {}{key_note} · {} rows · {} question{}. A row without a name is \
                 left out.",
                draft.set.columns_in_words(),
                draft.set.key_column,
                Thousands(draft.rows),
                draft.set.questions.count(),
                if draft.set.questions.count() == 1 {
                    ""
                } else {
                    "s"
                }
            ),
            rows,
            can_relabel: approved,
        }
    }
}

/// A preview's rows, as the page shows them.
#[derive(Template)]
#[template(path = "label_preview.html")]
struct LabelPreview {
    columns: Vec<String>,
    rows: Vec<Vec<String>>,
    summary: String,
    /// Why every row is labelled again, when the run would.
    relabel: Option<String>,
}

impl LabelPreview {
    fn of(report: &classify::Report, preview: &classify::Preview) -> Self {
        let outline = &report.outline;
        let labelling = match (outline.effect, outline.estimate()) {
            (Effect::AddsRows, Some(estimate)) => format!(
                " Labelling {} new rows takes {estimate}.",
                Thousands(outline.remaining)
            ),
            (_, Some(estimate)) => format!(
                " Labelling all {} rows takes {estimate}.",
                Thousands(outline.remaining)
            ),
            (_, None) => String::new(),
        };
        Self {
            columns: preview.compact.columns.clone(),
            rows: preview
                .compact
                .rows
                .iter()
                .map(|row| row.iter().map(|value| Cell(value).text()).collect())
                .collect(),
            summary: format!("{preview}{labelling}"),
            relabel: match outline.effect {
                Effect::ReplacesLabels { because } => Some(format!(
                    "Labels every row again: {because}. The current labels serve until then."
                )),
                Effect::NewTable | Effect::AddsRows => None,
            },
        }
    }
}

/// A message in place of a fragment, shown where the section keeps its
/// status.
#[derive(Template)]
#[template(path = "label_status.html")]
struct LabelStatus {
    message: String,
}

impl LabelStatus {
    /// The status area instead of the fragment's own target: the editor the
    /// person is working in stays, and this says why it did not change.
    fn refused(message: String) -> WebResult<Response> {
        let mut response = super::html(&Self { message })?;
        let headers = response.headers_mut();
        headers.insert("HX-Retarget", HeaderValue::from_static("#label-status"));
        headers.insert("HX-Reswap", HeaderValue::from_static("innerHTML"));
        Ok(response)
    }
}

/// The fields the section's forms post.
struct LabelForm(HashMap<String, String>);

impl LabelForm {
    /// One field, trimmed; empty when absent.
    fn field(&self, name: &str) -> &str {
        self.0.get(name).map_or("", |value| value.trim())
    }

    /// The questions the editor's rows hold, a row without a name left out.
    fn questions(&self) -> Result<Questions, String> {
        let mut pairs = Vec::new();
        for at in 0..MOST_ROWS {
            let name = self.field(&format!("n{at}_name"));
            if name.is_empty() {
                continue;
            }
            let spelled = self.field(&format!("n{at}_kind"));
            let Some(kind) = QuestionKind::parse(spelled) else {
                return Err(format!("'{spelled}' is not a kind of question"));
            };
            let question = DraftQuestion::from_lines(
                name,
                kind,
                self.field(&format!("n{at}_ask")),
                self.0
                    .get(&format!("n{at}_opts"))
                    .map_or("", String::as_str),
            );
            pairs.push(question.into_question().map_err(|e| e.to_string())?);
        }
        Questions::new(pairs).map_err(|e| e.to_string())
    }

    /// The request a posted editor makes: the questions as edited, the key
    /// and the text columns that rode along, and every row again when asked.
    fn request(&self) -> Result<classify::Request, String> {
        let questions = self.questions()?;
        let sentence = self.field("sentence");
        Ok(classify::Request {
            table: self.field("table").to_owned(),
            sentence: None,
            set: Some(LabelSet {
                key_column: self.field("key").to_owned(),
                key_reason: KeyReason::default(),
                text_columns: self
                    .field("text")
                    .split(',')
                    .map(str::trim)
                    .filter(|c| !c.is_empty())
                    .map(str::to_owned)
                    .collect(),
                questions,
                sentence: (!sentence.is_empty()).then(|| sentence.to_owned()),
            }),
            rows: if self.0.contains_key("all") {
                Rows::All
            } else {
                Rows::Missing
            },
        })
    }
}

/// `POST /w/{id}/tables/label/draft`: the chat model's questions for a
/// sentence, or the approved ones revised, as the editor.
pub(super) async fn draft(
    State(app): State<App>,
    WebUser(identity): WebUser,
    Path(id): Path<WorkspaceId>,
    Form(form): Form<HashMap<String, String>>,
) -> WebResult<Response> {
    let access = Access::resolve(&app, identity, &id, Need::READ).await?;
    let form = LabelForm(form);
    let sentence = form.field("sentence");
    let drafted = access
        .draft_questions(
            &app,
            form.field("table"),
            (!sentence.is_empty()).then_some(sentence),
        )
        .await;
    match drafted {
        Ok(draft) => super::html(&LabelEditor::of(&id, &draft, draft.approved_at.is_some())),
        Err(e) => LabelStatus::refused(e.message),
    }
}

/// `POST /w/{id}/tables/label/preview`: the labels of the first rows under
/// the questions as edited.
pub(super) async fn preview(
    State(app): State<App>,
    WebUser(identity): WebUser,
    Path(id): Path<WorkspaceId>,
    Form(form): Form<HashMap<String, String>>,
) -> WebResult<Response> {
    let access = Access::resolve(&app, identity, &id, Need::READ).await?;
    let form = LabelForm(form);
    let request = match form.request() {
        Ok(request) => request,
        Err(message) => return LabelStatus::refused(message),
    };
    match access
        .preview_classification(&app, &request, PREVIEW_ROWS)
        .await
    {
        Ok(report) => match &report.preview {
            Some(preview) => super::html(&LabelPreview::of(&report, preview)),
            None => LabelStatus::refused(String::from("the preview came back empty")),
        },
        Err(e) => LabelStatus::refused(e.message),
    }
}

/// `POST /w/{id}/tables/label/run`: start the run with the questions as
/// edited; they become the table's approved questions.
pub(super) async fn run(
    State(app): State<App>,
    WebUser(identity): WebUser,
    Path(id): Path<WorkspaceId>,
    Form(form): Form<HashMap<String, String>>,
) -> WebResult<Response> {
    let access = Access::resolve(&app, identity, &id, Need::WRITE).await?;
    let form = LabelForm(form);
    let back = format!("/w/{id}/tables");
    let table = form.field("table").to_owned();
    let request = match form.request() {
        Ok(request) => request,
        Err(message) => {
            return Ok(Flash::error(back, message).opening(table).into_response());
        }
    };
    Ok(match access.classify(&app, request).await {
        Ok(started) => Flash::notice(
            back,
            format!(
                "Labelling {} rows of {} as job {}; the job strip shows its progress.",
                Thousands(started.outline.remaining),
                started.outline.source_table,
                started.job
            ),
        ),
        Err(e) => Flash::error(back, e.message),
    }
    .opening(table)
    .into_response())
}
