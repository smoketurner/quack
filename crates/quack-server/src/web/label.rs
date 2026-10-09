//! The Tables page's "Label rows" form and "Labelled by" block (issue
//! #472): a decision model labels a table's text into a new table, from
//! the same `Access` methods the API uses.

use axum::Form;
use axum::extract::{Path, State};
use axum::response::{IntoResponse, Response};
use quack_core::classify::{
    Classification, ClassificationPreview, ClassificationRun, QuestionSet, Rows,
};
use quack_core::ids::WorkspaceId;
use quack_core::llm::decision::DecisionModel;
use quack_core::storage::profile::TableProfile;
use quack_core::storage::workspace::Cell;
use serde::Deserialize;

use super::flash::Flash;
use super::{TableView, TablesPage, WebResult, WebUser};
use crate::auth::{Access, Need};
use crate::state::App;

/// What the form offers when no run has used the table yet: the questions
/// of the issue's support-ticket example.
const EXAMPLE_SET: &str = r#"{
  "name": "ticket_triage",
  "questions": {
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
    "churn_risk": {
      "type": "noul",
      "instructions": "Does the customer threaten to cancel?"
    }
  }
}"#;

/// Rows the form previews unless it says otherwise.
const PREVIEW_ROWS: u32 = 20;
/// The most rows a preview labels.
const MOST_PREVIEW_ROWS: u32 = 100;

/// Whether the form is offered: the workspace has a decision model its
/// provider list allows. A setting that cannot be used hides the form and
/// is logged, not raised on a page about something else.
pub(super) async fn offered(app: &App) -> bool {
    match DecisionModel::offered(&app.config).await {
        Ok(model) => model.is_some(),
        Err(e) => {
            tracing::warn!(error = %e, "the Label rows form is not offered");
            false
        }
    }
}

/// The run a labels table was made by, as the page says it.
pub(super) struct LabelledView {
    pub set: String,
    pub source: String,
    pub key: String,
    pub model: String,
    pub status: String,
    pub counts: String,
    pub started_at: String,
}

impl LabelledView {
    pub(super) fn of(run: &ClassificationRun) -> Self {
        Self {
            set: run.question_set.name.to_string(),
            source: run.source_table.clone(),
            key: run.key_column.clone(),
            model: run.model.clone(),
            status: run.status.to_string(),
            counts: format!(
                "{} labelled, {} cut to fit the model, {} empty, {} skipped",
                run.labelled, run.cut, run.empty, run.skipped
            ),
            started_at: run.started_at.clone(),
        }
    }
}

/// The fields of the "Label rows" form.
#[derive(Clone)]
pub(super) struct LabelForm {
    pub text: String,
    pub key: String,
    pub questions: String,
    pub preview_rows: String,
    pub all: bool,
}

impl LabelForm {
    /// What the form holds for `table`: its text columns, its id column,
    /// and the questions the newest run on it used, or the example.
    pub(super) fn for_table(
        table: &TableView,
        profile: Option<&TableProfile>,
        runs: &[ClassificationRun],
    ) -> Self {
        let text = table
            .columns
            .iter()
            .filter(|c| c.kind == "VARCHAR")
            .map(|c| c.name.as_str())
            .collect::<Vec<_>>()
            .join(", ");
        let key = profile
            .and_then(TableProfile::key_column)
            .filter(|name| TableProfile::is_id_name(name))
            .unwrap_or_default()
            .to_owned();
        let questions = runs
            .iter()
            .find(|run| run.source_table == table.name)
            .and_then(|run| serde_json::to_string_pretty(&run.question_set).ok())
            .unwrap_or_else(|| String::from(EXAMPLE_SET));
        Self {
            text,
            key,
            questions,
            preview_rows: PREVIEW_ROWS.to_string(),
            all: false,
        }
    }

    /// The rows to preview: 1 to 100, or the default when the field is
    /// empty.
    fn preview_count(&self) -> Result<u32, String> {
        let text = self.preview_rows.trim();
        if text.is_empty() {
            return Ok(PREVIEW_ROWS);
        }
        text.parse::<u32>()
            .ok()
            .filter(|rows| (1..=MOST_PREVIEW_ROWS).contains(rows))
            .ok_or_else(|| {
                format!(
                    "rows to preview is a whole number from 1 to {MOST_PREVIEW_ROWS}, not '{text}'"
                )
            })
    }

    /// The request the form makes of `table`.
    fn classification(&self, table: &str) -> Result<Classification, String> {
        let question_set = QuestionSet::parse(&self.questions).map_err(|e| e.to_string())?;
        let key = self.key.trim();
        Ok(Classification {
            table: table.to_owned(),
            text_columns: self
                .text
                .split(',')
                .map(str::trim)
                .filter(|c| !c.is_empty())
                .map(str::to_owned)
                .collect(),
            key: (!key.is_empty()).then(|| key.to_owned()),
            question_set,
            rows: if self.all { Rows::All } else { Rows::Missing },
        })
    }
}

/// A preview's rows, as the page shows them.
pub(super) struct PreviewView {
    pub key: String,
    pub columns: Vec<String>,
    pub rows: Vec<Vec<String>>,
    pub summary: String,
}

impl PreviewView {
    fn of(preview: &ClassificationPreview) -> Self {
        Self {
            key: preview.key_column.clone(),
            columns: preview.result.columns.clone(),
            rows: preview
                .result
                .rows
                .iter()
                .map(|row| row.iter().map(|value| Cell(value).text()).collect())
                .collect(),
            summary: preview.to_string(),
        }
    }
}

/// The form as the page shows it again after a preview: the values typed
/// and the preview below them.
pub(super) struct Labelling {
    pub form: LabelForm,
    pub preview: Option<PreviewView>,
}

/// The submitted "Label rows" form: the table, its fields, and which
/// button was pressed.
#[derive(Deserialize)]
pub(super) struct LabelSubmission {
    name: String,
    #[serde(default)]
    text: String,
    #[serde(default)]
    key: String,
    #[serde(default)]
    questions: String,
    #[serde(default)]
    preview_rows: String,
    #[serde(default)]
    all: Option<String>,
    action: String,
}

impl LabelSubmission {
    fn form(&self) -> LabelForm {
        LabelForm {
            text: self.text.clone(),
            key: self.key.clone(),
            questions: self.questions.clone(),
            preview_rows: self.preview_rows.clone(),
            all: self.all.is_some(),
        }
    }
}

/// `POST /w/{id}/tables/classify`: preview the labels on the page, or start
/// the run as a job.
pub(super) async fn table_classify(
    State(app): State<App>,
    WebUser(identity): WebUser,
    Path(id): Path<WorkspaceId>,
    Form(submitted): Form<LabelSubmission>,
) -> WebResult<Response> {
    let previewing = submitted.action == "preview";
    let need = if previewing { Need::READ } else { Need::WRITE };
    let access = Access::resolve(&app, identity, &id, need).await?;
    let back = format!("/w/{id}/tables");
    let form = submitted.form();
    let request = match form.classification(&submitted.name) {
        Ok(request) => request,
        Err(message) => {
            return Ok(Flash::error(back, message)
                .opening(submitted.name)
                .into_response());
        }
    };
    if previewing {
        let outcome = match form.preview_count() {
            Ok(rows) => access
                .preview_classification(&app, &request, rows)
                .await
                .map_err(|e| e.message),
            Err(message) => Err(message),
        };
        let (error, preview) = match outcome {
            Ok(preview) => (None, Some(PreviewView::of(&preview))),
            Err(message) => (Some(message), None),
        };
        return TablesPage::render(
            &app,
            access.identity.clone(),
            &id,
            Some(submitted.name),
            (error, None),
            Some(Labelling { form, preview }),
        )
        .await;
    }
    let table = submitted.name;
    Ok(match access.classify(&app, request).await {
        Ok(started) => Flash::notice(
            back,
            format!(
                "Labelling {} rows of {} as job {}; the job strip shows its progress.",
                started.outline.remaining, started.outline.source_table, started.job
            ),
        ),
        Err(e) => Flash::error(back, e.message),
    }
    .opening(table)
    .into_response())
}
