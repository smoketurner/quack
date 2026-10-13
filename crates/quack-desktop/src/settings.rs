//! The Settings view: every recognized setting with its value and where it
//! came from (`quack config`'s inspection), the paths the app uses, and the
//! models in force.

use quack_core::config::inspect::{FileState, Inspection, SettingFilter};
use serde::Serialize;
use tauri::State;

use crate::app::AppState;
use crate::log::Logging;

/// What the Settings view shows.
#[derive(Debug, Clone, Serialize, utoipa::ToSchema)]
pub(crate) struct SettingsView {
    pub(crate) version: &'static str,
    pub(crate) config_path: String,
    /// `missing`, `loaded`, or `rejected`.
    pub(crate) file_state: &'static str,
    /// Why the file was refused, when it was.
    pub(crate) file_error: Option<String>,
    pub(crate) data_dir: String,
    pub(crate) log_dir: String,
    pub(crate) chat_model: String,
    pub(crate) settings: Vec<SettingView>,
    /// Keys in the file that no setting corresponds to.
    pub(crate) unknown_keys: Vec<String>,
    pub(crate) environment: Vec<EnvView>,
}

/// One setting: `section.key`, its value in force, and its origin.
#[derive(Debug, Clone, Serialize, utoipa::ToSchema)]
pub(crate) struct SettingView {
    pub(crate) path: String,
    pub(crate) value: Option<String>,
    pub(crate) default: Option<String>,
    /// `default`, `file`, `env VAR`, or `registration`.
    pub(crate) origin: String,
    pub(crate) is_default: bool,
}

/// An environment variable the app reads, and whether it is set; never
/// its value.
#[derive(Debug, Clone, Serialize, utoipa::ToSchema)]
pub(crate) struct EnvView {
    pub(crate) name: String,
    pub(crate) set: bool,
    pub(crate) purpose: String,
}

/// The configuration as it stands on disk and in the environment.
#[tauri::command]
#[expect(
    clippy::needless_pass_by_value,
    reason = "a command's arguments arrive by value from the frontend"
)]
pub(crate) fn settings(state: State<'_, AppState>, logging: State<'_, Logging>) -> SettingsView {
    let inspection = Inspection::load();
    let (file_state, file_error) = match &inspection.file_state {
        FileState::Missing => ("missing", None),
        FileState::Loaded => ("loaded", None),
        FileState::Rejected(why) => ("rejected", Some(why.clone())),
    };
    SettingsView {
        version: env!("CARGO_PKG_VERSION"),
        config_path: inspection.config_path.display().to_string(),
        file_state,
        file_error,
        data_dir: state.config.data_dir().display().to_string(),
        log_dir: logging.dir.display().to_string(),
        chat_model: state.config.chat_model_label(),
        settings: inspection
            .shown(SettingFilter::All)
            .into_iter()
            .map(|s| SettingView {
                path: s.path(),
                value: s.value.clone(),
                default: s.default.clone(),
                origin: s.origin.to_string(),
                is_default: s.is_default(),
            })
            .collect(),
        unknown_keys: inspection.unknown.iter().map(|k| k.path.clone()).collect(),
        environment: inspection
            .environment
            .iter()
            .map(|v| EnvView {
                name: v.name.clone(),
                set: v.set,
                purpose: v.purpose.clone(),
            })
            .collect(),
    }
}
