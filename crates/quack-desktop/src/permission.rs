//! A write a turn waits on, answered in a native dialog built from
//! `Decision`'s words (design 3.5): the heading, the hold's notice, the
//! statement, and the three answers in their order. Closing the dialog
//! any other way is a refusal.

use std::fmt::Write;

use quack_core::analysis::events::Decision;
use tauri::{AppHandle, Manager};
use tauri_plugin_dialog::{
    DialogExt, MessageDialogButtons, MessageDialogKind, MessageDialogResult,
};
use tokio::sync::oneshot;

/// Characters of the statement the dialog shows; the chat shows it all.
const SQL_SHOWN: usize = 1_500;

/// Show the dialog for a statement and wait for the answer.
pub(crate) async fn show(app: &AppHandle, sql: &str, notice: Option<&str>) -> Decision {
    let (tx, rx) = oneshot::channel();
    let mut dialog = app
        .dialog()
        .message(body(sql, notice))
        .title(Decision::HEADING)
        .kind(MessageDialogKind::Warning)
        .buttons(MessageDialogButtons::YesNoCancelCustom(
            Decision::Allow.label().to_owned(),
            Decision::AllowTurn.label().to_owned(),
            Decision::Deny.label().to_owned(),
        ));
    if let Some(window) = app.get_webview_window("main") {
        // Shown and focused first if it was hidden, so the question is seen.
        if let Err(e) = window.show().and_then(|()| window.set_focus()) {
            tracing::debug!(error = %e, "could not bring the window forward for a write prompt");
        }
        dialog = dialog.parent(&window);
    }
    dialog.show_with_result(move |result| {
        if tx.send(result).is_err() {
            tracing::debug!("a write prompt was answered after its turn stopped listening");
        }
    });
    match rx.await {
        Ok(result) => decision_of(&result),
        Err(_) => Decision::Deny,
    }
}

/// The `Decision` a dialog result stands for. The three custom buttons
/// come back as `Yes`, `No`, and `Cancel` on some platforms and as their
/// labels on others; anything else is a refusal.
pub(crate) fn decision_of(result: &MessageDialogResult) -> Decision {
    match result {
        MessageDialogResult::Yes => Decision::Allow,
        MessageDialogResult::No => Decision::AllowTurn,
        MessageDialogResult::Ok | MessageDialogResult::Cancel => Decision::Deny,
        MessageDialogResult::Custom(label) => Decision::CHOICES
            .into_iter()
            .find(|choice| choice.label() == label)
            .unwrap_or(Decision::Deny),
    }
}

/// The dialog's body: the hold's notice when there is one, then the
/// statement, cut when long.
pub(crate) fn body(sql: &str, notice: Option<&str>) -> String {
    let mut text = String::new();
    if let Some(notice) = notice {
        text.push_str(notice);
        text.push_str("\n\n");
    }
    let shown: String = sql.chars().take(SQL_SHOWN).collect();
    text.push_str(&shown);
    if shown.len() < sql.len() {
        let more = sql.lines().count().saturating_sub(shown.lines().count());
        if write!(text, "\n… and {more} more lines, shown in the chat").is_err() {
            // Writing to a String does not fail.
        }
    }
    text
}

#[cfg(test)]
mod tests {
    use quack_core::analysis::events::Decision;
    use quack_core::analysis::policy::Hold;
    use tauri_plugin_dialog::MessageDialogResult;

    use super::{SQL_SHOWN, body, decision_of};

    /// The buttons are `Decision`'s words in `CHOICES` order, and each
    /// result form maps back to its decision; anything else refuses.
    #[test]
    fn every_result_form_is_a_decision() {
        assert_eq!(decision_of(&MessageDialogResult::Yes), Decision::Allow);
        assert_eq!(decision_of(&MessageDialogResult::No), Decision::AllowTurn);
        assert_eq!(decision_of(&MessageDialogResult::Cancel), Decision::Deny);
        assert_eq!(decision_of(&MessageDialogResult::Ok), Decision::Deny);
        for choice in Decision::CHOICES {
            let result = MessageDialogResult::Custom(choice.label().to_owned());
            assert_eq!(decision_of(&result), choice);
        }
        let other = MessageDialogResult::Custom(String::from("Maybe"));
        assert_eq!(decision_of(&other), Decision::Deny);
    }

    /// The notice comes first when the hold has one; a long statement is
    /// cut and says how much the chat still shows.
    #[test]
    fn the_body_carries_the_notice_and_cuts_a_long_statement() {
        let short = body("DELETE FROM t", None);
        assert_eq!(short, "DELETE FROM t");
        let read = body("DELETE FROM t", Hold::ReadDocuments.notice());
        assert!(read.starts_with("This turn read document text"));
        assert!(read.ends_with("\n\nDELETE FROM t"));
        let long = "UPDATE t SET n = 1;\n".repeat(200);
        let cut = body(&long, None);
        assert!(cut.chars().count() < long.chars().count());
        assert!(cut.contains("more lines, shown in the chat"), "{cut}");
        let shown: String = long.chars().take(SQL_SHOWN).collect();
        assert!(cut.starts_with(&shown));
    }
}
