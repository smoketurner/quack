//! Where a web form lands once it is handled: back to a page, with a notice
//! or an error in the query string for that page's flash slot
//! (`FlashQuery` reads it back).

use axum::response::{IntoResponse, Redirect, Response};

use crate::server::error::ApiError;

/// A redirect after a form, carrying at most one message.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Flash {
    path: String,
    message: Option<(Level, String)>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Level {
    Notice,
    Error,
}

impl Level {
    fn key(self) -> &'static str {
        match self {
            Self::Notice => "notice",
            Self::Error => "error",
        }
    }
}

impl Flash {
    /// Back to `path` with nothing to say.
    pub(crate) fn to(path: impl Into<String>) -> Self {
        Self {
            path: path.into(),
            message: None,
        }
    }

    /// Back to `path` with a success message.
    pub(crate) fn notice(path: impl Into<String>, text: impl Into<String>) -> Self {
        Self {
            path: path.into(),
            message: Some((Level::Notice, text.into())),
        }
    }

    /// Back to `path` with what went wrong.
    pub(crate) fn error(path: impl Into<String>, text: impl Into<String>) -> Self {
        Self {
            path: path.into(),
            message: Some((Level::Error, text.into())),
        }
    }

    /// Back to `path`: with `success`'s message when the operation worked,
    /// else with its error's.
    pub(crate) fn after<T>(
        path: impl Into<String>,
        result: Result<T, ApiError>,
        success: impl FnOnce(T) -> Option<String>,
    ) -> Self {
        match result {
            Ok(value) => match success(value) {
                Some(text) => Self::notice(path, text),
                None => Self::to(path),
            },
            Err(e) => Self::error(path, e.message),
        }
    }

    fn url(&self) -> String {
        match &self.message {
            None => self.path.clone(),
            Some((level, text)) => {
                let joiner = if self.path.contains('?') { '&' } else { '?' };
                format!("{}{joiner}{}={}", self.path, level.key(), urlencoded(text))
            }
        }
    }
}

impl IntoResponse for Flash {
    fn into_response(self) -> Response {
        Redirect::to(&self.url()).into_response()
    }
}

/// `application/x-www-form-urlencoded` for one value: unreserved bytes as
/// they are, space as `+`, everything else percent-encoded.
fn urlencoded(text: &str) -> String {
    text.bytes()
        .map(|b| match b {
            b'a'..=b'z' | b'A'..=b'Z' | b'0'..=b'9' | b'-' | b'_' | b'.' => {
                char::from(b).to_string()
            }
            b' ' => String::from("+"),
            other => format!("%{other:02X}"),
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn urlencoded_escapes_reserved_bytes() {
        assert_eq!(urlencoded("a b&c=d/é"), "a+b%26c%3Dd%2F%C3%A9");
        assert_eq!(urlencoded("plain-text_1.2"), "plain-text_1.2");
    }

    #[test]
    fn a_flash_puts_one_message_in_the_query_string() {
        assert_eq!(Flash::to("/w/x/ontology").url(), "/w/x/ontology");
        assert_eq!(
            Flash::notice("/w/x/graph", "3 merged").url(),
            "/w/x/graph?notice=3+merged"
        );
        assert_eq!(
            Flash::error("/login", "wrong username or password").url(),
            "/login?error=wrong+username+or+password"
        );
        let failed: Result<(), ApiError> = Err(ApiError::bad_request("no rows"));
        assert_eq!(
            Flash::after("/w/x/tables", failed, |()| None).url(),
            "/w/x/tables?error=no+rows"
        );
        assert_eq!(
            Flash::after("/w/x/tables", Ok(2), |n| Some(format!("{n} loaded"))).url(),
            "/w/x/tables?notice=2+loaded"
        );
        assert_eq!(
            Flash::error("/w/x/ontology?status=low_support", "no").url(),
            "/w/x/ontology?status=low_support&error=no"
        );
    }
}
