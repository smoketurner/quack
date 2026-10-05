//! Copying text out of the session: the system clipboard, and the
//! terminal's own through OSC 52, which also reaches the person's machine
//! over SSH.

use std::fmt;
use std::io::Write;

use anyhow::Result;
use crossterm::clipboard::CopyToClipboard;
use tracing::debug;

/// How much text was copied.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Extent {
    lines: usize,
    characters: usize,
}

impl Extent {
    fn of(text: &str) -> Self {
        Self {
            lines: text.lines().count(),
            characters: text.chars().count(),
        }
    }
}

impl fmt::Display for Extent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match (self.lines, self.characters) {
            (0 | 1, 1) => f.write_str("1 character"),
            (0 | 1, characters) => write!(f, "{characters} characters"),
            (lines, _) => write!(f, "{lines} lines"),
        }
    }
}

/// How a copy went, as the status line reports it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum CopyStatus {
    /// On the system clipboard.
    Copied(Extent),
    /// Handed to the terminal alone, which may or may not honor it.
    Sent(Extent),
    Failed(String),
}

impl fmt::Display for CopyStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Copied(extent) => write!(f, "copied {extent}"),
            Self::Sent(extent) => write!(f, "sent {extent} to the terminal's clipboard"),
            Self::Failed(reason) => write!(f, "copy failed: {reason}"),
        }
    }
}

enum System {
    Unopened,
    // Kept open for the session: on X11 the text is served by whoever
    // holds the clipboard.
    Open(arboard::Clipboard),
    #[cfg(test)]
    Off,
}

impl System {
    fn set(&mut self, text: &str) -> Result<()> {
        match self {
            Self::Open(clipboard) => Ok(clipboard.set_text(text)?),
            Self::Unopened => {
                let mut clipboard = arboard::Clipboard::new()?;
                let set = clipboard.set_text(text);
                *self = Self::Open(clipboard);
                Ok(set?)
            }
            #[cfg(test)]
            Self::Off => Err(anyhow::anyhow!("no system clipboard")),
        }
    }
}

/// Where copied text goes: the system clipboard and `terminal`, the
/// stream OSC 52 is written to.
pub(crate) struct Clipboard<W> {
    system: System,
    terminal: W,
}

impl<W: Write> Clipboard<W> {
    pub(crate) fn new(terminal: W) -> Self {
        Self {
            system: System::Unopened,
            terminal,
        }
    }

    /// Through the terminal alone, leaving the system clipboard as it is.
    #[cfg(test)]
    pub(crate) fn terminal_only(terminal: W) -> Self {
        Self {
            system: System::Off,
            terminal,
        }
    }

    pub(crate) fn copy(&mut self, text: &str) -> CopyStatus {
        let extent = Extent::of(text);
        let sent = crossterm::execute!(self.terminal, CopyToClipboard::to_clipboard_from(text));
        match (self.system.set(text), sent) {
            (Ok(()), _) => CopyStatus::Copied(extent),
            (Err(e), Ok(())) => {
                debug!(error = %e, "system clipboard unavailable");
                CopyStatus::Sent(extent)
            }
            (Err(e), Err(_)) => CopyStatus::Failed(e.to_string()),
        }
    }
}

#[cfg(test)]
mod tests {
    use base64::Engine as _;
    use base64::engine::general_purpose::STANDARD;

    use super::*;

    /// A stream that refuses every write.
    struct Closed;

    impl Write for Closed {
        fn write(&mut self, _: &[u8]) -> std::io::Result<usize> {
            Err(std::io::ErrorKind::BrokenPipe.into())
        }

        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    #[test]
    fn a_copy_reaches_the_terminal_as_osc_52() {
        let mut out = Vec::new();
        let status = Clipboard::terminal_only(&mut out).copy("one\ntwo");
        assert_eq!(
            status.to_string(),
            "sent 2 lines to the terminal's clipboard"
        );
        let sequence = String::from_utf8_lossy(&out);
        let payload = sequence
            .strip_prefix("\u{1b}]52;c;")
            .and_then(|rest| rest.strip_suffix("\u{1b}\\"));
        let decoded = payload.and_then(|payload| STANDARD.decode(payload).ok());
        assert_eq!(decoded.as_deref(), Some(b"one\ntwo".as_slice()));
    }

    #[test]
    fn a_copy_nothing_took_says_why() {
        let status = Clipboard::terminal_only(Closed).copy("x");
        assert_eq!(status.to_string(), "copy failed: no system clipboard");
    }

    #[test]
    fn the_extent_counts_lines_or_characters() {
        let extent = |text| Extent::of(text).to_string();
        assert_eq!(extent("a"), "1 character");
        assert_eq!(extent("\u{65E5}\u{672C}"), "2 characters");
        assert_eq!(extent("a\nb\nc"), "3 lines");
    }
}
