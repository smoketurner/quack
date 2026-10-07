//! Mail through `mail-parser`: one RFC 5322 message (`.eml`), or a
//! mailbox of them (`.mbox`). Each message is a section headed by its
//! subject, with its sender, recipients, and date above the body; a
//! mailbox's sections carry `message N` as their locator.

use std::io::BufReader;

use mail_parser::mailbox::mbox::MessageIterator;
use mail_parser::{Address, Message, MessageParser};

use super::html;
use super::parser::{DocumentMeta, Extracted, Flow, Section};
use crate::error::{Error, Result};

/// One message: its subject as the title and heading, its envelope as
/// the metadata.
///
/// # Errors
///
/// Returns an error when the bytes are not a message or it has no text.
pub fn eml(data: &[u8]) -> Result<Extracted> {
    let message = MessageParser::default()
        .parse(data)
        .ok_or_else(|| Error::Ingestion(String::from("not an email message")))?;
    let (section, meta) = Mail(&message).section();
    let section = section.ok_or_else(|| {
        Error::Ingestion(String::from(
            "no extractable text: the message has no text body",
        ))
    })?;
    Ok(Extracted {
        title: section.heading.clone(),
        sections: vec![section],
        flow: Flow::Sectioned,
        pages: None,
        meta,
        blank_pages: Vec::new(),
    })
}

/// A mailbox: one section per message, in file order.
///
/// # Errors
///
/// Returns an error when no message in the file has text.
pub fn mbox(data: &[u8]) -> Result<Extracted> {
    let mut sections = Vec::new();
    let mut number = 0u32;
    for entry in MessageIterator::new(BufReader::new(data)) {
        let raw = match entry {
            Ok(raw) => raw.unwrap_contents(),
            Err(e) => {
                tracing::warn!(error = %e, "skipping an unreadable mbox message");
                continue;
            }
        };
        number = number.saturating_add(1);
        let Some(message) = MessageParser::default().parse(&raw) else {
            continue;
        };
        if let (Some(section), _) = Mail(&message).section() {
            sections.push(section.at(format!("message {number}")));
        }
    }
    if sections.is_empty() {
        return Err(Error::Ingestion(String::from(
            "no extractable text: no message in the mailbox has a text body",
        )));
    }
    Ok(Extracted {
        sections,
        flow: Flow::Sectioned,
        ..Extracted::default()
    })
}

/// One parsed message.
struct Mail<'m, 'x>(&'m Message<'x>);

impl Mail<'_, '_> {
    /// A message as a section under its subject, with the envelope lines a
    /// reader expects first, and what the envelope says as metadata.
    fn section(&self) -> (Option<Section>, DocumentMeta) {
        let message = self.0;
        let mut meta = DocumentMeta::default();
        let from = message.from().map(Self::addresses);
        let to = message.to().map(Self::addresses);
        let date = message.date().map(mail_parser::DateTime::to_rfc3339);
        DocumentMeta::set(&mut meta.author, from.as_deref());
        DocumentMeta::set(&mut meta.authored_at, date.as_deref());
        meta.extra("to", to.as_deref());
        meta.extra("subject", message.subject());
        let mut text = String::new();
        for (label, value) in [("From", &from), ("To", &to), ("Date", &date)] {
            if let Some(v) = value {
                text.push_str(label);
                text.push_str(": ");
                text.push_str(v);
                text.push('\n');
            }
        }
        let body = self.body_text();
        if body.trim().is_empty() {
            return (None, meta);
        }
        text.push('\n');
        text.push_str(body.trim());
        let heading = message
            .subject()
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(str::to_owned);
        (Some(Section::body(heading, text)), meta)
    }

    /// Every text part, else every HTML part as its text.
    fn body_text(&self) -> String {
        let message = self.0;
        let mut parts: Vec<String> = (0..message.text_body_count())
            .filter_map(|i| message.body_text(i))
            .map(|t| t.trim().to_owned())
            .filter(|t| !t.is_empty())
            .collect();
        if parts.is_empty() {
            parts = (0..message.html_body_count())
                .filter_map(|i| message.body_html(i))
                .filter_map(|h| html::html(&h).ok())
                .map(|extracted| {
                    extracted
                        .sections
                        .iter()
                        .map(|s| s.text.as_str())
                        .collect::<Vec<_>>()
                        .join("\n")
                })
                .filter(|t| !t.trim().is_empty())
                .collect();
        }
        parts.join("\n\n")
    }

    /// `Name <address>, ...` for a header's addresses.
    fn addresses(address: &Address<'_>) -> String {
        address
            .clone()
            .into_list()
            .iter()
            .map(|a| match (a.name(), a.address()) {
                (Some(name), Some(addr)) => format!("{name} <{addr}>"),
                (Some(name), None) => name.to_owned(),
                (None, Some(addr)) => addr.to_owned(),
                (None, None) => String::new(),
            })
            .filter(|a| !a.is_empty())
            .collect::<Vec<_>>()
            .join(", ")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const MESSAGE: &str = "From: Ada Lovelace <ada@example.com>\r\nTo: Charles <charles@example.com>\r\nSubject: Renewal terms\r\nDate: Mon, 5 Jan 2026 14:30:00 +0100\r\nContent-Type: text/plain\r\n\r\nThirty days to renew.\r\n";

    #[test]
    fn an_eml_is_one_section_under_its_subject_with_its_envelope() {
        let extracted = eml(MESSAGE.as_bytes()).unwrap_or_default();
        assert_eq!(extracted.title.as_deref(), Some("Renewal terms"));
        assert_eq!(
            extracted.meta.author.as_deref(),
            Some("Ada Lovelace <ada@example.com>")
        );
        assert_eq!(
            extracted.meta.authored_at.as_deref(),
            Some("2026-01-05T14:30:00+01:00")
        );
        let section = extracted.sections.first().cloned().unwrap_or_default();
        assert_eq!(section.heading.as_deref(), Some("Renewal terms"));
        assert_eq!(
            section.text,
            "From: Ada Lovelace <ada@example.com>\nTo: Charles <charles@example.com>\nDate: 2026-01-05T14:30:00+01:00\n\nThirty days to renew."
        );
        assert!(eml(b"").is_err());
    }

    #[test]
    fn an_mbox_gives_one_section_per_message_with_its_number() {
        let second = MESSAGE
            .replace("Renewal terms", "Second")
            .replace("Thirty", "Sixty");
        let mbox_text = format!(
            "From ada@example.com Mon Jan  5 14:30:00 2026\n{MESSAGE}\nFrom ada@example.com Mon Jan  5 15:30:00 2026\n{second}\n"
        );
        let extracted = mbox(mbox_text.as_bytes()).unwrap_or_default();
        let summary: Vec<(Option<&str>, Option<&str>)> = extracted
            .sections
            .iter()
            .map(|s| (s.heading.as_deref(), s.locator.as_deref()))
            .collect();
        assert_eq!(
            summary,
            vec![
                (Some("Renewal terms"), Some("message 1")),
                (Some("Second"), Some("message 2")),
            ]
        );
        assert!(mbox(b"not a mailbox").is_err());
    }

    #[test]
    fn an_html_only_message_reads_as_its_text() {
        let message = "From: a@example.com\r\nSubject: Html\r\nContent-Type: text/html\r\n\r\n<html><body><p>Bold <b>text</b> here.</p></body></html>\r\n";
        let extracted = eml(message.as_bytes()).unwrap_or_default();
        assert!(
            extracted
                .sections
                .first()
                .is_some_and(|s| s.text.ends_with("Bold text here.")),
            "{extracted:?}"
        );
    }
}
