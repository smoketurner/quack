//! Captions and transcripts (`.vtt`, `.srt`) through `subtp`: cues in
//! order, merged into sections that start at a cue's start time, which is
//! the section's locator, so a citation says `12:04`.

use subtp::srt::SubRip;
use subtp::vtt::{VttBlock, WebVtt};

use super::parser::{Extracted, Flow, Section};
use crate::error::{Error, Result};

/// A run of cues becomes one section once it holds this much text, or
/// when the next cue starts this long after the last one ended.
const SECTION_CHARS: usize = 600;
const GAP_SECONDS: u64 = 10;

/// One cue: when it starts and ends, and its lines.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Cue {
    start: CueTime,
    end: CueTime,
    text: String,
}

impl Cue {
    fn new(start: CueTime, end: CueTime, lines: &[String]) -> Self {
        Self {
            start,
            end,
            text: lines
                .iter()
                .map(|l| l.trim())
                .filter(|l| !l.is_empty())
                .collect::<Vec<_>>()
                .join(" "),
        }
    }
}

/// When a cue starts or ends, in milliseconds from the start.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct CueTime(u64);

impl CueTime {
    fn from_hms(hours: u8, minutes: u8, seconds: u8, milliseconds: u16) -> Self {
        Self(
            u64::from(hours)
                .saturating_mul(3_600_000)
                .saturating_add(u64::from(minutes).saturating_mul(60_000))
                .saturating_add(u64::from(seconds).saturating_mul(1_000))
                .saturating_add(u64::from(milliseconds)),
        )
    }

    /// How long after `earlier` this is; zero when it is not after.
    fn since(self, earlier: Self) -> u64 {
        self.0.saturating_sub(earlier.0)
    }
}

/// `12:04`, or `1:02:03` past an hour.
impl std::fmt::Display for CueTime {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let seconds = self.0.div_euclid(1_000);
        let (h, m, s) = (
            seconds.div_euclid(3_600),
            seconds.rem_euclid(3_600).div_euclid(60),
            seconds.rem_euclid(60),
        );
        if h > 0 {
            write!(f, "{h}:{m:02}:{s:02}")
        } else {
            write!(f, "{m}:{s:02}")
        }
    }
}

/// `WebVTT` captions.
///
/// # Errors
///
/// Returns an error when the text is not `WebVTT` or holds no cue text.
pub fn vtt(text: &str) -> Result<Extracted> {
    let parsed = WebVtt::parse(text).map_err(|e| Error::Ingestion(format!("not WebVTT: {e}")))?;
    let cues: Vec<Cue> = parsed
        .blocks
        .iter()
        .filter_map(|block| match block {
            VttBlock::Que(cue) => Some(Cue::new(
                CueTime::from_hms(
                    cue.timings.start.hours,
                    cue.timings.start.minutes,
                    cue.timings.start.seconds,
                    cue.timings.start.milliseconds,
                ),
                CueTime::from_hms(
                    cue.timings.end.hours,
                    cue.timings.end.minutes,
                    cue.timings.end.seconds,
                    cue.timings.end.milliseconds,
                ),
                &cue.payload,
            )),
            VttBlock::Comment(_) | VttBlock::Style(_) | VttBlock::Region(_) => None,
        })
        .collect();
    Cues(cues).into_extracted("WebVTT")
}

/// `SubRip` captions.
///
/// # Errors
///
/// Returns an error when the text is not `SubRip` or holds no cue text.
pub fn srt(text: &str) -> Result<Extracted> {
    let parsed = SubRip::parse(text).map_err(|e| Error::Ingestion(format!("not SubRip: {e}")))?;
    let cues: Vec<Cue> = parsed
        .subtitles
        .iter()
        .map(|cue| {
            Cue::new(
                CueTime::from_hms(
                    cue.start.hours,
                    cue.start.minutes,
                    cue.start.seconds,
                    cue.start.milliseconds,
                ),
                CueTime::from_hms(
                    cue.end.hours,
                    cue.end.minutes,
                    cue.end.seconds,
                    cue.end.milliseconds,
                ),
                &cue.text,
            )
        })
        .collect();
    Cues(cues).into_extracted("SubRip")
}

/// A captions file's cues, in order.
pub(crate) struct Cues(Vec<Cue>);

impl Cues {
    /// The cues merged into timed sections: a gap of more than
    /// `GAP_SECONDS` or `SECTION_CHARS` of text starts the next.
    ///
    /// # Errors
    ///
    /// Returns an error when no cue has text.
    pub(crate) fn into_extracted(self, format: &str) -> Result<Extracted> {
        let cues = &self.0;
        let mut out = Vec::new();
        let mut run: Vec<&Cue> = Vec::new();
        let flush = |run: &mut Vec<&Cue>, out: &mut Vec<Section>| {
            let Some(first) = run.first() else {
                return;
            };
            let text = run
                .iter()
                .map(|c| c.text.as_str())
                .filter(|t| !t.is_empty())
                .collect::<Vec<_>>()
                .join("\n");
            if !text.is_empty() {
                out.push(Section::body(None, text).at(first.start.to_string()));
            }
            run.clear();
        };
        for cue in cues {
            let gap = run
                .last()
                .is_some_and(|last| cue.start.since(last.end) > GAP_SECONDS * 1_000);
            let full = run.iter().map(|c| c.text.len()).sum::<usize>() >= SECTION_CHARS;
            if gap || full {
                flush(&mut run, &mut out);
            }
            run.push(cue);
        }
        flush(&mut run, &mut out);
        if out.is_empty() {
            return Err(Error::Ingestion(format!(
                "no extractable text: the {format} file has no cue text"
            )));
        }
        Ok(Extracted {
            sections: out,
            flow: Flow::Sectioned,
            ..Extracted::default()
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clock_drops_the_hour_until_there_is_one() {
        assert_eq!(CueTime(724_500).to_string(), "12:04");
        assert_eq!(CueTime(3_723_000).to_string(), "1:02:03");
        assert_eq!(CueTime(0).to_string(), "0:00");
    }

    #[test]
    fn vtt_cues_merge_into_timed_sections_split_at_long_gaps() {
        let text = "WEBVTT\n\nNOTE a comment\n\n00:00:01.000 --> 00:00:04.000\n- Never drink liquid nitrogen.\n\n00:00:05.000 --> 00:00:09.000\n- It will perforate your stomach.\n\n00:12:04.000 --> 00:12:06.000\nLater on.\n";
        let extracted = vtt(text).unwrap_or_default();
        let summary: Vec<(Option<&str>, &str)> = extracted
            .sections
            .iter()
            .map(|s| (s.locator.as_deref(), s.text.as_str()))
            .collect();
        assert_eq!(
            summary,
            vec![
                (
                    Some("0:01"),
                    "- Never drink liquid nitrogen.\n- It will perforate your stomach."
                ),
                (Some("12:04"), "Later on."),
            ]
        );
    }

    #[test]
    fn srt_cues_parse_and_an_empty_file_is_an_error() {
        let text = "1\n00:00:01,000 --> 00:00:02,000\nHello, world!\n\n2\n01:00:03,000 --> 01:00:04,000\nAn hour later.\n";
        let extracted = srt(text).unwrap_or_default();
        let locators: Vec<Option<&str>> = extracted
            .sections
            .iter()
            .map(|s| s.locator.as_deref())
            .collect();
        assert_eq!(locators, [Some("0:01"), Some("1:00:03")]);
        assert!(srt("garbage").is_err());
        assert!(vtt("WEBVTT\n").is_err());
    }

    #[test]
    fn a_long_run_of_cues_is_cut_by_size() {
        let cues: Vec<Cue> = (0..40u64)
            .map(|i| {
                Cue::new(
                    CueTime(i * 1_000),
                    CueTime(i * 1_000 + 900),
                    &[format!("cue number {i} with some words")],
                )
            })
            .collect();
        let extracted = Cues(cues).into_extracted("test").unwrap_or_default();
        assert!(extracted.sections.len() > 1, "{}", extracted.sections.len());
        assert!(
            extracted
                .sections
                .iter()
                .all(|s| s.text.len() < SECTION_CHARS + 60)
        );
    }
}
