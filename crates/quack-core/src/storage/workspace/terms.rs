//! How text becomes the terms of the keyword index (`_quack_terms`): each
//! document is stemmed under the language it was detected as at ingest,
//! runs of scripts written without spaces (Chinese, Japanese, Korean) become
//! character bigrams, and a query is stemmed under every language the
//! workspace's documents were (issue #395).

use std::borrow::Cow;
use std::collections::BTreeMap;
use std::fmt;
use std::str::FromStr;

use rust_stemmers::{Algorithm, Stemmer};
use whatlang::{Detector, Lang, Script};

use crate::error::{Error, Result};

/// A Snowball stemmer the keyword index can reduce a document's words with.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Language(Algorithm);

impl Language {
    pub const ENGLISH: Self = Self(Algorithm::English);

    /// Every language `rust-stemmers` compiles in.
    pub const ALL: [Self; 18] = [
        Self(Algorithm::Arabic),
        Self(Algorithm::Danish),
        Self(Algorithm::Dutch),
        Self(Algorithm::English),
        Self(Algorithm::Finnish),
        Self(Algorithm::French),
        Self(Algorithm::German),
        Self(Algorithm::Greek),
        Self(Algorithm::Hungarian),
        Self(Algorithm::Italian),
        Self(Algorithm::Norwegian),
        Self(Algorithm::Portuguese),
        Self(Algorithm::Romanian),
        Self(Algorithm::Russian),
        Self(Algorithm::Spanish),
        Self(Algorithm::Swedish),
        Self(Algorithm::Tamil),
        Self(Algorithm::Turkish),
    ];

    /// The name configuration and listings use.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self.0 {
            Algorithm::Arabic => "arabic",
            Algorithm::Danish => "danish",
            Algorithm::Dutch => "dutch",
            Algorithm::English => "english",
            Algorithm::Finnish => "finnish",
            Algorithm::French => "french",
            Algorithm::German => "german",
            Algorithm::Greek => "greek",
            Algorithm::Hungarian => "hungarian",
            Algorithm::Italian => "italian",
            Algorithm::Norwegian => "norwegian",
            Algorithm::Portuguese => "portuguese",
            Algorithm::Romanian => "romanian",
            Algorithm::Russian => "russian",
            Algorithm::Spanish => "spanish",
            Algorithm::Swedish => "swedish",
            Algorithm::Tamil => "tamil",
            Algorithm::Turkish => "turkish",
        }
    }

    /// The language the detector names the same way.
    const fn lang(self) -> Lang {
        match self.0 {
            Algorithm::Arabic => Lang::Ara,
            Algorithm::Danish => Lang::Dan,
            Algorithm::Dutch => Lang::Nld,
            Algorithm::English => Lang::Eng,
            Algorithm::Finnish => Lang::Fin,
            Algorithm::French => Lang::Fra,
            Algorithm::German => Lang::Deu,
            Algorithm::Greek => Lang::Ell,
            Algorithm::Hungarian => Lang::Hun,
            Algorithm::Italian => Lang::Ita,
            Algorithm::Norwegian => Lang::Nob,
            Algorithm::Portuguese => Lang::Por,
            Algorithm::Romanian => Lang::Ron,
            Algorithm::Russian => Lang::Rus,
            Algorithm::Spanish => Lang::Spa,
            Algorithm::Swedish => Lang::Swe,
            Algorithm::Tamil => Lang::Tam,
            Algorithm::Turkish => Lang::Tur,
        }
    }

    fn of_lang(lang: Lang) -> Option<Self> {
        Self::ALL.into_iter().find(|l| l.lang() == lang)
    }

    fn stem(self, word: &str) -> Cow<'_, str> {
        Stemmer::create(self.0).stem(word)
    }
}

impl fmt::Display for Language {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

impl FromStr for Language {
    type Err = Error;

    fn from_str(s: &str) -> Result<Self> {
        let want = s.trim().to_ascii_lowercase();
        Self::ALL
            .into_iter()
            .find(|l| l.name() == want)
            .ok_or_else(|| {
                let names: Vec<&str> = Self::ALL.iter().map(|l| l.name()).collect();
                Error::Config(format!(
                    "unknown language '{s}'; use \"auto\" or one of {}",
                    names.join(", ")
                ))
            })
    }
}

/// How one document's words become terms.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stemming {
    Snowball(Language),
    /// A language with no Snowball stemmer: words are only lowercased.
    Unstemmed,
}

impl Stemming {
    const UNSTEMMED: &'static str = "unstemmed";

    /// The stemming a stored ISO 639-3 code stands for; a document with no
    /// code (written before detection, or holding no text) is English, as
    /// every document was before detection.
    #[must_use]
    pub fn of_code(code: Option<&str>) -> Self {
        match code {
            None => Self::Snowball(Language::ENGLISH),
            Some(code) => Lang::from_code(code)
                .and_then(Language::of_lang)
                .map_or(Self::Unstemmed, Self::Snowball),
        }
    }

    /// The name `_quack_meta` records the workspace's set by.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Snowball(language) => language.name(),
            Self::Unstemmed => Self::UNSTEMMED,
        }
    }

    fn parse(name: &str) -> Option<Self> {
        if name == Self::UNSTEMMED {
            return Some(Self::Unstemmed);
        }
        name.parse().ok().map(Self::Snowball)
    }

    fn reduce(self, word: &str) -> String {
        match self {
            Self::Snowball(language) => language.stem(word).into_owned(),
            Self::Unstemmed => word.to_owned(),
        }
    }
}

/// The stemmings a text is reduced under: one for a document, the
/// workspace's whole set for a query, so a short query needs no detection
/// and matches a document in any of the workspace's languages.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Analyzer(Vec<Stemming>);

impl Default for Analyzer {
    /// English, the stemming of every document before detection.
    fn default() -> Self {
        Self(vec![Stemming::Snowball(Language::ENGLISH)])
    }
}

impl Analyzer {
    /// One document's stemming.
    #[must_use]
    pub fn of(stemming: Stemming) -> Self {
        Self(vec![stemming])
    }

    /// The set `_quack_meta` recorded (comma-separated names); English
    /// when it recorded none.
    #[must_use]
    pub fn of_recorded(recorded: Option<&str>) -> Self {
        let mut set = Vec::new();
        for stemming in recorded
            .unwrap_or_default()
            .split(',')
            .filter_map(|name| Stemming::parse(name.trim()))
        {
            if !set.contains(&stemming) {
                set.push(stemming);
            }
        }
        if set.is_empty() {
            Self::default()
        } else {
            Self(set)
        }
    }

    /// The set `codes` (stored ISO 639-3 codes) stand for, as
    /// `_quack_meta` records it.
    #[must_use]
    pub fn record<'a>(codes: impl IntoIterator<Item = Option<&'a str>>) -> String {
        let mut names: Vec<&'static str> = Vec::new();
        for code in codes {
            let name = Stemming::of_code(code).name();
            if !names.contains(&name) {
                names.push(name);
            }
        }
        names.sort_unstable();
        names.join(",")
    }

    /// The stemmings, for a listing.
    #[must_use]
    pub fn stemmings(&self) -> &[Stemming] {
        &self.0
    }

    /// The terms of `text`: lowercased alphanumeric runs reduced under each
    /// stemming, and every run of an unspaced script as character bigrams
    /// (one character alone as itself). A run joined by identifier
    /// punctuation with no whitespace (`POL-8841`, `v1.2.3`, `ns/part:7`)
    /// also yields its punctuation-stripped, lowercased, unstemmed form
    /// (`pol8841`) beside its pieces, so the query `POL-8841` ranks a
    /// chunk holding that identifier above one holding `pol` and `8841`
    /// apart (issue #77). The same rule indexes chunks and parses queries.
    #[must_use]
    pub fn terms(&self, text: &str) -> Vec<String> {
        let mut terms = Vec::new();
        for run in text.split(|c: char| !(c.is_alphanumeric() || is_identifier_joiner(c))) {
            let run = run.trim_matches(|c: char| !c.is_alphanumeric());
            if run.is_empty() {
                continue;
            }
            for token in run.split(|c: char| !c.is_alphanumeric()) {
                for segment in Segment::split(token) {
                    match segment {
                        Segment::Spaced(word) => self.push_word(&word.to_lowercase(), &mut terms),
                        Segment::Unspaced(chars) => Segment::push_bigrams(&chars, &mut terms),
                    }
                }
            }
            if run.contains(is_identifier_joiner) && !run.chars().any(Unspaced::holds) {
                let alnum: String = run.chars().filter(|c| c.is_alphanumeric()).collect();
                terms.push(alnum.to_lowercase());
            }
        }
        terms
    }

    fn push_word(&self, word: &str, terms: &mut Vec<String>) {
        let first = terms.len();
        for stemming in &self.0 {
            let term = stemming.reduce(word);
            if !terms
                .get(first..)
                .is_some_and(|added| added.contains(&term))
            {
                terms.push(term);
            }
        }
    }
}

/// Punctuation that joins alphanumeric runs into one identifier (`POL-8841`,
/// `v1.2.3`, `ns/part:7`) without introducing whitespace.
fn is_identifier_joiner(c: char) -> bool {
    matches!(c, '-' | '.' | '_' | '/' | ':')
}

/// The scripts written without spaces between words: Han, Hiragana,
/// Katakana, and Hangul, as the language detector classifies them.
pub struct Unspaced;

impl Unspaced {
    /// Whether `c` is written in one of them.
    #[must_use]
    pub fn holds(c: char) -> bool {
        if c.is_ascii() {
            return false;
        }
        let mut buf = [0_u8; 4];
        matches!(
            whatlang::detect_script(c.encode_utf8(&mut buf)),
            Some(Script::Mandarin | Script::Hiragana | Script::Katakana | Script::Hangul)
        )
    }

    /// Whether a term is a bigram (or a single character) of these scripts.
    #[must_use]
    pub fn is_term(term: &str) -> bool {
        !term.is_empty() && term.chars().all(Self::holds)
    }
}

/// A piece of an alphanumeric token: a word of a spaced script, or a run
/// of an unspaced one.
enum Segment<'a> {
    Spaced(&'a str),
    Unspaced(Vec<char>),
}

impl<'a> Segment<'a> {
    fn split(token: &'a str) -> Vec<Self> {
        let mut segments = Vec::new();
        let mut spaced_from: Option<usize> = None;
        let mut unspaced: Vec<char> = Vec::new();
        for (at, c) in token.char_indices() {
            if Unspaced::holds(c) {
                if let Some(word) = spaced_from.take().and_then(|from| token.get(from..at)) {
                    segments.push(Self::Spaced(word));
                }
                unspaced.push(c);
            } else {
                if !unspaced.is_empty() {
                    segments.push(Self::Unspaced(std::mem::take(&mut unspaced)));
                }
                spaced_from.get_or_insert(at);
            }
        }
        if let Some(word) = spaced_from.and_then(|from| token.get(from..)) {
            segments.push(Self::Spaced(word));
        }
        if !unspaced.is_empty() {
            segments.push(Self::Unspaced(unspaced));
        }
        segments
    }

    fn push_bigrams(chars: &[char], terms: &mut Vec<String>) {
        if let [only] = chars {
            terms.push(only.to_string());
            return;
        }
        for pair in chars.windows(2) {
            terms.push(pair.iter().collect());
        }
    }
}

/// `[retrieval].languages`: which languages a document may be detected as.
#[derive(Debug, Clone, Default, PartialEq, Eq, serde::Deserialize)]
#[serde(try_from = "Vec<String>")]
pub enum LanguageSetting {
    /// Any language the detector knows (`["auto"]`, the default).
    #[default]
    Auto,
    /// Only these: one is fixed, several are detected among.
    Only(Vec<Language>),
}

impl TryFrom<Vec<String>> for LanguageSetting {
    type Error = Error;

    fn try_from(names: Vec<String>) -> Result<Self> {
        if names.iter().any(|n| n.trim().eq_ignore_ascii_case("auto")) {
            if names.len() > 1 {
                return Err(Error::Config(String::from(
                    "[retrieval].languages is [\"auto\"] or a list of languages, not both",
                )));
            }
            return Ok(Self::Auto);
        }
        let mut languages = Vec::with_capacity(names.len());
        for name in &names {
            let language: Language = name.parse()?;
            if !languages.contains(&language) {
                languages.push(language);
            }
        }
        if languages.is_empty() {
            return Err(Error::Config(String::from(
                "[retrieval].languages is empty; use [\"auto\"] or name a language",
            )));
        }
        Ok(Self::Only(languages))
    }
}

impl fmt::Display for LanguageSetting {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let names: Vec<String> = match self {
            Self::Auto => vec![String::from("\"auto\"")],
            Self::Only(languages) => languages.iter().map(|l| format!("\"{l}\"")).collect(),
        };
        write!(f, "[{}]", names.join(", "))
    }
}

impl LanguageSetting {
    /// The ISO 639-3 code of the language `sample` (a document's opening
    /// text) is indexed under. Under `auto` a guess the detector is unsure
    /// of falls back to English, except in an unspaced script, whose
    /// characters already say what it is; under a list the closest listed
    /// language is taken.
    #[must_use]
    pub fn detect(&self, sample: &str) -> &'static str {
        let lang = match self {
            Self::Auto => Detector::new()
                .detect(sample)
                .filter(|info| {
                    info.is_reliable()
                        || matches!(
                            info.script(),
                            Script::Mandarin | Script::Hiragana | Script::Katakana | Script::Hangul
                        )
                })
                .map_or(Lang::Eng, |info| info.lang()),
            Self::Only(languages) => match languages.as_slice() {
                [] => Lang::Eng,
                [only] => only.lang(),
                [first, ..] => {
                    Detector::with_allowlist(languages.iter().map(|l| l.lang()).collect())
                        .detect_lang(sample)
                        .unwrap_or_else(|| first.lang())
                }
            },
        };
        lang.code()
    }
}

/// How often each term occurs in a chunk's content and heading, by term.
pub(super) struct TermFrequencies(pub(super) Vec<(String, u32)>);

impl TermFrequencies {
    pub(super) fn of(analyzer: &Analyzer, content: &str, heading: Option<&str>) -> Self {
        let mut counts: BTreeMap<String, u32> = BTreeMap::new();
        for term in analyzer
            .terms(content)
            .into_iter()
            .chain(heading.map(|h| analyzer.terms(h)).unwrap_or_default())
        {
            let entry = counts.entry(term).or_insert(0);
            *entry = entry.saturating_add(1);
        }
        Self(counts.into_iter().collect())
    }

    /// The distinct terms, for a query.
    pub(super) fn distinct(analyzer: &Analyzer, query: &str) -> Vec<String> {
        Self::of(analyzer, query, None)
            .0
            .into_iter()
            .map(|(t, _)| t)
            .collect()
    }

    /// Total term occurrences, the chunk length BM25 normalizes by.
    pub(super) fn total(&self) -> i64 {
        self.0
            .iter()
            .fold(0i64, |acc, (_, tf)| acc.saturating_add(i64::from(*tf)))
    }
}

#[cfg(test)]
mod tests;
