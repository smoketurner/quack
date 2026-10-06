use super::*;

fn german() -> Analyzer {
    Analyzer::of(Stemming::Snowball(Language(Algorithm::German)))
}

#[test]
fn unspaced_scripts_become_bigrams_and_a_lone_character_stays_whole() {
    let terms = Analyzer::default().terms("保険契約の更新手続き");
    assert_eq!(
        terms,
        [
            "保険", "険契", "契約", "約の", "の更", "更新", "新手", "手続", "続き"
        ]
    );
    // A query for a run inside the sentence shares its bigrams.
    let query = Analyzer::default().terms("更新手続き");
    assert!(query.iter().all(|t| terms.contains(t)), "{query:?}");
    assert_eq!(Analyzer::default().terms("東"), ["東"]);
    // Hangul is bigrammed the same way.
    assert_eq!(
        Analyzer::default().terms("보험계약"),
        ["보험", "험계", "계약"]
    );
}

#[test]
fn a_mixed_token_splits_by_script_and_keeps_no_joined_form_for_unspaced_text() {
    assert_eq!(
        Analyzer::default().terms("Quack漢字Renewals"),
        ["quack", "漢字", "renew"]
    );
    // A joined identifier is still indexed whole when it is all spaced.
    assert!(
        Analyzer::default()
            .terms("POL-8841")
            .contains(&String::from("pol8841"))
    );
    let mixed = Analyzer::default().terms("ID-漢字");
    assert_eq!(mixed, ["id", "漢字"], "no joined form across scripts");
}

#[test]
fn german_inflections_meet_under_the_german_stemmer_but_not_the_english_one() {
    assert_eq!(
        german().terms("Versicherungsverträge"),
        german().terms("Versicherungsvertrag")
    );
    assert_ne!(
        Analyzer::default().terms("Versicherungsverträge"),
        Analyzer::default().terms("Versicherungsvertrag")
    );
}

#[test]
fn a_query_is_stemmed_under_every_language_once_per_distinct_stem() {
    let both = Analyzer::of_recorded(Some("english,german"));
    let terms = both.terms("renewals data");
    assert_eq!(
        terms,
        [
            Stemming::Snowball(Language::ENGLISH).reduce("renewals"),
            Stemming::Snowball(Language(Algorithm::German)).reduce("renewals"),
            String::from("data"),
        ]
    );
    let unstemmed = Analyzer::of_recorded(Some("unstemmed")).terms("Renewals");
    assert_eq!(unstemmed, ["renewals"]);
}

#[test]
fn the_recorded_set_round_trips_and_defaults_to_english() {
    let recorded = Analyzer::record([Some("deu"), Some("eng"), Some("cmn"), Some("deu"), None]);
    assert_eq!(recorded, "english,german,unstemmed");
    assert_eq!(
        Analyzer::of_recorded(Some(&recorded)).stemmings(),
        [
            Stemming::Snowball(Language::ENGLISH),
            Stemming::Snowball(Language(Algorithm::German)),
            Stemming::Unstemmed,
        ]
    );
    assert_eq!(Analyzer::of_recorded(None), Analyzer::default());
    assert_eq!(Analyzer::of_recorded(Some("")), Analyzer::default());
    assert_eq!(
        Analyzer::of_recorded(Some("klingon, german")).stemmings(),
        [Stemming::Snowball(Language(Algorithm::German))],
        "an unknown name is skipped"
    );
}

#[test]
fn a_stored_code_maps_to_its_stemmer_or_to_none() {
    assert_eq!(
        Stemming::of_code(None),
        Stemming::Snowball(Language::ENGLISH)
    );
    assert_eq!(
        Stemming::of_code(Some("deu")),
        Stemming::Snowball(Language(Algorithm::German))
    );
    assert_eq!(Stemming::of_code(Some("cmn")), Stemming::Unstemmed);
    assert_eq!(Stemming::of_code(Some("zzz")), Stemming::Unstemmed);
    for language in Language::ALL {
        assert_eq!(Language::of_lang(language.lang()), Some(language));
        assert_eq!(language.name().parse::<Language>().ok(), Some(language));
    }
}

#[test]
fn the_setting_is_auto_or_a_list_of_known_languages() {
    let parse = |names: &[&str]| {
        LanguageSetting::try_from(names.iter().map(|n| (*n).to_owned()).collect::<Vec<_>>())
    };
    assert_eq!(parse(&["auto"]).ok(), Some(LanguageSetting::Auto));
    assert_eq!(
        parse(&["German", "english", "german"]).ok(),
        Some(LanguageSetting::Only(vec![
            Language(Algorithm::German),
            Language::ENGLISH
        ]))
    );
    let mixed = parse(&["auto", "german"]).err().map(|e| e.to_string());
    assert!(mixed.is_some_and(|e| e.contains("not both")));
    let empty = parse(&[]).err().map(|e| e.to_string());
    assert!(empty.is_some_and(|e| e.contains("empty")));
    let unknown = parse(&["klingon"]).err().map(|e| e.to_string());
    assert!(unknown.is_some_and(|e| e.contains("klingon") && e.contains("german")));
    assert_eq!(LanguageSetting::Auto.to_string(), "[\"auto\"]");
    assert_eq!(
        LanguageSetting::Only(vec![Language::ENGLISH]).to_string(),
        "[\"english\"]"
    );
}

const GERMAN: &str = "Die Versicherungsverträge werden jedes Jahr erneuert. Der Kunde \
    erhält rechtzeitig eine Mitteilung über die neuen Bedingungen und kann widersprechen.";
const ENGLISH: &str = "The insurance policies are renewed every year. The customer \
    receives a notice about the new terms in time and may object to them.";

#[test]
fn detection_names_the_language_or_falls_back() {
    assert_eq!(LanguageSetting::Auto.detect(GERMAN), "deu");
    assert_eq!(LanguageSetting::Auto.detect(ENGLISH), "eng");
    assert_eq!(
        LanguageSetting::Auto.detect("保険契約の更新手続きについて説明します。"),
        "jpn"
    );
    assert_eq!(
        LanguageSetting::Auto.detect("保险合同每年续签一次。"),
        "cmn"
    );
    // Too little text to tell: English, as before detection.
    assert_eq!(LanguageSetting::Auto.detect("x"), "eng");
    assert_eq!(LanguageSetting::Auto.detect(""), "eng");
    let only_german = LanguageSetting::Only(vec![Language(Algorithm::German)]);
    assert_eq!(only_german.detect(ENGLISH), "deu", "one language is fixed");
    let pair = LanguageSetting::Only(vec![Language::ENGLISH, Language(Algorithm::German)]);
    assert_eq!(pair.detect(GERMAN), "deu");
    assert_eq!(pair.detect(ENGLISH), "eng");
    assert_eq!(
        pair.detect("12345"),
        "eng",
        "nothing to detect: the first listed"
    );
}

#[test]
fn unspaced_terms_are_recognized() {
    assert!(Unspaced::is_term("更新"));
    assert!(Unspaced::is_term("カ"));
    assert!(!Unspaced::is_term("abc"));
    assert!(!Unspaced::is_term(""));
    assert!(!Unspaced::holds('a'));
    assert!(!Unspaced::holds('ä'));
}

#[test]
fn term_frequencies_count_a_heading_and_give_distinct_query_terms() {
    let tf = TermFrequencies::of(&Analyzer::default(), "更新 更新", Some("更新"));
    assert_eq!(tf.0, [(String::from("更新"), 3)]);
    assert_eq!(tf.total(), 3);
    let distinct = TermFrequencies::distinct(&Analyzer::default(), "flood flood damage");
    assert_eq!(distinct, ["damag", "flood"]);
}
