//! `TextFormat::Pdf` on arbitrary bytes: the parser may refuse the input,
//! never panic or run away.
#![no_main]

use libfuzzer_sys::fuzz_target;
use quack_core::ingestion::budget::DecompressionBudget;
use quack_core::ingestion::parser::TextFormat;

fuzz_target!(|data: &[u8]| {
    let _ = TextFormat::Pdf.extract(data, DecompressionBudget::megabytes(64));
});
