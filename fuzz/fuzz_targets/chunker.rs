//! The chunker over whatever the Markdown and text parsers make of
//! arbitrary bytes: every window must fall inside the text.
#![no_main]

use libfuzzer_sys::fuzz_target;
use quack_core::ingestion::budget::DecompressionBudget;
use quack_core::ingestion::chunker::Chunker;
use quack_core::ingestion::parser::TextFormat;

fuzz_target!(|data: &[u8]| {
    let Ok(chunker) = Chunker::new(64, 16, "cl100k_base") else {
        return;
    };
    for format in [TextFormat::Markdown, TextFormat::Text] {
        if let Ok(extracted) = format.extract(data, DecompressionBudget::megabytes(64)) {
            let _ = chunker.document(&extracted, Some("fuzz"));
        }
    }
});
