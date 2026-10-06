//! A workbook's sheets from arbitrary bytes: refused or read, never a panic.
#![no_main]

use libfuzzer_sys::fuzz_target;
use quack_core::ingestion::budget::DecompressionBudget;
use quack_core::ingestion::xlsx;

fuzz_target!(|data: &[u8]| {
    let _ = xlsx::sheets(data, DecompressionBudget::megabytes(64));
});
