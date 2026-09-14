//! Process-wide crypto provider installation.
//!
//! rustls needs exactly one default `CryptoProvider` per process. This
//! project uses aws-lc-rs and nothing else (`deny.toml` bans OpenSSL and
//! `ring`), so every binary installs it before any TLS connection exists.

use crate::error::{Error, Result};

/// Install aws-lc-rs as the process-wide rustls crypto provider.
///
/// Call this first in `main`, before building any HTTP client or runtime.
///
/// # Errors
///
/// Returns an error if a provider was already installed, which means some
/// other code ran TLS setup first and the ordering guarantee is broken.
pub fn install_default_provider() -> Result<()> {
    rustls::crypto::aws_lc_rs::default_provider()
        .install_default()
        .map_err(|_| {
            Error::Config(String::from(
                "a rustls crypto provider was already installed",
            ))
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[expect(clippy::unwrap_used, reason = "test asserts Err")]
    fn second_install_in_same_process_is_an_error() {
        // The first call may succeed or fail depending on which test ran
        // first in this process; the second call must always fail.
        drop(install_default_provider());
        let err = install_default_provider().err().unwrap();
        assert!(err.to_string().contains("already installed"), "{err}");
    }
}
