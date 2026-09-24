//! Process-wide crypto provider installation.
//!
//! rustls needs exactly one default `CryptoProvider` per process. This
//! project uses aws-lc-rs and nothing else (`deny.toml` bans OpenSSL and
//! `ring`), so every binary installs it before any TLS connection exists.
//!
//! On Linux that provider is the FIPS-validated AWS-LC module: `quack-core`
//! enables aws-lc-rs and rustls with their `fips` features for
//! `cfg(target_os = "linux")`, the only family where `aws-lc-fips-sys` emits a
//! static library that a single-file release archive can carry
//! (`docs/crypto.md`). The cipher suite list narrows to the approved ones
//! there; macOS and Windows run the same code on aws-lc-sys.

use crate::error::{Error, Result};

/// Install aws-lc-rs as the process-wide rustls crypto provider.
///
/// Call this first in `main`, before building any HTTP client or runtime.
/// Call [`CryptoModule::log`] once a tracing subscriber exists to record
/// which module this installed.
///
/// # Errors
///
/// Returns an error if a provider was already installed, which means some
/// other code ran TLS setup first and the ordering guarantee is broken.
pub fn install_default_provider() -> Result<()> {
    // On Linux the rustls `fips` feature is enabled, which restricts the
    // provider's key-exchange groups to the FIPS-approved set.
    // `default_fips_provider` exists only under that feature and returns the
    // same provider, so naming it here fails the build if the feature is ever
    // dropped rather than silently restoring non-FIPS key exchange.
    #[cfg(target_os = "linux")]
    let provider = rustls::crypto::default_fips_provider();
    #[cfg(not(target_os = "linux"))]
    let provider = rustls::crypto::aws_lc_rs::default_provider();

    provider.install_default().map_err(|_| {
        Error::Config(String::from(
            "a rustls crypto provider was already installed",
        ))
    })
}

/// The AWS-LC module this binary links: the library version that matters for
/// a CVE or a certificate (`awslc_version`, the trailing token of
/// `AWS-LC FIPS 4.2.0`) and, for a FIPS build, the module version that names
/// the certification. aws-lc-rs has no runtime API for its own crate version,
/// which stays in `Cargo.lock`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CryptoModule {
    pub awslc: &'static str,
    pub fips_module: Option<u32>,
}

impl CryptoModule {
    /// The module linked into this binary.
    #[must_use]
    pub fn linked() -> Self {
        Self {
            awslc: aws_lc_rs::awslc_version(),
            fips_module: aws_lc_rs::fips_version(),
        }
    }

    /// Whether this is a Linux build without the FIPS module, which every
    /// release for Linux links.
    #[must_use]
    pub fn lacks_expected_fips(self) -> bool {
        self.fips_module.is_none() && cfg!(target_os = "linux")
    }

    /// Log which module the installed provider runs on.
    ///
    /// Belongs right after the tracing subscriber is installed, not next to
    /// [`install_default_provider`]: that runs at the top of `main`, before
    /// any subscriber exists, so a log there goes nowhere. A build that
    /// should be FIPS but reports no FIPS module still works, so this warns
    /// rather than failing.
    pub fn log(self) {
        let awslc = self.awslc;
        let fips =
            rustls::crypto::CryptoProvider::get_default().is_some_and(|provider| provider.fips());
        if fips {
            tracing::info!(
                awslc,
                fips_module = self.fips_module,
                "installed the FIPS AWS-LC crypto provider"
            );
        } else if cfg!(target_os = "linux") {
            tracing::warn!(
                awslc,
                "installed a non-FIPS AWS-LC crypto provider on Linux"
            );
        } else {
            tracing::info!(awslc, "installed the AWS-LC crypto provider");
        }
    }
}

/// One line for `--version`: `AWS-LC FIPS 4.2.0 (FIPS module 40200)` for a
/// FIPS build, `AWS-LC 5.7.0` otherwise.
impl std::fmt::Display for CryptoModule {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.fips_module {
            Some(module) => write!(f, "AWS-LC FIPS {} (FIPS module {module})", self.awslc),
            None => write!(f, "AWS-LC {}", self.awslc),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_provider_is_fips_on_linux_and_not_elsewhere() {
        // Guards the target-gated features in Cargo.toml: dropping `fips` from
        // the Linux dependency, or adding it where aws-lc-fips-sys links a
        // shared library instead of a static one, changes this.
        let provider = rustls::crypto::aws_lc_rs::default_provider();
        assert_eq!(provider.fips(), cfg!(target_os = "linux"));
        assert_eq!(aws_lc_rs::fips_version().is_some(), provider.fips());
    }

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
