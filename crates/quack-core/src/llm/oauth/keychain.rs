//! The OS keychain entry that holds a token cache's encryption key.
//!
//! One store per platform: the macOS Keychain, the Linux kernel keyring
//! (`keyutils`, always present, in-memory only, so a reboot needs a new
//! login), and the Windows Credential Manager. Every call runs on the
//! blocking pool because the stores talk to the OS synchronously.

use std::sync::{Arc, OnceLock};

use keyring_core::{CredentialStore, Entry};

use crate::error::{Error, Result};

const SERVICE: &str = "quack";

fn platform_store() -> keyring_core::Result<Arc<CredentialStore>> {
    #[cfg(target_os = "macos")]
    {
        apple_native_keyring_store::keychain::Store::new().map(|s| -> Arc<CredentialStore> { s })
    }
    #[cfg(target_os = "linux")]
    {
        linux_keyutils_keyring_store::Store::new().map(|s| -> Arc<CredentialStore> { s })
    }
    #[cfg(target_os = "windows")]
    {
        windows_native_keyring_store::Store::new().map(|s| -> Arc<CredentialStore> { s })
    }
    #[cfg(not(any(target_os = "macos", target_os = "linux", target_os = "windows")))]
    {
        Err(keyring_core::Error::NoDefaultStore)
    }
}

/// Install the platform store once per process. Later calls return the
/// first outcome, so an unavailable keychain stays unavailable instead of
/// flapping between the keychain and the key file.
fn ensure_store() -> keyring_core::Result<()> {
    static INIT: OnceLock<std::result::Result<(), String>> = OnceLock::new();
    INIT.get_or_init(|| {
        if keyring_core::get_default_store().is_some() {
            return Ok(());
        }
        platform_store()
            .map(keyring_core::set_default_store)
            .map_err(|e| e.to_string())
    })
    .clone()
    .map_err(keyring_core::Error::BadStoreFormat)
}

/// A keychain entry under the `quack` service, named by its account.
pub(super) struct KeychainEntry(String);

/// What is done to an entry.
enum KeychainOp {
    Read,
    Write(String),
    Delete,
}

impl KeychainOp {
    const fn verb(&self) -> &'static str {
        match self {
            Self::Read => "read",
            Self::Write(_) => "write",
            Self::Delete => "delete",
        }
    }
}

impl KeychainEntry {
    pub(super) const fn new(account: String) -> Self {
        Self(account)
    }

    /// Do `op` on the blocking pool, since the stores talk to the OS
    /// synchronously. A read of a missing entry is `None`; deleting one is
    /// not an error.
    async fn run(&self, op: KeychainOp) -> Result<Option<String>> {
        let account = self.0.clone();
        tokio::task::spawn_blocking(move || {
            let verb = op.verb();
            let outcome = ensure_store()
                .and_then(|()| Entry::new(SERVICE, &account))
                .and_then(|entry| match op {
                    KeychainOp::Read => entry.get_password().map(Some),
                    KeychainOp::Write(key) => entry.set_password(&key).map(|()| None),
                    KeychainOp::Delete => entry.delete_credential().map(|()| None),
                });
            match outcome {
                Ok(found) => Ok(found),
                Err(keyring_core::Error::NoEntry) => Ok(None),
                Err(e) => Err(Error::Llm(format!(
                    "keychain {verb} of '{account}' failed: {e}"
                ))),
            }
        })
        .await
        .map_err(|e| Error::Llm(format!("keychain task failed: {e}")))?
    }

    /// The stored key, or `None` when the keychain has no entry.
    ///
    /// # Errors
    ///
    /// Returns an error when the keychain itself is unavailable or refuses
    /// the read; callers fall back to the key file on that.
    pub(super) async fn get(&self) -> Result<Option<String>> {
        self.run(KeychainOp::Read).await
    }

    /// Store the key, replacing any previous one.
    ///
    /// # Errors
    ///
    /// Returns an error when the keychain is unavailable or refuses the
    /// write.
    pub(super) async fn set(&self, key: &str) -> Result<()> {
        self.run(KeychainOp::Write(key.to_owned())).await.map(drop)
    }

    /// Remove the key. A missing entry is not an error.
    ///
    /// # Errors
    ///
    /// Returns an error when the keychain is unavailable or refuses the
    /// delete.
    pub(super) async fn delete(&self) -> Result<()> {
        self.run(KeychainOp::Delete).await.map(drop)
    }
}
