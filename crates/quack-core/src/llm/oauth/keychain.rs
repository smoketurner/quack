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

fn entry(provider: &str) -> keyring_core::Result<Entry> {
    ensure_store()?;
    Entry::new(SERVICE, &format!("oauth:{provider}"))
}

fn keychain_error(provider: &str, action: &str, e: &keyring_core::Error) -> Error {
    Error::Llm(format!(
        "keychain {action} for provider '{provider}' failed: {e}"
    ))
}

/// The stored key, or `None` when the keychain has no entry for the provider.
///
/// # Errors
///
/// Returns an error when the keychain itself is unavailable or refuses the
/// read; callers fall back to the key file on that.
pub(super) async fn get(provider: &str) -> Result<Option<String>> {
    let provider = provider.to_owned();
    tokio::task::spawn_blocking(
        move || match entry(&provider).and_then(|e| e.get_password()) {
            Ok(key) => Ok(Some(key)),
            Err(keyring_core::Error::NoEntry) => Ok(None),
            Err(e) => Err(keychain_error(&provider, "read", &e)),
        },
    )
    .await
    .map_err(|e| Error::Llm(format!("keychain task failed: {e}")))?
}

/// Store the key for the provider, replacing any previous one.
///
/// # Errors
///
/// Returns an error when the keychain is unavailable or refuses the write.
pub(super) async fn set(provider: &str, key: &str) -> Result<()> {
    let provider = provider.to_owned();
    let key = key.to_owned();
    tokio::task::spawn_blocking(move || {
        entry(&provider)
            .and_then(|e| e.set_password(&key))
            .map_err(|e| keychain_error(&provider, "write", &e))
    })
    .await
    .map_err(|e| Error::Llm(format!("keychain task failed: {e}")))?
}

/// Remove the provider's key. A missing entry is not an error.
///
/// # Errors
///
/// Returns an error when the keychain is unavailable or refuses the delete.
pub(super) async fn delete(provider: &str) -> Result<()> {
    let provider = provider.to_owned();
    tokio::task::spawn_blocking(move || {
        match entry(&provider).and_then(|e| e.delete_credential()) {
            Ok(()) | Err(keyring_core::Error::NoEntry) => Ok(()),
            Err(e) => Err(keychain_error(&provider, "delete", &e)),
        }
    })
    .await
    .map_err(|e| Error::Llm(format!("keychain task failed: {e}")))?
}
