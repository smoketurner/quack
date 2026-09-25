//! Where a secret key lives: the OS keychain (`keychain.rs`) when it is
//! usable, else a key file with mode 0600.

use std::path::{Path, PathBuf};

use super::keychain::KeychainEntry;
use crate::error::Result;

/// Where the key may be kept.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KeySource {
    /// The OS keychain, falling back to the key file when it is unusable.
    Keychain,
    /// Only the key file (tests, and hosts with no keychain).
    File,
}

/// Where a key turned out to be.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KeyLocation {
    Keychain,
    File,
}

impl std::fmt::Display for KeyLocation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Keychain => "keychain",
            Self::File => "key file",
        })
    }
}

/// One key's keychain entry and fallback file. The stored text is whatever
/// the owner encodes (base64 of the key bytes).
#[derive(Debug, Clone)]
pub(crate) struct KeySlot {
    account: String,
    file: PathBuf,
    source: KeySource,
}

impl KeySlot {
    /// The key named `account` in the keychain, or in `file`.
    pub(crate) fn new(account: String, file: PathBuf, source: KeySource) -> Self {
        Self {
            account,
            file,
            source,
        }
    }

    /// Where the key is: the key file when there is one, else the keychain.
    pub(crate) fn location(&self) -> KeyLocation {
        if self.file.exists() {
            KeyLocation::File
        } else {
            KeyLocation::Keychain
        }
    }

    fn keychain(&self) -> KeychainEntry {
        KeychainEntry::new(self.account.clone())
    }

    /// The stored key, or `None` when there is none.
    ///
    /// # Errors
    ///
    /// Returns an error when the key file exists but cannot be read.
    pub(crate) async fn read(&self) -> Result<Option<String>> {
        if self.source == KeySource::Keychain {
            match self.keychain().get().await {
                Ok(Some(key)) => return Ok(Some(key)),
                Ok(None) => {}
                Err(e) => {
                    tracing::warn!(account = %self.account, error = %e, "keychain unavailable; using the key file");
                }
            }
        }
        match std::fs::read_to_string(&self.file) {
            Ok(key) => Ok(Some(key)),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(e) => Err(e.into()),
        }
    }

    /// The stored key, or a new one from `make`, stored where it will be
    /// found again: the keychain when it takes it, else the key file.
    ///
    /// # Errors
    ///
    /// Returns an error when the key cannot be made or stored anywhere.
    pub(crate) async fn read_or_create(
        &self,
        make: impl FnOnce() -> Result<String>,
    ) -> Result<String> {
        if let Some(key) = self.read().await? {
            return Ok(key);
        }
        let key = make()?;
        if self.source == KeySource::Keychain {
            match self.keychain().set(&key).await {
                Ok(()) => return Ok(key),
                Err(e) => {
                    tracing::warn!(account = %self.account, error = %e, "keychain unavailable; writing the key file");
                }
            }
        }
        if let Some(dir) = self.file.parent() {
            std::fs::create_dir_all(dir)?;
        }
        write_private(&self.file, key.as_bytes())?;
        Ok(key)
    }

    /// Remove the key wherever it is.
    ///
    /// # Errors
    ///
    /// Returns an error when the key file cannot be removed.
    pub(crate) async fn delete(&self) -> Result<()> {
        remove_if_present(&self.file)?;
        if self.source == KeySource::Keychain
            && let Err(e) = self.keychain().delete().await
        {
            tracing::warn!(account = %self.account, error = %e, "keychain entry not removed");
        }
        Ok(())
    }
}

pub(crate) fn remove_if_present(path: &Path) -> Result<()> {
    match std::fs::remove_file(path) {
        Ok(()) => Ok(()),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(e) => Err(e.into()),
    }
}

/// Write a file readable only by its owner.
pub(crate) fn write_private(path: &Path, bytes: &[u8]) -> Result<()> {
    use std::io::Write;
    let mut options = std::fs::OpenOptions::new();
    options.write(true).create(true).truncate(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    let mut file = options.open(path)?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        file.set_permissions(std::fs::Permissions::from_mode(0o600))?;
    }
    file.write_all(bytes)?;
    file.flush()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[expect(clippy::panic, reason = "test failure path")]
    fn fail(msg: &str) -> ! {
        panic!("{msg}")
    }

    #[cfg(unix)]
    #[test]
    fn private_files_are_owner_only() {
        use std::os::unix::fs::PermissionsExt;
        let dir = tempfile::tempdir().unwrap_or_else(|e| fail(&e.to_string()));
        let path = dir.path().join("k");
        assert!(write_private(&path, b"x").is_ok());
        let mode = std::fs::metadata(&path).map(|m| m.permissions().mode() & 0o777);
        assert!(mode.is_ok_and(|m| m == 0o600));
    }

    #[tokio::test]
    async fn a_file_slot_makes_its_key_once_and_forgets_it_on_delete() {
        let dir = tempfile::tempdir().unwrap_or_else(|e| fail(&e.to_string()));
        let slot = KeySlot::new(
            String::from("test"),
            dir.path().join("s.key"),
            KeySource::File,
        );
        assert!(slot.read().await.is_ok_and(|k| k.is_none()));
        let made = slot.read_or_create(|| Ok(String::from("first"))).await;
        assert!(made.is_ok_and(|k| k == "first"));
        let again = slot.read_or_create(|| Ok(String::from("second"))).await;
        assert!(again.is_ok_and(|k| k == "first"));
        assert_eq!(slot.location(), KeyLocation::File);
        assert!(slot.delete().await.is_ok());
        assert!(slot.read().await.is_ok_and(|k| k.is_none()));
        assert!(slot.delete().await.is_ok());
    }
}
