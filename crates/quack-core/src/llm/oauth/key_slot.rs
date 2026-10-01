//! Where a secret key lives: the OS keychain (`keychain.rs`) when it is
//! usable, else a key file with mode 0600.

use std::path::{Path, PathBuf};

use super::keychain::{Keychain, KeychainError};
use crate::error::{Error, Result};

/// Where the key may be kept.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KeySource {
    /// The OS keychain, falling back to the key file when this host has no
    /// usable keychain (not when a keychain is locked).
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
///
/// The keychain comes first. The key file is used when the keychain has no
/// entry and the file exists (a key made while this host had no keychain),
/// or when this host has no keychain at all. A keychain that is there but
/// refuses (locked, access denied) is an error, never a reason to use or
/// make the file: the keychain may hold the key that sealed what is
/// stored, and a key made in the file would be shadowed by it later (#226).
#[derive(Debug, Clone)]
pub(crate) struct KeySlot {
    account: String,
    file: PathBuf,
    /// `None` for [`KeySource::File`].
    keychain: Option<Keychain>,
}

/// What a lookup found.
enum Found {
    /// The key, and where it was.
    Key(String, KeyLocation),
    /// No key yet; a new one goes here.
    Missing(KeyLocation),
}

impl KeySlot {
    /// The key named `account` in the keychain, or in `file`.
    pub(crate) fn new(account: String, file: PathBuf, source: KeySource) -> Self {
        Self {
            account,
            file,
            keychain: match source {
                KeySource::Keychain => Some(Keychain::Os),
                KeySource::File => None,
            },
        }
    }

    /// A slot over `keychain` in place of the OS one.
    #[cfg(test)]
    pub(super) fn with_keychain(account: String, file: PathBuf, keychain: Keychain) -> Self {
        Self {
            account,
            file,
            keychain: Some(keychain),
        }
    }

    /// Where the key is, looked up in the order [`Self::read`] uses; where a
    /// new one would go when there is none yet.
    ///
    /// # Errors
    ///
    /// As [`Self::read`].
    pub(crate) async fn location(&self) -> Result<KeyLocation> {
        Ok(match self.find().await? {
            Found::Key(_, location) | Found::Missing(location) => location,
        })
    }

    /// The stored key, or `None` when there is none.
    ///
    /// # Errors
    ///
    /// Returns an error when the keychain refuses the read (locked or
    /// denied), or the key file exists but cannot be read.
    pub(crate) async fn read(&self) -> Result<Option<String>> {
        Ok(match self.find().await? {
            Found::Key(key, _) => Some(key),
            Found::Missing(_) => None,
        })
    }

    /// The one precedence every lookup follows: the keychain's entry, else
    /// the key file.
    async fn find(&self) -> Result<Found> {
        let Some(keychain) = &self.keychain else {
            return self.find_in_file(KeyLocation::File);
        };
        match keychain.get(&self.account).await {
            Ok(Some(key)) => Ok(Found::Key(key, KeyLocation::Keychain)),
            Ok(None) => self.find_in_file(KeyLocation::Keychain),
            Err(KeychainError::Unavailable(e)) => {
                tracing::warn!(account = %self.account, error = %e, "no keychain on this host; using the key file");
                self.find_in_file(KeyLocation::File)
            }
            Err(KeychainError::Refused(e)) => Err(self.refused("read", &e)),
        }
    }

    /// The key file's key, or `Missing(otherwise)`.
    fn find_in_file(&self, otherwise: KeyLocation) -> Result<Found> {
        match std::fs::read_to_string(&self.file) {
            Ok(key) => Ok(Found::Key(key, KeyLocation::File)),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(Found::Missing(otherwise)),
            Err(e) => Err(e.into()),
        }
    }

    fn refused(&self, verb: &str, reason: &str) -> Error {
        Error::Vault(format!(
            "the OS keychain refused to {verb} the '{account}' key ({reason}); unlock the \
             keychain or allow quack to use it, then retry. quack does not fall back to {file} \
             while a keychain is present, since the keychain may hold the key that sealed the \
             stored tokens",
            account = self.account,
            file = self.file.display(),
        ))
    }

    /// The stored key, or a new one from `make`, stored where it will be
    /// found again: the keychain when this host has one, else the key file.
    /// Every `KeySlot` over the same file, in this process or another,
    /// takes the file's lock first, so they all agree on one key.
    ///
    /// # Errors
    ///
    /// Returns an error when the key cannot be made, the keychain refuses
    /// (a locked keychain never leads to a new key in the file), or the key
    /// file or its lock cannot be written.
    pub(crate) async fn read_or_create(
        &self,
        make: impl FnOnce() -> Result<String>,
    ) -> Result<String> {
        let _lock = KeyFileLock::acquire(&self.file).await?;
        let location = match self.find().await? {
            Found::Key(key, _) => return Ok(key),
            Found::Missing(location) => location,
        };
        let key = make()?;
        if let (KeyLocation::Keychain, Some(keychain)) = (location, &self.keychain) {
            match keychain.set(&self.account, &key).await {
                Ok(()) => return Ok(key),
                Err(KeychainError::Unavailable(e)) => {
                    tracing::warn!(account = %self.account, error = %e, "no keychain on this host; writing the key file");
                }
                Err(KeychainError::Refused(e)) => return Err(self.refused("store", &e)),
            }
        }
        publish_key(&self.file, &key)
    }
}

/// An exclusive lock on `<key file>.lock`, held from lookup to write. It is
/// an OS file lock, so it also orders two quack processes on one data
/// directory; released on drop.
struct KeyFileLock {
    _held: std::fs::File,
}

impl KeyFileLock {
    async fn acquire(key_file: &Path) -> Result<Self> {
        let mut name = key_file.as_os_str().to_owned();
        name.push(".lock");
        let path = PathBuf::from(name);
        tokio::task::spawn_blocking(move || {
            if let Some(dir) = path.parent() {
                std::fs::create_dir_all(dir)?;
            }
            let file = private_options().create(true).truncate(false).open(&path)?;
            file.lock()?;
            Ok(Self { _held: file })
        })
        .await
        .map_err(|e| Error::Vault(format!("the key file lock task failed: {e}")))?
    }
}

/// Write options for a file only its owner can read.
fn private_options() -> std::fs::OpenOptions {
    let mut options = std::fs::OpenOptions::new();
    options.write(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    options
}

/// Store `key` at `path` unless a key is already there, returning the key
/// the file holds. The key is written to a temporary file and hard-linked
/// into place, so a reader never sees an empty or partial file and an
/// existing key is never overwritten.
fn publish_key(path: &Path, key: &str) -> Result<String> {
    use std::io::Write;
    let dir = path.parent().unwrap_or_else(|| Path::new("."));
    std::fs::create_dir_all(dir)?;
    let temp = dir.join(format!(".{}.tmp", uuid::Uuid::now_v7()));
    let written = private_options()
        .create_new(true)
        .open(&temp)
        .and_then(|mut file| {
            file.write_all(key.as_bytes())?;
            file.sync_all()
        })
        .and_then(|()| std::fs::hard_link(&temp, path));
    drop(std::fs::remove_file(&temp));
    match written {
        Ok(()) => Ok(key.to_owned()),
        Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
            Ok(std::fs::read_to_string(path)?)
        }
        Err(e) => Err(e.into()),
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::super::keychain::fake::{FakeKeychain, Mode};
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
        assert!(publish_key(&path, "x").is_ok());
        let mode = std::fs::metadata(&path).map(|m| m.permissions().mode() & 0o777);
        assert!(mode.is_ok_and(|m| m == 0o600));
    }

    #[tokio::test]
    async fn a_file_slot_makes_its_key_once() {
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
        assert!(slot.location().await.is_ok_and(|l| l == KeyLocation::File));
    }

    fn fake_slot(dir: &Path, mode: Mode) -> (Arc<FakeKeychain>, KeySlot) {
        let fake = Arc::new(FakeKeychain::new(mode));
        let slot = KeySlot::with_keychain(
            String::from("vault"),
            dir.join("vault.key"),
            Keychain::Fake(Arc::clone(&fake)),
        );
        (fake, slot)
    }

    #[tokio::test]
    async fn a_working_keychain_holds_the_key_and_no_file_is_written() {
        let dir = tempfile::tempdir().unwrap_or_else(|e| fail(&e.to_string()));
        let (fake, slot) = fake_slot(dir.path(), Mode::Open);
        assert!(
            slot.location()
                .await
                .is_ok_and(|l| l == KeyLocation::Keychain)
        );
        let made = slot.read_or_create(|| Ok(String::from("k1"))).await;
        assert!(made.is_ok_and(|k| k == "k1"));
        assert_eq!(fake.peek("vault").as_deref(), Some("k1"));
        assert!(!dir.path().join("vault.key").exists());
        assert!(
            slot.location()
                .await
                .is_ok_and(|l| l == KeyLocation::Keychain)
        );
    }

    #[tokio::test]
    async fn a_host_without_a_keychain_uses_the_key_file() {
        let dir = tempfile::tempdir().unwrap_or_else(|e| fail(&e.to_string()));
        let (fake, slot) = fake_slot(dir.path(), Mode::Absent);
        assert!(slot.location().await.is_ok_and(|l| l == KeyLocation::File));
        let made = slot.read_or_create(|| Ok(String::from("k1"))).await;
        assert!(made.is_ok_and(|k| k == "k1"));
        assert!(fake.peek("vault").is_none());
        let again = slot.read_or_create(|| Ok(String::from("k2"))).await;
        assert!(again.is_ok_and(|k| k == "k1"));
        assert!(slot.location().await.is_ok_and(|l| l == KeyLocation::File));
    }

    #[tokio::test]
    async fn a_locked_keychain_is_an_error_and_makes_no_key_file() {
        let dir = tempfile::tempdir().unwrap_or_else(|e| fail(&e.to_string()));
        let (fake, slot) = fake_slot(dir.path(), Mode::Open);
        assert!(slot.read_or_create(|| Ok(String::from("k1"))).await.is_ok());
        fake.set_mode(Mode::Locked);

        let created = slot.read_or_create(|| Ok(String::from("orphan"))).await;
        assert!(matches!(&created, Err(Error::Vault(m)) if m.contains("refused")));
        assert!(slot.read().await.is_err());
        assert!(slot.location().await.is_err());
        assert!(!dir.path().join("vault.key").exists());

        // Unlocked again, the original key is still the one found.
        fake.set_mode(Mode::Open);
        assert!(slot.read().await.is_ok_and(|k| k.as_deref() == Some("k1")));
    }

    #[tokio::test]
    async fn a_locked_empty_keychain_does_not_make_a_key_in_the_file() {
        let dir = tempfile::tempdir().unwrap_or_else(|e| fail(&e.to_string()));
        let (fake, slot) = fake_slot(dir.path(), Mode::Locked);
        assert!(
            slot.read_or_create(|| Ok(String::from("k1")))
                .await
                .is_err()
        );
        assert!(!dir.path().join("vault.key").exists());
        assert!(fake.peek("vault").is_none());
    }

    #[tokio::test]
    async fn location_follows_the_read_order() {
        let dir = tempfile::tempdir().unwrap_or_else(|e| fail(&e.to_string()));
        let (fake, slot) = fake_slot(dir.path(), Mode::Absent);
        // A key file left from a run without a keychain...
        assert!(
            slot.read_or_create(|| Ok(String::from("file-key")))
                .await
                .is_ok()
        );
        fake.set_mode(Mode::Open);
        // ...is still what an empty keychain defers to.
        assert!(
            slot.read()
                .await
                .is_ok_and(|k| k.as_deref() == Some("file-key"))
        );
        assert!(slot.location().await.is_ok_and(|l| l == KeyLocation::File));
        // Once the keychain has an entry, both read and location say keychain.
        assert!(
            Keychain::Fake(Arc::clone(&fake))
                .set("vault", "chain-key")
                .await
                .is_ok()
        );
        assert!(
            slot.read()
                .await
                .is_ok_and(|k| k.as_deref() == Some("chain-key"))
        );
        assert!(
            slot.location()
                .await
                .is_ok_and(|l| l == KeyLocation::Keychain)
        );
    }

    #[test]
    fn publish_key_creates_a_fresh_file_with_the_given_key() {
        let dir = tempfile::tempdir().unwrap_or_else(|e| fail(&e.to_string()));
        let path = dir.path().join("v.key");
        let key = publish_key(&path, "ours").unwrap_or_else(|e| fail(&e.to_string()));
        assert_eq!(key, "ours");
        assert_eq!(
            std::fs::read_to_string(&path).unwrap_or_else(|e| fail(&e.to_string())),
            "ours"
        );
    }

    #[test]
    fn publish_key_reuses_a_pre_existing_file_instead_of_overwriting() {
        let dir = tempfile::tempdir().unwrap_or_else(|e| fail(&e.to_string()));
        let path = dir.path().join("v.key");
        std::fs::write(&path, "winner").unwrap_or_else(|e| fail(&e.to_string()));
        let key = publish_key(&path, "loser").unwrap_or_else(|e| fail(&e.to_string()));
        assert_eq!(key, "winner");
        assert_eq!(
            std::fs::read_to_string(&path).unwrap_or_else(|e| fail(&e.to_string())),
            "winner"
        );
        let names = std::fs::read_dir(dir.path())
            .unwrap_or_else(|e| fail(&e.to_string()))
            .count();
        assert_eq!(names, 1, "the temporary file is removed");
    }

    #[tokio::test]
    async fn a_second_lock_on_one_key_file_waits_for_the_first() {
        let dir = tempfile::tempdir().unwrap_or_else(|e| fail(&e.to_string()));
        let file = dir.path().join("vault.key");
        let held = KeyFileLock::acquire(&file)
            .await
            .unwrap_or_else(|e| fail(&e.to_string()));
        let second = tokio::spawn({
            let file = file.clone();
            async move { KeyFileLock::acquire(&file).await.map(drop) }
        });
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
        assert!(!second.is_finished(), "a second handle must wait");
        drop(held);
        let got = tokio::time::timeout(std::time::Duration::from_secs(5), second).await;
        assert!(matches!(got, Ok(Ok(Ok(())))));
    }

    #[tokio::test]
    async fn a_slot_waits_for_the_lock_holder_and_uses_the_key_it_stored() {
        let dir = tempfile::tempdir().unwrap_or_else(|e| fail(&e.to_string()));
        let (fake, slot) = fake_slot(dir.path(), Mode::Open);
        // Another process making the key holds the lock.
        let held = KeyFileLock::acquire(&dir.path().join("vault.key"))
            .await
            .unwrap_or_else(|e| fail(&e.to_string()));
        let waiting = tokio::spawn(async move {
            slot.read_or_create(|| Err(Error::Vault(String::from("made a second key"))))
                .await
        });
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
        assert!(!waiting.is_finished(), "the lookup must wait for the lock");
        assert!(
            Keychain::Fake(Arc::clone(&fake))
                .set("vault", "winner")
                .await
                .is_ok()
        );
        drop(held);
        let got = tokio::time::timeout(std::time::Duration::from_secs(5), waiting).await;
        assert!(matches!(got, Ok(Ok(Ok(key))) if key == "winner"));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn two_slots_first_seal_on_one_fresh_file_agree_on_one_key() {
        // Two slots racing their first key on one fresh file must agree.
        for _ in 0..20 {
            let dir = tempfile::tempdir().unwrap_or_else(|e| fail(&e.to_string()));
            let keyfile = dir.path().join("vault.key");
            let slot_a = KeySlot::new(String::from("vault"), keyfile.clone(), KeySource::File);
            let slot_b = KeySlot::new(String::from("vault"), keyfile.clone(), KeySource::File);
            let barrier = Arc::new(tokio::sync::Barrier::new(2));
            let ha = tokio::spawn({
                let barrier = Arc::clone(&barrier);
                async move {
                    barrier.wait().await;
                    slot_a.read_or_create(|| Ok(String::from("a"))).await
                }
            });
            let hb = tokio::spawn({
                let barrier = Arc::clone(&barrier);
                async move {
                    barrier.wait().await;
                    slot_b.read_or_create(|| Ok(String::from("b"))).await
                }
            });
            let a = ha
                .await
                .unwrap_or_else(|e| fail(&e.to_string()))
                .unwrap_or_else(|e| fail(&e.to_string()));
            let b = hb
                .await
                .unwrap_or_else(|e| fail(&e.to_string()))
                .unwrap_or_else(|e| fail(&e.to_string()));
            assert_eq!(a, b);
            let slot_c = KeySlot::new(String::from("vault"), keyfile, KeySource::File);
            assert_eq!(
                slot_c
                    .read()
                    .await
                    .unwrap_or_else(|e| fail(&e.to_string()))
                    .as_deref(),
                Some(a.as_str())
            );
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn two_slots_first_seal_on_one_fresh_keychain_agree_on_one_key() {
        // The keychain's set is last-writer-wins, so only the lock makes them agree.
        for _ in 0..20 {
            let dir = tempfile::tempdir().unwrap_or_else(|e| fail(&e.to_string()));
            let fake = Arc::new(FakeKeychain::new(Mode::Open));
            let slot_a = KeySlot::with_keychain(
                String::from("vault"),
                dir.path().join("vault.key"),
                Keychain::Fake(Arc::clone(&fake)),
            );
            let slot_b = KeySlot::with_keychain(
                String::from("vault"),
                dir.path().join("vault.key"),
                Keychain::Fake(Arc::clone(&fake)),
            );
            let barrier = Arc::new(tokio::sync::Barrier::new(2));
            let ha = tokio::spawn({
                let barrier = Arc::clone(&barrier);
                async move {
                    barrier.wait().await;
                    slot_a.read_or_create(|| Ok(String::from("a"))).await
                }
            });
            let hb = tokio::spawn({
                let barrier = Arc::clone(&barrier);
                async move {
                    barrier.wait().await;
                    slot_b.read_or_create(|| Ok(String::from("b"))).await
                }
            });
            let a = ha
                .await
                .unwrap_or_else(|e| fail(&e.to_string()))
                .unwrap_or_else(|e| fail(&e.to_string()));
            let b = hb
                .await
                .unwrap_or_else(|e| fail(&e.to_string()))
                .unwrap_or_else(|e| fail(&e.to_string()));
            assert_eq!(a, b);
            assert_eq!(fake.peek("vault").as_deref(), Some(a.as_str()));
            assert!(!dir.path().join("vault.key").exists());
        }
    }
}
