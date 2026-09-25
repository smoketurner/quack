//! The OS keychain entry that holds a key: the vault's (`key_slot.rs`).
//!
//! One store per platform: the macOS Keychain, the Linux kernel keyring
//! (`keyutils`, always present, in-memory only, so a reboot needs a new
//! login), and the Windows Credential Manager. Every call runs on the
//! blocking pool because the stores talk to the OS synchronously.

use std::sync::{Arc, OnceLock};

use keyring_core::{CredentialStore, Entry};

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

/// Why the keychain could not be used. The two cases call for opposite
/// responses, so they are kept apart (#226).
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum KeychainError {
    /// This host has no keychain to talk to: no store could be installed,
    /// or an entry cannot even be addressed (Docker's seccomp profile
    /// blocks the Linux keyring syscalls). The key file stands in for it.
    Unavailable(String),
    /// A keychain exists but refused or failed the operation: locked,
    /// access denied, or an error of its own. It may hold the key, so the
    /// key file must not stand in for it: a key made there would be
    /// shadowed by the keychain's once it answers again.
    Refused(String),
}

impl std::fmt::Display for KeychainError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Unavailable(e) => write!(f, "no usable keychain: {e}"),
            Self::Refused(e) => write!(f, "the keychain refused: {e}"),
        }
    }
}

/// Sort a keychain failure. Only a missing store is `Unavailable`: an
/// error from an entry the store did build is the store refusing.
fn classify(e: &keyring_core::Error, stage: Stage) -> KeychainError {
    let unavailable = match stage {
        Stage::Store | Stage::Entry => true,
        Stage::Operation => matches!(
            e,
            keyring_core::Error::NoDefaultStore | keyring_core::Error::NotSupportedByStore(_)
        ),
    };
    if unavailable {
        KeychainError::Unavailable(e.to_string())
    } else {
        KeychainError::Refused(e.to_string())
    }
}

/// How far a keychain call got before it failed.
#[derive(Debug, Clone, Copy)]
enum Stage {
    /// Installing the platform store.
    Store,
    /// Addressing the entry; on Linux this already talks to the kernel.
    Entry,
    /// Reading or writing the entry.
    Operation,
}

/// Where keys are kept: the OS keychain, or (in tests) a fake one.
#[derive(Debug, Clone)]
pub(super) enum Keychain {
    Os,
    #[cfg(test)]
    Fake(std::sync::Arc<fake::FakeKeychain>),
}

impl Keychain {
    /// The key stored under `account`, or `None` when there is no entry.
    ///
    /// # Errors
    ///
    /// `Unavailable` when this host has no keychain, `Refused` when the
    /// keychain would not answer.
    pub(super) async fn get(&self, account: &str) -> Result<Option<String>, KeychainError> {
        match self {
            Self::Os => {
                KeychainEntry::new(account.to_owned())
                    .run(KeychainOp::Read)
                    .await
            }
            #[cfg(test)]
            Self::Fake(fake) => fake.get(account),
        }
    }

    /// Store `key` under `account`, replacing any previous one.
    ///
    /// # Errors
    ///
    /// As [`Self::get`].
    pub(super) async fn set(&self, account: &str, key: &str) -> Result<(), KeychainError> {
        match self {
            Self::Os => KeychainEntry::new(account.to_owned())
                .run(KeychainOp::Write(key.to_owned()))
                .await
                .map(drop),
            #[cfg(test)]
            Self::Fake(fake) => fake.set(account, key),
        }
    }
}

/// A keychain entry under the `quack` service, named by its account.
struct KeychainEntry(String);

/// What is done to an entry.
enum KeychainOp {
    Read,
    Write(String),
}

impl KeychainOp {
    const fn verb(&self) -> &'static str {
        match self {
            Self::Read => "read",
            Self::Write(_) => "write",
        }
    }
}

impl KeychainEntry {
    const fn new(account: String) -> Self {
        Self(account)
    }

    /// Do `op` on the blocking pool, since the stores talk to the OS
    /// synchronously. A read of a missing entry is `None`.
    async fn run(&self, op: KeychainOp) -> Result<Option<String>, KeychainError> {
        let account = self.0.clone();
        tokio::task::spawn_blocking(move || {
            let verb = op.verb();
            let outcome = ensure_store()
                .map_err(|e| classify(&e, Stage::Store))
                .and_then(|()| {
                    Entry::new(SERVICE, &account).map_err(|e| classify(&e, Stage::Entry))
                })
                .and_then(|entry| {
                    match op {
                        KeychainOp::Read => entry.get_password().map(Some),
                        KeychainOp::Write(key) => entry.set_password(&key).map(|()| None),
                    }
                    .or_else(|e| match e {
                        keyring_core::Error::NoEntry => Ok(None),
                        e => Err(classify(&e, Stage::Operation)),
                    })
                });
            outcome.map_err(|e| match e {
                KeychainError::Unavailable(m) => {
                    KeychainError::Unavailable(format!("keychain {verb} of '{account}': {m}"))
                }
                KeychainError::Refused(m) => {
                    KeychainError::Refused(format!("keychain {verb} of '{account}': {m}"))
                }
            })
        })
        .await
        .map_err(|e| KeychainError::Refused(format!("keychain task failed: {e}")))?
    }
}

/// An in-memory keychain whose availability a test controls.
#[cfg(test)]
pub(super) mod fake {
    use std::collections::HashMap;
    use std::sync::Mutex;

    use super::KeychainError;

    /// How the fake keychain answers.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub(in crate::llm::oauth) enum Mode {
        /// Reads and writes work.
        Open,
        /// The host has no keychain at all.
        Absent,
        /// The keychain is there but locked: every call is refused.
        Locked,
    }

    #[derive(Debug)]
    pub(in crate::llm::oauth) struct FakeKeychain {
        state: Mutex<(Mode, HashMap<String, String>)>,
    }

    impl FakeKeychain {
        pub(in crate::llm::oauth) fn new(mode: Mode) -> Self {
            Self {
                state: Mutex::new((mode, HashMap::new())),
            }
        }

        pub(in crate::llm::oauth) fn set_mode(&self, mode: Mode) {
            if let Ok(mut state) = self.state.lock() {
                state.0 = mode;
            }
        }

        /// The stored entry, whatever the mode (what the OS holds).
        pub(in crate::llm::oauth) fn peek(&self, account: &str) -> Option<String> {
            self.state
                .lock()
                .ok()
                .and_then(|state| state.1.get(account).cloned())
        }

        fn gate(mode: Mode) -> Result<(), KeychainError> {
            match mode {
                Mode::Open => Ok(()),
                Mode::Absent => Err(KeychainError::Unavailable(String::from("no store"))),
                Mode::Locked => Err(KeychainError::Refused(String::from("locked"))),
            }
        }

        pub(super) fn get(&self, account: &str) -> Result<Option<String>, KeychainError> {
            let state = self
                .state
                .lock()
                .map_err(|e| KeychainError::Refused(e.to_string()))?;
            Self::gate(state.0)?;
            Ok(state.1.get(account).cloned())
        }

        pub(super) fn set(&self, account: &str, key: &str) -> Result<(), KeychainError> {
            let mut state = self
                .state
                .lock()
                .map_err(|e| KeychainError::Refused(e.to_string()))?;
            Self::gate(state.0)?;
            state.1.insert(account.to_owned(), key.to_owned());
            Ok(())
        }
    }
}
