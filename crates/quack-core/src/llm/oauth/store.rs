//! Each OAuth provider's token, kept in `control.db` (`provider_tokens`) and
//! sealed by the vault for `Purpose::ProviderToken`, the provider name as the
//! subject. `control.db` is opened on first use, so a command that never
//! reaches the provider never opens it.

use tokio::sync::OnceCell;

use super::key_slot::{KeyLocation, KeySource};
use super::token::{CachedToken, Plaintext};
use crate::config::{Config, ProviderName};
use crate::error::Result;
use crate::storage::control::{ControlPlane, SealedOwner};
use crate::vault::{Opened, Purpose, Vault};

/// One provider's stored token.
pub(super) struct ProviderTokens {
    provider: ProviderName,
    config: Config,
    control: OnceCell<ControlPlane>,
    vault: Vault,
}

impl std::fmt::Debug for ProviderTokens {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ProviderTokens")
            .field("provider", &self.provider)
            .finish_non_exhaustive()
    }
}

impl ProviderTokens {
    pub(super) fn new(config: &Config, provider: &ProviderName, key_source: KeySource) -> Self {
        Self {
            provider: provider.clone(),
            config: config.clone(),
            control: OnceCell::new(),
            vault: Vault::new(config.data_dir(), key_source),
        }
    }

    async fn control(&self) -> Result<&ControlPlane> {
        self.control
            .get_or_try_init(|| ControlPlane::open(&self.config))
            .await
    }

    fn owner(&self) -> SealedOwner<'_> {
        SealedOwner::Provider(&self.provider)
    }

    /// Where the vault key is.
    ///
    /// # Errors
    ///
    /// As [`Vault::key_location`].
    pub(super) async fn key_location(&self) -> Result<KeyLocation> {
        self.vault.key_location().await
    }

    /// The stored token, or `None` when there is none or the key that sealed
    /// it is gone: a new login is needed.
    ///
    /// # Errors
    ///
    /// Returns an error when the row does not open or a query fails.
    pub(super) async fn load(&self) -> Result<Option<CachedToken>> {
        let control = self.control().await?;
        let Some(sealed) = control.sealed(self.owner()).await? else {
            return Ok(None);
        };
        match self
            .vault
            .open(Purpose::ProviderToken, self.provider.as_str(), &sealed)
            .await?
        {
            Opened::Plaintext(plaintext) => {
                let plain: Plaintext = serde_json::from_slice(&plaintext)?;
                Ok(Some(CachedToken::from(plain)))
            }
            Opened::KeyGone => {
                tracing::warn!(provider = %self.provider, key = %sealed.key_id, "the key that sealed this provider's token is gone; a new login is needed");
                Ok(None)
            }
        }
    }

    /// Seal and keep the token, replacing the one before.
    ///
    /// # Errors
    ///
    /// Returns an error when sealing or the write fails.
    pub(super) async fn store(&self, token: &CachedToken) -> Result<()> {
        let plaintext = serde_json::to_vec(&Plaintext::from(token))?;
        let sealed = self
            .vault
            .seal(Purpose::ProviderToken, self.provider.as_str(), &plaintext)
            .await?;
        self.control()
            .await?
            .put_sealed(self.owner(), &sealed)
            .await
    }

    /// Forget the token. The vault key stays: other tokens use it.
    ///
    /// # Errors
    ///
    /// Returns an error if the delete fails.
    pub(super) async fn clear(&self) -> Result<()> {
        self.control().await?.delete_sealed(self.owner()).await
    }
}
