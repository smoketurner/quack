//! Signed-in users' identity-provider tokens, kept in `control.db`
//! (`user_tokens`, deleted with the user) and sealed by the [`Vault`] for
//! [`Purpose::UserToken`] with the user id as the subject.

use crate::error::Result;
use crate::ids::UserId;
use crate::llm::oauth::{CachedToken, Plaintext};
use crate::storage::control::{ControlPlane, SealedOwner};
use crate::vault::{Opened, Purpose, Vault};

/// Keeps each signed-in user's token.
#[derive(Debug)]
pub struct UserTokens {
    vault: Vault,
}

impl UserTokens {
    #[must_use]
    pub const fn new(vault: Vault) -> Self {
        Self { vault }
    }

    /// Seal `token` for `user` and keep it, replacing the one before.
    ///
    /// # Errors
    ///
    /// Returns an error when sealing fails or the row cannot be written.
    pub async fn store(
        &self,
        control: &ControlPlane,
        user: &UserId,
        token: &CachedToken,
    ) -> Result<()> {
        let plaintext = serde_json::to_vec(&Plaintext::from(token))?;
        let sealed = self
            .vault
            .seal(Purpose::UserToken, user.as_str(), &plaintext)
            .await?;
        control.put_sealed(SealedOwner::User(user), &sealed).await
    }

    /// The user's token, or `None` when there is none or the key that sealed
    /// it is gone: they sign in again.
    ///
    /// # Errors
    ///
    /// Returns an error when the row does not open (altered, or moved from
    /// another user) or a query fails.
    pub async fn load(&self, control: &ControlPlane, user: &UserId) -> Result<Option<CachedToken>> {
        let Some(sealed) = control.sealed(SealedOwner::User(user)).await? else {
            return Ok(None);
        };
        match self
            .vault
            .open(Purpose::UserToken, user.as_str(), &sealed)
            .await?
        {
            Opened::Plaintext(plaintext) => {
                let plain: Plaintext = serde_json::from_slice(&plaintext)?;
                Ok(Some(CachedToken::from(plain)))
            }
            Opened::KeyGone => {
                tracing::warn!(user = %user, key = %sealed.key_id, "the key that sealed this token is gone; the user signs in again");
                Ok(None)
            }
        }
    }

    /// Forget the user's token.
    ///
    /// # Errors
    ///
    /// Returns an error if the delete fails.
    pub async fn clear(&self, control: &ControlPlane, user: &UserId) -> Result<()> {
        control.delete_sealed(SealedOwner::User(user)).await
    }
}

#[cfg(test)]
mod tests;
