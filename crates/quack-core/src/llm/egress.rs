//! Which model providers the current work may send to.
//!
//! A workspace's owner can restrict it to some of the configured providers
//! (`workspaces.allowed_providers`). The restriction is carried by a Tokio
//! task-local, like [`crate::priority::Priority`] and
//! [`super::acting::Acting`], so the one check, [`Egress::permit`], finds it
//! wherever a model request is made: when a model's client is built, and
//! again as each request passes its provider's gates
//! ([`super::limit::ProviderGates`]), which nothing quack sends goes around.
//!
//! Work enters a scope where it learns its workspace: `quack serve` scopes
//! an empty slot around each request and fills it when the request's
//! workspace is resolved; the CLI does the same around a command; the job
//! queue carries the submitter's into the job. A model request made with no
//! scope is an error, never an allow, so work that forgets to enter one
//! fails instead of sending.

use std::future::Future;

use super::slot::Slot;
use crate::config::{ProviderName, ProviderType};
use crate::error::{Error, Result};
use crate::storage::control::AllowedProviders;

/// Why a workspace's provider allow-list refused a model request.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum Refusal {
    /// The list does not name the provider.
    #[error("provider '{provider}' is not allowed in this workspace, which allows {allowed}")]
    Provider {
        provider: String,
        allowed: AllowedProviders,
    },
    /// An Ollama model that Ollama serves from its own hosts, asked for in
    /// a workspace restricted to some providers.
    #[error(
        "model '{model}' of provider '{provider}' runs in Ollama's cloud, not on the Ollama \
         server itself, and this workspace allows {allowed}; configure a model without the \
         `cloud` tag"
    )]
    CloudModel {
        provider: String,
        model: String,
        allowed: AllowedProviders,
    },
}

/// Where the current work's model requests may go.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Egress {
    /// Work on a workspace: the providers its owner allows.
    Workspace(AllowedProviders),
    /// Work that belongs to no workspace and carries none's content, such
    /// as `quack doctor` probing each configured provider.
    NoWorkspace,
}

tokio::task_local! {
    static SLOT: Slot<Egress>;
}

impl Egress {
    /// Run `work` with an empty slot that [`Egress::enter`] fills once the
    /// workspace is known: a server request, a CLI command.
    pub async fn request<F: Future>(work: F) -> F::Output {
        Slot::request(&SLOT, work).await
    }

    /// Run `work` under `egress`; under `None` its model requests fail.
    pub async fn scope<F: Future>(egress: Option<Self>, work: F) -> F::Output {
        Slot::scope(&SLOT, egress, work).await
    }

    /// Make this the scope of the current request. The first caller wins;
    /// outside [`Egress::request`] it does nothing.
    pub fn enter(self) {
        Slot::enter(&SLOT, self);
    }

    /// The current work's scope, if it entered one.
    #[must_use]
    pub fn current() -> Option<Self> {
        Slot::current(&SLOT)
    }

    /// Whether the current work may send `model` (when the request names
    /// one) to `provider`. Under a restricted list, an Ollama model with a
    /// `cloud` tag is refused as well: Ollama serves it from its own hosts
    /// through the local API.
    ///
    /// # Errors
    ///
    /// Returns [`Error::ProviderRefused`] when the workspace's list refuses
    /// the request, and [`Error::ModelRequestUnscoped`] when the work
    /// entered no scope.
    pub(crate) fn permit(
        provider: &ProviderName,
        kind: ProviderType,
        model: Option<&str>,
    ) -> Result<()> {
        let Some(egress) = Self::current() else {
            return Err(Error::ModelRequestUnscoped {
                provider: provider.to_string(),
            });
        };
        let Self::Workspace(allowed @ AllowedProviders::Only(_)) = egress else {
            return Ok(());
        };
        if !allowed.permits(provider.as_str()) {
            return Err(Refusal::Provider {
                provider: provider.to_string(),
                allowed,
            }
            .into());
        }
        match model {
            Some(model)
                if kind == ProviderType::Ollama
                    && (model.ends_with("-cloud") || model.ends_with(":cloud")) =>
            {
                Err(Refusal::CloudModel {
                    provider: provider.to_string(),
                    model: model.to_owned(),
                    allowed,
                }
                .into())
            }
            _ => Ok(()),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use super::*;

    #[expect(clippy::panic, reason = "test failure path")]
    fn name(text: &str) -> ProviderName {
        text.parse()
            .unwrap_or_else(|e: Error| panic!("provider name: {e}"))
    }

    fn only(names: &[&str]) -> Egress {
        Egress::Workspace(AllowedProviders::Only(
            names.iter().map(|n| (*n).to_owned()).collect(),
        ))
    }

    async fn permit_under(
        egress: Option<Egress>,
        provider: &str,
        kind: ProviderType,
        model: Option<&str>,
    ) -> Result<()> {
        Egress::scope(egress, async {
            Egress::permit(&name(provider), kind, model)
        })
        .await
    }

    #[tokio::test]
    async fn an_allowed_provider_and_an_unrestricted_workspace_pass() {
        for egress in [
            only(&["ollama"]),
            Egress::Workspace(AllowedProviders::All),
            Egress::NoWorkspace,
        ] {
            let permitted =
                permit_under(Some(egress), "ollama", ProviderType::Ollama, Some("m:8b")).await;
            assert!(permitted.is_ok(), "{permitted:?}");
        }
        // A request that names no model (a model listing) passes on its provider.
        let listing = permit_under(
            Some(only(&["ollama"])),
            "ollama",
            ProviderType::Ollama,
            None,
        );
        assert!(listing.await.is_ok());
    }

    #[tokio::test]
    async fn a_provider_off_the_list_is_refused_with_the_list() {
        let refused = permit_under(
            Some(only(&["ollama", "local"])),
            "hosted",
            ProviderType::Openai,
            Some("m"),
        )
        .await;
        assert!(
            matches!(
                &refused,
                Err(Error::ProviderRefused(Refusal::Provider { provider, allowed }))
                    if provider == "hosted"
                        && allowed.names()
                            == Some(&BTreeSet::from(["local".to_owned(), "ollama".to_owned()]))
            ),
            "{refused:?}"
        );
        assert_eq!(
            refused.err().map(|e| e.to_string()).unwrap_or_default(),
            "provider 'hosted' is not allowed in this workspace, which allows only: local, ollama"
        );
        // A list that names nothing allows nothing.
        let none = permit_under(Some(only(&[])), "ollama", ProviderType::Ollama, None).await;
        assert_eq!(
            none.err().map(|e| e.to_string()).unwrap_or_default(),
            "provider 'ollama' is not allowed in this workspace, which allows no provider"
        );
    }

    #[tokio::test]
    async fn a_cloud_model_is_refused_only_on_ollama_under_a_restricted_list() {
        for model in ["big:120b-cloud", "big:cloud"] {
            let refused = permit_under(
                Some(only(&["ollama"])),
                "ollama",
                ProviderType::Ollama,
                Some(model),
            )
            .await;
            assert!(
                matches!(
                    &refused,
                    Err(Error::ProviderRefused(Refusal::CloudModel { provider, model: named, .. }))
                        if provider == "ollama" && named == model
                ),
                "{refused:?}"
            );
            assert!(
                refused
                    .err()
                    .is_some_and(|e| e.to_string().contains(&format!("model '{model}'")))
            );
            // Every provider allowed: the workspace is not kept local.
            let unrestricted = Egress::Workspace(AllowedProviders::All);
            let open = permit_under(
                Some(unrestricted),
                "ollama",
                ProviderType::Ollama,
                Some(model),
            );
            assert!(open.await.is_ok());
            // Another provider type's model id is only a name.
            let other = permit_under(
                Some(only(&["ollama"])),
                "ollama",
                ProviderType::Openai,
                Some(model),
            );
            assert!(other.await.is_ok());
        }
        let local = permit_under(
            Some(only(&["ollama"])),
            "ollama",
            ProviderType::Ollama,
            Some("cloud-atlas:8b"),
        );
        assert!(local.await.is_ok());
    }

    #[tokio::test]
    async fn a_request_with_no_scope_is_an_error() {
        let bare = Egress::permit(&name("ollama"), ProviderType::Ollama, None);
        assert!(
            matches!(&bare, Err(Error::ModelRequestUnscoped { provider }) if provider == "ollama"),
            "{bare:?}"
        );
        let unfilled =
            Egress::request(async { Egress::permit(&name("ollama"), ProviderType::Ollama, None) })
                .await;
        assert!(matches!(unfilled, Err(Error::ModelRequestUnscoped { .. })));
        let carried_nothing = permit_under(None, "ollama", ProviderType::Ollama, None).await;
        assert!(matches!(
            carried_nothing,
            Err(Error::ModelRequestUnscoped { .. })
        ));
    }

    #[tokio::test]
    async fn a_request_takes_the_first_scope_entered_and_nothing_leaks_out() {
        let seen = Egress::request(async {
            assert!(Egress::current().is_none());
            only(&["ollama"]).enter();
            Egress::NoWorkspace.enter();
            Egress::current()
        })
        .await;
        assert_eq!(seen, Some(only(&["ollama"])));
        assert!(Egress::current().is_none());
        // Entering outside a request scope does nothing.
        Egress::NoWorkspace.enter();
        assert!(Egress::current().is_none());
    }
}
