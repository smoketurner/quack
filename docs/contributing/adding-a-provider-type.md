# Adding a provider type

A provider type is a way of reaching a model API. Bedrock (`config/bedrock.rs` for the
settings, `llm/bedrock.rs` for the client) is the worked pattern for a type with its own
authentication and endpoint rules; `openai` for a plain HTTP API.

1. **`crates/quack-core/src/config.rs`**: the variant in `ProviderType` and its text; its
   default `base_url` and default `AuthMode` (`default_auth`); its default request limit
   (`default_request_limit`); and any setting only this type takes, as an optional field on
   `RawProviderConfig` validated in `TryFrom<RawProviderConfig> for ProviderConfig` (a
   setting another type cannot use is a config error, with the fix in the message). A type
   whose settings are a group of their own gets a `config/<type>.rs` like `config/bedrock.rs`.
2. **`crates/quack-core/src/config/inspect.rs`**: the new key in `PROVIDER_KEYS` (the
   unknown-key check reads it) and in `providers()`, so `quack config` prints it. The test
   `the_key_list_matches_the_config_structs` fails until both agree.
3. **`crates/quack-core/src/llm/mod.rs`**: build the chat client in `ChatClient::new` (every
   request goes through `LimitedHttp`, which holds the provider's permit and applies its
   headers; never a bare reqwest client) and the embedding client where the embedders are
   built. `ProviderModels` lists the provider's models for `quack doctor`.
4. **`crates/quack-core/src/llm/sampling.rs`**: how `temperature`, `effort`, and
   `max_tokens` go out on this API (`Wire`), and `check_tool_calls` if the API reports tool
   support.
5. **`crates/quack-core/src/doctor.rs`**: the probe for the type in `check_model` (a
   credential present, the model listed, the fix line when it is not).
6. **`docs/providers.md`**: the type in "Choosing a provider type", a recipe with a config
   snippet, and the credential it takes; `docs/design-doc.md` section 10.1.

## What bites

- `clippy::absolute_paths`: `use` the type; `crate::a::b::Type` at a call site is denied.
- No `unwrap`, `expect`, indexing, or slicing outside tests.
- Every outbound HTTP client takes its proxy from `proxy::Proxies` (`reqwest::Client::builder`
  is disallowed by `.clippy.toml`), and its TLS from rustls on aws-lc-rs: no `openssl`,
  `native-tls`, or `ring` feature on any new dependency (`deny.toml` bans them).
- A new dependency is pinned in the root `[workspace.dependencies]` with
  `default-features = false`.
- `Egress::permit` is checked when the client is built and `ProviderGates::permit` on every
  request, so a workspace's `allowed_providers` holds; a request outside any scope is
  `Error::ModelRequestUnscoped`. Tests that make a request enter a scope first.
- A model request carries workspace content, so nothing about a request may be logged
  beyond the provider, the model, and the status.
