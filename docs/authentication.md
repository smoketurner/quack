# Authentication

This guide shows how people and programs sign in to `quack serve`, and where quack keeps the
secrets involved. [`providers.md`](providers.md) shows how quack signs in to a model
provider. Design doc section 12 explains the mechanisms behind each step.

## Terms

| Term | Meaning |
|---|---|
| Identity provider, issuer | The organization's sign-in service, such as Microsoft Entra ID, Okta, or Auth0. It issues tokens. |
| OpenID Connect (OIDC) | The standard sign-in protocol that identity providers speak. |
| Access token | A short-lived credential that lets its holder call an API (application programming interface). |
| Refresh token | A longer-lived credential that gets new access tokens without a new sign-in. |
| Bearer | A token sent in the HTTP `Authorization: Bearer` header. Whoever holds it can use it. |
| JWT (JSON Web Token) | A signed token whose claims anyone can read, such as `aud` (audience), the service the token is for. |
| MCP (Model Context Protocol) | The protocol AI clients such as Claude Code use to call tools. quack serves it at `/mcp/v1/{workspace}`. |
| `private_key_jwt` | A way for quack to prove its identity to an issuer with a private key instead of a shared secret. |
| Dynamic client registration | quack creates its own client record at the issuer over HTTP, instead of a person creating it in the issuer's console. |

## Ways to sign in

`quack serve` accepts five kinds of caller:

| Mode | What the caller presents | How to turn it on |
|---|---|---|
| Local | nothing; loopback addresses only | `quack serve --local` |
| Password | a session cookie, or a `qs_…` bearer | `quack user add` |
| API token | a `qk_…` bearer for one workspace | `quack token create` |
| Identity-provider sign-in | a session cookie | `[server.oidc]` |
| Identity-provider access token | a JWT bearer | `[server.oidc].audience` |

Only `quack serve` checks who is calling. The terminal, print mode, `quack ingest`, and
`quack mcp` on stdio run as the operating-system user who starts them. That user can read the
whole data directory, so quack keeps it at mode `0700`: every command that opens it
removes any group or other access, with a warning. `quack serve --local` turns
authentication off for one person's browser and listens on loopback addresses only.

A caller's role in a workspace decides what it may do there: `viewer`, `member`, or `owner`.
The admin flag lets a user manage users, workspaces, and membership, but not read content.
quack audits every request that touches a workspace, including every refused one.

## Passwords and API tokens

An administrator creates password users and API tokens from the command line:

```bash
quack user add alice            # prompts for the password
quack user add admin --admin
quack token create -w sales --user alice --name reporting --scopes read,write --expires 90
quack token list -w sales
quack token revoke -w sales HASH          # a prefix of the hash is enough
```

A browser session ends at logout, 12 hours after login (`[server].session_max_age_hours`),
or after 120 idle minutes (`[server].session_idle_minutes`). quack holds sessions in memory,
so a restart signs everyone out. Behind a TLS-terminating proxy on the same host, set
`[server].secure_cookies = "always"` so the browser sends the cookie over HTTPS only.

quack shows an API token once. The token acts as its user in one workspace, limited to its
scopes: `read`, `write`, or `admin`. Workspace owners can also create and revoke tokens on
the workspace's Settings page.

## Sign-in through the organization's identity provider

With `[server.oidc]` set, the login page offers "Sign in with *issuer host*" beside the
password form. Password users keep working.

```toml
[server.oidc]
issuer_url = "https://login.example.com"
client_id = "quack"
client_secret_env = "QUACK_OIDC_SECRET"
redirect_uri = "https://quack.example.com/auth/oidc/callback"   # register it with the issuer
# scopes = ["openid", "profile", "email", "offline_access"]     # the default
# subject_claim = "sub"                                         # "oid" for Entra ID
# client_auth = "client_secret_post"                            # or "client_secret_basic", "private_key_jwt"
```

A person's first sign-in creates a user with no workspaces. That user sees nothing until an
owner adds them with `quack member add` or on the workspace's Settings page. quack stores
the person's tokens and renews them. When the issuer revokes the grant or disables the
account, quack ends all of that person's sessions. Without the `offline_access` scope, the
issuer sends no refresh token, and a sign-in lasts only as long as quack's session limits.

### Accepting the identity provider's access tokens

Set `audience`, and the REST API and MCP also accept access tokens the issuer mints for
quack. A script or an MCP client then needs no quack API token.

```toml
[server.oidc]
# ...as above
audience = "api://quack"
```

quack then publishes where to get such a token (OAuth protected-resource metadata, RFC
9728), and every `401` response points to it. An MCP client pointed at
`https://quack.example.com/mcp/v1/sales` reads it, signs the user in with the issuer, and
retries. The token maps to the same quack user as that person's browser sign-in.

### Proving quack's identity with a key instead of a secret

With `client_auth = "private_key_jwt"`, quack proves its identity with a private key it
creates and keeps. The issuer holds only the public key, so no shared secret sits in quack's
environment. Remove `client_secret_env`, then give the issuer the public key:

```bash
quack auth jwks                               # the public key set to register for [server.oidc]
quack auth jwks --rotate                      # adds a new key beside the old one; register both
quack auth jwks --rotate --activate           # switches to the new key; register it alone
```

Restart `quack serve` after `--activate`. Until the restart, it signs with the old key, and
the issuer refuses it. Microsoft Entra ID accepts only certificates, not this key, so use a
client secret there.

### Letting quack register itself

When the issuer supports dynamic client registration, leave `client_id` out and let quack
create the client record:

```bash
quack auth register                         # you sign in, and the client is registered as yours
quack auth register --device-code           # the same, on a host with no browser
quack auth register --token-env IDP_TOKEN   # uses an initial access token instead (Okta)
quack auth register --replace               # registers a new client and key, then deletes the old client
quack auth unregister                       # deletes the client at the issuer
```

quack registers the callback address, the grants, the scopes, and its public key, then prints
the new `client_id`. Every section without a `client_id` uses it from then on. When a model
provider acts on behalf of each person through the same issuer, one registration serves both
([`providers.md`](providers.md#recipe-on-behalf-of-each-person-with-vouch)). For a client quack
registered, `quack auth jwks --rotate` updates the issuer directly.

Four issuers, four answers: Vouch accepts registration. Auth0 accepts it once the tenant turns
it on. Okta accepts it with `--token-env`. Entra ID does not offer it.

## Identity-provider settings

These settings come from each vendor's documentation as of September 2026. quack has not yet
run against a live tenant of any of them. Settings for acting on behalf of each person are in
[`providers.md`](providers.md#identity-provider-notes).

**Microsoft Entra ID.** Set `issuer_url` to
`https://login.microsoftonline.com/{tenant_id}/v2.0`, register the callback as a Web redirect
URI, and use a client secret. Set `subject_claim = "oid"`, because Entra gives one person a
different `sub` in each application. To accept access tokens, expose an API on the app
registration and set `requestedAccessTokenVersion` to `2` in its manifest. Then set
`audience` to the app's client ID, a GUID (globally unique identifier), since that is a
version 2.0 token's `aud`.

**Okta.** Use a custom authorization server, such as `https://{domain}/oauth2/default`.
Turn on the Refresh Token grant. Set `audience` to the server's Audience (`api://default`)
and `client_auth = "client_secret_basic"`, Okta's default.

**Auth0.** Set `issuer_url` to `https://{tenant}.auth0.com/` or your custom domain. Turn on
"Allow Offline Access" on the API to get refresh tokens. quack's own sign-in cannot yet ask
Auth0 for a JWT access token, but an MCP client that gets one can present it.

**Vouch.** Set up sign-in and on-behalf-of together, as
[`providers.md`](providers.md#recipe-on-behalf-of-each-person-with-vouch) shows.

## Where quack keeps secrets

quack encrypts every stored token and key with one vault key, which it keeps outside
`control.db`:

| Secret | Where |
|---|---|
| The vault key | the OS keychain (entry `quack` / `vault`), else `<data_dir>/vault.key` with mode `0600` |
| Signed-in users' tokens, providers' OAuth tokens, client keys, registration tokens | `control.db`, encrypted with the vault key |
| On-behalf-of tokens, browser sessions | memory only |
| Passwords, API tokens | `control.db`, as argon2id and SHA-256 hashes |
| Client secrets, API keys | the environment variables the configuration names |
| AWS credentials | wherever the AWS SDK finds them |

Three consequences for operators:

- Linux keeps its keychain in memory, so after a reboot, users sign in again and each OAuth
  provider needs `quack auth login` again. Docker's default seccomp profile (its system-call
  filter) blocks that keychain, so in a container quack writes `vault.key` to the data volume.
- A locked keychain stops quack with an error; quack does not fall back to `vault.key`.
  Unlock the keychain and retry.
- A backup of `control.db` without the vault key cannot decrypt its tokens. Restore both, or
  expect everyone to sign in again.

## Troubleshooting

`quack doctor` checks `[server.oidc]` against the issuer. `quack config` shows every setting
and where its value came from.

- **"access token refused: InvalidAudience"**: the token's `aud` differs from
  `[server.oidc].audience`. Decode the token and compare the two.
- **"the discovery document for X names issuer …"**: `issuer_url` must match the issuer's own
  `issuer` value exactly, apart from a trailing slash.
- **"the sign-in could not be matched to this browser"**: the sign-in started in another
  browser, or at a host name other than the one in `redirect_uri`.
- **"the redirect names issuer …, not this one"**: the redirect came from a different server.
  Check `issuer_url`, proxies, and mixed tenants.
- **`invalid_client` with `private_key_jwt`**: the issuer does not hold quack's current key.
  Register the output of `quack auth jwks`, and restart `quack serve` after `--activate`.
