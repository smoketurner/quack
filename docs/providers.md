# Model providers

This guide shows how to connect quack to a model provider and how quack proves its identity
to it. It ends with recipes for Amazon Bedrock, a LiteLLM gateway, and Azure API Management in
front of LiteLLM. [`authentication.md`](authentication.md) shows how people sign in to quack.
Design doc section 10 explains the mechanisms.

## Terms

| Term | Meaning |
|---|---|
| Provider | One `[providers.NAME]` entry: an endpoint that serves chat or embedding models. |
| Gateway | A proxy in front of model endpoints, such as LiteLLM or Azure API Management (APIM). quack treats it as one provider. |
| Identity provider, issuer | The organization's sign-in service, such as Microsoft Entra ID, Okta, or Auth0. It issues tokens. |
| OAuth 2.0 | The standard protocol for issuing access tokens. |
| Access token | A short-lived credential that lets its holder call an API (application programming interface). |
| Refresh token | A longer-lived credential that gets new access tokens without a new sign-in. |
| Bearer | A token sent in the HTTP `Authorization: Bearer` header. Whoever holds it can use it. |
| On behalf of (OBO) | quack exchanges a signed-in person's token for a token to the provider, so the provider sees the person rather than quack. |
| `private_key_jwt` | A way for quack to prove its identity to an issuer with a private key instead of a shared secret. |
| Dynamic client registration | quack creates its own client record at the issuer over HTTP, instead of a person creating it in the issuer's console. |
| AWS SDK | The Amazon Web Services software development kit. quack uses it to find AWS credentials and sign Bedrock requests. |
| SigV4 | The AWS request-signing scheme. |

## Choosing a provider type

Each provider entry names one of five types. A model reference is `PROVIDER/MODEL`, such as
`bedrock/us.anthropic.claude-opus-5-5`.

| `type` | Serves | Chat | Embeddings |
|---|---|---|---|
| `ollama` | a local or remote Ollama server | yes | yes |
| `openai` | OpenAI, or any OpenAI-compatible endpoint at `base_url`, such as LiteLLM or vLLM | yes | yes |
| `anthropic` | the Anthropic API | yes | no |
| `bedrock` | Amazon Bedrock's `bedrock-runtime` endpoint | yes | yes |
| `bedrock-mantle` | Amazon Bedrock's `bedrock-mantle` endpoint | yes | no |

Three settings apply to every type:

- `base_url` sets the endpoint. Each type has a default except `openai`, whose default is
  OpenAI itself.
- `max_concurrent_requests` caps the requests in flight to each model. The default is 1 for
  Ollama and 8 for the others. Raise it to match a gateway's capacity.
- `headers` adds fixed HTTP headers to every model request, for a gateway that routes or
  bills on one. quack refuses `Authorization` and `x-api-key`, since the credential comes
  from `auth`. `quack config` shows header names, never values. Bedrock's `api = "converse"`
  takes no headers, because the AWS SDK sends those requests.
- `max_retries` (3) and `retry_backoff_ms` (500) say how a failed request is sent again: a
  429, a 5xx, or a connection that dropped before the response head is retried after the
  wait, which doubles each time with up to 25% jitter and never exceeds 60 seconds; a
  `Retry-After` header sets the wait when the provider sends one. A stream that fails after
  its head is not retried, since part of it was delivered. `max_retries = 0` sends once.
  Bedrock's `converse` API hands the same numbers to the AWS SDK's standard retry mode.
  Each retry counts in `quack_provider_retries_total` on `/metrics`.

`type = "openai"` also takes `api`: `"responses"` (the default for OpenAI itself) or
`"chat-completions"` (the default with a `base_url`, since many compatible servers offer
nothing else).

## Restricting a workspace to some providers

A workspace's owner can limit which configured providers receive its content: the
**Allowed providers** setting on the workspace's settings page, or `allowed_providers` in
`PATCH /api/v1/workspaces/{id}`. An empty list allows every provider. The list names
`[providers.NAME]` entries, so it applies to the chat model, the embedding model, and the
rerank model alike.

quack enforces the list on every model request from every interface: questions, search,
uploads, imports, graph extraction, the ontology's document pass, embeddings refresh, MCP, the
command line, and the terminal. It checks twice, with one function
(`quack_core::llm::egress::Egress::permit`):

1. When it builds a model's client, so the work is refused before it starts.
2. When each request passes the provider's concurrency gate
   (`llm::limit::ProviderGates::permit`). Every HTTP request quack sends to a provider goes
   through that gate, Bedrock's AWS SDK requests included, so no request goes around it.

A refused request is never sent. The error names the provider and the list:

```
provider 'hosted' is not allowed in this workspace, which allows only: local
```

The server answers `403` and writes a `denied` row to the audit log for the action that was
refused. An MCP tool returns the same text as a tool error. The command line prints it and
exits 1.

The models are set once for the whole installation, so restrict a workspace only to providers
that serve every model it needs. If `[embedding].model` is on a provider the list leaves out,
that workspace refuses uploads, search, and questions until the list or the configuration
changes.

Ollama serves some models from its own hosts through the local API; their ids carry a `cloud`
tag (`NAME:cloud` or `NAME:SIZE-cloud`). In a workspace with a restricted list, quack refuses
those models on a `type = "ollama"` provider, so a list of local providers keeps content on
the machine. A workspace that allows every provider is not affected.

`quack doctor` and the model listings carry no workspace content. `quack doctor` probes every
configured provider whatever any workspace allows.

## Temperature and reasoning effort

quack sends `temperature` only through Ollama's own API (`type = "ollama"`). Current Claude
and OpenAI reasoning models reject it with a 400, and every API accepts a request without it.

`[analysis].effort` and `background_effort` go out as the field each API takes:
`output_config.effort` for Claude, `reasoning.effort` on Responses, `reasoning_effort` on Chat
Completions, `think` on Ollama. quack knows which levels Claude, OpenAI's reasoning models, and
gpt-oss take, and refuses any other level before sending a request. A model it does not
recognize, such as a gateway alias or an open-weight model on vLLM, gets the effort on Chat
Completions and Responses, and the server decides whether it accepts that level.
Elsewhere, quack sends no effort and logs a warning; `quack doctor` shows the same warning.

`temperature`, `effort`, and `background_effort` can be set on a provider for all its models,
or under `models."ID"` for one model. quack takes each key from the model first, then the
provider, then `[analysis]`:

```toml
[providers.gateway]
type = "openai"
base_url = "https://llm.example.com/v1"
effort = "medium"                        # every model here, over [analysis].effort

[providers.gateway.models."corp-reasoner-pro"]
effort = "high"                          # this model, over the provider's
background_effort = "low"                # graph extraction and ontology induction

[providers.ollama.models."qwen3:32b"]
temperature = false                      # use the model's own sampling defaults
```

`temperature = true` sends quack's temperature (0.1 for chat turns, 0.0 for extraction). If only
some of a gateway's models reason, set `effort` on those models rather than in `[analysis]`,
because a model that does not reason rejects the field. `quack config` lists every key, and
`quack doctor` shows what the chat model is sent. It checks `background_effort` too when that differs from
`effort`. A level the model refuses fails graph extraction and the ontology's document pass;
a chat turn still answers, without model reranking and history summaries, and logs a warning.

## Credentials

Each provider entry picks one `auth` mode:

| `auth` | What quack sends | Used for |
|---|---|---|
| `none` (default) | nothing | a local Ollama server |
| `api-key` | the key in the environment variable named by `api_key_env`, as `Authorization: Bearer` (`x-api-key` for `type = "anthropic"`) | OpenAI, Anthropic, LiteLLM virtual keys |
| `oauth` | an access token from the `[providers.NAME.oauth]` issuer, as `Authorization: Bearer` for every type | gateways behind Entra ID, Okta, or Auth0 |
| `aws` (default for Bedrock) | a SigV4 signature from the AWS SDK's credentials | `bedrock`, `bedrock-mantle` |

`type = "openai"` needs `api-key` or `oauth`. For API keys, quack reads the variable at use
and stores nothing:

```toml
[providers.anthropic]
type = "anthropic"
auth = "api-key"
api_key_env = "ANTHROPIC_API_KEY"
```

## OAuth as quack

With `auth = "oauth"`, quack gets an access token from the organization's issuer and sends it
as the bearer. `grant` decides who signs in:

| `grant` | Who signs in | How the token renews |
|---|---|---|
| `authorization-code` (default) | a person, in a browser | with the refresh token |
| `device-code` | a person, who types a code on another device | with the refresh token |
| `client-credentials` | nobody; quack uses its own secret or key | quack asks again |
| `on-behalf-of` | each person who signs in to `quack serve` | per request ([next section](#on-behalf-of-each-person)) |

```toml
[providers.gateway]
type = "openai"
base_url = "https://models.example.com/v1"
auth = "oauth"

[providers.gateway.oauth]
issuer_url = "https://login.example.com"
client_id = "quack"
scopes = ["model.use", "offline_access"]
# grant = "authorization-code"
# client_secret_env = "GATEWAY_SECRET"     # client-credentials and on-behalf-of need a secret or a key
# client_auth = "client_secret_post"       # or "client_secret_basic", or "private_key_jwt"
# redirect_uri = "http://127.0.0.1:19876/callback"
```

```bash
quack auth login gateway    # a browser, or a device code where none can open (--device-code forces it)
quack auth status           # each OAuth provider: when its token expires and how it renews
quack auth logout gateway
```

quack encrypts the token in `control.db`, so one login serves every later process on that data
directory, `quack serve` included. A `client-credentials` provider needs no login; its first
request gets a token. When a person must sign in but cannot, the command-line interface exits
with code 4 and the server answers `503`, both naming `quack auth login NAME`.

Two practices keep logins working:

- **Prefer the browser flow.** Use `device-code` only on a host with no browser, such as one
  reached over SSH. An attacker can start a device-code login, send the code to a victim, and
  receive the victim's token when they approve it.
- **Make model calls through one long-lived process.** When two processes on one data
  directory, such as `quack serve` and `quack -p`, renew a token at the same moment, an issuer
  that rotates refresh tokens (Okta, Auth0) treats the second renewal as theft. It revokes the
  token, and every process needs `quack auth login` again.

## On behalf of each person

With `grant = "on-behalf-of"`, each request reaches the provider as the person who made it,
so the provider's own logs, quotas, and access policies see individual users. It works only
in `quack serve`, with people who sign in through the same issuer
([`authentication.md`](authentication.md#sign-in-through-the-organizations-identity-provider)).

```toml
[providers.gateway.oauth]
issuer_url = "https://login.example.com"
client_id = "quack"
client_secret_env = "GATEWAY_SECRET"   # or client_auth = "private_key_jwt"
grant = "on-behalf-of"
exchange = "token-exchange"            # or "entra"
audience = "api://model-gateway"       # Okta and Auth0: the provider's audience
# resource = "https://model.example.com"
# actor = true                         # the default: quack also names itself in the exchange
# scopes = ["model.use"]
```

For each request, quack trades the person's token for a token to the provider and caches it
in memory until 60 seconds before it expires. `exchange` picks the request format:
`token-exchange` (RFC 8693) for Okta, Auth0, and Vouch, or `entra` for Microsoft Entra ID.

quack acts for these people:

- a request to `quack serve`: its signed-in caller;
- a background job, such as the embeddings an upload triggers: the person who submitted it,
  even hours later;
- an MCP `query` or `search`: the user of that MCP connection.

quack refuses every other request and never falls back to its own identity. The command-line
interface, the terminal, local mode, and password users who never signed in through the
issuer all get "acts on behalf of the signed-in person and could not": `403` from the server,
exit code 4 from the command line. Give them a second provider entry if they need a model.

## Proving quack's identity with a key

With `client_auth = "private_key_jwt"`, quack proves its identity with a private key it creates
and keeps; the issuer holds only the public key. The `client-credentials` and `on-behalf-of`
grants accept the key in place of a secret.

```bash
quack auth jwks gateway                       # the public key set to register for [providers.gateway.oauth]
quack auth jwks --rotate gateway              # adds a new key beside the old one; register both
quack auth jwks --rotate --activate gateway   # switches to the new key; register it alone
```

Restart `quack serve` after `--activate`, since it signs with the old key until then.
Microsoft Entra ID accepts only certificates, not this key, so use a client secret there.

When the issuer supports dynamic client registration, leave `client_id` out and run `quack
auth register`, which creates the client and fills in its `client_id`. It also updates the
issuer at each rotation step. [`authentication.md`](authentication.md#letting-quack-register-itself)
lists its options. When `[server.oidc]` names the same issuer, one registration and one key
serve both.

## Proxies

quack sends every outbound HTTP request through the forward proxy the environment names:
model providers, Amazon Bedrock and its AWS credential calls, OAuth and OpenID Connect
issuers, and `quack import` of an HTTP(S) file.

| Variable | Used for |
|---|---|
| `HTTPS_PROXY` | `https://` requests |
| `HTTP_PROXY` | `http://` requests |
| `ALL_PROXY` | a scheme whose own variable is unset |
| `NO_PROXY` | hosts to reach directly, comma-separated |

The upper-case name wins over the lower-case one. A value without a scheme is an `http://`
proxy. Credentials go in the URL (`http://user:password@proxy.corp:8080`); quack never
prints them.

**Always direct.** `localhost`, `127.0.0.0/8`, `::1`, and `169.254.0.0/16` never go through
the proxy, whatever `NO_PROXY` holds. A local Ollama and the EC2 and ECS credential
endpoints therefore need no entry. A model server on another host does: under Docker
Compose, add the `ollama` service name to `NO_PROXY`.

**`NO_PROXY` forms.** A domain matches itself and its subdomains (`corp.example` and
`.corp.example` are the same). An address (`10.1.2.3`) and a range (`10.0.0.0/8`) match
addresses written in the URL. A lone `*` matches every hostname, but no address. Globs
(`*.corp.example`) and entries with a port (`host:8443`) match nothing; `quack doctor` names
them.

**Limits.**

- SOCKS proxies are not supported. Requests through one fail, and `quack doctor` fails the
  check.
- Amazon Bedrock's `converse` API and AWS credential calls take one proxy. When
  `HTTP_PROXY` and `HTTPS_PROXY` differ they use `HTTPS_PROXY`, and an `http://` Bedrock
  `base_url` is reached directly.
- A proxy that inspects TLS presents its own certificate. Add its certificate authority
  to the operating system's trust store.
- With `[import].allow_private_hosts` off, an import through a proxy checks only an
  address written in the URL. The proxy resolves names, so the proxy decides which hosts
  a name may reach.

`quack doctor` prints the proxy in effect, and `quack config` lists which of the four
variables are set.

## Recipe: Amazon Bedrock

Bedrock has two endpoints that host different models, so quack has one provider type for
each. Both sign every request with AWS credentials. quack stores none of them.

```toml
[general]
chat_model = "bedrock/us.anthropic.claude-opus-5-5"   # or "mantle/openai.gpt-oss-120b"

[embedding]
model = "bedrock/amazon.titan-embed-text-v2:0"
dimension = 1024

[providers.bedrock]
type = "bedrock"                      # bedrock-runtime
# api = "converse"                    # the default; or "chat-completions", "responses"
# aws_profile = "my-sso-profile"      # else AWS_PROFILE, else "default"
# region = "us-east-1"                # else base_url's, AWS_REGION, or the profile's

[providers.mantle]
type = "bedrock-mantle"               # bedrock-mantle
# api = "responses"                   # the default; or "chat-completions"
# aws_profile = "my-sso-profile"
```

- **Credentials.** quack finds credentials as the AWS CLI does: environment variables, then
  the profile (with `role_arn`, `credential_process`, and `aws sso login`), then web identity
  on EKS, then the ECS and EC2 instance roles. Run `aws sso login --profile my-sso-profile`
  before starting quack with an SSO profile.
- **Which endpoint.** `bedrock` serves Converse, cross-region inference profiles (`us.…`
  model IDs), embeddings, and a FIPS endpoint. `bedrock-mantle` serves models and Responses
  features only it has, such as Responses for GPT OSS. An entry calls one endpoint, so a
  model on the other needs a second entry.
- **Private networking.** An interface VPC endpoint with private DNS needs no setting.
  Without private DNS, set `base_url` to the endpoint's root, such as
  `https://vpce-0123456789abcdef0.bedrock-mantle.us-east-1.vpce.amazonaws.com`.
- **FIPS.** Set `use_fips_endpoint = true` in the profile or `AWS_USE_FIPS_ENDPOINT=true` to
  reach `bedrock-runtime-fips`. `bedrock-mantle` has no FIPS endpoint.
- **Retention.** quack sends every Responses request with `store: false`, so Bedrock keeps no
  copy of the conversation.

`quack doctor` resolves the credentials and the endpoint for each entry, and on
`bedrock-mantle` confirms the model is listed.

## Recipe: a dedicated rerank model

`[retrieval].rerank = "reranker"` orders search results with a cross-encoder instead of
the chat model: one fast `/rerank` call per search, which leaves the chat model's request
permits free. Serve the model with vLLM (`vllm serve BAAI/bge-reranker-v2-m3`), llama.cpp
(`llama-server --reranking`), or Text Embeddings Inference, and add that server as a
`type = "openai"` provider with its `base_url`:

```toml
[retrieval]
rerank = "reranker"
rerank_model = "rerank/BAAI/bge-reranker-v2-m3"

[providers.rerank]
type = "openai"
base_url = "http://localhost:8000/v1"   # quack posts to {base_url}/rerank
# auth = "api-key"                      # only when the server requires a key
# api_key_env = "RERANK_API_KEY"
```

The bearer is sent only when the provider has a key, since `llama-server` refuses one it
was not started with. Ollama, Anthropic, and Bedrock serve no rerank endpoint, so quack
refuses them for `rerank_model` when it loads the config. `quack doctor` checks that the
server lists the model and answers one small rerank call.

**Offline or air-gapped: Qwen3-Reranker-0.6B on llama.cpp.** Of the rerankers small enough
to run beside Ollama on one machine, Qwen3-Reranker-0.6B scores highest on the MTEB-R
reranking benchmark (65.80, against 57.03 for bge-reranker-v2-m3; the model card has the
table). llama.cpp's server serves it at `/v1/rerank` with no network at runtime:

```bash
llama-server --reranking -m Qwen3-Reranker-0.6B-Q8_0.gguf --port 8000
```

```toml
[retrieval]
rerank = "reranker"
rerank_model = "rerank/Qwen3-Reranker-0.6B"
# rerank_candidates = 24

[providers.rerank]
type = "openai"
base_url = "http://localhost:8000/v1"
```

`rerank_candidates` (24) is how many fused hits the reranker scores before `top_k` are
kept. Raise it, to 50 or so, when the workspace holds many near-duplicate passages (versions
of one policy, templated reports): the right passage is then often below rank 24 in the fused
list, and a reranker scores a pair in a few milliseconds, so the cost is small. Leave it
when the search already returns the right document in its first page.

## Recipe: LiteLLM

A LiteLLM proxy speaks the OpenAI API, so quack reaches it as one `openai` provider with a
LiteLLM virtual key. quack's model names are the `model_name` values in LiteLLM's
configuration, and LiteLLM holds every backend credential.

```toml
[general]
chat_model = "litellm/claude-opus"

[embedding]
model = "litellm/embed"
dimension = 1024

[providers.litellm]
type = "openai"
base_url = "https://litellm.example.com/v1"
auth = "api-key"
api_key_env = "LITELLM_API_KEY"         # a virtual key from LiteLLM's /key/generate
# max_concurrent_requests = 16          # match the key's parallel-request limit
```

quack uses Chat Completions here, the default for an OpenAI-compatible `base_url`. Set
`api = "responses"` only when every model the key reaches supports LiteLLM's Responses route.

## Recipe: Azure API Management, LiteLLM, Bedrock, and Azure OpenAI

In this layout, APIM is the front door, LiteLLM routes each model, and the models run on
Bedrock and Azure OpenAI. Each hop authenticates the next, and quack talks only to APIM:

```mermaid
flowchart LR
    Q[quack serve] -- "Entra token for the person" --> A[Azure API Management]
    A -- "LiteLLM key" --> L[LiteLLM]
    L -- "AWS credentials" --> B[Amazon Bedrock]
    L -- "Azure credentials" --> O[Azure OpenAI]
```

quack acts on behalf of each person, so APIM's logs, quotas, and policies see the person
rather than one shared quack identity. quack holds no LiteLLM key, AWS credential, or Azure
OpenAI key.

### 1. Register two applications in Entra ID

1. **The gateway.** Register an application for APIM and expose an API on it. Set
   `requestedAccessTokenVersion` to `2` in its manifest. Note its client ID (`{gateway_client_id}`).
2. **quack.** Register quack's application with the web redirect
   `https://quack.example.com/auth/oidc/callback` and a client secret. Expose an API on it
   with the scope `access_as_user`, so a person's sign-in yields a token for quack that quack
   can exchange. Add the delegated permission to the gateway's API, and grant admin consent.

### 2. Validate the token in APIM and forward to LiteLLM

In the API's inbound policy, APIM checks that the token came from quack's application for the
gateway, then replaces it with LiteLLM's key:

```xml
<inbound>
  <base />
  <validate-azure-ad-token tenant-id="{tenant_id}">
    <client-application-ids>
      <application-id>{quack_client_id}</application-id>
    </client-application-ids>
    <audiences>
      <audience>{gateway_client_id}</audience>
    </audiences>
  </validate-azure-ad-token>
  <set-backend-service base-url="https://litellm.internal.example.com/v1" />
  <set-header name="Authorization" exists-action="override">
    <value>Bearer {{litellm-key}}</value>
  </set-header>
</inbound>
```

`{{litellm-key}}` is an APIM named value holding a LiteLLM virtual key. APIM can key its
logs and rate limits on the validated token's `oid` claim, the person's Entra object ID.

### 3. Route models in LiteLLM

LiteLLM maps each model name quack uses to a backend, with the backend's credentials:

```yaml
model_list:
  - model_name: claude-opus
    litellm_params:
      model: bedrock/us.anthropic.claude-opus-5-5
      aws_region_name: us-east-1
  - model_name: gpt
    litellm_params:
      model: azure/{deployment}
      api_base: https://{resource}.openai.azure.com
      api_key: os.environ/AZURE_API_KEY
  - model_name: embed
    litellm_params:
      model: bedrock/amazon.titan-embed-text-v2:0
      aws_region_name: us-east-1
```

LiteLLM's documentation covers each backend's other credential options.

### 4. Configure quack

```toml
[general]
chat_model = "gateway/claude-opus"      # or "gateway/gpt"

[embedding]
model = "gateway/embed"
dimension = 1024

[server.oidc]
issuer_url = "https://login.microsoftonline.com/{tenant_id}/v2.0"
client_id = "{quack_client_id}"
client_secret_env = "QUACK_ENTRA_SECRET"
redirect_uri = "https://quack.example.com/auth/oidc/callback"
subject_claim = "oid"
scopes = ["openid", "profile", "email", "offline_access", "api://{quack_client_id}/access_as_user"]

[providers.gateway]
type = "openai"
base_url = "https://{apim_name}.azure-api.net/llm"   # the APIM API's URL
auth = "oauth"
# headers = { "Ocp-Apim-Subscription-Key" = "…" }    # only if the APIM API requires a subscription

[providers.gateway.oauth]
issuer_url = "https://login.microsoftonline.com/{tenant_id}/v2.0"
client_id = "{quack_client_id}"
client_secret_env = "QUACK_ENTRA_SECRET"
grant = "on-behalf-of"
exchange = "entra"
scopes = ["api://{gateway_client_id}/.default"]
```

quack appends `/chat/completions` and `/embeddings` to `base_url`, and APIM forwards them to
LiteLLM's `/v1`. The Entra token already authenticates each request, so turn off the APIM
subscription requirement for this API. If policy requires a subscription key, `headers` can
carry it, but the key then sits in the configuration file in plain text.

### 5. Check it

```bash
quack doctor                    # checks the issuer, the exchange grant, and the gateway
quack serve
```

People sign in with "Sign in with login.microsoftonline.com". Their first model request
exchanges their token, and APIM sees a token issued to them. The command line and password
users cannot use `gateway`; give them a second provider entry if they need a model.

## Recipe: on behalf of each person, with Vouch

With [Vouch](https://vouch.sh) as the issuer, one client serves both the sign-in to `quack
serve` and the token exchange, and it proves its identity with quack's key instead of a
secret.

```toml
[server.oidc]
issuer_url = "https://us.vouch.sh"
client_auth = "private_key_jwt"
redirect_uri = "https://quack.example.com/auth/oidc/callback"
scopes = ["openid", "email"]

[providers.gateway]
type = "openai"
base_url = "https://models.example.com/v1"
auth = "oauth"

[providers.gateway.oauth]
issuer_url = "https://us.vouch.sh"
client_auth = "private_key_jwt"
grant = "on-behalf-of"
exchange = "token-exchange"
actor = false
# audience = "https://models.example.com"   # when the model API expects one
```

```bash
quack auth register             # you sign in to Vouch; quack registers the client as yours
quack user add admin --admin    # Vouch sign-ins start with no workspaces
quack doctor
quack serve
```

Neither section sets `client_id`, so `quack auth register` fills it in for both. Vouch records
you as the client's owner with access scope Personal, so only you can sign in at first. On the
Applications page of the Vouch console, set **Access scope** to **Organization**. The step is
manual on purpose: it decides who in the organization can use quack.

Four settings matter:

- **`private_key_jwt`**: only quack's key can turn a person's Vouch token into a model token,
  and no shared secret exists to leak.
- **`actor = false`**: Vouch accepts an actor token only from a Vouch user, and quack's token
  names a client, so Vouch would refuse the exchange with "Actor token user not found".
- **The standard security profile, not FAPI 2.0**: FAPI binds tokens to a key, and a model API
  that takes bearer tokens cannot use a bound token.
- **No refresh tokens**: Vouch issues none and offers only the `openid` and `email` scopes, so
  a sign-in lasts for Vouch's session.

To rotate the key, run `quack auth jwks --rotate`, then `quack auth jwks --rotate --activate`,
then restart `quack serve`. quack updates Vouch at both steps.

Without registration, create a Web application in the Vouch console with the callback as its
redirect URI, access scope Organization, security profile Standard OAuth, and client
authentication `private_key_jwt`. Add its `client_id` to both sections, then paste the output
of `quack auth jwks` into the application's JWKS (JSON Web Key Set) field. Paste again at each
rotation step. A Vouch console without the client-authentication choice issues a client
secret instead: set `client_secret_env` and `client_auth = "client_secret_basic"` in both
sections.

## One person on the command line

For one person on their own machine, register a public client, with no secret and no key:

```toml
[providers.gateway.oauth]
issuer_url = "https://login.example.com"
client_id = "quack-cli"
# redirect_uri = "http://127.0.0.1:19876/callback"   # register this with the issuer
```

`quack auth login gateway` opens the browser and catches the redirect on the loopback
address. A secret or a key adds nothing when the person can read it anyway.

## Identity-provider notes

These settings come from each vendor's documentation as of September 2026. quack has not yet
run against a live tenant of any of them.

**Microsoft Entra ID.** Set `exchange = "entra"` and list the downstream API's scope, such as
`api://{gateway_client_id}/.default`. The person's token must be for quack's own API, so add
that API's scope, such as `api://{quack_client_id}/access_as_user`, to
`[server.oidc].scopes`. Entra's exchange has no actor, so quack ignores `actor`.

**Okta.** Create an API Services application with the Token Exchange grant, set
`client_auth = "client_secret_basic"`, and set `audience` to the downstream authorization
server's audience. An exchange across two authorization servers needs Okta's trusted
servers, and an NHI (non-human identity) subscription bought or renewed on or after
August 14, 2026.

**Auth0.** Turn on On-Behalf-Of Token Exchange on quack's own client, and set `audience` to
the downstream API's identifier.

## Troubleshooting

`quack doctor` checks every provider: the endpoint answers, the credential works, and the
model exists. It checks an on-behalf-of provider with quack's own token, or only the issuer's
advertised grants when `actor = false`. `--offline` skips network checks.

- **"provider 'X' needs a login"** (exit code 4, or `503` from the server): run
  `quack auth login X` as the server's operating-system user, on the same data directory.
- **"provider 'X' acts on behalf of the signed-in person and could not"**: the request came
  from the command line, the terminal, local mode, or a user without an identity-provider
  sign-in. Sign in through the issuer, or use another provider entry.
- **`invalid_client` with `private_key_jwt`**: the issuer does not hold quack's current key.
  Register the output of `quack auth jwks NAME`, and restart `quack serve` after
  `--activate`.
- **"provider 'X' is not allowed in this workspace"** (`403` from the server, exit code 1):
  the workspace's allowed providers leave out the provider of a configured model. Add the
  provider on the workspace's settings page, or move the model to an allowed provider.
- **"model 'M' of provider 'X' runs in Ollama's cloud"**: the workspace restricts its
  providers and the model has a `cloud` tag. Configure a local model, or allow every provider.
- **A Bedrock credential error at the first request**: the AWS SDK found no valid
  credentials. Run `aws sso login` for the profile, or check `aws_profile` and `AWS_PROFILE`.

**Does quack support DPoP (proof-of-possession tokens)?** No. A bound token needs a fresh
proof on every request, and model APIs that accept bearer tokens refuse it. Issuers that
support DPoP, Vouch among them, still issue bearer tokens to quack.
