//! The forward proxy every outbound HTTP client uses, from `HTTP_PROXY`,
//! `HTTPS_PROXY`, `ALL_PROXY`, and `NO_PROXY`.
//!
//! reqwest reads these on its own, but the AWS SDK client does not, and
//! neither keeps a local model server off the proxy. [`Proxies`] reads them
//! once and configures both clients the same way.

use std::borrow::Cow;
use std::fmt;
use std::net::IpAddr;
use std::sync::LazyLock;

use aws_smithy_http_client::proxy::ProxyConfig;
use http::uri::Authority;
use hyper_util::client::proxy::matcher::Matcher;
use reqwest::{NoProxy, Proxy, Url};

/// Reached directly whatever `NO_PROXY` says: this machine, and the
/// link-local range, where the EC2 and ECS credential endpoints live.
const ALWAYS_DIRECT: &str = "localhost,127.0.0.0/8,::1,169.254.0.0/16";

static ENVIRONMENT: LazyLock<Proxies> = LazyLock::new(|| Proxies::new(Environment::read()));

/// One proxy variable that is set and not empty.
#[derive(Debug, Clone)]
pub(crate) struct Variable {
    name: &'static str,
    value: String,
}

impl Variable {
    /// The first of `names` that is set and not empty.
    fn read(names: [&'static str; 2]) -> Option<Self> {
        names.into_iter().find_map(|name| {
            let value = std::env::var(name).ok()?;
            (!value.trim().is_empty()).then_some(Self { name, value })
        })
    }

    #[cfg(test)]
    #[expect(clippy::unnecessary_wraps, reason = "fills an `Environment` field")]
    pub(crate) fn named(name: &'static str, value: &str) -> Option<Self> {
        Some(Self {
            name,
            value: value.to_owned(),
        })
    }
}

/// The proxy variables as the process was started with them.
#[derive(Debug, Clone, Default)]
pub(crate) struct Environment {
    pub(crate) http: Option<Variable>,
    pub(crate) https: Option<Variable>,
    pub(crate) all: Option<Variable>,
    pub(crate) no: Option<Variable>,
}

impl Environment {
    /// The upper-case name wins over the lower-case one, as in curl.
    fn read() -> Self {
        Self {
            http: Variable::read(["HTTP_PROXY", "http_proxy"]),
            https: Variable::read(["HTTPS_PROXY", "https_proxy"]),
            all: Variable::read(["ALL_PROXY", "all_proxy"]),
            no: Variable::read(["NO_PROXY", "no_proxy"]),
        }
    }
}

/// A proxy to send through, and the variable that named it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProxyUrl {
    url: Url,
    variable: &'static str,
}

impl ProxyUrl {
    /// A value without a scheme is an `http://` proxy, as reqwest reads it.
    fn parse(variable: &Variable) -> Result<Self, Problem> {
        let unusable = || Problem::Unusable {
            variable: variable.name,
        };
        let value = variable.value.trim();
        let value = if value.contains("://") {
            Cow::Borrowed(value)
        } else {
            Cow::Owned(format!("http://{value}"))
        };
        let url = Url::parse(&value).map_err(|_| unusable())?;
        if !url.has_host() {
            return Err(unusable());
        }
        match url.scheme() {
            "http" | "https" | "socks4" | "socks4a" | "socks5" | "socks5h" => Ok(Self {
                url,
                variable: variable.name,
            }),
            _ => Err(unusable()),
        }
    }

    /// The scheme, when it is SOCKS: neither HTTP client here speaks it.
    #[must_use]
    pub fn unsupported_scheme(&self) -> Option<&str> {
        match self.url.scheme() {
            "http" | "https" => None,
            scheme => Some(scheme),
        }
    }
}

/// The host and port alone: the userinfo may hold a password.
impl fmt::Display for ProxyUrl {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.url.host_str().unwrap_or_default())?;
        match self.url.port_or_known_default() {
            Some(port) => write!(f, ":{port}"),
            None => Ok(()),
        }
    }
}

/// Something in the proxy variables that does not do what it says.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Problem {
    /// Not a proxy URL; the variable is left out.
    Unusable { variable: &'static str },
    /// A SOCKS proxy: requests through it fail.
    Socks {
        variable: &'static str,
        scheme: String,
    },
    /// A `NO_PROXY` entry in a form the matcher never matches.
    NeverMatches { entry: String, instead: String },
}

impl Problem {
    /// The `NO_PROXY` entries that match nothing: a glob other than a lone
    /// `*`, and a host with a port.
    fn in_no_proxy(entry: &str) -> Option<Self> {
        let never = |instead: &str| Self::NeverMatches {
            entry: entry.to_owned(),
            instead: instead.to_owned(),
        };
        if entry.contains('*') && entry != "*" {
            return Some(never(&entry.replace('*', "")));
        }
        if entry.parse::<IpAddr>().is_ok() {
            return None;
        }
        let authority = entry.parse::<Authority>().ok()?;
        authority.port()?;
        Some(never(authority.host()))
    }

    /// Whether requests fail or leave unproxied because of it.
    #[must_use]
    pub const fn is_failure(&self) -> bool {
        match self {
            Self::Unusable { .. } | Self::Socks { .. } => true,
            Self::NeverMatches { .. } => false,
        }
    }

    /// What to change.
    #[must_use]
    pub fn fix(&self) -> String {
        match self {
            Self::Unusable { variable } | Self::Socks { variable, .. } => {
                format!("set {variable} to an http:// or https:// proxy URL, or unset it")
            }
            Self::NeverMatches { instead, .. } => format!("write \"{instead}\""),
        }
    }
}

impl fmt::Display for Problem {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unusable { variable } => write!(
                f,
                "{variable} is not an http:// or https:// proxy URL; requests go direct until \
                 it is fixed"
            ),
            Self::Socks { variable, scheme } => write!(
                f,
                "{variable} is a {scheme} proxy, which quack does not support; requests \
                 through it fail"
            ),
            Self::NeverMatches { entry, .. } => {
                write!(f, "NO_PROXY entry \"{entry}\" never matches")
            }
        }
    }
}

/// Whether a request to a URL goes through a proxy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Route {
    Direct,
    Proxied,
}

/// The proxy settings of this process.
#[derive(Debug)]
pub struct Proxies {
    http: Option<ProxyUrl>,
    https: Option<ProxyUrl>,
    /// How many entries `NO_PROXY` holds.
    no_proxy_entries: usize,
    /// `NO_PROXY` and [`ALWAYS_DIRECT`], in the form both clients take.
    direct: String,
    /// The matcher reqwest and the AWS client run, over the same values.
    matcher: Matcher,
    problems: Vec<Problem>,
}

impl Proxies {
    /// The settings in the environment, read once.
    #[must_use]
    pub fn from_env() -> &'static Self {
        &ENVIRONMENT
    }

    /// `ALL_PROXY` stands in for a scheme whose own variable is unset or
    /// unusable.
    pub(crate) fn new(environment: Environment) -> Self {
        let mut problems = Vec::new();
        let mut usable = |variable: Option<Variable>| {
            let variable = variable?;
            ProxyUrl::parse(&variable)
                .map_err(|problem| problems.push(problem))
                .ok()
        };
        let http = usable(environment.http);
        let https = usable(environment.https);
        let all = if http.is_none() || https.is_none() {
            usable(environment.all)
        } else {
            None
        };
        let http = http.or_else(|| all.clone());
        let https = https.or(all);
        for proxy in [&http, &https].into_iter().flatten() {
            let Some(scheme) = proxy.unsupported_scheme() else {
                continue;
            };
            let problem = Problem::Socks {
                variable: proxy.variable,
                scheme: scheme.to_owned(),
            };
            if !problems.contains(&problem) {
                problems.push(problem);
            }
        }

        let no_proxy = environment.no.map(|v| v.value).unwrap_or_default();
        let mut no_proxy_entries = 0_usize;
        for entry in no_proxy.split(',').map(str::trim) {
            if entry.is_empty() {
                continue;
            }
            no_proxy_entries = no_proxy_entries.saturating_add(1);
            problems.extend(Problem::in_no_proxy(entry));
        }
        let direct = if no_proxy_entries == 0 {
            ALWAYS_DIRECT.to_owned()
        } else {
            format!("{no_proxy},{ALWAYS_DIRECT}")
        };

        for problem in &problems {
            tracing::warn!("{problem}; {}", problem.fix());
        }
        let as_value = |proxy: &Option<ProxyUrl>| {
            proxy
                .as_ref()
                .map(|p| p.url.as_str().to_owned())
                .unwrap_or_default()
        };
        let matcher = Matcher::builder()
            .http(as_value(&http))
            .https(as_value(&https))
            .no(direct.as_str())
            .build();
        Self {
            http,
            https,
            no_proxy_entries,
            direct,
            matcher,
            problems,
        }
    }

    /// The builder every reqwest client starts from: these proxies in place
    /// of reqwest's own reading of the environment.
    #[expect(
        clippy::disallowed_methods,
        reason = "the one place a reqwest client starts"
    )]
    pub fn client(&self) -> reqwest::ClientBuilder {
        let mut builder = reqwest::Client::builder().no_proxy();
        let direct = NoProxy::from_string(&self.direct);
        let proxies = [
            self.http.as_ref().map(|p| Proxy::http(p.url.clone())),
            self.https.as_ref().map(|p| Proxy::https(p.url.clone())),
        ];
        for proxy in proxies.into_iter().flatten() {
            match proxy {
                Ok(proxy) => builder = builder.proxy(proxy.no_proxy(direct.clone())),
                Err(e) => tracing::warn!(error = %e, "reqwest refused a proxy URL"),
            }
        }
        builder
    }

    /// The same settings for the AWS SDK's connector. It takes one proxy:
    /// with two different ones the HTTPS proxy is used, and plain HTTP
    /// requests go direct.
    #[must_use]
    pub fn aws(&self) -> ProxyConfig {
        let config = match (&self.http, &self.https) {
            (Some(http), Some(https)) if http.url == https.url => {
                ProxyConfig::all(https.url.as_str())
            }
            (_, Some(https)) => ProxyConfig::https(https.url.as_str()),
            (Some(http), None) => ProxyConfig::http(http.url.as_str()),
            (None, None) => return ProxyConfig::disabled(),
        };
        match config {
            Ok(config) => config.no_proxy(&self.direct),
            Err(e) => {
                tracing::debug!(error = %e, "the AWS client goes direct");
                ProxyConfig::disabled()
            }
        }
    }

    /// Whether a request to `url` goes through a proxy.
    #[must_use]
    pub fn route(&self, url: &Url) -> Route {
        let Ok(uri) = url.as_str().parse::<http::Uri>() else {
            return Route::Direct;
        };
        match self.matcher.intercept(&uri) {
            Some(_) => Route::Proxied,
            None => Route::Direct,
        }
    }

    /// The proxy for `http://` requests.
    #[must_use]
    pub const fn http(&self) -> Option<&ProxyUrl> {
        self.http.as_ref()
    }

    /// The proxy for `https://` requests.
    #[must_use]
    pub const fn https(&self) -> Option<&ProxyUrl> {
        self.https.as_ref()
    }

    /// How many entries `NO_PROXY` holds.
    #[must_use]
    pub const fn no_proxy_entries(&self) -> usize {
        self.no_proxy_entries
    }

    /// What in the variables does not do what it says.
    #[must_use]
    pub fn problems(&self) -> &[Problem] {
        &self.problems
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use tokio::io::{AsyncReadExt as _, AsyncWriteExt as _};
    use tokio::net::TcpListener;

    use super::*;

    #[expect(clippy::panic, reason = "test helper")]
    fn fail(msg: &str) -> ! {
        panic!("{msg}")
    }

    fn route(proxies: &Proxies, url: &str) -> Route {
        proxies.route(&Url::parse(url).unwrap_or_else(|e| fail(&e.to_string())))
    }

    fn both(proxy: &str, no_proxy: &str) -> Proxies {
        Proxies::new(Environment {
            http: Variable::named("HTTP_PROXY", proxy),
            https: Variable::named("HTTPS_PROXY", proxy),
            all: None,
            no: Variable::named("NO_PROXY", no_proxy),
        })
    }

    #[test]
    fn nothing_set_sends_everything_direct() {
        let proxies = Proxies::new(Environment::default());
        assert_eq!(route(&proxies, "https://api.example.com/v1"), Route::Direct);
        assert!(proxies.aws().is_disabled());
        assert!(proxies.problems().is_empty());
        assert!(proxies.http().is_none() && proxies.https().is_none());
    }

    #[test]
    fn this_machine_and_link_local_are_never_proxied() {
        let proxies = both("http://proxy.corp:8080", "");
        for direct in [
            "http://localhost:11434/api/tags",
            "http://127.0.0.1:11434/",
            "http://127.8.9.1/",
            "http://[::1]:11434/",
            "http://169.254.169.254/latest/meta-data/",
            "http://169.254.170.2/v2/credentials",
        ] {
            assert_eq!(route(&proxies, direct), Route::Direct, "{direct}");
        }
        for proxied in [
            "https://api.example.com/",
            "http://example.com/",
            "http://10.0.0.5/",
        ] {
            assert_eq!(route(&proxies, proxied), Route::Proxied, "{proxied}");
        }
    }

    #[test]
    fn no_proxy_adds_to_what_is_direct() {
        let proxies = both(
            "http://proxy.corp:8080",
            "internal.corp, 10.0.0.0/8,192.168.1.7",
        );
        assert_eq!(proxies.no_proxy_entries(), 3);
        for direct in [
            "https://internal.corp/",
            "https://models.internal.corp/",
            "http://10.1.2.3/",
            "http://192.168.1.7/",
            "http://localhost/",
        ] {
            assert_eq!(route(&proxies, direct), Route::Direct, "{direct}");
        }
        assert_eq!(route(&proxies, "https://notinternal.corp/"), Route::Proxied);
        assert_eq!(route(&proxies, "http://192.168.1.8/"), Route::Proxied);
    }

    #[test]
    fn all_proxy_stands_in_for_an_unset_scheme() {
        let proxies = Proxies::new(Environment {
            http: Variable::named("http_proxy", "http://plain.corp:3128"),
            all: Variable::named("ALL_PROXY", "http://all.corp:8080"),
            ..Environment::default()
        });
        assert_eq!(
            proxies.http().map(ToString::to_string).as_deref(),
            Some("plain.corp:3128")
        );
        assert_eq!(
            proxies.https().map(ToString::to_string).as_deref(),
            Some("all.corp:8080")
        );
        assert_eq!(route(&proxies, "https://example.com/"), Route::Proxied);
    }

    #[test]
    fn one_scheme_alone_leaves_the_other_direct() {
        let proxies = Proxies::new(Environment {
            https: Variable::named("HTTPS_PROXY", "http://proxy.corp:8080"),
            ..Environment::default()
        });
        assert_eq!(route(&proxies, "https://example.com/"), Route::Proxied);
        assert_eq!(route(&proxies, "http://example.com/"), Route::Direct);
        assert!(!proxies.aws().is_disabled());
    }

    #[test]
    fn a_value_without_a_scheme_is_an_http_proxy_and_the_password_is_not_shown() {
        let proxies = both("user:s3cret@proxy.corp:8080", "");
        assert!(proxies.problems().is_empty(), "{:?}", proxies.problems());
        let shown = proxies.https().map(ToString::to_string).unwrap_or_default();
        assert_eq!(shown, "proxy.corp:8080");
        assert_eq!(route(&proxies, "https://example.com/"), Route::Proxied);
    }

    #[test]
    fn an_unusable_value_is_left_out_and_reported() {
        let proxies = Proxies::new(Environment {
            http: Variable::named("HTTP_PROXY", "ftp://proxy.corp"),
            https: Variable::named("HTTPS_PROXY", "http://"),
            ..Environment::default()
        });
        assert_eq!(route(&proxies, "http://example.com/"), Route::Direct);
        assert_eq!(route(&proxies, "https://example.com/"), Route::Direct);
        assert_eq!(
            proxies.problems(),
            [
                Problem::Unusable {
                    variable: "HTTP_PROXY"
                },
                Problem::Unusable {
                    variable: "HTTPS_PROXY"
                },
            ]
        );
        assert!(proxies.problems().iter().all(Problem::is_failure));
    }

    #[test]
    fn a_socks_proxy_is_kept_and_reported_once() {
        let proxies = Proxies::new(Environment {
            all: Variable::named("ALL_PROXY", "socks5://127.0.0.1:1080"),
            ..Environment::default()
        });
        assert_eq!(route(&proxies, "https://example.com/"), Route::Proxied);
        assert_eq!(
            proxies.problems(),
            [Problem::Socks {
                variable: "ALL_PROXY",
                scheme: String::from("socks5"),
            }]
        );
        assert!(proxies.aws().is_disabled());
    }

    #[test]
    fn all_proxy_is_not_judged_when_both_schemes_have_their_own() {
        let proxies = Proxies::new(Environment {
            http: Variable::named("HTTP_PROXY", "http://proxy.corp:8080"),
            https: Variable::named("HTTPS_PROXY", "http://proxy.corp:8080"),
            all: Variable::named("ALL_PROXY", "socks5://127.0.0.1:1080"),
            no: None,
        });
        assert!(proxies.problems().is_empty(), "{:?}", proxies.problems());
    }

    #[test]
    fn no_proxy_entries_that_never_match_are_named_with_the_form_that_does() {
        let proxies = both(
            "http://proxy.corp:8080",
            "*.internal,models.corp:8443,::1,10.0.0.0/8,.ok.corp",
        );
        assert_eq!(
            proxies.problems(),
            [
                Problem::NeverMatches {
                    entry: String::from("*.internal"),
                    instead: String::from(".internal"),
                },
                Problem::NeverMatches {
                    entry: String::from("models.corp:8443"),
                    instead: String::from("models.corp"),
                },
            ]
        );
        assert!(!proxies.problems().iter().any(Problem::is_failure));
        assert_eq!(route(&proxies, "https://a.internal/"), Route::Proxied);

        let star = both("http://proxy.corp:8080", "*");
        assert!(star.problems().is_empty());
        assert_eq!(route(&star, "https://example.com/"), Route::Direct);
    }

    /// A listener standing in for the proxy: it returns the first line of
    /// the one request it receives, or nothing when none arrives.
    async fn proxy() -> (String, tokio::task::JoinHandle<String>) {
        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .unwrap_or_else(|e| fail(&e.to_string()));
        let addr = listener
            .local_addr()
            .unwrap_or_else(|e| fail(&e.to_string()));
        let seen = tokio::spawn(async move {
            let accepted = tokio::time::timeout(Duration::from_secs(10), listener.accept());
            let Ok(Ok((mut socket, _))) = accepted.await else {
                return String::new();
            };
            let mut buf = [0_u8; 2048];
            let read = socket.read(&mut buf).await.unwrap_or_default();
            drop(
                socket
                    .write_all(
                        b"HTTP/1.1 200 OK\r\ncontent-length: 2\r\nconnection: close\r\n\r\nok",
                    )
                    .await,
            );
            let request = String::from_utf8_lossy(buf.get(..read).unwrap_or_default());
            request.lines().next().unwrap_or_default().to_owned()
        });
        (format!("http://{addr}"), seen)
    }

    #[tokio::test]
    async fn a_client_sends_other_hosts_to_the_proxy_and_loopback_direct() {
        let (address, seen) = proxy().await;
        let client = both(&address, "")
            .client()
            .build()
            .unwrap_or_else(|e| fail(&e.to_string()));
        // A name no resolver answers: only the proxy can take this request.
        let answer = client
            .get("http://files.invalid/data.csv")
            .send()
            .await
            .unwrap_or_else(|e| fail(&e.to_string()));
        assert!(answer.status().is_success());
        assert_eq!(
            seen.await.unwrap_or_default(),
            "GET http://files.invalid/data.csv HTTP/1.1"
        );

        // The "proxy" here refuses connections, so an answer came direct.
        let (origin, seen) = proxy().await;
        let client = both("http://127.0.0.1:9", "")
            .client()
            .build()
            .unwrap_or_else(|e| fail(&e.to_string()));
        let answer = client
            .get(format!("{origin}/health"))
            .send()
            .await
            .unwrap_or_else(|e| fail(&e.to_string()));
        assert!(answer.status().is_success());
        assert_eq!(seen.await.unwrap_or_default(), "GET /health HTTP/1.1");
    }

    #[tokio::test]
    async fn a_client_ignores_what_reqwest_would_read_itself() {
        // No proxy in these settings, so the request goes straight to the
        // listener even when the developer's shell sets proxy variables.
        let (origin, seen) = proxy().await;
        let client = Proxies::new(Environment::default())
            .client()
            .build()
            .unwrap_or_else(|e| fail(&e.to_string()));
        let answer = client
            .get(format!("{origin}/direct"))
            .send()
            .await
            .unwrap_or_else(|e| fail(&e.to_string()));
        assert!(answer.status().is_success());
        assert_eq!(seen.await.unwrap_or_default(), "GET /direct HTTP/1.1");
    }
}
