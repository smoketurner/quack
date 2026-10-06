//! Who a request came from when a proxy stands in front of `quack serve`.
//! The TCP peer is the one address a caller cannot choose, so the forwarded
//! headers are read only when that peer is in `[server].trusted_proxies`,
//! and then from the right, stopping at the first address no trusted proxy
//! wrote: the model of nginx's `set_real_ip_from` and Caddy's
//! `trusted_proxies`. With no trusted range (the default) the peer is the
//! client, whatever the headers say.

use std::net::IpAddr;

use forwarded_header_value::{ForwardedHeaderValue, ForwardedStanza, Protocol};
use http::HeaderMap;
use ipnet::IpNet;
use serde::Deserialize;

/// The forwarded headers a request carried, as the proxy wrote them.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Forwarded<'a> {
    /// RFC 7239 `Forwarded`.
    pub rfc7239: Option<&'a str>,
    /// `X-Forwarded-For`.
    pub x_forwarded_for: Option<&'a str>,
    /// `X-Forwarded-Proto`.
    pub x_forwarded_proto: Option<&'a str>,
}

/// Where a request came from, once the trusted proxies are peeled off.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Client {
    pub ip: IpAddr,
    /// Whether the hop the client made was TLS: what a trusted proxy said
    /// in `Forwarded; proto=` or `X-Forwarded-Proto`, else `None`.
    pub https: Option<bool>,
}

impl<'a> Forwarded<'a> {
    /// The forwarded headers of a request, as the proxy wrote them.
    #[must_use]
    pub fn of(headers: &'a HeaderMap) -> Self {
        let text = |name: &str| headers.get(name).and_then(|v| v.to_str().ok());
        Self {
            rfc7239: text("forwarded"),
            x_forwarded_for: text("x-forwarded-for"),
            x_forwarded_proto: text("x-forwarded-proto"),
        }
    }

    /// Each hop's address, in the order the proxies wrote them, with the
    /// scheme it names: `Forwarded` when present (one per stanza), else
    /// `X-Forwarded-For` with the one `X-Forwarded-Proto` scheme. A header
    /// that does not parse (`forwarded-header-value` refuses it whole)
    /// counts as absent.
    fn hops(self) -> Vec<(IpAddr, Option<bool>)> {
        if let Some(value) = self.rfc7239 {
            let Ok(value) = ForwardedHeaderValue::from_forwarded(value) else {
                return Vec::new();
            };
            return value
                .iter()
                .filter_map(|stanza| {
                    let https = stanza.forwarded_proto.map(|p| p == Protocol::Https);
                    stanza.forwarded_for_ip().map(|ip| (ip, https))
                })
                .collect();
        }
        let https = self
            .x_forwarded_proto
            .map(|proto| proto.trim().eq_ignore_ascii_case("https"));
        let Some(Ok(value)) = self
            .x_forwarded_for
            .map(ForwardedHeaderValue::from_x_forwarded_for)
        else {
            return Vec::new();
        };
        value
            .iter()
            .filter_map(ForwardedStanza::forwarded_for_ip)
            .map(|ip| (ip, https))
            .collect()
    }
}

/// `[server].trusted_proxies`: the address ranges of the proxies in front
/// of the server. Empty (the default) trusts nobody's forwarded headers.
#[derive(Debug, Clone, Default, PartialEq, Eq, Deserialize)]
#[serde(transparent)]
pub struct TrustedProxies(Vec<IpNet>);

impl From<Vec<IpNet>> for TrustedProxies {
    fn from(ranges: Vec<IpNet>) -> Self {
        Self(ranges)
    }
}

impl TrustedProxies {
    /// The ranges, as configured.
    #[must_use]
    pub fn ranges(&self) -> &[IpNet] {
        &self.0
    }

    fn contains(&self, ip: IpAddr) -> bool {
        self.0.iter().any(|net| net.contains(&ip))
    }

    /// The client behind `peer`: the peer itself unless it is a trusted
    /// proxy, in which case the rightmost forwarded address that is not
    /// itself a trusted proxy. A forwarded chain made only of trusted
    /// proxies resolves to its leftmost entry, the one the first proxy saw.
    #[must_use]
    pub fn client(&self, peer: IpAddr, headers: Forwarded<'_>) -> Client {
        let mut client = Client {
            ip: peer,
            https: None,
        };
        if !self.contains(peer) {
            return client;
        }
        for (ip, https) in headers.hops().into_iter().rev() {
            client = Client { ip, https };
            if !self.contains(ip) {
                break;
            }
        }
        client
    }

    /// [`Self::client`]'s address for a request's headers: what audit rows,
    /// the `Secure` cookie decision, and the rate limiter all take.
    #[must_use]
    pub fn client_ip(&self, peer: IpAddr, headers: &HeaderMap) -> IpAddr {
        self.client(peer, Forwarded::of(headers)).ip
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ip(s: &str) -> IpAddr {
        s.parse()
            .unwrap_or_else(|e: std::net::AddrParseError| unreachable_ip(&e.to_string()))
    }

    fn net(s: &str) -> IpNet {
        s.parse()
            .unwrap_or_else(|e: ipnet::AddrParseError| unreachable_ip(&e.to_string()))
    }

    fn proxies_client(peer: IpAddr, headers: Forwarded<'_>, trusted: &[IpNet]) -> Client {
        TrustedProxies::from(trusted.to_vec()).client(peer, headers)
    }

    #[expect(clippy::panic, reason = "test failure path")]
    fn unreachable_ip(msg: &str) -> ! {
        panic!("{msg}")
    }

    #[test]
    fn an_untrusted_peer_is_the_client_whatever_it_says() {
        let client = proxies_client(
            ip("203.0.113.9"),
            Forwarded {
                x_forwarded_for: Some("10.0.0.1"),
                x_forwarded_proto: Some("https"),
                ..Forwarded::default()
            },
            &[net("127.0.0.0/8")],
        );
        assert_eq!(client.ip, ip("203.0.113.9"));
        assert_eq!(client.https, None);
        let bare = proxies_client(ip("203.0.113.9"), Forwarded::default(), &[]);
        assert_eq!(bare.ip, ip("203.0.113.9"));
    }

    #[test]
    fn a_trusted_peer_yields_the_rightmost_untrusted_hop() {
        let trusted = [net("127.0.0.0/8"), net("10.0.0.0/8")];
        let client = proxies_client(
            ip("127.0.0.1"),
            Forwarded {
                x_forwarded_for: Some("198.51.100.7, 203.0.113.9, 10.0.0.2"),
                x_forwarded_proto: Some("https"),
                ..Forwarded::default()
            },
            &trusted,
        );
        // 10.0.0.2 is a trusted proxy; 203.0.113.9 is the first address no
        // trusted proxy wrote; 198.51.100.7 is whatever the client claimed.
        assert_eq!(client.ip, ip("203.0.113.9"));
        assert_eq!(client.https, Some(true));
        let forwarded = proxies_client(
            ip("127.0.0.1"),
            Forwarded {
                rfc7239: Some("for=\"[2001:db8::1]:4711\";proto=http, for=10.0.0.3"),
                x_forwarded_for: Some("1.1.1.1"),
                ..Forwarded::default()
            },
            &trusted,
        );
        assert_eq!(forwarded.ip, ip("2001:db8::1"));
        assert_eq!(forwarded.https, Some(false));
    }

    #[test]
    fn a_chain_of_only_proxies_resolves_to_its_first_hop() {
        let trusted = [net("10.0.0.0/8")];
        let client = proxies_client(
            ip("10.0.0.1"),
            Forwarded {
                x_forwarded_for: Some("10.0.0.5, 10.0.0.2"),
                ..Forwarded::default()
            },
            &trusted,
        );
        assert_eq!(client.ip, ip("10.0.0.5"));
        let none = proxies_client(ip("10.0.0.1"), Forwarded::default(), &trusted);
        assert_eq!(none.ip, ip("10.0.0.1"));
    }

    #[test]
    fn a_header_that_does_not_parse_counts_as_absent() {
        let trusted = [net("10.0.0.0/8")];
        let client = proxies_client(
            ip("10.0.0.1"),
            Forwarded {
                x_forwarded_for: Some("not an address"),
                ..Forwarded::default()
            },
            &trusted,
        );
        assert_eq!(client.ip, ip("10.0.0.1"));
    }
}
