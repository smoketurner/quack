//! Who a request came from when a proxy stands in front of `quack serve`.
//! The TCP peer is the one address a caller cannot choose, so the forwarded
//! addresses are read only when that peer is in `[server].trusted_proxies`,
//! and then from the right, stopping at the first address no trusted proxy
//! wrote: the model of nginx's `set_real_ip_from` and Caddy's
//! `trusted_proxies`. With no trusted range (the default) the peer is the
//! client, whatever the headers say.
//!
//! Only `X-Forwarded-For` is read, every line of it in order: it is what
//! nginx, `HAProxy`, Caddy, and Traefik write. A proxy that manages that
//! header passes an RFC 7239 `Forwarded` header from the client through
//! untouched, so reading `Forwarded` too would let any client name itself.

use std::net::IpAddr;

use forwarded_header_value::{ForwardedHeaderValue, ForwardedStanza};
use http::HeaderMap;
use ipnet::IpNet;
use serde::Deserialize;

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

    /// The client behind `peer`, for audit rows, the `Secure` cookie
    /// decision, and the rate limiter: the peer itself unless it is a
    /// trusted proxy, in which case the rightmost forwarded address that is
    /// not itself a trusted proxy. An entry that is not an address stops
    /// the walk at the last proxy, since no trusted proxy wrote it; a chain
    /// made only of trusted proxies resolves to its leftmost entry, the one
    /// the first proxy saw.
    #[must_use]
    pub fn client_ip(&self, peer: IpAddr, headers: &HeaderMap) -> IpAddr {
        let mut client = peer;
        if !self.contains(peer) {
            return client;
        }
        for hop in ForwardedFor::of(headers).0.into_iter().rev() {
            let Some(ip) = hop else {
                break;
            };
            client = ip;
            if !self.contains(ip) {
                break;
            }
        }
        client
    }
}

/// Every `X-Forwarded-For` entry a request carried, across all its lines,
/// in the order the proxies wrote them; `None` for an entry that is not an
/// address.
struct ForwardedFor(Vec<Option<IpAddr>>);

impl ForwardedFor {
    fn of(headers: &HeaderMap) -> Self {
        Self(
            headers
                .get_all("x-forwarded-for")
                .iter()
                .filter_map(|line| line.to_str().ok())
                .flat_map(|line| line.split(','))
                .map(|entry| {
                    ForwardedHeaderValue::from_x_forwarded_for(entry.trim())
                        .ok()
                        .and_then(|value| value.iter().find_map(ForwardedStanza::forwarded_for_ip))
                })
                .collect(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use http::HeaderValue;

    fn ip(s: &str) -> IpAddr {
        s.parse()
            .unwrap_or_else(|e: std::net::AddrParseError| unreachable_ip(&e.to_string()))
    }

    fn net(s: &str) -> IpNet {
        s.parse()
            .unwrap_or_else(|e: ipnet::AddrParseError| unreachable_ip(&e.to_string()))
    }

    /// Headers with each `(name, value)` appended, so a name may repeat.
    fn headers(lines: &[(&'static str, &'static str)]) -> HeaderMap {
        let mut map = HeaderMap::new();
        for (name, value) in lines {
            map.append(*name, HeaderValue::from_static(value));
        }
        map
    }

    fn client(peer: &str, lines: &[(&'static str, &'static str)], trusted: &[IpNet]) -> IpAddr {
        TrustedProxies::from(trusted.to_vec()).client_ip(ip(peer), &headers(lines))
    }

    #[expect(clippy::panic, reason = "test failure path")]
    fn unreachable_ip(msg: &str) -> ! {
        panic!("{msg}")
    }

    #[test]
    fn an_untrusted_peer_is_the_client_whatever_it_says() {
        let lines = [("x-forwarded-for", "10.0.0.1")];
        assert_eq!(
            client("203.0.113.9", &lines, &[net("127.0.0.0/8")]),
            ip("203.0.113.9")
        );
        assert_eq!(client("203.0.113.9", &[], &[]), ip("203.0.113.9"));
    }

    #[test]
    fn a_trusted_peer_yields_the_rightmost_untrusted_hop() {
        let trusted = [net("127.0.0.0/8"), net("10.0.0.0/8")];
        // 10.0.0.2 is a trusted proxy; 203.0.113.9 is the first address no
        // trusted proxy wrote; 198.51.100.7 is whatever the client claimed.
        let lines = [("x-forwarded-for", "198.51.100.7, 203.0.113.9, 10.0.0.2")];
        assert_eq!(client("127.0.0.1", &lines, &trusted), ip("203.0.113.9"));
        let v6 = [("x-forwarded-for", "2001:db8::1, 10.0.0.3")];
        assert_eq!(client("127.0.0.1", &v6, &trusted), ip("2001:db8::1"));
    }

    /// A client's own `Forwarded` header, which a proxy that manages
    /// `X-Forwarded-For` passes through, names nobody.
    #[test]
    fn a_forwarded_header_from_the_client_is_ignored() {
        let lines = [
            ("forwarded", "for=1.2.3.4"),
            ("x-forwarded-for", "203.0.113.9"),
        ];
        assert_eq!(
            client("10.0.0.1", &lines, &[net("10.0.0.0/8")]),
            ip("203.0.113.9")
        );
    }

    /// `HAProxy` appends a line of its own: every line counts, and the
    /// client's own line is the leftmost, not the one read.
    #[test]
    fn every_line_counts_and_the_last_is_the_proxys() {
        let lines = [
            ("x-forwarded-for", "1.2.3.4"),
            ("x-forwarded-for", "203.0.113.9"),
        ];
        assert_eq!(
            client("10.0.0.1", &lines, &[net("10.0.0.0/8")]),
            ip("203.0.113.9")
        );
    }

    /// A garbage entry the client wrote does not void the address the
    /// proxy appended after it; one where the proxy's own entry should be
    /// stops at the proxy.
    #[test]
    fn an_entry_that_is_no_address_stops_the_walk_where_it_stands() {
        let trusted = [net("10.0.0.0/8")];
        let after = [("x-forwarded-for", "not an address, 203.0.113.9")];
        assert_eq!(client("10.0.0.1", &after, &trusted), ip("203.0.113.9"));
        let last = [("x-forwarded-for", "203.0.113.9, not an address")];
        assert_eq!(client("10.0.0.1", &last, &trusted), ip("10.0.0.1"));
    }

    #[test]
    fn a_chain_of_only_proxies_resolves_to_its_first_hop() {
        let trusted = [net("10.0.0.0/8")];
        let lines = [("x-forwarded-for", "10.0.0.5, 10.0.0.2")];
        assert_eq!(client("10.0.0.1", &lines, &trusted), ip("10.0.0.5"));
        assert_eq!(client("10.0.0.1", &[], &trusted), ip("10.0.0.1"));
    }
}
