//! Who a request came from when a proxy stands in front of `quack serve`.
//! The TCP peer is the one address a caller cannot choose, so the forwarded
//! headers are read only when that peer is in `[server].trusted_proxies`,
//! and then from the right, stopping at the first address no trusted proxy
//! wrote: the model of nginx's `set_real_ip_from` and Caddy's
//! `trusted_proxies`. With no trusted range (the default) the peer is the
//! client, whatever the headers say.

use std::net::IpAddr;

use ipnet::IpNet;

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

/// The client behind `peer`: the peer itself unless it is a trusted proxy,
/// in which case the rightmost forwarded address that is not itself a
/// trusted proxy. A forwarded chain made only of trusted proxies resolves
/// to its leftmost entry, the one the first proxy saw.
#[must_use]
pub fn client_addr(peer: IpAddr, headers: Forwarded<'_>, trusted: &[IpNet]) -> Client {
    let is_trusted = |ip: IpAddr| trusted.iter().any(|net| net.contains(&ip));
    if !is_trusted(peer) {
        return Client {
            ip: peer,
            https: None,
        };
    }
    let (chain, proto) = match headers.rfc7239 {
        Some(forwarded) => parse_forwarded(forwarded),
        None => (
            headers
                .x_forwarded_for
                .map(|v| v.split(',').filter_map(parse_ip).collect::<Vec<_>>())
                .unwrap_or_default(),
            headers.x_forwarded_proto.map(|p| {
                p.split(',')
                    .next_back()
                    .unwrap_or(p)
                    .trim()
                    .to_ascii_lowercase()
            }),
        ),
    };
    let https = proto.as_deref().map(|p| p == "https");
    let mut client = peer;
    for hop in chain.iter().rev() {
        client = *hop;
        if !is_trusted(*hop) {
            break;
        }
    }
    Client { ip: client, https }
}

/// The `for=` addresses of every element of a `Forwarded` header, in
/// order, and the last `proto=` it names.
fn parse_forwarded(value: &str) -> (Vec<IpAddr>, Option<String>) {
    let mut chain = Vec::new();
    let mut proto = None;
    for element in value.split(',') {
        for pair in element.split(';') {
            let Some((name, raw)) = pair.split_once('=') else {
                continue;
            };
            let raw = raw.trim().trim_matches('"');
            match name.trim().to_ascii_lowercase().as_str() {
                "for" => {
                    if let Some(ip) = parse_ip(raw) {
                        chain.push(ip);
                    }
                }
                "proto" => proto = Some(raw.to_ascii_lowercase()),
                _ => {}
            }
        }
    }
    (chain, proto)
}

/// An address as a forwarded header writes it: bare, with a port, or an
/// IPv6 in brackets; an obfuscated identifier (`_hidden`) is none.
fn parse_ip(raw: &str) -> Option<IpAddr> {
    let raw = raw.trim().trim_matches('"');
    if let Some(inner) = raw.strip_prefix('[') {
        let end = inner.find(']')?;
        return inner.get(..end)?.parse().ok();
    }
    if let Ok(ip) = raw.parse::<IpAddr>() {
        return Some(ip);
    }
    raw.rsplit_once(':')
        .and_then(|(host, _port)| host.parse::<std::net::Ipv4Addr>().ok())
        .map(IpAddr::V4)
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

    #[expect(clippy::panic, reason = "test failure path")]
    fn unreachable_ip(msg: &str) -> ! {
        panic!("{msg}")
    }

    #[test]
    fn an_untrusted_peer_is_the_client_whatever_it_says() {
        let client = client_addr(
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
        let bare = client_addr(ip("203.0.113.9"), Forwarded::default(), &[]);
        assert_eq!(bare.ip, ip("203.0.113.9"));
    }

    #[test]
    fn a_trusted_peer_yields_the_rightmost_untrusted_hop() {
        let trusted = [net("127.0.0.0/8"), net("10.0.0.0/8")];
        let client = client_addr(
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
        let forwarded = client_addr(
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
        let client = client_addr(
            ip("10.0.0.1"),
            Forwarded {
                x_forwarded_for: Some("10.0.0.5:1234, 10.0.0.2"),
                ..Forwarded::default()
            },
            &trusted,
        );
        assert_eq!(client.ip, ip("10.0.0.5"));
        let none = client_addr(ip("10.0.0.1"), Forwarded::default(), &trusted);
        assert_eq!(none.ip, ip("10.0.0.1"));
    }

    #[test]
    fn unparseable_hops_are_skipped() {
        assert_eq!(parse_ip("_hidden"), None);
        assert_eq!(parse_ip("192.0.2.1:8080"), Some(ip("192.0.2.1")));
        assert_eq!(parse_ip("\"[::1]\""), Some(ip("::1")));
    }
}
