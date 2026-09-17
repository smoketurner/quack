# Security Policy

## Reporting a Vulnerability

Please report security vulnerabilities responsibly. **Do not** open a public GitHub issue.

Use GitHub's private vulnerability reporting on this repository (Security tab, "Report a
vulnerability").

Include where possible:

- A description of the vulnerability and its impact
- Steps to reproduce
- Affected versions, components, or commit

## Response

- Acknowledgment within 48 hours of receipt
- Initial assessment within 5 business days
- Fix timeline based on severity; critical issues are prioritized immediately

## Security principles

- **No custom cryptography** — audited libraries only (aws-lc-rs, rustls). See
  [docs/crypto.md](docs/crypto.md).
- **No secrets in the repo** — provider keys come from environment variables named in
  `config.toml`, OAuth tokens live in an encrypted cache keyed from the OS keychain, and an
  import URL's password is used once and never stored.
- **Dependencies pinned and scanned** — exact versions in `Cargo.toml`, enforced by
  `cargo-deny`, Dependabot, and dependency-review in CI.
