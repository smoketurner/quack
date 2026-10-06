# Security Policy

## Reporting a Vulnerability

Please report security vulnerabilities responsibly. **Do not** open a public GitHub issue.

Use GitHub's private vulnerability reporting on this repository (Security tab, "Report a
vulnerability"). The machine-readable form of this policy (RFC 9116) is
[`.well-known/security.txt`](.well-known/security.txt); publish it at the project's web
origin, not from `quack serve`, where the contact would be the operator's.

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
  `cargo-deny`, Dependabot, and dependency-review in CI. `cargo deny check` also runs every
  Monday on its own (`.github/workflows/advisories.yml`), so an advisory published while
  the repository is quiet opens one issue within a week.
- **Parsers are fuzzed** — every parser that takes an uploaded file (PDF, Markdown, text,
  HTML, DOCX, PPTX, XLSX) and the chunker run under libFuzzer for ten minutes each night
  (`fuzz/`, `.github/workflows/fuzz.yml`); a crash's input is uploaded with the run.
- **Admin self-grants are marked** — an administrator who gives themself a role in a
  workspace they are not a member of must state a reason, and the access audit records the
  grant as `break_glass` rather than an ordinary membership change.
