#!/bin/bash
# Claude Code SessionStart hook: prepare a fresh container to build and gate this repo.
#
# Every step is idempotent and skips silently when already satisfied or when the
# network policy blocks a download. The hook must never fail the session, so it
# always exits 0. Linux-only; a no-op elsewhere.

[ "$(uname)" = "Linux" ] || exit 0

# The contributor setup script installs the toolchain and tools (idempotent;
# `make setup` runs the same). Quiet, and never fatal here: the session must
# start even when a download is blocked.
"$(dirname "$0")/../../scripts/setup.sh" >/dev/null 2>&1 || true

# Tailwind from the npm registry when the GitHub download was blocked (the
# registry stays reachable when github.com egress is restricted).
if ! command -v tailwindcss >/dev/null 2>&1 && command -v npm >/dev/null 2>&1; then
  npm install -g --silent @tailwindcss/cli >/dev/null 2>&1
  # Unlike the standalone binary, the npm CLI resolves `@import "tailwindcss"`
  # by walking node_modules up from the input CSS file, so the package must
  # also be installed. Put it in the checkout's parent directory: that is on
  # the resolution path but keeps the repo working tree clean (--no-save
  # writes no package.json there either).
  (cd "$(dirname "$PWD")" && npm install --no-save --silent tailwindcss >/dev/null 2>&1)
fi
if [ -x "$HOME/.local/bin/prek" ] && ! command -v prek >/dev/null 2>&1; then
  sudo ln -sf "$HOME/.local/bin/prek" /usr/local/bin/prek >/dev/null 2>&1
fi

# Claude remote-execution containers only (detected by the agent-proxy config
# directory). These tweaks must not touch local developer machines.
if [ -d /root/.ccr ]; then
  # Incremental compilation roughly doubles target/ (~30 GB vs ~14 GB for a full
  # --all-features cycle) and has little value in a container that starts cold,
  # while the session's disk allowance is fixed.
  cargo_config="$HOME/.cargo/config.toml"
  if grep -qsE '^[[:space:]]*incremental[[:space:]]*=' "$cargo_config"; then
    # Flip any existing setting in place; appending a second [build] table
    # would be invalid TOML.
    sed -i -E 's/^([[:space:]]*incremental[[:space:]]*=[[:space:]]*).*/\1false/' "$cargo_config"
  else
    mkdir -p "$HOME/.cargo"
    printf '\n[build]\nincremental = false\n' >>"$cargo_config"
  fi

  # git uses one program (gpg.ssh.program) for both signing and verification, but
  # the provisioned signer implements only "-Y sign", so locally every signed
  # commit reports as unverifiable (%G? = N/E) and the stop hook flags good
  # commits. Route verification subcommands to the real ssh-keygen instead; the
  # allowed-signers file is already provisioned.
  # Read config unscoped: the signer program is global but the allowed-signers
  # file is provisioned in the repo-local .git/config (cwd is the project root
  # when SessionStart hooks run).
  shim=/usr/local/bin/git-ssh-sign-shim
  sign_prog=$(git config --get gpg.ssh.program 2>/dev/null)
  signers=$(git config --get gpg.ssh.allowedSignersFile 2>/dev/null)
  if [ "$(git config --get gpg.format 2>/dev/null)" = "ssh" ] &&
    { [ "$sign_prog" = "/tmp/code-sign" ] || [ "$sign_prog" = "$shim" ]; } &&
    [ -f "$signers" ] && [ -e /tmp/code-sign ] &&
    command -v ssh-keygen >/dev/null 2>&1; then
    cat >"$shim" <<'EOF'
#!/bin/sh
# Route git SSH signing to the provisioned signer and everything else
# (-Y verify, -Y find-principals, ...) to the real ssh-keygen.
if [ "$1" = "-Y" ] && [ "$2" = "sign" ]; then
  exec /tmp/code-sign "$@"
fi
exec ssh-keygen "$@"
EOF
    chmod 755 "$shim"
    git config --global gpg.ssh.program "$shim"
    # The provisioned value may live in repo-local .git/config, which overrides
    # the global write above; if the effective value still isn't the shim, set
    # it in the local scope too (.git/config is never committed).
    if [ "$(git config --get gpg.ssh.program 2>/dev/null)" != "$shim" ]; then
      git config --local gpg.ssh.program "$shim"
    fi
  fi
fi

exit 0
