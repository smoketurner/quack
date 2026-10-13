# Makefile for quack.

-include .env
export

CARGO ?= cargo

BIN ?= quack

# Address for `make run-server`.
BIND ?= 127.0.0.1:8080

# Largest retrieval benchmark size; the 1M-chunk point takes a quarter of an hour to fill.
BENCH_CHUNKS ?= 100000

# Example under examples/ that `make demo-data` loads, and the workspace it fills.
EXAMPLE ?= storms
WORKSPACE ?= storms

# The crate whose templates Tailwind scans; its built CSS is committed.
SERVER_CRATE ?= quack-server

# The desktop app's frontend (crates/quack-desktop/ui), built with pnpm.
UI_DIR ?= crates/quack-desktop/ui

# On Linux the desktop crate links WebKitGTK; without its development
# package the workspace-wide targets leave the crate out and say so. CI
# installs the package and never skips.
DESKTOP_EXCLUDE := $(shell if [ "$$(uname -s)" = Linux ] && ! pkg-config --exists webkit2gtk-4.1 2>/dev/null; then echo "--exclude quack-desktop"; fi)

.PHONY: all build check clean setup doc fmt fmt-check lint test test-coverage test-mutants deny crypto-gates release-gates eval eval-offline eval-scale build-debug image hooks css-dev css-build editor-build ui-install ui-types ui-build ui-lint ui-test desktop run run-server demo-data help

all: build

##@ Build

build: ## Build the workspace (release)
	$(CARGO) build

check: ## Type-check the workspace
	$(CARGO) check --workspace --all-targets --all-features

clean: ## Remove the cargo target/ build artifacts
	$(CARGO) clean

setup: ## Install the toolchain and tools a contributor needs (idempotent), then the git hooks
	./scripts/setup.sh
	prek install

doc: ## Build the API documentation with broken links denied (what the CI docs job runs)
	RUSTDOCFLAGS="-D warnings" $(CARGO) doc --workspace --no-deps

##@ Quality

fmt: ## Format all code
	$(CARGO) fmt --all

fmt-check: ## Verify formatting without writing
	$(CARGO) fmt --all --check

lint: ## Run clippy with warnings denied
	@if [ -n "$(DESKTOP_EXCLUDE)" ]; then echo "note: quack-desktop skipped (install libwebkit2gtk-4.1-dev to lint it)"; fi
	$(CARGO) clippy --workspace $(DESKTOP_EXCLUDE) --all-targets --all-features -- -D warnings

test: ## Run unit tests
	@if [ -n "$(DESKTOP_EXCLUDE)" ]; then echo "note: quack-desktop skipped (install libwebkit2gtk-4.1-dev to test it)"; fi
	$(CARGO) test --workspace $(DESKTOP_EXCLUDE) --all-features

bench: ## Run the criterion benchmarks (BENCH_CHUNKS=1000000 for the 1M-chunk retrieval point)
	QUACK_BENCH_CHUNKS=$(BENCH_CHUNKS) $(CARGO) bench --workspace

test-coverage: ## Generate an HTML coverage report (requires cargo-llvm-cov)
	$(CARGO) llvm-cov --workspace --html
	@echo "Coverage report: target/llvm-cov/html/index.html"

test-mutants: ## Run mutation testing (requires cargo-mutants)
	$(CARGO) mutants

eval: build-debug ## Run the evaluation suite against Ollama: the built quack binary driven as a person would, scored from its JSON (crates/quack-eval/README.md)
	$(CARGO) run -p quack-eval -- --ollama

eval-offline: build-debug ## Run the evaluation suite with no model: ingestion by format, keyword and hybrid retrieval, ontology induction
	$(CARGO) run -p quack-eval

# The SPC tornado database: every US tornado since 1950, one CSV of about 8 MB, public domain.
EVAL_SCALE_URL ?= https://www.spc.noaa.gov/wcm/data/1950-2023_all_tornadoes.csv
EVAL_CACHE ?= target/eval-cache

eval-scale: build-debug ## make eval, then the agent over a real 70,000-row table (downloads the SPC tornado database once into target/eval-cache)
	@mkdir -p $(EVAL_CACHE)
	@test -s $(EVAL_CACHE)/tornadoes.csv || { echo "downloading $(EVAL_SCALE_URL)"; curl -fsSL -o $(EVAL_CACHE)/tornadoes.csv.part "$(EVAL_SCALE_URL)" && mv $(EVAL_CACHE)/tornadoes.csv.part $(EVAL_CACHE)/tornadoes.csv; }
	$(CARGO) run -p quack-eval -- --ollama --scale $(EVAL_CACHE)/tornadoes.csv

build-debug: ## Build the quack binary in the dev profile (what the eval drives)
	$(CARGO) build --bin quack

crypto-gates: ## No ring or OpenSSL in the runtime dependency tree (needs no extra tools)
	@if $(CARGO) tree -i ring -e normal 2>/dev/null | grep -q ring; then echo "ring is in the runtime dependency tree"; exit 1; fi
	@if $(CARGO) tree -i openssl-sys -e normal 2>/dev/null | grep -q openssl; then echo "openssl-sys is in the runtime dependency tree"; exit 1; fi
	@echo "runtime tree is clean of ring and OpenSSL"

release-gates: crypto-gates ## The crypto gates a release must pass: crypto-gates plus cargo deny clean
	$(CARGO) deny check

image: ## Build the quack serve container image from source: make image [TAG=quack:dev]
	docker buildx build --tag $(or $(TAG),quack:dev) .

deny: ## Check advisories, licenses, bans, and sources
	$(CARGO) deny check

hooks: ## Install prek git hooks (pre-commit + pre-push)
	prek install

##@ UI assets

css-dev: ## Watch and rebuild Tailwind CSS for the web UI (commit the result)
	cd crates/$(SERVER_CRATE) && tailwindcss -i styles/input.css -o static/css/output.css --watch

css-build: ## Build minified Tailwind CSS for the web UI (commit the result)
	cd crates/$(SERVER_CRATE) && tailwindcss -i styles/input.css -o static/css/output.css --minify

##@ Desktop app (crates/quack-desktop)

ui-install: ## Install the desktop frontend's packages (pnpm, lockfile pinned)
	cd $(UI_DIR) && pnpm install --frozen-lockfile

ui-types: ## Regenerate the frontend's types from the commands' schema (commit the result)
	QUACK_WRITE_SCHEMA=1 $(CARGO) test --workspace --all-features --bin quack-desktop the_committed_schema_is_current
	cd $(UI_DIR) && pnpm run types
	@echo "written $(UI_DIR)/src/schema.json and src/schema.d.ts"

ui-build: ## Build the desktop frontend into crates/quack-desktop/ui/dist
	cd $(UI_DIR) && pnpm run build

ui-lint: ## Type-check, lint, and check the formatting of the desktop frontend
	cd $(UI_DIR) && pnpm run lint

ui-test: ## Run the desktop frontend's tests (Vitest)
	cd $(UI_DIR) && pnpm run test

desktop: ## Run the desktop app in development (needs tauri-cli: cargo install tauri-cli --version =2.12.2 --locked)
	cd crates/quack-desktop && cargo tauri dev

editor-build: ## Bundle the web console's SQL editor (CodeMirror) with pnpm (commit the result)
	cd crates/$(SERVER_CRATE)/editor && pnpm install --frozen-lockfile && pnpm run build

##@ Run

run: build ## Build and run a binary: make run [BIN=quack] [ARGS="..."]
	$(CARGO) run --bin $(BIN) -- $(ARGS)

run-server: ## Serve the web UI and API locally without login: make run-server [BIND=127.0.0.1:8080]
	$(CARGO) run --bin quack -- serve --local --bind $(BIND)

demo-data: ## Load an example into a workspace: make demo-data [EXAMPLE=storms] [WORKSPACE=storms]
	$(CARGO) build --bin quack
	examples/$(EXAMPLE)/load.sh $(WORKSPACE)

##@ Help

help: ## Show this help
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make \033[36m<target>\033[0m\n"} /^[a-zA-Z_0-9-]+:.*?##/ { printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2 } /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) }' $(MAKEFILE_LIST)
