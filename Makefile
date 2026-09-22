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
EXAMPLE ?= nutrition
WORKSPACE ?= nutrition

# The crate whose templates Tailwind scans; its built CSS is committed.
SERVER_CRATE ?= quack

.PHONY: all build check clean fmt fmt-check lint test test-coverage test-mutants deny crypto-gates release-gates image hooks css-dev css-build run run-server demo-data help

all: build

##@ Build

build: ## Build the workspace (release)
	$(CARGO) build --release

check: ## Type-check the workspace
	$(CARGO) check --workspace --all-targets --all-features

clean: ## Remove the cargo target/ build artifacts
	$(CARGO) clean

##@ Quality

fmt: ## Format all code
	$(CARGO) fmt --all

fmt-check: ## Verify formatting without writing
	$(CARGO) fmt --all --check

lint: ## Run clippy with warnings denied
	$(CARGO) clippy --workspace --all-targets --all-features -- -D warnings

test: ## Run unit tests
	$(CARGO) test --workspace --all-features

bench: ## Run the criterion benchmarks (BENCH_CHUNKS=1000000 for the 1M-chunk retrieval point)
	QUACK_BENCH_CHUNKS=$(BENCH_CHUNKS) $(CARGO) bench --workspace

test-coverage: ## Generate an HTML coverage report (requires cargo-llvm-cov)
	$(CARGO) llvm-cov --workspace --html
	@echo "Coverage report: target/llvm-cov/html/index.html"

test-mutants: ## Run mutation testing (requires cargo-mutants)
	$(CARGO) mutants

eval: ## Run the evaluation harness: retrieval, induction, extraction, citations
	$(CARGO) run -p quack-core --example eval

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

##@ Run

run: build ## Build and run a binary: make run [BIN=quack] [ARGS="..."]
	$(CARGO) run --release --bin $(BIN) -- $(ARGS)

run-server: ## Serve the web UI and API locally without login: make run-server [BIND=127.0.0.1:8080]
	$(CARGO) run --bin quack -- serve --local --bind $(BIND)

demo-data: ## Load an example into a workspace: make demo-data [EXAMPLE=nutrition] [WORKSPACE=nutrition]
	$(CARGO) build --bin quack
	examples/$(EXAMPLE)/load.sh $(WORKSPACE)

##@ Help

help: ## Show this help
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make \033[36m<target>\033[0m\n"} /^[a-zA-Z_0-9-]+:.*?##/ { printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2 } /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) }' $(MAKEFILE_LIST)
