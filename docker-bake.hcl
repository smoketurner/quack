# Static musl builds of `quack` (Dockerfile.build). One invocation per
# architecture, TARGET set to the musl triple:
#   TARGET=aarch64-unknown-linux-musl docker buildx bake ci
# The binary lands under ./target/<TARGET>/release/quack (output type=local).

variable "TARGET" {
  default = "x86_64-unknown-linux-musl"
}

variable "SOURCE_DATE_EPOCH" {
  default = "0"
}

variable "GENERATE_SBOM" {
  default = "false"
}

group "default" {
  targets = ["ci"]
}

target "_common" {
  dockerfile = "Dockerfile.build"
  context    = "."
  output     = ["type=local,dest=."]
}

# The release build: reproducible, with a CycloneDX SBOM beside the binary.
#
# No `type=gha` cache. This target runs only from the release workflow, and an
# Actions cache written on a tag ref can only be read back by that same tag, so
# the entry every release uploaded was never readable by the next one. Writing
# it was worse than useless: `mode=max` pushes the whole cooked musl dependency
# tree, several GB per architecture, into the repository's 10 GB cache budget,
# where it evicts the CI entries that keep ordinary pushes fast (a
# `workflow_dispatch` run on main shares main's cache scope, so it could evict
# them directly). Releases are rare and reproducibility matters more there than
# minutes; local `docker buildx bake ci` still uses the daemon's own cache.
target "ci" {
  inherits = ["_common"]
  args = {
    TARGET            = TARGET
    SOURCE_DATE_EPOCH = SOURCE_DATE_EPOCH
    GENERATE_SBOM     = GENERATE_SBOM
  }
}
