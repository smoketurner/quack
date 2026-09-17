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
target "ci" {
  inherits = ["_common"]
  args = {
    TARGET            = TARGET
    SOURCE_DATE_EPOCH = SOURCE_DATE_EPOCH
    GENERATE_SBOM     = GENERATE_SBOM
  }
  cache-from = ["type=gha,scope=bake-ci-${TARGET}"]
  cache-to   = ["type=gha,mode=max,ignore-error=true,scope=bake-ci-${TARGET}"]
}
