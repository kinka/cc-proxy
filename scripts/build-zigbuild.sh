#!/bin/bash
#
# Dual-arch build that never pulls a toolchain image.
#
# `scripts/build.sh` builds inside `rust:1.94-bookworm`, which is ~1.5GB. Behind a slow
# registry mirror that pull alone takes hours, and it has to happen twice (once per
# platform). This script instead cross-compiles on the host with cargo-zigbuild and then
# layers the binary onto an *existing* runtime image, so nothing over a few MB moves.
#
# The tradeoff is explicit: the base image's OS packages are NOT refreshed. Use
# `scripts/build.sh` when you want a fresh debian-slim + ca-certificates layer, and this
# one for iterating on the Rust code.
#
# Prerequisites (one time):
#   brew install zig
#   cargo install cargo-zigbuild
#   rustup target add x86_64-unknown-linux-musl aarch64-unknown-linux-musl
#
# Usage:
#   ./scripts/build-zigbuild.sh                       # local images only
#   VERSION=v0.3.0 REGISTRY=registry.example.com/ns \
#       ./scripts/build-zigbuild.sh --push            # + push arch tags and manifest

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE_NAME="${IMAGE_NAME:-cc-proxy}"
VERSION="${VERSION:-latest}"
REGISTRY="${REGISTRY:-}"
PUSH=false

# Runtime layer to reuse; it only supplies debian-slim + ca-certificates. Deliberately has
# no default: the obvious one would be this script's own output tag, which then grows a
# fresh binary layer on every run without ever saying so. Point these at a published tag
# (`.../cc-proxy:latest-arm64`) or at whatever `scripts/build.sh` last left locally.
BASE_ARM64="${BASE_ARM64:-}"
BASE_AMD64="${BASE_AMD64:-}"

for arg in "$@"; do
    case "$arg" in
        --push) PUSH=true ;;
        --no-push) PUSH=false ;;
        *) echo "Unknown argument: $arg" >&2; exit 1 ;;
    esac
done

if [ "$PUSH" = true ] && [ -z "$REGISTRY" ]; then
    echo "REGISTRY is required when --push is enabled" >&2
    exit 1
fi

FULL_IMAGE="$IMAGE_NAME"
if [ -n "$REGISTRY" ]; then
    FULL_IMAGE="$REGISTRY/$IMAGE_NAME"
fi

for tool in zig cargo-zigbuild; do
    command -v "$tool" >/dev/null 2>&1 || { echo "missing $tool — see the header of this script" >&2; exit 1; }
done

if [ -z "$BASE_ARM64" ] || [ -z "$BASE_AMD64" ]; then
    cat >&2 <<'MSG'
BASE_ARM64 and BASE_AMD64 are required: this script only swaps the binary, it does not
build a runtime layer. Both must already exist locally or be pullable, e.g.

    BASE_ARM64=cc-proxy:arm64 BASE_AMD64=cc-proxy:amd64 ./scripts/build-zigbuild.sh
    BASE_ARM64=<registry>/cc-proxy:latest-arm64 \
    BASE_AMD64=<registry>/cc-proxy:latest-amd64 ./scripts/build-zigbuild.sh
MSG
    exit 1
fi

STAGE_DIR="$(mktemp -d)"
trap 'rm -rf "$STAGE_DIR"' EXIT

# The binary is copied in, so the build context only needs the binary itself.
cat > "$STAGE_DIR/Dockerfile" <<'EOF'
ARG BASE
FROM ${BASE}
ARG BIN
COPY ${BIN} /usr/local/bin/cc-proxy
EOF

build_for_arch() {
    local arch="$1" target="$2" platform="$3" base="$4"

    echo "  [${arch}] cargo zigbuild --target ${target}"
    (cd "$ROOT_DIR" && cargo zigbuild --release --target "$target")

    cp "$ROOT_DIR/target/$target/release/$IMAGE_NAME" "$STAGE_DIR/$arch"

    echo "  [${arch}] docker build on ${base}"
    docker build \
        --platform "$platform" \
        --build-arg BASE="$base" \
        --build-arg BIN="$arch" \
        -t "$IMAGE_NAME:$arch" \
        "$STAGE_DIR"
}

echo "Building $IMAGE_NAME dual-arch images via cargo-zigbuild"
echo "Version: $VERSION"
echo "Runtime base: $BASE_ARM64 / $BASE_AMD64"

build_for_arch arm64 aarch64-unknown-linux-musl linux/arm64 "$BASE_ARM64"
build_for_arch amd64 x86_64-unknown-linux-musl linux/amd64 "$BASE_AMD64"

if [ "$PUSH" = true ]; then
    docker tag "$IMAGE_NAME:arm64" "$FULL_IMAGE:$VERSION-arm64"
    docker tag "$IMAGE_NAME:amd64" "$FULL_IMAGE:$VERSION-amd64"
    docker push "$FULL_IMAGE:$VERSION-arm64"
    docker push "$FULL_IMAGE:$VERSION-amd64"
    docker manifest rm "$FULL_IMAGE:$VERSION" >/dev/null 2>&1 || true
    docker manifest create "$FULL_IMAGE:$VERSION" \
        "$FULL_IMAGE:$VERSION-arm64" \
        "$FULL_IMAGE:$VERSION-amd64"
    docker manifest annotate "$FULL_IMAGE:$VERSION" "$FULL_IMAGE:$VERSION-arm64" --arch arm64
    docker manifest annotate "$FULL_IMAGE:$VERSION" "$FULL_IMAGE:$VERSION-amd64" --arch amd64
    docker manifest push "$FULL_IMAGE:$VERSION"
fi

echo
docker images "$IMAGE_NAME" --format "table {{.Repository}}:{{.Tag}}\t{{.Size}}\t{{.CreatedSince}}"
