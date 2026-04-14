#!/bin/bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE_NAME="${IMAGE_NAME:-cc-proxy}"
VERSION="${VERSION:-latest}"
SEQUENTIAL=true
PUSH=false
REGISTRY="${REGISTRY:-}"
CARGO_REGISTRY_INDEX="${CARGO_REGISTRY_INDEX:-sparse+https://rsproxy.cn/index/}"
CARGO_HTTP_TIMEOUT="${CARGO_HTTP_TIMEOUT:-600}"
CARGO_NET_RETRY="${CARGO_NET_RETRY:-10}"

for arg in "$@"; do
    case "$arg" in
        --parallel)
            SEQUENTIAL=false
            ;;
        --sequential)
            SEQUENTIAL=true
            ;;
        --push)
            PUSH=true
            ;;
        --no-push)
            PUSH=false
            ;;
        *)
            echo "Unknown argument: $arg" >&2
            exit 1
            ;;
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

build_for_arch() {
    local arch="$1"
    local platform="$2"

    local arch_label
    arch_label="$(printf '%s' "$arch" | tr '[:lower:]' '[:upper:]')"
    echo "  [${arch_label}] docker build --platform ${platform}"
    docker build \
        --platform "$platform" \
        --build-arg CARGO_HTTP_TIMEOUT="$CARGO_HTTP_TIMEOUT" \
        --build-arg CARGO_NET_RETRY="$CARGO_NET_RETRY" \
        --build-arg CARGO_REGISTRIES_CRATES_IO_INDEX="$CARGO_REGISTRY_INDEX" \
        -t "$IMAGE_NAME:$arch" \
        -f "$ROOT_DIR/Dockerfile" \
        "$ROOT_DIR"
}

echo "Building $IMAGE_NAME multi-arch images"
echo "Version: $VERSION"
echo "Platforms: linux/arm64, linux/amd64"
echo "Cargo index: $CARGO_REGISTRY_INDEX"

docker image rm "$IMAGE_NAME:arm64" "$IMAGE_NAME:amd64" "$IMAGE_NAME:latest" "$IMAGE_NAME:$VERSION" >/dev/null 2>&1 || true

if [ "$SEQUENTIAL" = true ]; then
    build_for_arch arm64 linux/arm64
    build_for_arch amd64 linux/amd64
else
    build_for_arch arm64 linux/arm64 &
    ARM64_PID=$!
    build_for_arch amd64 linux/amd64 &
    AMD64_PID=$!
    wait "$ARM64_PID"
    wait "$AMD64_PID"
fi

CURRENT_ARCH="$(uname -m)"
if [ "$CURRENT_ARCH" = "arm64" ] || [ "$CURRENT_ARCH" = "aarch64" ]; then
    docker tag "$IMAGE_NAME:arm64" "$IMAGE_NAME:latest"
elif [ "$CURRENT_ARCH" = "x86_64" ] || [ "$CURRENT_ARCH" = "amd64" ]; then
    docker tag "$IMAGE_NAME:amd64" "$IMAGE_NAME:latest"
fi

docker tag "$IMAGE_NAME:latest" "$IMAGE_NAME:$VERSION"

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
