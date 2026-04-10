#!/bin/bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONTAINER_NAME="cc-proxy"
HOST_PORT="${HOST_PORT:-15721}"
CONTAINER_PORT="${CONTAINER_PORT:-15721}"
IMAGE_TAG="${IMAGE_TAG:-}"

if [ -z "${CONFIG_PATH:-}" ]; then
    for candidate in \
        "$ROOT_DIR/config/proxy.local.yaml" \
        "$ROOT_DIR/config/proxy.responses.local.yaml" \
        "$ROOT_DIR/config/proxy.internal-chat.local.yaml"
    do
        if [ -f "$candidate" ]; then
            CONFIG_PATH="$candidate"
            break
        fi
    done
fi

CONFIG_PATH="${CONFIG_PATH:-$ROOT_DIR/config/proxy.local.yaml}"

if [ -z "$IMAGE_TAG" ]; then
    CURRENT_ARCH="$(uname -m)"
    if [ "$CURRENT_ARCH" = "arm64" ] || [ "$CURRENT_ARCH" = "aarch64" ]; then
        IMAGE_TAG="cc-proxy:arm64"
    elif [ "$CURRENT_ARCH" = "x86_64" ] || [ "$CURRENT_ARCH" = "amd64" ]; then
        IMAGE_TAG="cc-proxy:amd64"
    else
        IMAGE_TAG="cc-proxy:latest"
    fi
fi

if [ ! -f "$CONFIG_PATH" ]; then
    echo "Config not found: $CONFIG_PATH" >&2
    echo "Create one from config/openai-chat.example.yaml or config/openai-responses.example.yaml" >&2
    exit 1
fi

docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true

RUN_ARGS=(
    -d
    --name "$CONTAINER_NAME"
    --restart unless-stopped
    -p "127.0.0.1:${HOST_PORT}:${CONTAINER_PORT}"
    -v "${CONFIG_PATH}:/app/config/proxy.yaml:ro"
    --add-host host.docker.internal:host-gateway
)

if [ -n "${RUST_LOG:-}" ]; then
    RUN_ARGS+=(-e "RUST_LOG=${RUST_LOG}")
fi

if [ -n "${HTTP_PROXY:-}" ]; then
    RUN_ARGS+=(-e "HTTP_PROXY=${HTTP_PROXY}")
fi

if [ -n "${HTTPS_PROXY:-}" ]; then
    RUN_ARGS+=(-e "HTTPS_PROXY=${HTTPS_PROXY}")
fi

if [ -n "${NO_PROXY:-}" ]; then
    RUN_ARGS+=(-e "NO_PROXY=${NO_PROXY}")
fi

docker run "${RUN_ARGS[@]}" "$IMAGE_TAG" >/dev/null

echo "Started ${CONTAINER_NAME}"
echo "Image: ${IMAGE_TAG}"
echo "Config: ${CONFIG_PATH}"
echo "Endpoint: http://127.0.0.1:${HOST_PORT}"
