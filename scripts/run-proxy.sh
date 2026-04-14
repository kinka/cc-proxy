#!/bin/bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONTAINER_NAME="cc-proxy"
HOST_PORT="${HOST_PORT:-25721}"
CONTAINER_PORT="${CONTAINER_PORT:-25721}"
IMAGE_TAG="${IMAGE_TAG:-}"
RUST_LOG="${RUST_LOG:-info}"
SKIP_REAL_REQUEST="${SKIP_REAL_REQUEST:-0}"

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
    -p "${HOST_PORT}:${CONTAINER_PORT}"
    -v "${CONFIG_PATH}:/app/config/proxy.yaml:ro"
)

DOCKER_VERSION_TEXT="$(docker version 2>/dev/null || true)"
if [[ "$DOCKER_VERSION_TEXT" != *"Podman Engine"* ]]; then
    RUN_ARGS+=(--add-host host.docker.internal:host-gateway)
fi

RUN_ARGS+=( -e "RUST_LOG=${RUST_LOG}" )

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

REQUEST_URL="http://127.0.0.1:${HOST_PORT}/v1/messages"
REQUEST_PAYLOAD='{
  "model": "claude-sonnet-4-20250514",
  "max_tokens": 16,
  "messages": [
    {
      "role": "user",
      "content": "reply with exactly OK"
    }
  ]
}'

echo "Waiting for service to be ready..."
ready=false
for _ in $(seq 1 20); do
    if curl -fsS "http://127.0.0.1:${HOST_PORT}/health" >/dev/null 2>&1; then
        ready=true
        break
    fi
    sleep 1
done

if [ "$ready" != true ]; then
    echo "Health check failed" >&2
    docker logs --tail 20 "$CONTAINER_NAME" 2>&1 || true
    exit 1
fi

echo "Service ready"

if [ "$SKIP_REAL_REQUEST" != "1" ]; then
    echo "Verifying /v1/messages endpoint..."
    response_file="$(mktemp)"
    http_code="$(curl -sS -o "$response_file" -w '%{http_code}' \
        -X POST "$REQUEST_URL" \
        -H 'content-type: application/json' \
        -d "$REQUEST_PAYLOAD" 2>/dev/null || echo "000")"
    response_body="$(cat "$response_file" 2>/dev/null || echo "")"
    rm -f "$response_file"

    if [ "$http_code" = "200" ] && printf '%s' "$response_body" | grep -q '"id"' 2>/dev/null; then
        echo "✓ /v1/messages endpoint verified"
    else
        echo "⚠ /v1/messages verification failed (HTTP $http_code)" >&2
        if [ -n "$response_body" ]; then
            echo "Response: ${response_body:0:200}..." >&2
        fi
    fi
fi

echo
echo "Recent logs:"
docker logs -f "$CONTAINER_NAME" 2>&1 || true
