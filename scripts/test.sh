#!/bin/bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PORT="${PORT:-25721}"
CONTAINER_NAME="${CONTAINER_NAME:-cc-proxy-smoke}"
SKIP_REAL_REQUEST="${SKIP_REAL_REQUEST:-0}"

if [ -z "${IMAGE_TAG:-}" ]; then
    CURRENT_ARCH="$(uname -m)"
    if [ "$CURRENT_ARCH" = "arm64" ] || [ "$CURRENT_ARCH" = "aarch64" ]; then
        IMAGE_TAG="cc-proxy:arm64"
    elif [ "$CURRENT_ARCH" = "x86_64" ] || [ "$CURRENT_ARCH" = "amd64" ]; then
        IMAGE_TAG="cc-proxy:amd64"
    else
        IMAGE_TAG="cc-proxy:latest"
    fi
fi

if [ -z "${CONFIG_PATH:-}" ]; then
    for candidate in \
        "$ROOT_DIR/config/proxy.local.yaml" \
        "$ROOT_DIR/config/proxy.responses.local.yaml" \
        "$ROOT_DIR/config/proxy.internal-chat.local.yaml" \
        "$ROOT_DIR/config/proxy.example.yaml"
    do
        if [ -f "$candidate" ]; then
            CONFIG_PATH="$candidate"
            break
        fi
    done
fi

CONFIG_PATH="${CONFIG_PATH:-$ROOT_DIR/config/proxy.local.yaml}"
REQUEST_URL="http://127.0.0.1:${PORT}/v1/messages"
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

cleanup() {
    docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
}

dump_logs() {
    docker logs "$CONTAINER_NAME" 2>/dev/null || true
}

fail_with_logs() {
    echo "$1" >&2
    dump_logs >&2
    exit 1
}

trap cleanup EXIT

if [ ! -f "$CONFIG_PATH" ]; then
    echo "Config not found: $CONFIG_PATH" >&2
    exit 1
fi

cleanup

docker run -d \
    --name "$CONTAINER_NAME" \
    -p "127.0.0.1:${PORT}:15721" \
    -v "$CONFIG_PATH:/app/config/proxy.yaml:ro" \
    "$IMAGE_TAG" >/dev/null

ready=false
for _ in $(seq 1 20); do
    if curl -fsS "$REQUEST_URL" >/dev/null 2>&1; then
        :
    fi
    if health_response="$(curl -fsS "http://127.0.0.1:${PORT}/health" 2>/dev/null)"; then
        ready=true
        break
    fi
    sleep 1
done

if [ "$ready" != true ]; then
    fail_with_logs "Health check failed for $IMAGE_TAG on port $PORT"
fi

printf '%s' "$health_response"
printf '\n'
printf '%s' "$health_response" | grep -q '"status":"ok"'
printf '%s' "$health_response" | grep -q '"service":"cc-proxy"'

echo

echo "Health check passed for $IMAGE_TAG"

if [ "$SKIP_REAL_REQUEST" = "1" ]; then
    echo "Skipped real /v1/messages check"
    exit 0
fi

response_file="$(mktemp)"
http_code="$(curl -sS -o "$response_file" -w '%{http_code}' \
    -X POST "$REQUEST_URL" \
    -H 'content-type: application/json' \
    -d "$REQUEST_PAYLOAD")"
response_body="$(cat "$response_file")"
rm -f "$response_file"

if [ "$http_code" != "200" ]; then
    echo "Real request failed: $REQUEST_URL" >&2
    echo "Config: $CONFIG_PATH" >&2
    echo "HTTP status: $http_code" >&2
    echo "Response body: $response_body" >&2
    fail_with_logs "Health check passed, but real message request failed"
fi

printf '%s' "$response_body" | grep -q '"id"' || fail_with_logs "Unexpected /v1/messages response: missing id"
printf '%s' "$response_body" | grep -q '"model"' || fail_with_logs "Unexpected /v1/messages response: missing model"
printf '%s' "$response_body" | grep -q '"role":"assistant"' || fail_with_logs "Unexpected /v1/messages response: missing assistant role"
printf '%s' "$response_body" | grep -q '"content":\[' || fail_with_logs "Unexpected /v1/messages response: missing content array"
printf '%s' "$response_body" | grep -q '"text"' || fail_with_logs "Unexpected /v1/messages response: missing text content"

echo "Real /v1/messages check passed for $IMAGE_TAG"
