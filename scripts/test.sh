#!/bin/bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE_TAG="${IMAGE_TAG:-cc-proxy:latest}"
PORT="${PORT:-15721}"
CONTAINER_NAME="${CONTAINER_NAME:-cc-proxy-smoke}"

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

cleanup() {
    docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
}

dump_logs() {
    docker logs "$CONTAINER_NAME" 2>/dev/null || true
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
    if curl -fsS "http://127.0.0.1:${PORT}/health" >/tmp/cc-proxy-health.json 2>/dev/null; then
        ready=true
        break
    fi
    sleep 1
done

if [ "$ready" != true ]; then
    echo "Health check failed for $IMAGE_TAG on port $PORT" >&2
    dump_logs >&2
    exit 1
fi

cat /tmp/cc-proxy-health.json
grep -q '"status":"ok"' /tmp/cc-proxy-health.json
grep -q '"service":"cc-proxy"' /tmp/cc-proxy-health.json

echo
echo "Health check passed for $IMAGE_TAG"
