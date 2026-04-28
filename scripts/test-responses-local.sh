#!/bin/bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_PATH="${CONFIG_PATH:-$ROOT_DIR/config/proxy.internal-chat.local.yaml}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-25721}"
MODEL="${MODEL:-external-qwen3.6-plus}"

if [ ! -f "$CONFIG_PATH" ]; then
    echo "Config not found: $CONFIG_PATH" >&2
    exit 1
fi

RESPONSE_FILE="$(mktemp)"
SERVER_LOG="$(mktemp)"
cleanup() {
    if [ -n "${SERVER_PID:-}" ] && kill -0 "$SERVER_PID" >/dev/null 2>&1; then
        kill "$SERVER_PID" >/dev/null 2>&1 || true
        wait "$SERVER_PID" >/dev/null 2>&1 || true
    fi
    rm -f "$RESPONSE_FILE" "$SERVER_LOG"
}
trap cleanup EXIT

echo "Starting cc-proxy without outbound proxy env..."
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy \
    NO_PROXY='localhost,127.0.0.1,.gf.com.cn,llm.smart-zone-dev.gf.com.cn' \
    RUST_LOG=info \
    cargo run -- --config "$CONFIG_PATH" >"$SERVER_LOG" 2>&1 &
SERVER_PID=$!

echo "Waiting for health endpoint..."
ready=false
for _ in $(seq 1 30); do
    if curl --noproxy '*' -fsS "http://$HOST:$PORT/health" >/dev/null 2>&1; then
        ready=true
        break
    fi
    sleep 1
done

if [ "$ready" != true ]; then
    echo "Health check failed" >&2
    cat "$SERVER_LOG" >&2
    exit 1
fi

echo "Calling /v1/responses..."
HTTP_CODE="$(curl --noproxy '*' -sS -o "$RESPONSE_FILE" -w '%{http_code}' \
    -X POST "http://$HOST:$PORT/v1/responses" \
    -H 'content-type: application/json' \
    -d "{\"model\":\"$MODEL\",\"input\":\"reply exactly OK\",\"max_output_tokens\":16}")"

echo "HTTP $HTTP_CODE"
cat "$RESPONSE_FILE"
echo

echo "Recent server log:"
tail -n 40 "$SERVER_LOG"

if [ "$HTTP_CODE" != "200" ]; then
    exit 1
fi
