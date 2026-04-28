#!/bin/bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_PATH="${CONFIG_PATH:-$ROOT_DIR/config/proxy.internal-chat.local.yaml}"

exec env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy \
    NO_PROXY='localhost,127.0.0.1,.gf.com.cn,llm.smart-zone-dev.gf.com.cn' \
    RUST_LOG="${RUST_LOG:-info}" \
    cargo run -- --config "$CONFIG_PATH"
