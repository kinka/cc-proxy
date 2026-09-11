# cc-proxy

`cc-proxy` is a standalone Claude-compatible proxy for OpenAI-compatible upstreams.

It accepts Anthropic-style `POST /v1/messages` requests from tools such as Claude Code, forwards them to an OpenAI-compatible upstream, and translates the response back into Anthropic message format.

## Features

- Supports `POST /v1/messages`
- Supports `POST /claude/v1/messages`
- Supports `api_format: openai_chat`
- Supports `api_format: openai_responses`
- Supports non-streaming responses
- Supports streaming SSE conversion
- Supports model mapping
- Supports tool calls and image input
- Supports Docker-first deployment

## Non-goals

- Multi-provider failover
- Persistent state or database storage
- Desktop UI integration
- Host config takeover and restore
- Raw header ordering and casing preservation

## Layout

- `src/`: HTTP server, config loading, protocol transforms, SSE adapters
- `config/`: sanitized examples plus ignored local configs
- `scripts/build.sh`: local dual-arch Docker build helper
- `scripts/test.sh`: smoke test helper
- `run.sh`: quick local runner with config mount

## Config

Start from one of these examples:

- `config/openai-chat.example.yaml`
- `config/openai-responses.example.yaml`

Recommended local workflow:

```bash
cp config/openai-responses.example.yaml config/proxy.local.yaml
```

or

```bash
cp config/openai-chat.example.yaml config/proxy.local.yaml
```

Then edit:

- `upstream.base_url`
- `upstream.api_key`
- `upstream.api_format`
- `upstream.model_map`
- `upstream.extra_headers`
- `upstream.extra_body`

Local configs use `*.local.yaml` and are ignored by git.

### `extra_body`

Top-level fields merged into the upstream request body after translation. This is the
escape hatch for upstream-specific sampling knobs that have no Anthropic equivalent, so
clients never have to send them:

```yaml
upstream:
  extra_body:
    repetition_penalty: 1.1
```

Notes:

- It applies to **every model routed through that upstream**. Give a model its own
  `providers` entry when the knob should not be shared.
- Values override whatever the translation produced.
- Structural fields (`model`, `messages`, `input`, `tools`, `stream`, `max_tokens`,
  `max_completion_tokens`) are refused and logged, so a config typo cannot turn into an
  opaque upstream `400`.
- Unknown fields are forwarded as-is; an upstream that does not recognize a knob may
  either ignore it or reject the request, so verify against that upstream before relying
  on it.

## Local Run

```bash
cargo run -- --config ./config/proxy.local.yaml
```

Or:

```bash
CC_PROXY_CONFIG=./config/proxy.local.yaml cargo run
```

If no config path is provided, `cc-proxy` looks for:

1. `config/proxy.local.yaml`
2. `config/proxy.responses.local.yaml`
3. `config/proxy.internal-chat.local.yaml`
4. `config/proxy.example.yaml`

## Docker Build

Build the current platform image:

```bash
docker build -t cc-proxy-local .
```

Build local dual-arch images:

```bash
./scripts/build.sh
```

This creates:

- `cc-proxy:arm64`
- `cc-proxy:amd64`

The build script defaults to `sparse+https://rsproxy.cn/index/` for Cargo. Override it if needed:

```bash
CARGO_REGISTRY_INDEX="sparse+https://your-mirror.example.com/index/" ./scripts/build.sh
```

## Quick Run

Use the bundled runner:

```bash
./run.sh
```

Useful overrides:

```bash
CONFIG_PATH=./config/proxy.local.yaml ./run.sh
```

```bash
HOST_PORT=25721 IMAGE_TAG=cc-proxy:arm64 ./run.sh
```

```bash
HTTP_PROXY=http://host.docker.internal:8028 ./run.sh
```

`run.sh` always starts the container as `cc-proxy`.

## Smoke Test

```bash
./scripts/test.sh
```

Or target a specific config:

```bash
CONFIG_PATH=./config/proxy.local.yaml ./scripts/test.sh
```

To validate a real upstream request with `config/gpt.yaml`:

```bash
CONFIG_PATH=./config/gpt.yaml ./scripts/test.sh
```

If you only want the local `/health` check:

```bash
CONFIG_PATH=./config/gpt.yaml SKIP_REAL_REQUEST=1 ./scripts/test.sh
```

## Claude Code

Point Claude Code at the local proxy:

```bash
export ANTHROPIC_BASE_URL=http://127.0.0.1:25721
export ANTHROPIC_AUTH_TOKEN=PROXY_MANAGED
```

`ANTHROPIC_AUTH_TOKEN` is only a placeholder for Claude-side clients. The upstream bearer token comes from the proxy config.

## Notes

- Some OpenAI-compatible gateways return slightly different SSE or response shapes. This project already handles several common variants, but provider-specific adjustments may still be needed.
- Debug logging can be enabled with `RUST_LOG=debug`.
- Do not commit local config files containing real credentials.
- **Reasoning must round-trip.** Anthropic clients send prior reasoning back as `thinking`
  blocks on the assistant turn, and the request conversion restores them as
  `reasoning_content`. This is not optional bookkeeping: DeepSeek rejects the entire request
  with `The 'reasoning_content' in the thinking mode must be passed back to the API` as soon
  as an assistant turn carries `tool_calls` without it, so dropping the block breaks every
  multi-turn tool-using session on those models. The failure is all-or-nothing and behaves
  identically under `stream: true` and `stream: false`. When the client sends no thinking
  block at all but the turn does carry `tool_calls`, the field is emitted as an empty string
  — clients legitimately lose reasoning (pi downgrades a signature-less thinking block to
  plain text after an aborted stream), and an empty value satisfies the check instead of
  failing the request.
