mod config;
mod sse;
mod streaming;
mod streaming_responses;
mod transform;
mod transform_responses;

use crate::config::{ApiFormat, ProxyConfig};
use anyhow::{Context, Result};
use axum::{
    body::Body,
    extract::{Request, State},
    http::{HeaderMap, HeaderValue, StatusCode},
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use bytes::Bytes;
use futures::{Stream, StreamExt};
use http_body_util::BodyExt;
use reqwest::Client;
use serde_json::{json, Value};
use std::{
    net::SocketAddr,
    path::{Path, PathBuf},
    pin::Pin,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
    time::{Duration, Instant},
};

static NEXT_REQUEST_ID: AtomicU64 = AtomicU64::new(1);

#[derive(Clone)]
pub(crate) struct RequestLogContext {
    pub(crate) req_id: u64,
    pub(crate) path: String,
    pub(crate) api_format: &'static str,
    pub(crate) started_at: Instant,
}

#[derive(Debug, Default)]
struct ResponseLogSummary {
    response_id: Option<String>,
    response_model: Option<String>,
    stop_reason: Option<String>,
    input_tokens: u64,
    output_tokens: u64,
    cache_read_input_tokens: Option<u64>,
    cache_creation_input_tokens: Option<u64>,
    has_tool_use: bool,
    has_thinking: bool,
}

#[derive(Clone)]
struct AppState {
    config: Arc<ProxyConfig>,
    client: Client,
}

#[tokio::main]
async fn main() -> Result<()> {
    env_logger::init();

    let config_path = resolve_config_path()?;
    let config = Arc::new(ProxyConfig::load(&config_path)?);
    let client = Client::builder()
        .timeout(Duration::from_secs(config.upstream.timeout_secs))
        .connect_timeout(Duration::from_secs(30))
        .tcp_keepalive(Duration::from_secs(60))
        .no_gzip()
        .no_brotli()
        .no_deflate()
        .build()
        .context("failed to build reqwest client")?;

    let state = AppState { config, client };

    let app = Router::new()
        .route("/health", get(health))
        .route("/v1/messages", post(handle_messages))
        .route("/claude/v1/messages", post(handle_messages))
        .with_state(state.clone());

    let addr: SocketAddr = format!(
        "{}:{}",
        state.config.listen.host, state.config.listen.port
    )
    .parse()
    .context("invalid listen address")?;

    log::info!(
        "startup addr={} config_path={} api_format={} upstream_base_url={} timeout_secs={} model_map_count={} extra_headers_count={} http_proxy_set={} https_proxy_set={} no_proxy_set={}",
        addr,
        config_path.display(),
        api_format_name(state.config.upstream.api_format),
        state.config.upstream.base_url,
        state.config.upstream.timeout_secs,
        state.config.upstream.model_map.len(),
        state.config.upstream.extra_headers.len(),
        env_var_is_set("HTTP_PROXY") || env_var_is_set("http_proxy"),
        env_var_is_set("HTTPS_PROXY") || env_var_is_set("https_proxy"),
        env_var_is_set("NO_PROXY") || env_var_is_set("no_proxy"),
    );
    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .context("failed to bind listener")?;
    axum::serve(listener, app).await.context("server failed")?;

    Ok(())
}

async fn health() -> impl IntoResponse {
    Json(json!({
        "status": "ok",
        "service": "cc-proxy",
    }))
}

async fn handle_messages(
    State(state): State<AppState>,
    request: Request,
) -> Result<Response, Response> {
    let (parts, body) = request.into_parts();
    let path = parts.uri.path().to_string();
    let req_id = NEXT_REQUEST_ID.fetch_add(1, Ordering::Relaxed);
    let started_at = Instant::now();
    let body_bytes = body
        .collect()
        .await
        .map_err(|err| internal_error_with_context(req_id, &path, "read_request_body", None, None, started_at, err))?
        .to_bytes();
    let request_body: Value = serde_json::from_slice(&body_bytes)
        .map_err(bad_request)?;
    let is_stream = request_body
        .get("stream")
        .and_then(|value| value.as_bool())
        .unwrap_or(false);
    let requested_model = request_body
        .get("model")
        .and_then(|value| value.as_str())
        .map(str::to_string);
    let message_count = request_body
        .get("messages")
        .and_then(|value| value.as_array())
        .map_or(0, |messages| messages.len());
    let tool_count = request_body
        .get("tools")
        .and_then(|value| value.as_array())
        .map_or(0, |tools| tools.len());

    let mapped_body = apply_model_mapping(request_body, &state.config.upstream.model_map);
    let mapped_model = mapped_body
        .get("model")
        .and_then(|value| value.as_str())
        .map(str::to_string);
    let api_format = state.config.upstream.api_format;
    let api_format_name = api_format_name(api_format);
    let request_context = RequestLogContext {
        req_id,
        path: path.clone(),
        api_format: api_format_name,
        started_at,
    };

    log_request_start(
        req_id,
        &path,
        is_stream,
        api_format_name,
        requested_model.as_deref(),
        mapped_model.as_deref(),
        message_count,
        tool_count,
    );

    let transformed_body = match api_format {
        ApiFormat::OpenAiChat => transform::anthropic_to_openai(
            mapped_body,
            state.config.upstream.prompt_cache_key.as_deref(),
        ),
        ApiFormat::OpenAiResponses => transform_responses::anthropic_to_responses(
            mapped_body,
            state.config.upstream.prompt_cache_key.as_deref(),
        ),
    }
    .map_err(|err| internal_error_with_context(req_id, &path, "transform_request", Some(api_format_name), Some(is_stream), started_at, err))?;

    let endpoint = match api_format {
        ApiFormat::OpenAiChat => "/chat/completions",
        ApiFormat::OpenAiResponses => "/responses",
    };
    let upstream_url = build_upstream_url(&state.config.upstream.base_url, endpoint);

    let payload = serde_json::to_vec(&transformed_body)
        .map_err(|err| internal_error_with_context(req_id, &path, "serialize_upstream_request", Some(api_format_name), Some(is_stream), started_at, err))?;
    let mut upstream_request = state
        .client
        .post(&upstream_url)
        .header("authorization", format!("Bearer {}", state.config.upstream.api_key))
        .header("content-type", "application/json")
        .header("accept-encoding", "identity")
        .body(payload);

    for (name, value) in &state.config.upstream.extra_headers {
        upstream_request = upstream_request.header(name, value);
    }

    if let Some(user_agent) = parts
        .headers
        .get("user-agent")
        .and_then(|value| value.to_str().ok())
    {
        upstream_request = upstream_request.header("user-agent", user_agent);
    }

    let upstream_response = upstream_request
        .send()
        .await
        .map_err(|err| upstream_transport_error_with_context(req_id, &path, api_format_name, is_stream, started_at, err))?;
    let upstream_status = upstream_response.status();

    if !upstream_status.is_success() {
        let body = upstream_response.text().await.unwrap_or_default();
        let message = extract_error_message(&body);
        log::warn!(
            "request.upstream_error req_id={} path={} stream={} api_format={} upstream_status={} latency_ms={} error_message={}",
            req_id,
            path,
            is_stream,
            api_format_name,
            upstream_status.as_u16(),
            started_at.elapsed().as_millis(),
            sanitize_error_message(&message),
        );
        return Err(anthropic_error_response(upstream_status, &message));
    }

    if is_stream {
        let stream = upstream_response
            .bytes_stream()
            .map(|chunk| chunk.map_err(|err| std::io::Error::other(err.to_string())));

        let anthropic_stream: Pin<
            Box<dyn Stream<Item = Result<Bytes, std::io::Error>> + Send>,
        > = match api_format {
            ApiFormat::OpenAiChat => Box::pin(streaming::create_anthropic_sse_stream(stream, request_context.clone())),
            ApiFormat::OpenAiResponses => Box::pin(
                streaming_responses::create_anthropic_sse_stream_from_responses(stream, request_context.clone()),
            ),
        };

        let mut headers = HeaderMap::new();
        headers.insert("content-type", HeaderValue::from_static("text/event-stream"));
        headers.insert("cache-control", HeaderValue::from_static("no-cache"));
        headers.insert("connection", HeaderValue::from_static("keep-alive"));
        return Ok((headers, Body::from_stream(anthropic_stream)).into_response());
    }

    let upstream_body = upstream_response
        .bytes()
        .await
        .map_err(|err| upstream_transport_error_with_context(req_id, &path, api_format_name, is_stream, started_at, err))?;
    let parsed: Value = serde_json::from_slice(&upstream_body)
        .map_err(|err| internal_error_with_context(req_id, &path, "parse_upstream_json", Some(api_format_name), Some(is_stream), started_at, err))?;
    let anthropic_body = match api_format {
        ApiFormat::OpenAiChat => transform::openai_to_anthropic(parsed),
        ApiFormat::OpenAiResponses => transform_responses::responses_to_anthropic(parsed),
    }
    .map_err(|err| internal_error_with_context(req_id, &path, "transform_upstream_response", Some(api_format_name), Some(is_stream), started_at, err))?;

    log_request_done(&request_context, upstream_status, &summarize_anthropic_response(&anthropic_body), false);

    let mut headers = HeaderMap::new();
    headers.insert("content-type", HeaderValue::from_static("application/json"));
    Ok((StatusCode::OK, headers, Json(anthropic_body)).into_response())
}

fn api_format_name(api_format: ApiFormat) -> &'static str {
    match api_format {
        ApiFormat::OpenAiChat => "openai_chat",
        ApiFormat::OpenAiResponses => "openai_responses",
    }
}

fn env_var_is_set(name: &str) -> bool {
    std::env::var_os(name).is_some()
}

fn log_request_start(
    req_id: u64,
    path: &str,
    is_stream: bool,
    api_format: &str,
    requested_model: Option<&str>,
    mapped_model: Option<&str>,
    message_count: usize,
    tool_count: usize,
) {
    match (requested_model, mapped_model) {
        (Some(requested_model), Some(mapped_model)) if requested_model != mapped_model => {
            log::info!(
                "request.start req_id={} path={} stream={} api_format={} requested_model={} mapped_model={} message_count={} tool_count={}",
                req_id,
                path,
                is_stream,
                api_format,
                requested_model,
                mapped_model,
                message_count,
                tool_count,
            );
        }
        (_, Some(model)) | (Some(model), None) => {
            log::info!(
                "request.start req_id={} path={} stream={} api_format={} model={} message_count={} tool_count={}",
                req_id,
                path,
                is_stream,
                api_format,
                model,
                message_count,
                tool_count,
            );
        }
        _ => {
            log::info!(
                "request.start req_id={} path={} stream={} api_format={} message_count={} tool_count={}",
                req_id,
                path,
                is_stream,
                api_format,
                message_count,
                tool_count,
            );
        }
    }
}

fn summarize_anthropic_response(body: &Value) -> ResponseLogSummary {
    let usage = body.get("usage").unwrap_or(&Value::Null);
    let content = body
        .get("content")
        .and_then(|value| value.as_array())
        .cloned()
        .unwrap_or_default();

    ResponseLogSummary {
        response_id: body.get("id").and_then(|value| value.as_str()).map(str::to_string),
        response_model: body.get("model").and_then(|value| value.as_str()).map(str::to_string),
        stop_reason: body.get("stop_reason").and_then(|value| value.as_str()).map(str::to_string),
        input_tokens: usage.get("input_tokens").and_then(|value| value.as_u64()).unwrap_or(0),
        output_tokens: usage.get("output_tokens").and_then(|value| value.as_u64()).unwrap_or(0),
        cache_read_input_tokens: usage.get("cache_read_input_tokens").and_then(|value| value.as_u64()),
        cache_creation_input_tokens: usage
            .get("cache_creation_input_tokens")
            .and_then(|value| value.as_u64()),
        has_tool_use: content.iter().any(|part| part.get("type").and_then(|value| value.as_str()) == Some("tool_use")),
        has_thinking: content.iter().any(|part| part.get("type").and_then(|value| value.as_str()) == Some("thinking")),
    }
}

pub(crate) fn log_request_done(
    context: &RequestLogContext,
    status: reqwest::StatusCode,
    summary: &ResponseLogSummary,
    is_stream: bool,
) {
    log::info!(
        "request.done req_id={} path={} stream={} api_format={} status={} latency_ms={} response_id={} response_model={} stop_reason={} input_tokens={} output_tokens={} cache_read_input_tokens={} cache_creation_input_tokens={} has_tool_use={} has_thinking={}",
        context.req_id,
        context.path,
        is_stream,
        context.api_format,
        status.as_u16(),
        context.started_at.elapsed().as_millis(),
        summary.response_id.as_deref().unwrap_or(""),
        summary.response_model.as_deref().unwrap_or(""),
        summary.stop_reason.as_deref().unwrap_or(""),
        summary.input_tokens,
        summary.output_tokens,
        summary
            .cache_read_input_tokens
            .map(|value| value.to_string())
            .as_deref()
            .unwrap_or(""),
        summary
            .cache_creation_input_tokens
            .map(|value| value.to_string())
            .as_deref()
            .unwrap_or(""),
        summary.has_tool_use,
        summary.has_thinking,
    );
}

fn sanitize_error_message(message: &str) -> String {
    let single_line = message.split_whitespace().collect::<Vec<_>>().join(" ");
    let truncated: String = single_line.chars().take(200).collect();
    if single_line.chars().count() > 200 {
        format!("{}...", truncated)
    } else {
        truncated
    }
}

fn internal_error_with_context(
    req_id: u64,
    path: &str,
    stage: &str,
    api_format: Option<&str>,
    is_stream: Option<bool>,
    started_at: Instant,
    err: impl std::fmt::Display,
) -> Response {
    log::error!(
        "request.error req_id={} path={} stage={} api_format={} stream={} latency_ms={} error={}",
        req_id,
        path,
        stage,
        api_format.unwrap_or(""),
        is_stream.map(|value| value.to_string()).as_deref().unwrap_or(""),
        started_at.elapsed().as_millis(),
        sanitize_error_message(&err.to_string()),
    );
    internal_error(err)
}

fn upstream_transport_error_with_context(
    req_id: u64,
    path: &str,
    api_format: &str,
    is_stream: bool,
    started_at: Instant,
    err: impl std::fmt::Display,
) -> Response {
    log::error!(
        "request.error req_id={} path={} stage=send_upstream api_format={} stream={} latency_ms={} error={}",
        req_id,
        path,
        api_format,
        is_stream,
        started_at.elapsed().as_millis(),
        sanitize_error_message(&err.to_string()),
    );
    upstream_transport_error(err)
}

fn resolve_config_path() -> Result<PathBuf> {
    let mut args = std::env::args().skip(1);
    let mut explicit: Option<PathBuf> = None;

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--config" | "-c" => {
                let value = args.next().context("missing value for --config")?;
                explicit = Some(PathBuf::from(value));
            }
            other if !other.starts_with('-') && explicit.is_none() => {
                explicit = Some(PathBuf::from(other));
            }
            _ => {}
        }
    }

    if let Some(path) = explicit {
        return Ok(path);
    }

    if let Ok(path) = std::env::var("CC_PROXY_CONFIG") {
        return Ok(PathBuf::from(path));
    }

    for candidate in [
        Path::new("config").join("proxy.local.yaml"),
        Path::new("config").join("proxy.responses.local.yaml"),
        Path::new("config").join("proxy.internal-chat.local.yaml"),
        Path::new("config").join("proxy.example.yaml"),
    ] {
        if candidate.exists() {
            return Ok(candidate);
        }
    }

    Ok(Path::new("config").join("proxy.local.yaml"))
}

fn build_upstream_url(base_url: &str, endpoint: &str) -> String {
    let base = base_url.trim_end_matches('/');
    let endpoint = endpoint.trim_start_matches('/');

    let already_has_v1 = base.ends_with("/v1");
    let origin_only = match base.split_once("://") {
        Some((_, rest)) => !rest.contains('/'),
        None => !base.contains('/'),
    };

    let mut url = if already_has_v1 {
        format!("{base}/{endpoint}")
    } else if origin_only {
        format!("{base}/v1/{endpoint}")
    } else {
        format!("{base}/{endpoint}")
    };

    while url.contains("/v1/v1") {
        url = url.replace("/v1/v1", "/v1");
    }

    url
}

fn apply_model_mapping(mut body: Value, model_map: &std::collections::HashMap<String, String>) -> Value {
    let Some(model) = body.get("model").and_then(|value| value.as_str()) else {
        return body;
    };

    if let Some(mapped) = model_map.get(model) {
        body["model"] = json!(mapped);
    }

    body
}

fn anthropic_error_response(status: reqwest::StatusCode, message: &str) -> Response {
    let status = StatusCode::from_u16(status.as_u16()).unwrap_or(StatusCode::BAD_GATEWAY);
    let body = json!({
        "type": "error",
        "error": {
            "type": "api_error",
            "message": message,
        }
    });
    (status, Json(body)).into_response()
}

fn extract_error_message(body: &str) -> String {
    if let Ok(json) = serde_json::from_str::<Value>(body) {
        for candidate in [
            json.pointer("/error/message"),
            json.pointer("/message"),
            json.pointer("/detail"),
            json.pointer("/error"),
        ] {
            if let Some(text) = candidate.and_then(|value| value.as_str()) {
                return text.to_string();
            }
        }
    }

    if body.trim().is_empty() {
        "upstream request failed".to_string()
    } else {
        body.trim().to_string()
    }
}

fn bad_request(err: impl std::fmt::Display) -> Response {
    (
        StatusCode::BAD_REQUEST,
        Json(json!({
            "type": "error",
            "error": {
                "type": "invalid_request_error",
                "message": err.to_string(),
            }
        })),
    )
        .into_response()
}

fn internal_error(err: impl std::fmt::Display) -> Response {
    log::error!("{err}");
    (
        StatusCode::INTERNAL_SERVER_ERROR,
        Json(json!({
            "type": "error",
            "error": {
                "type": "api_error",
                "message": err.to_string(),
            }
        })),
    )
        .into_response()
}

fn upstream_transport_error(err: impl std::fmt::Display) -> Response {
    log::error!("upstream transport error: {err}");
    (
        StatusCode::BAD_GATEWAY,
        Json(json!({
            "type": "error",
            "error": {
                "type": "api_error",
                "message": format!("upstream transport error: {err}"),
            }
        })),
    )
        .into_response()
}
