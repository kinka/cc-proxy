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
    sync::Arc,
    time::Duration,
};

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

    log::info!("cc-proxy listening on {addr}");
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
    let body_bytes = body
        .collect()
        .await
        .map_err(internal_error)?
        .to_bytes();
    let request_body: Value = serde_json::from_slice(&body_bytes).map_err(bad_request)?;
    let is_stream = request_body
        .get("stream")
        .and_then(|value| value.as_bool())
        .unwrap_or(false);

    let mapped_body = apply_model_mapping(request_body, &state.config.upstream.model_map);
    let transformed_body = match state.config.upstream.api_format {
        ApiFormat::OpenAiChat => transform::anthropic_to_openai(
            mapped_body,
            state.config.upstream.prompt_cache_key.as_deref(),
        ),
        ApiFormat::OpenAiResponses => transform_responses::anthropic_to_responses(
            mapped_body,
            state.config.upstream.prompt_cache_key.as_deref(),
        ),
    }
    .map_err(internal_error)?;

    let endpoint = match state.config.upstream.api_format {
        ApiFormat::OpenAiChat => "/chat/completions",
        ApiFormat::OpenAiResponses => "/responses",
    };
    let upstream_url = build_upstream_url(&state.config.upstream.base_url, endpoint);

    log::info!(
        "proxying Claude request to upstream {} ({})",
        upstream_url,
        match state.config.upstream.api_format {
            ApiFormat::OpenAiChat => "openai_chat",
            ApiFormat::OpenAiResponses => "openai_responses",
        }
    );

    let payload = serde_json::to_vec(&transformed_body).map_err(internal_error)?;
    log::debug!(
        "upstream request body: {}",
        String::from_utf8_lossy(&payload)
    );
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

    let upstream_response = upstream_request.send().await.map_err(upstream_transport_error)?;
    let upstream_status = upstream_response.status();

    if !upstream_status.is_success() {
        let body = upstream_response.text().await.unwrap_or_default();
        let message = extract_error_message(&body);
        return Err(anthropic_error_response(upstream_status, &message));
    }

    if is_stream {
        let stream = upstream_response
            .bytes_stream()
            .map(|chunk| chunk.map_err(|err| std::io::Error::other(err.to_string())));

        let anthropic_stream: Pin<
            Box<dyn Stream<Item = Result<Bytes, std::io::Error>> + Send>,
        > = match state.config.upstream.api_format {
            ApiFormat::OpenAiChat => Box::pin(streaming::create_anthropic_sse_stream(stream)),
            ApiFormat::OpenAiResponses => Box::pin(
                streaming_responses::create_anthropic_sse_stream_from_responses(stream),
            ),
        };

        let mut headers = HeaderMap::new();
        headers.insert("content-type", HeaderValue::from_static("text/event-stream"));
        headers.insert("cache-control", HeaderValue::from_static("no-cache"));
        headers.insert("connection", HeaderValue::from_static("keep-alive"));
        return Ok((headers, Body::from_stream(anthropic_stream)).into_response());
    }

    let upstream_body = upstream_response.bytes().await.map_err(upstream_transport_error)?;
    log::debug!(
        "upstream response body: {}",
        String::from_utf8_lossy(&upstream_body)
    );
    let parsed: Value = serde_json::from_slice(&upstream_body).map_err(internal_error)?;
    let anthropic_body = match state.config.upstream.api_format {
        ApiFormat::OpenAiChat => transform::openai_to_anthropic(parsed),
        ApiFormat::OpenAiResponses => transform_responses::responses_to_anthropic(parsed),
    }
    .map_err(internal_error)?;

    let mut headers = HeaderMap::new();
    headers.insert("content-type", HeaderValue::from_static("application/json"));
    Ok((StatusCode::OK, headers, Json(anthropic_body)).into_response())
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
