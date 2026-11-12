mod config;
mod sse;
mod streaming;
mod streaming_anthropic;
mod streaming_chat_responses;
mod streaming_responses;
mod transform;
mod transform_responses;
mod usage;

use crate::config::{ApiFormat, ProxyConfig, UpstreamConfig};
use crate::usage::{Outcome, QueryError, RequestIdentity, TokenCounts, UsageMeta, UsageQuery, UsageRecorder, UsageTracking};
use anyhow::{Context, Result};
use axum::{
    body::Body,
    extract::{Query, Request, State},
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
    /// Ledger handle; `None` when the `usage` config section is absent.
    pub(crate) usage: Option<Arc<UsageTracking>>,
}

impl RequestLogContext {
    #[cfg(test)]
    pub(crate) fn for_test(path: &str, api_format: &'static str) -> Self {
        Self { req_id: 1, path: path.to_string(), api_format, started_at: Instant::now(), usage: None }
    }

    /// Native Anthropic usage and the Chat Completions → Anthropic conversion both report
    /// `input_tokens` net of cache; every Responses-API-shaped usage counts cached input inside
    /// `input_tokens`.
    pub(crate) fn input_includes_cache(&self) -> bool {
        !matches!(self.api_format, "openai_chat" | "anthropic")
    }

    /// Settle this request in the ledger. The first call wins; later ones are no-ops.
    pub(crate) fn record(&self, outcome: Outcome, status: Option<u16>, tokens: TokenCounts, response_model: Option<&str>) {
        if let Some(usage) = &self.usage {
            usage.settle(self.req_id, &self.path, self.api_format, self.started_at, outcome, status, tokens, response_model);
        }
    }

    fn internal_error(&self, stage: &str, is_stream: bool, err: impl std::fmt::Display) -> Response {
        self.record(Outcome::InternalError, Some(500), TokenCounts::default(), None);
        internal_error_with_context(self.req_id, &self.path, stage, Some(self.api_format), Some(is_stream), self.started_at, err)
    }

    fn transport_error(&self, is_stream: bool, err: impl std::fmt::Display) -> Response {
        self.record(Outcome::TransportError, Some(502), TokenCounts::default(), None);
        upstream_transport_error_with_context(self.req_id, &self.path, self.api_format, is_stream, self.started_at, err)
    }
}

/// Records `client_aborted` for a request nothing else settled.
///
/// The case it exists for: the caller cancels a turn or its process dies mid-request, axum
/// drops the handler future or the response body, and none of the terminal paths runs. The
/// upstream may well have billed for it. `partial` carries whatever usage the stream had seen
/// so far — usually nothing, since providers send usage in the last chunk, but the request is
/// still counted so the gap is measurable.
pub(crate) struct UsageGuard {
    context: RequestLogContext,
    pub(crate) partial: TokenCounts,
    pub(crate) response_model: Option<String>,
    armed: bool,
}

impl UsageGuard {
    pub(crate) fn new(context: RequestLogContext) -> Self {
        Self { context, partial: TokenCounts::default(), response_model: None, armed: true }
    }

    /// Hand responsibility to someone else (the handler does this once a stream takes over).
    pub(crate) fn disarm(&mut self) {
        self.armed = false;
    }
}

impl Drop for UsageGuard {
    fn drop(&mut self) {
        if !self.armed || self.context.usage.as_ref().is_none_or(|usage| usage.is_settled()) {
            return;
        }
        log::warn!(
            "request.client_aborted req_id={} path={} api_format={} latency_ms={} partial_total_tokens={}",
            self.context.req_id,
            self.context.path,
            self.context.api_format,
            self.context.started_at.elapsed().as_millis(),
            self.partial.total(),
        );
        self.context.record(Outcome::ClientAborted, None, self.partial, self.response_model.as_deref());
    }
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
    /// Visible-text and reasoning sizes, in Unicode chars rather than bytes so a
    /// Chinese reply is not counted as three times its length.
    ///
    /// These exist to make one specific failure visible without capturing bodies:
    /// a model that drafts the whole answer inside its reasoning and then writes it
    /// out again. It shows up as reasoning and text being of comparable size on a
    /// turn that produced one short answer, and it costs real output tokens on top
    /// of leaking the reasoning into the user-visible reply. Counts only — the
    /// content itself is business data and must not reach the log.
    thinking_chars: usize,
    text_chars: usize,
}

#[derive(Clone)]
struct AppState {
    config: Arc<ProxyConfig>,
    client: Client,
    usage_recorder: Option<UsageRecorder>,
    usage_query: Option<Arc<UsageQuery>>,
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

    let (usage_recorder, usage_query) = match &config.usage {
        Some(usage) => {
            let dir = PathBuf::from(&usage.dir);
            let recorder = UsageRecorder::spawn(dir.clone(), usage.retention_days)
                .with_context(|| format!("failed to start usage recorder in {}", usage.dir))?;
            (Some(recorder), Some(Arc::new(UsageQuery::new(dir))))
        }
        None => (None, None),
    };

    let state = AppState { config, client, usage_recorder, usage_query };

    let app = Router::new()
        .route("/health", get(health))
        .route("/usage/daily", get(usage_daily))
        .route("/usage/sessions", get(usage_sessions))
        .route("/v1/messages", post(handle_messages))
        .route("/claude/v1/messages", post(handle_messages))
        .route("/v1/responses", post(handle_responses))
        .route("/:provider/v1/messages", post(handle_messages_provider))
        .route("/:provider/claude/v1/messages", post(handle_messages_provider))
        .route("/:provider/v1/responses", post(handle_responses_provider))
        .with_state(state.clone());

    let addr: SocketAddr = format!(
        "{}:{}",
        state.config.listen.host, state.config.listen.port
    )
    .parse()
    .context("invalid listen address")?;

    log::info!(
        "startup addr={} config_path={} api_format={} upstream_base_url={} timeout_secs={} model_map_count={} extra_headers_count={} usage_ledger={} http_proxy_set={} https_proxy_set={} no_proxy_set={}",
        addr,
        config_path.display(),
        api_format_name(state.config.upstream.api_format),
        state.config.upstream.base_url,
        state.config.upstream.timeout_secs,
        state.config.upstream.model_map.len(),
        state.config.upstream.extra_headers.len(),
        state.config.usage.as_ref().map_or("off", |usage| usage.dir.as_str()),
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

#[derive(serde::Deserialize)]
struct UsageParams {
    date: Option<String>,
    env: Option<String>,
    trace_id: Option<String>,
}

fn usage_query_response(result: Result<Value, QueryError>) -> Response {
    match result {
        Ok(value) => Json(value).into_response(),
        Err(QueryError::BadRequest(message)) => (StatusCode::BAD_REQUEST, Json(json!({ "error": message }))).into_response(),
        Err(QueryError::Internal(message)) => {
            log::warn!("usage.query_failed error={}", sanitize_error_message(&message));
            (StatusCode::INTERNAL_SERVER_ERROR, Json(json!({ "error": "usage ledger read failed" }))).into_response()
        }
    }
}

async fn run_usage_query(
    state: AppState,
    query_fn: impl FnOnce(&UsageQuery) -> Result<Value, QueryError> + Send + 'static,
) -> Response {
    let Some(query) = state.usage_query.clone() else {
        return StatusCode::NOT_FOUND.into_response();
    };
    let result = tokio::task::spawn_blocking(move || query_fn(&query))
        .await
        .unwrap_or_else(|err| Err(QueryError::Internal(err.to_string())));
    usage_query_response(result)
}

async fn usage_daily(State(state): State<AppState>, Query(params): Query<UsageParams>) -> Response {
    run_usage_query(state, move |query| query.daily(params.date.as_deref(), params.env.as_deref())).await
}

async fn usage_sessions(State(state): State<AppState>, Query(params): Query<UsageParams>) -> Response {
    run_usage_query(state, move |query| query.sessions(params.date.as_deref(), params.trace_id.as_deref())).await
}

/// The trace headers to copy onto the upstream request. A header the deployment pins via
/// `extra_headers` wins — reqwest appends rather than replaces, and two values for one
/// identity header is worse than either.
fn trace_headers_to_forward(
    incoming: &HeaderMap,
    extra_headers: &std::collections::HashMap<String, String>,
) -> Vec<(&'static str, HeaderValue)> {
    usage::FORWARDED_TRACE_HEADERS
        .iter()
        .filter(|name| !extra_headers.keys().any(|key| key.eq_ignore_ascii_case(name)))
        .filter_map(|name| incoming.get(*name).map(|value| (*name, value.clone())))
        .collect()
}

/// Mirrors the routing in `resolve_upstream_and_body`, for the ledger only.
fn resolved_provider_name(body: &Value, config: &ProxyConfig, provider_from_path: Option<&str>) -> Option<String> {
    if let Some(provider) = provider_from_path.filter(|provider| config.providers.contains_key(*provider)) {
        return Some(provider.to_string());
    }
    let (provider, _) = body.get("model")?.as_str()?.split_once('/')?;
    config.providers.contains_key(provider).then(|| provider.to_string())
}

fn build_usage_tracking(
    state: &AppState,
    headers: &HeaderMap,
    provider: Option<String>,
    is_stream: bool,
    requested_model: Option<&str>,
    mapped_model: Option<&str>,
) -> Option<Arc<UsageTracking>> {
    let recorder = state.usage_recorder.clone()?;
    Some(Arc::new(UsageTracking::new(
        recorder,
        UsageMeta {
            provider,
            stream: is_stream,
            requested_model: requested_model.map(str::to_string),
            mapped_model: mapped_model.map(str::to_string),
            identity: RequestIdentity::from_headers(headers),
        },
    )))
}

async fn handle_messages(
    State(state): State<AppState>,
    request: Request,
) -> Result<Response, Response> {
    handle_messages_internal(state, None, request).await
}

async fn handle_messages_provider(
    State(state): State<AppState>,
    axum::extract::Path(provider): axum::extract::Path<String>,
    request: Request,
) -> Result<Response, Response> {
    handle_messages_internal(state, Some(provider), request).await
}

async fn handle_messages_internal(
    state: AppState,
    provider_from_path: Option<String>,
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

    let provider_name = resolved_provider_name(&request_body, &state.config, provider_from_path.as_deref());
    let (upstream, request_body) = resolve_upstream_and_body(request_body, &state.config, provider_from_path.as_deref());

    let mut mapped_body = apply_model_mapping(request_body, &upstream.model_map, upstream.default_model.as_deref());
    if let Some(ref vision_model) = upstream.vision_model {
        if transform::request_has_image(&mapped_body) {
            mapped_body["model"] = json!(vision_model);
        }
    }
    let mapped_model = mapped_body
        .get("model")
        .and_then(|value| value.as_str())
        .map(str::to_string);
    let api_format = upstream.api_format;
    let api_format_name = api_format_name(api_format);
    let usage = build_usage_tracking(
        &state,
        &parts.headers,
        provider_name,
        is_stream,
        requested_model.as_deref(),
        mapped_model.as_deref(),
    );
    let request_context = RequestLogContext {
        req_id,
        path: path.clone(),
        api_format: api_format_name,
        started_at,
        usage,
    };
    let mut usage_guard = UsageGuard::new(request_context.clone());

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
            upstream.prompt_cache_key.as_deref(),
            upstream.strip_tool_result_images,
        ),
        ApiFormat::OpenAiResponses => transform_responses::anthropic_to_responses(
            mapped_body,
            upstream.prompt_cache_key.as_deref(),
        ),
        ApiFormat::Anthropic => Ok(mapped_body),
    }
    .map_err(|err| request_context.internal_error("transform_request", is_stream, err))?;

    let transformed_body = apply_extra_body(req_id, transformed_body, &upstream.extra_body);

    let endpoint = match api_format {
        ApiFormat::OpenAiChat => "/chat/completions",
        ApiFormat::OpenAiResponses => "/responses",
        ApiFormat::Anthropic => "/messages",
    };
    let upstream_url = build_upstream_url(&upstream.base_url, endpoint);

    let payload = serde_json::to_vec(&transformed_body)
        .map_err(|err| request_context.internal_error("serialize_upstream_request", is_stream, err))?;
    let api_key = get_request_api_key(&parts.headers)
        .unwrap_or_else(|| upstream.api_key.clone());
    let mut upstream_request = state
        .client
        .post(&upstream_url)
        .timeout(Duration::from_secs(upstream.timeout_secs))
        .header("authorization", format!("Bearer {}", api_key))
        .header("content-type", "application/json")
        .header("accept-encoding", "identity")
        .body(payload);

    for (name, value) in trace_headers_to_forward(&parts.headers, &upstream.extra_headers) {
        upstream_request = upstream_request.header(name, value);
    }
    if api_format == ApiFormat::Anthropic {
        for (name, value) in anthropic_headers_to_forward(&parts.headers, &upstream.extra_headers) {
            upstream_request = upstream_request.header(name, value);
        }
    }
    for (name, value) in &upstream.extra_headers {
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
        .map_err(|err| request_context.transport_error(is_stream, err))?;
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
        request_context.record(Outcome::UpstreamError, Some(upstream_status.as_u16()), TokenCounts::default(), None);
        // An Anthropic upstream already speaks the client's error shape; keep its error type
        // (`overloaded_error`, `rate_limit_error` …) so the client's retry logic still works.
        if api_format == ApiFormat::Anthropic {
            if let Some(response) = passthrough_anthropic_error(upstream_status, &body) {
                return Err(response);
            }
        }
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
            ApiFormat::Anthropic => Box::pin(
                streaming_anthropic::create_passthrough_sse_stream(stream, request_context.clone()),
            ),
        };

        let mut headers = HeaderMap::new();
        headers.insert("content-type", HeaderValue::from_static("text/event-stream"));
        headers.insert("cache-control", HeaderValue::from_static("no-cache"));
        headers.insert("connection", HeaderValue::from_static("keep-alive"));
        // The stream owns settlement from here, with its own guard for client aborts.
        usage_guard.disarm();
        return Ok((headers, Body::from_stream(anthropic_stream)).into_response());
    }

    let upstream_body = upstream_response
        .bytes()
        .await
        .map_err(|err| request_context.transport_error(is_stream, err))?;
    let parsed: Value = serde_json::from_slice(&upstream_body)
        .map_err(|err| request_context.internal_error("parse_upstream_json", is_stream, err))?;

    let mut headers = HeaderMap::new();
    headers.insert("content-type", HeaderValue::from_static("application/json"));

    if api_format == ApiFormat::Anthropic {
        // Parsed only for the ledger; the client gets the upstream bytes as they came.
        log_request_done(&request_context, upstream_status, &summarize_anthropic_response(&parsed), false);
        return Ok((StatusCode::OK, headers, Body::from(upstream_body)).into_response());
    }

    let anthropic_body = match api_format {
        ApiFormat::OpenAiChat => transform::openai_to_anthropic(parsed),
        ApiFormat::OpenAiResponses => transform_responses::responses_to_anthropic(parsed),
        ApiFormat::Anthropic => Ok(parsed),
    }
    .map_err(|err| request_context.internal_error("transform_upstream_response", is_stream, err))?;

    log_request_done(&request_context, upstream_status, &summarize_anthropic_response(&anthropic_body), false);

    Ok((StatusCode::OK, headers, Json(anthropic_body)).into_response())
}

async fn handle_responses(
    State(state): State<AppState>,
    request: Request,
) -> Result<Response, Response> {
    handle_responses_internal(state, None, request).await
}

async fn handle_responses_provider(
    State(state): State<AppState>,
    axum::extract::Path(provider): axum::extract::Path<String>,
    request: Request,
) -> Result<Response, Response> {
    handle_responses_internal(state, Some(provider), request).await
}

async fn handle_responses_internal(
    state: AppState,
    provider_from_path: Option<String>,
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
    let input_count = request_body
        .get("input")
        .and_then(|value| value.as_array())
        .map_or(usize::from(request_body.get("input").is_some()), |input| input.len());
    let tool_count = request_body
        .get("tools")
        .and_then(|value| value.as_array())
        .map_or(0, |tools| tools.len());

    let provider_name = resolved_provider_name(&request_body, &state.config, provider_from_path.as_deref());
    let (upstream, request_body) = resolve_upstream_and_body(request_body, &state.config, provider_from_path.as_deref());

    // This route always translates to Chat Completions; an Anthropic upstream has no such endpoint.
    if upstream.api_format == ApiFormat::Anthropic {
        log::warn!(
            "request.rejected req_id={} path={} reason=responses_route_on_anthropic_upstream",
            req_id,
            path,
        );
        return Err(openai_error_response(
            reqwest::StatusCode::BAD_REQUEST,
            "this upstream uses api_format=anthropic; send Anthropic Messages requests to /v1/messages instead of /v1/responses",
        ));
    }

    let mapped_body = apply_model_mapping(request_body, &upstream.model_map, upstream.default_model.as_deref());
    let mapped_model = mapped_body
        .get("model")
        .and_then(|value| value.as_str())
        .map(str::to_string);
    let api_format_name = "responses_to_openai_chat";
    let usage = build_usage_tracking(
        &state,
        &parts.headers,
        provider_name,
        is_stream,
        requested_model.as_deref(),
        mapped_model.as_deref(),
    );
    let request_context = RequestLogContext {
        req_id,
        path: path.clone(),
        api_format: api_format_name,
        started_at,
        usage,
    };
    let mut usage_guard = UsageGuard::new(request_context.clone());

    log_request_start(
        req_id,
        &path,
        is_stream,
        api_format_name,
        requested_model.as_deref(),
        mapped_model.as_deref(),
        input_count,
        tool_count,
    );

    let transformed_body = transform_responses::responses_request_to_openai_chat(mapped_body)
        .map_err(|err| request_context.internal_error("transform_request", is_stream, err))?;
    let transformed_body = apply_extra_body(req_id, transformed_body, &upstream.extra_body);
    let upstream_url = build_upstream_url(&upstream.base_url, "/chat/completions");
    let payload = serde_json::to_vec(&transformed_body)
        .map_err(|err| request_context.internal_error("serialize_upstream_request", is_stream, err))?;
    let api_key = get_request_api_key(&parts.headers)
        .unwrap_or_else(|| upstream.api_key.clone());
    let mut upstream_request = state
        .client
        .post(&upstream_url)
        .timeout(Duration::from_secs(upstream.timeout_secs))
        .header("authorization", format!("Bearer {}", api_key))
        .header("content-type", "application/json")
        .header("accept-encoding", "identity")
        .body(payload);

    for (name, value) in trace_headers_to_forward(&parts.headers, &upstream.extra_headers) {
        upstream_request = upstream_request.header(name, value);
    }
    for (name, value) in &upstream.extra_headers {
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
        .map_err(|err| request_context.transport_error(is_stream, err))?;
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
        request_context.record(Outcome::UpstreamError, Some(upstream_status.as_u16()), TokenCounts::default(), None);
        return Err(openai_error_response(upstream_status, &message));
    }

    if is_stream {
        let stream = upstream_response
            .bytes_stream()
            .map(|chunk| chunk.map_err(|err| std::io::Error::other(err.to_string())));
        let responses_stream = streaming_chat_responses::create_responses_sse_stream_from_chat(
            stream,
            request_context.clone(),
        );

        let mut headers = HeaderMap::new();
        headers.insert("content-type", HeaderValue::from_static("text/event-stream"));
        headers.insert("cache-control", HeaderValue::from_static("no-cache"));
        headers.insert("connection", HeaderValue::from_static("keep-alive"));
        // The stream owns settlement from here, with its own guard for client aborts.
        usage_guard.disarm();
        return Ok((headers, Body::from_stream(responses_stream)).into_response());
    }

    let upstream_body = upstream_response
        .bytes()
        .await
        .map_err(|err| request_context.transport_error(is_stream, err))?;
    let parsed: Value = serde_json::from_slice(&upstream_body)
        .map_err(|err| request_context.internal_error("parse_upstream_json", is_stream, err))?;
    let responses_body = transform_responses::openai_chat_to_responses(parsed)
        .map_err(|err| request_context.internal_error("transform_upstream_response", is_stream, err))?;

    log_request_done(&request_context, upstream_status, &summarize_responses_response(&responses_body), false);

    let mut headers = HeaderMap::new();
    headers.insert("content-type", HeaderValue::from_static("application/json"));
    Ok((StatusCode::OK, headers, Json(responses_body)).into_response())
}


fn api_format_name(api_format: ApiFormat) -> &'static str {
    match api_format {
        ApiFormat::OpenAiChat => "openai_chat",
        ApiFormat::OpenAiResponses => "openai_responses",
        ApiFormat::Anthropic => "anthropic",
    }
}

/// Protocol headers an Anthropic upstream needs from the client: the API version, and the
/// beta flags that switch on features the body already relies on (a request using a beta
/// field without its flag is a 400). `extra_headers` wins, same rule as the trace headers.
fn anthropic_headers_to_forward(
    incoming: &HeaderMap,
    extra_headers: &std::collections::HashMap<String, String>,
) -> Vec<(&'static str, HeaderValue)> {
    const DEFAULT_ANTHROPIC_VERSION: &str = "2023-06-01";
    let pinned = |name: &str| extra_headers.keys().any(|key| key.eq_ignore_ascii_case(name));
    let mut headers = Vec::new();
    if !pinned("anthropic-version") {
        let version = incoming
            .get("anthropic-version")
            .cloned()
            .unwrap_or_else(|| HeaderValue::from_static(DEFAULT_ANTHROPIC_VERSION));
        headers.push(("anthropic-version", version));
    }
    if !pinned("anthropic-beta") {
        if let Some(beta) = incoming.get("anthropic-beta") {
            headers.push(("anthropic-beta", beta.clone()));
        }
    }
    headers
}

/// Relay an Anthropic-shaped error body as-is; `None` when the body is not one.
fn passthrough_anthropic_error(status: reqwest::StatusCode, body: &str) -> Option<Response> {
    let parsed: Value = serde_json::from_str(body).ok()?;
    if parsed.get("type").and_then(Value::as_str) != Some("error") || parsed.get("error").is_none() {
        return None;
    }
    let status = StatusCode::from_u16(status.as_u16()).unwrap_or(StatusCode::BAD_GATEWAY);
    Some((status, Json(parsed)).into_response())
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
        thinking_chars: count_part_chars(&content, "thinking", "thinking"),
        text_chars: count_part_chars(&content, "text", "text"),
    }
}

/// Sum the chars of every `type == part_type` block, reading the field named `field`.
fn count_part_chars(parts: &[Value], part_type: &str, field: &str) -> usize {
    parts
        .iter()
        .filter(|part| part.get("type").and_then(|value| value.as_str()) == Some(part_type))
        .filter_map(|part| part.get(field).and_then(|value| value.as_str()))
        .map(|text| text.chars().count())
        .sum()
}

fn summarize_responses_response(body: &Value) -> ResponseLogSummary {
    let usage = body.get("usage").unwrap_or(&Value::Null);
    let output = body.get("output").and_then(|value| value.as_array());

    ResponseLogSummary {
        response_id: body.get("id").and_then(|value| value.as_str()).map(str::to_string),
        response_model: body.get("model").and_then(|value| value.as_str()).map(str::to_string),
        stop_reason: body.get("status").and_then(|value| value.as_str()).map(str::to_string),
        input_tokens: usage.get("input_tokens").and_then(|value| value.as_u64()).unwrap_or(0),
        output_tokens: usage.get("output_tokens").and_then(|value| value.as_u64()).unwrap_or(0),
        cache_read_input_tokens: usage
            .pointer("/input_tokens_details/cached_tokens")
            .and_then(|value| value.as_u64()),
        cache_creation_input_tokens: None,
        has_tool_use: output.is_some_and(|items| items.iter().any(|part| part.get("type").and_then(|value| value.as_str()) == Some("function_call"))),
        has_thinking: output.is_some_and(|items| items.iter().any(|part| part.get("type").and_then(|value| value.as_str()) == Some("reasoning"))),
        // Responses nests one level deeper than Chat: reasoning carries `summary[].summary_text`,
        // the answer carries `message.content[].output_text`.
        thinking_chars: output.map_or(0, |items| count_nested_chars(items, "reasoning", "summary", &["summary_text"])),
        text_chars: output.map_or(0, |items| count_nested_chars(items, "message", "content", &["output_text", "text"])),
    }
}

/// Sum the chars of `child_types` parts found under `child_key` of every `item_type` item.
fn count_nested_chars(items: &[Value], item_type: &str, child_key: &str, child_types: &[&str]) -> usize {
    items
        .iter()
        .filter(|item| item.get("type").and_then(|value| value.as_str()) == Some(item_type))
        .filter_map(|item| item.get(child_key).and_then(|value| value.as_array()))
        .flatten()
        .filter(|part| {
            part.get("type")
                .and_then(|value| value.as_str())
                .is_some_and(|found| child_types.contains(&found))
        })
        .filter_map(|part| part.get("text").and_then(|value| value.as_str()))
        .map(|text| text.chars().count())
        .sum()
}

pub(crate) fn log_request_done(
    context: &RequestLogContext,
    status: reqwest::StatusCode,
    summary: &ResponseLogSummary,
    is_stream: bool,
) {
    log::info!(
        "request.done req_id={} path={} stream={} api_format={} status={} latency_ms={} response_id={} response_model={} stop_reason={} input_tokens={} output_tokens={} cache_read_input_tokens={} cache_creation_input_tokens={} has_tool_use={} has_thinking={} thinking_chars={} text_chars={}",
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
        summary.thinking_chars,
        summary.text_chars,
    );
    context.record(
        Outcome::Done,
        Some(status.as_u16()),
        TokenCounts::from_parts(
            summary.input_tokens,
            summary.output_tokens,
            summary.cache_read_input_tokens,
            summary.cache_creation_input_tokens,
            context.input_includes_cache(),
        ),
        summary.response_model.as_deref(),
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

/// Merge configured `extra_body` fields into the request that is about to leave for
/// the upstream, so sampling knobs with no Anthropic equivalent (`repetition_penalty`
/// and friends) can be pinned per provider without the client knowing about them.
///
/// Reserved keys are dropped with a warning rather than applied: a typo there would
/// otherwise surface as an opaque upstream 400 with nothing pointing back at the config.
fn apply_extra_body(
    req_id: u64,
    mut body: Value,
    extra_body: &std::collections::HashMap<String, Value>,
) -> Value {
    if extra_body.is_empty() {
        return body;
    }
    let Some(object) = body.as_object_mut() else {
        return body;
    };
    for (key, value) in extra_body {
        if config::RESERVED_EXTRA_BODY_KEYS.contains(&key.as_str()) {
            log::warn!(
                "req_id={} ignoring reserved extra_body field key={}",
                req_id,
                key
            );
            continue;
        }
        object.insert(key.clone(), value.clone());
    }
    body
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

fn apply_model_mapping(
    mut body: Value,
    model_map: &std::collections::HashMap<String, String>,
    default_model: Option<&str>,
) -> Value {
    let Some(model) = body.get("model").and_then(|value| value.as_str()) else {
        return body;
    };

    if let Some(mapped) = model_map.get(model) {
        body["model"] = json!(mapped);
    } else if let Some(def) = default_model {
        body["model"] = json!(def);
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

fn openai_error_response(status: reqwest::StatusCode, message: &str) -> Response {
    let status = StatusCode::from_u16(status.as_u16()).unwrap_or(StatusCode::BAD_GATEWAY);
    let body = json!({
        "error": {
            "message": message,
            "type": "api_error",
        }
    });
    (status, Json(body)).into_response()
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

fn get_request_api_key(headers: &HeaderMap) -> Option<String> {
    if let Some(key) = headers.get("x-api-key").and_then(|v| v.to_str().ok()) {
        let trimmed = key.trim();
        if !trimmed.is_empty() && !trimmed.starts_with("sk-ant-") {
            return Some(trimmed.to_string());
        }
    }
    if let Some(auth) = headers.get("authorization").and_then(|v| v.to_str().ok()) {
        let trimmed_auth = auth.trim();
        if !trimmed_auth.is_empty() {
            if auth.get(..7).map_or(false, |s| s.eq_ignore_ascii_case("Bearer ")) {
                let token = auth[7..].trim();
                if !token.is_empty() && !token.starts_with("sk-ant-") {
                    return Some(token.to_string());
                }
            } else if trimmed_auth.get(..7).map_or(false, |s| s.eq_ignore_ascii_case("Bearer ")) {
                let token = trimmed_auth[7..].trim();
                if !token.is_empty() && !token.starts_with("sk-ant-") {
                    return Some(token.to_string());
                }
            } else if trimmed_auth.eq_ignore_ascii_case("bearer") {
                return None;
            } else if !trimmed_auth.starts_with("sk-ant-") {
                return Some(trimmed_auth.to_string());
            }
        }
    }
    None
}

fn resolve_upstream_and_body<'a>(
    mut body: Value,
    config: &'a ProxyConfig,
    provider_from_path: Option<&str>,
) -> (&'a UpstreamConfig, Value) {
    if let Some(provider) = provider_from_path {
        if let Some(upstream_cfg) = config.providers.get(provider) {
            return (upstream_cfg, body);
        }
    }

    let Some(model_str) = body.get("model").and_then(|v| v.as_str()) else {
        return (&config.upstream, body);
    };

    if let Some((provider, model_name)) = model_str.split_once('/') {
        if let Some(upstream_cfg) = config.providers.get(provider) {
            body["model"] = json!(model_name);
            return (upstream_cfg, body);
        }
    }

    (&config.upstream, body)
}

#[cfg(test)]
mod main_tests {
    use super::*;

    #[test]
    fn only_trace_headers_are_forwarded_and_extra_headers_win() {
        let mut incoming = HeaderMap::new();
        incoming.insert("x-trace-identifier", HeaderValue::from_static("alice"));
        incoming.insert("x-trace-identifier-type", HeaderValue::from_static("oa"));
        incoming.insert("x-cura-session-id", HeaderValue::from_static("s1"));
        incoming.insert("x-cura-env", HeaderValue::from_static("prod"));

        let forwarded = trace_headers_to_forward(&incoming, &HashMap::new());
        let names: Vec<_> = forwarded.iter().map(|(name, _)| *name).collect();
        assert_eq!(names, vec!["x-trace-identifier", "x-trace-identifier-type"]);

        let pinned = HashMap::from([("X-Trace-Identifier-Type".to_string(), "portal".to_string())]);
        let forwarded = trace_headers_to_forward(&incoming, &pinned);
        let names: Vec<_> = forwarded.iter().map(|(name, _)| *name).collect();
        assert_eq!(names, vec!["x-trace-identifier"]);
    }

    use crate::config::{ListenConfig, UpstreamConfig, ApiFormat};
    use axum::http::HeaderMap;
    use std::collections::HashMap;

    #[test]
    fn anthropic_api_format_parses_and_targets_messages() {
        let parsed: ApiFormat = serde_yaml::from_str("anthropic").unwrap();
        assert_eq!(parsed, ApiFormat::Anthropic);
        assert_eq!(build_upstream_url("https://gw.example.com", "/messages"), "https://gw.example.com/v1/messages");
        assert_eq!(build_upstream_url("https://gw.example.com/v1/", "/messages"), "https://gw.example.com/v1/messages");
    }

    #[test]
    fn anthropic_input_tokens_exclude_cache() {
        assert!(!RequestLogContext::for_test("/v1/messages", "anthropic").input_includes_cache());
        assert!(!RequestLogContext::for_test("/v1/messages", "openai_chat").input_includes_cache());
        assert!(RequestLogContext::for_test("/v1/messages", "openai_responses").input_includes_cache());
    }

    #[test]
    fn anthropic_protocol_headers_default_version_and_respect_extra_headers() {
        let forwarded = anthropic_headers_to_forward(&HeaderMap::new(), &HashMap::new());
        assert_eq!(forwarded.len(), 1);
        assert_eq!(forwarded[0].0, "anthropic-version");
        assert_eq!(forwarded[0].1, "2023-06-01");

        let mut incoming = HeaderMap::new();
        incoming.insert("anthropic-version", HeaderValue::from_static("2024-01-01"));
        incoming.insert("anthropic-beta", HeaderValue::from_static("interleaved-thinking-2025-05-14"));
        let forwarded = anthropic_headers_to_forward(&incoming, &HashMap::new());
        let pairs: Vec<_> = forwarded.iter().map(|(name, value)| (*name, value.to_str().unwrap())).collect();
        assert_eq!(pairs, vec![("anthropic-version", "2024-01-01"), ("anthropic-beta", "interleaved-thinking-2025-05-14")]);

        let pinned = HashMap::from([("Anthropic-Version".to_string(), "2023-06-01".to_string())]);
        let forwarded = anthropic_headers_to_forward(&incoming, &pinned);
        let names: Vec<_> = forwarded.iter().map(|(name, _)| *name).collect();
        assert_eq!(names, vec!["anthropic-beta"]);
    }

    #[test]
    fn anthropic_error_bodies_pass_through_and_others_do_not() {
        let body = r#"{"type":"error","error":{"type":"overloaded_error","message":"Overloaded"}}"#;
        let response = passthrough_anthropic_error(reqwest::StatusCode::from_u16(529).unwrap(), body).unwrap();
        assert_eq!(response.status().as_u16(), 529);

        assert!(passthrough_anthropic_error(reqwest::StatusCode::BAD_GATEWAY, "<html>bad gateway</html>").is_none());
        assert!(passthrough_anthropic_error(reqwest::StatusCode::BAD_REQUEST, r#"{"error":{"message":"x"}}"#).is_none());
    }

    /// The 2026-09-10 shape this field exists for: the model drafts the whole answer inside
    /// its reasoning, then writes it again as visible text. Both sides must be counted in
    /// chars — measured in bytes a Chinese reply reads as three times its real length and
    /// the "reasoned and answered twice" ratio stops meaning anything.
    #[test]
    fn test_summarize_counts_thinking_and_text_in_chars() {
        let body = json!({
            "content": [
                { "type": "thinking", "thinking": "用户要求补充工时" },
                { "type": "text", "text": "已成功追加" },
                { "type": "text", "text": "ok" },
            ]
        });
        let summary = summarize_anthropic_response(&body);
        assert!(summary.has_thinking);
        assert_eq!(summary.thinking_chars, 8);
        assert_eq!(summary.text_chars, 7);
    }

    #[test]
    fn test_summarize_responses_counts_nested_parts() {
        let body = json!({
            "output": [
                { "type": "reasoning", "summary": [{ "type": "summary_text", "text": "推理" }] },
                { "type": "message", "content": [
                    { "type": "output_text", "text": "答案" },
                    { "type": "refusal", "text": "不该计入" },
                ] },
            ]
        });
        let summary = summarize_responses_response(&body);
        assert_eq!(summary.thinking_chars, 2);
        assert_eq!(summary.text_chars, 2);
    }

    #[test]
    fn test_get_request_api_key() {
        let mut headers = HeaderMap::new();
        assert_eq!(get_request_api_key(&headers), None);

        // 1. Should ignore keys starting with sk-ant-
        headers.insert("x-api-key", "sk-ant-dummykey12345".parse().unwrap());
        assert_eq!(get_request_api_key(&headers), None);

        headers.insert("x-api-key", "my-secret-key-1".parse().unwrap());
        assert_eq!(get_request_api_key(&headers), Some("my-secret-key-1".to_string()));

        headers.remove("x-api-key");
        
        // 2. Should ignore Bearer keys starting with sk-ant-
        headers.insert("authorization", "Bearer sk-ant-dummykey12345".parse().unwrap());
        assert_eq!(get_request_api_key(&headers), None);

        headers.insert("authorization", "Bearer my-secret-key-2".parse().unwrap());
        assert_eq!(get_request_api_key(&headers), Some("my-secret-key-2".to_string()));

        headers.insert("authorization", "bearer my-secret-key-3".parse().unwrap());
        assert_eq!(get_request_api_key(&headers), Some("my-secret-key-3".to_string()));

        headers.insert("authorization", "Bearer   ".parse().unwrap());
        assert_eq!(get_request_api_key(&headers), None);

        headers.insert("authorization", "Bearer".parse().unwrap());
        assert_eq!(get_request_api_key(&headers), None);

        headers.insert("authorization", "raw-token".parse().unwrap());
        assert_eq!(get_request_api_key(&headers), Some("raw-token".to_string()));
    }

    #[test]
    fn test_apply_extra_body_merges_and_overrides() {
        let mut extra = HashMap::new();
        extra.insert("repetition_penalty".to_string(), json!(1.1));
        extra.insert("temperature".to_string(), json!(0.2));

        let body = json!({ "model": "qwen", "messages": [], "temperature": 1.0 });
        let merged = apply_extra_body(1, body, &extra);

        assert_eq!(merged["repetition_penalty"], json!(1.1));
        assert_eq!(merged["temperature"], json!(0.2));
        assert_eq!(merged["model"], json!("qwen"));
    }

    #[test]
    fn test_apply_extra_body_refuses_reserved_keys() {
        let mut extra = HashMap::new();
        extra.insert("model".to_string(), json!("hijacked"));
        extra.insert("messages".to_string(), json!([]));
        extra.insert("repetition_penalty".to_string(), json!(1.2));

        let body = json!({ "model": "qwen", "messages": [{ "role": "user" }] });
        let merged = apply_extra_body(1, body, &extra);

        assert_eq!(merged["model"], json!("qwen"));
        assert_eq!(merged["messages"].as_array().unwrap().len(), 1);
        assert_eq!(merged["repetition_penalty"], json!(1.2));
    }

    #[test]
    fn test_apply_extra_body_is_a_noop_when_empty() {
        let body = json!({ "model": "qwen" });
        let merged = apply_extra_body(1, body.clone(), &HashMap::new());
        assert_eq!(merged, body);
    }

    #[test]
    fn test_resolve_upstream_and_body() {
        let default_upstream = UpstreamConfig {
            base_url: "http://default".to_string(),
            api_key: "default-key".to_string(),
            api_format: ApiFormat::OpenAiChat,
            timeout_secs: 10,
            prompt_cache_key: None,
            default_model: None,
            model_map: HashMap::new(),
            vision_model: None,
            strip_tool_result_images: false,
            extra_headers: HashMap::new(),
            extra_body: HashMap::new(),
        };

        let provider_upstream = UpstreamConfig {
            base_url: "http://provider".to_string(),
            api_key: "provider-key".to_string(),
            api_format: ApiFormat::OpenAiResponses,
            timeout_secs: 20,
            prompt_cache_key: None,
            default_model: None,
            model_map: HashMap::new(),
            vision_model: None,
            strip_tool_result_images: false,
            extra_headers: HashMap::new(),
            extra_body: HashMap::new(),
        };

        let mut providers = HashMap::new();
        providers.insert("my-provider".to_string(), provider_upstream.clone());

        let config = ProxyConfig {
            listen: ListenConfig { host: "127.0.0.1".to_string(), port: 8080 },
            upstream: default_upstream.clone(),
            providers,
            usage: None,
        };

        // 1. No provider prefix
        let body1 = json!({ "model": "claude-3-sonnet" });
        let (up1, body1_res) = resolve_upstream_and_body(body1, &config, None);
        assert_eq!(up1.base_url, "http://default");
        assert_eq!(body1_res["model"], "claude-3-sonnet");

        // 2. With provider prefix
        let body2 = json!({ "model": "my-provider/claude-3-sonnet" });
        let (up2, body2_res) = resolve_upstream_and_body(body2, &config, None);
        assert_eq!(up2.base_url, "http://provider");
        assert_eq!(body2_res["model"], "claude-3-sonnet");

        // 3. With non-existent provider prefix
        let body3 = json!({ "model": "other-provider/claude-3-sonnet" });
        let (up3, body3_res) = resolve_upstream_and_body(body3, &config, None);
        assert_eq!(up3.base_url, "http://default");
        assert_eq!(body3_res["model"], "other-provider/claude-3-sonnet");

        // 4. With provider from path
        let body4 = json!({ "model": "claude-3-sonnet" });
        let (up4, body4_res) = resolve_upstream_and_body(body4, &config, Some("my-provider"));
        assert_eq!(up4.base_url, "http://provider");
        assert_eq!(body4_res["model"], "claude-3-sonnet");
    }

    #[test]
    fn test_apply_model_mapping() {
        let mut model_map = HashMap::new();
        model_map.insert("claude-4".to_string(), "mimo-v4.0".to_string());

        // 1. Matched in map
        let body1 = json!({ "model": "claude-4" });
        let res1 = apply_model_mapping(body1, &model_map, Some("fallback-model"));
        assert_eq!(res1["model"], "mimo-v4.0");

        // 2. Fallback to default_model when not matched in map
        let body2 = json!({ "model": "other-model" });
        let res2 = apply_model_mapping(body2, &model_map, Some("fallback-model"));
        assert_eq!(res2["model"], "fallback-model");

        // 3. Keep original when no match and no default_model
        let body3 = json!({ "model": "other-model" });
        let res3 = apply_model_mapping(body3, &model_map, None);
        assert_eq!(res3["model"], "other-model");
    }
}
