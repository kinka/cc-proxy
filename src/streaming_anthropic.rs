use crate::sse::strip_sse_field;
use crate::usage::{Outcome, TokenCounts};
use crate::{log_request_done, RequestLogContext, UsageGuard};
use bytes::Bytes;
use futures::{Stream, StreamExt};
use serde_json::{json, Value};

/// Relay an Anthropic Messages SSE stream unchanged, reading a copy of it for the ledger.
///
/// Bytes go to the client exactly as the upstream sent them, chunk for chunk; the parsing
/// below only ever looks. Usage arrives in two halves — input and cache in `message_start`,
/// cumulative output in `message_delta` (some gateways repeat the input fields there) — so
/// fields are merged as they appear, and `message_stop` settles the request.
pub fn create_passthrough_sse_stream<E: std::error::Error + Send + 'static>(
    stream: impl Stream<Item = Result<Bytes, E>> + Send + 'static,
    request_context: RequestLogContext,
) -> impl Stream<Item = Result<Bytes, std::io::Error>> + Send {
    async_stream::stream! {
        let mut buffer = String::new();
        let mut message_id: Option<String> = None;
        let mut current_model: Option<String> = None;
        let mut pending_usage = json!({});
        let mut stop_reason: Option<String> = None;
        let mut saw_thinking = false;
        let mut saw_tool_use = false;
        let mut thinking_chars = 0usize;
        let mut text_chars = 0usize;
        let mut saw_any_event = false;
        let mut finished = false;

        // Settles `client_aborted` if the caller drops the body before any terminal path runs.
        let mut usage_guard = UsageGuard::new(request_context.clone());

        tokio::pin!(stream);

        while let Some(chunk) = stream.next().await {
            let bytes = match chunk {
                Ok(bytes) => bytes,
                Err(err) => {
                    log::error!(
                        "request.error req_id={} path={} stage=stream_chunk api_format={} stream=true latency_ms={} error={}",
                        request_context.req_id,
                        request_context.path,
                        request_context.api_format,
                        request_context.started_at.elapsed().as_millis(),
                        err,
                    );
                    request_context.record(Outcome::TransportError, Some(200), usage_guard.partial, usage_guard.response_model.as_deref());
                    yield Err(std::io::Error::other(err.to_string()));
                    continue;
                }
            };

            buffer.push_str(&String::from_utf8_lossy(&bytes));
            yield Ok(bytes);

            while let Some(pos) = buffer.find("\n\n") {
                let block = buffer[..pos].to_string();
                buffer = buffer[pos + 2..].to_string();

                for line in block.lines() {
                    let Some(data) = strip_sse_field(line, "data") else {
                        continue;
                    };
                    let Ok(event) = serde_json::from_str::<Value>(data) else {
                        continue;
                    };
                    saw_any_event = true;

                    match event.get("type").and_then(Value::as_str) {
                        Some("message_start") => {
                            let message = event.get("message").unwrap_or(&Value::Null);
                            message_id = message.get("id").and_then(Value::as_str).map(str::to_string);
                            current_model = message.get("model").and_then(Value::as_str).map(str::to_string);
                            merge_usage(&mut pending_usage, message.get("usage"));
                        }
                        Some("content_block_start") => {
                            match event.pointer("/content_block/type").and_then(Value::as_str) {
                                Some("tool_use" | "server_tool_use") => saw_tool_use = true,
                                Some("thinking" | "redacted_thinking") => saw_thinking = true,
                                _ => {}
                            }
                        }
                        Some("content_block_delta") => {
                            let delta = event.get("delta").unwrap_or(&Value::Null);
                            match delta.get("type").and_then(Value::as_str) {
                                Some("thinking_delta") => {
                                    thinking_chars += delta.get("thinking").and_then(Value::as_str).map_or(0, |text| text.chars().count());
                                }
                                Some("text_delta") => {
                                    text_chars += delta.get("text").and_then(Value::as_str).map_or(0, |text| text.chars().count());
                                }
                                _ => {}
                            }
                        }
                        Some("message_delta") => {
                            if let Some(reason) = event.pointer("/delta/stop_reason").and_then(Value::as_str) {
                                stop_reason = Some(reason.to_string());
                            }
                            merge_usage(&mut pending_usage, event.get("usage"));
                        }
                        Some("message_stop") if !finished => {
                            finished = true;
                            log_request_done(
                                &request_context,
                                reqwest::StatusCode::OK,
                                &crate::ResponseLogSummary {
                                    response_id: message_id.clone(),
                                    response_model: current_model.clone(),
                                    stop_reason: stop_reason.clone(),
                                    input_tokens: pending_usage.get("input_tokens").and_then(Value::as_u64).unwrap_or(0),
                                    output_tokens: pending_usage.get("output_tokens").and_then(Value::as_u64).unwrap_or(0),
                                    cache_read_input_tokens: pending_usage.get("cache_read_input_tokens").and_then(Value::as_u64),
                                    cache_creation_input_tokens: pending_usage.get("cache_creation_input_tokens").and_then(Value::as_u64),
                                    has_tool_use: saw_tool_use,
                                    has_thinking: saw_thinking,
                                    thinking_chars,
                                    text_chars,
                                },
                                true,
                            );
                        }
                        Some("error") if !finished => {
                            finished = true;
                            let message = event.pointer("/error/message").and_then(Value::as_str).unwrap_or("");
                            log::warn!(
                                "request.upstream_error req_id={} path={} stream=true api_format={} upstream_status=200 latency_ms={} error_type={} error_message={}",
                                request_context.req_id,
                                request_context.path,
                                request_context.api_format,
                                request_context.started_at.elapsed().as_millis(),
                                event.pointer("/error/type").and_then(Value::as_str).unwrap_or(""),
                                crate::sanitize_error_message(message),
                            );
                            request_context.record(Outcome::UpstreamError, Some(200), usage_guard.partial, current_model.as_deref());
                        }
                        _ => {}
                    }

                    usage_guard.partial = TokenCounts::from_anthropic_usage(&pending_usage, request_context.input_includes_cache());
                    usage_guard.response_model = current_model.clone();
                }
            }
        }

        // Upstream closed the body without `message_stop` (or an `error` event). Passed through
        // as-is the client would hold a `200` stream that just stops, which agents read as an
        // empty reply; an `error` event makes the SDK raise instead, and the ledger says why.
        if !finished {
            let event = json!({
                "type": "error",
                "error": {
                    "type": "api_error",
                    "message": "upstream closed the stream before message_stop",
                }
            });
            yield Ok(Bytes::from(format!("event: error\ndata: {}\n\n", serde_json::to_string(&event).unwrap_or_default())));

            request_context.record(Outcome::StreamTruncated, Some(200), usage_guard.partial, current_model.as_deref());
            log::error!(
                "request.error req_id={} path={} stage=stream_truncated api_format={} stream=true latency_ms={} response_id={} response_model={} stop_reason={} saw_any_event={} has_tool_use={} has_thinking={} thinking_chars={} text_chars={} trailing_bytes={} error=upstream stream ended without message_stop",
                request_context.req_id,
                request_context.path,
                request_context.api_format,
                request_context.started_at.elapsed().as_millis(),
                message_id.as_deref().unwrap_or(""),
                current_model.as_deref().unwrap_or(""),
                stop_reason.as_deref().unwrap_or(""),
                saw_any_event,
                saw_tool_use,
                saw_thinking,
                thinking_chars,
                text_chars,
                buffer.len(),
            );
        }
    }
}

/// Copy the numeric usage fields present in `source` over `target`. Missing or `null` fields
/// leave the earlier value alone: `message_delta` usually carries only `output_tokens`.
fn merge_usage(target: &mut Value, source: Option<&Value>) {
    let (Some(target), Some(source)) = (target.as_object_mut(), source.and_then(Value::as_object)) else {
        return;
    };
    for (key, value) in source {
        if value.is_u64() {
            target.insert(key.clone(), value.clone());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn chunks(chunks: Vec<&'static str>) -> impl Stream<Item = Result<Bytes, std::io::Error>> + Send {
        futures::stream::iter(chunks.into_iter().map(|chunk| Ok::<_, std::io::Error>(Bytes::from(chunk))))
    }

    fn collect(out: impl Stream<Item = Result<Bytes, std::io::Error>>) -> String {
        futures::executor::block_on(async move {
            futures::pin_mut!(out);
            let mut acc = String::new();
            while let Some(item) = out.next().await {
                acc.push_str(&String::from_utf8_lossy(&item.unwrap()));
            }
            acc
        })
    }

    const START: &str = "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\",\"model\":\"m\",\"usage\":{\"input_tokens\":40,\"cache_read_input_tokens\":60,\"cache_creation_input_tokens\":0,\"output_tokens\":1}}}\n\n";
    const TEXT: &str = "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"你好\"}}\n\n";
    const DELTA: &str = "event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":7}}\n\n";
    const STOP: &str = "event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n";

    #[test]
    fn bytes_pass_through_unchanged() {
        let context = RequestLogContext::for_test("/v1/messages", "anthropic");
        // Split mid-event to make sure re-chunking never happens.
        let body = collect(create_passthrough_sse_stream(chunks(vec![START, &TEXT[..30], &TEXT[30..], DELTA, STOP]), context));
        assert_eq!(body, [START, TEXT, DELTA, STOP].concat());
    }

    #[test]
    fn completed_stream_records_done_with_cache_added_to_prompt() {
        let (context, dir) = crate::usage::tests::tracked_context("/v1/messages", "anthropic");
        collect(create_passthrough_sse_stream(chunks(vec![START, TEXT, DELTA, STOP]), context));
        let records = crate::usage::tests::take_records(dir, 1);
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].outcome, Outcome::Done);
        // Native Anthropic input_tokens (40) excludes the 60 cached; prompt must count both.
        assert_eq!(records[0].tokens.prompt_tokens, 100);
        assert_eq!(records[0].tokens.cache_read_tokens, 60);
        assert_eq!(records[0].tokens.output_tokens, 7);
    }

    #[test]
    fn stream_without_message_stop_is_truncated_and_ends_in_an_error_event() {
        let (context, dir) = crate::usage::tests::tracked_context("/v1/messages", "anthropic");
        let body = collect(create_passthrough_sse_stream(chunks(vec![START, TEXT]), context));
        assert!(body.ends_with("\n\n") && body.contains("event: error"));
        let records = crate::usage::tests::take_records(dir, 1);
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].outcome, Outcome::StreamTruncated);
        assert_eq!(records[0].tokens.total(), 101);
    }

    #[test]
    fn upstream_error_event_records_upstream_error_and_adds_nothing() {
        let (context, dir) = crate::usage::tests::tracked_context("/v1/messages", "anthropic");
        let error = "event: error\ndata: {\"type\":\"error\",\"error\":{\"type\":\"overloaded_error\",\"message\":\"Overloaded\"}}\n\n";
        let body = collect(create_passthrough_sse_stream(chunks(vec![START, error]), context));
        assert_eq!(body, [START, error].concat());
        let records = crate::usage::tests::take_records(dir, 1);
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].outcome, Outcome::UpstreamError);
    }

    #[test]
    fn dropping_the_stream_early_records_client_aborted() {
        let (context, dir) = crate::usage::tests::tracked_context("/v1/messages", "anthropic");
        let out = create_passthrough_sse_stream(chunks(vec![START, TEXT, DELTA, STOP]), context);
        futures::executor::block_on(async move {
            futures::pin_mut!(out);
            let _ = out.next().await;
            let _ = out.next().await;
        });
        let records = crate::usage::tests::take_records(dir, 1);
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].outcome, Outcome::ClientAborted);
        assert_eq!(records[0].tokens.prompt_tokens, 100);
    }
}
