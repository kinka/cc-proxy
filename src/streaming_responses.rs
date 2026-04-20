use crate::sse::strip_sse_field;
use crate::transform_responses::{build_anthropic_usage_from_responses, map_responses_stop_reason};
use crate::{log_request_done, RequestLogContext, ResponseLogSummary};
use bytes::Bytes;
use futures::{Stream, StreamExt};
use serde_json::{json, Value};
use std::collections::{HashMap, HashSet};

pub fn create_anthropic_sse_stream_from_responses<E: std::error::Error + Send + 'static>(
    stream: impl Stream<Item = Result<Bytes, E>> + Send + 'static,
    request_context: RequestLogContext,
) -> impl Stream<Item = Result<Bytes, std::io::Error>> + Send {
    async_stream::stream! {
        let mut buffer = String::new();
        let mut message_id: Option<String> = None;
        let mut current_model: Option<String> = None;
        let mut has_sent_message_start = false;
        let mut has_tool_use = false;
        let mut next_content_index: u32 = 0;
        let mut index_by_key: HashMap<String, u32> = HashMap::new();
        let mut open_indices: HashSet<u32> = HashSet::new();
        let mut fallback_open_index: Option<u32> = None;
        let mut current_text_index: Option<u32> = None;
        let mut tool_index_by_item_id: HashMap<String, u32> = HashMap::new();
        let mut last_tool_index: Option<u32> = None;
        let mut saw_thinking = false;
        let mut logged_completion = false;
        let mut sent_message_stop = false;

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
                    yield Err(std::io::Error::other(err.to_string()));
                    continue;
                }
            };

            let text = String::from_utf8_lossy(&bytes);
            buffer.push_str(&text);

            while let Some(pos) = buffer.find("\n\n") {
                let block = buffer[..pos].to_string();
                buffer = buffer[pos + 2..].to_string();

                if block.trim().is_empty() {
                    continue;
                }

                let mut event_name: Option<String> = None;
                let mut data_lines: Vec<String> = Vec::new();
                for line in block.lines() {
                    if let Some(event) = strip_sse_field(line, "event") {
                        event_name = Some(event.trim().to_string());
                    } else if let Some(data) = strip_sse_field(line, "data") {
                        data_lines.push(data.to_string());
                    }
                }

                if data_lines.is_empty() {
                    continue;
                }

                let data_str = data_lines.join("\n");
                let data = match serde_json::from_str::<Value>(&data_str) {
                    Ok(data) => data,
                    Err(_) => continue,
                };

                match event_name.as_deref().unwrap_or("") {
                    "response.created" => {
                        let response = response_object_from_event(&data);
                        if let Some(id) = response.get("id").and_then(|value| value.as_str()) {
                            message_id = Some(id.to_string());
                        }
                        if let Some(model) = response.get("model").and_then(|value| value.as_str()) {
                            current_model = Some(model.to_string());
                        }

                        let event = json!({
                            "type": "message_start",
                            "message": {
                                "id": message_id.clone().unwrap_or_default(),
                                "type": "message",
                                "role": "assistant",
                                "model": current_model.clone().unwrap_or_default(),
                                "usage": build_anthropic_usage_from_responses(response.get("usage")),
                            }
                        });
                        yield Ok(Bytes::from(format!("event: message_start\ndata: {}\n\n", serde_json::to_string(&event).unwrap_or_default())));
                        has_sent_message_start = true;
                    }
                    "response.content_part.added" => {
                        if !has_sent_message_start {
                            let event = json!({
                                "type": "message_start",
                                "message": {
                                    "id": message_id.clone().unwrap_or_default(),
                                    "type": "message",
                                    "role": "assistant",
                                    "model": current_model.clone().unwrap_or_default(),
                                    "usage": {
                                        "input_tokens": 0,
                                        "output_tokens": 0,
                                    }
                                }
                            });
                            yield Ok(Bytes::from(format!("event: message_start\ndata: {}\n\n", serde_json::to_string(&event).unwrap_or_default())));
                            has_sent_message_start = true;
                        }

                        if let Some(part) = data.get("part") {
                            let part_type = part.get("type").and_then(|value| value.as_str());
                            if matches!(part_type, Some("output_text") | Some("refusal")) {
                                let index = current_text_index.unwrap_or_else(|| {
                                    let assigned = resolve_content_index(
                                        &data,
                                        &mut next_content_index,
                                        &mut index_by_key,
                                        &mut fallback_open_index,
                                    );
                                    current_text_index = Some(assigned);
                                    assigned
                                });

                                if !open_indices.contains(&index) {
                                    let event = json!({
                                        "type": "content_block_start",
                                        "index": index,
                                        "content_block": {
                                            "type": "text",
                                            "text": "",
                                        }
                                    });
                                    yield Ok(Bytes::from(format!("event: content_block_start\ndata: {}\n\n", serde_json::to_string(&event).unwrap_or_default())));
                                    open_indices.insert(index);
                                }
                            }
                        }
                    }
                    "response.output_text.delta" | "response.refusal.delta" => {
                        let Some(delta) = data.get("delta").and_then(|value| value.as_str()) else {
                            continue;
                        };

                        let index = current_text_index.unwrap_or_else(|| {
                            let assigned = resolve_content_index(
                                &data,
                                &mut next_content_index,
                                &mut index_by_key,
                                &mut fallback_open_index,
                            );
                            current_text_index = Some(assigned);
                            assigned
                        });

                        if !open_indices.contains(&index) {
                            let event = json!({
                                "type": "content_block_start",
                                "index": index,
                                "content_block": {
                                    "type": "text",
                                    "text": "",
                                }
                            });
                            yield Ok(Bytes::from(format!("event: content_block_start\ndata: {}\n\n", serde_json::to_string(&event).unwrap_or_default())));
                            open_indices.insert(index);
                        }

                        let event = json!({
                            "type": "content_block_delta",
                            "index": index,
                            "delta": {
                                "type": "text_delta",
                                "text": delta,
                            }
                        });
                        yield Ok(Bytes::from(format!("event: content_block_delta\ndata: {}\n\n", serde_json::to_string(&event).unwrap_or_default())));
                    }
                    "response.output_item.added" => {
                        let Some(item) = data.get("item") else {
                            continue;
                        };
                        if item.get("type").and_then(|value| value.as_str()) != Some("function_call") {
                            continue;
                        }

                        has_tool_use = true;
                        let tool_index = if let Some(index) = tool_item_key_from_added(&data, item)
                            .and_then(|key| tool_index_by_item_id.get(&key).copied())
                        {
                            index
                        } else {
                            let assigned = next_content_index;
                            next_content_index += 1;
                            if let Some(key) = tool_item_key_from_added(&data, item) {
                                tool_index_by_item_id.insert(key, assigned);
                            }
                            assigned
                        };
                        last_tool_index = Some(tool_index);

                        let event = json!({
                            "type": "content_block_start",
                            "index": tool_index,
                            "content_block": {
                                "type": "tool_use",
                                "id": item.get("call_id").and_then(|value| value.as_str()).unwrap_or(""),
                                "name": item.get("name").and_then(|value| value.as_str()).unwrap_or(""),
                                "input": {},
                            }
                        });
                        yield Ok(Bytes::from(format!("event: content_block_start\ndata: {}\n\n", serde_json::to_string(&event).unwrap_or_default())));
                        open_indices.insert(tool_index);
                    }
                    "response.function_call_arguments.delta" => {
                        let Some(delta) = data.get("delta").and_then(|value| value.as_str()) else {
                            continue;
                        };
                        let index = tool_item_key_from_event(&data)
                            .and_then(|key| tool_index_by_item_id.get(&key).copied())
                            .or(last_tool_index);

                        if let Some(index) = index {
                            let event = json!({
                                "type": "content_block_delta",
                                "index": index,
                                "delta": {
                                    "type": "input_json_delta",
                                    "partial_json": delta,
                                }
                            });
                            yield Ok(Bytes::from(format!("event: content_block_delta\ndata: {}\n\n", serde_json::to_string(&event).unwrap_or_default())));
                        }
                    }
                    "response.reasoning.delta" => {
                        let Some(delta) = data
                            .pointer("/delta/summary_text")
                            .and_then(|value| value.as_str())
                            .or_else(|| data.get("delta").and_then(|value| value.as_str()))
                        else {
                            continue;
                        };
                        saw_thinking = true;

                        let index = resolve_content_index(
                            &data,
                            &mut next_content_index,
                            &mut index_by_key,
                            &mut fallback_open_index,
                        );
                        if !open_indices.contains(&index) {
                            let event = json!({
                                "type": "content_block_start",
                                "index": index,
                                "content_block": {
                                    "type": "thinking",
                                    "thinking": "",
                                }
                            });
                            yield Ok(Bytes::from(format!("event: content_block_start\ndata: {}\n\n", serde_json::to_string(&event).unwrap_or_default())));
                            open_indices.insert(index);
                        }
                        let event = json!({
                            "type": "content_block_delta",
                            "index": index,
                            "delta": {
                                "type": "thinking_delta",
                                "thinking": delta,
                            }
                        });
                        yield Ok(Bytes::from(format!("event: content_block_delta\ndata: {}\n\n", serde_json::to_string(&event).unwrap_or_default())));
                    }
                    "response.completed" => {
                        for index in open_indices.drain() {
                            let event = json!({ "type": "content_block_stop", "index": index });
                            yield Ok(Bytes::from(format!("event: content_block_stop\ndata: {}\n\n", serde_json::to_string(&event).unwrap_or_default())));
                        }
                        current_text_index = None;

                        let response = response_object_from_event(&data);
                        let event = json!({
                            "type": "message_delta",
                            "delta": {
                                "stop_reason": map_responses_stop_reason(
                                    response.get("status").and_then(|value| value.as_str()),
                                    has_tool_use,
                                    response.pointer("/incomplete_details/reason").and_then(|value| value.as_str()),
                                ),
                                "stop_sequence": null,
                            },
                            "usage": build_anthropic_usage_from_responses(response.get("usage")),
                        });
                        yield Ok(Bytes::from(format!("event: message_delta\ndata: {}\n\n", serde_json::to_string(&event).unwrap_or_default())));

                        if !sent_message_stop {
                            yield Ok(Bytes::from("event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n"));
                            sent_message_stop = true;
                        }

                        if !logged_completion {
                            let usage = build_anthropic_usage_from_responses(response.get("usage"));
                            log_request_done(
                                &request_context,
                                reqwest::StatusCode::OK,
                                &ResponseLogSummary {
                                    response_id: message_id.clone(),
                                    response_model: current_model.clone(),
                                    stop_reason: map_responses_stop_reason(
                                        response.get("status").and_then(|value| value.as_str()),
                                        has_tool_use,
                                        response.pointer("/incomplete_details/reason").and_then(|value| value.as_str()),
                                    )
                                    .map(str::to_string),
                                    input_tokens: usage.get("input_tokens").and_then(|value| value.as_u64()).unwrap_or(0),
                                    output_tokens: usage.get("output_tokens").and_then(|value| value.as_u64()).unwrap_or(0),
                                    cache_read_input_tokens: usage.get("cache_read_input_tokens").and_then(|value| value.as_u64()),
                                    cache_creation_input_tokens: usage.get("cache_creation_input_tokens").and_then(|value| value.as_u64()),
                                    has_tool_use,
                                    has_thinking: saw_thinking,
                                },
                                true,
                            );
                            logged_completion = true;
                        }
                    }
                    _ => {}
                }
            }
        }
    }
}

fn response_object_from_event(data: &Value) -> &Value {
    data.get("response").unwrap_or(data)
}

fn content_part_key(data: &Value) -> Option<String> {
    if let (Some(item_id), Some(content_index)) = (
        data.get("item_id").and_then(|value| value.as_str()),
        data.get("content_index").and_then(|value| value.as_u64()),
    ) {
        return Some(format!("part:{item_id}:{content_index}"));
    }
    if let (Some(output_index), Some(content_index)) = (
        data.get("output_index").and_then(|value| value.as_u64()),
        data.get("content_index").and_then(|value| value.as_u64()),
    ) {
        return Some(format!("part:out:{output_index}:{content_index}"));
    }
    None
}

fn resolve_content_index(
    data: &Value,
    next_content_index: &mut u32,
    index_by_key: &mut HashMap<String, u32>,
    fallback_open_index: &mut Option<u32>,
) -> u32 {
    if let Some(key) = content_part_key(data) {
        if let Some(index) = index_by_key.get(&key).copied() {
            index
        } else {
            let assigned = *next_content_index;
            *next_content_index += 1;
            index_by_key.insert(key, assigned);
            assigned
        }
    } else if let Some(index) = *fallback_open_index {
        index
    } else {
        let assigned = *next_content_index;
        *next_content_index += 1;
        *fallback_open_index = Some(assigned);
        assigned
    }
}

fn tool_item_key_from_added(data: &Value, item: &Value) -> Option<String> {
    if let Some(item_id) = item.get("id").and_then(|value| value.as_str()) {
        return Some(format!("tool:{item_id}"));
    }
    if let Some(item_id) = data.get("item_id").and_then(|value| value.as_str()) {
        return Some(format!("tool:{item_id}"));
    }
    data.get("output_index")
        .and_then(|value| value.as_u64())
        .map(|index| format!("tool:out:{index}"))
}

fn tool_item_key_from_event(data: &Value) -> Option<String> {
    if let Some(item_id) = data.get("item_id").and_then(|value| value.as_str()) {
        return Some(format!("tool:{item_id}"));
    }
    data.get("output_index")
        .and_then(|value| value.as_u64())
        .map(|index| format!("tool:out:{index}"))
}
