use crate::sse::strip_sse_field;
use crate::transform::{DeltaToolCall, OpenAIStreamChunk};
use crate::transform_responses::openai_chat_usage_to_responses_usage;
use crate::{log_request_done, RequestLogContext, ResponseLogSummary};
use bytes::Bytes;
use futures::{Stream, StreamExt};
use serde_json::{json, Value};
use std::collections::{HashMap, HashSet};

#[derive(Debug, Clone)]
struct ToolCallState {
    id: String,
    name: String,
    arguments: String,
    output_index: usize,
    started: bool,
}

pub fn create_responses_sse_stream_from_chat<E: std::error::Error + Send + 'static>(
    stream: impl Stream<Item = Result<Bytes, E>> + Send + 'static,
    request_context: RequestLogContext,
) -> impl Stream<Item = Result<Bytes, std::io::Error>> + Send {
    async_stream::stream! {
        let mut buffer = String::new();
        let mut response_id: Option<String> = None;
        let mut current_model: Option<String> = None;
        let mut output_index: usize = 0;
        let mut text_item_id: Option<String> = None;
        let mut text_buffer = String::new();
        let mut text_started = false;
        let mut completed_output: Vec<Value> = Vec::new();
        let mut completed_output_text = String::new();
        let mut completed_tool_call_ids: HashSet<String> = HashSet::new();
        let mut tool_calls_by_index: HashMap<usize, ToolCallState> = HashMap::new();
        let mut pending_usage = json!({
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
        });
        let mut pending_finish_reason: Option<String> = None;
        let mut logged_completion = false;

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

                for line in block.lines() {
                    let Some(data) = strip_sse_field(line, "data") else {
                        continue;
                    };

                    if data.trim() == "[DONE]" {
                        for event in finish_text_item(
                            &mut text_started,
                            &mut text_item_id,
                            &mut text_buffer,
                            &mut output_index,
                            &mut completed_output,
                            &mut completed_output_text,
                        ) {
                            yield Ok(event);
                        }

                        for event in finish_pending_tool_calls(
                            &tool_calls_by_index,
                            &mut completed_tool_call_ids,
                            &mut completed_output,
                        ) {
                            yield Ok(event);
                        }

                        let id = response_id.clone().unwrap_or_default();
                        let model = current_model.clone().unwrap_or_default();
                        let status = if pending_finish_reason.as_deref() == Some("length") { "incomplete" } else { "completed" };
                        let mut response = json!({
                            "id": id,
                            "object": "response",
                            "status": status,
                            "model": model,
                            "output": std::mem::take(&mut completed_output),
                            "output_text": completed_output_text,
                            "usage": pending_usage.clone(),
                        });
                        if pending_finish_reason.as_deref() == Some("length") {
                            response["incomplete_details"] = json!({ "reason": "max_output_tokens" });
                        }
                        let completed = json!({
                            "type": "response.completed",
                            "response": response,
                        });
                        yield Ok(Bytes::from(format!("event: response.completed\ndata: {}\n\n", serde_json::to_string(&completed).unwrap_or_default())));
                        yield Ok(Bytes::from("data: [DONE]\n\n"));

                        if !logged_completion {
                            log_request_done(
                                &request_context,
                                reqwest::StatusCode::OK,
                                &ResponseLogSummary {
                                    response_id: response_id.clone(),
                                    response_model: current_model.clone(),
                                    stop_reason: pending_finish_reason.clone(),
                                    input_tokens: pending_usage.get("input_tokens").and_then(|value| value.as_u64()).unwrap_or(0),
                                    output_tokens: pending_usage.get("output_tokens").and_then(|value| value.as_u64()).unwrap_or(0),
                                    cache_read_input_tokens: pending_usage.pointer("/input_tokens_details/cached_tokens").and_then(|value| value.as_u64()),
                                    cache_creation_input_tokens: None,
                                    has_tool_use: !tool_calls_by_index.is_empty(),
                                    has_thinking: false,
                                },
                                true,
                            );
                            logged_completion = true;
                        }
                        continue;
                    }

                    let parsed = match serde_json::from_str::<OpenAIStreamChunk>(data) {
                        Ok(parsed) => parsed,
                        Err(_) => continue,
                    };

                    if response_id.is_none() {
                        response_id = Some(parsed.id.clone());
                        current_model = Some(parsed.model.clone());
                        let response = json!({
                            "id": parsed.id,
                            "object": "response",
                            "status": "in_progress",
                            "model": parsed.model,
                            "output": [],
                            "usage": null,
                        });
                        let event = json!({
                            "type": "response.created",
                            "response": response,
                        });
                        yield Ok(Bytes::from(format!("event: response.created\ndata: {}\n\n", serde_json::to_string(&event).unwrap_or_default())));
                    }

                    if let Some(usage) = &parsed.usage {
                        let mut usage_value = json!({
                            "prompt_tokens": usage.prompt_tokens,
                            "completion_tokens": usage.completion_tokens,
                            "total_tokens": usage.prompt_tokens + usage.completion_tokens,
                        });
                        if let Some(details) = &usage.prompt_tokens_details {
                            usage_value["prompt_tokens_details"] = json!({ "cached_tokens": details.cached_tokens });
                        }
                        pending_usage = openai_chat_usage_to_responses_usage(Some(&usage_value));
                    }

                    let Some(choice) = parsed.choices.first() else {
                        continue;
                    };

                    if let Some(content) = &choice.delta.content {
                        if !content.is_empty() {
                            if !text_started {
                                let item_id = format!("msg_{}", response_id.as_deref().unwrap_or_default());
                                text_item_id = Some(item_id.clone());
                                let item = json!({
                                    "id": item_id,
                                    "type": "message",
                                    "status": "in_progress",
                                    "role": "assistant",
                                    "content": [],
                                });
                                let item_event = json!({
                                    "type": "response.output_item.added",
                                    "output_index": output_index,
                                    "item": item,
                                });
                                yield Ok(Bytes::from(format!("event: response.output_item.added\ndata: {}\n\n", serde_json::to_string(&item_event).unwrap_or_default())));

                                let part_event = json!({
                                    "type": "response.content_part.added",
                                    "item_id": text_item_id.clone().unwrap_or_default(),
                                    "output_index": output_index,
                                    "content_index": 0,
                                    "part": { "type": "output_text", "text": "", "annotations": [] },
                                });
                                yield Ok(Bytes::from(format!("event: response.content_part.added\ndata: {}\n\n", serde_json::to_string(&part_event).unwrap_or_default())));
                                text_started = true;
                            }

                            let delta_event = json!({
                                "type": "response.output_text.delta",
                                "item_id": text_item_id.clone().unwrap_or_default(),
                                "output_index": output_index,
                                "content_index": 0,
                                "delta": content,
                            });
                            text_buffer.push_str(content);
                            yield Ok(Bytes::from(format!("event: response.output_text.delta\ndata: {}\n\n", serde_json::to_string(&delta_event).unwrap_or_default())));
                        }
                    }

                    if let Some(tool_calls) = &choice.delta.tool_calls {
                        if text_started {
                            for event in finish_text_item(
                                &mut text_started,
                                &mut text_item_id,
                                &mut text_buffer,
                                &mut output_index,
                                &mut completed_output,
                                &mut completed_output_text,
                            ) {
                                yield Ok(event);
                            }
                        }

                        for tool_call in tool_calls {
                            let state = tool_calls_by_index.entry(tool_call.index).or_insert_with(|| build_tool_call_state(tool_call, output_index));
                            update_tool_call_state(state, tool_call);
                            if !state.started {
                                let item = json!({
                                    "id": state.id,
                                    "type": "function_call",
                                    "status": "in_progress",
                                    "call_id": state.id,
                                    "name": state.name,
                                    "arguments": "",
                                });
                                let event = json!({
                                    "type": "response.output_item.added",
                                    "output_index": state.output_index,
                                    "item": item,
                                });
                                yield Ok(Bytes::from(format!("event: response.output_item.added\ndata: {}\n\n", serde_json::to_string(&event).unwrap_or_default())));
                                state.started = true;
                                output_index = output_index.max(state.output_index + 1);
                            }
                            if let Some(arguments) = tool_call.function.as_ref().and_then(|function| function.arguments.as_ref()) {
                                if !arguments.is_empty() {
                                    state.arguments.push_str(arguments);
                                    let event = json!({
                                        "type": "response.function_call_arguments.delta",
                                        "item_id": state.id,
                                        "output_index": state.output_index,
                                        "delta": arguments,
                                    });
                                    yield Ok(Bytes::from(format!("event: response.function_call_arguments.delta\ndata: {}\n\n", serde_json::to_string(&event).unwrap_or_default())));
                                }
                            }
                        }
                    }

                    if let Some(finish_reason) = choice.finish_reason.as_deref() {
                        pending_finish_reason = Some(finish_reason.to_string());
                        if text_started {
                            for event in finish_text_item(
                                &mut text_started,
                                &mut text_item_id,
                                &mut text_buffer,
                                &mut output_index,
                                &mut completed_output,
                                &mut completed_output_text,
                            ) {
                                yield Ok(event);
                            }
                        }

                        for event in finish_pending_tool_calls(
                            &tool_calls_by_index,
                            &mut completed_tool_call_ids,
                            &mut completed_output,
                        ) {
                            yield Ok(event);
                        }
                    }
                }
            }
        }
    }
}

fn build_tool_call_state(tool_call: &DeltaToolCall, output_index: usize) -> ToolCallState {
    ToolCallState {
        id: tool_call.id.clone().unwrap_or_else(|| format!("call_{}", tool_call.index)),
        name: tool_call
            .function
            .as_ref()
            .and_then(|function| function.name.clone())
            .unwrap_or_default(),
        arguments: String::new(),
        output_index,
        started: false,
    }
}

fn update_tool_call_state(state: &mut ToolCallState, tool_call: &DeltaToolCall) {
    if let Some(id) = &tool_call.id {
        state.id = id.clone();
    }
    if let Some(name) = tool_call.function.as_ref().and_then(|function| function.name.as_ref()) {
        state.name = name.clone();
    }
}

fn finish_text_item(
    text_started: &mut bool,
    text_item_id: &mut Option<String>,
    text_buffer: &mut String,
    output_index: &mut usize,
    completed_output: &mut Vec<Value>,
    completed_output_text: &mut String,
) -> Vec<Bytes> {
    if !*text_started {
        return Vec::new();
    }

    let item_id = text_item_id.clone().unwrap_or_default();
    let text = std::mem::take(text_buffer);
    completed_output_text.push_str(&text);

    let text_done = json!({
        "type": "response.output_text.done",
        "item_id": item_id.clone(),
        "output_index": *output_index,
        "content_index": 0,
        "text": text.clone(),
    });
    let part_done = json!({
        "type": "response.content_part.done",
        "item_id": item_id.clone(),
        "output_index": *output_index,
        "content_index": 0,
        "part": { "type": "output_text", "text": text.clone(), "annotations": [] },
    });
    let item = json!({
        "id": item_id,
        "type": "message",
        "status": "completed",
        "role": "assistant",
        "content": [{ "type": "output_text", "text": text, "annotations": [] }],
    });
    completed_output.push(item.clone());
    let item_done = json!({
        "type": "response.output_item.done",
        "output_index": *output_index,
        "item": item,
    });

    *output_index += 1;
    *text_started = false;
    *text_item_id = None;

    vec![
        sse_event("response.output_text.done", text_done),
        sse_event("response.content_part.done", part_done),
        sse_event("response.output_item.done", item_done),
    ]
}

fn finish_pending_tool_calls(
    tool_calls_by_index: &HashMap<usize, ToolCallState>,
    completed_tool_call_ids: &mut HashSet<String>,
    completed_output: &mut Vec<Value>,
) -> Vec<Bytes> {
    let mut events = Vec::new();
    for state in tool_calls_by_index.values() {
        if completed_tool_call_ids.contains(&state.id) {
            continue;
        }
        let done = json!({
            "type": "response.function_call_arguments.done",
            "item_id": state.id,
            "output_index": state.output_index,
            "arguments": state.arguments,
        });
        events.push(sse_event("response.function_call_arguments.done", done));

        let item = json!({
            "id": state.id,
            "type": "function_call",
            "status": "completed",
            "call_id": state.id,
            "name": state.name,
            "arguments": state.arguments,
        });
        completed_output.push(item.clone());
        completed_tool_call_ids.insert(state.id.clone());
        let item_done = json!({
            "type": "response.output_item.done",
            "output_index": state.output_index,
            "item": item,
        });
        events.push(sse_event("response.output_item.done", item_done));
    }
    events
}

fn sse_event(event: &str, data: Value) -> Bytes {
    Bytes::from(format!(
        "event: {}\ndata: {}\n\n",
        event,
        serde_json::to_string(&data).unwrap_or_default()
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::TryStreamExt;
    use std::time::Instant;

    #[tokio::test]
    async fn chat_stream_emits_completed_output_item_and_response_output() {
        let chunks = futures::stream::iter(vec![
            Ok::<_, std::io::Error>(Bytes::from("data: {\"id\":\"chatcmpl_1\",\"model\":\"test-model\",\"choices\":[{\"delta\":{\"content\":\"he\"}}]}\n\n")),
            Ok::<_, std::io::Error>(Bytes::from("data: {\"id\":\"chatcmpl_1\",\"model\":\"test-model\",\"choices\":[{\"delta\":{\"content\":\"llo\"},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":3,\"completion_tokens\":2}}\n\n")),
            Ok::<_, std::io::Error>(Bytes::from("data: [DONE]\n\n")),
        ]);
        let context = RequestLogContext {
            req_id: 1,
            path: "/v1/responses".to_string(),
            api_format: "responses_to_openai_chat",
            started_at: Instant::now(),
        };

        let body = create_responses_sse_stream_from_chat(chunks, context)
            .try_fold(String::new(), |mut body, bytes| async move {
                body.push_str(&String::from_utf8(bytes.to_vec()).unwrap());
                Ok(body)
            })
            .await
            .unwrap();

        let output_item_done = sse_event_json(&body, "response.output_item.done");
        assert_eq!(output_item_done["item"]["content"][0]["text"], "hello");
        let completed = sse_event_json(&body, "response.completed");
        assert_eq!(completed["response"]["output"][0]["content"][0]["text"], "hello");
        assert_eq!(completed["response"]["output_text"], "hello");
    }

    fn sse_event_json(body: &str, event_name: &str) -> Value {
        for block in body.split("\n\n") {
            if !block.lines().any(|line| line == format!("event: {event_name}")) {
                continue;
            }
            let data = block
                .lines()
                .find_map(|line| strip_sse_field(line, "data"))
                .unwrap();
            return serde_json::from_str(data).unwrap();
        }
        panic!("missing event {event_name}");
    }
}
