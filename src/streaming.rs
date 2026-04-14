use crate::sse::strip_sse_field;
use crate::transform::{
    extract_cache_read_tokens, DeltaToolCall, OpenAIStreamChunk,
};
use bytes::Bytes;
use futures::{Stream, StreamExt};
use serde_json::json;
use std::collections::{HashMap, HashSet};

#[derive(Debug, Clone)]
struct ToolBlockState {
    anthropic_index: u32,
    id: String,
    name: String,
    started: bool,
    pending_args: String,
}

pub fn create_anthropic_sse_stream<E: std::error::Error + Send + 'static>(
    stream: impl Stream<Item = Result<Bytes, E>> + Send + 'static,
) -> impl Stream<Item = Result<Bytes, std::io::Error>> + Send {
    async_stream::stream! {
        let mut buffer = String::new();
        let mut message_id: Option<String> = None;
        let mut current_model: Option<String> = None;
        let mut next_content_index: u32 = 0;
        let mut has_sent_message_start = false;
        let mut current_non_tool_block_type: Option<&'static str> = None;
        let mut current_non_tool_block_index: Option<u32> = None;
        let mut tool_blocks_by_index: HashMap<usize, ToolBlockState> = HashMap::new();
        let mut open_tool_block_indices: HashSet<u32> = HashSet::new();
        let mut pending_usage = json!({
            "input_tokens": 0,
            "output_tokens": 0,
        });
        let mut sent_message_stop = false;

        tokio::pin!(stream);

        while let Some(chunk) = stream.next().await {
            let bytes = match chunk {
                Ok(bytes) => bytes,
                Err(err) => {
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
                        if !sent_message_stop {
                            yield Ok(Bytes::from("event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n"));
                            sent_message_stop = true;
                        }
                        continue;
                    }

                    let parsed = match serde_json::from_str::<OpenAIStreamChunk>(data) {
                        Ok(parsed) => parsed,
                        Err(_) => continue,
                    };

                    if message_id.is_none() {
                        message_id = Some(parsed.id.clone());
                    }
                    if current_model.is_none() {
                        current_model = Some(parsed.model.clone());
                    }

                    if let Some(usage) = &parsed.usage {
                        pending_usage["input_tokens"] = json!(usage.prompt_tokens);
                        pending_usage["output_tokens"] = json!(usage.completion_tokens);
                        if let Some(cached) = extract_cache_read_tokens(usage) {
                            pending_usage["cache_read_input_tokens"] = json!(cached);
                        }
                        if let Some(created) = usage.cache_creation_input_tokens {
                            pending_usage["cache_creation_input_tokens"] = json!(created);
                        }
                    }

                    let Some(choice) = parsed.choices.first() else {
                        continue;
                    };

                    if !has_sent_message_start {
                        let event = json!({
                            "type": "message_start",
                            "message": {
                                "id": message_id.clone().unwrap_or_default(),
                                "type": "message",
                                "role": "assistant",
                                "model": current_model.clone().unwrap_or_default(),
                                "usage": pending_usage.clone(),
                            }
                        });
                        yield Ok(Bytes::from(format!("event: message_start\ndata: {}\n\n", serde_json::to_string(&event).unwrap_or_default())));
                        has_sent_message_start = true;
                    }

                    // Support both reasoning and reasoning_content fields
                    let reasoning_text = choice.delta.reasoning.as_ref()
                        .or(choice.delta.reasoning_content.as_ref());

                    if let Some(reasoning) = reasoning_text {
                        if current_non_tool_block_type != Some("thinking") {
                            if let Some(index) = current_non_tool_block_index.take() {
                                let event = json!({ "type": "content_block_stop", "index": index });
                                yield Ok(Bytes::from(format!("event: content_block_stop\ndata: {}\n\n", serde_json::to_string(&event).unwrap_or_default())));
                            }
                            let index = next_content_index;
                            next_content_index += 1;
                            let event = json!({
                                "type": "content_block_start",
                                "index": index,
                                "content_block": {
                                    "type": "thinking",
                                    "thinking": "",
                                }
                            });
                            yield Ok(Bytes::from(format!("event: content_block_start\ndata: {}\n\n", serde_json::to_string(&event).unwrap_or_default())));
                            current_non_tool_block_type = Some("thinking");
                            current_non_tool_block_index = Some(index);
                        }

                        if let Some(index) = current_non_tool_block_index {
                            let event = json!({
                                "type": "content_block_delta",
                                "index": index,
                                "delta": {
                                    "type": "thinking_delta",
                                    "thinking": reasoning,
                                }
                            });
                            yield Ok(Bytes::from(format!("event: content_block_delta\ndata: {}\n\n", serde_json::to_string(&event).unwrap_or_default())));
                        }
                    }

                    if let Some(content) = &choice.delta.content {
                        if !content.is_empty() {
                            if current_non_tool_block_type != Some("text") {
                                if let Some(index) = current_non_tool_block_index.take() {
                                    let event = json!({ "type": "content_block_stop", "index": index });
                                    yield Ok(Bytes::from(format!("event: content_block_stop\ndata: {}\n\n", serde_json::to_string(&event).unwrap_or_default())));
                                }

                                let index = next_content_index;
                                next_content_index += 1;
                                let event = json!({
                                    "type": "content_block_start",
                                    "index": index,
                                    "content_block": {
                                        "type": "text",
                                        "text": "",
                                    }
                                });
                                yield Ok(Bytes::from(format!("event: content_block_start\ndata: {}\n\n", serde_json::to_string(&event).unwrap_or_default())));
                                current_non_tool_block_type = Some("text");
                                current_non_tool_block_index = Some(index);
                            }

                            if let Some(index) = current_non_tool_block_index {
                                let event = json!({
                                    "type": "content_block_delta",
                                    "index": index,
                                    "delta": {
                                        "type": "text_delta",
                                        "text": content,
                                    }
                                });
                                yield Ok(Bytes::from(format!("event: content_block_delta\ndata: {}\n\n", serde_json::to_string(&event).unwrap_or_default())));
                            }
                        }
                    }

                    if let Some(tool_calls) = &choice.delta.tool_calls {
                        if let Some(index) = current_non_tool_block_index.take() {
                            let event = json!({ "type": "content_block_stop", "index": index });
                            yield Ok(Bytes::from(format!("event: content_block_stop\ndata: {}\n\n", serde_json::to_string(&event).unwrap_or_default())));
                            current_non_tool_block_type = None;
                        }

                        for tool_call in tool_calls {
                            let state = tool_blocks_by_index
                                .entry(tool_call.index)
                                .or_insert_with(|| build_tool_state(tool_call, next_content_index));

                            if state.anthropic_index == next_content_index {
                                next_content_index += 1;
                            }

                            update_tool_state(state, tool_call);

                            if !state.started {
                                let event = json!({
                                    "type": "content_block_start",
                                    "index": state.anthropic_index,
                                    "content_block": {
                                        "type": "tool_use",
                                        "id": state.id,
                                        "name": state.name,
                                        "input": {},
                                    }
                                });
                                yield Ok(Bytes::from(format!("event: content_block_start\ndata: {}\n\n", serde_json::to_string(&event).unwrap_or_default())));
                                state.started = true;
                                open_tool_block_indices.insert(state.anthropic_index);
                            }

                            if let Some(arguments) = tool_call
                                .function
                                .as_ref()
                                .and_then(|function| function.arguments.as_ref())
                            {
                                if !arguments.is_empty() {
                                    state.pending_args.push_str(arguments);
                                    let event = json!({
                                        "type": "content_block_delta",
                                        "index": state.anthropic_index,
                                        "delta": {
                                            "type": "input_json_delta",
                                            "partial_json": arguments,
                                        }
                                    });
                                    yield Ok(Bytes::from(format!("event: content_block_delta\ndata: {}\n\n", serde_json::to_string(&event).unwrap_or_default())));
                                }
                            }
                        }
                    }

                    if let Some(finish_reason) = choice.finish_reason.as_deref() {
                        if let Some(index) = current_non_tool_block_index.take() {
                            let event = json!({ "type": "content_block_stop", "index": index });
                            yield Ok(Bytes::from(format!("event: content_block_stop\ndata: {}\n\n", serde_json::to_string(&event).unwrap_or_default())));
                        }
                        current_non_tool_block_type = None;

                        for open_index in open_tool_block_indices.drain() {
                            let event = json!({ "type": "content_block_stop", "index": open_index });
                            yield Ok(Bytes::from(format!("event: content_block_stop\ndata: {}\n\n", serde_json::to_string(&event).unwrap_or_default())));
                        }

                        let stop_reason = match finish_reason {
                            "stop" => Some("end_turn"),
                            "length" => Some("max_tokens"),
                            "tool_calls" | "function_call" => Some("tool_use"),
                            _ => Some("end_turn"),
                        };

                        let event = json!({
                            "type": "message_delta",
                            "delta": {
                                "stop_reason": stop_reason,
                                "stop_sequence": null,
                            },
                            "usage": pending_usage.clone(),
                        });
                        yield Ok(Bytes::from(format!("event: message_delta\ndata: {}\n\n", serde_json::to_string(&event).unwrap_or_default())));

                        if !sent_message_stop {
                            yield Ok(Bytes::from("event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n"));
                            sent_message_stop = true;
                        }
                    }
                }
            }
        }
    }
}

fn build_tool_state(tool_call: &DeltaToolCall, next_content_index: u32) -> ToolBlockState {
    ToolBlockState {
        anthropic_index: next_content_index,
        id: tool_call.id.clone().unwrap_or_else(|| format!("call_{}", tool_call.index)),
        name: tool_call
            .function
            .as_ref()
            .and_then(|function| function.name.clone())
            .unwrap_or_default(),
        started: false,
        pending_args: String::new(),
    }
}

fn update_tool_state(state: &mut ToolBlockState, tool_call: &DeltaToolCall) {
    if let Some(id) = &tool_call.id {
        state.id = id.clone();
    }
    if let Some(name) = tool_call
        .function
        .as_ref()
        .and_then(|function| function.name.as_ref())
    {
        state.name = name.clone();
    }
}
