use serde_json::{json, Value};

pub fn anthropic_to_responses(body: Value, cache_key: Option<&str>) -> anyhow::Result<Value> {
    let mut result = json!({});

    if let Some(model) = body.get("model").and_then(|value| value.as_str()) {
        result["model"] = json!(model);
    }

    if let Some(system) = body.get("system") {
        let instructions = if let Some(text) = system.as_str() {
            text.to_string()
        } else if let Some(items) = system.as_array() {
            items
                .iter()
                .filter_map(|item| item.get("text").and_then(|value| value.as_str()))
                .collect::<Vec<_>>()
                .join("\n\n")
        } else {
            String::new()
        };
        if !instructions.is_empty() {
            result["instructions"] = json!(instructions);
        }
    }

    if let Some(messages) = body.get("messages").and_then(|value| value.as_array()) {
        result["input"] = json!(convert_messages_to_input(messages)?);
    }

    if let Some(max_tokens) = body.get("max_tokens") {
        result["max_output_tokens"] = max_tokens.clone();
    }
    if let Some(temperature) = body.get("temperature") {
        result["temperature"] = temperature.clone();
    }
    if let Some(top_p) = body.get("top_p") {
        result["top_p"] = top_p.clone();
    }
    if let Some(stream) = body.get("stream") {
        result["stream"] = stream.clone();
    }

    if let Some(model) = body.get("model").and_then(|value| value.as_str()) {
        if super::transform::supports_reasoning_effort(model) {
            if let Some(effort) = super::transform::resolve_reasoning_effort(&body) {
                result["reasoning"] = json!({ "effort": effort });
            }
        }
    }

    if let Some(tools) = body.get("tools").and_then(|value| value.as_array()) {
        let mapped_tools: Vec<Value> = tools
            .iter()
            .filter(|tool| tool.get("type").and_then(|value| value.as_str()) != Some("BatchTool"))
            .map(|tool| {
                json!({
                    "type": "function",
                    "name": tool.get("name").and_then(|value| value.as_str()).unwrap_or(""),
                    "description": tool.get("description"),
                    "parameters": super::transform::clean_schema(tool.get("input_schema").cloned().unwrap_or(json!({}))),
                })
            })
            .collect();

        if !mapped_tools.is_empty() {
            result["tools"] = json!(mapped_tools);
        }
    }

    if let Some(tool_choice) = body.get("tool_choice") {
        result["tool_choice"] = map_tool_choice_to_responses(tool_choice);
    }

    if let Some(cache_key) = cache_key {
        result["prompt_cache_key"] = json!(cache_key);
    }

    Ok(result)
}

pub fn responses_to_anthropic(body: Value) -> anyhow::Result<Value> {
    let output = body
        .get("output")
        .and_then(|value| value.as_array())
        .ok_or_else(|| anyhow::anyhow!("missing output"))?;

    let mut content = Vec::new();
    let mut has_tool_use = false;

    for item in output {
        match item.get("type").and_then(|value| value.as_str()).unwrap_or("") {
            "message" => {
                if let Some(parts) = item.get("content").and_then(|value| value.as_array()) {
                    for part in parts {
                        match part.get("type").and_then(|value| value.as_str()).unwrap_or("") {
                            "output_text" | "text" => {
                                if let Some(text) = part.get("text").and_then(|value| value.as_str()) {
                                    if !text.is_empty() {
                                        content.push(json!({ "type": "text", "text": text }));
                                    }
                                }
                            }
                            "refusal" => {
                                if let Some(refusal) = part.get("refusal").and_then(|value| value.as_str()) {
                                    if !refusal.is_empty() {
                                        content.push(json!({ "type": "text", "text": refusal }));
                                    }
                                }
                            }
                            _ => {}
                        }
                    }
                }
            }
            "output_text" | "text" => {
                if let Some(text) = item.get("text").and_then(|value| value.as_str()) {
                    if !text.is_empty() {
                        content.push(json!({ "type": "text", "text": text }));
                    }
                }
            }
            "function_call" => {
                let arguments = item
                    .get("arguments")
                    .and_then(|value| value.as_str())
                    .unwrap_or("{}");
                let parsed_arguments =
                    serde_json::from_str::<Value>(arguments).unwrap_or_else(|_| json!({}));
                content.push(json!({
                    "type": "tool_use",
                    "id": item.get("call_id").and_then(|value| value.as_str()).unwrap_or(""),
                    "name": item.get("name").and_then(|value| value.as_str()).unwrap_or(""),
                    "input": parsed_arguments,
                }));
                has_tool_use = true;
            }
            "reasoning" => {
                if let Some(summary) = item.get("summary").and_then(|value| value.as_array()) {
                    let text = summary
                        .iter()
                        .filter_map(|part| {
                            if part.get("type").and_then(|value| value.as_str()) == Some("summary_text") {
                                part.get("text").and_then(|value| value.as_str())
                            } else {
                                None
                            }
                        })
                        .collect::<Vec<_>>()
                        .join("");

                    if !text.is_empty() {
                        content.push(json!({
                            "type": "thinking",
                            "thinking": text,
                        }));
                    }
                }
            }
            _ => {}
        }
    }

    if content.is_empty() {
        if let Some(text) = body.get("output_text").and_then(|value| value.as_str()) {
            if !text.is_empty() {
                content.push(json!({ "type": "text", "text": text }));
            }
        }
    }

    let stop_reason = map_responses_stop_reason(
        body.get("status").and_then(|value| value.as_str()),
        has_tool_use,
        body.pointer("/incomplete_details/reason")
            .and_then(|value| value.as_str()),
    );

    Ok(json!({
        "id": body.get("id").and_then(|value| value.as_str()).unwrap_or(""),
        "type": "message",
        "role": "assistant",
        "content": content,
        "model": body.get("model").and_then(|value| value.as_str()).unwrap_or(""),
        "stop_reason": stop_reason,
        "stop_sequence": null,
        "usage": build_anthropic_usage_from_responses(body.get("usage")),
    }))
}

pub fn build_anthropic_usage_from_responses(usage: Option<&Value>) -> Value {
    let Some(usage) = usage else {
        return json!({
            "input_tokens": 0,
            "output_tokens": 0,
        });
    };

    let mut result = json!({
        "input_tokens": usage.get("input_tokens").and_then(|value| value.as_u64()).unwrap_or(0),
        "output_tokens": usage.get("output_tokens").and_then(|value| value.as_u64()).unwrap_or(0),
    });

    if let Some(cached) = usage
        .pointer("/input_tokens_details/cached_tokens")
        .and_then(|value| value.as_u64())
    {
        result["cache_read_input_tokens"] = json!(cached);
    }
    if let Some(cached) = usage
        .pointer("/prompt_tokens_details/cached_tokens")
        .and_then(|value| value.as_u64())
    {
        if result.get("cache_read_input_tokens").is_none() {
            result["cache_read_input_tokens"] = json!(cached);
        }
    }
    if let Some(cache_read) = usage.get("cache_read_input_tokens") {
        result["cache_read_input_tokens"] = cache_read.clone();
    }
    if let Some(cache_creation) = usage.get("cache_creation_input_tokens") {
        result["cache_creation_input_tokens"] = cache_creation.clone();
    }

    result
}

pub fn map_responses_stop_reason(
    status: Option<&str>,
    has_tool_use: bool,
    incomplete_reason: Option<&str>,
) -> Option<&'static str> {
    status.map(|status| match status {
        "completed" => {
            if has_tool_use {
                "tool_use"
            } else {
                "end_turn"
            }
        }
        "incomplete" => {
            if matches!(
                incomplete_reason,
                Some("max_output_tokens") | Some("max_tokens")
            ) || incomplete_reason.is_none()
            {
                "max_tokens"
            } else {
                "end_turn"
            }
        }
        _ => "end_turn",
    })
}

fn map_tool_choice_to_responses(tool_choice: &Value) -> Value {
    match tool_choice {
        Value::String(_) => tool_choice.clone(),
        Value::Object(object) => match object.get("type").and_then(|value| value.as_str()) {
            Some("any") => json!("required"),
            Some("auto") => json!("auto"),
            Some("none") => json!("none"),
            Some("tool") => json!({
                "type": "function",
                "name": object.get("name").and_then(|value| value.as_str()).unwrap_or(""),
            }),
            _ => tool_choice.clone(),
        },
        _ => tool_choice.clone(),
    }
}

fn convert_messages_to_input(messages: &[Value]) -> anyhow::Result<Vec<Value>> {
    let mut input = Vec::new();

    for message in messages {
        let role = message
            .get("role")
            .and_then(|value| value.as_str())
            .unwrap_or("user");
        match message.get("content") {
            Some(Value::String(text)) => {
                let item_type = if role == "assistant" {
                    "output_text"
                } else {
                    "input_text"
                };
                input.push(json!({
                    "role": role,
                    "content": [{ "type": item_type, "text": text }],
                }));
            }
            Some(Value::Array(blocks)) => {
                let mut message_content = Vec::new();

                for block in blocks {
                    match block.get("type").and_then(|value| value.as_str()).unwrap_or("") {
                        "text" => {
                            if let Some(text) = block.get("text").and_then(|value| value.as_str()) {
                                let item_type = if role == "assistant" {
                                    "output_text"
                                } else {
                                    "input_text"
                                };
                                message_content.push(json!({ "type": item_type, "text": text }));
                            }
                        }
                        "image" => {
                            if let Some(source) = block.get("source") {
                                if let Some(data_url) = super::transform::image_source_to_data_url(source) {
                                    message_content.push(json!({
                                        "type": "input_image",
                                        "image_url": data_url,
                                    }));
                                }
                            }
                        }
                        "tool_use" => {
                            if !message_content.is_empty() {
                                input.push(json!({
                                    "role": role,
                                    "content": message_content.clone(),
                                }));
                                message_content.clear();
                            }

                            let arguments = serde_json::to_string(
                                &block.get("input").cloned().unwrap_or_else(|| json!({})),
                            )?;
                            input.push(json!({
                                "type": "function_call",
                                "call_id": block.get("id").and_then(|value| value.as_str()).unwrap_or(""),
                                "name": block.get("name").and_then(|value| value.as_str()).unwrap_or(""),
                                "arguments": arguments,
                            }));
                        }
                        "tool_result" => {
                            if !message_content.is_empty() {
                                input.push(json!({
                                    "role": role,
                                    "content": message_content.clone(),
                                }));
                                message_content.clear();
                            }

                            let parts = super::transform::normalize_tool_result_content(block.get("content"))?;
                            let output = super::transform::tool_result_parts_to_text(&parts)
                                .unwrap_or_else(|| serde_json::to_string(&block.get("content").cloned().unwrap_or(Value::Null)).unwrap_or_default());
                            let call_id = block.get("tool_use_id").and_then(|value| value.as_str()).unwrap_or("");
                            input.push(json!({
                                "type": "function_call_output",
                                "call_id": call_id,
                                "output": output,
                            }));

                            if super::transform::tool_result_parts_contain_image(&parts) {
                                let mut multimodal_content = vec![json!({
                                    "type": "input_text",
                                    "text": format!("Tool result for call_id={call_id} follows."),
                                })];
                                multimodal_content.extend(super::transform::tool_result_parts_to_responses_content(&parts));
                                input.push(json!({
                                    "role": role,
                                    "content": multimodal_content,
                                }));
                            }
                        }
                        "thinking" => {}
                        _ => {}
                    }
                }

                if !message_content.is_empty() {
                    input.push(json!({
                        "role": role,
                        "content": message_content,
                    }));
                }
            }
            _ => input.push(json!({ "role": role })),
        }
    }

    Ok(input)
}

pub fn responses_request_to_openai_chat(body: Value) -> anyhow::Result<Value> {
    let mut result = json!({});

    if let Some(model) = body.get("model").and_then(|value| value.as_str()) {
        result["model"] = json!(model);
    }

    let mut messages = Vec::new();
    if let Some(instructions) = body.get("instructions").and_then(|value| value.as_str()) {
        if !instructions.is_empty() {
            messages.push(json!({ "role": "system", "content": instructions }));
        }
    }

    if let Some(input) = body.get("input") {
        messages.extend(responses_input_to_chat_messages(input)?);
    }
    result["messages"] = json!(messages);

    let model = body.get("model").and_then(|value| value.as_str()).unwrap_or("");
    if let Some(max_tokens) = body.get("max_output_tokens") {
        if super::transform::is_openai_o_series(model) {
            result["max_completion_tokens"] = max_tokens.clone();
        } else {
            result["max_tokens"] = max_tokens.clone();
        }
    }
    for field in ["temperature", "top_p", "stream", "stop", "user", "metadata", "parallel_tool_calls"] {
        if let Some(value) = body.get(field) {
            result[field] = value.clone();
        }
    }
    if let Some(format) = body.get("text").and_then(|value| value.get("format")) {
        result["response_format"] = format.clone();
    }
    if let Some(effort) = body.pointer("/reasoning/effort").and_then(|value| value.as_str()) {
        result["reasoning_effort"] = json!(effort);
    }

    if let Some(tools) = body.get("tools").and_then(|value| value.as_array()) {
        let mapped_tools: Vec<Value> = tools
            .iter()
            .filter_map(responses_tool_to_chat_tool)
            .collect();
        if !mapped_tools.is_empty() {
            result["tools"] = json!(mapped_tools);
        }
    }

    if let Some(tool_choice) = body.get("tool_choice") {
        result["tool_choice"] = responses_tool_choice_to_chat(tool_choice);
    }

    Ok(result)
}

pub fn openai_chat_to_responses(body: Value) -> anyhow::Result<Value> {
    let choices = body
        .get("choices")
        .and_then(|value| value.as_array())
        .ok_or_else(|| anyhow::anyhow!("missing choices"))?;
    let choice = choices
        .first()
        .ok_or_else(|| anyhow::anyhow!("empty choices array"))?;
    let message = choice
        .get("message")
        .ok_or_else(|| anyhow::anyhow!("missing message"))?;

    let mut output = Vec::new();
    let mut output_text = String::new();

    let mut message_content = Vec::new();
    if let Some(content) = message.get("content") {
        append_chat_content_as_response_output_text(content, &mut message_content, &mut output_text);
    }
    if let Some(refusal) = message.get("refusal").and_then(|value| value.as_str()) {
        if !refusal.is_empty() {
            message_content.push(json!({ "type": "refusal", "refusal": refusal }));
            output_text.push_str(refusal);
        }
    }
    if !message_content.is_empty() {
        output.push(json!({
            "type": "message",
            "id": format!("msg_{}", body.get("id").and_then(|value| value.as_str()).unwrap_or("")),
            "status": "completed",
            "role": "assistant",
            "content": message_content,
        }));
    }

    if let Some(tool_calls) = message.get("tool_calls").and_then(|value| value.as_array()) {
        for tool_call in tool_calls {
            let empty = json!({});
            let function = tool_call.get("function").unwrap_or(&empty);
            output.push(json!({
                "type": "function_call",
                "id": tool_call.get("id").and_then(|value| value.as_str()).unwrap_or(""),
                "call_id": tool_call.get("id").and_then(|value| value.as_str()).unwrap_or(""),
                "name": function.get("name").and_then(|value| value.as_str()).unwrap_or(""),
                "arguments": function.get("arguments").and_then(|value| value.as_str()).unwrap_or("{}"),
                "status": "completed",
            }));
        }
    }

    let finish_reason = choice.get("finish_reason").and_then(|value| value.as_str());
    let mut result = json!({
        "id": body.get("id").and_then(|value| value.as_str()).unwrap_or(""),
        "object": "response",
        "created_at": body.get("created").and_then(|value| value.as_i64()).unwrap_or(0),
        "status": if finish_reason == Some("length") { "incomplete" } else { "completed" },
        "model": body.get("model").and_then(|value| value.as_str()).unwrap_or(""),
        "output": output,
        "output_text": output_text,
        "usage": openai_chat_usage_to_responses_usage(body.get("usage")),
    });

    if finish_reason == Some("length") {
        result["incomplete_details"] = json!({ "reason": "max_output_tokens" });
    }

    Ok(result)
}

pub fn openai_chat_usage_to_responses_usage(usage: Option<&Value>) -> Value {
    let Some(usage) = usage else {
        return json!({
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
        });
    };

    let input_tokens = usage.get("prompt_tokens").and_then(|value| value.as_u64()).unwrap_or(0);
    let output_tokens = usage.get("completion_tokens").and_then(|value| value.as_u64()).unwrap_or(0);
    let mut result = json!({
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": usage.get("total_tokens").and_then(|value| value.as_u64()).unwrap_or(input_tokens + output_tokens),
    });

    if let Some(details) = usage.get("prompt_tokens_details") {
        result["input_tokens_details"] = details.clone();
    }
    if let Some(details) = usage.get("completion_tokens_details") {
        result["output_tokens_details"] = details.clone();
    }

    result
}

fn responses_input_to_chat_messages(input: &Value) -> anyhow::Result<Vec<Value>> {
    if let Some(text) = input.as_str() {
        return Ok(vec![json!({ "role": "user", "content": text })]);
    }

    let Some(items) = input.as_array() else {
        return Ok(vec![json!({ "role": "user", "content": input })]);
    };

    let mut messages = Vec::new();
    for item in items {
        let item_type = item.get("type").and_then(|value| value.as_str()).unwrap_or("");
        match item_type {
            "message" => messages.push(response_message_item_to_chat_message(item)),
            "function_call" => messages.push(response_function_call_to_chat_message(item)),
            "function_call_output" => messages.push(json!({
                "role": "tool",
                "tool_call_id": item.get("call_id").and_then(|value| value.as_str()).unwrap_or(""),
                "content": item.get("output").cloned().unwrap_or_else(|| json!("")),
            })),
            _ if item.get("role").is_some() => messages.push(response_message_item_to_chat_message(item)),
            _ => messages.push(json!({ "role": "user", "content": response_content_to_chat_content(Some(item)) })),
        }
    }

    Ok(messages)
}

fn response_message_item_to_chat_message(item: &Value) -> Value {
    let role = normalize_responses_role_for_chat(
        item.get("role")
            .and_then(|value| value.as_str())
            .unwrap_or("user"),
    );
    json!({
        "role": role,
        "content": response_content_to_chat_content(item.get("content")),
    })
}

fn normalize_responses_role_for_chat(role: &str) -> &str {
    match role {
        "developer" => "system",
        other => other,
    }
}

fn response_function_call_to_chat_message(item: &Value) -> Value {
    json!({
        "role": "assistant",
        "content": null,
        "tool_calls": [{
            "id": item.get("call_id").or_else(|| item.get("id")).and_then(|value| value.as_str()).unwrap_or(""),
            "type": "function",
            "function": {
                "name": item.get("name").and_then(|value| value.as_str()).unwrap_or(""),
                "arguments": item.get("arguments").and_then(|value| value.as_str()).unwrap_or("{}"),
            },
        }],
    })
}

fn response_content_to_chat_content(content: Option<&Value>) -> Value {
    let Some(content) = content else {
        return Value::Null;
    };
    if let Some(text) = content.as_str() {
        return json!(text);
    }
    if let Some(parts) = content.as_array() {
        let mut chat_parts = Vec::new();
        for part in parts {
            match part.get("type").and_then(|value| value.as_str()).unwrap_or("") {
                "input_text" | "output_text" | "text" => {
                    if let Some(text) = part.get("text").and_then(|value| value.as_str()) {
                        chat_parts.push(json!({ "type": "text", "text": text }));
                    }
                }
                "input_image" => {
                    if let Some(image_url) = part.get("image_url") {
                        chat_parts.push(json!({ "type": "image_url", "image_url": normalize_chat_image_url(image_url) }));
                    }
                }
                _ => {}
            }
        }
        if chat_parts.len() == 1 && chat_parts[0].get("type").and_then(|value| value.as_str()) == Some("text") {
            return chat_parts[0].get("text").cloned().unwrap_or_else(|| json!(""));
        }
        return json!(chat_parts);
    }
    content.clone()
}

fn normalize_chat_image_url(image_url: &Value) -> Value {
    if image_url.is_string() {
        json!({ "url": image_url })
    } else {
        image_url.clone()
    }
}

fn responses_tool_to_chat_tool(tool: &Value) -> Option<Value> {
    if tool.get("type").and_then(|value| value.as_str()) != Some("function") {
        return None;
    }
    if tool.get("function").is_some() {
        return Some(tool.clone());
    }
    Some(json!({
        "type": "function",
        "function": {
            "name": tool.get("name").and_then(|value| value.as_str()).unwrap_or(""),
            "description": tool.get("description"),
            "parameters": super::transform::clean_schema(tool.get("parameters").cloned().unwrap_or_else(|| json!({}))),
        },
    }))
}

fn responses_tool_choice_to_chat(tool_choice: &Value) -> Value {
    match tool_choice {
        Value::Object(object) => match object.get("type").and_then(|value| value.as_str()) {
            Some("function") => json!({
                "type": "function",
                "function": {
                    "name": object.get("name").and_then(|value| value.as_str()).unwrap_or(""),
                },
            }),
            _ => tool_choice.clone(),
        },
        _ => tool_choice.clone(),
    }
}

fn append_chat_content_as_response_output_text(content: &Value, message_content: &mut Vec<Value>, output_text: &mut String) {
    if let Some(text) = content.as_str() {
        if !text.is_empty() {
            message_content.push(json!({ "type": "output_text", "text": text, "annotations": [] }));
            output_text.push_str(text);
        }
    } else if let Some(parts) = content.as_array() {
        for part in parts {
            match part.get("type").and_then(|value| value.as_str()).unwrap_or("") {
                "text" | "output_text" => {
                    if let Some(text) = part.get("text").and_then(|value| value.as_str()) {
                        if !text.is_empty() {
                            message_content.push(json!({ "type": "output_text", "text": text, "annotations": [] }));
                            output_text.push_str(text);
                        }
                    }
                }
                "refusal" => {
                    if let Some(refusal) = part.get("refusal").and_then(|value| value.as_str()) {
                        if !refusal.is_empty() {
                            message_content.push(json!({ "type": "refusal", "refusal": refusal }));
                            output_text.push_str(refusal);
                        }
                    }
                }
                _ => {}
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn standard_user_image_maps_to_input_image() {
        let body = json!({
            "model": "test-model",
            "messages": [{
                "role": "user",
                "content": [{
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": "YWJj"
                    }
                }]
            }]
        });

        let converted = anthropic_to_responses(body, None).unwrap();
        let input = converted.get("input").and_then(Value::as_array).unwrap();
        assert_eq!(input[0]["content"][0]["type"], "input_image");
        assert_eq!(input[0]["content"][0]["image_url"], "data:image/png;base64,YWJj");
    }

    #[test]
    fn tool_result_with_text_and_image_adds_multimodal_followup() {
        let body = json!({
            "model": "test-model",
            "messages": [
                {
                    "role": "assistant",
                    "content": [{
                        "type": "tool_use",
                        "id": "toolu_1",
                        "name": "Read",
                        "input": {"file_path": "a.png"}
                    }]
                },
                {
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": "toolu_1",
                        "content": [
                            {"type": "text", "text": "截图如下"},
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": "YWJj"
                                }
                            }
                        ]
                    }]
                }
            ]
        });

        let converted = anthropic_to_responses(body, None).unwrap();
        let input = converted.get("input").and_then(Value::as_array).unwrap();
        assert_eq!(input[1]["type"], "function_call_output");
        assert_eq!(input[1]["call_id"], "toolu_1");
        assert!(input[1]["output"].as_str().unwrap().contains("image"));
        assert_eq!(input[2]["role"], "user");
        assert_eq!(input[2]["content"][0]["type"], "input_text");
        assert_eq!(input[2]["content"][1]["type"], "input_text");
        assert_eq!(input[2]["content"][1]["text"], "截图如下");
        assert_eq!(input[2]["content"][2]["type"], "input_image");
        assert_eq!(input[2]["content"][2]["image_url"], "data:image/png;base64,YWJj");
    }

    #[test]
    fn text_only_tool_result_stays_function_call_output_only() {
        let body = json!({
            "model": "test-model",
            "messages": [{
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": "toolu_1",
                    "content": "plain text"
                }]
            }]
        });

        let converted = anthropic_to_responses(body, None).unwrap();
        let input = converted.get("input").and_then(Value::as_array).unwrap();
        assert_eq!(input.len(), 1);
        assert_eq!(input[0]["type"], "function_call_output");
        assert_eq!(input[0]["output"], "plain text");
    }

    #[test]
    fn responses_request_maps_to_openai_chat() {
        let body = json!({
            "model": "gpt-4.1",
            "instructions": "You are helpful.",
            "input": [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "hello"},
                        {"type": "input_image", "image_url": "data:image/png;base64,YWJj"}
                    ]
                }
            ],
            "max_output_tokens": 123,
            "temperature": 0.2,
            "stream": true,
            "tools": [{
                "type": "function",
                "name": "Read",
                "description": "read file",
                "parameters": {
                    "type": "object",
                    "properties": {"file_path": {"type": "string", "format": "uri"}}
                }
            }],
            "tool_choice": {"type": "function", "name": "Read"}
        });

        let converted = responses_request_to_openai_chat(body).unwrap();
        let messages = converted.get("messages").and_then(Value::as_array).unwrap();
        assert_eq!(messages[0]["role"], "system");
        assert_eq!(messages[0]["content"], "You are helpful.");
        assert_eq!(messages[1]["role"], "user");
        assert_eq!(messages[1]["content"][0]["type"], "text");
        assert_eq!(messages[1]["content"][0]["text"], "hello");
        assert_eq!(messages[1]["content"][1]["type"], "image_url");
        assert_eq!(messages[1]["content"][1]["image_url"]["url"], "data:image/png;base64,YWJj");
        assert_eq!(converted["max_tokens"], 123);
        assert_eq!(converted["stream"], true);
        assert_eq!(converted["tools"][0]["function"]["name"], "Read");
        assert!(converted["tools"][0]["function"]["parameters"]["properties"]["file_path"].get("format").is_none());
        assert_eq!(converted["tool_choice"]["function"]["name"], "Read");
    }

    #[test]
    fn developer_role_maps_to_system_for_chat() {
        let body = json!({
            "model": "gpt-4.1",
            "input": [{
                "role": "developer",
                "content": [{"type": "input_text", "text": "be terse"}]
            }]
        });

        let converted = responses_request_to_openai_chat(body).unwrap();
        let messages = converted.get("messages").and_then(Value::as_array).unwrap();
        assert_eq!(messages[0]["role"], "system");
        assert_eq!(messages[0]["content"], "be terse");
    }

    #[test]
    fn openai_chat_response_maps_to_responses() {
        let body = json!({
            "id": "chatcmpl_test",
            "created": 123,
            "model": "gpt-4.1",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "hello",
                    "tool_calls": [{
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "Read",
                            "arguments": "{\"file_path\":\"a.txt\"}"
                        }
                    }]
                },
                "finish_reason": "tool_calls"
            }],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
                "prompt_tokens_details": {"cached_tokens": 3}
            }
        });

        let converted = openai_chat_to_responses(body).unwrap();
        assert_eq!(converted["id"], "chatcmpl_test");
        assert_eq!(converted["object"], "response");
        assert_eq!(converted["status"], "completed");
        assert_eq!(converted["output"][0]["type"], "message");
        assert_eq!(converted["output"][0]["content"][0]["type"], "output_text");
        assert_eq!(converted["output"][0]["content"][0]["text"], "hello");
        assert_eq!(converted["output"][1]["type"], "function_call");
        assert_eq!(converted["output"][1]["call_id"], "call_1");
        assert_eq!(converted["usage"]["input_tokens"], 10);
        assert_eq!(converted["usage"]["output_tokens"], 5);
        assert_eq!(converted["usage"]["input_tokens_details"]["cached_tokens"], 3);
    }

    #[test]
    fn openai_chat_length_maps_to_incomplete_response() {
        let body = json!({
            "id": "chatcmpl_test",
            "created": 123,
            "model": "gpt-4.1",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "truncated"
                },
                "finish_reason": "length"
            }],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15
            }
        });

        let converted = openai_chat_to_responses(body).unwrap();
        assert_eq!(converted["status"], "incomplete");
        assert_eq!(converted["incomplete_details"]["reason"], "max_output_tokens");
    }
}
