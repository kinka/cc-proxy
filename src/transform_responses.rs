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
}
