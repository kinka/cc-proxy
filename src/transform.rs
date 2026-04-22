use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

pub fn is_openai_o_series(model: &str) -> bool {
    model.len() > 1
        && model.starts_with('o')
        && model.as_bytes().get(1).is_some_and(|byte| byte.is_ascii_digit())
}

pub fn supports_reasoning_effort(model: &str) -> bool {
    is_openai_o_series(model)
        || model
            .to_lowercase()
            .strip_prefix("gpt-")
            .and_then(|rest| rest.chars().next())
            .is_some_and(|c| c.is_ascii_digit() && c >= '5')
}

pub fn resolve_reasoning_effort(body: &Value) -> Option<&'static str> {
    if let Some(effort) = body
        .pointer("/output_config/effort")
        .and_then(|value| value.as_str())
    {
        return match effort {
            "low" => Some("low"),
            "medium" => Some("medium"),
            "high" => Some("high"),
            "max" => Some("xhigh"),
            _ => None,
        };
    }

    let thinking = body.get("thinking")?;
    match thinking.get("type").and_then(|value| value.as_str()) {
        Some("adaptive") => Some("high"),
        Some("enabled") => {
            let budget = thinking
                .get("budget_tokens")
                .and_then(|value| value.as_u64());
            match budget {
                Some(tokens) if tokens < 4_000 => Some("low"),
                Some(tokens) if tokens < 16_000 => Some("medium"),
                Some(_) => Some("high"),
                None => Some("high"),
            }
        }
        _ => None,
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum ToolResultPart {
    Text(String),
    Image { data_url: String },
}

pub(crate) fn image_source_to_data_url(source: &Value) -> Option<String> {
    let media_type = source
        .get("media_type")
        .and_then(|value| value.as_str())
        .unwrap_or("image/png");
    let data = source
        .get("data")
        .and_then(|value| value.as_str())
        .unwrap_or("");

    if data.is_empty() {
        None
    } else {
        Some(format!("data:{media_type};base64,{data}"))
    }
}

pub(crate) fn normalize_tool_result_content(content: Option<&Value>) -> anyhow::Result<Vec<ToolResultPart>> {
    match content {
        None => Ok(Vec::new()),
        Some(Value::String(text)) => Ok(vec![ToolResultPart::Text(text.clone())]),
        Some(Value::Array(blocks)) => {
            let mut parts = Vec::new();

            for block in blocks {
                match block.get("type").and_then(|value| value.as_str()).unwrap_or("") {
                    "text" => {
                        if let Some(text) = block.get("text").and_then(|value| value.as_str()) {
                            parts.push(ToolResultPart::Text(text.to_string()));
                        }
                    }
                    "image" => {
                        if let Some(source) = block.get("source") {
                            if let Some(data_url) = image_source_to_data_url(source) {
                                parts.push(ToolResultPart::Image { data_url });
                            } else {
                                parts.push(ToolResultPart::Text(serde_json::to_string(block)?));
                            }
                        } else {
                            parts.push(ToolResultPart::Text(serde_json::to_string(block)?));
                        }
                    }
                    _ => parts.push(ToolResultPart::Text(serde_json::to_string(block)?)),
                }
            }

            Ok(parts)
        }
        Some(other) => Ok(vec![ToolResultPart::Text(serde_json::to_string(other)?)]),
    }
}

pub(crate) fn tool_result_parts_contain_image(parts: &[ToolResultPart]) -> bool {
    parts.iter()
        .any(|part| matches!(part, ToolResultPart::Image { .. }))
}

pub(crate) fn tool_result_parts_to_text(parts: &[ToolResultPart]) -> Option<String> {
    if tool_result_parts_contain_image(parts) {
        return None;
    }

    Some(
        parts.iter()
            .map(|part| match part {
                ToolResultPart::Text(text) => text.as_str(),
                ToolResultPart::Image { .. } => unreachable!(),
            })
            .collect::<Vec<_>>()
            .join(""),
    )
}

pub fn compute_anthropic_input_tokens(prompt_tokens: u32, cache_read_tokens: Option<u32>, cache_creation_tokens: Option<u32>) -> u32 {
    let cached = cache_read_tokens.unwrap_or(0);
    let created = cache_creation_tokens.unwrap_or(0);
    prompt_tokens.saturating_sub(cached.saturating_add(created))
}

pub(crate) fn tool_result_parts_to_openai_chat_content(parts: &[ToolResultPart]) -> Vec<Value> {
    parts.iter()
        .map(|part| match part {
            ToolResultPart::Text(text) => json!({ "type": "text", "text": text }),
            ToolResultPart::Image { data_url } => json!({
                "type": "image_url",
                "image_url": { "url": data_url },
            }),
        })
        .collect()
}

pub(crate) fn tool_result_parts_to_responses_content(parts: &[ToolResultPart]) -> Vec<Value> {
    parts.iter()
        .map(|part| match part {
            ToolResultPart::Text(text) => json!({ "type": "input_text", "text": text }),
            ToolResultPart::Image { data_url } => json!({
                "type": "input_image",
                "image_url": data_url,
            }),
        })
        .collect()
}

pub fn anthropic_to_openai(body: Value, cache_key: Option<&str>) -> anyhow::Result<Value> {
    let mut result = json!({});

    if let Some(model) = body.get("model").and_then(|value| value.as_str()) {
        result["model"] = json!(model);
    }

    let mut messages = Vec::new();

    if let Some(system) = body.get("system") {
        if let Some(text) = system.as_str() {
            messages.push(json!({ "role": "system", "content": text }));
        } else if let Some(items) = system.as_array() {
            for item in items {
                if let Some(text) = item.get("text").and_then(|value| value.as_str()) {
                    let mut system_message = json!({ "role": "system", "content": text });
                    if let Some(cache_control) = item.get("cache_control") {
                        system_message["cache_control"] = cache_control.clone();
                    }
                    messages.push(system_message);
                }
            }
        }
    }

    if let Some(input_messages) = body.get("messages").and_then(|value| value.as_array()) {
        for message in input_messages {
            let role = message
                .get("role")
                .and_then(|value| value.as_str())
                .unwrap_or("user");
            let converted = convert_message_to_openai(role, message.get("content"))?;
            messages.extend(converted);
        }
    }

    result["messages"] = json!(messages);

    let model = body.get("model").and_then(|value| value.as_str()).unwrap_or("");
    if let Some(max_tokens) = body.get("max_tokens") {
        if is_openai_o_series(model) {
            result["max_completion_tokens"] = max_tokens.clone();
        } else {
            result["max_tokens"] = max_tokens.clone();
        }
    }
    if let Some(temperature) = body.get("temperature") {
        result["temperature"] = temperature.clone();
    }
    if let Some(top_p) = body.get("top_p") {
        result["top_p"] = top_p.clone();
    }
    if let Some(stop) = body.get("stop_sequences") {
        result["stop"] = stop.clone();
    }
    if let Some(stream) = body.get("stream") {
        result["stream"] = stream.clone();
    }

    // Always try to pass reasoning_effort if thinking is requested
    if let Some(effort) = resolve_reasoning_effort(&body) {
        result["reasoning_effort"] = json!(effort);
    }

    if let Some(tools) = body.get("tools").and_then(|value| value.as_array()) {
        let mapped_tools: Vec<Value> = tools
            .iter()
            .filter(|tool| tool.get("type").and_then(|value| value.as_str()) != Some("BatchTool"))
            .map(|tool| {
                let mut mapped = json!({
                    "type": "function",
                    "function": {
                        "name": tool.get("name").and_then(|value| value.as_str()).unwrap_or(""),
                        "description": tool.get("description"),
                        "parameters": clean_schema(tool.get("input_schema").cloned().unwrap_or(json!({}))),
                    }
                });
                if let Some(cache_control) = tool.get("cache_control") {
                    mapped["cache_control"] = cache_control.clone();
                }
                mapped
            })
            .collect();

        if !mapped_tools.is_empty() {
            result["tools"] = json!(mapped_tools);
        }
    }

    if let Some(tool_choice) = body.get("tool_choice") {
        result["tool_choice"] = tool_choice.clone();
    }

    if let Some(cache_key) = cache_key {
        result["prompt_cache_key"] = json!(cache_key);
    }

    Ok(result)
}

pub fn openai_to_anthropic(body: Value) -> anyhow::Result<Value> {
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

    let mut content = Vec::new();
    let mut has_tool_use = false;

    // Handle reasoning_content from OpenAI response
    if let Some(reasoning) = message.get("reasoning_content").and_then(|v| v.as_str()) {
        if !reasoning.is_empty() {
            content.push(json!({
                "type": "thinking",
                "thinking": reasoning
            }));
        }
    }

    if let Some(message_content) = message.get("content") {
        if let Some(text) = message_content.as_str() {
            if !text.is_empty() {
                content.push(json!({ "type": "text", "text": text }));
            }
        } else if let Some(parts) = message_content.as_array() {
            for part in parts {
                match part.get("type").and_then(|value| value.as_str()).unwrap_or("") {
                    "text" | "output_text" => {
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

    if let Some(refusal) = message.get("refusal").and_then(|value| value.as_str()) {
        if !refusal.is_empty() {
            content.push(json!({ "type": "text", "text": refusal }));
        }
    }

    if let Some(tool_calls) = message.get("tool_calls").and_then(|value| value.as_array()) {
        if !tool_calls.is_empty() {
            has_tool_use = true;
        }
        for tool_call in tool_calls {
            let empty = json!({});
            let function = tool_call.get("function").unwrap_or(&empty);
            let arguments = function
                .get("arguments")
                .and_then(|value| value.as_str())
                .unwrap_or("{}");
            let parsed_arguments =
                serde_json::from_str::<Value>(arguments).unwrap_or_else(|_| json!({}));
            content.push(json!({
                "type": "tool_use",
                "id": tool_call.get("id").and_then(|value| value.as_str()).unwrap_or(""),
                "name": function.get("name").and_then(|value| value.as_str()).unwrap_or(""),
                "input": parsed_arguments,
            }));
        }
    }

    let stop_reason = choice
        .get("finish_reason")
        .and_then(|value| value.as_str())
        .map(|reason| match reason {
            "stop" => "end_turn",
            "length" => "max_tokens",
            "tool_calls" | "function_call" => "tool_use",
            _ => "end_turn",
        })
        .or(if has_tool_use {
            Some("tool_use")
        } else {
            None
        });

    let usage = body.get("usage").cloned().unwrap_or_else(|| json!({}));
    let prompt_tokens = usage
        .get("prompt_tokens")
        .and_then(|value| value.as_u64())
        .unwrap_or(0);
    let cache_read = usage
        .pointer("/prompt_tokens_details/cached_tokens")
        .and_then(|value| value.as_u64())
        .or_else(|| usage.get("cache_read_input_tokens").and_then(|value| value.as_u64()));
    let cache_creation = usage
        .get("cache_creation_input_tokens")
        .and_then(|value| value.as_u64());

    let mut usage_json = json!({
        "input_tokens": compute_anthropic_input_tokens(
            prompt_tokens.min(u64::from(u32::MAX)) as u32,
            cache_read.map(|value| value.min(u64::from(u32::MAX)) as u32),
            cache_creation.map(|value| value.min(u64::from(u32::MAX)) as u32),
        ),
        "output_tokens": usage.get("completion_tokens").and_then(|value| value.as_u64()).unwrap_or(0),
    });

    if let Some(cached) = cache_read {
        usage_json["cache_read_input_tokens"] = json!(cached);
    }
    if let Some(created) = cache_creation {
        usage_json["cache_creation_input_tokens"] = json!(created);
    }

    Ok(json!({
        "id": body.get("id").and_then(|value| value.as_str()).unwrap_or(""),
        "type": "message",
        "role": "assistant",
        "content": content,
        "model": body.get("model").and_then(|value| value.as_str()).unwrap_or(""),
        "stop_reason": stop_reason,
        "stop_sequence": null,
        "usage": usage_json,
    }))
}

fn convert_message_to_openai(role: &str, content: Option<&Value>) -> anyhow::Result<Vec<Value>> {
    let mut result = Vec::new();
    let Some(content) = content else {
        result.push(json!({ "role": role, "content": null }));
        return Ok(result);
    };

    if let Some(text) = content.as_str() {
        result.push(json!({ "role": role, "content": text }));
        return Ok(result);
    }

    if let Some(blocks) = content.as_array() {
        let mut content_parts = Vec::new();
        let mut tool_calls = Vec::new();

        for block in blocks {
            match block.get("type").and_then(|value| value.as_str()).unwrap_or("") {
                "text" => {
                    if let Some(text) = block.get("text").and_then(|value| value.as_str()) {
                        let mut part = json!({ "type": "text", "text": text });
                        if let Some(cache_control) = block.get("cache_control") {
                            part["cache_control"] = cache_control.clone();
                        }
                        content_parts.push(part);
                    }
                }
                "image" => {
                    if let Some(source) = block.get("source") {
                        if let Some(data_url) = image_source_to_data_url(source) {
                            content_parts.push(json!({
                                "type": "image_url",
                                "image_url": { "url": data_url },
                            }));
                        }
                    }
                }
                "tool_use" => {
                    let arguments = serde_json::to_string(
                        &block.get("input").cloned().unwrap_or_else(|| json!({})),
                    )?;
                    tool_calls.push(json!({
                        "id": block.get("id").and_then(|value| value.as_str()).unwrap_or(""),
                        "type": "function",
                        "function": {
                            "name": block.get("name").and_then(|value| value.as_str()).unwrap_or(""),
                            "arguments": arguments,
                        }
                    }));
                }
                "tool_result" => {
                    let parts = normalize_tool_result_content(block.get("content"))?;
                    let content = if let Some(text) = tool_result_parts_to_text(&parts) {
                        json!(text)
                    } else {
                        json!(tool_result_parts_to_openai_chat_content(&parts))
                    };
                    result.push(json!({
                        "role": "tool",
                        "tool_call_id": block.get("tool_use_id").and_then(|value| value.as_str()).unwrap_or(""),
                        "content": content,
                    }));
                }
                "thinking" => {}
                _ => {}
            }
        }

        if !content_parts.is_empty() || !tool_calls.is_empty() {
            let mut message = json!({ "role": role });
            if content_parts.is_empty() {
                message["content"] = Value::Null;
            } else if content_parts.len() == 1 && content_parts[0].get("cache_control").is_none() {
                if let Some(text) = content_parts[0].get("text") {
                    message["content"] = text.clone();
                } else {
                    message["content"] = json!(content_parts);
                }
            } else {
                message["content"] = json!(content_parts);
            }

            if !tool_calls.is_empty() {
                message["tool_calls"] = json!(tool_calls);
            }

            result.push(message);
        }

        return Ok(result);
    }

    result.push(json!({ "role": role, "content": content }));
    Ok(result)
}

pub fn clean_schema(mut schema: Value) -> Value {
    if let Some(object) = schema.as_object_mut() {
        if object.get("format").and_then(|value| value.as_str()) == Some("uri") {
            object.remove("format");
        }

        if let Some(properties) = object.get_mut("properties").and_then(|value| value.as_object_mut())
        {
            for value in properties.values_mut() {
                *value = clean_schema(value.clone());
            }
        }

        if let Some(items) = object.get_mut("items") {
            *items = clean_schema(items.clone());
        }
    }

    schema
}

#[derive(Debug, Deserialize)]
pub struct OpenAIStreamChunk {
    pub id: String,
    pub model: String,
    pub choices: Vec<StreamChoice>,
    #[serde(default)]
    pub usage: Option<Usage>,
}

#[derive(Debug, Deserialize)]
pub struct StreamChoice {
    pub delta: Delta,
    #[serde(default)]
    pub finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct Delta {
    #[serde(default)]
    pub content: Option<String>,
    #[serde(default)]
    pub reasoning: Option<String>,
    #[serde(default)]
    pub reasoning_content: Option<String>,
    #[serde(default)]
    pub tool_calls: Option<Vec<DeltaToolCall>>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct DeltaToolCall {
    pub index: usize,
    #[serde(default)]
    pub id: Option<String>,
    #[serde(rename = "type", default)]
    pub call_type: Option<String>,
    #[serde(default)]
    pub function: Option<DeltaFunction>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct DeltaFunction {
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub arguments: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct Usage {
    #[serde(default)]
    pub prompt_tokens: u32,
    #[serde(default)]
    pub completion_tokens: u32,
    #[serde(default)]
    pub prompt_tokens_details: Option<PromptTokensDetails>,
    #[serde(default)]
    pub cache_creation_input_tokens: Option<u32>,
}

#[derive(Debug, Deserialize)]
pub struct PromptTokensDetails {
    #[serde(default)]
    pub cached_tokens: u32,
}

pub fn extract_cache_read_tokens(usage: &Usage) -> Option<u32> {
    usage.prompt_tokens_details.as_ref().map(|details| details.cached_tokens)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn standard_user_image_maps_to_image_url() {
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

        let converted = anthropic_to_openai(body, None).unwrap();
        let messages = converted.get("messages").and_then(Value::as_array).unwrap();
        assert_eq!(messages[0]["content"][0]["type"], "image_url");
        assert_eq!(messages[0]["content"][0]["image_url"]["url"], "data:image/png;base64,YWJj");
    }

    #[test]
    fn tool_result_with_text_and_image_stays_structured_for_chat() {
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

        let converted = anthropic_to_openai(body, None).unwrap();
        let messages = converted.get("messages").and_then(Value::as_array).unwrap();
        let tool_message = messages.iter().find(|message| message["role"] == "tool").unwrap();
        let content = tool_message.get("content").and_then(Value::as_array).unwrap();
        assert_eq!(content[0]["type"], "text");
        assert_eq!(content[0]["text"], "截图如下");
        assert_eq!(content[1]["type"], "image_url");
        assert_eq!(content[1]["image_url"]["url"], "data:image/png;base64,YWJj");
    }

    #[test]
    fn text_only_tool_result_stays_plain_text_for_chat() {
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

        let converted = anthropic_to_openai(body, None).unwrap();
        let messages = converted.get("messages").and_then(Value::as_array).unwrap();
        assert_eq!(messages[0]["role"], "tool");
        assert_eq!(messages[0]["content"], "plain text");
    }

    #[test]
    fn openai_chat_usage_excludes_cache_from_input_tokens() {
        let body = json!({
            "id": "chatcmpl_test",
            "model": "test-model",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "ok"
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 120,
                "completion_tokens": 30,
                "prompt_tokens_details": {
                    "cached_tokens": 70
                },
                "cache_creation_input_tokens": 10
            }
        });

        let converted = openai_to_anthropic(body).unwrap();
        assert_eq!(converted["usage"]["input_tokens"], 40);
        assert_eq!(converted["usage"]["cache_read_input_tokens"], 70);
        assert_eq!(converted["usage"]["cache_creation_input_tokens"], 10);
        assert_eq!(converted["usage"]["output_tokens"], 30);
    }
}
