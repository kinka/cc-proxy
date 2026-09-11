use anyhow::{Context, Result};
use serde::Deserialize;
use std::{collections::HashMap, fs, path::Path};

#[derive(Debug, Clone, Deserialize)]
pub struct ProxyConfig {
    pub listen: ListenConfig,
    pub upstream: UpstreamConfig,
    #[serde(default)]
    pub providers: HashMap<String, UpstreamConfig>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ListenConfig {
    pub host: String,
    pub port: u16,
}

#[derive(Debug, Clone, Deserialize)]
pub struct UpstreamConfig {
    pub base_url: String,
    pub api_key: String,
    pub api_format: ApiFormat,
    #[serde(default = "default_timeout_secs")]
    pub timeout_secs: u64,
    #[serde(default)]
    pub prompt_cache_key: Option<String>,
    #[serde(default)]
    pub default_model: Option<String>,
    #[serde(default)]
    pub model_map: HashMap<String, String>,
    #[serde(default)]
    pub vision_model: Option<String>,
    #[serde(default)]
    pub strip_tool_result_images: bool,
    #[serde(default)]
    pub extra_headers: HashMap<String, String>,
    /// Extra top-level fields merged into the upstream request body after transform.
    ///
    /// This is the escape hatch for upstream-specific sampling knobs that have no
    /// Anthropic equivalent, so the client never has to know about them — e.g.
    /// `repetition_penalty` on a Qwen-backed gateway, which is the only guardrail
    /// against a model looping on the same reasoning verbatim.
    ///
    /// Values override whatever the transform produced. Structural fields
    /// (see `RESERVED_EXTRA_BODY_KEYS`) are refused, since overriding those turns a
    /// config typo into a malformed request that is hard to trace back here.
    #[serde(default)]
    pub extra_body: HashMap<String, serde_json::Value>,
}

/// Fields `extra_body` must never touch: they carry the translated request itself.
pub const RESERVED_EXTRA_BODY_KEYS: &[&str] =
    &["model", "messages", "input", "tools", "stream", "max_tokens", "max_completion_tokens"];

#[derive(Debug, Clone, Copy, Deserialize, PartialEq, Eq)]
pub enum ApiFormat {
    #[serde(rename = "openai_chat", alias = "open_ai_chat")]
    OpenAiChat,
    #[serde(rename = "openai_responses", alias = "open_ai_responses")]
    OpenAiResponses,
}

impl ProxyConfig {
    pub fn load(path: &Path) -> Result<Self> {
        let raw = fs::read_to_string(path)
            .with_context(|| format!("failed to read config: {}", path.display()))?;
        let config = serde_yaml::from_str::<Self>(&raw)
            .with_context(|| format!("failed to parse config: {}", path.display()))?;
        Ok(config)
    }
}

fn default_timeout_secs() -> u64 {
    600
}
