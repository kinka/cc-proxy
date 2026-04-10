use anyhow::{Context, Result};
use serde::Deserialize;
use std::{collections::HashMap, fs, path::Path};

#[derive(Debug, Clone, Deserialize)]
pub struct ProxyConfig {
    pub listen: ListenConfig,
    pub upstream: UpstreamConfig,
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
    pub model_map: HashMap<String, String>,
    #[serde(default)]
    pub extra_headers: HashMap<String, String>,
}

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
