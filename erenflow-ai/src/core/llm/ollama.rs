//! Ollama LLM Provider
//!
//! Supports running local LLMs via Ollama.
//! Ollama runs locally (typically on http://localhost:11434).
//!
//! To use:
//! 1. Install Ollama from https://ollama.ai
//! 2. Pull a model: `ollama pull mistral`
//! 3. Run: `ollama serve`
//! 4. Set provider to "ollama" in config

use super::{LLMClient, LLMConfig, Message};
use crate::core::error::ErenFlowError;
use serde_json::json;

/// Ollama LLM client for local LLM execution
#[derive(Clone)]
pub struct OllamaClient {
    config: LLMConfig,
    client: reqwest::Client,
}

impl OllamaClient {
    /// Create a new Ollama client
    pub fn new(config: LLMConfig) -> Self {
        Self {
            config,
            client: reqwest::Client::new(),
        }
    }

    /// Get the API endpoint URL
    fn get_endpoint(&self) -> String {
        format!("{}/chat", self.config.provider.base_url())
    }
}

#[async_trait::async_trait]
impl LLMClient for OllamaClient {
    async fn chat(&self, messages: Vec<Message>) -> crate::core::error::Result<Message> {
        let payload = json!({
            "model": self.config.model,
            "messages": messages.iter().map(|m| json!({
                "role": match m.role {
                    super::MessageRole::System => "system",
                    super::MessageRole::User => "user",
                    super::MessageRole::Assistant => "assistant",
                    super::MessageRole::Tool => "tool",
                },
                "content": m.content,
            })).collect::<Vec<_>>(),
            "stream": false,
            "options": {
                "temperature": self.config.temperature.unwrap_or(0.7),
                "num_predict": self.config.max_tokens.unwrap_or(2048),
                "top_p": self.config.top_p.unwrap_or(1.0),
            }
        });

        let response = self
            .client
            .post(self.get_endpoint())
            .header("Content-Type", "application/json")
            .json(&payload)
            .send()
            .await
            .map_err(|e| {
                ErenFlowError::LLMError(format!(
                    "Ollama request failed (is Ollama running on {}?): {}",
                    self.config.provider.base_url(),
                    e
                ))
            })?;

        if !response.status().is_success() {
            let error_text = response.text().await.unwrap_or_default();
            return Err(ErenFlowError::LLMError(format!(
                "Ollama API error: {}",
                error_text
            )));
        }

        let data: serde_json::Value = response.json().await.map_err(|e| {
            ErenFlowError::LLMError(format!("Failed to parse Ollama response: {}", e))
        })?;

        let content = data
            .get("message")
            .and_then(|m| m.get("content"))
            .and_then(|c| c.as_str())
            .ok_or_else(|| ErenFlowError::LLMError("Invalid Ollama response format".to_string()))?;

        Ok(Message::assistant(content))
    }

    async fn chat_stream(
        &self,
        _messages: Vec<Message>,
    ) -> crate::core::error::Result<tokio::sync::mpsc::Receiver<String>> {
        // TODO: Implement Ollama streaming
        Err(ErenFlowError::LLMError(
            "Ollama streaming not yet implemented".to_string(),
        ))
    }
}
