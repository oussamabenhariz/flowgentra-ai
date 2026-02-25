//! Groq Cloud LLM Provider
//!
//! Supports Groq Cloud inference API with high-speed LLM access.
//! Supports models like LLaMA, Mixtral, and others.

use super::{LLMClient, LLMConfig, Message};
use crate::core::error::ErenFlowError;
use serde_json::json;

/// Groq Cloud LLM client
#[derive(Clone)]
pub struct GroqClient {
    config: LLMConfig,
    client: reqwest::Client,
}

impl GroqClient {
    /// Create a new Groq client
    pub fn new(config: LLMConfig) -> Self {
        Self {
            config,
            client: reqwest::Client::new(),
        }
    }

    /// Get the API endpoint URL
    fn get_endpoint(&self) -> String {
        format!("{}/chat/completions", self.config.provider.base_url())
    }
}

#[async_trait::async_trait]
impl LLMClient for GroqClient {
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
            "temperature": self.config.temperature.unwrap_or(0.7),
            "max_tokens": self.config.max_tokens.unwrap_or(2048),
            "top_p": self.config.top_p.unwrap_or(1.0),
        });

        let response = self
            .client
            .post(self.get_endpoint())
            .header("Authorization", format!("Bearer {}", self.config.api_key))
            .header("Content-Type", "application/json")
            .json(&payload)
            .send()
            .await
            .map_err(|e| ErenFlowError::LLMError(format!("Groq API request failed: {}", e)))?;

        if !response.status().is_success() {
            let error_text = response.text().await.unwrap_or_default();
            return Err(ErenFlowError::LLMError(format!(
                "Groq API error: {}",
                error_text
            )));
        }

        let data: serde_json::Value = response.json().await.map_err(|e| {
            ErenFlowError::LLMError(format!("Failed to parse Groq response: {}", e))
        })?;

        let content = data
            .get("choices")
            .and_then(|c| c.get(0))
            .and_then(|c| c.get("message"))
            .and_then(|m| m.get("content"))
            .and_then(|c| c.as_str())
            .ok_or_else(|| ErenFlowError::LLMError("Invalid Groq response format".to_string()))?;

        Ok(Message::assistant(content))
    }

    async fn chat_stream(
        &self,
        _messages: Vec<Message>,
    ) -> crate::core::error::Result<tokio::sync::mpsc::Receiver<String>> {
        // TODO: Implement Groq streaming
        Err(ErenFlowError::LLMError(
            "Groq streaming not yet implemented".to_string(),
        ))
    }
}
