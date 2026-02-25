//! Mistral AI LLM Provider
//!
//! Supports Mistral AI models (Mistral-7B, Mistral-Medium, Mistral-Large).
//! Uses the Mistral AI API.

use super::{LLMClient, LLMConfig, Message};
use crate::core::error::ErenFlowError;
use serde_json::json;

/// Mistral AI LLM client
#[derive(Clone)]
pub struct MistralClient {
    config: LLMConfig,
    client: reqwest::Client,
}

impl MistralClient {
    /// Create a new Mistral client
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
impl LLMClient for MistralClient {
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
            .map_err(|e| ErenFlowError::LLMError(format!("Mistral API request failed: {}", e)))?;

        if !response.status().is_success() {
            let error_text = response.text().await.unwrap_or_default();
            return Err(ErenFlowError::LLMError(format!(
                "Mistral API error: {}",
                error_text
            )));
        }

        let data: serde_json::Value = response.json().await.map_err(|e| {
            ErenFlowError::LLMError(format!("Failed to parse Mistral response: {}", e))
        })?;

        let content = data
            .get("choices")
            .and_then(|c| c.get(0))
            .and_then(|c| c.get("message"))
            .and_then(|m| m.get("content"))
            .and_then(|c| c.as_str())
            .ok_or_else(|| {
                ErenFlowError::LLMError("Invalid Mistral response format".to_string())
            })?;

        Ok(Message::assistant(content))
    }

    async fn chat_stream(
        &self,
        _messages: Vec<Message>,
    ) -> crate::core::error::Result<tokio::sync::mpsc::Receiver<String>> {
        // TODO: Implement Mistral streaming
        Err(ErenFlowError::LLMError(
            "Mistral streaming not yet implemented".to_string(),
        ))
    }
}
