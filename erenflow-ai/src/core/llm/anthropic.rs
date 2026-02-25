//! Anthropic LLM Provider
//!
//! Supports Claude models (3 Opus, 3 Sonnet, 3 Haiku, and others).
//! Uses the Anthropic Messages API.

use super::{LLMClient, LLMConfig, Message};
use crate::core::error::ErenFlowError;
use serde_json::json;

/// Anthropic Claude LLM client
#[derive(Clone)]
pub struct AnthropicClient {
    config: LLMConfig,
    client: reqwest::Client,
}

impl AnthropicClient {
    /// Create a new Anthropic client
    pub fn new(config: LLMConfig) -> Self {
        Self {
            config,
            client: reqwest::Client::new(),
        }
    }

    /// Get the API endpoint URL
    fn get_endpoint(&self) -> String {
        format!("{}/messages", self.config.provider.base_url())
    }
}

#[async_trait::async_trait]
impl LLMClient for AnthropicClient {
    async fn chat(&self, messages: Vec<Message>) -> crate::core::error::Result<Message> {
        // Filter out system messages - Anthropic uses a separate system parameter
        let (system_prompt, messages) = {
            let mut sys = String::new();
            let mut msgs = Vec::new();
            for msg in messages {
                if msg.role == super::MessageRole::System {
                    sys = msg.content.clone();
                } else {
                    msgs.push(msg);
                }
            }
            (sys, msgs)
        };

        let mut payload = json!({
            "model": self.config.model,
            "messages": messages.iter().map(|m| json!({
                "role": match m.role {
                    super::MessageRole::User => "user",
                    super::MessageRole::Assistant => "assistant",
                    _ => "user", // System and Tool handled separately
                },
                "content": m.content,
            })).collect::<Vec<_>>(),
            "max_tokens": self.config.max_tokens.unwrap_or(2048),
        });

        // Add system prompt if present
        if !system_prompt.is_empty() {
            payload["system"] = json!(system_prompt);
        }

        // Note: Anthropic doesn't directly support temperature the same way
        // but we include it in extra_params if the user wants to use it
        if let Some(temp) = self.config.temperature {
            payload["temperature"] = json!(temp);
        }

        if let Some(top_p) = self.config.top_p {
            payload["top_p"] = json!(top_p);
        }

        let response = self
            .client
            .post(self.get_endpoint())
            .header("x-api-key", &self.config.api_key)
            .header("anthropic-version", "2023-06-01")
            .header("Content-Type", "application/json")
            .json(&payload)
            .send()
            .await
            .map_err(|e| ErenFlowError::LLMError(format!("Anthropic API request failed: {}", e)))?;

        if !response.status().is_success() {
            let error_text = response.text().await.unwrap_or_default();
            return Err(ErenFlowError::LLMError(format!(
                "Anthropic API error: {}",
                error_text
            )));
        }

        let data: serde_json::Value = response.json().await.map_err(|e| {
            ErenFlowError::LLMError(format!("Failed to parse Anthropic response: {}", e))
        })?;

        let content = data
            .get("content")
            .and_then(|c| c.get(0))
            .and_then(|c| c.get("text"))
            .and_then(|t| t.as_str())
            .ok_or_else(|| {
                ErenFlowError::LLMError("Invalid Anthropic response format".to_string())
            })?;

        Ok(Message::assistant(content))
    }

    async fn chat_stream(
        &self,
        _messages: Vec<Message>,
    ) -> crate::core::error::Result<tokio::sync::mpsc::Receiver<String>> {
        // TODO: Implement Anthropic streaming
        Err(ErenFlowError::LLMError(
            "Anthropic streaming not yet implemented".to_string(),
        ))
    }
}
