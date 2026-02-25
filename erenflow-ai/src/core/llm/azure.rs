//! Azure OpenAI LLM Provider
//!
//! Supports Azure OpenAI deployment with Azure authentication.
//! Requires additional configuration for Azure-specific settings.

use super::{LLMClient, LLMConfig, Message};
use crate::core::error::ErenFlowError;
use serde_json::json;

/// Azure OpenAI LLM client
#[derive(Clone)]
pub struct AzureOpenAIClient {
    config: LLMConfig,
    client: reqwest::Client,
    deployment_id: String,
    api_version: String,
}

impl AzureOpenAIClient {
    /// Create a new Azure OpenAI client
    ///
    /// Configuration should have:
    /// - api_key: Azure API key
    /// - model: Deployment ID (e.g., "gpt-4-deployment")
    /// - extra_params["resource_name"]: Azure resource name
    /// - extra_params["api_version"]: API version (default: "2024-02-15-preview")
    pub fn new(mut config: LLMConfig) -> crate::core::error::Result<Self> {
        let resource_name = config
            .extra_params
            .get("resource_name")
            .and_then(|v| v.as_str())
            .ok_or_else(|| {
                ErenFlowError::ConfigError(
                    "Azure OpenAI requires 'resource_name' in extra_params".to_string(),
                )
            })?
            .to_string();

        let api_version = config
            .extra_params
            .get("api_version")
            .and_then(|v| v.as_str())
            .unwrap_or("2024-02-15-preview")
            .to_string();

        // Override the provider's base URL with Azure endpoint
        config.provider = super::LLMProvider::Custom(format!(
            "https://{}.openai.azure.com/openai/deployments",
            resource_name
        ));

        Ok(Self {
            deployment_id: config.model.clone(),
            client: reqwest::Client::new(),
            config,
            api_version,
        })
    }

    /// Get the API endpoint URL
    fn get_endpoint(&self) -> String {
        format!(
            "{}/{}/chat/completions?api-version={}",
            self.config.provider.base_url(),
            self.deployment_id,
            self.api_version
        )
    }
}

#[async_trait::async_trait]
impl LLMClient for AzureOpenAIClient {
    async fn chat(&self, messages: Vec<Message>) -> crate::core::error::Result<Message> {
        let payload = json!({
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
            .header("api-key", &self.config.api_key)
            .header("Content-Type", "application/json")
            .json(&payload)
            .send()
            .await
            .map_err(|e| {
                ErenFlowError::LLMError(format!("Azure OpenAI API request failed: {}", e))
            })?;

        if !response.status().is_success() {
            let error_text = response.text().await.unwrap_or_default();
            return Err(ErenFlowError::LLMError(format!(
                "Azure OpenAI API error: {}",
                error_text
            )));
        }

        let data: serde_json::Value = response.json().await.map_err(|e| {
            ErenFlowError::LLMError(format!("Failed to parse Azure OpenAI response: {}", e))
        })?;

        let content = data
            .get("choices")
            .and_then(|c| c.get(0))
            .and_then(|c| c.get("message"))
            .and_then(|m| m.get("content"))
            .and_then(|c| c.as_str())
            .ok_or_else(|| {
                ErenFlowError::LLMError("Invalid Azure OpenAI response format".to_string())
            })?;

        Ok(Message::assistant(content))
    }

    async fn chat_stream(
        &self,
        _messages: Vec<Message>,
    ) -> crate::core::error::Result<tokio::sync::mpsc::Receiver<String>> {
        // TODO: Implement Azure OpenAI streaming
        Err(ErenFlowError::LLMError(
            "Azure OpenAI streaming not yet implemented".to_string(),
        ))
    }
}
