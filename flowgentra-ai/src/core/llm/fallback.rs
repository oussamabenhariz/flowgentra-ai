//! Fallback LLM Client — try multiple providers in order
//!
//! When the primary provider fails, automatically falls through to the next
//! one in the chain. Useful for reliability and cost optimization.

use std::sync::Arc;

use super::{LLMClient, Message, TokenUsage, ToolDefinition};

/// An LLM client that tries multiple providers in order.
///
/// If the primary client fails, it tries the next one, and so on.
/// All clients must implement `LLMClient`.
///
/// # Example
/// ```ignore
/// let primary = create_llm_client(&openai_config)?;
/// let fallback = create_llm_client(&anthropic_config)?;
/// let local = create_llm_client(&ollama_config)?;
///
/// let client = FallbackLLMClient::new(primary)
///     .with_fallback(fallback)
///     .with_fallback(local);
///
/// // Tries OpenAI first, then Anthropic, then Ollama
/// let response = client.chat(messages).await?;
/// ```
pub struct FallbackLLMClient {
    clients: Vec<Arc<dyn LLMClient>>,
}

impl FallbackLLMClient {
    /// Create with a primary client.
    pub fn new(primary: Arc<dyn LLMClient>) -> Self {
        Self {
            clients: vec![primary],
        }
    }

    /// Add a fallback client. Clients are tried in the order they are added.
    pub fn with_fallback(mut self, client: Arc<dyn LLMClient>) -> Self {
        self.clients.push(client);
        self
    }

    /// Create from a list of clients (first = primary).
    pub fn from_clients(clients: Vec<Arc<dyn LLMClient>>) -> Self {
        assert!(!clients.is_empty(), "At least one client is required");
        Self { clients }
    }
}

#[async_trait::async_trait]
impl LLMClient for FallbackLLMClient {
    async fn chat(&self, messages: Vec<Message>) -> crate::core::error::Result<Message> {
        let mut last_error = None;

        for (i, client) in self.clients.iter().enumerate() {
            match client.chat(messages.clone()).await {
                Ok(response) => {
                    if i > 0 {
                        tracing::info!("Fallback LLM succeeded with client #{}", i);
                    }
                    return Ok(response);
                }
                Err(e) => {
                    tracing::warn!("LLM client #{} failed: {}, trying next...", i, e);
                    last_error = Some(e);
                }
            }
        }

        Err(last_error.unwrap_or_else(|| {
            crate::core::error::FlowgentraError::LLMError("All fallback clients failed".to_string())
        }))
    }

    async fn chat_with_usage(
        &self,
        messages: Vec<Message>,
    ) -> crate::core::error::Result<(Message, Option<TokenUsage>)> {
        let mut last_error = None;

        for (i, client) in self.clients.iter().enumerate() {
            match client.chat_with_usage(messages.clone()).await {
                Ok(response) => {
                    if i > 0 {
                        tracing::info!("Fallback LLM succeeded with client #{}", i);
                    }
                    return Ok(response);
                }
                Err(e) => {
                    tracing::warn!("LLM client #{} failed: {}, trying next...", i, e);
                    last_error = Some(e);
                }
            }
        }

        Err(last_error.unwrap_or_else(|| {
            crate::core::error::FlowgentraError::LLMError("All fallback clients failed".to_string())
        }))
    }

    async fn chat_with_tools(
        &self,
        messages: Vec<Message>,
        tools: &[ToolDefinition],
    ) -> crate::core::error::Result<Message> {
        let mut last_error = None;

        for (i, client) in self.clients.iter().enumerate() {
            match client.chat_with_tools(messages.clone(), tools).await {
                Ok(response) => {
                    if i > 0 {
                        tracing::info!("Fallback LLM succeeded with client #{}", i);
                    }
                    return Ok(response);
                }
                Err(e) => {
                    tracing::warn!("LLM client #{} failed: {}, trying next...", i, e);
                    last_error = Some(e);
                }
            }
        }

        Err(last_error.unwrap_or_else(|| {
            crate::core::error::FlowgentraError::LLMError("All fallback clients failed".to_string())
        }))
    }

    async fn chat_stream(
        &self,
        messages: Vec<Message>,
    ) -> crate::core::error::Result<tokio::sync::mpsc::Receiver<String>> {
        let mut last_error = None;

        for (i, client) in self.clients.iter().enumerate() {
            match client.chat_stream(messages.clone()).await {
                Ok(rx) => {
                    if i > 0 {
                        tracing::info!("Fallback LLM stream succeeded with client #{}", i);
                    }
                    return Ok(rx);
                }
                Err(e) => {
                    tracing::warn!("LLM stream client #{} failed: {}, trying next...", i, e);
                    last_error = Some(e);
                }
            }
        }

        Err(last_error.unwrap_or_else(|| {
            crate::core::error::FlowgentraError::LLMError(
                "All fallback clients failed for streaming".to_string(),
            )
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fallback_creation() {
        // Just test that the types compose correctly
        // (Can't test actual LLM calls without a mock)
        let _ = FallbackLLMClient::from_clients;
    }
}
