//! LLM Client Factory
//!
//! Factory function to create the appropriate LLM client
//! based on the provider configuration.

use super::{LLMClient, LLMConfig, LLMProvider};
use crate::core::error::Result;
use std::sync::Arc;

/// Create an LLM client based on the configuration
///
/// # Example
/// ```ignore
/// let config = LLMConfig::new(
///     LLMProvider::OpenAI,
///     "gpt-4".to_string(),
///     api_key
/// );
/// let client = create_llm_client(&config)?;
/// ```
pub fn create_llm_client(config: &LLMConfig) -> Result<Arc<dyn LLMClient>> {
    match &config.provider {
        LLMProvider::OpenAI => Ok(Arc::new(super::openai::OpenAIClient::new(config.clone()))),
        LLMProvider::Anthropic => Ok(Arc::new(super::anthropic::AnthropicClient::new(
            config.clone(),
        ))),
        LLMProvider::Mistral => Ok(Arc::new(super::mistral::MistralClient::new(config.clone()))),
        LLMProvider::Groq => Ok(Arc::new(super::groq::GroqClient::new(config.clone()))),
        LLMProvider::Ollama => Ok(Arc::new(super::ollama::OllamaClient::new(config.clone()))),
        LLMProvider::Azure => Ok(Arc::new(super::azure::AzureOpenAIClient::new(
            config.clone(),
        )?)),
        LLMProvider::Custom(url) => {
            // For custom providers, check if it's Azure
            if url.contains("openai.azure.com") {
                Ok(Arc::new(super::azure::AzureOpenAIClient::new(
                    config.clone(),
                )?))
            } else {
                // For other custom providers, default to OpenAI-compatible API
                Ok(Arc::new(super::openai::OpenAIClient::new(config.clone())))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_openai_client() {
        let config = LLMConfig::new(
            LLMProvider::OpenAI,
            "gpt-4".to_string(),
            "test-key".to_string(),
        );
        assert!(create_llm_client(&config).is_ok());
    }

    #[test]
    fn test_create_anthropic_client() {
        let config = LLMConfig::new(
            LLMProvider::Anthropic,
            "claude-3-opus".to_string(),
            "test-key".to_string(),
        );
        assert!(create_llm_client(&config).is_ok());
    }

    #[test]
    fn test_create_ollama_client() {
        let config = LLMConfig::new(
            LLMProvider::Ollama,
            "mistral".to_string(),
            "".to_string(), // Ollama usually doesn't need an API key
        );
        assert!(create_llm_client(&config).is_ok());
    }
}
