//! LLM Factory
//!
//! Factory function to create the appropriate LLM
//! based on the provider configuration.
//!
//! All providers (except HuggingFace) now use [`HttpLLM`] with
//! a provider-specific [`ProviderAdapter`].

use super::adapter::{
    AnthropicAdapter, AzureAdapter, HttpLLM, OllamaAdapter, OpenAICompatibleAdapter,
};
use super::{LLMConfig, LLMProvider, LLM};
use crate::core::error::Result;
use std::sync::Arc;

/// Create an LLM based on the configuration
///
/// # Example
/// ```ignore
/// let config = LLMConfig::new(
///     LLMProvider::OpenAI,
///     "gpt-4".to_string(),
///     api_key
/// );
/// let client = create_llm(&config)?;
/// ```
pub fn create_llm(config: &LLMConfig) -> Result<Arc<dyn LLM>> {
    match &config.provider {
        LLMProvider::OpenAI => Ok(Arc::new(HttpLLM::new(
            config.clone(),
            OpenAICompatibleAdapter::new("OpenAI"),
        ))),
        LLMProvider::Anthropic => Ok(Arc::new(HttpLLM::new(config.clone(), AnthropicAdapter))),
        LLMProvider::Mistral => Ok(Arc::new(HttpLLM::new(
            config.clone(),
            OpenAICompatibleAdapter::new("Mistral"),
        ))),
        LLMProvider::Groq => Ok(Arc::new(HttpLLM::new(
            config.clone(),
            OpenAICompatibleAdapter::new("Groq"),
        ))),
        LLMProvider::HuggingFace => Ok(Arc::new(super::huggingface::HuggingFaceClient::new(
            config.clone(),
        ))),
        LLMProvider::Ollama => Ok(Arc::new(HttpLLM::new(config.clone(), OllamaAdapter))),
        LLMProvider::Azure => {
            let adapter = AzureAdapter::from_config(config)?;
            Ok(Arc::new(HttpLLM::new(config.clone(), adapter)))
        }
        LLMProvider::Custom(url) => {
            // For custom providers, check if it's Azure
            if url.contains("openai.azure.com") {
                let adapter = AzureAdapter::from_config(config)?;
                Ok(Arc::new(HttpLLM::new(config.clone(), adapter)))
            } else {
                // For other custom providers, default to OpenAI-compatible API
                Ok(Arc::new(HttpLLM::new(
                    config.clone(),
                    OpenAICompatibleAdapter::new("Custom"),
                )))
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
        assert!(create_llm(&config).is_ok());
    }

    #[test]
    fn test_create_anthropic_client() {
        let config = LLMConfig::new(
            LLMProvider::Anthropic,
            "claude-3-opus".to_string(),
            "test-key".to_string(),
        );
        assert!(create_llm(&config).is_ok());
    }

    #[test]
    fn test_create_ollama_client() {
        let config = LLMConfig::new(
            LLMProvider::Ollama,
            "mistral".to_string(),
            "".to_string(), // Ollama usually doesn't need an API key
        );
        assert!(create_llm(&config).is_ok());
    }

    #[test]
    fn test_create_huggingface_client() {
        let config = LLMConfig::new(
            LLMProvider::HuggingFace,
            "mistralai/Mistral-7B-Instruct-v0.1".to_string(),
            "hf_test_token".to_string(),
        );
        assert!(create_llm(&config).is_ok());
    }
}
