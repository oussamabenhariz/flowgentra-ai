//! Azure OpenAI LLM Provider
//!
//! Supports Azure OpenAI deployment with Azure authentication.
//! Requires additional configuration for Azure-specific settings.
//!
//! Internally delegates to [`HttpLLMClient`] with [`AzureAdapter`].

use super::adapter::{AzureAdapter, HttpLLMClient};
use super::LLMConfig;

/// Azure OpenAI LLM client
///
/// Thin wrapper around [`HttpLLMClient`] using the Azure adapter.
pub type AzureOpenAIClient = HttpLLMClient;

/// Create a new Azure OpenAI client
///
/// Configuration should have:
/// - api_key: Azure API key
/// - model: Deployment ID (e.g., "gpt-4-deployment")
/// - extra_params["resource_name"]: Azure resource name
/// - extra_params["api_version"]: API version (default: "2024-02-15-preview")
#[allow(dead_code)]
pub fn new_azure_client(config: LLMConfig) -> crate::core::error::Result<AzureOpenAIClient> {
    let adapter = AzureAdapter::from_config(&config)?;
    Ok(HttpLLMClient::new(config, adapter))
}
