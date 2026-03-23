//! Mistral AI LLM Provider
//!
//! Supports Mistral AI models (Mistral-7B, Mistral-Medium, Mistral-Large).
//! Uses the Mistral AI API (OpenAI-compatible format).
//!
//! Internally delegates to [`HttpLLMClient`] with [`OpenAICompatibleAdapter`].

use super::adapter::{HttpLLMClient, OpenAICompatibleAdapter};
use super::LLMConfig;

/// Mistral AI LLM client
///
/// Thin wrapper around [`HttpLLMClient`] using the OpenAI-compatible adapter.
pub type MistralClient = HttpLLMClient;

/// Create a new Mistral client
#[allow(dead_code)]
pub fn new_mistral_client(config: LLMConfig) -> MistralClient {
    HttpLLMClient::new(config, OpenAICompatibleAdapter::new("Mistral"))
}
