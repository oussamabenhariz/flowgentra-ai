//! Mistral AI LLM Provider
//!
//! Supports Mistral AI models (Mistral-7B, Mistral-Medium, Mistral-Large).
//! Uses the Mistral AI API (OpenAI-compatible format).
//!
//! Internally delegates to [`HttpLLM`] with [`OpenAICompatibleAdapter`].

use super::adapter::{HttpLLM, OpenAICompatibleAdapter};
use super::LLMConfig;

/// Mistral AI LLM
///
/// Thin wrapper around [`HttpLLM`] using the OpenAI-compatible adapter.
pub type MistralClient = HttpLLM;

/// Create a new Mistral client
#[allow(dead_code)]
pub fn new_mistral_client(config: LLMConfig) -> MistralClient {
    HttpLLM::new(config, OpenAICompatibleAdapter::new("Mistral"))
}
