//! Groq Cloud LLM Provider
//!
//! Supports Groq Cloud inference API with high-speed LLM access.
//! Supports models like LLaMA, Mixtral, and others.
//!
//! Internally delegates to [`HttpLLMClient`] with [`OpenAICompatibleAdapter`].

use super::adapter::{HttpLLMClient, OpenAICompatibleAdapter};
use super::LLMConfig;

/// Groq Cloud LLM client
///
/// Thin wrapper around [`HttpLLMClient`] using the OpenAI-compatible adapter.
pub type GroqClient = HttpLLMClient;

/// Create a new Groq client
#[allow(dead_code)]
pub fn new_groq_client(config: LLMConfig) -> GroqClient {
    HttpLLMClient::new(config, OpenAICompatibleAdapter::new("Groq"))
}
