//! Groq Cloud LLM Provider
//!
//! Supports Groq Cloud inference API with high-speed LLM access.
//! Supports models like LLaMA, Mixtral, and others.
//!
//! Internally delegates to [`HttpLLM`] with [`OpenAICompatibleAdapter`].

use super::adapter::{HttpLLM, OpenAICompatibleAdapter};
use super::LLMConfig;

/// Groq Cloud LLM
///
/// Thin wrapper around [`HttpLLM`] using the OpenAI-compatible adapter.
pub type GroqClient = HttpLLM;

/// Create a new Groq client
#[allow(dead_code)]
pub fn new_groq_client(config: LLMConfig) -> GroqClient {
    HttpLLM::new(config, OpenAICompatibleAdapter::new("Groq"))
}
