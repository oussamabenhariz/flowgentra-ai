//! OpenAI LLM Provider
//!
//! Supports GPT-4, GPT-3.5-turbo, and other OpenAI models.
//! Uses the OpenAI chat completion API.
//!
//! Internally delegates to [`HttpLLMClient`] with [`OpenAICompatibleAdapter`].

use super::adapter::{HttpLLMClient, OpenAICompatibleAdapter};
use super::LLMConfig;

/// OpenAI LLM client
///
/// Thin wrapper around [`HttpLLMClient`] using the OpenAI-compatible adapter.
pub type OpenAIClient = HttpLLMClient;

/// Create a new OpenAI client
#[allow(dead_code)]
pub fn new_openai_client(config: LLMConfig) -> OpenAIClient {
    HttpLLMClient::new(config, OpenAICompatibleAdapter::new("OpenAI"))
}
