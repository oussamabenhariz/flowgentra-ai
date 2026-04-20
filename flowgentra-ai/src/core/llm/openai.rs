//! OpenAI LLM Provider
//!
//! Supports GPT-4, GPT-3.5-turbo, and other OpenAI models.
//! Uses the OpenAI chat completion API.
//!
//! Internally delegates to [`HttpLLM`] with [`OpenAICompatibleAdapter`].

use super::adapter::{HttpLLM, OpenAICompatibleAdapter};
use super::LLMConfig;

/// OpenAI LLM
///
/// Thin wrapper around [`HttpLLM`] using the OpenAI-compatible adapter.
pub type OpenAIClient = HttpLLM;

/// Create a new OpenAI client
#[allow(dead_code)]
pub fn new_openai_client(config: LLMConfig) -> OpenAIClient {
    HttpLLM::new(config, OpenAICompatibleAdapter::new("OpenAI"))
}
