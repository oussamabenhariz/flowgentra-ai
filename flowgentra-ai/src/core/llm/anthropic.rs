//! Anthropic LLM Provider
//!
//! Supports Claude models (3 Opus, 3 Sonnet, 3 Haiku, and others).
//! Uses the Anthropic Messages API.
//!
//! Internally delegates to [`HttpLLMClient`] with [`AnthropicAdapter`].

use super::adapter::{AnthropicAdapter, HttpLLMClient};
use super::LLMConfig;

/// Anthropic Claude LLM client
///
/// Thin wrapper around [`HttpLLMClient`] using the Anthropic adapter.
pub type AnthropicClient = HttpLLMClient;

/// Create a new Anthropic client
#[allow(dead_code)]
pub fn new_anthropic_client(config: LLMConfig) -> AnthropicClient {
    HttpLLMClient::new(config, AnthropicAdapter)
}
