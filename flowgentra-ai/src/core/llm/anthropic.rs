//! Anthropic LLM Provider
//!
//! Supports Claude models (3 Opus, 3 Sonnet, 3 Haiku, and others).
//! Uses the Anthropic Messages API.
//!
//! Internally delegates to [`HttpLLM`] with [`AnthropicAdapter`].

use super::adapter::{AnthropicAdapter, HttpLLM};
use super::LLMConfig;

/// Anthropic Claude LLM
///
/// Thin wrapper around [`HttpLLM`] using the Anthropic adapter.
pub type AnthropicClient = HttpLLM;

/// Create a new Anthropic client
#[allow(dead_code)]
pub fn new_anthropic_client(config: LLMConfig) -> AnthropicClient {
    HttpLLM::new(config, AnthropicAdapter)
}
