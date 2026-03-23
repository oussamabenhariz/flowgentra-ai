//! Ollama LLM Provider
//!
//! Supports running local LLMs via Ollama.
//! Ollama runs locally (typically on http://localhost:11434).
//!
//! To use:
//! 1. Install Ollama from https://ollama.ai
//! 2. Pull a model: `ollama pull mistral`
//! 3. Run: `ollama serve`
//! 4. Set provider to "ollama" in config
//!
//! Internally delegates to [`HttpLLMClient`] with [`OllamaAdapter`].

use super::adapter::{HttpLLMClient, OllamaAdapter};
use super::LLMConfig;

/// Ollama LLM client for local LLM execution
///
/// Thin wrapper around [`HttpLLMClient`] using the Ollama adapter.
pub type OllamaClient = HttpLLMClient;

/// Create a new Ollama client
#[allow(dead_code)]
pub fn new_ollama_client(config: LLMConfig) -> OllamaClient {
    HttpLLMClient::new(config, OllamaAdapter)
}
