//! # LLM Integration
//!
//! Support for multiple Large Language Model providers with a unified interface.
//!
//! ## Supported Providers
//!
//! - **OpenAI** - GPT-4, GPT-3.5-turbo
//! - **Anthropic** - Claude models
//! - **Mistral** - Mistral AI models
//! - **Groq** - Groq Cloud models
//! - **Ollama** - Local LLM via Ollama
//! - **Custom** - Any provider with compatible API
//!
//! ## Configuration
//!
//! Specify your LLM in the agent's YAML config:
//!
//! ```yaml
//! llm:
//!   provider: openai
//!   model: gpt-4
//!   temperature: 0.7
//!   api_key: ${OPENAI_API_KEY}
//! ```
//!
//! Environment variable substitution is supported via `${VAR_NAME}` syntax.
//!
//! ## Usage in Handlers
//!
//! ```no_run
//! use erenflow_ai::core::state::State;
//! use erenflow_ai::core::llm::Message;
//!
//! async fn my_handler(mut state: State) -> Result<State, Box<dyn std::error::Error>> {
//!     // Get LLM client from state (context-dependent)
//!     // Send a message to the LLM
//!     let messages = vec![
//!         Message::system("You are helpful assistant"),
//!         Message::user("What is Rust?"),
//!     ];
//!     
//!     // Response would be processed and added to state
//!     Ok(state)
//! }
//! ```

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;

/// Token usage from an LLM API response
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TokenUsage {
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    pub total_tokens: u64,
}

impl TokenUsage {
    pub fn new(prompt: u64, completion: u64) -> Self {
        Self {
            prompt_tokens: prompt,
            completion_tokens: completion,
            total_tokens: prompt + completion,
        }
    }

    pub fn add(&self, other: &TokenUsage) -> TokenUsage {
        TokenUsage {
            prompt_tokens: self.prompt_tokens + other.prompt_tokens,
            completion_tokens: self.completion_tokens + other.completion_tokens,
            total_tokens: self.total_tokens + other.total_tokens,
        }
    }
}

// Re-export provider-specific clients
pub mod anthropic;
pub mod azure;
pub mod factory;
pub mod groq;
pub mod mistral;
pub mod ollama;
pub mod openai;

pub use anthropic::AnthropicClient;
pub use azure::AzureOpenAIClient;
pub use factory::create_llm_client;
pub use groq::GroqClient;
pub use mistral::MistralClient;
pub use ollama::OllamaClient;
pub use openai::OpenAIClient;

// =============================================================================
// LLM Provider
// =============================================================================

/// Supported LLM providers
///
/// The framework provides built-in support for major providers,
/// plus a Custom option for flexible integration.
///
/// ## Configuration Examples
///
/// ```yaml
/// # OpenAI
/// llm:
///   provider: openai
///   model: gpt-4
///   api_key: ${OPENAI_API_KEY}
///
/// # Anthropic Claude
/// llm:
///   provider: anthropic
///   model: claude-3-opus-20240229
///   api_key: ${ANTHROPIC_API_KEY}
///
/// # Local Ollama
/// llm:
///   provider: ollama
///   model: mistral
///   # No API key needed for local Ollama
///
/// # Azure OpenAI
/// llm:
///   provider: azure
///   model: gpt-4-deployment
///   api_key: ${AZURE_OPENAI_KEY}
///   extra_params:
///     resource_name: my-azure-resource
///     api_version: "2024-02-15-preview"
/// ```
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum LLMProvider {
    /// OpenAI (GPT-4, GPT-3.5-turbo, etc.)
    OpenAI,

    /// Anthropic Claude
    Anthropic,

    /// Mistral AI
    Mistral,

    /// Groq Cloud
    Groq,

    /// Local LLM via Ollama (http://localhost:11434)
    Ollama,

    /// Azure OpenAI
    Azure,

    /// Custom provider with specified base URL
    Custom(String),
}

impl LLMProvider {
    /// Get the base API URL for this provider
    pub fn base_url(&self) -> String {
        match self {
            LLMProvider::OpenAI => "https://api.openai.com/v1".to_string(),
            LLMProvider::Anthropic => "https://api.anthropic.com/v1".to_string(),
            LLMProvider::Mistral => "https://api.mistral.ai/v1".to_string(),
            LLMProvider::Groq => "https://api.groq.com/v1".to_string(),
            LLMProvider::Ollama => "http://localhost:11434/api".to_string(),
            LLMProvider::Azure => "".to_string(), // Will be constructed per-request
            LLMProvider::Custom(url) => url.clone(),
        }
    }

    /// Get the provider name as a string
    pub fn name(&self) -> &str {
        match self {
            LLMProvider::OpenAI => "openai",
            LLMProvider::Anthropic => "anthropic",
            LLMProvider::Mistral => "mistral",
            LLMProvider::Groq => "groq",
            LLMProvider::Ollama => "ollama",
            LLMProvider::Azure => "azure",
            LLMProvider::Custom(_) => "custom",
        }
    }
}

// =============================================================================
// LLM Configuration
// =============================================================================

/// Configuration for LLM integration
///
/// Contains all settings needed to connect to and use an LLM provider.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LLMConfig {
    /// Which provider to use
    pub provider: LLMProvider,

    /// Model identifier (e.g., "gpt-4", "claude-3-opus")
    pub model: String,

    /// Controls response randomness (0.0 to 2.0)
    /// - 0.0 = deterministic
    /// - 1.0 = balanced (default)
    /// - 2.0 = very random
    pub temperature: Option<f32>,

    /// Maximum tokens to generate in response
    pub max_tokens: Option<usize>,

    /// Nucleus sampling parameter (0.0 to 1.0)
    /// Only tokens with cumulative probability up to top_p are considered
    pub top_p: Option<f32>,

    /// API key or authentication token for the provider
    /// Use environment variable syntax like `${OPENAI_API_KEY}`
    pub api_key: String,

    /// Provider-specific parameters
    #[serde(default)]
    pub extra_params: HashMap<String, serde_json::Value>,
}

impl LLMConfig {
    /// Create a new LLM configuration with required parameters
    ///
    /// # Example
    /// ```
    /// use erenflow_ai::core::llm::{LLMConfig, LLMProvider};
    ///
    /// let config = LLMConfig::new(
    ///     LLMProvider::OpenAI,
    ///     "gpt-4".to_string(),
    ///     "sk-...".to_string()
    /// );
    /// ```
    pub fn new(provider: LLMProvider, model: String, api_key: String) -> Self {
        LLMConfig {
            provider,
            model,
            temperature: None,
            max_tokens: None,
            top_p: None,
            api_key,
            extra_params: HashMap::new(),
        }
    }

    /// Set the temperature (response randomness)
    pub fn with_temperature(mut self, temperature: f32) -> Self {
        self.temperature = Some(temperature.clamp(0.0, 2.0));
        self
    }

    /// Set the maximum number of tokens in the response
    pub fn with_max_tokens(mut self, max_tokens: usize) -> Self {
        self.max_tokens = Some(max_tokens);
        self
    }

    /// Set the nucleus sampling parameter (top_p)
    pub fn with_top_p(mut self, top_p: f32) -> Self {
        self.top_p = Some(top_p.clamp(0.0, 1.0));
        self
    }

    /// Add an extra parameter (provider-specific)
    pub fn with_extra_param(mut self, key: String, value: serde_json::Value) -> Self {
        self.extra_params.insert(key, value);
        self
    }

    /// Create an LLM client based on this configuration
    ///
    /// # Example
    /// ```ignore
    /// let config = LLMConfig::new(LLMProvider::OpenAI, "gpt-4".to_string(), api_key);
    /// let client = config.create_client()?;
    /// ```
    pub fn create_client(&self) -> crate::core::error::Result<std::sync::Arc<dyn LLMClient>> {
        factory::create_llm_client(self)
    }
}

// =============================================================================
// Message Types
// =============================================================================

/// A role in a conversation
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum MessageRole {
    /// System prompt that sets behavior
    System,

    /// User message or query
    User,

    /// LLM response
    Assistant,

    /// Tool/function call result
    Tool,
}

/// A single message in a conversation
///
/// Messages are organized by role and accumulate to form a conversation context
/// that the LLM uses to generate responses.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    /// The role of the message sender
    pub role: MessageRole,

    /// The message content
    pub content: String,

    /// Optional tool calls made by the LLM
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
}

impl Message {
    /// Create a system message (sets LLM behavior)
    pub fn system(content: impl Into<String>) -> Self {
        Message {
            role: MessageRole::System,
            content: content.into(),
            tool_calls: None,
        }
    }

    /// Create a user message
    pub fn user(content: impl Into<String>) -> Self {
        Message {
            role: MessageRole::User,
            content: content.into(),
            tool_calls: None,
        }
    }

    /// Create an assistant message (LLM response)
    pub fn assistant(content: impl Into<String>) -> Self {
        Message {
            role: MessageRole::Assistant,
            content: content.into(),
            tool_calls: None,
        }
    }

    /// Create a tool result message
    pub fn tool(content: impl Into<String>) -> Self {
        Message {
            role: MessageRole::Tool,
            content: content.into(),
            tool_calls: None,
        }
    }

    /// Whether this message is from the system (behavior prompt).
    pub fn is_system(&self) -> bool {
        self.role == MessageRole::System
    }

    /// Whether this message is from the user.
    pub fn is_user(&self) -> bool {
        self.role == MessageRole::User
    }

    /// Whether this message is from the assistant (LLM).
    pub fn is_assistant(&self) -> bool {
        self.role == MessageRole::Assistant
    }

    /// Whether this message is a tool result.
    pub fn is_tool(&self) -> bool {
        self.role == MessageRole::Tool
    }
}

impl fmt::Display for Message {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let role = match self.role {
            MessageRole::System => "system",
            MessageRole::User => "user",
            MessageRole::Assistant => "assistant",
            MessageRole::Tool => "tool",
        };
        write!(f, "[{}] {}", role, self.content)
    }
}

/// A tool call made by the LLM (when using function calling)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    /// Unique identifier for this tool call
    pub id: String,

    /// Name of the tool to call
    pub name: String,

    /// Arguments to pass to the tool (JSON)
    pub arguments: serde_json::Value,
}

// =============================================================================
// LLM Client Trait
// =============================================================================

/// Trait for interacting with LLM providers
///
/// Implementations handle provider-specific API details,
/// authentication, and response parsing.
#[async_trait::async_trait]
pub trait LLMClient: Send + Sync {
    /// Send messages to the LLM and get a response
    ///
    /// # Arguments
    /// - `messages`: Conversation history
    ///
    /// # Returns
    /// The LLM's response as a new message
    async fn chat(&self, messages: Vec<Message>) -> crate::core::error::Result<Message>;

    /// Send messages and get response with token usage (for observability).
    /// Default implementation calls chat() and returns None for usage.
    async fn chat_with_usage(
        &self,
        messages: Vec<Message>,
    ) -> crate::core::error::Result<(Message, Option<TokenUsage>)> {
        let msg = self.chat(messages).await?;
        Ok((msg, None))
    }

    /// Receive a streaming response from the LLM
    ///
    /// Useful for long responses or UI updates
    /// Returns a channel that emits response chunks
    async fn chat_stream(
        &self,
        messages: Vec<Message>,
    ) -> crate::core::error::Result<tokio::sync::mpsc::Receiver<String>>;
}

// Provider-specific clients are in separate modules
// Use factory::create_llm_client() to create the appropriate client
