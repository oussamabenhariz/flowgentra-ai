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
//! use flowgentra_ai::core::state::DynState;
//! use flowgentra_ai::core::llm::Message;
//!
//! async fn my_handler(state: DynState) -> Result<DynState, Box<dyn std::error::Error>> {
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

    /// Estimate the cost in USD for this token usage based on model pricing.
    ///
    /// Returns `None` if the model is not in the pricing table.
    pub fn estimated_cost(&self, model: &str) -> Option<f64> {
        let (input_per_m, output_per_m) = model_pricing(model)?;
        let cost = (self.prompt_tokens as f64 * input_per_m
            + self.completion_tokens as f64 * output_per_m)
            / 1_000_000.0;
        Some(cost)
    }
}

/// Returns (input_price_per_million_tokens, output_price_per_million_tokens) in USD.
pub fn model_pricing(model: &str) -> Option<(f64, f64)> {
    let m = model.to_lowercase();
    match m.as_str() {
        // OpenAI
        s if s.starts_with("gpt-4o-mini") => Some((0.15, 0.60)),
        s if s.starts_with("gpt-4o") => Some((2.50, 10.00)),
        s if s.starts_with("gpt-4-turbo") => Some((10.00, 30.00)),
        "gpt-4" => Some((30.00, 60.00)),
        s if s.starts_with("gpt-3.5-turbo") => Some((0.50, 1.50)),
        s if s.starts_with("o1-mini") => Some((3.00, 12.00)),
        s if s.starts_with("o1") => Some((15.00, 60.00)),
        s if s.starts_with("o3-mini") => Some((1.10, 4.40)),
        // Anthropic
        s if s.contains("claude-3-5-sonnet") || s.contains("claude-sonnet-4") => {
            Some((3.00, 15.00))
        }
        s if s.contains("claude-3-5-haiku") || s.contains("claude-haiku-4") => Some((0.80, 4.00)),
        s if s.contains("claude-3-opus") || s.contains("claude-opus-4") => Some((15.00, 75.00)),
        s if s.contains("claude-3-sonnet") => Some((3.00, 15.00)),
        s if s.contains("claude-3-haiku") => Some((0.25, 1.25)),
        // Mistral
        s if s.contains("mistral-large") => Some((2.00, 6.00)),
        s if s.contains("mistral-small") => Some((0.20, 0.60)),
        s if s.contains("mistral-medium") => Some((2.70, 8.10)),
        // Groq (pricing varies, these are estimates)
        s if s.contains("llama-3") => Some((0.05, 0.08)),
        s if s.contains("mixtral") => Some((0.24, 0.24)),
        _ => None,
    }
}

// Re-export provider-specific clients
mod adapter;
mod anthropic;
mod azure;
pub mod cache;
mod factory;
pub mod fallback;
mod groq;
mod huggingface;
mod mistral;
mod ollama;
mod openai;
pub mod output_parser;
pub mod prompt_template;
mod retry;
pub mod token_counter;

pub use adapter::{HttpLLMClient, ProviderAdapter};
pub use anthropic::AnthropicClient;
pub use azure::AzureOpenAIClient;
pub use cache::CachedLLMClient;
pub use factory::create_llm_client;
pub use fallback::FallbackLLMClient;
pub use groq::GroqClient;
pub use huggingface::HuggingFaceClient;
pub use mistral::MistralClient;
pub use ollama::OllamaClient;
pub use openai::OpenAIClient;
pub use retry::RetryLLMClient;

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
/// # HuggingFace Inference API
/// llm:
///   provider: huggingface
///   model: mistralai/Mistral-7B-Instruct-v0.1
///   api_key: ${HF_API_TOKEN}
///
/// # HuggingFace Local TGI Server
/// llm:
///   provider: huggingface
///   model: mistralai/Mistral-7B-Instruct-v0.1
///   api_key: ""
///   extra_params:
///     mode: tgi
///     endpoint: "http://localhost:8080"
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

    /// HuggingFace (cloud API or local TGI server)
    HuggingFace,

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
            LLMProvider::HuggingFace => "https://api-inference.huggingface.co".to_string(),
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
            LLMProvider::HuggingFace => "huggingface",
            LLMProvider::Ollama => "ollama",
            LLMProvider::Azure => "azure",
            LLMProvider::Custom(_) => "custom",
        }
    }

    /// The conventional environment variable name for this provider's API key.
    /// Returns `None` for providers that don't require a key (Ollama).
    pub fn env_var(&self) -> Option<&'static str> {
        match self {
            LLMProvider::OpenAI => Some("OPENAI_API_KEY"),
            LLMProvider::Anthropic => Some("ANTHROPIC_API_KEY"),
            LLMProvider::Mistral => Some("MISTRAL_API_KEY"),
            LLMProvider::Groq => Some("GROQ_API_KEY"),
            LLMProvider::HuggingFace => Some("HUGGINGFACEHUB_API_TOKEN"),
            LLMProvider::Azure => Some("AZURE_OPENAI_KEY"),
            LLMProvider::Ollama => None,
            LLMProvider::Custom(_) => None,
        }
    }
}

// =============================================================================
// LLM Configuration
// =============================================================================

/// Response format for structured output.
///
/// When set to `Json`, providers that support `response_format` will include
/// `{ "type": "json_object" }` in the API request, guaranteeing valid JSON output.
/// For providers without native support, a system-prompt fallback is used.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
#[serde(rename_all = "snake_case")]
pub enum ResponseFormat {
    /// Default text output
    #[default]
    Text,
    /// Force JSON output (OpenAI `json_object`, Anthropic prompt-based)
    Json,
    /// JSON output constrained to a specific JSON Schema (OpenAI `json_schema`)
    JsonSchema {
        /// A name for the schema (required by OpenAI)
        name: String,
        /// The JSON Schema the output must conform to
        schema: serde_json::Value,
    },
}

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

    /// API key or authentication token for the provider.
    /// Pass it directly, or leave empty to fall back to the provider's
    /// conventional environment variable (e.g. `OPENAI_API_KEY`).
    /// A `.env` file in the working directory is loaded automatically as a fallback.
    pub api_key: String,

    /// Response format — set to `Json` or `JsonSchema` for structured output
    #[serde(default)]
    pub response_format: ResponseFormat,

    /// Provider-specific parameters
    #[serde(default)]
    pub extra_params: HashMap<String, serde_json::Value>,
}

impl LLMConfig {
    /// Create a new LLM configuration.
    ///
    /// If `api_key` is empty the constructor tries to resolve it automatically:
    /// 1. Reads the provider's conventional env var (e.g. `OPENAI_API_KEY`).
    /// 2. If not found in the environment, attempts to load a `.env` file from
    ///    the current working directory and retries.
    ///
    /// # Example
    /// ```
    /// use flowgentra_ai::core::llm::{LLMConfig, LLMProvider};
    ///
    /// // Explicit key
    /// let config = LLMConfig::new(LLMProvider::OpenAI, "gpt-4".to_string(), "sk-...".to_string());
    ///
    /// // Auto-resolve from OPENAI_API_KEY env var or .env file
    /// let config = LLMConfig::new(LLMProvider::OpenAI, "gpt-4".to_string(), String::new());
    /// ```
    pub fn new(provider: LLMProvider, model: String, api_key: String) -> Self {
        let resolved_key = if api_key.is_empty() {
            if let Some(var) = provider.env_var() {
                // Try the process environment first
                std::env::var(var).unwrap_or_else(|_| {
                    // Fall back to .env file in the working directory
                    let _ = dotenv::dotenv();
                    std::env::var(var).unwrap_or_default()
                })
            } else {
                String::new()
            }
        } else {
            api_key
        };

        LLMConfig {
            provider,
            model,
            temperature: None,
            max_tokens: None,
            top_p: None,
            api_key: resolved_key,
            response_format: ResponseFormat::default(),
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

    /// Set the response format for structured output
    pub fn with_response_format(mut self, format: ResponseFormat) -> Self {
        self.response_format = format;
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
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum MessageRole {
    /// System prompt that sets behavior
    System,

    /// User message or query
    #[default]
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
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Message {
    /// The role of the message sender
    pub role: MessageRole,

    /// The message content
    pub content: String,

    /// Optional tool calls made by the LLM
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,

    /// Tool call ID (for tool result messages, links back to the ToolCall.id)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

impl Message {
    /// Create a system message (sets LLM behavior)
    pub fn system(content: impl Into<String>) -> Self {
        Message {
            role: MessageRole::System,
            content: content.into(),
            ..Default::default()
        }
    }

    /// Create a user message
    pub fn user(content: impl Into<String>) -> Self {
        Message {
            role: MessageRole::User,
            content: content.into(),
            ..Default::default()
        }
    }

    /// Create an assistant message (LLM response)
    pub fn assistant(content: impl Into<String>) -> Self {
        Message {
            role: MessageRole::Assistant,
            content: content.into(),
            ..Default::default()
        }
    }

    /// Create a tool result message
    pub fn tool(content: impl Into<String>) -> Self {
        Message {
            role: MessageRole::Tool,
            content: content.into(),
            ..Default::default()
        }
    }

    /// Create a tool result message linked to a specific tool call
    pub fn tool_result(tool_call_id: impl Into<String>, content: impl Into<String>) -> Self {
        Message {
            role: MessageRole::Tool,
            content: content.into(),
            tool_call_id: Some(tool_call_id.into()),
            ..Default::default()
        }
    }

    /// Check if this message contains tool calls from the LLM
    pub fn has_tool_calls(&self) -> bool {
        self.tool_calls
            .as_ref()
            .map(|tc| !tc.is_empty())
            .unwrap_or(false)
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

/// A tool definition passed to the LLM so it knows what tools are available.
///
/// Follows the OpenAI function-calling format, which is also supported by
/// Mistral, Groq, and other OpenAI-compatible providers.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDefinition {
    /// Tool name (must match the MCP tool name for automatic routing)
    pub name: String,

    /// Human-readable description shown to the LLM
    pub description: String,

    /// JSON Schema describing the tool's input parameters
    pub parameters: serde_json::Value,
}

impl ToolDefinition {
    pub fn new(
        name: impl Into<String>,
        description: impl Into<String>,
        parameters: serde_json::Value,
    ) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            parameters,
        }
    }
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

    /// Default implementation calls chat() and returns None for usage.
    async fn chat_with_usage(
        &self,
        messages: Vec<Message>,
    ) -> crate::core::error::Result<(Message, Option<TokenUsage>)> {
        let msg = self.chat(messages).await?;
        Ok((msg, None))
    }

    /// Send messages with tool definitions. The LLM may respond with tool calls.
    ///
    /// When the LLM wants to call a tool, the returned `Message` will have
    /// `tool_calls` populated. The caller is responsible for executing the tools
    /// and feeding results back as `MessageRole::Tool` messages.
    ///
    /// Default implementation falls back to `chat()` (ignoring tools).
    async fn chat_with_tools(
        &self,
        messages: Vec<Message>,
        _tools: &[ToolDefinition],
    ) -> crate::core::error::Result<Message> {
        self.chat(messages).await
    }

    /// Receive a streaming response from the LLM
    ///
    /// Useful for long responses or UI updates
    /// Returns a channel that emits response chunks
    async fn chat_stream(
        &self,
        messages: Vec<Message>,
    ) -> crate::core::error::Result<tokio::sync::mpsc::Receiver<String>>;

    /// Receive a structured response from the LLM parsed into JSON
    async fn chat_structured(
        &self,
        messages: Vec<Message>,
    ) -> crate::core::error::Result<serde_json::Value> {
        let mut structured_messages = messages;
        structured_messages.push(Message::system(
            "You must respond with valid JSON matching the requested schema. \
            Do not include markdown formatting like ```json or any other text.",
        ));

        let msg = self.chat(structured_messages).await?;
        serde_json::from_str(&msg.content).map_err(|e| {
            crate::core::error::FlowgentraError::LLMError(format!(
                "Failed to parse structured output: {}",
                e
            ))
        })
    }
}

// Provider-specific clients are in separate modules
// Use factory::create_llm_client() to create the appropriate client
