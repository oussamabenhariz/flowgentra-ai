//! # Pluggable Node Architecture
//!
//! Trait-based system for composable, reusable node implementations.
//!
//! ## Core Abstraction
//!
//! All nodes implement the `PluggableNode` trait which defines:
//! - Input/output compatibility
//! - Error handling
//! - Metadata and configuration
//!
//! ## Built-in Nodes
//!
//! - **LLMNode** - Integration with language models
//! - **ToolNode** - Execute MCP tools
//! - **ConditionalRouter** - Route based on state conditions
//! - **ParallelExecutor** - Run branches concurrently
//! - **RetryNode** - Automatic retry with backoff
//! - **TimeoutNode** - Enforce execution timeout
//! - **HumanInTheLoopNode** - Wait for human approval
//!
//! ## Example
//!
//! ```ignore
//! use flowgentra_ai::core::nodes::{PluggableNode, LLMNode};
//! use flowgentra_ai::core::state::State;
//!
//! let mut node = LLMNode::new("analysis")
//!     .with_prompt("Analyze the following: {data}")
//!     .with_temperature(0.7);
//!
//! let result = node.run(State::new(Default::default())).await?;
//! ```

use crate::core::error::Result;
use crate::core::state::State;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::task::{Context, Poll};
use std::time::Duration;
use tower::Service;

/// Output from a node execution
#[derive(Debug, Clone)]
pub struct NodeOutput<T: State> {
    /// The resulting state
    pub state: T,

    /// Metadata about the execution
    pub metadata: HashMap<String, serde_json::Value>,

    /// Whether execution succeeded
    pub success: bool,

    /// Error message if failed
    pub error: Option<String>,

    /// Execution time in milliseconds
    pub execution_time_ms: u128,
}

impl<T: State> NodeOutput<T> {
    /// Create a successful output
    pub fn success(state: T) -> Self {
        NodeOutput {
            state,
            metadata: HashMap::new(),
            success: true,
            error: None,
            execution_time_ms: 0,
        }
    }

    /// Create a failed output
    pub fn error(state: T, error: impl Into<String>) -> Self {
        NodeOutput {
            state,
            metadata: HashMap::new(),
            success: false,
            error: Some(error.into()),
            execution_time_ms: 0,
        }
    }

    /// Add metadata to the output
    pub fn with_metadata(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        self.metadata.insert(key.into(), value);
        self
    }

    /// Set execution time
    pub fn with_duration(mut self, duration: Duration) -> Self {
        self.execution_time_ms = duration.as_millis();
        self
    }
}

/// Core trait for pluggable nodes
///
/// All nodes must implement this trait, which defines:
/// - Execution logic
/// - Configuration and metadata
/// - Input/output validation
///
/// # Type Parameter
/// - `T` - The state type (typically `State`)
#[async_trait]
pub trait PluggableNode<T: State>: Send + Sync {
    /// Execute this node with the given state
    async fn run(&self, state: T) -> Result<NodeOutput<T>>;

    /// Get the name of this node
    fn name(&self) -> &str;

    /// Get the node type
    fn node_type(&self) -> &str;

    /// Get node configuration
    fn config(&self) -> &HashMap<String, serde_json::Value>;

    /// Validate that the node can execute with this state
    fn validate_input(&self, state: &T) -> Result<()> {
        let _state = state;
        Ok(())
    }

    /// Get metadata about this node
    fn metadata(&self) -> HashMap<String, String> {
        HashMap::new()
    }

    /// Check if this node can be composed with another
    fn is_composable_with(&self, other_type: &str) -> bool {
        let _other_type = other_type;
        true
    }

    /// Clone the node as a trait object
    fn clone_box(&self) -> Box<dyn PluggableNode<T>>;

    /// Convert this node into a Tower Service
    fn into_service(self) -> NodeService<T>
    where
        Self: Sized + 'static,
    {
        NodeService::new(Box::new(self))
    }
}

// Implement Clone for trait objects with generic state
impl<T: State> Clone for Box<dyn PluggableNode<T>> {
    fn clone(&self) -> Self {
        self.clone_box()
    }
}

// =============================================================================
// Tower Service Integration
// =============================================================================

/// A Tower Service adapter for PluggableNodes
///
/// This provides direct compatibility with `tower` ecosystem middleware
/// (e.g., Retry, Timeout, Rate Limiting, Load Balancing, Tracing).
#[derive(Clone)]
pub struct NodeService<T: State> {
    node: std::sync::Arc<Box<dyn PluggableNode<T>>>,
}

impl<T: State> NodeService<T> {
    /// Create a new NodeService from a pluggable node
    pub fn new(node: Box<dyn PluggableNode<T>>) -> Self {
        Self {
            node: std::sync::Arc::new(node),
        }
    }
}

impl<T: State> Service<T> for NodeService<T> {
    type Response = NodeOutput<T>;
    type Error = crate::core::error::FlowgentraError;
    type Future =
        Pin<Box<dyn Future<Output = std::result::Result<Self::Response, Self::Error>> + Send>>;

    fn poll_ready(&mut self, _cx: &mut Context<'_>) -> Poll<std::result::Result<(), Self::Error>> {
        Poll::Ready(Ok(()))
    }

    fn call(&mut self, state: T) -> Self::Future {
        let node = self.node.clone();
        Box::pin(async move { node.run(state).await })
    }
}

/// Configuration for LLM nodes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LLMNodeConfig {
    /// Name of the LLM node
    pub name: String,

    /// Prompt template for the LLM
    /// Can contain {variable_name} placeholders
    pub prompt: String,

    /// Temperature for LLM (0.0 - 1.0)
    #[serde(default = "default_temperature")]
    pub temperature: f32,

    /// Maximum tokens in response
    #[serde(default)]
    pub max_tokens: Option<usize>,

    /// System prompt/instructions
    #[serde(default)]
    pub system_prompt: Option<String>,

    /// Variables to extract from state for the prompt
    #[serde(default)]
    pub input_variables: Vec<String>,

    /// Where to store the LLM response in state
    #[serde(default = "default_output_key")]
    pub output_key: String,

    /// Additional configuration
    #[serde(default)]
    pub config: HashMap<String, serde_json::Value>,
}

fn default_temperature() -> f32 {
    0.7
}

fn default_output_key() -> String {
    "llm_response".to_string()
}

impl LLMNodeConfig {
    /// Create a new LLM node configuration
    pub fn new(name: impl Into<String>, prompt: impl Into<String>) -> Self {
        LLMNodeConfig {
            name: name.into(),
            prompt: prompt.into(),
            temperature: 0.7,
            max_tokens: None,
            system_prompt: None,
            input_variables: Vec::new(),
            output_key: "llm_response".to_string(),
            config: HashMap::new(),
        }
    }

    /// Set temperature
    pub fn with_temperature(mut self, temp: f32) -> Self {
        self.temperature = temp.clamp(0.0, 1.0);
        self
    }

    /// Set max tokens
    pub fn with_max_tokens(mut self, max: usize) -> Self {
        self.max_tokens = Some(max);
        self
    }

    /// Set system prompt
    pub fn with_system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.system_prompt = Some(prompt.into());
        self
    }

    /// Add an input variable
    pub fn add_input_variable(mut self, var: impl Into<String>) -> Self {
        self.input_variables.push(var.into());
        self
    }

    /// Set output key
    pub fn with_output_key(mut self, key: impl Into<String>) -> Self {
        self.output_key = key.into();
        self
    }
}

/// Configuration for Tool nodes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolNodeConfig {
    /// Name of the tool node
    pub name: String,

    /// MCP tool to execute
    pub tool_name: String,

    /// Tool method/operation
    pub method: String,

    /// Input parameter mapping from state
    #[serde(default)]
    pub input_mapping: HashMap<String, String>,

    /// Output key where to store result
    #[serde(default = "default_output_key")]
    pub output_key: String,

    /// Whether to fail if tool fails
    #[serde(default = "default_fail_on_error")]
    pub fail_on_error: bool,

    /// Additional configuration
    #[serde(default)]
    pub config: HashMap<String, serde_json::Value>,
}

fn default_fail_on_error() -> bool {
    true
}

impl ToolNodeConfig {
    /// Create a new tool node configuration
    pub fn new(
        name: impl Into<String>,
        tool_name: impl Into<String>,
        method: impl Into<String>,
    ) -> Self {
        ToolNodeConfig {
            name: name.into(),
            tool_name: tool_name.into(),
            method: method.into(),
            input_mapping: HashMap::new(),
            output_key: "tool_result".to_string(),
            fail_on_error: true,
            config: HashMap::new(),
        }
    }

    /// Add input mapping
    pub fn map_input(
        mut self,
        state_key: impl Into<String>,
        tool_param: impl Into<String>,
    ) -> Self {
        self.input_mapping
            .insert(state_key.into(), tool_param.into());
        self
    }

    /// Set output key
    pub fn with_output_key(mut self, key: impl Into<String>) -> Self {
        self.output_key = key.into();
        self
    }

    /// Set fail_on_error
    pub fn with_fail_on_error(mut self, fail: bool) -> Self {
        self.fail_on_error = fail;
        self
    }
}

/// Configuration for Conditional Router nodes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConditionalRouterConfig {
    /// Name of the router
    pub name: String,

    /// Routing rules (condition -> next_node)
    pub rules: Vec<(String, String)>, // (condition_name, target_node)

    /// Default route if no conditions match
    pub default_route: Option<String>,

    /// Additional configuration
    #[serde(default)]
    pub config: HashMap<String, serde_json::Value>,
}

impl ConditionalRouterConfig {
    /// Create a new router configuration
    pub fn new(name: impl Into<String>) -> Self {
        ConditionalRouterConfig {
            name: name.into(),
            rules: Vec::new(),
            default_route: None,
            config: HashMap::new(),
        }
    }

    /// Add a routing rule
    pub fn add_rule(mut self, condition: impl Into<String>, target: impl Into<String>) -> Self {
        self.rules.push((condition.into(), target.into()));
        self
    }

    /// Set default route
    pub fn with_default(mut self, route: impl Into<String>) -> Self {
        self.default_route = Some(route.into());
        self
    }
}

/// Configuration for Retry nodes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetryNodeConfig {
    /// Name of the retry node
    pub name: String,

    /// Maximum number of retry attempts
    #[serde(default = "default_max_retries")]
    pub max_retries: usize,

    /// Initial backoff delay in milliseconds
    #[serde(default = "default_backoff_ms")]
    pub backoff_ms: u64,

    /// Exponential backoff multiplier
    #[serde(default = "default_backoff_multiplier")]
    pub backoff_multiplier: f32,

    /// Maximum backoff delay
    #[serde(default = "default_max_backoff_ms")]
    pub max_backoff_ms: u64,

    /// Condition for retryable errors
    #[serde(default)]
    pub retry_condition: Option<String>,

    /// Additional configuration
    #[serde(default)]
    pub config: HashMap<String, serde_json::Value>,
}

fn default_max_retries() -> usize {
    3
}

fn default_backoff_ms() -> u64 {
    1000
}

fn default_backoff_multiplier() -> f32 {
    2.0
}

fn default_max_backoff_ms() -> u64 {
    30000
}

impl RetryNodeConfig {
    /// Build a RetryNodeConfig from a NodeConfig (YAML deserialization target).
    /// Fields are read from `node.config` map.
    pub fn from_node_config(node: &crate::core::node::NodeConfig) -> crate::core::error::Result<Self> {
        let max_retries = node.config.get("max_retries")
            .and_then(|v| v.as_u64())
            .unwrap_or(3) as usize;
        let backoff_ms = node.config.get("backoff_ms")
            .and_then(|v| v.as_u64())
            .unwrap_or(1000);
        let backoff_multiplier = node.config.get("backoff_multiplier")
            .and_then(|v| v.as_f64())
            .unwrap_or(2.0) as f32;
        let max_backoff_ms = node.config.get("max_backoff_ms")
            .and_then(|v| v.as_u64())
            .unwrap_or(30000);
        let retry_condition = node.config.get("retry_condition")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());
        Ok(RetryNodeConfig {
            name: node.name.clone(),
            max_retries,
            backoff_ms,
            backoff_multiplier,
            max_backoff_ms,
            retry_condition,
            config: node.config.clone(),
        })
    }

    /// Create a new retry node configuration
    pub fn new(name: impl Into<String>) -> Self {
        RetryNodeConfig {
            name: name.into(),
            max_retries: 3,
            backoff_ms: 1000,
            backoff_multiplier: 2.0,
            max_backoff_ms: 30000,
            retry_condition: None,
            config: HashMap::new(),
        }
    }

    /// Set maximum retries
    pub fn with_max_retries(mut self, max: usize) -> Self {
        self.max_retries = max;
        self
    }

    /// Set initial backoff
    pub fn with_backoff_ms(mut self, backoff: u64) -> Self {
        self.backoff_ms = backoff;
        self
    }

    /// Set backoff multiplier
    pub fn with_backoff_multiplier(mut self, multiplier: f32) -> Self {
        self.backoff_multiplier = multiplier;
        self
    }

    /// Set maximum backoff
    pub fn with_max_backoff(mut self, max: u64) -> Self {
        self.max_backoff_ms = max;
        self
    }
}

/// Configuration for Timeout nodes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeoutNodeConfig {
    /// Name of the timeout node
    pub name: String,

    /// Timeout duration in milliseconds
    pub timeout_ms: u64,

    /// Action on timeout: "error", "skip", "default_value"
    #[serde(default = "default_timeout_action")]
    pub on_timeout: String,

    /// Default value if timeout action is "default_value"
    #[serde(default)]
    pub default_value: Option<serde_json::Value>,

    /// Additional configuration
    #[serde(default)]
    pub config: HashMap<String, serde_json::Value>,
}

fn default_timeout_action() -> String {
    "error".to_string()
}

impl TimeoutNodeConfig {
    /// Build a TimeoutNodeConfig from a NodeConfig (YAML deserialization target).
    /// Fields are read from `node.config` map.
    pub fn from_node_config(node: &crate::core::node::NodeConfig) -> crate::core::error::Result<Self> {
        let timeout_ms = node.config.get("timeout_ms")
            .and_then(|v| v.as_u64())
            .unwrap_or(30_000);
        let on_timeout = node.config.get("on_timeout")
            .and_then(|v| v.as_str())
            .unwrap_or("error")
            .to_string();
        let default_value = node.config.get("default_value").cloned();
        Ok(TimeoutNodeConfig {
            name: node.name.clone(),
            timeout_ms,
            on_timeout,
            default_value,
            config: node.config.clone(),
        })
    }

    /// Create a new timeout node configuration
    pub fn new(name: impl Into<String>, timeout_ms: u64) -> Self {
        TimeoutNodeConfig {
            name: name.into(),
            timeout_ms,
            on_timeout: "error".to_string(),
            default_value: None,
            config: HashMap::new(),
        }
    }

    /// Set timeout action
    pub fn with_on_timeout(mut self, action: impl Into<String>) -> Self {
        self.on_timeout = action.into();
        self
    }

    /// Set default value for timeout
    pub fn with_default_value(mut self, value: serde_json::Value) -> Self {
        self.default_value = Some(value);
        self
    }
}

/// Configuration for Human-in-the-Loop nodes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HumanInTheLoopConfig {
    /// Name of the HitL node
    pub name: String,

    /// Prompt to show the human
    pub prompt: String,

    /// Whether to require approval
    #[serde(default)]
    pub require_approval: bool,

    /// Fields that can be edited by the human
    #[serde(default)]
    pub editable_fields: Vec<String>,

    /// Timeout for human response in milliseconds
    #[serde(default)]
    pub timeout_ms: Option<u64>,

    /// Additional configuration
    #[serde(default)]
    pub config: HashMap<String, serde_json::Value>,
}

impl HumanInTheLoopConfig {
    /// Build a HumanInTheLoopConfig from a NodeConfig (YAML deserialization target).
    /// Fields are read from `node.config` map.
    pub fn from_node_config(node: &crate::core::node::NodeConfig) -> crate::core::error::Result<Self> {
        let prompt = node.config.get("prompt")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        let require_approval = node.config.get("require_approval")
            .and_then(|v| v.as_bool())
            .unwrap_or(true);
        let editable_fields: Vec<String> = node.config.get("editable_fields")
            .and_then(|v| v.as_array())
            .map(|arr| arr.iter().filter_map(|v| v.as_str().map(|s| s.to_string())).collect())
            .unwrap_or_default();
        let timeout_ms = node.config.get("timeout_ms")
            .and_then(|v| v.as_u64());
        Ok(HumanInTheLoopConfig {
            name: node.name.clone(),
            prompt,
            require_approval,
            editable_fields,
            timeout_ms,
            config: node.config.clone(),
        })
    }

    /// Create a new HitL node configuration
    pub fn new(name: impl Into<String>, prompt: impl Into<String>) -> Self {
        HumanInTheLoopConfig {
            name: name.into(),
            prompt: prompt.into(),
            require_approval: true,
            editable_fields: Vec::new(),
            timeout_ms: None,
            config: HashMap::new(),
        }
    }

    /// Set whether approval is required
    pub fn require_approval(mut self, require: bool) -> Self {
        self.require_approval = require;
        self
    }

    /// Add an editable field
    pub fn add_editable_field(mut self, field: impl Into<String>) -> Self {
        self.editable_fields.push(field.into());
        self
    }

    /// Set timeout
    pub fn with_timeout(mut self, timeout_ms: u64) -> Self {
        self.timeout_ms = Some(timeout_ms);
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_llm_node_config() {
        let config = LLMNodeConfig::new("analyzer", "Analyze: {data}")
            .with_temperature(0.5)
            .with_max_tokens(1000);

        assert_eq!(config.name, "analyzer");
        assert_eq!(config.temperature, 0.5);
        assert_eq!(config.max_tokens, Some(1000));
    }

    #[test]
    fn test_tool_node_config() {
        let config = ToolNodeConfig::new("searcher", "web_search", "search")
            .map_input("query", "q")
            .with_output_key("search_results");

        assert_eq!(config.tool_name, "web_search");
        assert_eq!(config.output_key, "search_results");
    }

    #[test]
    fn test_retry_node_config() {
        let config = RetryNodeConfig::new("retry")
            .with_max_retries(5)
            .with_backoff_ms(2000);

        assert_eq!(config.max_retries, 5);
        assert_eq!(config.backoff_ms, 2000);
    }

    #[test]
    fn test_node_output() {
        use crate::core::state::SharedState;
        let state = SharedState::new(Default::default());
        let output = NodeOutput::success(state).with_duration(Duration::from_millis(100));

        assert!(output.success);
        assert_eq!(output.execution_time_ms, 100);
    }
}
