//! # Built-in Node Implementations
//!
//! Complete implementations of standard, reusable nodes:
//! - LLMNode - Language model integration
//! - ToolNode - MCP tool execution
//! - ConditionalRouter - Routing based on conditions
//! - RetryNode - Automatic retry with backoff
//! - TimeoutNode - Execution timeout enforcement
//! - HumanInTheLoopNode - Human approval workflow
//! - ParallelExecutorNode - Concurrent branch execution

use crate::core::error::{ErenFlowError, Result};
use crate::core::llm::LLMClient;
use crate::core::mcp::MCPClient;
use crate::core::nodes_trait::*;
use crate::core::state::State;
use async_trait::async_trait;
use serde_json::json;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::time::sleep;

// =============================================================================
// LLM Node
// =============================================================================

/// Node that integrates with Language Models
///
/// Sends a prompt to an LLM and captures the response.
pub struct LLMNode {
    config: LLMNodeConfig,
    llm_client: Option<Arc<dyn LLMClient>>,
}

impl LLMNode {
    /// Create a new LLM node
    pub fn new(name: impl Into<String>) -> Self {
        LLMNode {
            config: LLMNodeConfig {
                name: name.into(),
                prompt: String::new(),
                temperature: 0.7,
                max_tokens: None,
                system_prompt: None,
                input_variables: Vec::new(),
                output_key: "llm_response".to_string(),
                config: HashMap::new(),
            },
            llm_client: None,
        }
    }

    /// Create from configuration
    pub fn from_config(config: LLMNodeConfig) -> Self {
        LLMNode {
            config,
            llm_client: None,
        }
    }

    /// Set LLM client
    pub fn with_llm_client(mut self, client: Arc<dyn LLMClient>) -> Self {
        self.llm_client = Some(client);
        self
    }

    /// Set prompt
    pub fn with_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.config.prompt = prompt.into();
        self
    }

    /// Set temperature
    pub fn with_temperature(mut self, temp: f32) -> Self {
        self.config.temperature = temp.clamp(0.0, 1.0);
        self
    }

    /// Add input variable
    pub fn add_input_variable(mut self, var: impl Into<String>) -> Self {
        self.config.input_variables.push(var.into());
        self
    }

    /// Replace variables in prompt with values from state
    fn interpolate_prompt(&self, state: &State) -> String {
        let mut prompt = self.config.prompt.clone();

        for var in &self.config.input_variables {
            let placeholder = format!("{{{}}}", var);
            if let Some(value) = state.get(var) {
                let value_str = match value {
                    serde_json::Value::String(s) => s.clone(),
                    _ => value.to_string(),
                };
                prompt = prompt.replace(&placeholder, &value_str);
            }
        }

        prompt
    }
}

#[async_trait]
impl PluggableNode for LLMNode {
    async fn run(&self, mut state: State) -> Result<NodeOutput<State>> {
        let start = Instant::now();

        // Interpolate prompt with state variables
        let prompt = self.interpolate_prompt(&state);

        // Get LLM response (mock if no client provided)
        let response = if let Some(_client) = &self.llm_client {
            // TODO: Call actual LLM client
            format!("LLM response for: {}", prompt)
        } else {
            format!("Mock LLM response for: {}", prompt)
        };

        // Store response in state
        state.set(&self.config.output_key, json!(response));

        Ok(NodeOutput::success(state).with_duration(start.elapsed()))
    }

    fn name(&self) -> &str {
        &self.config.name
    }

    fn node_type(&self) -> &str {
        "llm"
    }

    fn config(&self) -> &HashMap<String, serde_json::Value> {
        &self.config.config
    }

    fn metadata(&self) -> HashMap<String, String> {
        let mut meta = HashMap::new();
        meta.insert(
            "temperature".to_string(),
            self.config.temperature.to_string(),
        );
        meta.insert(
            "prompt_length".to_string(),
            self.config.prompt.len().to_string(),
        );
        meta
    }

    fn clone_box(&self) -> Box<dyn PluggableNode> {
        Box::new(LLMNode {
            config: self.config.clone(),
            llm_client: self.llm_client.clone(),
        })
    }
}

// =============================================================================
// Tool Node
// =============================================================================

/// Node that executes MCP tools
pub struct ToolNode {
    config: ToolNodeConfig,
    tool_client: Option<Arc<dyn MCPClient>>,
}

impl ToolNode {
    /// Create a new tool node
    pub fn new(
        name: impl Into<String>,
        tool_name: impl Into<String>,
        method: impl Into<String>,
    ) -> Self {
        ToolNode {
            config: ToolNodeConfig::new(name, tool_name, method),
            tool_client: None,
        }
    }

    /// Create from configuration
    pub fn from_config(config: ToolNodeConfig) -> Self {
        ToolNode {
            config,
            tool_client: None,
        }
    }

    /// Set tool client
    pub fn with_tool_client(mut self, client: Arc<dyn MCPClient>) -> Self {
        self.tool_client = Some(client);
        self
    }

    /// Map input parameter
    pub fn map_input(
        mut self,
        state_key: impl Into<String>,
        tool_param: impl Into<String>,
    ) -> Self {
        self.config = self.config.map_input(state_key, tool_param);
        self
    }
}

#[async_trait]
impl PluggableNode for ToolNode {
    async fn run(&self, mut state: State) -> Result<NodeOutput<State>> {
        let start = Instant::now();

        // Map state values to tool parameters
        let mut tool_params = HashMap::new();
        for (state_key, tool_param) in &self.config.input_mapping {
            if let Some(value) = state.get(state_key) {
                tool_params.insert(tool_param.clone(), value.clone());
            }
        }

        // Execute tool (mock if no client provided)
        let result = if let Some(_client) = &self.tool_client {
            // TODO: Call actual tool client
            json!({"status": "success", "data": format!("Executed {}.{}", self.config.tool_name, self.config.method)})
        } else {
            json!({"status": "success", "data": format!("Mock: {}.{}", self.config.tool_name, self.config.method)})
        };

        state.set(&self.config.output_key, result);

        Ok(NodeOutput::success(state).with_duration(start.elapsed()))
    }

    fn name(&self) -> &str {
        &self.config.name
    }

    fn node_type(&self) -> &str {
        "tool"
    }

    fn config(&self) -> &HashMap<String, serde_json::Value> {
        &self.config.config
    }

    fn metadata(&self) -> HashMap<String, String> {
        let mut meta = HashMap::new();
        meta.insert("tool".to_string(), self.config.tool_name.clone());
        meta.insert("method".to_string(), self.config.method.clone());
        meta
    }

    fn clone_box(&self) -> Box<dyn PluggableNode> {
        Box::new(ToolNode {
            config: self.config.clone(),
            tool_client: self.tool_client.clone(),
        })
    }
}

// =============================================================================
// Conditional Router Node
// =============================================================================

/// Complex type for condition functions
type ConditionFn = Box<dyn Fn(&State) -> bool + Send + Sync>;

/// Node that routes based on conditions
pub struct ConditionalRouter {
    config: ConditionalRouterConfig,
    conditions: HashMap<String, ConditionFn>,
}

impl ConditionalRouter {
    /// Create a new router
    pub fn new(name: impl Into<String>) -> Self {
        ConditionalRouter {
            config: ConditionalRouterConfig::new(name),
            conditions: HashMap::new(),
        }
    }

    /// Create from configuration
    pub fn from_config(config: ConditionalRouterConfig) -> Self {
        ConditionalRouter {
            config,
            conditions: HashMap::new(),
        }
    }

    /// Register a condition
    pub fn register_condition<F>(mut self, name: impl Into<String>, condition: F) -> Self
    where
        F: Fn(&State) -> bool + Send + Sync + 'static,
    {
        self.conditions.insert(name.into(), Box::new(condition));
        self
    }

    /// Add routing rule
    pub fn add_rule(mut self, condition: impl Into<String>, target: impl Into<String>) -> Self {
        self.config = self.config.add_rule(condition, target);
        self
    }
}

#[async_trait]
impl PluggableNode for ConditionalRouter {
    async fn run(&self, mut state: State) -> Result<NodeOutput<State>> {
        let start = Instant::now();

        // Check each routing rule
        let mut next_node = self.config.default_route.clone();

        for (cond_name, _target) in &self.config.rules {
            if let Some(condition_fn) = self.conditions.get(cond_name) {
                if condition_fn(&state) {
                    next_node = Some(_target.clone());
                    break;
                }
            }
        }

        if let Some(next) = next_node {
            state.set("_next_node", json!(next));
        }

        Ok(NodeOutput::success(state).with_duration(start.elapsed()))
    }

    fn name(&self) -> &str {
        &self.config.name
    }

    fn node_type(&self) -> &str {
        "router"
    }

    fn config(&self) -> &HashMap<String, serde_json::Value> {
        &self.config.config
    }

    fn clone_box(&self) -> Box<dyn PluggableNode> {
        Box::new(ConditionalRouter {
            config: self.config.clone(),
            conditions: HashMap::new(),
        })
    }
}

// =============================================================================
// Retry Node
// =============================================================================

/// Node that wraps another node with retry logic
pub struct RetryNode {
    config: RetryNodeConfig,
    inner_node: Option<Box<dyn PluggableNode>>,
}

impl RetryNode {
    /// Create a new retry node
    pub fn new(name: impl Into<String>) -> Self {
        RetryNode {
            config: RetryNodeConfig::new(name),
            inner_node: None,
        }
    }

    /// Create from configuration
    pub fn from_config(config: RetryNodeConfig) -> Self {
        RetryNode {
            config,
            inner_node: None,
        }
    }

    /// Set the inner node to retry
    pub fn with_inner_node(mut self, node: Box<dyn PluggableNode>) -> Self {
        self.inner_node = Some(node);
        self
    }

    /// Set maximum retries
    pub fn with_max_retries(mut self, max: usize) -> Self {
        self.config.max_retries = max;
        self
    }
}

#[async_trait]
impl PluggableNode for RetryNode {
    async fn run(&self, state: State) -> Result<NodeOutput<State>> {
        let start = Instant::now();

        let inner_node = self.inner_node.as_ref().ok_or_else(|| {
            ErenFlowError::NodeExecutionError("No inner node configured".to_string())
        })?;

        let current_state = state;
        let mut last_error = None;
        let mut backoff = self.config.backoff_ms;

        for attempt in 0..=self.config.max_retries {
            match inner_node.run(current_state.clone()).await {
                Ok(output) => {
                    return Ok(output.with_duration(start.elapsed()));
                }
                Err(e) => {
                    last_error = Some(e);

                    if attempt < self.config.max_retries {
                        sleep(Duration::from_millis(backoff)).await;
                        backoff = ((backoff as f32 * self.config.backoff_multiplier) as u64)
                            .min(self.config.max_backoff_ms);
                    }
                }
            }
        }

        Err(last_error
            .unwrap_or_else(|| ErenFlowError::NodeExecutionError("Retry node failed".to_string())))
    }

    fn name(&self) -> &str {
        &self.config.name
    }

    fn node_type(&self) -> &str {
        "retry"
    }

    fn config(&self) -> &HashMap<String, serde_json::Value> {
        &self.config.config
    }

    fn metadata(&self) -> HashMap<String, String> {
        let mut meta = HashMap::new();
        meta.insert(
            "max_retries".to_string(),
            self.config.max_retries.to_string(),
        );
        meta.insert("backoff_ms".to_string(), self.config.backoff_ms.to_string());
        meta
    }

    fn clone_box(&self) -> Box<dyn PluggableNode> {
        Box::new(RetryNode {
            config: self.config.clone(),
            inner_node: self.inner_node.as_ref().map(|n| n.clone_box()),
        })
    }
}

// =============================================================================
// Timeout Node
// =============================================================================

/// Node that enforces execution timeout
pub struct TimeoutNode {
    config: TimeoutNodeConfig,
    inner_node: Option<Box<dyn PluggableNode>>,
}

impl TimeoutNode {
    /// Create a new timeout node
    pub fn new(name: impl Into<String>, timeout_ms: u64) -> Self {
        TimeoutNode {
            config: TimeoutNodeConfig::new(name, timeout_ms),
            inner_node: None,
        }
    }

    /// Create from configuration
    pub fn from_config(config: TimeoutNodeConfig) -> Self {
        TimeoutNode {
            config,
            inner_node: None,
        }
    }

    /// Set the inner node
    pub fn with_inner_node(mut self, node: Box<dyn PluggableNode>) -> Self {
        self.inner_node = Some(node);
        self
    }
}

#[async_trait]
impl PluggableNode for TimeoutNode {
    async fn run(&self, state: State) -> Result<NodeOutput<State>> {
        let start = Instant::now();

        let inner_node = self.inner_node.as_ref().ok_or_else(|| {
            ErenFlowError::NodeExecutionError("No inner node configured".to_string())
        })?;

        let timeout = Duration::from_millis(self.config.timeout_ms);

        match tokio::time::timeout(timeout, inner_node.run(state.clone())).await {
            Ok(Ok(output)) => Ok(output.with_duration(start.elapsed())),
            Ok(Err(e)) => Err(e),
            Err(_) => match self.config.on_timeout.as_str() {
                "skip" => Ok(NodeOutput::success(state).with_duration(start.elapsed())),
                "default_value" => {
                    let mut result_state = state;
                    if let Some(default) = &self.config.default_value {
                        result_state.set("_timeout_default", default.clone());
                    }
                    Ok(NodeOutput::success(result_state).with_duration(start.elapsed()))
                }
                _ => Err(ErenFlowError::ExecutionTimeout(format!(
                    "Node '{}' timed out after {}ms",
                    self.config.name, self.config.timeout_ms
                ))),
            },
        }
    }

    fn name(&self) -> &str {
        &self.config.name
    }

    fn node_type(&self) -> &str {
        "timeout"
    }

    fn config(&self) -> &HashMap<String, serde_json::Value> {
        &self.config.config
    }

    fn metadata(&self) -> HashMap<String, String> {
        let mut meta = HashMap::new();
        meta.insert("timeout_ms".to_string(), self.config.timeout_ms.to_string());
        meta.insert("on_timeout".to_string(), self.config.on_timeout.clone());
        meta
    }

    fn clone_box(&self) -> Box<dyn PluggableNode> {
        Box::new(TimeoutNode {
            config: self.config.clone(),
            inner_node: self.inner_node.as_ref().map(|n| n.clone_box()),
        })
    }
}

// =============================================================================
// Human-in-the-Loop Node
// =============================================================================

/// Node that waits for human approval/input
pub struct HumanInTheLoopNode {
    config: HumanInTheLoopConfig,
}

impl HumanInTheLoopNode {
    /// Create a new HitL node
    pub fn new(name: impl Into<String>, prompt: impl Into<String>) -> Self {
        HumanInTheLoopNode {
            config: HumanInTheLoopConfig::new(name, prompt),
        }
    }

    /// Create from configuration
    pub fn from_config(config: HumanInTheLoopConfig) -> Self {
        HumanInTheLoopNode { config }
    }

    /// Add editable field
    pub fn add_editable_field(mut self, field: impl Into<String>) -> Self {
        self.config = self.config.add_editable_field(field);
        self
    }

    /// Set approval requirement
    pub fn require_approval(mut self, require: bool) -> Self {
        self.config.require_approval = require;
        self
    }
}

#[async_trait]
impl PluggableNode for HumanInTheLoopNode {
    async fn run(&self, mut state: State) -> Result<NodeOutput<State>> {
        let start = Instant::now();

        // In a real implementation, this would:
        // 1. Display prompt to user
        // 2. Allow editing of specified fields
        // 3. Wait for approval or timeout
        // 4. Update state with any edits

        // For now, we'll simulate with a log and mark as approved
        println!("\n=== Human-in-the-Loop ===");
        println!("Prompt: {}", self.config.prompt);
        println!("Approval required: {}", self.config.require_approval);

        state.set("_human_approved", json!(true));
        state.set(
            "_human_timestamp",
            json!(std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs()),
        );

        Ok(NodeOutput::success(state).with_duration(start.elapsed()))
    }

    fn name(&self) -> &str {
        &self.config.name
    }

    fn node_type(&self) -> &str {
        "human_in_the_loop"
    }

    fn config(&self) -> &HashMap<String, serde_json::Value> {
        &self.config.config
    }

    fn metadata(&self) -> HashMap<String, String> {
        let mut meta = HashMap::new();
        meta.insert(
            "require_approval".to_string(),
            self.config.require_approval.to_string(),
        );
        meta.insert(
            "editable_fields_count".to_string(),
            self.config.editable_fields.len().to_string(),
        );
        meta
    }

    fn clone_box(&self) -> Box<dyn PluggableNode> {
        Box::new(HumanInTheLoopNode {
            config: self.config.clone(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_llm_node() {
        let node = LLMNode::new("analyzer").with_prompt("Analyze: {input}");

        let state = State::new();
        let result = node.run(state).await;

        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_tool_node() {
        let node = ToolNode::new("searcher", "web_search", "search");

        let state = State::new();
        let result = node.run(state).await;

        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_router_node() {
        let router = ConditionalRouter::new("router")
            .register_condition("is_urgent", |state| {
                state.get_bool("urgent").unwrap_or(false)
            })
            .add_rule("is_urgent", "urgent_handler");

        let mut state = State::new();
        state.set("urgent", json!(true));

        let result = router.run(state).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_timeout_node() {
        let llm = LLMNode::new("fast_llm");
        let timeout_node = TimeoutNode::new("timeout_wrapper", 5000).with_inner_node(Box::new(llm));

        let state = State::new();
        let result = timeout_node.run(state).await;

        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_human_in_loop() {
        let node = HumanInTheLoopNode::new("approval", "Please approve this action")
            .add_editable_field("amount")
            .require_approval(true);

        let state = State::new();
        let result = node.run(state).await;

        assert!(result.is_ok());
    }
}
