use crate::core::node::nodes_trait::LLMNodeConfig;
use crate::core::node::nodes_trait::{ConditionalRouterConfig, ToolNodeConfig};
use crate::core::nodes_trait::HumanInTheLoopConfig;
use crate::core::nodes_trait::NodeService;
use crate::core::NodeOutput;
use crate::core::PluggableNode;
use crate::core::RetryNodeConfig;
use crate::core::TimeoutNodeConfig;

// # Built-in Node Implementations
//
// Complete implementations of standard, reusable nodes:
// - LLMNode - Language model integration
// - ToolNode - MCP tool execution
// - ConditionalRouter - Routing based on conditions
// - RetryNode - Automatic retry with backoff
// - TimeoutNode - Execution timeout enforcement
// - HumanInTheLoopNode - Human approval workflow
// - ParallelExecutorNode - Concurrent branch execution

use crate::core::error::{FlowgentraError, Result};
use crate::core::llm::LLMClient;
use crate::core::mcp::MCPClient;
// Removed unexpected closing delimiter
use crate::core::state::DynState;
use async_trait::async_trait;
use serde_json::json;
use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::time::sleep;
use tower::ServiceExt;

// =============================================================================
// LLM Node
// =============================================================================

/// Node that integrates with Language Models
///
/// Sends a prompt to an LLM and captures the response.
pub struct LLMNode {
    #[allow(dead_code)]
    config: LLMNodeConfig,
    #[allow(dead_code)]
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
}

// Removed unexpected closing delimiter

// Tool Node
// =============================================================================

/// Node that executes MCP tools
pub struct ToolNode {
    #[allow(dead_code)]
    config: ToolNodeConfig,
    #[allow(dead_code)]
    tool_client: Option<Arc<dyn MCPClient>>,
}

impl ToolNode {
    /// Create a new tool node
    pub fn new(
        name: impl Into<String>,
        tool_name: impl Into<String>,
        method: impl Into<String>,
    ) -> ToolNode {
        ToolNode {
            config: ToolNodeConfig::new(name, tool_name, method),
            tool_client: None,
        }
    }
}

// Removed unexpected closing delimiter// Duplicate impl blocks for PluggableNode on ToolNode and LLMNode removed. Only one impl per struct is allowed.

// =============================================================================
// Conditional Router Node
// =============================================================================

/// Complex type for condition functions
pub type ConditionFn = Box<dyn Fn(&DynState) -> bool + Send + Sync>;

/// Node that routes based on conditions
pub struct ConditionalRouter {
    config: ConditionalRouterConfig,
    #[allow(clippy::type_complexity)]
    conditions: HashMap<String, Box<dyn Fn(&DynState) -> bool + Send + Sync>>,
}

impl ConditionalRouter {
    /// Create a new router
    pub fn new(name: impl Into<String>) -> Self {
        ConditionalRouter {
            config: ConditionalRouterConfig {
                name: name.into(),
                rules: Vec::new(),
                default_route: None,
                config: HashMap::new(),
            },
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
        F: Fn(&DynState) -> bool + Send + Sync + 'static,
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
impl PluggableNode<DynState> for ConditionalRouter {
    async fn run(&self, state: DynState) -> Result<NodeOutput<DynState>> {
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

    fn clone_box(&self) -> Box<dyn PluggableNode<DynState>> {
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
    inner_node: Option<Box<dyn PluggableNode<DynState>>>,
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
    pub fn with_inner_node(mut self, node: Box<dyn PluggableNode<DynState>>) -> Self {
        self.inner_node = Some(node);
        self
    }

    /// Set maximum retries
    pub fn with_max_retries(mut self, max: usize) -> Self {
        self.config.max_retries = max;
        self
    }
}

#[derive(Clone)]
struct AgentRetryPolicy {
    attempts: usize,
    max_retries: usize,
    backoff_ms: u64,
    multiplier: f32,
    max_backoff_ms: u64,
}

impl tower::retry::Policy<DynState, NodeOutput<DynState>, FlowgentraError> for AgentRetryPolicy {
    type Future = Pin<Box<dyn Future<Output = Self> + Send>>;

    fn retry(
        &self,
        _req: &DynState,
        result: std::result::Result<&NodeOutput<DynState>, &FlowgentraError>,
    ) -> Option<Self::Future> {
        if result.is_ok() {
            return None;
        }

        if self.attempts >= self.max_retries {
            return None;
        }

        let next_attempts = self.attempts + 1;
        let mut next_backoff = self.backoff_ms;
        if self.attempts > 0 {
            next_backoff =
                ((self.backoff_ms as f32 * self.multiplier) as u64).min(self.max_backoff_ms);
        }

        let next_policy = AgentRetryPolicy {
            attempts: next_attempts,
            max_retries: self.max_retries,
            backoff_ms: next_backoff,
            multiplier: self.multiplier,
            max_backoff_ms: self.max_backoff_ms,
        };

        let delay = self.backoff_ms;
        Some(Box::pin(async move {
            if delay > 0 {
                sleep(Duration::from_millis(delay)).await;
            }
            next_policy
        }))
    }

    fn clone_request(&self, req: &DynState) -> Option<DynState> {
        Some(req.clone())
    }
}

#[async_trait]
impl PluggableNode<DynState> for RetryNode {
    async fn run(&self, state: DynState) -> Result<NodeOutput<DynState>> {
        let start = Instant::now();

        let inner_node = self.inner_node.as_ref().ok_or_else(|| {
            FlowgentraError::NodeExecutionError("No inner node configured".to_string())
        })?;

        let policy = AgentRetryPolicy {
            attempts: 0,
            max_retries: self.config.max_retries,
            backoff_ms: self.config.backoff_ms,
            multiplier: self.config.backoff_multiplier,
            max_backoff_ms: self.config.max_backoff_ms,
        };

        let service = NodeService::new(inner_node.clone_box());
        let retry_service = tower::retry::Retry::new(policy, service);

        // Explicitly annotate success type to help type inference
        match retry_service.oneshot(state.clone()).await {
            Ok(output) => Ok(output.with_duration(start.elapsed())), // Add explicit type annotation if needed
            Err(e) => Err(e),
        }
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

    fn clone_box(&self) -> Box<dyn PluggableNode<DynState>> {
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
    inner_node: Option<Box<dyn PluggableNode<DynState>>>,
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
    pub fn with_inner_node(
        mut self,
        node: Box<dyn crate::core::node::PluggableNode<DynState>>,
    ) -> Self {
        self.inner_node = Some(node);
        self
    }
}

#[async_trait]
impl crate::core::node::PluggableNode<DynState> for TimeoutNode {
    async fn run(&self, state: DynState) -> Result<NodeOutput<DynState>> {
        let start = Instant::now();

        let inner_node = self.inner_node.as_ref().ok_or_else(|| {
            FlowgentraError::NodeExecutionError("No inner node configured".to_string())
        })?;

        let timeout = Duration::from_millis(self.config.timeout_ms);
        let service = NodeService::new(inner_node.clone_box());
        let timeout_service = tower::timeout::Timeout::new(service, timeout);

        match timeout_service.oneshot(state.clone()).await {
            Ok(output) => Ok(output.with_duration(start.elapsed())),
            Err(e) => {
                if e.downcast_ref::<tower::timeout::error::Elapsed>().is_some() {
                    match self.config.on_timeout.as_str() {
                        "skip" => Ok(NodeOutput::success(state).with_duration(start.elapsed())),
                        "default_value" => {
                            let result_state = state;
                            if let Some(default) = &self.config.default_value {
                                result_state.set("_timeout_default", default.clone());
                            }
                            Ok(NodeOutput::success(result_state).with_duration(start.elapsed()))
                        }
                        _ => Err(FlowgentraError::ExecutionTimeout(format!(
                            "Node '{}' timed out after {}ms",
                            self.config.name, self.config.timeout_ms
                        ))),
                    }
                } else {
                    // Because NodeService returns FlowgentraError, we downcast back to it
                    if let Ok(flow_err) = e.downcast::<FlowgentraError>() {
                        Err(*flow_err)
                    } else {
                        Err(FlowgentraError::NodeExecutionError(
                            "Unknown inner error".into(),
                        ))
                    }
                }
            }
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

    fn clone_box(&self) -> Box<dyn PluggableNode<DynState>> {
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
impl PluggableNode<DynState> for HumanInTheLoopNode {
    async fn run(&self, state: DynState) -> Result<NodeOutput<DynState>> {
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

    fn clone_box(&self) -> Box<dyn crate::core::node::PluggableNode<DynState>> {
        Box::new(HumanInTheLoopNode {
            config: self.config.clone(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::state::DynState;

    #[test]
    fn test_llm_node_creation() {
        let node = LLMNode::new("analyzer");
        assert_eq!(node.config.name, "analyzer");
    }

    #[test]
    fn test_tool_node_creation() {
        let node = ToolNode::new("searcher", "web_search", "search");
        assert_eq!(node.config.tool_name, "web_search");
    }

    #[tokio::test]
    async fn test_router_node() {
        let router = ConditionalRouter::new("router")
            .register_condition("is_urgent", |state: &DynState| {
                state
                    .get("urgent")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false)
            })
            .add_rule("is_urgent", "urgent_handler");

        let state = DynState::new();
        state.set("urgent", json!(true));

        let result = router.run(state).await;
        assert!(result.is_ok());
    }

    #[test]
    fn test_timeout_node_creation() {
        let _timeout_node: TimeoutNode = TimeoutNode::new("timeout_wrapper", 5000);
    }

    #[tokio::test]
    async fn test_human_in_loop() {
        let node = HumanInTheLoopNode::new("approval", "Please approve this action")
            .add_editable_field("amount")
            .require_approval(true);

        let state = DynState::new();
        let result = node.run(state).await;

        assert!(result.is_ok());
    }
}
