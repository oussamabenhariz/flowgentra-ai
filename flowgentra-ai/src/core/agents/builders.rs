use super::graph_nodes::ToolExecutorFn;
/// Agent Builder Factory
///
/// Provides builder patterns for creating predefined agents with sensible defaults.
/// Similar to LangChain's `initialize_agent` but tailored for FlowgentraAI.
use super::{
    reasoning_router, Agent, AgentReasoningNode, AgentType, ConversationalNode, ToolExecutorNode,
    ToolSpec,
};
use crate::core::error::FlowgentraError;
use crate::core::llm::LLMConfig;
use crate::core::mcp::MCPConfig;
use crate::core::state::context::Context;
use crate::core::state::{DynState, DynStateUpdate};
use crate::core::state_graph::{FunctionNode, StateGraph};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tracing::debug;

/// Agent configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrebuiltAgentConfig {
    /// Agent name
    pub name: String,

    /// Agent type
    pub agent_type: String,

    /// LLM model identifier (used when `llm` is None)
    pub llm_model: String,

    /// Full LLM configuration (takes precedence over `llm_model` / `api_key` / `temperature` / `max_tokens`).
    #[serde(skip)]
    pub llm: Option<LLMConfig>,

    /// API key for the LLM provider. Required for cloud providers (OpenAI, Anthropic, etc.).
    /// Leave as `None` for local providers that don't need authentication (e.g. Ollama).
    pub api_key: Option<String>,

    /// System prompt/instructions
    pub system_prompt: String,

    /// Tools available to agent
    pub tools: HashMap<String, ToolSpec>,

    /// MCP (Model Context Protocol) configurations
    pub mcps: Vec<MCPConfig>,

    /// Temperature (0.0 - 1.0) for LLM sampling
    pub temperature: f32,

    /// Maximum tokens in response
    pub max_tokens: usize,

    /// Memory configuration
    pub memory_enabled: bool,
    pub memory_steps: usize,

    /// Evaluation enabled
    pub evaluation_enabled: bool,

    /// Retry configuration
    pub max_retries: usize,
    pub retry_on_failure: bool,

    /// Custom parameters
    pub custom_params: HashMap<String, String>,
}

impl Default for PrebuiltAgentConfig {
    fn default() -> Self {
        Self {
            name: "agent".to_string(),
            agent_type: "default".to_string(),
            llm_model: "gpt-4".to_string(),
            llm: None,
            api_key: None,
            system_prompt: String::new(),
            tools: HashMap::new(),
            mcps: Vec::new(),
            temperature: 0.7,
            max_tokens: 2000,
            memory_enabled: false,
            memory_steps: 5,
            evaluation_enabled: false,
            max_retries: 3,
            retry_on_failure: true,
            custom_params: HashMap::new(),
        }
    }
}

impl PrebuiltAgentConfig {
    /// Create new agent configuration
    pub fn new(name: impl Into<String>, agent_type: AgentType) -> Self {
        Self {
            name: name.into(),
            agent_type: agent_type.to_string(),
            system_prompt: super::SystemPrompts::get_default(agent_type),
            ..Default::default()
        }
    }
}

/// Graph-based agent that wraps StateGraph for predefined agent types
pub struct GraphBasedAgent {
    config: PrebuiltAgentConfig,
    name: String,
    graph: StateGraph<DynState>,
    /// Conversation history for multi-turn context: Vec<(role, content)>
    conversation_history: std::sync::Mutex<Vec<(String, String)>>,
}

impl GraphBasedAgent {
    /// Create a new graph-based agent
    pub fn new(
        config: PrebuiltAgentConfig,
        tool_executor: Option<ToolExecutorFn>,
    ) -> Result<Self, FlowgentraError> {
        let name = config.name.clone();
        let agent_type = AgentType::from_type_str(&config.agent_type);

        // Build appropriate graph based on agent type
        let graph = match agent_type {
            AgentType::ZeroShotReAct => Self::build_zero_shot_react_graph(&config, tool_executor)?,
            AgentType::FewShotReAct => Self::build_few_shot_react_graph(&config, tool_executor)?,
            AgentType::Conversational => Self::build_conversational_graph(&config)?,
        };

        Ok(Self {
            config,
            name,
            graph,
            conversation_history: std::sync::Mutex::new(Vec::new()),
        })
    }

    /// Build a ZeroShotReAct agent graph
    fn build_zero_shot_react_graph(
        config: &PrebuiltAgentConfig,
        tool_executor: Option<ToolExecutorFn>,
    ) -> Result<StateGraph<DynState>, FlowgentraError> {
        let agent_config = config.clone();
        let tool_config = config.clone();
        let tool_executor = tool_executor.unwrap_or_else(|| Arc::new(|name: &str, _args: &str| {
            format!("Tool '{}' has no executor registered. Use .with_tool_executor() on the AgentBuilder.", name)
        }));

        // Create agent reasoning node (wraps the actual node logic)
        let agent_node = Arc::new(FunctionNode::new(
            "agent",
            move |state: &DynState, _ctx: &Context| {
                let config = agent_config.clone();
                let state = state.clone();
                Box::pin(async move {
                    let reasoning_node = AgentReasoningNode::new(config);
                    reasoning_node.execute(&state).await.map_err(|e| {
                        crate::core::state_graph::StateGraphError::ExecutionError {
                            node: "agent".to_string(),
                            reason: e.to_string(),
                        }
                    })?;
                    Ok(DynStateUpdate::new())
                })
            },
        ));

        // Create tool executor node with the user-provided executor
        let tool_exec_fn = tool_executor.clone();
        let tool_executor_node = Arc::new(FunctionNode::new(
            "tool_executor",
            move |state: &DynState, _ctx: &Context| {
                let config = tool_config.clone();
                let executor = tool_exec_fn.clone();
                let state = state.clone();
                Box::pin(async move {
                    let tool_node = ToolExecutorNode::new(config).with_executor(executor);
                    tool_node.execute(&state).await.map_err(|e| {
                        crate::core::state_graph::StateGraphError::ExecutionError {
                            node: "tool_executor".to_string(),
                            reason: e.to_string(),
                        }
                    })?;
                    Ok(DynStateUpdate::new())
                })
            },
        ));

        // Create end node (terminal node that just returns state)
        let end_node = Arc::new(FunctionNode::new(
            "END",
            |_state: &DynState, _ctx: &Context| Box::pin(async move { Ok(DynStateUpdate::new()) }),
        ));

        // Build the graph with routing
        let graph = StateGraph::<DynState>::builder()
            .add_node("agent", agent_node)
            .add_node("tool_executor", tool_executor_node)
            .add_node("END", end_node)
            .set_entry_point("agent")
            // Agent decides: use tools or finish
            .add_conditional_edge(
                "agent",
                Box::new(|state: &DynState| {
                    reasoning_router(state).map_err(|e| {
                        crate::core::state_graph::StateGraphError::ExecutionError {
                            node: "agent".to_string(),
                            reason: e.to_string(),
                        }
                    })
                }),
            )
            // Tools always go back to agent for next iteration
            .add_edge("tool_executor", "agent")
            .compile()
            .map_err(|e| {
                FlowgentraError::GraphError(format!("ZeroShotReAct graph build failed: {}", e))
            })?;

        debug!("Built ZeroShotReAct graph");
        Ok(graph)
    }

    /// Build a FewShotReAct agent graph (similar to ZeroShotReAct but with examples)
    fn build_few_shot_react_graph(
        config: &PrebuiltAgentConfig,
        tool_executor: Option<ToolExecutorFn>,
    ) -> Result<StateGraph<DynState>, FlowgentraError> {
        // For now, use same structure as ZeroShotReAct
        // In real implementation, would include example demonstrations
        Self::build_zero_shot_react_graph(config, tool_executor)
    }

    /// Build a Conversational agent graph
    fn build_conversational_graph(
        config: &PrebuiltAgentConfig,
    ) -> Result<StateGraph<DynState>, FlowgentraError> {
        let config_clone = config.clone();

        // Single node for conversational response
        let conversation_node = Arc::new(FunctionNode::new(
            "conversation",
            move |state: &DynState, _ctx: &Context| {
                let config = config_clone.clone();
                let state = state.clone();
                Box::pin(async move {
                    let conv_node = ConversationalNode::new(config);
                    conv_node.execute(&state).await.map_err(|e| {
                        crate::core::state_graph::StateGraphError::ExecutionError {
                            node: "conversation".to_string(),
                            reason: e.to_string(),
                        }
                    })?;
                    Ok(DynStateUpdate::new())
                })
            },
        ));

        let graph = StateGraph::<DynState>::builder()
            .add_node("conversation", conversation_node)
            .set_entry_point("conversation")
            .compile()
            .map_err(|e| {
                FlowgentraError::GraphError(format!("Conversational graph build failed: {}", e))
            })?;

        debug!("Built Conversational graph");
        Ok(graph)
    }

    /// Execute the agent with given input
    pub async fn execute_input(&self, input: &str) -> Result<String, FlowgentraError> {
        // Create initial state
        let initial_state = DynState::default();
        initial_state.set("input", serde_json::json!(input));

        // Inject conversation history into state for multi-turn context
        {
            let history = self.conversation_history.lock().map_err(|_| {
                FlowgentraError::StateError("Failed to lock conversation history".to_string())
            })?;
            if !history.is_empty() {
                let history_json: Vec<serde_json::Value> = history
                    .iter()
                    .map(|(role, content)| serde_json::json!({ "role": role, "content": content }))
                    .collect();
                initial_state.set("conversation_history", serde_json::json!(history_json));
            }
        }

        // Execute the graph - StateGraph handles routing, loops, checkpointing
        let final_state =
            self.graph.invoke(initial_state).await.map_err(|e| {
                FlowgentraError::StateError(format!("Graph execution failed: {}", e))
            })?;

        // Extract response based on agent type
        let response = match AgentType::from_type_str(&self.config.agent_type) {
            AgentType::ZeroShotReAct | AgentType::FewShotReAct => {
                let full_response = final_state
                    .get("llm_response")
                    .and_then(|v| v.as_str().map(|s| s.to_string()))
                    .unwrap_or_else(|| "No response generated".to_string());

                // Try to extract content from <answer> tags
                if let Some(start) = full_response.find("<answer>") {
                    if let Some(end) = full_response.find("</answer>") {
                        let start_idx = start + 8; // Length of "<answer>"
                        if start_idx < end {
                            return Ok(full_response[start_idx..end].trim().to_string());
                        }
                    }
                }

                // If no tags, return the response as-is (remove thinking blocks)
                if let Some(thinking_end) = full_response.find("</thinking>") {
                    full_response[thinking_end + 11..].trim().to_string()
                } else {
                    full_response
                }
            }
            AgentType::Conversational => final_state
                .get("response")
                .and_then(|v| v.as_str().map(|s| s.to_string()))
                .unwrap_or_else(|| "No response generated".to_string()),
        };

        // Append to conversation history for future turns
        {
            let mut history = self.conversation_history.lock().map_err(|_| {
                FlowgentraError::StateError("Failed to lock conversation history".to_string())
            })?;
            history.push(("user".to_string(), input.to_string()));
            history.push(("assistant".to_string(), response.clone()));
        }

        Ok(response)
    }

    /// Get reference to the underlying StateGraph for advanced usage
    pub fn graph(&self) -> &StateGraph<DynState> {
        &self.graph
    }

    /// Get config reference
    pub fn config(&self) -> &PrebuiltAgentConfig {
        &self.config
    }

    /// Get tools from config
    pub fn tools(&self) -> Vec<&ToolSpec> {
        self.config.tools.values().collect()
    }
}

impl Agent for GraphBasedAgent {
    fn name(&self) -> &str {
        &self.name
    }

    fn agent_type(&self) -> AgentType {
        AgentType::from_type_str(&self.config.agent_type)
    }

    fn initialize(&mut self, _state: &mut DynState) -> Result<(), FlowgentraError> {
        // Initialize any resources needed
        if self.config.memory_enabled {
            debug!(
                "Agent memory enabled with {} steps",
                self.config.memory_steps
            );
        }
        Ok(())
    }

    fn process(&self, _input: &str, _state: &DynState) -> Result<String, FlowgentraError> {
        // Synchronous wrapper - use execute_input() for async execution
        Err(FlowgentraError::StateError(
            "Use execute_input() for async execution".to_string(),
        ))
    }

    fn config(&self) -> &PrebuiltAgentConfig {
        &self.config
    }

    fn add_tool(&mut self, tool_name: &str, tool_spec: ToolSpec) -> Result<(), FlowgentraError> {
        self.config.tools.insert(tool_name.to_string(), tool_spec);
        debug!("Added tool: {}", tool_name);
        Ok(())
    }

    fn tools(&self) -> Vec<&ToolSpec> {
        self.config.tools.values().collect()
    }
}

/// Agent builder for fluent configuration
pub struct AgentBuilder {
    config: PrebuiltAgentConfig,
    tool_executor: Option<ToolExecutorFn>,
}

impl AgentBuilder {
    /// Create new agent builder
    pub fn new(agent_type: AgentType) -> Self {
        Self {
            config: PrebuiltAgentConfig::new("agent", agent_type),
            tool_executor: None,
        }
    }

    /// Set agent name
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.config.name = name.into();
        self
    }

    /// Set a full LLM configuration object (provider, model, api_key, temperature, etc.).
    /// Takes precedence over `with_llm_config` when both are set.
    pub fn with_llm(mut self, llm: LLMConfig) -> Self {
        self.config.llm = Some(llm);
        self
    }

    /// Set LLM model by name (string). Use `with_llm` to pass a full `LLMConfig` object.
    pub fn with_llm_config(mut self, model: impl Into<String>) -> Self {
        self.config.llm_model = model.into();
        self
    }

    /// Set temperature
    pub fn with_temperature(mut self, temperature: f32) -> Self {
        self.config.temperature = temperature.clamp(0.0, 1.0);
        self
    }

    /// Set max tokens
    pub fn with_max_tokens(mut self, tokens: usize) -> Self {
        self.config.max_tokens = tokens;
        self
    }

    /// Add tool to agent
    pub fn with_tool(mut self, tool: ToolSpec) -> Self {
        self.config.tools.insert(tool.name.clone(), tool);
        self
    }

    /// Add multiple tools
    pub fn with_tools(mut self, tools: Vec<ToolSpec>) -> Self {
        for tool in tools {
            self.config.tools.insert(tool.name.clone(), tool);
        }
        self
    }

    /// Add MCP configuration
    pub fn with_mcp(mut self, mcp_config: MCPConfig) -> Self {
        self.config.mcps.push(mcp_config);
        self
    }

    /// Add multiple MCPs
    pub fn with_mcps(mut self, mcp_configs: Vec<MCPConfig>) -> Self {
        self.config.mcps.extend(mcp_configs);
        self
    }

    /// Set system prompt
    pub fn with_system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.config.system_prompt = prompt.into();
        self
    }

    /// Enable memory
    pub fn with_memory_steps(mut self, steps: usize) -> Self {
        self.config.memory_enabled = true;
        self.config.memory_steps = steps;
        self
    }

    /// Disable memory
    pub fn without_memory(mut self) -> Self {
        self.config.memory_enabled = false;
        self
    }

    /// Enable evaluation
    pub fn with_evaluation(mut self) -> Self {
        self.config.evaluation_enabled = true;
        self
    }

    /// Set retry policy
    pub fn with_retries(mut self, max_retries: usize) -> Self {
        self.config.max_retries = max_retries;
        self.config.retry_on_failure = true;
        self
    }

    /// Add custom parameter
    pub fn with_param(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.config.custom_params.insert(key.into(), value.into());
        self
    }

    /// Get configuration
    pub fn config(&self) -> &PrebuiltAgentConfig {
        &self.config
    }

    /// Set the tool executor function.
    ///
    /// The tool executor receives `(tool_name, arguments)` and returns the result string.
    /// This is how user-defined Rust functions get called when the LLM decides to use a tool.
    ///
    /// # Example
    /// ```ignore
    /// let agent = AgentBuilder::new(AgentType::ZeroShotReAct)
    ///     .with_tool_executor(|name, args| match name {
    ///         "calculator" => format!("Result: {}", eval(args)),
    ///         _ => format!("Unknown tool: {}", name),
    ///     })
    ///     .build_graph()?;
    /// ```
    pub fn with_tool_executor<F>(mut self, executor: F) -> Self
    where
        F: Fn(&str, &str) -> String + Send + Sync + 'static,
    {
        self.tool_executor = Some(Arc::new(executor));
        self
    }

    /// Build agent - returns a boxed Agent trait object
    /// This now creates a GraphBasedAgent which wraps StateGraph internally
    pub fn build(self) -> Result<Box<dyn Agent>, FlowgentraError> {
        // Create the graph-based agent (which builds StateGraph internally)
        let agent = GraphBasedAgent::new(self.config, self.tool_executor)?;
        Ok(Box::new(agent))
    }

    /// Build agent - returns the concrete GraphBasedAgent for async execution
    /// Use this when you need to call execute_input() for async LLM calls
    pub fn build_graph(self) -> Result<GraphBasedAgent, FlowgentraError> {
        GraphBasedAgent::new(self.config, self.tool_executor)
    }
}

impl Default for AgentBuilder {
    fn default() -> Self {
        Self::new(AgentType::ZeroShotReAct)
    }
}

impl AgentType {
    /// Parse string as agent type
    pub fn from_type_str(s: &str) -> Self {
        match s {
            "zero-shot-react" => AgentType::ZeroShotReAct,
            "few-shot-react" => AgentType::FewShotReAct,
            "conversational" => AgentType::Conversational,
            _ => AgentType::ZeroShotReAct,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_agent_builder_fluent_api() {
        let builder = AgentBuilder::new(AgentType::ZeroShotReAct)
            .with_name("my_agent")
            .with_llm_config("gpt-4-turbo")
            .with_temperature(0.5)
            .with_max_tokens(4000)
            .with_memory_steps(10)
            .with_evaluation();

        let config = builder.config();
        assert_eq!(config.name, "my_agent");
        assert_eq!(config.llm_model, "gpt-4-turbo");
        assert_eq!(config.temperature, 0.5);
        assert_eq!(config.max_tokens, 4000);
        assert!(config.memory_enabled);
        assert!(config.evaluation_enabled);
    }

    #[test]
    fn test_temperature_clamping() {
        let builder = AgentBuilder::new(AgentType::Conversational).with_temperature(2.0); // Out of range

        assert_eq!(builder.config().temperature, 1.0);
    }

    #[test]
    fn test_agent_type_parsing() {
        assert_eq!(
            AgentType::from_type_str("zero-shot-react"),
            AgentType::ZeroShotReAct
        );
        assert_eq!(
            AgentType::from_type_str("conversational"),
            AgentType::Conversational
        );
        assert_eq!(
            AgentType::from_type_str("invalid"),
            AgentType::ZeroShotReAct
        );
    }

    #[test]
    fn test_agent_builder_builds_graph_based_agent() {
        let agent = AgentBuilder::new(AgentType::ZeroShotReAct)
            .with_name("test_agent")
            .with_tool(ToolSpec::new("calculator", "Perform math"))
            .with_tool_executor(|name, args| format!("Executed {} with {}", name, args))
            .build()
            .expect("Failed to build agent");

        assert_eq!(agent.name(), "test_agent");
        assert_eq!(agent.agent_type(), AgentType::ZeroShotReAct);
        assert_eq!(agent.tools().len(), 1);
    }
}
