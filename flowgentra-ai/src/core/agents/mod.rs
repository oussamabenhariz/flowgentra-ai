/// Predefined Agent Types
///
/// Similar to LangChain, FlowgentraAI provides factory builders for common agent patterns.
/// Users can initialize and use agents directly without building complex graph configurations.
///
/// # Examples
///
/// ```ignore
/// use flowgentra_ai::core::agents::{AgentBuilder, AgentType, ToolSpec};
///
/// // Create a zero-shot ReAct agent
/// let calculator = ToolSpec::new("calculator", "Perform calculations");
/// let search = ToolSpec::new("search", "Search the web");
///
/// let agent = AgentBuilder::new(AgentType::ZeroShotReAct)
///     .with_llm_config("gpt-4")
///     .with_tool(calculator)
///     .with_tool(search)
///     .build()?;
///
/// // Create a conversational agent with memory
/// let agent = AgentBuilder::new(AgentType::Conversational)
///     .with_memory_steps(10)
///     .with_system_prompt("You are a helpful assistant")
///     .build()?;
/// ```
mod builders;
mod conversational;
mod few_shot_react;
pub mod graph_nodes;
mod prompts;
pub mod supervisor;
mod zero_shot_react;

pub use builders::{AgentBuilder, GraphBasedAgent, PrebuiltAgentConfig};
pub use conversational::ConversationalAgent;
pub use few_shot_react::FewShotReActAgent;
pub use graph_nodes::{reasoning_router, AgentReasoningNode, ConversationalNode, ToolExecutorNode};
pub use prompts::{PromptTemplates, SystemPrompts};
pub use supervisor::Supervisor;
pub use zero_shot_react::ZeroShotReActAgent;

use crate::core::error::FlowgentraError;
use std::collections::HashMap;

/// Agent type enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AgentType {
    /// Zero-shot ReAct: General purpose reasoning + action without examples
    ZeroShotReAct,
    /// Few-shot ReAct: ReAct with example demonstrations
    FewShotReAct,
    /// Conversational: Multi-turn dialogue with memory
    Conversational,
}

impl std::fmt::Display for AgentType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AgentType::ZeroShotReAct => write!(f, "zero-shot-react"),
            AgentType::FewShotReAct => write!(f, "few-shot-react"),
            AgentType::Conversational => write!(f, "conversational"),
        }
    }
}

/// Trait for agent implementations
pub trait Agent: Send + Sync {
    /// Get agent name
    fn name(&self) -> &str;

    /// Get agent type
    fn agent_type(&self) -> AgentType;

    /// Initialize agent with state
    fn initialize(
        &mut self,
        state: &mut crate::core::state::DynState,
    ) -> Result<(), FlowgentraError>;

    /// Process user input and generate response
    fn process(
        &self,
        input: &str,
        state: &crate::core::state::DynState,
    ) -> Result<String, FlowgentraError>;

    /// Get agent configuration
    fn config(&self) -> &PrebuiltAgentConfig;

    /// Add tool to agent
    fn add_tool(&mut self, tool_name: &str, tool_spec: ToolSpec) -> Result<(), FlowgentraError>;

    /// Get available tools
    fn tools(&self) -> Vec<&ToolSpec>;
}

/// Tool specification
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ToolSpec {
    pub name: String,
    pub description: String,
    pub parameters: HashMap<String, String>,
    pub required: Vec<String>,
}

impl ToolSpec {
    /// Create new tool specification
    pub fn new(name: impl Into<String>, description: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            parameters: HashMap::new(),
            required: Vec::new(),
        }
    }

    /// Add parameter to tool
    pub fn with_parameter(
        mut self,
        name: impl Into<String>,
        param_type: impl Into<String>,
    ) -> Self {
        self.parameters.insert(name.into(), param_type.into());
        self
    }

    /// Mark parameter as required
    pub fn required(mut self, param: impl Into<String>) -> Self {
        self.required.push(param.into());
        self
    }
}

/// Agent execution result
#[derive(Debug, Clone)]
pub struct AgentResult {
    pub output: String,
    pub reasoning: Option<String>,
    pub tool_calls: Vec<ToolCall>,
    pub success: bool,
}

impl AgentResult {
    /// Create successful result
    pub fn success(output: impl Into<String>) -> Self {
        Self {
            output: output.into(),
            reasoning: None,
            tool_calls: Vec::new(),
            success: true,
        }
    }

    /// Create result with reasoning
    pub fn with_reasoning(output: impl Into<String>, reasoning: impl Into<String>) -> Self {
        Self {
            output: output.into(),
            reasoning: Some(reasoning.into()),
            tool_calls: Vec::new(),
            success: true,
        }
    }

    /// Create failed result
    pub fn failed(error: impl Into<String>) -> Self {
        Self {
            output: error.into(),
            reasoning: None,
            tool_calls: Vec::new(),
            success: false,
        }
    }
}

/// Tool call in agent execution
#[derive(Debug, Clone)]
pub struct ToolCall {
    pub tool_name: String,
    pub arguments: HashMap<String, String>,
    pub result: Option<String>,
}

impl ToolCall {
    /// Create new tool call
    pub fn new(tool_name: impl Into<String>) -> Self {
        Self {
            tool_name: tool_name.into(),
            arguments: HashMap::new(),
            result: None,
        }
    }

    /// Add argument to tool call
    pub fn with_arg(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.arguments.insert(key.into(), value.into());
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tool_spec_creation() {
        let tool = ToolSpec::new("search", "Search the web")
            .with_parameter("query", "string")
            .required("query");

        assert_eq!(tool.name, "search");
        assert_eq!(tool.parameters.len(), 1);
        assert!(tool.required.contains(&"query".to_string()));
    }

    #[test]
    fn test_agent_result_creation() {
        let result = AgentResult::success("Done");
        assert!(result.success);
        assert_eq!(result.output, "Done");

        let failed = AgentResult::failed("Error occurred");
        assert!(!failed.success);
    }

    #[test]
    fn test_tool_call_creation() {
        let call = ToolCall::new("calculator")
            .with_arg("operation", "add")
            .with_arg("a", "5")
            .with_arg("b", "3");

        assert_eq!(call.tool_name, "calculator");
        assert_eq!(call.arguments.len(), 3);
    }
}
