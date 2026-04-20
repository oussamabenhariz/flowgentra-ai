/// Predefined Agent Types
///
/// FlowgentraAI provides typed agent constructors for common agent patterns.
/// Instantiate directly with [`AgentConfig`]:
///
/// ```ignore
/// use flowgentra_ai::core::agents::{FewShotReAct, AgentConfig, ToolSpec};
///
/// let agent = FewShotReAct::new(AgentConfig {
///     name: "classifier".into(),
///     llm: llm,
///     system_prompt: Some("Example 1: urgent bug → Priority: HIGH".into()),
///     tools: vec![ToolSpec::new("search", "Search the web")],
///     retries: 2,
///     memory_steps: Some(10),
///     ..Default::default()
/// })?;
///
/// let result = agent.execute_input("App crashes on login").await?;
/// ```
mod builders;
mod conversational;
mod few_shot_react;
pub mod graph_nodes;
mod prompts;
mod react_docstore;
mod self_ask_with_search;
mod structured_chat;
pub mod supervisor;
mod tool_calling;
mod zero_shot_react;

pub use builders::{
    AgentConfig, Conversational, FewShotReAct, GraphBasedAgent, PrebuiltAgentConfig,
    ReactDocstore, SelfAskWithSearch, StructuredChat, ToolCalling, ZeroShotReAct,
};
pub use conversational::ConversationalAgent;
pub use few_shot_react::FewShotReActAgent;
pub use graph_nodes::{
    docstore_router, reasoning_router, self_ask_router, tool_calling_router, AgentReasoningNode,
    ConversationalNode, DocstoreNode, SelfAskNode, StructuredChatNode, ToolCallingNode,
    ToolExecutorNode,
};
pub use prompts::{PromptTemplates, SystemPrompts};
pub use react_docstore::ReactDocstoreAgent;
pub use self_ask_with_search::SelfAskWithSearchAgent;
pub use structured_chat::StructuredChatZeroShotReActAgent;
pub use supervisor::Supervisor;
pub use tool_calling::ToolCallingAgent;
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
    /// Tool Calling: Provider-agnostic native function/tool calling API
    ToolCalling,
    /// Structured Chat Zero-Shot ReAct: ReAct with JSON-structured actions
    StructuredChatZeroShotReAct,
    /// Self Ask With Search: Decomposes questions into sub-questions answered by search
    SelfAskWithSearch,
    /// ReAct Docstore: ReAct loop specialized for Search + Lookup in a document store
    ReactDocstore,
}

impl std::fmt::Display for AgentType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AgentType::ZeroShotReAct => write!(f, "zero-shot-react"),
            AgentType::FewShotReAct => write!(f, "few-shot-react"),
            AgentType::Conversational => write!(f, "conversational"),
            AgentType::ToolCalling => write!(f, "tool-calling"),
            AgentType::StructuredChatZeroShotReAct => {
                write!(f, "structured-chat-zero-shot-react")
            }
            AgentType::SelfAskWithSearch => write!(f, "self-ask-with-search"),
            AgentType::ReactDocstore => write!(f, "react-docstore"),
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

    #[test]
    fn test_new_agent_type_display() {
        assert_eq!(AgentType::ToolCalling.to_string(), "tool-calling");
        assert_eq!(
            AgentType::StructuredChatZeroShotReAct.to_string(),
            "structured-chat-zero-shot-react"
        );
        assert_eq!(
            AgentType::SelfAskWithSearch.to_string(),
            "self-ask-with-search"
        );
        assert_eq!(AgentType::ReactDocstore.to_string(), "react-docstore");
    }
}
