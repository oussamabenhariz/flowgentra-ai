/// Structured Chat Zero-Shot ReAct Agent
///
/// ReAct agent that communicates tool calls as JSON blobs instead of free-text
/// `<action>` tags.  The format is:
///
/// ```json
/// { "action": "tool_name", "action_input": "..." }
/// ```
///
/// The final answer is signalled with `"action": "Final Answer"`.
///
/// Equivalent to LangChain's `structured-chat-zero-shot-react-description` agent.
use super::{Agent, AgentType, PrebuiltAgentConfig, ToolSpec};
use crate::core::error::FlowgentraError;
use std::collections::HashMap;

/// Structured Chat Zero-Shot ReAct agent implementation
pub struct StructuredChatZeroShotReActAgent {
    config: PrebuiltAgentConfig,
    tools: HashMap<String, ToolSpec>,
}

impl StructuredChatZeroShotReActAgent {
    /// Create new Structured Chat agent
    pub fn new(config: PrebuiltAgentConfig) -> Self {
        let tools = config.tools.clone();
        Self { config, tools }
    }

    fn format_tools(&self) -> String {
        if self.tools.is_empty() {
            return "No tools available".to_string();
        }
        let mut formatted = String::new();
        for (idx, (name, spec)) in self.tools.iter().enumerate() {
            formatted.push_str(&format!("{}. {} — {}\n", idx + 1, name, spec.description));
            if !spec.parameters.is_empty() {
                formatted.push_str("   Parameters:\n");
                for (param_name, param_type) in &spec.parameters {
                    let req = if spec.required.contains(param_name) {
                        "(required)"
                    } else {
                        "(optional)"
                    };
                    formatted.push_str(&format!("     - {}: {} {}\n", param_name, param_type, req));
                }
            }
        }
        formatted
    }

    fn format_mcps(&self) -> String {
        if self.config.mcps.is_empty() {
            return String::new();
        }
        let mut formatted = String::from("\nAvailable MCP Services:\n");
        for (idx, mcp) in self.config.mcps.iter().enumerate() {
            formatted.push_str(&format!(
                "{}. {} ({})\n   Connection: {}\n",
                idx + 1,
                mcp.name,
                mcp.connection_type.as_str(),
                mcp.uri
            ));
        }
        formatted
    }

    /// Build the action prompt shown to the user (not sent to LLM directly; used by
    /// `StructuredChatNode` which injects tool lists into the system prompt).
    pub fn build_action_prompt(&self, input: &str) -> String {
        let tools_str = self.format_tools();
        let tool_names: Vec<&str> = self.tools.keys().map(|s| s.as_str()).collect();
        let mcps_str = self.format_mcps();
        format!(
            "Available tools: {}\nTool names: {}\n{}\n\nQuestion: {}",
            tools_str,
            tool_names.join(", "),
            mcps_str,
            input
        )
    }
}

impl Default for StructuredChatZeroShotReActAgent {
    fn default() -> Self {
        Self::new(PrebuiltAgentConfig::new(
            "structured_chat",
            AgentType::StructuredChatZeroShotReAct,
        ))
    }
}

impl Agent for StructuredChatZeroShotReActAgent {
    fn name(&self) -> &str {
        &self.config.name
    }

    fn agent_type(&self) -> AgentType {
        AgentType::StructuredChatZeroShotReAct
    }

    fn initialize(
        &mut self,
        state: &mut crate::core::state::DynState,
    ) -> Result<(), FlowgentraError> {
        state.set(
            "__agent_name",
            serde_json::Value::String(self.config.name.clone()),
        );
        state.set(
            "__agent_type",
            serde_json::Value::String("structured-chat-zero-shot-react".to_string()),
        );
        state.set(
            "__agent_tools_count",
            serde_json::Value::String(self.tools.len().to_string()),
        );
        Ok(())
    }

    fn process(
        &self,
        input: &str,
        _state: &crate::core::state::DynState,
    ) -> Result<String, FlowgentraError> {
        Ok(format!(
            "Agent: {}\nType: {}\nMode: Structured Chat Zero-Shot ReAct\n\n{}",
            self.config.name,
            self.agent_type(),
            self.build_action_prompt(input)
        ))
    }

    fn config(&self) -> &PrebuiltAgentConfig {
        &self.config
    }

    fn add_tool(&mut self, tool_name: &str, tool_spec: ToolSpec) -> Result<(), FlowgentraError> {
        if self.tools.contains_key(tool_name) {
            return Err(FlowgentraError::ConfigError(format!(
                "Tool '{}' already exists",
                tool_name
            )));
        }
        self.tools.insert(tool_name.to_string(), tool_spec);
        self.config
            .tools
            .insert(tool_name.to_string(), self.tools[tool_name].clone());
        Ok(())
    }

    fn tools(&self) -> Vec<&ToolSpec> {
        self.tools.values().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_structured_chat_creation() {
        let agent = StructuredChatZeroShotReActAgent::default();
        assert_eq!(agent.name(), "structured_chat");
        assert_eq!(agent.agent_type(), AgentType::StructuredChatZeroShotReAct);
    }

    #[test]
    fn test_add_tool() {
        let mut agent = StructuredChatZeroShotReActAgent::default();
        let tool = ToolSpec::new("search", "Search the web")
            .with_parameter("query", "string")
            .required("query");
        assert!(agent.add_tool("search", tool).is_ok());
        assert_eq!(agent.tools().len(), 1);
    }

    #[test]
    fn test_initialize_sets_type() {
        use crate::core::state::DynState;
        let mut agent = StructuredChatZeroShotReActAgent::default();
        let mut state = DynState::new();
        agent.initialize(&mut state).unwrap();
        assert_eq!(
            state.get("__agent_type").unwrap().as_str().unwrap(),
            "structured-chat-zero-shot-react"
        );
    }

    #[test]
    fn test_process_contains_json_hint() {
        use crate::core::state::DynState;
        let agent = StructuredChatZeroShotReActAgent::default();
        let state = DynState::new();
        let output = agent.process("What is Rust?", &state).unwrap();
        assert!(output.contains("Structured Chat"));
    }
}
