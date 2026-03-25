/// Zero-Shot ReAct Agent
///
/// General-purpose reasoning + action agent that works without examples.
/// Best for open-ended tasks requiring flexible problem-solving.
use super::{Agent, AgentType, PrebuiltAgentConfig, ToolSpec};
use crate::core::error::FlowgentraError;
use std::collections::HashMap;

/// Zero-shot ReAct agent implementation
pub struct ZeroShotReActAgent {
    config: PrebuiltAgentConfig,
    tools: HashMap<String, ToolSpec>,
}

impl ZeroShotReActAgent {
    /// Create new zero-shot ReAct agent
    pub fn new(config: PrebuiltAgentConfig) -> Self {
        let tools = config.tools.clone();
        Self { config, tools }
    }

    /// Format tools for agent prompt
    fn format_tools(&self) -> String {
        if self.tools.is_empty() {
            return "No tools available".to_string();
        }

        let mut formatted = String::new();
        for (idx, (name, spec)) in self.tools.iter().enumerate() {
            formatted.push_str(&format!("{}. {}: {}\n", idx + 1, name, spec.description));
            if !spec.parameters.is_empty() {
                formatted.push_str("   Parameters:\n");
                for (param_name, param_type) in &spec.parameters {
                    let required = if spec.required.contains(param_name) {
                        "(required)"
                    } else {
                        "(optional)"
                    };
                    formatted.push_str(&format!(
                        "     - {}: {} {}\n",
                        param_name, param_type, required
                    ));
                }
            }
        }
        formatted
    }

    /// Format MCP configurations for agent prompt
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

    /// Generate reasoning chain
    fn generate_reasoning_chain(&self, input: &str) -> String {
        let tools_str = self.format_tools();
        let mcps_str = self.format_mcps();

        format!(
            r#"{}

Available Tools:
{}{}

Input: {}

Think through this step by step:
1. What is the core problem?
2. What tools or MCP services would help?
3. How will you verify your answer?

Provide your reasoning and solution."#,
            self.config.system_prompt, tools_str, mcps_str, input
        )
    }
}

impl Default for ZeroShotReActAgent {
    fn default() -> Self {
        Self::new(PrebuiltAgentConfig::new(
            "zero_shot_react",
            AgentType::ZeroShotReAct,
        ))
    }
}

impl Agent for ZeroShotReActAgent {
    fn name(&self) -> &str {
        &self.config.name
    }

    fn agent_type(&self) -> AgentType {
        AgentType::ZeroShotReAct
    }

    fn initialize(
        &mut self,
        state: &mut crate::core::state::SharedState,
    ) -> Result<(), FlowgentraError> {
        // Store agent metadata in state
        state.set(
            "__agent_name",
            serde_json::Value::String(self.config.name.clone()),
        );
        state.set(
            "__agent_type",
            serde_json::Value::String("zero-shot-react".to_string()),
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
        _state: &crate::core::state::SharedState,
    ) -> Result<String, FlowgentraError> {
        // For production, this would integrate with actual LLM
        // For now, return structured thinking process

        let reasoning = self.generate_reasoning_chain(input);

        Ok(format!(
            "Agent: {}\nType: {}\nMode: Zero-Shot ReAct\n\nReasoning:\n{}",
            self.config.name,
            self.agent_type(),
            reasoning
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
        // Update config
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
    fn test_zero_shot_react_creation() {
        let agent = ZeroShotReActAgent::default();
        assert_eq!(agent.name(), "zero_shot_react");
        assert_eq!(agent.agent_type(), AgentType::ZeroShotReAct);
    }

    #[test]
    fn test_add_tool() {
        let mut agent = ZeroShotReActAgent::default();
        let tool = ToolSpec::new("search", "Search the web")
            .with_parameter("query", "string")
            .required("query");

        assert!(agent.add_tool("search", tool).is_ok());
        assert_eq!(agent.tools().len(), 1);
    }

    #[test]
    fn test_duplicate_tool_error() {
        let mut agent = ZeroShotReActAgent::default();
        let tool1 = ToolSpec::new("calc", "Calculator");
        let tool2 = ToolSpec::new("calc", "Another calculator");

        agent.add_tool("calc", tool1).unwrap();
        let result = agent.add_tool("calc", tool2);
        assert!(result.is_err());
    }

    #[test]
    fn test_format_tools() {
        let mut agent = ZeroShotReActAgent::default();
        agent
            .add_tool(
                "math",
                ToolSpec::new("math", "Perform math operations")
                    .with_parameter("operation", "string")
                    .required("operation"),
            )
            .unwrap();

        let formatted = agent.format_tools();
        assert!(formatted.contains("math"));
        assert!(formatted.contains("Perform math operations"));
    }

    #[test]
    fn test_process() {
        use crate::core::state::SharedState;
        let agent = ZeroShotReActAgent::default();
        let state = SharedState::new(Default::default());
        let result = agent.process("What is 2+2?", &state);

        assert!(result.is_ok());
        let output = result.unwrap();
        assert!(output.contains("zero-shot-react"));
    }

    #[test]
    fn test_initialize() {
        use crate::core::state::SharedState;
        let mut agent = ZeroShotReActAgent::default();
        let mut state = SharedState::new(Default::default());

        assert!(agent.initialize(&mut state).is_ok());
        assert!(state.get("__agent_name").is_some());
        assert!(state.get("__agent_type").is_some());
    }
}
