/// Tool Calling Agent
///
/// Uses the provider's native function/tool-calling API instead of text-based
/// `<action>` tags.  Supported by OpenAI, Anthropic, Mistral, Groq, and any
/// OpenAI-compatible endpoint.
///
/// Equivalent to LangChain's `tool-calling` agent type.
use super::{Agent, AgentType, PrebuiltAgentConfig, ToolSpec};
use crate::core::error::FlowgentraError;
use std::collections::HashMap;

/// Tool Calling agent implementation
pub struct ToolCallingAgent {
    config: PrebuiltAgentConfig,
    tools: HashMap<String, ToolSpec>,
}

impl ToolCallingAgent {
    /// Create new Tool Calling agent
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
            formatted.push_str(&format!("{}. {}: {}\n", idx + 1, name, spec.description));
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
}

impl Default for ToolCallingAgent {
    fn default() -> Self {
        Self::new(PrebuiltAgentConfig::new(
            "tool_calling",
            AgentType::ToolCalling,
        ))
    }
}

impl Agent for ToolCallingAgent {
    fn name(&self) -> &str {
        &self.config.name
    }

    fn agent_type(&self) -> AgentType {
        AgentType::ToolCalling
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
            serde_json::Value::String("tool-calling".to_string()),
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
        let tools_str = self.format_tools();
        let mcps_str = self.format_mcps();
        Ok(format!(
            "Agent: {}\nType: {}\nMode: Tool Calling\n\nSystem: {}\n\nAvailable Tools:\n{}{}Input: {}",
            self.config.name,
            self.agent_type(),
            self.config.system_prompt,
            tools_str,
            mcps_str,
            input
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
    fn test_tool_calling_agent_creation() {
        let agent = ToolCallingAgent::default();
        assert_eq!(agent.name(), "tool_calling");
        assert_eq!(agent.agent_type(), AgentType::ToolCalling);
    }

    #[test]
    fn test_add_tool() {
        let mut agent = ToolCallingAgent::default();
        let tool = ToolSpec::new("get_weather", "Get current weather")
            .with_parameter("location", "string")
            .required("location");
        assert!(agent.add_tool("get_weather", tool).is_ok());
        assert_eq!(agent.tools().len(), 1);
    }

    #[test]
    fn test_duplicate_tool_error() {
        let mut agent = ToolCallingAgent::default();
        agent.add_tool("t", ToolSpec::new("t", "tool")).unwrap();
        assert!(agent.add_tool("t", ToolSpec::new("t", "dup")).is_err());
    }

    #[test]
    fn test_initialize() {
        use crate::core::state::DynState;
        let mut agent = ToolCallingAgent::default();
        let mut state = DynState::new();
        assert!(agent.initialize(&mut state).is_ok());
        assert_eq!(
            state.get("__agent_type").unwrap().as_str().unwrap(),
            "tool-calling"
        );
    }
}
