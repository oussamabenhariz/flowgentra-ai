/// Agent Builder Factory
///
/// Provides builder patterns for creating predefined agents with sensible defaults.
/// Similar to LangChain's `initialize_agent` but tailored for ErenFlowAI.
use super::{Agent, AgentType, ToolSpec};
use crate::core::error::ErenFlowError;
use crate::core::mcp::MCPConfig;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Agent configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentConfig {
    /// Agent name
    pub name: String,

    /// Agent type
    pub agent_type: String,

    /// LLM configuration
    pub llm_model: String,

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

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            name: "agent".to_string(),
            agent_type: "default".to_string(),
            llm_model: "gpt-4".to_string(),
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

impl AgentConfig {
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

/// Agent builder for fluent configuration
pub struct AgentBuilder {
    config: AgentConfig,
}

impl AgentBuilder {
    /// Create new agent builder
    pub fn new(agent_type: AgentType) -> Self {
        Self {
            config: AgentConfig::new("agent", agent_type),
        }
    }

    /// Set agent name
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.config.name = name.into();
        self
    }

    /// Set LLM configuration
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
    pub fn config(&self) -> &AgentConfig {
        &self.config
    }

    /// Build agent (returns boxed trait object)
    pub fn build(self) -> Result<Box<dyn Agent>, ErenFlowError> {
        match AgentType::from_type_str(&self.config.agent_type) {
            AgentType::ZeroShotReAct => Ok(Box::new(super::ZeroShotReActAgent::new(self.config))),
            AgentType::FewShotReAct => Ok(Box::new(super::FewShotReActAgent::new(self.config))),
            AgentType::Conversational => Ok(Box::new(super::ConversationalAgent::new(self.config))),
        }
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
        assert_eq!(AgentType::from_type_str("invalid"), AgentType::ZeroShotReAct);
    }
}
