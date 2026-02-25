//! # Builder Patterns for Complex Configuration
//!
//! Provides fluent builder APIs for constructing complex configuration objects
//! in a type-safe, ergonomic manner.

use crate::core::config::{AgentConfig, GraphConfig};
use crate::core::error::Result;
use crate::core::llm::LLMConfig;
use crate::core::mcp::MCPConfig;
use crate::core::node::{EdgeConfig, NodeConfig};
use std::collections::HashMap;

/// Builder for AgentConfig
pub struct AgentConfigBuilder {
    name: Option<String>,
    description: Option<String>,
    llm: Option<LLMConfig>,
    nodes: Vec<NodeConfig>,
    edges: Vec<EdgeConfig>,
    mcps: HashMap<String, MCPConfig>,
}

impl AgentConfigBuilder {
    /// Create a new builder
    pub fn new() -> Self {
        Self {
            name: None,
            description: None,
            llm: None,
            nodes: Vec::new(),
            edges: Vec::new(),
            mcps: HashMap::new(),
        }
    }

    /// Set agent name
    pub fn name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Set agent description
    pub fn description(mut self, desc: impl Into<String>) -> Self {
        self.description = Some(desc.into());
        self
    }

    /// Set LLM configuration
    pub fn llm(mut self, config: LLMConfig) -> Self {
        self.llm = Some(config);
        self
    }

    /// Add a node configuration
    pub fn with_node(mut self, node: NodeConfig) -> Self {
        self.nodes.push(node);
        self
    }

    /// Add an edge configuration
    pub fn with_edge(mut self, edge: EdgeConfig) -> Self {
        self.edges.push(edge);
        self
    }

    /// Add MCP configuration
    pub fn with_mcp(mut self, name: impl Into<String>, config: MCPConfig) -> Self {
        self.mcps.insert(name.into(), config);
        self
    }

    /// Build the agent configuration
    pub fn build(self) -> Result<AgentConfig> {
        let name = self.name.ok_or_else(|| {
            crate::core::error::ErenFlowError::ConfigError("Agent name is required".into())
        })?;

        let llm = self.llm.ok_or_else(|| {
            crate::core::error::ErenFlowError::ConfigError("LLM config is required".into())
        })?;

        let graph = GraphConfig {
            nodes: self.nodes,
            edges: self.edges,
            mcps: self.mcps,
            parallel: Vec::new(),
            planner: Default::default(),
        };

        let config = AgentConfig {
            name,
            description: self.description,
            llm,
            graph,
            state_schema: HashMap::new(),
            validation_schema: None,
            memory: Default::default(),
        };

        tracing::info!(agent_name = %config.name, "Agent config built");
        Ok(config)
    }
}

impl Default for AgentConfigBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builder_requires_name() {
        let result = AgentConfigBuilder::new().build();
        assert!(result.is_err());
    }
}
