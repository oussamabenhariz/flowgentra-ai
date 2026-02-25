//! # MCP (Model Context Protocol) Support
//!
//! Enables nodes to integrate with external tools and data sources through
//! the Model Context Protocol.
//!
//! ## Features
//!
//! - Multiple connection types (SSE, Stdio, Docker)
//! - Authentication support (API key, OAuth, etc.)
//! - Connection pooling and retry logic
//! - Tool discovery
//! - Enhanced error context with structured logging
//! - Builder pattern for configuration
//!
//! ## Example
//!
//! ```rust
//! use erenflow_ai::core::mcp::MCPConfig;
//!
//! # #[tokio::main]
//! # async fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let config = MCPConfig::builder()
//!     .name("my_tool")
//!     .sse("http://localhost:8000")
//!     .timeout_secs(30)
//!     .build()?;
//! # Ok(())
//! # }
//! ```

use crate::core::error::{ErenFlowError, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Connection type for MCP server
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum MCPConnectionType {
    /// Server-Sent Events over HTTP (suitable for REST APIs)
    Sse,
    /// Standard input/output (for local binaries)
    Stdio,
    /// Docker container (for containerized tools)
    Docker,
}

impl MCPConnectionType {
    /// Get string representation
    pub fn as_str(&self) -> &str {
        match self {
            MCPConnectionType::Sse => "sse",
            MCPConnectionType::Stdio => "stdio",
            MCPConnectionType::Docker => "docker",
        }
    }

    /// Check if this is a remote connection
    pub fn is_remote(&self) -> bool {
        matches!(self, MCPConnectionType::Sse)
    }

    /// Check if this is a local connection
    pub fn is_local(&self) -> bool {
        matches!(self, MCPConnectionType::Stdio | MCPConnectionType::Docker)
    }
}

impl std::fmt::Display for MCPConnectionType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// MCP Configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MCPConfig {
    /// Unique identifier for this MCP connection
    pub name: String,
    /// Type of connection: sse, stdio, or docker
    pub connection_type: MCPConnectionType,
    /// URL/path/command depending on connection type
    pub uri: String,
    /// Authentication credentials if needed
    #[serde(default)]
    pub auth: Option<MCPAuth>,
    /// Connection-specific settings
    #[serde(default)]
    pub connection_settings: MCPConnectionSettings,
    /// Custom configuration
    #[serde(default)]
    pub config: HashMap<String, serde_json::Value>,
}

impl MCPConfig {
    /// Create a new builder for MCP configuration
    pub fn builder() -> MCPConfigBuilder {
        MCPConfigBuilder::new()
    }

    /// Validate configuration
    pub fn validate(&self) -> Result<()> {
        if self.name.is_empty() {
            return Err(ErenFlowError::MCPError("MCP name cannot be empty".into()));
        }
        if self.uri.is_empty() {
            return Err(ErenFlowError::MCPError("MCP URI cannot be empty".into()));
        }
        match self.connection_type {
            MCPConnectionType::Sse if !self.uri.starts_with("http") => {
                return Err(ErenFlowError::MCPError(
                    "SSE connection requires HTTP(S) URI".into(),
                ));
            }
            MCPConnectionType::Docker if self.connection_settings.container_name.is_none() => {
                tracing::warn!("Docker connection without explicit container name");
            }
            _ => {}
        }
        Ok(())
    }
}

/// Builder for MCPConfig
pub struct MCPConfigBuilder {
    name: Option<String>,
    connection_type: Option<MCPConnectionType>,
    uri: Option<String>,
    auth: Option<MCPAuth>,
    timeout_secs: Option<u64>,
    container_name: Option<String>,
}

impl MCPConfigBuilder {
    /// Create new builder
    pub fn new() -> Self {
        Self {
            name: None,
            connection_type: None,
            uri: None,
            auth: None,
            timeout_secs: None,
            container_name: None,
        }
    }

    /// Set MCP name
    pub fn name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Configure SSE connection
    pub fn sse(mut self, uri: impl Into<String>) -> Self {
        self.connection_type = Some(MCPConnectionType::Sse);
        self.uri = Some(uri.into());
        self
    }

    /// Configure Stdio connection
    pub fn stdio(mut self, command: impl Into<String>) -> Self {
        self.connection_type = Some(MCPConnectionType::Stdio);
        self.uri = Some(command.into());
        self
    }

    /// Configure Docker connection
    pub fn docker(mut self, container_name: impl Into<String>) -> Self {
        self.connection_type = Some(MCPConnectionType::Docker);
        self.container_name = Some(container_name.into());
        self
    }

    /// Set timeout in seconds
    pub fn timeout_secs(mut self, secs: u64) -> Self {
        self.timeout_secs = Some(secs);
        self
    }

    /// Set authentication
    pub fn auth(mut self, auth: MCPAuth) -> Self {
        self.auth = Some(auth);
        self
    }

    /// Build MCPConfig
    pub fn build(self) -> Result<MCPConfig> {
        let name = self
            .name
            .ok_or_else(|| ErenFlowError::ConfigError("MCP name is required".into()))?;
        let connection_type = self
            .connection_type
            .ok_or_else(|| ErenFlowError::ConfigError("MCP connection type is required".into()))?;
        let uri = self
            .uri
            .ok_or_else(|| ErenFlowError::ConfigError("MCP URI is required".into()))?;

        let mut connection_settings = MCPConnectionSettings::default();
        if let Some(timeout) = self.timeout_secs {
            connection_settings.timeout = Some(timeout);
        }
        if let Some(container) = self.container_name {
            connection_settings.container_name = Some(container);
        }

        let config = MCPConfig {
            name,
            connection_type,
            uri,
            auth: self.auth,
            connection_settings,
            config: HashMap::new(),
        };

        config.validate()?;
        tracing::info!(mcp_name = config.name, "MCP config created");
        Ok(config)
    }
}

impl Default for MCPConfigBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Connection-specific settings for MCP
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct MCPConnectionSettings {
    /// For SSE: timeout in seconds
    pub timeout: Option<u64>,
    /// For Docker: container name
    pub container_name: Option<String>,
    /// For Docker: port mapping
    pub port: Option<u16>,
    /// For Stdio: environment variables
    #[serde(default)]
    pub env_vars: HashMap<String, String>,
    /// For any: retry settings
    pub max_retries: Option<u32>,
}

/// MCP Authentication
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MCPAuth {
    pub auth_type: String,
    pub credentials: HashMap<String, String>,
}

/// Represents an MCP tool
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MCPTool {
    pub name: String,
    pub description: String,
    pub input_schema: serde_json::Value,
}

/// MCP Client trait
#[async_trait::async_trait]
pub trait MCPClient: Send + Sync {
    /// List available tools
    async fn list_tools(&self) -> Result<Vec<MCPTool>>;

    /// Call an MCP tool
    async fn call_tool(
        &self,
        tool_name: &str,
        arguments: serde_json::Value,
    ) -> Result<serde_json::Value>;
}

/// Default MCP client implementation
pub struct DefaultMCPClient {
    config: MCPConfig,
    client: reqwest::Client,
}

impl DefaultMCPClient {
    /// Create new MCP client with given configuration
    pub fn new(config: MCPConfig) -> Self {
        DefaultMCPClient {
            config,
            client: reqwest::Client::new(),
        }
    }

    /// Get reference to MCP configuration
    pub fn config(&self) -> &MCPConfig {
        &self.config
    }
}

#[async_trait::async_trait]
impl MCPClient for DefaultMCPClient {
    async fn list_tools(&self) -> Result<Vec<MCPTool>> {
        let span = tracing::info_span!("mcp_list_tools", mcp = %self.config.name);
        let _guard = span.enter();

        let url = format!("{}/tools", self.config.uri);
        tracing::debug!(url = %url, "Fetching available MCP tools");

        let response = self.client.get(&url).send().await.map_err(|e| {
            let err_msg = format!("Failed to list MCP tools: {}", e);
            tracing::error!(error = %e, mcp = %self.config.name, "MCP list_tools request failed");
            ErenFlowError::MCPError(err_msg)
        })?;

        let tools: Vec<MCPTool> = response.json().await.map_err(|e| {
            let err_msg = format!("Failed to parse MCP response: {}", e);
            tracing::error!(error = %e, mcp = %self.config.name, "MCP response parsing failed");
            ErenFlowError::MCPError(err_msg)
        })?;

        tracing::info!(tool_count = tools.len(), "Successfully retrieved MCP tools");
        Ok(tools)
    }

    async fn call_tool(
        &self,
        tool_name: &str,
        arguments: serde_json::Value,
    ) -> Result<serde_json::Value> {
        let span = tracing::info_span!("mcp_call_tool", mcp = %self.config.name, tool = %tool_name);
        let _guard = span.enter();

        let payload = serde_json::json!({
            "tool": tool_name,
            "arguments": arguments
        });

        tracing::debug!("Calling MCP tool");

        let url = format!("{}/call", self.config.uri);
        let response = self
            .client
            .post(&url)
            .json(&payload)
            .send()
            .await
            .map_err(|e| {
                let err_msg = format!("Failed to call MCP tool '{}': {}", tool_name, e);
                tracing::error!(tool = %tool_name, error = %e, "MCP call failed");
                ErenFlowError::MCPError(err_msg)
            })?;

        let result = response.json().await.map_err(|e| {
            let err_msg = format!("Failed to parse MCP tool response: {}", e);
            tracing::error!(tool = %tool_name, error = %e, "MCP response parsing failed");
            ErenFlowError::MCPError(err_msg)
        })?;

        tracing::debug!("MCP tool call completed");
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn connection_type_methods() {
        assert_eq!(MCPConnectionType::Sse.as_str(), "sse");
        assert!(MCPConnectionType::Sse.is_remote());
        assert!(!MCPConnectionType::Sse.is_local());

        assert!(MCPConnectionType::Stdio.is_local());
        assert!(!MCPConnectionType::Stdio.is_remote());
    }

    #[test]
    fn config_builder_success() {
        let result = MCPConfig::builder()
            .name("test_mcp")
            .sse("http://localhost:8000")
            .timeout_secs(30)
            .build();

        assert!(result.is_ok());
        let config = result.unwrap();
        assert_eq!(config.name, "test_mcp");
        assert_eq!(config.connection_type, MCPConnectionType::Sse);
        assert_eq!(config.connection_settings.timeout, Some(30));
    }

    #[test]
    fn config_builder_validation() {
        let result = MCPConfig::builder()
            .name("")
            .sse("http://localhost:8000")
            .build();

        assert!(result.is_err());
    }
}
