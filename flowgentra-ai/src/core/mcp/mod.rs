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
//! use flowgentra_ai::core::mcp::MCPConfig;
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

// Module declarations
pub mod factory;
pub mod sse;
pub mod stdio;
pub mod docker;

// Re-exports for convenience
pub use factory::MCPClientFactory;
pub use sse::{SSEConnection, SSEMessage, SSEResponse, SSEStatus, SSEStreamReceiver};
pub use stdio::{
    StdioConnection, StdioConnectionBuilder, JsonRpcRequest, JsonRpcResponse, JsonRpcError,
};
pub use docker::{
    DockerConnection, DockerConnectionBuilder, DockerConfig, ContainerStatus, ContainerState,
};

use crate::core::error::{FlowgentraError, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio_util::sync::CancellationToken;
use tracing::Instrument;

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
    #[serde(default)]
    pub uri: String,
    /// For stdio: the command to run (e.g. "python")
    #[serde(default)]
    pub command: Option<String>,
    /// For stdio: arguments to the command (e.g. ["mcp_server.py"])
    #[serde(default)]
    pub args: Vec<String>,
    /// For docker: the image to run (e.g. "my-mcp-server:latest")
    #[serde(default)]
    pub image: Option<String>,
    /// Authentication credentials if needed
    #[serde(default)]
    pub auth: Option<MCPAuth>,
    /// Connection-specific settings
    #[serde(default)]
    pub connection_settings: MCPConnectionSettings,
    /// Optional namespace prefix for tools (e.g. "math_server" → tools become "math_server.calculate")
    #[serde(default)]
    pub namespace: Option<String>,
    /// If set, only expose tools whose names are in this list (applied after namespace prefix)
    #[serde(default)]
    pub tool_include: Option<Vec<String>>,
    /// If set, hide tools whose names are in this list (applied after namespace prefix)
    #[serde(default)]
    pub tool_exclude: Option<Vec<String>>,
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
            return Err(FlowgentraError::MCPError("MCP name cannot be empty".into()));
        }
        match self.connection_type {
            MCPConnectionType::Sse => {
                if self.uri.is_empty() {
                    return Err(FlowgentraError::MCPError("SSE connection requires a URI".into()));
                }
                if !self.uri.starts_with("http") {
                    return Err(FlowgentraError::MCPError(
                        "SSE connection requires HTTP(S) URI".into(),
                    ));
                }
            }
            MCPConnectionType::Stdio => {
                if self.command.is_none() && self.uri.is_empty() {
                    return Err(FlowgentraError::MCPError(
                        "Stdio connection requires a 'command' or 'uri' field".into(),
                    ));
                }
            }
            MCPConnectionType::Docker => {
                if self.image.is_none() && self.uri.is_empty() {
                    return Err(FlowgentraError::MCPError(
                        "Docker connection requires an 'image' or 'uri' field".into(),
                    ));
                }
            }
        }
        Ok(())
    }

    /// Get the effective command for stdio connections.
    /// Uses `command` field if present, otherwise falls back to `uri`.
    pub fn stdio_command(&self) -> &str {
        self.command.as_deref().unwrap_or(&self.uri)
    }

    /// Strip the namespace prefix from a tool name before sending to the server.
    pub fn strip_namespace<'a>(&self, tool_name: &'a str) -> &'a str {
        if let Some(ref ns) = self.namespace {
            let prefix = format!("{}.", ns);
            tool_name.strip_prefix(&prefix).unwrap_or(tool_name)
        } else {
            tool_name
        }
    }

    /// Check if a tool name is excluded by `tool_exclude`.
    pub fn is_tool_excluded(&self, tool_name: &str) -> bool {
        if let Some(ref exclude) = self.tool_exclude {
            let raw = self.strip_namespace(tool_name);
            exclude.iter().any(|e| e == raw || e == tool_name)
        } else {
            false
        }
    }
}

/// Builder for MCPConfig
pub struct MCPConfigBuilder {
    name: Option<String>,
    connection_type: Option<MCPConnectionType>,
    uri: Option<String>,
    command: Option<String>,
    args: Vec<String>,
    image: Option<String>,
    auth: Option<MCPAuth>,
    timeout_secs: Option<u64>,
    container_name: Option<String>,
    namespace: Option<String>,
    tool_include: Option<Vec<String>>,
    tool_exclude: Option<Vec<String>>,
}

impl MCPConfigBuilder {
    /// Create new builder
    pub fn new() -> Self {
        Self {
            name: None,
            connection_type: None,
            uri: None,
            command: None,
            args: Vec::new(),
            image: None,
            auth: None,
            timeout_secs: None,
            container_name: None,
            namespace: None,
            tool_include: None,
            tool_exclude: None,
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

    /// Configure Stdio connection with command (e.g. "python")
    pub fn stdio(mut self, command: impl Into<String>) -> Self {
        self.connection_type = Some(MCPConnectionType::Stdio);
        self.command = Some(command.into());
        self
    }

    /// Set command arguments (for stdio connections)
    pub fn args(mut self, args: Vec<String>) -> Self {
        self.args = args;
        self
    }

    /// Configure Docker connection with image name
    pub fn docker(mut self, image: impl Into<String>) -> Self {
        self.connection_type = Some(MCPConnectionType::Docker);
        self.image = Some(image.into());
        self
    }

    /// Set Docker container name
    pub fn container_name(mut self, name: impl Into<String>) -> Self {
        self.container_name = Some(name.into());
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

    /// Set namespace prefix for tools
    pub fn namespace(mut self, ns: impl Into<String>) -> Self {
        self.namespace = Some(ns.into());
        self
    }

    /// Set tool include filter
    pub fn tool_include(mut self, tools: Vec<String>) -> Self {
        self.tool_include = Some(tools);
        self
    }

    /// Set tool exclude filter
    pub fn tool_exclude(mut self, tools: Vec<String>) -> Self {
        self.tool_exclude = Some(tools);
        self
    }

    /// Build MCPConfig
    pub fn build(self) -> Result<MCPConfig> {
        let name = self
            .name
            .ok_or_else(|| FlowgentraError::ConfigError("MCP name is required".into()))?;
        let connection_type = self
            .connection_type
            .ok_or_else(|| FlowgentraError::ConfigError("MCP connection type is required".into()))?;
        let uri = self.uri.unwrap_or_default();

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
            command: self.command,
            args: self.args,
            image: self.image,
            auth: self.auth,
            connection_settings,
            namespace: self.namespace,
            tool_include: self.tool_include,
            tool_exclude: self.tool_exclude,
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
    /// General timeout in seconds (used as fallback for both connect and call)
    pub timeout: Option<u64>,
    /// Timeout for establishing a connection (seconds). Falls back to `timeout`.
    pub connect_timeout: Option<u64>,
    /// Timeout for individual tool calls (seconds). Falls back to `timeout`.
    pub call_timeout: Option<u64>,
    /// For Docker: container name (auto-generated if not set)
    pub container_name: Option<String>,
    /// For Docker: port inside the container (default: 8080)
    pub port: Option<u16>,
    /// For Docker: port on the host to bind to (auto-assigned if not set)
    pub host_port: Option<u16>,
    /// For Stdio: working directory for the subprocess
    pub working_dir: Option<String>,
    /// For Stdio/Docker: environment variables
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
    #[serde(default)]
    pub description: Option<String>,
    #[serde(alias = "inputSchema", default)]
    pub input_schema: serde_json::Value,
}

/// An MCP resource (per the MCP spec `resources/list` response).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MCPResource {
    /// Unique URI for this resource (e.g., "file:///path/to/file")
    pub uri: String,
    /// Human-readable name
    #[serde(default)]
    pub name: Option<String>,
    /// Description of what this resource contains
    #[serde(default)]
    pub description: Option<String>,
    /// MIME type of the resource content
    #[serde(alias = "mimeType", default)]
    pub mime_type: Option<String>,
}

/// Content of a read resource.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MCPResourceContent {
    pub uri: String,
    #[serde(alias = "mimeType", default)]
    pub mime_type: Option<String>,
    #[serde(default)]
    pub text: Option<String>,
    #[serde(default)]
    pub blob: Option<String>,
}

/// An MCP prompt template (per the MCP spec `prompts/list` response).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MCPPrompt {
    /// Unique name for this prompt
    pub name: String,
    /// Description of what the prompt does
    #[serde(default)]
    pub description: Option<String>,
    /// Arguments the prompt accepts
    #[serde(default)]
    pub arguments: Vec<MCPPromptArgument>,
}

/// An argument for an MCP prompt.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MCPPromptArgument {
    pub name: String,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default)]
    pub required: bool,
}

/// Result of getting a prompt (rendered messages).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MCPPromptResult {
    #[serde(default)]
    pub description: Option<String>,
    pub messages: Vec<MCPPromptMessage>,
}

/// A message in a rendered prompt result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MCPPromptMessage {
    pub role: String,
    pub content: serde_json::Value,
}

/// MCP protocol version supported by this implementation.
pub const MCP_PROTOCOL_VERSION: &str = "2024-11-05";

/// MCP Client trait
#[async_trait::async_trait]
pub trait MCPClient: Send + Sync {
    /// Perform the MCP initialize handshake (protocol version negotiation).
    /// Default implementation is a no-op returning the current protocol version.
    /// Stdio clients should override to send `initialize` JSON-RPC.
    async fn initialize(&self) -> Result<String> {
        Ok(MCP_PROTOCOL_VERSION.to_string())
    }

    /// List available tools
    async fn list_tools(&self) -> Result<Vec<MCPTool>>;

    /// Call an MCP tool
    async fn call_tool(
        &self,
        tool_name: &str,
        arguments: serde_json::Value,
    ) -> Result<serde_json::Value>;

    /// Call multiple tools in parallel and collect results.
    /// Default implementation fans out to `call_tool` concurrently.
    async fn call_tools_parallel(
        &self,
        calls: Vec<(String, serde_json::Value)>,
    ) -> Result<Vec<serde_json::Value>> {
        let futs: Vec<_> = calls
            .into_iter()
            .map(|(name, args)| {
                let name = name.clone();
                let args = args.clone();
                async move { self.call_tool(&name, args).await }
            })
            .collect();
        futures::future::try_join_all(futs).await
    }

    /// Check if the MCP server is reachable.
    /// Default implementation tries `list_tools()` and returns true on success.
    async fn health_check(&self) -> Result<bool> {
        match self.list_tools().await {
            Ok(_) => Ok(true),
            Err(_) => Ok(false),
        }
    }

    /// Call a tool and receive streaming chunks.
    /// Default implementation buffers the full result and sends it as one chunk.
    /// Clients that support streaming (e.g. SSE) can override this.
    async fn call_tool_streaming(
        &self,
        tool_name: &str,
        arguments: serde_json::Value,
    ) -> Result<tokio::sync::mpsc::UnboundedReceiver<Result<serde_json::Value>>> {
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        let result = self.call_tool(tool_name, arguments).await;
        let _ = tx.send(result);
        Ok(rx)
    }

    /// Call a tool with cancellation support.
    /// Default implementation races `call_tool` against the token.
    async fn call_tool_cancellable(
        &self,
        tool_name: &str,
        arguments: serde_json::Value,
        cancel: CancellationToken,
    ) -> Result<serde_json::Value> {
        tokio::select! {
            result = self.call_tool(tool_name, arguments) => result,
            _ = cancel.cancelled() => {
                Err(FlowgentraError::ExecutionAborted(
                    format!("MCP call to '{}' was cancelled", tool_name)
                ))
            }
        }
    }

    /// Gracefully shut down the connection (stop subprocess, remove container, etc.).
    /// Default implementation is a no-op for stateless HTTP clients.
    async fn shutdown(&self) -> Result<()> {
        Ok(())
    }

    // =========================================================================
    // Resources protocol (MCP spec: resources/list, resources/read)
    // =========================================================================

    /// List available resources from the MCP server.
    /// Default returns an empty list (server may not support resources).
    async fn list_resources(&self) -> Result<Vec<MCPResource>> {
        Ok(vec![])
    }

    /// Read a resource by URI.
    /// Default returns NotImplemented.
    async fn read_resource(&self, _uri: &str) -> Result<MCPResourceContent> {
        Err(FlowgentraError::MCPError(
            "resources/read not supported by this MCP client".to_string(),
        ))
    }

    // =========================================================================
    // Prompts protocol (MCP spec: prompts/list, prompts/get)
    // =========================================================================

    /// List available prompt templates from the MCP server.
    /// Default returns an empty list.
    async fn list_prompts(&self) -> Result<Vec<MCPPrompt>> {
        Ok(vec![])
    }

    /// Get a rendered prompt by name, with arguments.
    /// Default returns NotImplemented.
    async fn get_prompt(
        &self,
        _name: &str,
        _arguments: serde_json::Value,
    ) -> Result<MCPPromptResult> {
        Err(FlowgentraError::MCPError(
            "prompts/get not supported by this MCP client".to_string(),
        ))
    }
}

/// Apply namespace prefix and include/exclude filtering to a tool list.
pub fn apply_tool_filters(
    tools: Vec<MCPTool>,
    namespace: Option<&str>,
    include: Option<&[String]>,
    exclude: Option<&[String]>,
) -> Vec<MCPTool> {
    let mut result: Vec<MCPTool> = tools
        .into_iter()
        .map(|mut t| {
            if let Some(ns) = namespace {
                t.name = format!("{}.{}", ns, t.name);
            }
            t
        })
        .collect();

    if let Some(include) = include {
        let suffixes: Vec<String> = include.iter().map(|i| format!(".{}", i)).collect();
        result.retain(|t| {
            include.iter().zip(&suffixes).any(|(i, suf)| t.name == *i || t.name.ends_with(suf))
        });
    }
    if let Some(exclude) = exclude {
        let suffixes: Vec<String> = exclude.iter().map(|e| format!(".{}", e)).collect();
        result.retain(|t| {
            !exclude.iter().zip(&suffixes).any(|(e, suf)| t.name == *e || t.name.ends_with(suf))
        });
    }

    result
}

/// Apply auth headers from an MCPConfig to a request builder.
fn apply_auth(config: &MCPConfig, req: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
    match &config.auth {
        Some(auth) => match auth.auth_type.as_str() {
            "bearer" => {
                if let Some(token) = auth.credentials.get("token") {
                    return req.bearer_auth(token);
                }
                req
            }
            "api_key" => {
                let header = auth.credentials.get("header").map(|s| s.as_str()).unwrap_or("X-API-Key");
                if let Some(key) = auth.credentials.get("key") {
                    return req.header(header, key);
                }
                req
            }
            "basic" => {
                let user = auth.credentials.get("username").cloned().unwrap_or_default();
                let pass = auth.credentials.get("password").cloned().unwrap_or_default();
                req.basic_auth(user, Some(pass))
            }
            _ => {
                tracing::warn!(auth_type = %auth.auth_type, "Unknown auth type, skipping");
                req
            }
        },
        None => req,
    }
}

/// Merge tool lists from multiple MCP clients into a single list.
pub async fn merge_tool_lists(clients: &[Arc<dyn MCPClient>]) -> Result<Vec<MCPTool>> {
    let futs = clients.iter().map(|c| c.list_tools());
    let results = futures::future::try_join_all(futs).await?;
    Ok(results.into_iter().flatten().collect())
}

/// Classify a reqwest error into transport (retryable) vs other.
fn classify_reqwest_error(msg: String, e: &reqwest::Error) -> FlowgentraError {
    if e.is_timeout() || e.is_connect() || e.is_request() {
        FlowgentraError::MCPTransportError(msg)
    } else {
        FlowgentraError::MCPError(msg)
    }
}

/// Classify an HTTP status code into server (5xx) vs client error.
fn classify_http_status(status: reqwest::StatusCode, msg: &str) -> FlowgentraError {
    if status.is_server_error() {
        FlowgentraError::MCPServerError(msg.to_string())
    } else {
        FlowgentraError::MCPError(msg.to_string())
    }
}

/// Default MCP client implementation (HTTP/SSE)
pub struct DefaultMCPClient {
    config: MCPConfig,
    client: reqwest::Client,
}

impl DefaultMCPClient {
    /// Create new MCP client with given configuration
    pub fn new(config: MCPConfig) -> Self {
        let general = config.connection_settings.timeout.unwrap_or(30);
        let connect = config.connection_settings.connect_timeout.unwrap_or(general);
        let call = config.connection_settings.call_timeout.unwrap_or(general);

        DefaultMCPClient {
            client: reqwest::Client::builder()
                .connect_timeout(std::time::Duration::from_secs(connect))
                .timeout(std::time::Duration::from_secs(call))
                .build()
                .unwrap_or_default(),
            config,
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
        let url = format!("{}/tools", self.config.uri);
        tracing::debug!(url = %url, "Fetching available MCP tools");

        let req = apply_auth(&self.config, self.client.get(&url));
        let response = req.send().await.map_err(|e| {
            let err_msg = format!("Failed to list MCP tools: {}", e);
            tracing::error!(error = %e, mcp = %self.config.name, "MCP list_tools request failed");
            classify_reqwest_error(err_msg, &e)
        })?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(classify_http_status(status, &format!(
                "MCP server returned {}: {}", status, body
            )));
        }

        let tools: Vec<MCPTool> = response.json().await.map_err(|e| {
            let err_msg = format!("Failed to parse MCP response: {}", e);
            tracing::error!(error = %e, mcp = %self.config.name, "MCP response parsing failed");
            FlowgentraError::MCPError(err_msg)
        })?;

        // Apply namespace and filtering
        let tools = apply_tool_filters(
            tools,
            self.config.namespace.as_deref(),
            self.config.tool_include.as_deref(),
            self.config.tool_exclude.as_deref(),
        );

        tracing::info!(tool_count = tools.len(), "Successfully retrieved MCP tools");
        Ok(tools)
    }

    async fn call_tool(
        &self,
        tool_name: &str,
        arguments: serde_json::Value,
    ) -> Result<serde_json::Value> {
        // Enforce tool_exclude at call time
        if self.config.is_tool_excluded(tool_name) {
            return Err(FlowgentraError::ToolError(format!(
                "Tool '{}' is excluded by configuration", tool_name
            )));
        }

        let raw_name = self.config.strip_namespace(tool_name);

        async {
            let payload = serde_json::json!({
                "tool": raw_name,
                "arguments": arguments
            });

            tracing::debug!("Calling MCP tool");

            let url = format!("{}/call", self.config.uri);
            let req = apply_auth(&self.config, self.client.post(&url).json(&payload));
            let response = req
                .send()
                .await
                .map_err(|e| {
                    let err_msg = format!("Failed to call MCP tool '{}': {}", tool_name, e);
                    tracing::error!(tool = %tool_name, error = %e, "MCP call failed");
                    classify_reqwest_error(err_msg, &e)
                })?;

            if !response.status().is_success() {
                let status = response.status();
                let body = response.text().await.unwrap_or_default();
                return Err(classify_http_status(status, &format!(
                    "MCP tool '{}' returned {}: {}", tool_name, status, body
                )));
            }

            let result = response.json().await.map_err(|e| {
                let err_msg = format!("Failed to parse MCP tool response: {}", e);
                tracing::error!(tool = %tool_name, error = %e, "MCP response parsing failed");
                FlowgentraError::MCPError(err_msg)
            })?;

            tracing::debug!("MCP tool call completed");
            Ok(result)
        }
        .instrument(tracing::info_span!("mcp_call_tool", mcp = %self.config.name, tool = %tool_name))
        .await
    }

    async fn list_resources(&self) -> Result<Vec<MCPResource>> {
        let url = format!("{}/resources", self.config.uri);
        let req = apply_auth(&self.config, self.client.get(&url));
        match req.send().await {
            Ok(resp) if resp.status().is_success() => {
                resp.json().await.map_err(|e| {
                    FlowgentraError::MCPError(format!("Failed to parse resources: {}", e))
                })
            }
            Ok(resp) if resp.status().as_u16() == 404 => Ok(vec![]),
            Ok(resp) => Err(FlowgentraError::MCPError(format!(
                "resources/list returned {}", resp.status()
            ))),
            Err(e) => Err(classify_reqwest_error(format!("resources/list failed: {}", e), &e)),
        }
    }

    async fn read_resource(&self, uri: &str) -> Result<MCPResourceContent> {
        let url = format!("{}/resources/read", self.config.uri);
        let payload = serde_json::json!({ "uri": uri });
        let req = apply_auth(&self.config, self.client.post(&url).json(&payload));
        let resp = req.send().await.map_err(|e| {
            classify_reqwest_error(format!("resources/read failed: {}", e), &e)
        })?;
        if !resp.status().is_success() {
            return Err(FlowgentraError::MCPError(format!(
                "resources/read returned {}", resp.status()
            )));
        }
        resp.json().await.map_err(|e| {
            FlowgentraError::MCPError(format!("Failed to parse resource content: {}", e))
        })
    }

    async fn list_prompts(&self) -> Result<Vec<MCPPrompt>> {
        let url = format!("{}/prompts", self.config.uri);
        let req = apply_auth(&self.config, self.client.get(&url));
        match req.send().await {
            Ok(resp) if resp.status().is_success() => {
                resp.json().await.map_err(|e| {
                    FlowgentraError::MCPError(format!("Failed to parse prompts: {}", e))
                })
            }
            Ok(resp) if resp.status().as_u16() == 404 => Ok(vec![]),
            Ok(resp) => Err(FlowgentraError::MCPError(format!(
                "prompts/list returned {}", resp.status()
            ))),
            Err(e) => Err(classify_reqwest_error(format!("prompts/list failed: {}", e), &e)),
        }
    }

    async fn get_prompt(&self, name: &str, arguments: serde_json::Value) -> Result<MCPPromptResult> {
        let url = format!("{}/prompts/get", self.config.uri);
        let payload = serde_json::json!({ "name": name, "arguments": arguments });
        let req = apply_auth(&self.config, self.client.post(&url).json(&payload));
        let resp = req.send().await.map_err(|e| {
            classify_reqwest_error(format!("prompts/get failed: {}", e), &e)
        })?;
        if !resp.status().is_success() {
            return Err(FlowgentraError::MCPError(format!(
                "prompts/get returned {}", resp.status()
            )));
        }
        resp.json().await.map_err(|e| {
            FlowgentraError::MCPError(format!("Failed to parse prompt result: {}", e))
        })
    }
}

/// MCP client that communicates with a subprocess over JSON-RPC stdin/stdout
pub struct StdioMCPClient {
    config: MCPConfig,
    connection: StdioConnection,
    started: tokio::sync::Mutex<bool>,
}

impl StdioMCPClient {
    /// Create a new stdio MCP client from config
    pub fn new(config: MCPConfig) -> Self {
        let command = config.stdio_command().to_string();
        let args = config.args.clone();
        let env_vars = config.connection_settings.env_vars.clone();
        let timeout_secs = config.connection_settings.timeout.unwrap_or(30);

        let mut conn = StdioConnection::new(&command, args);
        conn = conn.with_timeout(std::time::Duration::from_secs(timeout_secs));
        if let Some(ref dir) = config.connection_settings.working_dir {
            conn = conn.with_working_dir(dir);
        }
        for (k, v) in &env_vars {
            conn = conn.with_env(k, v);
        }

        StdioMCPClient {
            config,
            connection: conn,
            started: tokio::sync::Mutex::new(false),
        }
    }

    /// Ensure the subprocess is started (and still alive).
    /// If it crashed or was killed (e.g. after a timeout), restart it.
    async fn ensure_started(&self) -> Result<()> {
        let mut started = self.started.lock().await;
        if *started {
            // Check if the process is still alive
            if !self.connection.is_running().await {
                tracing::warn!(mcp = %self.config.name, "Stdio subprocess died, restarting");
                *started = false;
            }
        }
        if !*started {
            self.connection.start().await?;
            *started = true;
        }
        Ok(())
    }
}

#[async_trait::async_trait]
impl MCPClient for StdioMCPClient {
    async fn initialize(&self) -> Result<String> {
        self.ensure_started().await?;

        let params = serde_json::json!({
            "protocolVersion": MCP_PROTOCOL_VERSION,
            "capabilities": {},
            "clientInfo": {
                "name": "flowgentra-ai",
                "version": env!("CARGO_PKG_VERSION"),
            }
        });

        let result = self.connection.call("initialize", params).await?;
        let server_version = result
            .get("protocolVersion")
            .and_then(|v| v.as_str())
            .unwrap_or(MCP_PROTOCOL_VERSION)
            .to_string();

        tracing::info!(
            mcp = %self.config.name,
            server_version = %server_version,
            "MCP initialize handshake completed"
        );

        // Send initialized notification (no response expected, but still use call)
        let _ = self.connection.call("notifications/initialized", serde_json::json!({})).await;

        Ok(server_version)
    }

    async fn list_tools(&self) -> Result<Vec<MCPTool>> {
        self.ensure_started().await?;

        tracing::debug!(mcp = %self.config.name, "Listing tools via stdio");
        let result = self.connection.call("tools/list", serde_json::json!({})).await?;

        // Parse the response — expect {"tools": [...]} or just [...]
        let tools_value = if let Some(tools) = result.get("tools") {
            tools.clone()
        } else if result.is_array() {
            result
        } else {
            return Err(FlowgentraError::MCPError(format!(
                "Unexpected list_tools response: {}", result
            )));
        };

        let tools: Vec<MCPTool> = serde_json::from_value(tools_value).map_err(|e| {
            FlowgentraError::MCPError(format!("Failed to parse tools: {}", e))
        })?;

        let tools = apply_tool_filters(
            tools,
            self.config.namespace.as_deref(),
            self.config.tool_include.as_deref(),
            self.config.tool_exclude.as_deref(),
        );

        tracing::info!(mcp = %self.config.name, tool_count = tools.len(), "Listed stdio MCP tools");
        Ok(tools)
    }

    async fn call_tool(
        &self,
        tool_name: &str,
        arguments: serde_json::Value,
    ) -> Result<serde_json::Value> {
        self.ensure_started().await?;

        // Enforce tool_exclude at call time
        if self.config.is_tool_excluded(tool_name) {
            return Err(FlowgentraError::ToolError(format!(
                "Tool '{}' is excluded by configuration", tool_name
            )));
        }

        let raw_name = self.config.strip_namespace(tool_name);

        tracing::debug!(mcp = %self.config.name, tool = %tool_name, "Calling tool via stdio");
        let params = serde_json::json!({
            "name": raw_name,
            "arguments": arguments,
        });

        let result = self.connection.call("tools/call", params).await?;

        // MCP spec: result has "content" array with {type, text} items
        if let Some(content) = result.get("content") {
            if let Some(arr) = content.as_array() {
                // Extract text from first text content block
                for item in arr {
                    if item.get("type").and_then(|t| t.as_str()) == Some("text") {
                        if let Some(text) = item.get("text").and_then(|t| t.as_str()) {
                            // Try to parse as JSON, otherwise return as string
                            if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(text) {
                                return Ok(parsed);
                            }
                            return Ok(serde_json::json!(text));
                        }
                    }
                }
            }
        }

        Ok(result)
    }

    async fn shutdown(&self) -> Result<()> {
        let mut started = self.started.lock().await;
        if *started {
            self.connection.stop().await?;
            *started = false;
        }
        Ok(())
    }
}

impl Drop for StdioMCPClient {
    fn drop(&mut self) {
        let started = self.started.get_mut();
        if *started {
            tracing::info!(mcp = %self.config.name, "Killing stdio MCP subprocess on drop");
            // Best-effort: try to get the process lock synchronously via get_mut
            // (only works if no async task holds the lock, which is expected during drop)
            if let Some(mut proc) = self.connection.process.try_lock().ok().and_then(|mut g| g.take()) {
                // Start kill but can't await it in Drop — fire and forget
                let _ = proc.child.start_kill();
            }
        }
    }
}

/// MCP client backed by SSE streaming.
///
/// Uses the underlying `SSEConnection` for streaming responses while
/// conforming to the `MCPClient` trait. `call_tool` collects the
/// full streamed response; `call_tool_streaming` yields chunks.
pub struct SSEMCPClient {
    config: MCPConfig,
    connection: SSEConnection,
}

impl SSEMCPClient {
    pub fn new(config: MCPConfig) -> Self {
        let timeout_secs = config.connection_settings.timeout.unwrap_or(30);
        let connection = SSEConnection::new(config.uri.clone(), timeout_secs);
        Self { config, connection }
    }
}

#[async_trait::async_trait]
impl MCPClient for SSEMCPClient {
    async fn list_tools(&self) -> Result<Vec<MCPTool>> {
        let mut stream = self.connection.stream(
            "tools/list",
            serde_json::json!({}),
        ).await?;

        let mut tools = Vec::new();
        while let Some(result) = stream.next().await {
            let msg = result?;
            if let Ok(batch) = serde_json::from_value::<Vec<MCPTool>>(msg.data.clone()) {
                tools.extend(batch);
            } else if let Ok(tool) = serde_json::from_value::<MCPTool>(msg.data) {
                tools.push(tool);
            }
        }

        let tools = apply_tool_filters(
            tools,
            self.config.namespace.as_deref(),
            self.config.tool_include.as_deref(),
            self.config.tool_exclude.as_deref(),
        );

        Ok(tools)
    }

    async fn call_tool(
        &self,
        tool_name: &str,
        arguments: serde_json::Value,
    ) -> Result<serde_json::Value> {
        if self.config.is_tool_excluded(tool_name) {
            return Err(FlowgentraError::ToolError(format!(
                "Tool '{}' is excluded by configuration", tool_name
            )));
        }

        let raw_name = self.config.strip_namespace(tool_name);
        let payload = serde_json::json!({
            "tool": raw_name,
            "arguments": arguments,
        });

        let mut stream = self.connection.stream("tools/call", payload).await?;

        // Collect all streamed chunks into a single result
        let mut result = serde_json::Value::Null;
        while let Some(msg_result) = stream.next().await {
            let msg = msg_result?;
            result = msg.data;
        }

        Ok(result)
    }

    async fn call_tool_streaming(
        &self,
        tool_name: &str,
        arguments: serde_json::Value,
    ) -> Result<tokio::sync::mpsc::UnboundedReceiver<Result<serde_json::Value>>> {
        if self.config.is_tool_excluded(tool_name) {
            return Err(FlowgentraError::ToolError(format!(
                "Tool '{}' is excluded by configuration", tool_name
            )));
        }

        let raw_name = self.config.strip_namespace(tool_name);
        let payload = serde_json::json!({
            "tool": raw_name,
            "arguments": arguments,
        });

        let mut stream = self.connection.stream("tools/call", payload).await?;
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();

        tokio::spawn(async move {
            while let Some(msg_result) = stream.next().await {
                let result = msg_result.map(|msg| msg.data);
                if tx.send(result).is_err() {
                    break;
                }
            }
        });

        Ok(rx)
    }

    async fn shutdown(&self) -> Result<()> {
        self.connection.close_all().await;
        Ok(())
    }
}

/// MCP client that manages a Docker container and communicates via HTTP.
///
/// On first use, the client:
/// 1. Pulls the image (if not already present)
/// 2. Starts a container with the configured port mapping
/// 3. Waits for the HTTP server inside to become healthy
/// 4. Communicates via HTTP GET/POST (same as SSE/DefaultMCPClient)
///
/// The container is stopped and removed when the client is dropped.
pub struct DockerMCPClient {
    config: MCPConfig,
    client: reqwest::Client,
    container_name: String,
    /// Resolved host port — set once during initialization via OnceCell.
    /// OnceCell ensures only one caller performs init; others await it
    /// WITHOUT holding a mutex across the 30s startup.
    host_port: tokio::sync::OnceCell<u16>,
    /// The configured host port (0 = auto-assign).
    configured_host_port: u16,
}

impl DockerMCPClient {
    /// Create a new Docker MCP client from config.
    pub fn new(config: MCPConfig) -> Self {
        let container_name = config.connection_settings.container_name
            .clone()
            .unwrap_or_else(|| format!("flowgentra-mcp-{}", &config.name));
        let configured_host_port = config.connection_settings.host_port.unwrap_or(0);
        let general = config.connection_settings.timeout.unwrap_or(30);
        let connect = config.connection_settings.connect_timeout.unwrap_or(general);
        let call = config.connection_settings.call_timeout.unwrap_or(general);

        DockerMCPClient {
            config,
            client: reqwest::Client::builder()
                .connect_timeout(std::time::Duration::from_secs(connect))
                .timeout(std::time::Duration::from_secs(call))
                .build()
                .unwrap_or_default(),
            container_name,
            host_port: tokio::sync::OnceCell::new(),
            configured_host_port,
        }
    }

    /// Ensure the container is running. Uses OnceCell so the long startup
    /// only happens once and concurrent callers wait without holding a mutex.
    async fn ensure_started(&self) -> Result<()> {
        self.host_port.get_or_try_init(|| async {
            self.do_start().await
        }).await?;
        Ok(())
    }

    /// Perform the actual container startup (called at most once).
    async fn do_start(&self) -> Result<u16> {
        let image = self.config.image.as_deref().ok_or_else(|| {
            FlowgentraError::MCPError("Docker MCP requires an 'image' field".into())
        })?;
        let container_port = self.config.connection_settings.port.unwrap_or(8080);

        // If host_port is 0, pick a random free port
        let host_port = if self.configured_host_port == 0 {
            let listener = std::net::TcpListener::bind("127.0.0.1:0").map_err(|e| {
                FlowgentraError::MCPError(format!("Failed to find free port: {}", e))
            })?;
            let port = listener.local_addr().unwrap().port();
            drop(listener);
            port
        } else {
            self.configured_host_port
        };

        tracing::info!(
            image = %image,
            container = %self.container_name,
            port = %format!("{}:{}", host_port, container_port),
            "Starting Docker MCP container"
        );

        // Remove any existing container with the same name (ignore errors)
        let _ = tokio::process::Command::new("docker")
            .args(["rm", "-f", &self.container_name])
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status()
            .await;

        // Build docker run command
        let mut args = vec![
            "run".to_string(),
            "-d".to_string(),
            "--name".to_string(),
            self.container_name.clone(),
            "-p".to_string(),
            format!("{}:{}", host_port, container_port),
        ];

        // Add environment variables
        for (k, v) in &self.config.connection_settings.env_vars {
            args.push("-e".to_string());
            args.push(format!("{}={}", k, v));
        }

        args.push(image.to_string());

        let output = tokio::process::Command::new("docker")
            .args(&args)
            .output()
            .await
            .map_err(|e| {
                FlowgentraError::MCPError(format!("Failed to run docker: {}", e))
            })?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(FlowgentraError::MCPError(format!(
                "docker run failed: {}", stderr.trim()
            )));
        }

        let container_id = String::from_utf8_lossy(&output.stdout).trim().to_string();
        tracing::info!(container_id = %container_id, "Container started, waiting for health...");

        // Wait for the HTTP server inside to become ready
        let url = format!("http://127.0.0.1:{}/tools", host_port);
        let timeout_secs = self.config.connection_settings.timeout.unwrap_or(30);
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(timeout_secs);

        loop {
            if std::time::Instant::now() > deadline {
                // Cleanup on timeout
                let _ = tokio::process::Command::new("docker")
                    .args(["rm", "-f", &self.container_name])
                    .status()
                    .await;
                return Err(FlowgentraError::MCPError(format!(
                    "Container did not become healthy within {}s", timeout_secs
                )));
            }

            match self.client.get(&url).send().await {
                Ok(resp) if resp.status().is_success() => break,
                _ => tokio::time::sleep(std::time::Duration::from_millis(500)).await,
            }
        }

        tracing::info!(host_port = %host_port, "Docker MCP container ready");
        Ok(host_port)
    }

    /// Get the base URL for HTTP calls to the container.
    fn base_url(&self) -> String {
        let port = self.host_port.get().copied().unwrap_or(0);
        format!("http://127.0.0.1:{}", port)
    }
}

#[async_trait::async_trait]
impl MCPClient for DockerMCPClient {
    async fn list_tools(&self) -> Result<Vec<MCPTool>> {
        self.ensure_started().await?;

        let url = format!("{}/tools", self.base_url());
        tracing::debug!(mcp = %self.config.name, url = %url, "Listing tools via Docker");

        let req = apply_auth(&self.config, self.client.get(&url));
        let response = req.send().await.map_err(|e| {
            let msg = format!("Failed to list Docker MCP tools: {}", e);
            classify_reqwest_error(msg, &e)
        })?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(classify_http_status(status, &format!(
                "Docker MCP server returned {}: {}", status, body
            )));
        }

        let tools: Vec<MCPTool> = response.json().await.map_err(|e| {
            FlowgentraError::MCPError(format!("Failed to parse Docker MCP tools: {}", e))
        })?;

        let tools = apply_tool_filters(
            tools,
            self.config.namespace.as_deref(),
            self.config.tool_include.as_deref(),
            self.config.tool_exclude.as_deref(),
        );

        tracing::info!(mcp = %self.config.name, tool_count = tools.len(), "Listed Docker MCP tools");
        Ok(tools)
    }

    async fn call_tool(
        &self,
        tool_name: &str,
        arguments: serde_json::Value,
    ) -> Result<serde_json::Value> {
        self.ensure_started().await?;

        // Enforce tool_exclude at call time
        if self.config.is_tool_excluded(tool_name) {
            return Err(FlowgentraError::ToolError(format!(
                "Tool '{}' is excluded by configuration", tool_name
            )));
        }

        let raw_name = self.config.strip_namespace(tool_name);

        let url = format!("{}/call", self.base_url());
        let payload = serde_json::json!({
            "tool": raw_name,
            "arguments": arguments,
        });

        tracing::debug!(mcp = %self.config.name, tool = %tool_name, "Calling tool via Docker");

        let req = apply_auth(&self.config, self.client.post(&url).json(&payload));
        let response = req.send().await.map_err(|e| {
            let msg = format!("Failed to call Docker MCP tool '{}': {}", tool_name, e);
            classify_reqwest_error(msg, &e)
        })?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(classify_http_status(status, &format!(
                "Docker MCP tool '{}' returned {}: {}", tool_name, status, body
            )));
        }

        let result = response.json().await.map_err(|e| {
            FlowgentraError::MCPError(format!("Failed to parse Docker MCP response: {}", e))
        })?;

        Ok(result)
    }

    async fn shutdown(&self) -> Result<()> {
        if self.host_port.get().is_some() {
            tracing::info!(container = %self.container_name, "Shutting down Docker MCP container");
            let _ = tokio::process::Command::new("docker")
                .args(["rm", "-f", &self.container_name])
                .stdout(std::process::Stdio::null())
                .stderr(std::process::Stdio::null())
                .status()
                .await;
        }
        Ok(())
    }
}

impl Drop for DockerMCPClient {
    fn drop(&mut self) {
        if self.host_port.get().is_some() {
            let name = self.container_name.clone();
            tracing::info!(container = %name, "Stopping Docker MCP container");
            std::thread::spawn(move || {
                let _ = std::process::Command::new("docker")
                    .args(["rm", "-f", &name])
                    .stdout(std::process::Stdio::null())
                    .stderr(std::process::Stdio::null())
                    .status();
            });
        }
    }
}

/// Circuit breaker state for `RetryMCPClient`.
struct CircuitBreakerState {
    consecutive_failures: u32,
    opened_at: Option<std::time::Instant>,
}

/// MCP client wrapper that adds:
/// - Retry with exponential backoff (transport errors only)
/// - Tool list caching with TTL
/// - Circuit breaker (opens after N consecutive failures, resets after cooldown)
///
/// Wraps any `MCPClient` implementation. The factory applies this automatically.
pub struct RetryMCPClient {
    inner: Arc<dyn MCPClient>,
    max_retries: u32,
    cached_tools: tokio::sync::Mutex<Option<(Vec<MCPTool>, std::time::Instant)>>,
    cache_ttl: std::time::Duration,
    circuit: tokio::sync::Mutex<CircuitBreakerState>,
    /// Number of consecutive failures before the circuit opens.
    circuit_threshold: u32,
    /// How long the circuit stays open before allowing a probe request.
    circuit_cooldown: std::time::Duration,
}

impl RetryMCPClient {
    /// Wrap an existing MCP client with retry, caching, and circuit breaker.
    pub fn new(inner: Arc<dyn MCPClient>, max_retries: u32) -> Self {
        Self {
            inner,
            max_retries,
            cached_tools: tokio::sync::Mutex::new(None),
            cache_ttl: std::time::Duration::from_secs(300), // 5 min default
            circuit: tokio::sync::Mutex::new(CircuitBreakerState {
                consecutive_failures: 0,
                opened_at: None,
            }),
            circuit_threshold: 5,
            circuit_cooldown: std::time::Duration::from_secs(30),
        }
    }

    /// Create with custom cache TTL and circuit breaker settings.
    pub fn with_settings(
        inner: Arc<dyn MCPClient>,
        max_retries: u32,
        cache_ttl: std::time::Duration,
        circuit_threshold: u32,
        circuit_cooldown: std::time::Duration,
    ) -> Self {
        Self {
            inner,
            max_retries,
            cached_tools: tokio::sync::Mutex::new(None),
            cache_ttl,
            circuit: tokio::sync::Mutex::new(CircuitBreakerState {
                consecutive_failures: 0,
                opened_at: None,
            }),
            circuit_threshold,
            circuit_cooldown,
        }
    }

    /// Invalidate the cached tool list so the next `list_tools()` hits the server.
    pub async fn invalidate_cache(&self) {
        let mut cache = self.cached_tools.lock().await;
        *cache = None;
    }

    /// Check circuit breaker state. Returns Err if the circuit is open.
    async fn check_circuit(&self) -> Result<()> {
        let mut cb = self.circuit.lock().await;
        if let Some(opened_at) = cb.opened_at {
            if opened_at.elapsed() < self.circuit_cooldown {
                return Err(FlowgentraError::MCPTransportError(
                    "Circuit breaker is open — MCP server appears down".into()
                ));
            }
            // Cooldown elapsed — allow one probe (half-open)
            tracing::info!("Circuit breaker half-open, allowing probe request");
            cb.opened_at = None;
            cb.consecutive_failures = 0;
        }
        Ok(())
    }

    /// Record a successful call — resets the circuit breaker.
    async fn record_success(&self) {
        let mut cb = self.circuit.lock().await;
        cb.consecutive_failures = 0;
        cb.opened_at = None;
    }

    /// Record a failed call — may open the circuit.
    async fn record_failure(&self) {
        let mut cb = self.circuit.lock().await;
        cb.consecutive_failures += 1;
        if cb.consecutive_failures >= self.circuit_threshold {
            tracing::error!(
                failures = cb.consecutive_failures,
                "Circuit breaker opened after {} consecutive failures", cb.consecutive_failures
            );
            cb.opened_at = Some(std::time::Instant::now());
        }
    }

    /// Execute an async operation with exponential backoff retry.
    /// Only retries if the error is classified as retryable (transport-level).
    async fn with_retry<F, Fut, T>(&self, op_name: &str, f: F) -> Result<T>
    where
        F: Fn() -> Fut,
        Fut: std::future::Future<Output = Result<T>>,
    {
        self.check_circuit().await?;

        let mut last_err = None;
        for attempt in 0..=self.max_retries {
            match f().await {
                Ok(val) => {
                    self.record_success().await;
                    return Ok(val);
                }
                Err(e) => {
                    if !e.is_retryable() || attempt == self.max_retries {
                        self.record_failure().await;
                        return Err(e);
                    }
                    last_err = Some(e);
                    let delay_ms = 100 * 2u64.pow(attempt);
                    tracing::warn!(
                        op = %op_name,
                        attempt = attempt + 1,
                        max = self.max_retries,
                        delay_ms = delay_ms,
                        "MCP call failed, retrying"
                    );
                    tokio::time::sleep(std::time::Duration::from_millis(delay_ms)).await;
                }
            }
        }
        self.record_failure().await;
        Err(last_err.unwrap())
    }
}

#[async_trait::async_trait]
impl MCPClient for RetryMCPClient {
    async fn list_tools(&self) -> Result<Vec<MCPTool>> {
        // Return cached tools if still within TTL
        {
            let cache = self.cached_tools.lock().await;
            if let Some((ref tools, cached_at)) = *cache {
                if cached_at.elapsed() < self.cache_ttl {
                    return Ok(tools.clone());
                }
            }
        }

        let inner = self.inner.clone();
        let tools: Vec<MCPTool> = self.with_retry("list_tools", || {
            let inner = inner.clone();
            async move { inner.list_tools().await }
        }).await?;

        // Cache with timestamp
        {
            let mut cache = self.cached_tools.lock().await;
            *cache = Some((tools.clone(), std::time::Instant::now()));
        }

        Ok(tools)
    }

    async fn call_tool(
        &self,
        tool_name: &str,
        arguments: serde_json::Value,
    ) -> Result<serde_json::Value> {
        let inner = self.inner.clone();
        let name = tool_name.to_string();
        let args = arguments.clone();
        self.with_retry(&format!("call_tool:{}", tool_name), move || {
            let inner = inner.clone();
            let name = name.clone();
            let args = args.clone();
            async move { inner.call_tool(&name, args).await }
        }).await
    }

    async fn shutdown(&self) -> Result<()> {
        self.inner.shutdown().await
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

// =============================================================================
// Reconnecting MCP Client
// =============================================================================

/// MCP client wrapper that recreates the inner client on connection failure.
///
/// When an HTTP/SSE MCP server goes down and comes back, the `ReconnectingMCPClient`
/// detects the failure and uses a factory to create a fresh client for the next attempt.
///
/// # Example
/// ```ignore
/// let config = MCPConfig::builder().name("tools").http("http://localhost:8000").build()?;
/// let client = ReconnectingMCPClient::new(move || {
///     Arc::new(DefaultMCPClient::new(config.clone())) as Arc<dyn MCPClient>
/// });
/// ```
pub struct ReconnectingMCPClient<F: Fn() -> Arc<dyn MCPClient> + Send + Sync> {
    factory: F,
    inner: tokio::sync::Mutex<Arc<dyn MCPClient>>,
    max_reconnects: u32,
}

impl<F: Fn() -> Arc<dyn MCPClient> + Send + Sync> ReconnectingMCPClient<F> {
    pub fn new(factory: F) -> Self {
        let inner = (factory)();
        Self {
            factory,
            inner: tokio::sync::Mutex::new(inner),
            max_reconnects: 3,
        }
    }

    pub fn with_max_reconnects(mut self, max: u32) -> Self {
        self.max_reconnects = max;
        self
    }

    /// Try the operation; on connection error, recreate the client and retry.
    async fn with_reconnect<T, Fut>(
        &self,
        op: impl Fn(Arc<dyn MCPClient>) -> Fut,
    ) -> Result<T>
    where
        Fut: std::future::Future<Output = Result<T>>,
    {
        let client = self.inner.lock().await.clone();
        match op(client).await {
            Ok(val) => Ok(val),
            Err(e) if is_connection_error(&e) => {
                tracing::warn!("MCP connection error, attempting reconnect: {}", e);
                for attempt in 1..=self.max_reconnects {
                    let new_client = (self.factory)();
                    *self.inner.lock().await = new_client.clone();
                    tracing::info!(attempt, "Reconnected MCP client");
                    match op(new_client).await {
                        Ok(val) => return Ok(val),
                        Err(e) if is_connection_error(&e) && attempt < self.max_reconnects => {
                            tokio::time::sleep(std::time::Duration::from_millis(500 * attempt as u64)).await;
                            continue;
                        }
                        Err(e) => return Err(e),
                    }
                }
                Err(e)
            }
            Err(e) => Err(e),
        }
    }
}

fn is_connection_error(err: &FlowgentraError) -> bool {
    let msg = err.to_string().to_lowercase();
    msg.contains("connection") || msg.contains("timeout") || msg.contains("refused")
        || msg.contains("reset") || msg.contains("broken pipe")
}

#[async_trait::async_trait]
impl<F: Fn() -> Arc<dyn MCPClient> + Send + Sync> MCPClient for ReconnectingMCPClient<F> {
    async fn list_tools(&self) -> Result<Vec<MCPTool>> {
        self.with_reconnect(|c| async move { c.list_tools().await }).await
    }

    async fn call_tool(&self, name: &str, args: serde_json::Value) -> Result<serde_json::Value> {
        let name = name.to_string();
        self.with_reconnect(move |c| {
            let n = name.clone();
            let a = args.clone();
            async move { c.call_tool(&n, a).await }
        }).await
    }
}
