//! Factory for creating MCP clients based on connection type

use super::{
    DefaultMCPClient, DockerMCPClient, MCPClient, MCPConfig, MCPConnectionType,
    MCPSseProtocolClient, RetryMCPClient, StdioMCPClient,
};
use crate::core::error::Result;
use std::sync::Arc;

/// Factory for creating MCP clients based on connection type.
///
/// Every client is automatically wrapped with `RetryMCPClient` which provides:
/// - Exponential backoff retry (configurable via `max_retries` in connection_settings)
/// - Tool list caching (first `list_tools()` call is cached for the lifetime of the client)
pub struct MCPClientFactory;

impl MCPClientFactory {
    /// Create an MCP client based on the connection type in the config.
    /// The returned client is wrapped with retry and caching.
    pub fn create(config: MCPConfig) -> Result<Arc<dyn MCPClient>> {
        config.validate()?;

        let max_retries = config.connection_settings.max_retries.unwrap_or(2);

        let inner: Arc<dyn MCPClient> = match config.connection_type {
            MCPConnectionType::Sse => {
                tracing::info!(name = %config.name, uri = %config.uri, "Creating SSE MCP client");
                Arc::new(MCPSseProtocolClient::new(config))
            }
            MCPConnectionType::Stdio => {
                tracing::info!(name = %config.name, command = %config.stdio_command(), "Creating Stdio MCP client");
                Arc::new(StdioMCPClient::new(config))
            }
            MCPConnectionType::Docker => {
                let image = config.image.as_deref().unwrap_or("(uri)");
                tracing::info!(name = %config.name, image = %image, "Creating Docker MCP client");
                if config.image.is_some() {
                    Arc::new(DockerMCPClient::new(config))
                } else {
                    Arc::new(DefaultMCPClient::new(config))
                }
            }
        };

        Ok(Arc::new(RetryMCPClient::new(inner, max_retries)))
    }

    /// Create multiple MCP clients from a list of configs
    pub fn create_multiple(configs: Vec<MCPConfig>) -> Result<Vec<Arc<dyn MCPClient>>> {
        configs.into_iter().map(Self::create).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn create_sse_client() {
        let config = MCPConfig::builder()
            .name("test_sse")
            .sse("http://localhost:8000")
            .build()
            .unwrap();

        let result = MCPClientFactory::create(config);
        assert!(result.is_ok());
    }

    #[test]
    fn create_stdio_client() {
        let config = MCPConfig::builder()
            .name("test_stdio")
            .stdio("python /path/to/tool.py")
            .build()
            .unwrap();

        let result = MCPClientFactory::create(config);
        assert!(result.is_ok());
    }

    #[test]
    fn create_docker_client() {
        let config = MCPConfig::builder()
            .name("test_docker")
            .docker("my-tool-container")
            .build()
            .unwrap();

        let result = MCPClientFactory::create(config);
        assert!(result.is_ok());
    }

    #[test]
    fn create_multiple_clients() {
        let configs = vec![
            MCPConfig::builder()
                .name("mcp1")
                .sse("http://localhost:8000")
                .build()
                .unwrap(),
            MCPConfig::builder()
                .name("mcp2")
                .stdio("python /path/to/tool.py")
                .build()
                .unwrap(),
        ];

        let result = MCPClientFactory::create_multiple(configs);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 2);
    }
}
