//! Stdio Protocol Handler for MCP
//!
//! Implements communication with MCP servers running as local processes
//! via JSON-RPC over stdin/stdout.

use crate::core::error::{FlowgentraError, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::Child as TokioChild;
use tokio::sync::Mutex;

/// JSON-RPC 2.0 Request structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonRpcRequest {
    pub jsonrpc: String,
    pub method: String,
    pub params: serde_json::Value,
    pub id: String,
}

impl JsonRpcRequest {
    /// Create a new JSON-RPC request
    pub fn new(method: impl Into<String>, params: serde_json::Value) -> Self {
        Self {
            jsonrpc: "2.0".to_string(),
            method: method.into(),
            params,
            id: uuid::Uuid::new_v4().to_string(),
        }
    }
}

/// JSON-RPC 2.0 Response structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonRpcResponse {
    pub jsonrpc: String,
    pub result: Option<serde_json::Value>,
    pub error: Option<JsonRpcError>,
    pub id: String,
}

/// JSON-RPC error response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonRpcError {
    pub code: i32,
    pub message: String,
    pub data: Option<serde_json::Value>,
}

/// Holds the running subprocess and a persistent buffered reader over its stdout.
pub(crate) struct StdioProcess {
    pub(crate) child: TokioChild,
    pub(crate) reader: BufReader<tokio::process::ChildStdout>,
}

/// Stdio connection for local process-based MCP servers
pub struct StdioConnection {
    command: String,
    args: Vec<String>,
    env_vars: HashMap<String, String>,
    working_dir: Option<String>,
    pub(crate) process: Arc<Mutex<Option<StdioProcess>>>,
    timeout: std::time::Duration,
}

impl StdioConnection {
    /// Create a new Stdio connection handler
    pub fn new(command: impl Into<String>, args: Vec<String>) -> Self {
        Self {
            command: command.into(),
            args,
            env_vars: HashMap::new(),
            working_dir: None,
            process: Arc::new(Mutex::new(None)),
            timeout: std::time::Duration::from_secs(30),
        }
    }

    /// Add environment variable for the subprocess
    pub fn with_env(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.env_vars.insert(key.into(), value.into());
        self
    }

    /// Set working directory for the subprocess
    pub fn with_working_dir(mut self, dir: impl Into<String>) -> Self {
        self.working_dir = Some(dir.into());
        self
    }

    /// Set timeout for operations
    pub fn with_timeout(mut self, timeout: std::time::Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Start the subprocess
    pub async fn start(&self) -> Result<()> {
        tracing::info!(command = %self.command, args = ?self.args, "Starting Stdio MCP process");

        let mut cmd = tokio::process::Command::new(&self.command);
        cmd.args(&self.args)
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped());

        // Set working directory if configured
        if let Some(ref dir) = self.working_dir {
            cmd.current_dir(dir);
        }

        // Add environment variables
        for (key, value) in &self.env_vars {
            cmd.env(key, value);
        }

        let mut child = cmd.spawn().map_err(|e| {
            let err_msg = format!("Failed to start subprocess: {}", e);
            tracing::error!(error = %e, command = %self.command, "Subprocess start failed");
            FlowgentraError::MCPError(err_msg)
        })?;

        let stdout = child.stdout.take().ok_or_else(|| {
            FlowgentraError::MCPError("Failed to capture subprocess stdout".into())
        })?;
        let reader = BufReader::new(stdout);

        let mut process = self.process.lock().await;
        *process = Some(StdioProcess { child, reader });

        tracing::info!("Stdio MCP process started successfully");
        Ok(())
    }

    /// Send a JSON-RPC request and receive response
    pub async fn call(
        &self,
        method: impl Into<String>,
        params: serde_json::Value,
    ) -> Result<serde_json::Value> {
        let request = JsonRpcRequest::new(method, params);
        let request_json = serde_json::to_string(&request).map_err(|e| {
            let err_msg = format!("Failed to serialize request: {}", e);
            tracing::error!(error = %e, "Request serialization failed");
            FlowgentraError::MCPError(err_msg)
        })?;

        tracing::debug!(method = %request.method, "Sending JSON-RPC request");

        let read_response = async {
            let mut process = self.process.lock().await;
            let proc = process
                .as_mut()
                .ok_or_else(|| FlowgentraError::MCPError("Process not started".into()))?;

            // Write request to stdin
            let stdin = proc
                .child
                .stdin
                .as_mut()
                .ok_or_else(|| FlowgentraError::MCPError("Failed to get stdin".into()))?;

            stdin
                .write_all(request_json.as_bytes())
                .await
                .map_err(|e| {
                    let err_msg = format!("Failed to write to stdin: {}", e);
                    tracing::error!(error = %e, "stdin write failed");
                    FlowgentraError::MCPError(err_msg)
                })?;
            stdin.write_all(b"\n").await.map_err(|e| {
                FlowgentraError::MCPError(format!("Failed to write newline: {}", e))
            })?;
            stdin
                .flush()
                .await
                .map_err(|e| FlowgentraError::MCPError(format!("Failed to flush stdin: {}", e)))?;

            // Read response from the persistent BufReader (no take/restore needed)
            let mut line = String::new();

            proc.reader.read_line(&mut line).await.map_err(|e| {
                let err_msg = format!("Failed to read from stdout: {}", e);
                tracing::error!(error = %e, "stdout read failed");
                FlowgentraError::MCPError(err_msg)
            })?;

            let response: JsonRpcResponse = serde_json::from_str(&line).map_err(|e| {
                tracing::error!(error = %e, raw = %line.trim(), "Response parsing failed");
                FlowgentraError::MCPError(format!("Failed to parse JSON-RPC response: {}", e))
            })?;

            Ok::<_, FlowgentraError>(response)
        };

        // Apply timeout
        tokio::select! {
            result = read_response => {
                let response = result?;
                if let Some(error) = response.error {
                    return Err(FlowgentraError::MCPError(format!(
                        "JSON-RPC error: {} ({})",
                        error.message, error.code
                    )));
                }
                Ok(response.result.unwrap_or(serde_json::Value::Null))
            }
            _ = tokio::time::sleep(self.timeout) => {
                tracing::warn!("Stdio request timeout — killing subprocess to prevent state corruption");
                // Kill the process so the next call triggers a fresh restart
                // rather than reading stale/partial data from the buffer.
                let mut process = self.process.lock().await;
                if let Some(mut proc) = process.take() {
                    let _ = proc.child.kill().await;
                }
                Err(FlowgentraError::MCPTransportError("Stdio request timeout".into()))
            }
        }
    }

    /// Stop the subprocess
    pub async fn stop(&self) -> Result<()> {
        let mut process = self.process.lock().await;

        if let Some(mut proc) = process.take() {
            proc.child.kill().await.map_err(|e| {
                let err_msg = format!("Failed to kill process: {}", e);
                tracing::error!(error = %e, "Process kill failed");
                FlowgentraError::MCPError(err_msg)
            })?;

            tracing::info!("Stdio MCP process stopped");
        }

        Ok(())
    }

    /// Check if process is still running
    pub async fn is_running(&self) -> bool {
        let mut process = self.process.lock().await;

        if let Some(proc) = process.as_mut() {
            matches!(proc.child.try_wait(), Ok(None))
        } else {
            false
        }
    }
}

/// Builder for StdioConnection
pub struct StdioConnectionBuilder {
    command: String,
    args: Vec<String>,
    env_vars: HashMap<String, String>,
    working_dir: Option<String>,
    timeout: std::time::Duration,
}

impl StdioConnectionBuilder {
    /// Create new builder with command
    pub fn new(command: impl Into<String>) -> Self {
        Self {
            command: command.into(),
            args: Vec::new(),
            env_vars: HashMap::new(),
            working_dir: None,
            timeout: std::time::Duration::from_secs(30),
        }
    }

    /// Add argument to command
    pub fn arg(mut self, arg: impl Into<String>) -> Self {
        self.args.push(arg.into());
        self
    }

    /// Add multiple arguments
    pub fn args(mut self, args: Vec<String>) -> Self {
        self.args.extend(args);
        self
    }

    /// Add environment variable
    pub fn env(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.env_vars.insert(key.into(), value.into());
        self
    }

    /// Set timeout
    pub fn timeout(mut self, duration: std::time::Duration) -> Self {
        self.timeout = duration;
        self
    }

    /// Set working directory
    pub fn working_dir(mut self, dir: impl Into<String>) -> Self {
        self.working_dir = Some(dir.into());
        self
    }

    /// Build StdioConnection
    pub fn build(self) -> StdioConnection {
        StdioConnection {
            command: self.command,
            args: self.args,
            env_vars: self.env_vars,
            working_dir: self.working_dir,
            process: Arc::new(Mutex::new(None)),
            timeout: self.timeout,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_jsonrpc_request_creation() {
        let req = JsonRpcRequest::new("tools/list", serde_json::json!({}));
        assert_eq!(req.jsonrpc, "2.0");
        assert_eq!(req.method, "tools/list");
    }

    #[test]
    fn test_jsonrpc_request_serialization() {
        let req = JsonRpcRequest::new("test", serde_json::json!({"key": "value"}));
        let json = serde_json::to_string(&req).unwrap();
        assert!(json.contains("\"method\":\"test\""));
    }

    #[test]
    fn test_stdio_connection_builder() {
        let conn = StdioConnectionBuilder::new("python")
            .arg("/path/to/tool.py")
            .env("PYTHONUNBUFFERED", "1")
            .timeout(std::time::Duration::from_secs(60))
            .build();

        assert_eq!(conn.command, "python");
        assert_eq!(conn.args.len(), 1);
        assert!(conn.env_vars.contains_key("PYTHONUNBUFFERED"));
    }

    #[tokio::test]
    async fn test_stdio_connection_creation() {
        let conn = StdioConnection::new("echo", vec!["hello".to_string()]);
        assert!(!conn.is_running().await);
    }
}
