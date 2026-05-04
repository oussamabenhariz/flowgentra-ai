//! Standard MCP HTTP/SSE transport client.
//!
//! Implements the MCP wire protocol over HTTP/SSE:
//!   1. `GET /sse` — opens the SSE event stream
//!   2. Server emits `event: endpoint` with the session URL (e.g. `/messages/?session_id=xxx`)
//!   3. Client POSTs JSON-RPC 2.0 requests to `{base_url}{session_path}`
//!   4. Responses arrive via the SSE stream as `event: message` events

use crate::core::error::{FlowgentraError, Result};
use crate::core::mcp::{apply_tool_filters, MCPClient, MCPConfig, MCPTool, MCP_PROTOCOL_VERSION};
use futures::StreamExt;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use tokio::sync::{oneshot, Mutex, RwLock};

// ── Shared state ─────────────────────────────────────────────────────────────

struct SseState {
    base_url: String,
    http_client: reqwest::Client,
    /// Set once the server sends the `endpoint` event.
    session_url: RwLock<Option<String>>,
    /// Pending requests awaiting a response, keyed by JSON-RPC id.
    pending: Mutex<HashMap<u64, oneshot::Sender<std::result::Result<serde_json::Value, String>>>>,
    next_id: AtomicU64,
}

// ── Client ───────────────────────────────────────────────────────────────────

/// MCP client using the standard HTTP/SSE transport.
pub struct MCPSseProtocolClient {
    config: MCPConfig,
    state: Arc<SseState>,
}

impl MCPSseProtocolClient {
    pub fn new(config: MCPConfig) -> Self {
        let timeout = config.connection_settings.timeout.unwrap_or(30);
        let base_url = config.uri.trim_end_matches('/').to_string();

        let http_client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(timeout))
            .build()
            .unwrap_or_else(|_| reqwest::Client::new());

        let state = Arc::new(SseState {
            base_url,
            http_client,
            session_url: RwLock::new(None),
            pending: Mutex::new(HashMap::new()),
            next_id: AtomicU64::new(1),
        });

        Self { config, state }
    }

    /// Open `GET /sse`, wait for the `endpoint` event, then keep the stream alive
    /// in a background task that routes JSON-RPC responses to waiting callers.
    async fn connect(&self) -> Result<()> {
        if self.state.session_url.read().await.is_some() {
            return Ok(());
        }

        let sse_url = format!("{}/sse", self.state.base_url);
        tracing::debug!(url = %sse_url, "Opening MCP SSE connection");

        let response = self
            .state
            .http_client
            .get(&sse_url)
            .header("Accept", "text/event-stream")
            .header("Cache-Control", "no-cache")
            .send()
            .await
            .map_err(|e| FlowgentraError::MCPError(format!("SSE connect failed: {}", e)))?;

        if !response.status().is_success() {
            return Err(FlowgentraError::MCPError(format!(
                "SSE connect returned {}",
                response.status()
            )));
        }

        let state = self.state.clone();
        let base_url = self.state.base_url.clone();
        let (session_tx, session_rx) = oneshot::channel::<String>();
        let mut session_tx_opt = Some(session_tx);

        tokio::spawn(async move {
            let mut byte_stream = response.bytes_stream();
            let mut buf = String::new();

            while let Some(chunk) = byte_stream.next().await {
                let bytes = match chunk {
                    Ok(b) => b,
                    Err(e) => {
                        tracing::error!("SSE stream error: {}", e);
                        break;
                    }
                };

                buf.push_str(&String::from_utf8_lossy(&bytes));

                // SSE events are separated by a blank line (\n\n)
                while let Some(pos) = buf.find("\n\n") {
                    let event_str = buf[..pos].to_string();
                    buf.drain(..pos + 2);

                    let (event_type, data) = parse_sse_fields(&event_str);

                    match event_type.as_str() {
                        "endpoint" => {
                            // data is a relative path: /messages/?session_id=xxx
                            let full_url = format!("{}{}", base_url, data);
                            tracing::info!(session_url = %full_url, "MCP session established");
                            *state.session_url.write().await = Some(full_url.clone());
                            if let Some(tx) = session_tx_opt.take() {
                                let _ = tx.send(full_url);
                            }
                        }
                        "message" | "" => {
                            route_response(&state, &data).await;
                        }
                        _ => {}
                    }
                }
            }

            // Stream ended — fail any in-flight requests
            let mut pending = state.pending.lock().await;
            for (_, tx) in pending.drain() {
                let _ = tx.send(Err("SSE connection closed".into()));
            }
        });

        tokio::time::timeout(std::time::Duration::from_secs(10), session_rx)
            .await
            .map_err(|_| FlowgentraError::MCPError("Timed out waiting for MCP session URL".into()))?
            .map_err(|_| {
                FlowgentraError::MCPError("SSE closed before session URL received".into())
            })?;

        Ok(())
    }

    /// Send a JSON-RPC request and await the matching response from the SSE stream.
    async fn call(&self, method: &str, params: serde_json::Value) -> Result<serde_json::Value> {
        // Auto-connect when called directly without initialize()
        if self.state.session_url.read().await.is_none() {
            self.connect().await?;
        }

        let id = self.state.next_id.fetch_add(1, Ordering::SeqCst);
        let (tx, rx) = oneshot::channel();
        self.state.pending.lock().await.insert(id, tx);

        let session_url = self
            .state
            .session_url
            .read()
            .await
            .clone()
            .ok_or_else(|| FlowgentraError::MCPError("Not connected to MCP server".into()))?;

        let body = serde_json::json!({
            "jsonrpc": "2.0",
            "id": id,
            "method": method,
            "params": params,
        });

        let resp = self
            .state
            .http_client
            .post(&session_url)
            .json(&body)
            .send()
            .await
            .map_err(|e| FlowgentraError::MCPError(format!("POST '{}' failed: {}", method, e)))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            self.state.pending.lock().await.remove(&id);
            return Err(FlowgentraError::MCPError(format!(
                "MCP server returned {}: {}",
                status, text
            )));
        }

        rx.await
            .map_err(|_| FlowgentraError::MCPError("Response channel closed".into()))?
            .map_err(|e| FlowgentraError::MCPError(format!("MCP error: {}", e)))
    }

    /// Send a JSON-RPC notification (no id, no response expected).
    async fn notify(&self, method: &str, params: serde_json::Value) -> Result<()> {
        let session_url = match self.state.session_url.read().await.clone() {
            Some(u) => u,
            None => return Ok(()), // not connected, skip
        };

        let body = serde_json::json!({
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
        });

        self.state
            .http_client
            .post(&session_url)
            .json(&body)
            .send()
            .await
            .map_err(|e| {
                FlowgentraError::MCPError(format!("Notification '{}' failed: {}", method, e))
            })?;

        Ok(())
    }
}

// ── MCPClient impl ────────────────────────────────────────────────────────────

#[async_trait::async_trait]
impl MCPClient for MCPSseProtocolClient {
    async fn initialize(&self) -> Result<String> {
        self.connect().await?;

        let params = serde_json::json!({
            "protocolVersion": MCP_PROTOCOL_VERSION,
            "capabilities": {
                "roots": { "listChanged": true },
                "sampling": {}
            },
            "clientInfo": {
                "name": "flowgentra-ai",
                "version": env!("CARGO_PKG_VERSION")
            }
        });

        let result = self.call("initialize", params).await?;
        let _ = self
            .notify("notifications/initialized", serde_json::json!({}))
            .await;

        Ok(result["protocolVersion"]
            .as_str()
            .unwrap_or(MCP_PROTOCOL_VERSION)
            .to_string())
    }

    async fn list_tools(&self) -> Result<Vec<MCPTool>> {
        let result = self.call("tools/list", serde_json::json!({})).await?;

        let tools: Vec<MCPTool> = serde_json::from_value(
            result
                .get("tools")
                .cloned()
                .unwrap_or(serde_json::Value::Array(vec![])),
        )
        .map_err(|e| FlowgentraError::MCPError(format!("Failed to parse tools: {}", e)))?;

        Ok(apply_tool_filters(
            tools,
            self.config.namespace.as_deref(),
            self.config.tool_include.as_deref(),
            self.config.tool_exclude.as_deref(),
        ))
    }

    async fn call_tool(
        &self,
        tool_name: &str,
        arguments: serde_json::Value,
    ) -> Result<serde_json::Value> {
        if self.config.is_tool_excluded(tool_name) {
            return Err(FlowgentraError::ToolError(format!(
                "Tool '{}' is excluded by the MCP server's deny configuration.",
                tool_name
            )));
        }

        let raw_name = self.config.strip_namespace(tool_name);

        let result = self
            .call(
                "tools/call",
                serde_json::json!({
                    "name": raw_name,
                    "arguments": arguments,
                }),
            )
            .await?;

        // MCP tools/call response: { "content": [{"type":"text","text":"..."}], "isError": false }
        // Extract the first text item for a clean return value.
        if let Some(items) = result.get("content").and_then(|c| c.as_array()) {
            if let Some(text) = items
                .iter()
                .find_map(|item| item.get("text").and_then(|t| t.as_str()))
            {
                return Ok(serde_json::Value::String(text.to_string()));
            }
        }

        Ok(result)
    }

    async fn shutdown(&self) -> Result<()> {
        *self.state.session_url.write().await = None;
        Ok(())
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Route a raw SSE data string to the waiting oneshot channel for its request id.
async fn route_response(state: &Arc<SseState>, data: &str) {
    let Ok(value) = serde_json::from_str::<serde_json::Value>(data) else {
        return;
    };
    let Some(id) = value.get("id").and_then(|v| v.as_u64()) else {
        return;
    };
    let Some(tx) = state.pending.lock().await.remove(&id) else {
        return;
    };

    if let Some(result) = value.get("result") {
        let _ = tx.send(Ok(result.clone()));
    } else if let Some(err) = value.get("error") {
        let msg = err
            .get("message")
            .and_then(|m| m.as_str())
            .unwrap_or("unknown MCP error")
            .to_string();
        let _ = tx.send(Err(msg));
    } else {
        let _ = tx.send(Ok(serde_json::Value::Null));
    }
}

/// Extract `event:` and `data:` fields from a raw SSE event block.
fn parse_sse_fields(event_str: &str) -> (String, String) {
    let mut event_type = String::new();
    let mut data = String::new();

    for line in event_str.lines() {
        if let Some(v) = line.strip_prefix("event:") {
            event_type = v.trim().to_string();
        } else if let Some(v) = line.strip_prefix("data:") {
            data = v.trim().to_string();
        }
    }

    (event_type, data)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_endpoint_event() {
        let (etype, data) = parse_sse_fields("event: endpoint\ndata: /messages/?session_id=abc123");
        assert_eq!(etype, "endpoint");
        assert_eq!(data, "/messages/?session_id=abc123");
    }

    #[test]
    fn parse_message_event() {
        let json = r#"{"jsonrpc":"2.0","id":1,"result":{"tools":[]}}"#;
        let (etype, data) = parse_sse_fields(&format!("event: message\ndata: {}", json));
        assert_eq!(etype, "message");
        assert_eq!(data, json);
    }

    #[test]
    fn parse_event_no_type() {
        let (etype, data) = parse_sse_fields("data: hello");
        assert_eq!(etype, "");
        assert_eq!(data, "hello");
    }
}
