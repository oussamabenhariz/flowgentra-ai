//! SSE (Server-Sent Events) Protocol Handler for MCP
//!
//! Implements streaming communication with MCP servers over HTTP/SSE.

use crate::core::error::{FlowgentraError, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{mpsc, RwLock};

/// SSE message from MCP server
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SSEMessage {
    pub id: String,
    pub event_type: String,
    pub data: serde_json::Value,
    pub timestamp: String,
}

/// SSE streaming response
#[derive(Debug, Clone)]
pub struct SSEResponse {
    pub id: String,
    pub status: SSEStatus,
    pub data: serde_json::Value,
    pub metadata: ResponseMetadata,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SSEStatus {
    Streaming,
    Completed,
    Failed,
}

#[derive(Debug, Clone, Default)]
pub struct ResponseMetadata {
    pub duration_ms: u64,
    pub message_count: usize,
    pub error: Option<String>,
}

/// SSE Connection handler
pub struct SSEConnection {
    base_url: String,
    client: reqwest::Client,
    timeout: std::time::Duration,
    active_streams: Arc<RwLock<HashMap<String, SSEStreamState>>>,
}

#[derive(Clone)]
#[allow(dead_code)]
struct SSEStreamState {
    id: String,
    message_count: usize,
    start_time: std::time::Instant,
}

impl SSEConnection {
    /// Create new SSE connection handler
    pub fn new(base_url: String, timeout_secs: u64) -> Self {
        Self {
            base_url,
            client: reqwest::Client::new(),
            timeout: std::time::Duration::from_secs(timeout_secs.max(30)),
            active_streams: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Open SSE stream and listen for events
    pub async fn stream(
        &self,
        endpoint: impl AsRef<str>,
        request_body: serde_json::Value,
    ) -> Result<SSEStreamReceiver> {
        let endpoint = endpoint.as_ref();
        let url = format!("{}/{}", self.base_url, endpoint);
        let stream_id = uuid::Uuid::new_v4().to_string();

        tracing::debug!(stream_id = %stream_id, url = %url, "Opening SSE stream");

        let response = self
            .client
            .post(&url)
            .timeout(self.timeout)
            .header("Accept", "text/event-stream")
            .json(&request_body)
            .send()
            .await
            .map_err(|e| {
                let err_msg = format!("Failed to open SSE stream: {}", e);
                tracing::error!(error = %e, url = %url, "SSE stream failed to open");
                FlowgentraError::MCPError(err_msg)
            })?;

        if !response.status().is_success() {
            return Err(FlowgentraError::MCPError(format!(
                "SSE stream returned status {}",
                response.status()
            )));
        }

        let (tx, rx) = mpsc::unbounded_channel();

        // Store stream state
        {
            let mut streams = self.active_streams.write().await;
            streams.insert(
                stream_id.clone(),
                SSEStreamState {
                    id: stream_id.clone(),
                    message_count: 0,
                    start_time: std::time::Instant::now(),
                },
            );
        }

        let stream_id_clone = stream_id.clone();
        let active_streams = self.active_streams.clone();
        let timeout = self.timeout;

        // Spawn task to read SSE stream
        tokio::spawn(async move {
            let mut stream = response.bytes_stream();
            let mut buffer = String::new();

            loop {
                tokio::select! {
                    _ = tokio::time::sleep(timeout) => {
                        tracing::warn!(stream_id = %stream_id_clone, "SSE stream timeout");
                        let _ = tx.send(Err(FlowgentraError::MCPError("Stream timeout".into())));
                        break;
                    }
                    result = futures::stream::StreamExt::next(&mut stream) => {
                        match result {
                            Some(Ok(bytes)) => {
                                buffer.push_str(&String::from_utf8_lossy(&bytes));

                                // Parse SSE events
                                while let Some(event_end) = buffer.find("\n\n") {
                                    let event_data = buffer[..event_end].to_string();
                                    buffer.drain(..event_end + 2);

                                    match parse_sse_event(&event_data) {
                                        Ok(message) => {
                                            let mut streams = active_streams.write().await;
                                            if let Some(state) = streams.get_mut(&stream_id_clone) {
                                                state.message_count += 1;
                                            }
                                            drop(streams);

                                            let _ = tx.send(Ok(message));
                                        }
                                        Err(e) => {
                                            tracing::warn!(error = %e, "Failed to parse SSE event");
                                        }
                                    }
                                }
                            }
                            Some(Err(e)) => {
                                tracing::error!(error = %e, stream_id = %stream_id_clone, "SSE stream error");
                                let _ = tx.send(Err(FlowgentraError::MCPError(format!("Stream error: {}", e))));
                                break;
                            }
                            None => {
                                tracing::debug!(stream_id = %stream_id_clone, "SSE stream closed");
                                break;
                            }
                        }
                    }
                }
            }

            active_streams.write().await.remove(&stream_id_clone);
        });

        Ok(SSEStreamReceiver {
            id: stream_id,
            receiver: rx,
        })
    }

    /// Get active stream count
    pub async fn active_stream_count(&self) -> usize {
        self.active_streams.read().await.len()
    }

    /// Close all streams
    pub async fn close_all(&self) {
        self.active_streams.write().await.clear();
    }
}

/// Receiver for SSE stream messages
pub struct SSEStreamReceiver {
    pub id: String,
    receiver: mpsc::UnboundedReceiver<Result<SSEMessage>>,
}

impl SSEStreamReceiver {
    /// Receive next message from stream
    pub async fn next(&mut self) -> Option<Result<SSEMessage>> {
        self.receiver.recv().await
    }

    /// Collect all remaining messages
    pub async fn collect_all(mut self) -> Result<Vec<SSEMessage>> {
        let mut messages = Vec::new();

        while let Some(result) = self.next().await {
            messages.push(result?);
        }

        Ok(messages)
    }
}

/// Parse SSE event format: event: type\ndata: {json}\n\n
fn parse_sse_event(event_data: &str) -> Result<SSEMessage> {
    let mut event_type = "message".to_string();
    let mut data = serde_json::Value::Null;

    for line in event_data.lines() {
        if let Some(event_line) = line.strip_prefix("event:") {
            event_type = event_line.trim().to_string();
        } else if let Some(data_line) = line.strip_prefix("data:") {
            let json_str = data_line.trim();
            data = serde_json::from_str(json_str)
                .unwrap_or(serde_json::Value::String(json_str.to_string()));
        }
    }

    Ok(SSEMessage {
        id: uuid::Uuid::new_v4().to_string(),
        event_type,
        data,
        timestamp: chrono::Local::now().to_rfc3339(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_sse_event_basic() {
        let event_data = r#"event: message
data: {"type": "tool_result", "content": "Hello"}
"#;
        let result = parse_sse_event(event_data);
        assert!(result.is_ok());

        let msg = result.unwrap();
        assert_eq!(msg.event_type, "message");
        assert_eq!(msg.data["type"], "tool_result");
    }

    #[test]
    fn parse_sse_event_with_complex_data() {
        let event_data = r#"event: tool_call
data: {"tool": "search", "args": {"query": "test"}}
"#;
        let result = parse_sse_event(event_data);
        assert!(result.is_ok());

        let msg = result.unwrap();
        assert_eq!(msg.event_type, "tool_call");
    }

    #[test]
    fn sse_connection_creation() {
        let conn = SSEConnection::new("http://localhost:8000".to_string(), 30);
        assert_eq!(conn.base_url, "http://localhost:8000");
    }

    #[tokio::test]
    async fn sse_active_stream_count() {
        let conn = SSEConnection::new("http://localhost:8000".to_string(), 30);
        assert_eq!(conn.active_stream_count().await, 0);

        conn.close_all().await;
        assert_eq!(conn.active_stream_count().await, 0);
    }
}
