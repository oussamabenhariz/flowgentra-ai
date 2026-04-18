//! Full HTTP request tool supporting all methods, custom headers, and request bodies.
//!
//! Reuses the SSRF protection and response-size cap from `web.rs`.

use super::{JsonSchema, Tool, ToolDefinition};
use crate::core::tools::web::{validate_fetch_url, MAX_RESPONSE_BYTES};
use crate::prelude::*;
use async_trait::async_trait;
use futures::StreamExt;
use serde_json::{json, Value};
use std::collections::HashMap;

pub struct WebRequestTool {
    client: reqwest::Client,
}

impl WebRequestTool {
    pub fn new() -> Self {
        Self {
            client: reqwest::Client::new(),
        }
    }
}

impl Default for WebRequestTool {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Tool for WebRequestTool {
    async fn call(&self, input: Value) -> Result<Value> {
        let raw_url = input
            .get("url")
            .and_then(|v| v.as_str())
            .ok_or_else(|| FlowgentraError::ToolError("Missing 'url' field".to_string()))?;

        let method = input
            .get("method")
            .and_then(|v| v.as_str())
            .unwrap_or("GET")
            .to_uppercase();

        let timeout_secs = input
            .get("timeout_secs")
            .and_then(|v| v.as_u64())
            .unwrap_or(30);

        let url = validate_fetch_url(raw_url)?;

        // Build the request for the chosen method
        let mut builder = match method.as_str() {
            "GET" => self.client.get(url),
            "POST" => self.client.post(url),
            "PUT" => self.client.put(url),
            "PATCH" => self.client.patch(url),
            "DELETE" => self.client.delete(url),
            "HEAD" => self.client.head(url),
            other => {
                return Err(FlowgentraError::ToolError(format!(
                    "Unsupported HTTP method: {}",
                    other
                )));
            }
        };

        // Attach custom headers
        if let Some(headers_map) = input.get("headers").and_then(|v| v.as_object()) {
            for (k, v) in headers_map {
                if let Some(v_str) = v.as_str() {
                    builder = builder.header(k.as_str(), v_str);
                }
            }
        }

        // Attach body
        if let Some(body) = input.get("body").and_then(|v| v.as_str()) {
            builder = builder.body(body.to_string());
        }

        let response =
            tokio::time::timeout(std::time::Duration::from_secs(timeout_secs), builder.send())
                .await
                .map_err(|_| FlowgentraError::ToolError("Request timeout".to_string()))?
                .map_err(|e| FlowgentraError::ToolError(format!("HTTP error: {}", e)))?;

        let status = response.status().as_u16();

        // Stream body with size cap
        let bytes = tokio::time::timeout(std::time::Duration::from_secs(timeout_secs), async {
            let mut buf = Vec::new();
            let mut stream = response.bytes_stream();
            while let Some(chunk) = stream.next().await {
                let chunk = chunk.map_err(|e| {
                    FlowgentraError::ToolError(format!("Failed to read response chunk: {}", e))
                })?;
                if buf.len() + chunk.len() > MAX_RESPONSE_BYTES {
                    return Err(FlowgentraError::ToolError(format!(
                        "Response body exceeds {} byte limit",
                        MAX_RESPONSE_BYTES
                    )));
                }
                buf.extend_from_slice(&chunk);
            }
            Ok::<Vec<u8>, FlowgentraError>(buf)
        })
        .await
        .map_err(|_| FlowgentraError::ToolError("Response read timeout".to_string()))??;

        let body_text = String::from_utf8_lossy(&bytes).into_owned();

        Ok(json!({"status": status, "body": body_text}))
    }

    fn definition(&self) -> ToolDefinition {
        let mut props = HashMap::new();
        props.insert(
            "url".to_string(),
            JsonSchema::string().with_description("URL to request (http/https only)"),
        );
        props.insert(
            "method".to_string(),
            JsonSchema::string().with_description(
                "HTTP method: GET, POST, PUT, PATCH, DELETE, HEAD (default: GET)",
            ),
        );
        props.insert(
            "headers".to_string(),
            JsonSchema::object().with_description("Optional request headers as key-value pairs"),
        );
        props.insert(
            "body".to_string(),
            JsonSchema::string().with_description("Optional request body"),
        );
        props.insert(
            "timeout_secs".to_string(),
            JsonSchema::integer().with_description("Request timeout in seconds (default: 30)"),
        );

        ToolDefinition::new(
            "web_request",
            "Make HTTP requests with any method, custom headers and body (SSRF-protected)",
            JsonSchema::object()
                .with_properties(props)
                .with_required(vec!["url".to_string()]),
            JsonSchema::object().with_properties({
                let mut out = HashMap::new();
                out.insert("status".to_string(), JsonSchema::integer());
                out.insert("body".to_string(), JsonSchema::string());
                out
            }),
        )
        .with_category("network")
        .with_example(
            json!({"url": "https://api.example.com/data", "method": "POST",
                   "headers": {"Content-Type": "application/json"},
                   "body": "{\"key\":\"value\"}"}),
            json!({"status": 200, "body": "{\"result\":\"ok\"}"}),
            "POST JSON to an API endpoint",
        )
    }
}
