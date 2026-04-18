//! Web tools: HTTP GET with SSRF protection.

use super::{JsonSchema, Tool, ToolDefinition};
use crate::prelude::*;
use futures::StreamExt;
use serde_json::json;
use std::collections::HashMap;

/// Maximum response body size to buffer in memory (10 MiB).
pub(crate) const MAX_RESPONSE_BYTES: usize = 10 * 1024 * 1024;

/// Validate that a URL is safe to fetch (SSRF protection).
///
/// Rejects non-http/https schemes, private/loopback/link-local IP ranges.
pub(crate) fn validate_fetch_url(raw: &str) -> Result<reqwest::Url> {
    let url = reqwest::Url::parse(raw)
        .map_err(|e| FlowgentraError::ToolError(format!("Invalid URL: {}", e)))?;

    match url.scheme() {
        "http" | "https" => {}
        scheme => {
            return Err(FlowgentraError::ToolError(format!(
                "URL scheme '{}' is not allowed; use http or https",
                scheme
            )));
        }
    }

    if let Some(host) = url.host_str() {
        let addrs: Vec<std::net::IpAddr> = if let Ok(ip) = host.parse::<std::net::IpAddr>() {
            vec![ip]
        } else {
            std::net::ToSocketAddrs::to_socket_addrs(&(host, 80))
                .map_err(|e| {
                    FlowgentraError::ToolError(format!(
                        "DNS resolution failed for '{}': {}",
                        host, e
                    ))
                })?
                .map(|sa| sa.ip())
                .collect()
        };

        for ip in addrs {
            if is_private_ip(ip) {
                return Err(FlowgentraError::ToolError(format!(
                    "Requests to private/internal addresses are not allowed (resolved '{}')",
                    ip
                )));
            }
        }
    } else {
        return Err(FlowgentraError::ToolError("URL has no host".to_string()));
    }

    Ok(url)
}

pub(crate) fn is_private_ip(ip: std::net::IpAddr) -> bool {
    match ip {
        std::net::IpAddr::V4(v4) => {
            v4.is_loopback()
                || v4.is_private()
                || v4.is_link_local()
                || v4.is_broadcast()
                || v4.is_multicast()
                || v4.is_unspecified()
        }
        std::net::IpAddr::V6(v6) => {
            v6.is_loopback()
                || v6.is_multicast()
                || v6.is_unspecified()
                || matches!(v6.segments()[0] & 0xfe00, 0xfc00)
                || matches!(v6.segments()[0] & 0xffc0, 0xfe80)
        }
    }
}

/// Tool for making HTTP GET requests with SSRF protection and response size cap.
pub struct FetchTool;

#[async_trait::async_trait]
impl Tool for FetchTool {
    fn definition(&self) -> ToolDefinition {
        let mut props = HashMap::new();
        props.insert(
            "url".to_string(),
            JsonSchema::string().with_description("URL to fetch (http/https only)"),
        );
        props.insert(
            "timeout_secs".to_string(),
            JsonSchema::integer().with_description("Request timeout in seconds (default: 30)"),
        );

        ToolDefinition::new(
            "http_get",
            "Fetch content from a URL via HTTP GET (SSRF-protected, 10 MiB cap)",
            JsonSchema::object()
                .with_properties(props)
                .with_required(vec!["url".to_string()]),
            JsonSchema::object(),
        )
        .with_category("network")
    }

    async fn call(&self, input: serde_json::Value) -> Result<serde_json::Value> {
        let raw_url = input
            .get("url")
            .and_then(|v| v.as_str())
            .ok_or_else(|| FlowgentraError::ToolError("Missing 'url' parameter".to_string()))?;

        let url = validate_fetch_url(raw_url)?;

        let timeout = input
            .get("timeout_secs")
            .and_then(|v| v.as_u64())
            .unwrap_or(30);

        let client = reqwest::Client::new();
        let response = tokio::time::timeout(
            std::time::Duration::from_secs(timeout),
            client.get(url).send(),
        )
        .await
        .map_err(|_| FlowgentraError::ToolError("Request timeout".to_string()))?
        .map_err(|e| FlowgentraError::ToolError(format!("HTTP error: {}", e)))?;

        let status = response.status().as_u16();

        let bytes = tokio::time::timeout(std::time::Duration::from_secs(timeout), async {
            let mut buf = Vec::new();
            let mut byte_stream = response.bytes_stream();
            while let Some(chunk) = byte_stream.next().await {
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

        let text = String::from_utf8_lossy(&bytes).into_owned();

        Ok(json!({
            "status": status,
            "content": text
        }))
    }
}
