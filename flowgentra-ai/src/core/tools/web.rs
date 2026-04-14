/// Web-related tools (HTTP requests, web search)
use super::{JsonSchema, Tool, ToolDefinition};
use crate::prelude::*;
use futures::StreamExt;
use serde_json::json;
use std::collections::HashMap;

/// Maximum response body size to buffer in memory (10 MiB).
const MAX_RESPONSE_BYTES: usize = 10 * 1024 * 1024;

/// Validate that a URL is safe to fetch (SSRF protection).
///
/// Rejects:
/// - Non-http/https schemes (blocks `file://`, `ftp://`, etc.)
/// - Private / link-local / loopback IPv4 ranges
/// - Private / link-local / loopback IPv6 ranges
fn validate_fetch_url(raw: &str) -> Result<reqwest::Url> {
    let url = reqwest::Url::parse(raw)
        .map_err(|e| FlowgentraError::ToolError(format!("Invalid URL: {}", e)))?;

    // Only allow http and https
    match url.scheme() {
        "http" | "https" => {}
        scheme => {
            return Err(FlowgentraError::ToolError(format!(
                "URL scheme '{}' is not allowed; use http or https",
                scheme
            )));
        }
    }

    // Resolve the host to an IP and block private ranges
    if let Some(host) = url.host_str() {
        // Attempt to parse as a literal IP first; if it is a hostname we
        // do a synchronous DNS lookup so we can inspect the result before
        // the actual request is made.
        let addrs: Vec<std::net::IpAddr> = if let Ok(ip) = host.parse::<std::net::IpAddr>() {
            vec![ip]
        } else {
            // Perform a blocking DNS resolution (acceptable here because
            // tool calls are not on a latency-critical path).
            std::net::ToSocketAddrs::to_socket_addrs(&(host, 80))
                .map_err(|e| {
                    FlowgentraError::ToolError(format!("DNS resolution failed for '{}': {}", host, e))
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

/// Returns true for loopback, private, and link-local IP addresses.
fn is_private_ip(ip: std::net::IpAddr) -> bool {
    match ip {
        std::net::IpAddr::V4(v4) => {
            v4.is_loopback()          // 127.0.0.0/8
                || v4.is_private()    // 10.0.0.0/8, 172.16.0.0/12, 192.168.0.0/16
                || v4.is_link_local() // 169.254.0.0/16
                || v4.is_broadcast()  // 255.255.255.255
                || v4.is_multicast()  // 224.0.0.0/4
                || v4.is_unspecified()// 0.0.0.0
        }
        std::net::IpAddr::V6(v6) => {
            v6.is_loopback()           // ::1
                || v6.is_multicast()   // ff00::/8
                || v6.is_unspecified() // ::
                // Unique-local (fc00::/7) and link-local (fe80::/10)
                || matches!(v6.segments()[0] & 0xfe00, 0xfc00)
                || matches!(v6.segments()[0] & 0xffc0, 0xfe80)
        }
    }
}

/// Tool for making HTTP GET requests
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
            JsonSchema::integer().with_description("Request timeout in seconds"),
        );

        ToolDefinition::new(
            "http_get",
            "Fetch content from a URL via HTTP GET request",
            JsonSchema::object()
                .with_properties(props)
                .with_required(vec!["url".to_string()]),
            JsonSchema::object(),
        )
    }

    async fn call(&self, input: serde_json::Value) -> Result<serde_json::Value> {
        let raw_url = input
            .get("url")
            .and_then(|v| v.as_str())
            .ok_or_else(|| FlowgentraError::ToolError("Missing 'url' parameter".to_string()))?;

        // Validate URL before making the request (SSRF protection)
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

        // Enforce a body size cap to prevent memory exhaustion
        let bytes = tokio::time::timeout(
            std::time::Duration::from_secs(timeout),
            async {
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
            },
        )
        .await
        .map_err(|_| FlowgentraError::ToolError("Response read timeout".to_string()))??;

        let text = String::from_utf8_lossy(&bytes).into_owned();

        // Do not echo back the raw user-supplied URL to avoid reflected-value issues
        Ok(json!({
            "status": status,
            "content": text
        }))
    }
}

/// Tool for web search (requires integration with search API)
pub struct SearchTool;

#[async_trait::async_trait]
impl Tool for SearchTool {
    fn definition(&self) -> ToolDefinition {
        let mut props = HashMap::new();
        props.insert(
            "query".to_string(),
            JsonSchema::string().with_description("Search query"),
        );
        props.insert(
            "max_results".to_string(),
            JsonSchema::integer().with_description("Maximum number of results"),
        );

        ToolDefinition::new(
            "web_search",
            "Search the web for relevant results",
            JsonSchema::object()
                .with_properties(props)
                .with_required(vec!["query".to_string()]),
            JsonSchema::object(),
        )
    }

    async fn call(&self, input: serde_json::Value) -> Result<serde_json::Value> {
        let query = input
            .get("query")
            .and_then(|v| v.as_str())
            .ok_or_else(|| FlowgentraError::ToolError("Missing 'query' parameter".to_string()))?;

        let max_results = input
            .get("max_results")
            .and_then(|v| v.as_u64())
            .unwrap_or(10) as usize;

        // Mock implementation - in production, integrate with search API
        // such as Google Custom Search, Bing, or DuckDuckGo
        Ok(json!({
            "query": query,
            "results": [
                {
                    "title": "Example Result",
                    "url": "https://example.com",
                    "snippet": "This is a mock search result"
                }
            ],
            "max_results": max_results
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_search_tool() {
        let tool = SearchTool;
        let result = tool
            .call(json!({
                "query": "Rust programming",
                "max_results": 5
            }))
            .await
            .unwrap();

        assert_eq!(result["query"], "Rust programming");
        assert_eq!(result["max_results"], 5);
    }
}
