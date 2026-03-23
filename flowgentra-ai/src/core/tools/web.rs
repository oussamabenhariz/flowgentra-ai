/// Web-related tools (HTTP requests, web search)
use super::{JsonSchema, Tool, ToolDefinition};
use crate::prelude::*;
use serde_json::json;
use std::collections::HashMap;

/// Tool for making HTTP GET requests
pub struct FetchTool;

#[async_trait::async_trait]
impl Tool for FetchTool {
    fn definition(&self) -> ToolDefinition {
        let mut props = HashMap::new();
        props.insert(
            "url".to_string(),
            JsonSchema::string().with_description("URL to fetch"),
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
        let url = input
            .get("url")
            .and_then(|v| v.as_str())
            .ok_or_else(|| FlowgentraError::ToolError("Missing 'url' parameter".to_string()))?;

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
        let text = response
            .text()
            .await
            .map_err(|e| FlowgentraError::ToolError(format!("Failed to read response: {}", e)))?;

        Ok(json!({
            "status": status,
            "content": text,
            "url": url
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
