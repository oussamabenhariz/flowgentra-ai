//! Built-in tools for common operations
//!
//! Includes: Calculator, Search, Web Requests, File Operations

use super::{JsonSchema, Tool, ToolDefinition};
use crate::core::error::{FlowgentraError, Result};
use async_trait::async_trait;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::fs;

// =============================================================================
// Calculator Tool
// =============================================================================

/// Simple calculator tool supporting basic math operations
pub struct CalculatorTool;

impl CalculatorTool {
    pub fn new() -> Self {
        CalculatorTool
    }

    fn evaluate(&self, operation: &str, a: f64, b: f64) -> Result<f64> {
        match operation {
            "add" => Ok(a + b),
            "subtract" => Ok(a - b),
            "multiply" => Ok(a * b),
            "divide" => {
                if b == 0.0 {
                    Err(FlowgentraError::ToolError("Division by zero".to_string()))
                } else {
                    Ok(a / b)
                }
            }
            "power" => Ok(a.powf(b)),
            "modulo" => {
                if b == 0.0 {
                    Err(FlowgentraError::ToolError("Modulo by zero".to_string()))
                } else {
                    Ok(a % b)
                }
            }
            _ => Err(FlowgentraError::ToolError(format!(
                "Unknown operation: {}",
                operation
            ))),
        }
    }
}

impl Default for CalculatorTool {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Tool for CalculatorTool {
    async fn call(&self, input: Value) -> Result<Value> {
        let operation = input
            .get("operation")
            .and_then(|v| v.as_str())
            .ok_or_else(|| FlowgentraError::ToolError("Missing 'operation' field".to_string()))?;

        let a = input.get("a").and_then(|v| v.as_f64()).ok_or_else(|| {
            FlowgentraError::ToolError("Missing or invalid 'a' field".to_string())
        })?;

        let b = input.get("b").and_then(|v| v.as_f64()).ok_or_else(|| {
            FlowgentraError::ToolError("Missing or invalid 'b' field".to_string())
        })?;

        let result = self.evaluate(operation, a, b)?;

        Ok(json!({
            "result": result,
            "operation": operation,
            "operands": {"a": a, "b": b}
        }))
    }

    fn definition(&self) -> ToolDefinition {
        let mut input_props = HashMap::new();
        input_props.insert(
            "operation".to_string(),
            JsonSchema::string()
                .with_description("Math operation: add, subtract, multiply, divide, power, modulo"),
        );
        input_props.insert(
            "a".to_string(),
            JsonSchema::number().with_description("First operand"),
        );
        input_props.insert(
            "b".to_string(),
            JsonSchema::number().with_description("Second operand"),
        );

        ToolDefinition::new(
            "calculator",
            "Perform mathematical calculations",
            JsonSchema::object()
                .with_properties(input_props)
                .with_required(vec![
                    "operation".to_string(),
                    "a".to_string(),
                    "b".to_string(),
                ]),
            JsonSchema::object().with_properties({
                let mut props = HashMap::new();
                props.insert("result".to_string(), JsonSchema::number());
                props.insert("operation".to_string(), JsonSchema::string());
                props
            }),
        )
        .with_category("math")
        .with_example(
            json!({"operation": "add", "a": 2, "b": 3}),
            json!({"result": 5.0, "operation": "add", "operands": {"a": 2, "b": 3}}),
            "Add two numbers",
        )
    }
}

// =============================================================================
// Search Tool
// =============================================================================

/// Simulated search tool (in real app, would query search API)
pub struct SearchTool;

impl SearchTool {
    pub fn new() -> Self {
        SearchTool
    }

    fn mock_search(&self, query: &str) -> Vec<SearchResult> {
        // Mock search results
        vec![
            SearchResult {
                title: format!("Result for: {}", query),
                url: format!("https://example.com/search?q={}", query),
                snippet: format!("This is a mock search result for '{}'", query),
            },
            SearchResult {
                title: format!("{} - Wikipedia", query),
                url: format!("https://en.wikipedia.org/wiki/{}", query),
                snippet: format!("Information about {} from Wikipedia", query),
            },
        ]
    }
}

#[derive(Debug, Clone, serde::Serialize)]
struct SearchResult {
    title: String,
    url: String,
    snippet: String,
}

impl Default for SearchTool {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Tool for SearchTool {
    async fn call(&self, input: Value) -> Result<Value> {
        let query = input
            .get("query")
            .and_then(|v| v.as_str())
            .ok_or_else(|| FlowgentraError::ToolError("Missing 'query' field".to_string()))?;

        let results = self.mock_search(query);

        Ok(json!({
            "query": query,
            "results": results,
            "count": results.len()
        }))
    }

    fn definition(&self) -> ToolDefinition {
        let mut input_props = HashMap::new();
        input_props.insert(
            "query".to_string(),
            JsonSchema::string().with_description("Search query"),
        );
        input_props.insert(
            "limit".to_string(),
            JsonSchema::integer().with_description("Maximum number of results (default: 10)"),
        );

        ToolDefinition::new(
            "search",
            "Search for information",
            JsonSchema::object()
                .with_properties(input_props)
                .with_required(vec!["query".to_string()]),
            JsonSchema::object()
                .with_properties({
                    let mut props = HashMap::new();
                    props.insert("query".to_string(), JsonSchema::string());
                    props.insert("results".to_string(), JsonSchema::array());
                    props.insert("count".to_string(), JsonSchema::integer());
                    props
                }),
        )
        .with_category("information")
        .with_example(
            json!({"query": "rust programming"}),
            json!({
                "query": "rust programming",
                "results": [
                    {"title": "Rust Official", "url": "https://rust-lang.org", "snippet": "The Rust programming language"}
                ],
                "count": 1
            }),
            "Search for Rust programming",
        )
    }
}

// =============================================================================
// Web Request Tool
// =============================================================================

/// Tool for making HTTP requests
pub struct WebRequestTool;

impl WebRequestTool {
    pub fn new() -> Self {
        WebRequestTool
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
        let url = input
            .get("url")
            .and_then(|v| v.as_str())
            .ok_or_else(|| FlowgentraError::ToolError("Missing 'url' field".to_string()))?;

        let method = input
            .get("method")
            .and_then(|v| v.as_str())
            .unwrap_or("GET");

        // Mock HTTP request - in real implementation would use reqwest
        let status_code = 200;
        let body = format!("Mock response from {}", url);

        Ok(json!({
            "url": url,
            "method": method,
            "status": status_code,
            "body": body
        }))
    }

    fn definition(&self) -> ToolDefinition {
        let mut input_props = HashMap::new();
        input_props.insert(
            "url".to_string(),
            JsonSchema::string().with_description("URL to request"),
        );
        input_props.insert(
            "method".to_string(),
            JsonSchema::string().with_description("HTTP method (GET, POST, etc)"),
        );
        input_props.insert(
            "headers".to_string(),
            JsonSchema::object().with_description("Optional HTTP headers"),
        );
        input_props.insert(
            "body".to_string(),
            JsonSchema::string().with_description("Optional request body"),
        );

        ToolDefinition::new(
            "web_request",
            "Make HTTP requests to web endpoints",
            JsonSchema::object()
                .with_properties(input_props)
                .with_required(vec!["url".to_string()]),
            JsonSchema::object()
                .with_properties({
                    let mut props = HashMap::new();
                    props.insert("url".to_string(), JsonSchema::string());
                    props.insert("method".to_string(), JsonSchema::string());
                    props.insert("status".to_string(), JsonSchema::integer());
                    props.insert("body".to_string(), JsonSchema::string());
                    props
                }),
        )
        .with_category("network")
        .with_example(
            json!({"url": "https://api.example.com/data"}),
            json!({"url": "https://api.example.com/data", "method": "GET", "status": 200, "body": "{}"}),
            "Fetch data from API",
        )
    }
}

// =============================================================================
// File Tool
// =============================================================================

/// Tool for file operations (read, write, list)
pub struct FilesTool;

impl FilesTool {
    pub fn new() -> Self {
        FilesTool
    }

    async fn read_file(&self, path: &str) -> Result<String> {
        fs::read_to_string(path)
            .map_err(|e| FlowgentraError::ToolError(format!("Failed to read file: {}", e)))
    }

    async fn write_file(&self, path: &str, content: &str) -> Result<()> {
        fs::write(path, content)
            .map_err(|e| FlowgentraError::ToolError(format!("Failed to write file: {}", e)))
    }

    async fn list_files(&self, path: &str) -> Result<Vec<String>> {
        let entries = fs::read_dir(path)
            .map_err(|e| FlowgentraError::ToolError(format!("Failed to list files: {}", e)))?;

        let files: Vec<String> = entries
            .filter_map(
                |entry: std::result::Result<std::fs::DirEntry, std::io::Error>| {
                    entry.ok().map(|e| e.path().to_string_lossy().to_string())
                },
            )
            .collect();

        Ok(files)
    }
}

impl Default for FilesTool {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Tool for FilesTool {
    async fn call(&self, input: Value) -> Result<Value> {
        let operation = input
            .get("operation")
            .and_then(|v| v.as_str())
            .ok_or_else(|| FlowgentraError::ToolError("Missing 'operation' field".to_string()))?;

        match operation {
            "read" => {
                let path = input.get("path").and_then(|v| v.as_str()).ok_or_else(|| {
                    FlowgentraError::ToolError("Missing 'path' field".to_string())
                })?;

                let content = self.read_file(path).await?;
                Ok(serde_json::json!({
                    "operation": "read",
                    "path": path,
                    "content": content
                }))
            }
            "write" => {
                let path = input.get("path").and_then(|v| v.as_str()).ok_or_else(|| {
                    FlowgentraError::ToolError("Missing 'path' field".to_string())
                })?;

                let content = input
                    .get("content")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| {
                        FlowgentraError::ToolError("Missing 'content' field".to_string())
                    })?;

                self.write_file(path, content).await?;
                Ok(serde_json::json!({
                    "operation": "write",
                    "path": path,
                    "success": true
                }))
            }
            "list" => {
                let path = input.get("path").and_then(|v| v.as_str()).unwrap_or(".");

                let files: Vec<String> = self.list_files(path).await?;
                Ok(serde_json::json!({
                    "operation": "list",
                    "path": path,
                    "files": files,
                    "count": files.len()
                }))
            }
            _ => Err(FlowgentraError::ToolError(format!(
                "Unknown file operation: {}",
                operation
            ))),
        }
    }

    fn definition(&self) -> ToolDefinition {
        let mut input_props = HashMap::new();
        input_props.insert(
            "operation".to_string(),
            JsonSchema::string().with_description("File operation: read, write, list"),
        );
        input_props.insert(
            "path".to_string(),
            JsonSchema::string().with_description("File or directory path"),
        );
        input_props.insert(
            "content".to_string(),
            JsonSchema::string().with_description("Content (for write operation)"),
        );

        ToolDefinition::new(
            "file",
            "Perform file operations (read, write, list)",
            JsonSchema::object()
                .with_properties(input_props)
                .with_required(vec!["operation".to_string(), "path".to_string()]),
            JsonSchema::object().with_properties({
                let mut props = HashMap::new();
                props.insert("operation".to_string(), JsonSchema::string());
                props.insert("path".to_string(), JsonSchema::string());
                props.insert("content".to_string(), JsonSchema::string());
                props.insert("files".to_string(), JsonSchema::array());
                props.insert("success".to_string(), JsonSchema::boolean());
                props
            }),
        )
        .with_category("file")
        .with_example(
            json!({"operation": "read", "path": "example.txt"}),
            json!({"operation": "read", "path": "example.txt", "content": "file contents"}),
            "Read a file",
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_calculator() {
        let calc = CalculatorTool::new();
        let result: Value = calc
            .call(json!({"operation": "add", "a": 2.0, "b": 3.0}))
            .await
            .unwrap();

        assert_eq!(result["result"], 5.0);
    }

    #[tokio::test]
    async fn test_search() {
        let search = SearchTool::new();
        let result: Value = search.call(json!({"query": "test"})).await.unwrap();

        assert_eq!(result["query"], "test");
        assert!(result["count"].as_u64().unwrap() > 0);
    }

    #[tokio::test]
    async fn test_web_request() {
        let web = WebRequestTool::new();
        let result: Value = web
            .call(json!({"url": "https://example.com"}))
            .await
            .unwrap();

        assert_eq!(result["status"], 200);
    }

    #[test]
    fn test_calculator_definition() {
        let calc = CalculatorTool::new();
        let def = calc.definition();
        assert_eq!(def.name, "calculator");
        assert!(def.categories.contains(&"math".to_string()));
    }
}
