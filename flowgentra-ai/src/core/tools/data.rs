//! Structured data tools: JSON introspection and CSV querying.

use super::{JsonSchema, Tool, ToolDefinition};
use crate::prelude::*;
use async_trait::async_trait;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::io::Cursor;

// =============================================================================
// JsonGetValueTool
// =============================================================================

/// Extract a value from a JSON string using dot-notation path.
pub struct JsonGetValueTool;

#[async_trait]
impl Tool for JsonGetValueTool {
    async fn call(&self, input: Value) -> Result<Value> {
        let json_str = input
            .get("json")
            .and_then(|v| v.as_str())
            .ok_or_else(|| FlowgentraError::ToolError("Missing 'json' field".to_string()))?;

        let path = input
            .get("path")
            .and_then(|v| v.as_str())
            .ok_or_else(|| FlowgentraError::ToolError("Missing 'path' field".to_string()))?;

        let parsed: Value = serde_json::from_str(json_str)
            .map_err(|e| FlowgentraError::ToolError(format!("Invalid JSON: {}", e)))?;

        let mut current = &parsed;
        for key in path.split('.') {
            match current.get(key) {
                Some(next) => current = next,
                None => {
                    return Ok(json!({"value": null, "path": path, "found": false}));
                }
            }
        }

        Ok(json!({"value": current, "path": path, "found": true}))
    }

    fn definition(&self) -> ToolDefinition {
        let mut props = HashMap::new();
        props.insert(
            "json".to_string(),
            JsonSchema::string().with_description("JSON string to query"),
        );
        props.insert(
            "path".to_string(),
            JsonSchema::string().with_description("Dot-notation path, e.g. \"user.address.city\""),
        );

        ToolDefinition::new(
            "json_get",
            "Extract a value from a JSON string using a dot-notation path",
            JsonSchema::object()
                .with_properties(props)
                .with_required(vec!["json".to_string(), "path".to_string()]),
            JsonSchema::object(),
        )
        .with_category("data")
        .with_example(
            json!({"json": "{\"user\":{\"name\":\"Alice\"}}", "path": "user.name"}),
            json!({"value": "Alice", "path": "user.name", "found": true}),
            "Extract nested value",
        )
    }
}

// =============================================================================
// JsonListKeysTool
// =============================================================================

/// List keys of a JSON object up to a given depth.
pub struct JsonListKeysTool;

fn collect_keys(
    val: &Value,
    depth: usize,
    current_depth: usize,
    prefix: &str,
    keys: &mut Vec<String>,
) {
    if let Value::Object(map) = val {
        for (k, v) in map {
            let full_key = if prefix.is_empty() {
                k.clone()
            } else {
                format!("{}.{}", prefix, k)
            };
            keys.push(full_key.clone());
            if current_depth < depth {
                collect_keys(v, depth, current_depth + 1, &full_key, keys);
            }
        }
    }
}

#[async_trait]
impl Tool for JsonListKeysTool {
    async fn call(&self, input: Value) -> Result<Value> {
        let json_str = input
            .get("json")
            .and_then(|v| v.as_str())
            .ok_or_else(|| FlowgentraError::ToolError("Missing 'json' field".to_string()))?;

        let depth = input.get("depth").and_then(|v| v.as_u64()).unwrap_or(1) as usize;

        let parsed: Value = serde_json::from_str(json_str)
            .map_err(|e| FlowgentraError::ToolError(format!("Invalid JSON: {}", e)))?;

        let mut keys = Vec::new();
        collect_keys(&parsed, depth, 1, "", &mut keys);

        Ok(json!({"keys": keys, "count": keys.len()}))
    }

    fn definition(&self) -> ToolDefinition {
        let mut props = HashMap::new();
        props.insert(
            "json".to_string(),
            JsonSchema::string().with_description("JSON string to inspect"),
        );
        props.insert(
            "depth".to_string(),
            JsonSchema::integer()
                .with_description("How many levels deep to list keys (default: 1)"),
        );

        ToolDefinition::new(
            "json_keys",
            "List all keys in a JSON object up to a given nesting depth",
            JsonSchema::object()
                .with_properties(props)
                .with_required(vec!["json".to_string()]),
            JsonSchema::object(),
        )
        .with_category("data")
        .with_example(
            json!({"json": "{\"a\":1,\"b\":{\"c\":2}}", "depth": 2}),
            json!({"keys": ["a", "b", "b.c"], "count": 3}),
            "List keys with depth 2",
        )
    }
}

// =============================================================================
// CsvQueryTool
// =============================================================================

/// Parse a CSV string and optionally filter rows by a column=value expression.
pub struct CsvQueryTool;

#[async_trait]
impl Tool for CsvQueryTool {
    async fn call(&self, input: Value) -> Result<Value> {
        let csv_str = input
            .get("csv")
            .and_then(|v| v.as_str())
            .ok_or_else(|| FlowgentraError::ToolError("Missing 'csv' field".to_string()))?
            .to_string();

        let filter = input
            .get("filter")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());

        // Parse filter expression "column=value"
        let filter_parts: Option<(String, String)> = filter.as_deref().and_then(|f| {
            let mut parts = f.splitn(2, '=');
            let col = parts.next()?.trim().to_string();
            let val = parts.next()?.trim().to_string();
            Some((col, val))
        });

        let mut reader = csv::Reader::from_reader(Cursor::new(csv_str.as_bytes()));
        let headers: Vec<String> = reader
            .headers()
            .map_err(|e| FlowgentraError::ToolError(format!("CSV header error: {}", e)))?
            .iter()
            .map(|s| s.to_string())
            .collect();

        let mut rows: Vec<Value> = Vec::new();
        for record in reader.records() {
            let record = record
                .map_err(|e| FlowgentraError::ToolError(format!("CSV parse error: {}", e)))?;

            let row: serde_json::Map<String, Value> = headers
                .iter()
                .zip(record.iter())
                .map(|(h, v)| (h.clone(), Value::String(v.to_string())))
                .collect();

            if let Some((ref col, ref val)) = filter_parts {
                if row.get(col).and_then(|v| v.as_str()) != Some(val.as_str()) {
                    continue;
                }
            }

            rows.push(Value::Object(row));
        }

        let count = rows.len();
        Ok(json!({"rows": rows, "count": count}))
    }

    fn definition(&self) -> ToolDefinition {
        let mut props = HashMap::new();
        props.insert(
            "csv".to_string(),
            JsonSchema::string().with_description("CSV data as a string (first row = headers)"),
        );
        props.insert(
            "filter".to_string(),
            JsonSchema::string().with_description("Optional filter expression: \"column=value\""),
        );

        ToolDefinition::new(
            "csv_query",
            "Parse a CSV string and optionally filter rows by a column=value expression",
            JsonSchema::object()
                .with_properties(props)
                .with_required(vec!["csv".to_string()]),
            JsonSchema::object(),
        )
        .with_category("data")
        .with_example(
            json!({"csv": "name,age\nAlice,30\nBob,25", "filter": "name=Alice"}),
            json!({"rows": [{"name": "Alice", "age": "30"}], "count": 1}),
            "Filter CSV rows",
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_json_get() {
        let tool = JsonGetValueTool;
        let result = tool
            .call(json!({"json": "{\"user\":{\"name\":\"Alice\"}}", "path": "user.name"}))
            .await
            .unwrap();
        assert_eq!(result["value"], "Alice");
        assert_eq!(result["found"], true);
    }

    #[tokio::test]
    async fn test_json_get_missing() {
        let tool = JsonGetValueTool;
        let result = tool
            .call(json!({"json": "{\"a\":1}", "path": "b.c"}))
            .await
            .unwrap();
        assert_eq!(result["found"], false);
    }

    #[tokio::test]
    async fn test_json_keys() {
        let tool = JsonListKeysTool;
        let result = tool
            .call(json!({"json": "{\"a\":1,\"b\":{\"c\":2}}", "depth": 2}))
            .await
            .unwrap();
        let keys: Vec<String> = result["keys"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_str().unwrap().to_string())
            .collect();
        assert!(keys.contains(&"a".to_string()));
        assert!(keys.contains(&"b".to_string()));
        assert!(keys.contains(&"b.c".to_string()));
    }

    #[tokio::test]
    async fn test_csv_query() {
        let tool = CsvQueryTool;
        let result = tool
            .call(json!({"csv": "name,age\nAlice,30\nBob,25", "filter": "name=Alice"}))
            .await
            .unwrap();
        assert_eq!(result["count"], 1);
        assert_eq!(result["rows"][0]["name"], "Alice");
    }

    #[tokio::test]
    async fn test_csv_no_filter() {
        let tool = CsvQueryTool;
        let result = tool
            .call(json!({"csv": "name,age\nAlice,30\nBob,25"}))
            .await
            .unwrap();
        assert_eq!(result["count"], 2);
    }
}
