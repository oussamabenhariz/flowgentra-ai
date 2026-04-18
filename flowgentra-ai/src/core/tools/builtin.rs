//! Built-in tools: Calculator and sandboxed File operations.

use super::{JsonSchema, Tool, ToolDefinition};
use crate::core::error::{FlowgentraError, Result};
use async_trait::async_trait;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::fs;

// =============================================================================
// Calculator Tool
// =============================================================================

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
// File Tool
// =============================================================================

/// Tool for file operations (read, write, list).
///
/// All paths are resolved relative to `sandbox_root` and must remain inside it.
pub struct FilesTool {
    sandbox_root: std::path::PathBuf,
}

impl FilesTool {
    pub fn new_with_root(root: impl AsRef<std::path::Path>) -> Result<Self> {
        let canonical = fs::canonicalize(root.as_ref()).map_err(|e| {
            FlowgentraError::ToolError(format!(
                "Cannot canonicalize sandbox root '{}': {}",
                root.as_ref().display(),
                e
            ))
        })?;
        Ok(Self {
            sandbox_root: canonical,
        })
    }

    fn safe_path(&self, user_path: &str) -> Result<std::path::PathBuf> {
        let p = std::path::Path::new(user_path);
        for component in p.components() {
            if component == std::path::Component::ParentDir {
                return Err(FlowgentraError::ToolError(
                    "Path traversal via '..' is not allowed".to_string(),
                ));
            }
        }
        let joined = if p.is_absolute() {
            p.to_path_buf()
        } else {
            self.sandbox_root.join(p)
        };
        let canonical = fs::canonicalize(&joined).map_err(|e| {
            FlowgentraError::ToolError(format!("Path '{}' is invalid: {}", user_path, e))
        })?;
        if !canonical.starts_with(&self.sandbox_root) {
            return Err(FlowgentraError::ToolError(format!(
                "Path '{}' resolves outside the allowed sandbox directory",
                user_path
            )));
        }
        Ok(canonical)
    }

    fn safe_write_path(&self, user_path: &str) -> Result<std::path::PathBuf> {
        let p = std::path::Path::new(user_path);
        for component in p.components() {
            if component == std::path::Component::ParentDir {
                return Err(FlowgentraError::ToolError(
                    "Path traversal via '..' is not allowed".to_string(),
                ));
            }
        }
        let joined = if p.is_absolute() {
            p.to_path_buf()
        } else {
            self.sandbox_root.join(p)
        };
        let parent = joined.parent().ok_or_else(|| {
            FlowgentraError::ToolError(format!("Path '{}' has no parent directory", user_path))
        })?;
        let canonical_parent = fs::canonicalize(parent).map_err(|e| {
            FlowgentraError::ToolError(format!(
                "Parent directory of '{}' is invalid: {}",
                user_path, e
            ))
        })?;
        if !canonical_parent.starts_with(&self.sandbox_root) {
            return Err(FlowgentraError::ToolError(format!(
                "Path '{}' resolves outside the allowed sandbox directory",
                user_path
            )));
        }
        let file_name = joined.file_name().ok_or_else(|| {
            FlowgentraError::ToolError(format!("Path '{}' has no file name", user_path))
        })?;
        Ok(canonical_parent.join(file_name))
    }

    async fn read_file(&self, path: &str) -> Result<String> {
        let safe = self.safe_path(path)?;
        fs::read_to_string(&safe)
            .map_err(|e| FlowgentraError::ToolError(format!("Failed to read file: {}", e)))
    }

    async fn write_file(&self, path: &str, content: &str) -> Result<()> {
        let safe = self.safe_write_path(path)?;
        fs::write(&safe, content)
            .map_err(|e| FlowgentraError::ToolError(format!("Failed to write file: {}", e)))
    }

    async fn list_files(&self, path: &str) -> Result<Vec<String>> {
        let safe = self.safe_path(path)?;
        let entries = fs::read_dir(&safe)
            .map_err(|e| FlowgentraError::ToolError(format!("Failed to list files: {}", e)))?;
        let files: Vec<String> = entries
            .filter_map(|entry| entry.ok().map(|e| e.path().to_string_lossy().to_string()))
            .collect();
        Ok(files)
    }
}

impl Default for FilesTool {
    fn default() -> Self {
        Self::new_with_root(std::env::current_dir().expect("Failed to get current directory"))
            .expect("Failed to canonicalize current directory")
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
                Ok(json!({"operation": "read", "path": path, "content": content}))
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
                Ok(json!({"operation": "write", "path": path, "success": true}))
            }
            "list" => {
                let path = input.get("path").and_then(|v| v.as_str()).unwrap_or(".");
                let files = self.list_files(path).await?;
                let count = files.len();
                Ok(json!({"operation": "list", "path": path, "files": files, "count": count}))
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
            JsonSchema::string().with_description("Content for write operation"),
        );

        ToolDefinition::new(
            "file",
            "Perform file operations (read, write, list) within a sandboxed directory",
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
        let result = calc
            .call(json!({"operation": "add", "a": 2.0, "b": 3.0}))
            .await
            .unwrap();
        assert_eq!(result["result"], 5.0);
    }

    #[test]
    fn test_calculator_definition() {
        let calc = CalculatorTool::new();
        let def = calc.definition();
        assert_eq!(def.name, "calculator");
        assert!(def.categories.contains(&"math".to_string()));
    }
}
