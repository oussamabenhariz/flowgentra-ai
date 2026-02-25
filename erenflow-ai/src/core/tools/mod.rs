//! # Tool System for ErenFlowAI
//!
//! Provides a trait-based tool abstraction that works seamlessly with LLM function calling.
//!
//! ## Core Concepts
//!
//! - **Tool Trait**: Unified interface for all tools with `async fn call(&self, input: Value) -> Result<Value>`
//! - **ToolDefinition**: Metadata (name, description, input/output schemas)
//! - **ToolRegistry**: Central registry for managing and discovering tools
//! - **JSON Schema**: Validation of tool inputs and outputs
//! - **Function Calling**: Automatic routing of LLM function calls to tools
//!
//! ## Quick Example
//!
//! ```ignore
//! // Define a tool
//! let calculator = CalculatorTool::new();
//!
//! // Register it
//! let mut registry = ToolRegistry::new();
//! registry.register("calculator", Box::new(calculator))?;
//!
//! // Use in a handler
//! let result = registry.call_tool("calculator", json!({"operation": "add", "a": 2, "b": 3})).await?;
//! assert_eq!(result["result"], 5);
//! ```

use crate::core::error::{ErenFlowError, Result};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::sync::Arc;

pub mod builtin;

pub use builtin::{CalculatorTool, FilesTool, SearchTool, WebRequestTool};

// =============================================================================
// Tool Trait
// =============================================================================

/// Universal tool interface - all tools implement this
#[async_trait]
pub trait Tool: Send + Sync {
    /// Execute the tool with the given input
    ///
    /// # Arguments
    /// * `input` - JSON value containing tool parameters
    ///
    /// # Returns
    /// * `Result<Value>` - JSON result or error
    async fn call(&self, input: Value) -> Result<Value>;

    /// Get tool definition (name, description, schema)
    fn definition(&self) -> ToolDefinition;
}

// =============================================================================
// JSON Schema Types
// =============================================================================

/// Represents a JSON schema for input/output validation
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct JsonSchema {
    /// Schema type (object, string, number, boolean, array, etc.)
    #[serde(rename = "type")]
    pub schema_type: String,

    /// Description of what this schema represents
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,

    /// For object types: properties definition
    #[serde(skip_serializing_if = "Option::is_none")]
    pub properties: Option<HashMap<String, JsonSchema>>,

    /// For object types: which properties are required
    #[serde(skip_serializing_if = "Option::is_none")]
    pub required: Option<Vec<String>>,

    /// For array types: schema of items
    #[serde(skip_serializing_if = "Option::is_none")]
    pub items: Option<Box<JsonSchema>>,

    /// Allowed enum values
    #[serde(skip_serializing_if = "Option::is_none")]
    pub enumeration: Option<Vec<Value>>,

    /// Minimum value (for numbers)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub minimum: Option<f64>,

    /// Maximum value (for numbers)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub maximum: Option<f64>,

    /// Pattern (for strings)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pattern: Option<String>,

    /// Minimum length (for strings/arrays)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub min_length: Option<usize>,

    /// Maximum length (for strings/arrays)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_length: Option<usize>,
}

impl JsonSchema {
    /// Create an object schema
    pub fn object() -> Self {
        JsonSchema {
            schema_type: "object".to_string(),
            properties: None,
            required: None,
            description: None,
            items: None,
            enumeration: None,
            minimum: None,
            maximum: None,
            pattern: None,
            min_length: None,
            max_length: None,
        }
    }

    /// Create a string schema
    pub fn string() -> Self {
        JsonSchema {
            schema_type: "string".to_string(),
            description: None,
            properties: None,
            required: None,
            items: None,
            enumeration: None,
            minimum: None,
            maximum: None,
            pattern: None,
            min_length: None,
            max_length: None,
        }
    }

    /// Create a number schema
    pub fn number() -> Self {
        JsonSchema {
            schema_type: "number".to_string(),
            description: None,
            properties: None,
            required: None,
            items: None,
            enumeration: None,
            minimum: None,
            maximum: None,
            pattern: None,
            min_length: None,
            max_length: None,
        }
    }

    /// Create an integer schema
    pub fn integer() -> Self {
        JsonSchema {
            schema_type: "integer".to_string(),
            description: None,
            properties: None,
            required: None,
            items: None,
            enumeration: None,
            minimum: None,
            maximum: None,
            pattern: None,
            min_length: None,
            max_length: None,
        }
    }

    /// Create a boolean schema
    pub fn boolean() -> Self {
        JsonSchema {
            schema_type: "boolean".to_string(),
            description: None,
            properties: None,
            required: None,
            items: None,
            enumeration: None,
            minimum: None,
            maximum: None,
            pattern: None,
            min_length: None,
            max_length: None,
        }
    }

    /// Create an array schema
    pub fn array() -> Self {
        JsonSchema {
            schema_type: "array".to_string(),
            items: Some(Box::new(JsonSchema::string())),
            description: None,
            properties: None,
            required: None,
            enumeration: None,
            minimum: None,
            maximum: None,
            pattern: None,
            min_length: None,
            max_length: None,
        }
    }

    /// Add description
    pub fn with_description(mut self, desc: impl Into<String>) -> Self {
        self.description = Some(desc.into());
        self
    }

    /// Add properties (for objects)
    pub fn with_properties(mut self, props: HashMap<String, JsonSchema>) -> Self {
        self.properties = Some(props);
        self
    }

    /// Add required fields (for objects)
    pub fn with_required(mut self, fields: Vec<String>) -> Self {
        self.required = Some(fields);
        self
    }

    /// Add items schema (for arrays)
    pub fn with_items(mut self, items: JsonSchema) -> Self {
        self.items = Some(Box::new(items));
        self
    }

    /// Validate a value against this schema
    pub fn validate(&self, value: &Value) -> Result<()> {
        match self.schema_type.as_str() {
            "object" => {
                if !value.is_object() {
                    return Err(ErenFlowError::ValidationError(
                        "Expected object".to_string(),
                    ));
                }
                if let Some(required) = &self.required {
                    for field in required {
                        if value.get(field).is_none() {
                            return Err(ErenFlowError::ValidationError(format!(
                                "Missing required field: {}",
                                field
                            )));
                        }
                    }
                }
                if let Some(properties) = &self.properties {
                    for (key, schema) in properties {
                        if let Some(val) = value.get(key) {
                            schema.validate(val)?;
                        }
                    }
                }
                Ok(())
            }
            "string" => {
                if !value.is_string() {
                    return Err(ErenFlowError::ValidationError(
                        "Expected string".to_string(),
                    ));
                }
                let s = value.as_str().unwrap();
                if let Some(min) = self.min_length {
                    if s.len() < min {
                        return Err(ErenFlowError::ValidationError(format!(
                            "String too short (min: {})",
                            min
                        )));
                    }
                }
                if let Some(max) = self.max_length {
                    if s.len() > max {
                        return Err(ErenFlowError::ValidationError(format!(
                            "String too long (max: {})",
                            max
                        )));
                    }
                }
                Ok(())
            }
            "number" | "integer" => {
                if !value.is_number() {
                    return Err(ErenFlowError::ValidationError(format!(
                        "Expected {}",
                        self.schema_type
                    )));
                }
                if let Some(val) = value.as_f64() {
                    if let Some(min) = self.minimum {
                        if val < min {
                            return Err(ErenFlowError::ValidationError(format!(
                                "Number too small (min: {})",
                                min
                            )));
                        }
                    }
                    if let Some(max) = self.maximum {
                        if val > max {
                            return Err(ErenFlowError::ValidationError(format!(
                                "Number too large (max: {})",
                                max
                            )));
                        }
                    }
                }
                Ok(())
            }
            "boolean" => {
                if !value.is_boolean() {
                    return Err(ErenFlowError::ValidationError(
                        "Expected boolean".to_string(),
                    ));
                }
                Ok(())
            }
            "array" => {
                if !value.is_array() {
                    return Err(ErenFlowError::ValidationError("Expected array".to_string()));
                }
                if let Some(items_schema) = &self.items {
                    for item in value.as_array().unwrap() {
                        items_schema.validate(item)?;
                    }
                }
                Ok(())
            }
            _ => Ok(()),
        }
    }
}

// =============================================================================
// Tool Definition
// =============================================================================

/// Complete metadata about a tool
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDefinition {
    /// Tool name (used for function calling)
    pub name: String,

    /// Human-readable description
    pub description: String,

    /// Input parameters schema
    pub input_schema: JsonSchema,

    /// Output schema
    pub output_schema: JsonSchema,

    /// List of categories/tags
    #[serde(default)]
    pub categories: Vec<String>,

    /// Example usage
    #[serde(default)]
    pub examples: Vec<ToolExample>,
}

impl ToolDefinition {
    /// Create a new tool definition
    pub fn new(
        name: impl Into<String>,
        description: impl Into<String>,
        input_schema: JsonSchema,
        output_schema: JsonSchema,
    ) -> Self {
        ToolDefinition {
            name: name.into(),
            description: description.into(),
            input_schema,
            output_schema,
            categories: Vec::new(),
            examples: Vec::new(),
        }
    }

    /// Add category/tag
    pub fn with_category(mut self, category: impl Into<String>) -> Self {
        self.categories.push(category.into());
        self
    }

    /// Add example
    pub fn with_example(
        mut self,
        input: Value,
        output: Value,
        description: impl Into<String>,
    ) -> Self {
        self.examples.push(ToolExample {
            input,
            output,
            description: description.into(),
        });
        self
    }
}

/// Example usage of a tool
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolExample {
    /// Example input
    pub input: Value,

    /// Example output
    pub output: Value,

    /// Description of the example
    pub description: String,
}

// =============================================================================
// Tool Registry
// =============================================================================

/// Central registry for managing tools
pub struct ToolRegistry {
    tools: HashMap<String, Arc<dyn Tool>>,
}

impl ToolRegistry {
    /// Create a new tool registry
    pub fn new() -> Self {
        ToolRegistry {
            tools: HashMap::new(),
        }
    }

    /// Register a tool
    pub fn register(&mut self, name: &str, tool: Arc<dyn Tool>) -> Result<()> {
        if self.tools.contains_key(name) {
            return Err(ErenFlowError::ToolError(format!(
                "Tool '{}' already registered",
                name
            )));
        }
        self.tools.insert(name.to_string(), tool);
        Ok(())
    }

    /// Unregister a tool
    pub fn unregister(&mut self, name: &str) -> Result<()> {
        self.tools
            .remove(name)
            .ok_or_else(|| ErenFlowError::ToolError(format!("Tool '{}' not found", name)))?;
        Ok(())
    }

    /// Get a tool definition
    pub fn get_definition(&self, name: &str) -> Result<ToolDefinition> {
        self.tools
            .get(name)
            .ok_or_else(|| ErenFlowError::ToolError(format!("Tool '{}' not found", name)))
            .map(|tool| tool.definition())
    }

    /// Get all tool definitions
    pub fn list_definitions(&self) -> Vec<ToolDefinition> {
        self.tools
            .values()
            .map(|tool: &Arc<dyn Tool>| tool.definition())
            .collect()
    }

    /// Call a tool by name
    pub async fn call_tool(&self, name: &str, input: Value) -> Result<Value> {
        let tool = self
            .tools
            .get(name)
            .ok_or_else(|| ErenFlowError::ToolError(format!("Tool '{}' not found", name)))?;

        // Validate input against schema
        let definition = tool.definition();
        definition.input_schema.validate(&input)?;

        // Call the tool
        let result = tool.call(input).await?;

        // Validate output against schema
        definition.output_schema.validate(&result)?;

        Ok(result)
    }

    /// Validate tool input without calling
    pub fn validate_input(&self, name: &str, input: &Value) -> Result<()> {
        let definition = self.get_definition(name)?;
        definition.input_schema.validate(input)
    }

    /// Get tool count
    pub fn len(&self) -> usize {
        self.tools.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.tools.is_empty()
    }

    /// Clear all tools
    pub fn clear(&mut self) {
        self.tools.clear();
    }
}

impl Default for ToolRegistry {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// Tool Call from LLM
// =============================================================================

/// Represents a function call made by the LLM
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCallRequest {
    /// Unique ID for this call
    pub id: String,

    /// Tool name to call
    pub name: String,

    /// Arguments (JSON)
    pub input: Value,
}

/// Represents a tool call result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCallResult {
    /// Call ID that this result corresponds to
    pub call_id: String,

    /// Tool name
    pub tool_name: String,

    /// The result (or error message)
    pub result: Value,

    /// Whether this was successful
    pub success: bool,
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_json_schema_string_validation() {
        let schema = JsonSchema::string();
        assert!(schema.validate(&Value::String("test".to_string())).is_ok());
        assert!(schema.validate(&Value::Number(123.into())).is_err());
    }

    #[test]
    fn test_json_schema_number_validation() {
        let schema = JsonSchema::number();
        assert!(schema.validate(&Value::Number(123u64.into())).is_ok());
        assert!(schema.validate(&Value::String("test".to_string())).is_err());
    }

    #[test]
    fn test_json_schema_object_validation() {
        let mut props = HashMap::new();
        props.insert("name".to_string(), JsonSchema::string());
        props.insert("age".to_string(), JsonSchema::integer());

        let schema = JsonSchema::object()
            .with_properties(props)
            .with_required(vec!["name".to_string()]);

        let valid = json!({
            "name": "John",
            "age": 30
        });
        assert!(schema.validate(&valid).is_ok());

        let missing_required = json!({
            "age": 30
        });
        assert!(schema.validate(&missing_required).is_err());
    }

    #[tokio::test]
    async fn test_tool_registry() {
        let registry = ToolRegistry::new();

        // Try to call non-existent tool
        let result: Result<Value> = registry.call_tool("nonexistent", json!({})).await;
        assert!(result.is_err());
    }
}
