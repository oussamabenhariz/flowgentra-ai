//! # State Validation & Schema
//!
//! Provides schema-based validation for JSON values.
//! With typed state, compile-time validation replaces most runtime checks,
//! but this module is kept for config-driven validation scenarios.

use serde_json::Value;
use std::collections::HashMap;
use std::fmt;

/// Type of a field in state
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FieldType {
    String,
    Integer,
    Float,
    Boolean,
    Array,
    Object,
    Null,
    Any,
}

impl FieldType {
    /// Check if a value matches this type
    pub fn matches(&self, value: &Value) -> bool {
        match self {
            FieldType::String => value.is_string(),
            FieldType::Integer => value.is_i64(),
            FieldType::Float => value.is_f64(),
            FieldType::Boolean => value.is_boolean(),
            FieldType::Array => value.is_array(),
            FieldType::Object => value.is_object(),
            FieldType::Null => value.is_null(),
            FieldType::Any => true,
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            FieldType::String => "string",
            FieldType::Integer => "integer",
            FieldType::Float => "float",
            FieldType::Boolean => "boolean",
            FieldType::Array => "array",
            FieldType::Object => "object",
            FieldType::Null => "null",
            FieldType::Any => "any",
        }
    }
}

impl fmt::Display for FieldType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// Custom validation function for a field
pub type FieldValidatorFn = Box<dyn Fn(&Value) -> Result<(), String> + Send + Sync>;

/// Validator for a single field
pub struct FieldValidator {
    pub field_type: FieldType,
    pub required: bool,
    pub immutable: bool,
    pub custom: Option<FieldValidatorFn>,
    pub description: Option<String>,
}

impl FieldValidator {
    pub fn new(field_type: FieldType) -> Self {
        Self {
            field_type,
            required: false,
            immutable: false,
            custom: None,
            description: None,
        }
    }

    pub fn required(mut self) -> Self {
        self.required = true;
        self
    }

    pub fn immutable(mut self) -> Self {
        self.immutable = true;
        self
    }

    pub fn with_validator(mut self, validator: FieldValidatorFn) -> Self {
        self.custom = Some(validator);
        self
    }

    pub fn with_description(mut self, desc: impl Into<String>) -> Self {
        self.description = Some(desc.into());
        self
    }

    pub fn validate(&self, value: &Value) -> Result<(), String> {
        if !self.field_type.matches(value) {
            return Err(format!(
                "Expected type {}, got {}",
                self.field_type,
                match value {
                    Value::String(_) => "string",
                    Value::Number(n) if n.is_i64() => "integer",
                    Value::Number(n) if n.is_f64() => "float",
                    Value::Bool(_) => "boolean",
                    Value::Array(_) => "array",
                    Value::Object(_) => "object",
                    Value::Null => "null",
                    _ => "unknown",
                }
            ));
        }
        if let Some(ref validator) = self.custom {
            validator(value)?;
        }
        Ok(())
    }
}

impl fmt::Debug for FieldValidator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FieldValidator")
            .field("field_type", &self.field_type)
            .field("required", &self.required)
            .field("immutable", &self.immutable)
            .field("custom", &self.custom.is_some())
            .field("description", &self.description)
            .finish()
    }
}

/// Validation error with detailed information
#[derive(Debug, Clone)]
pub struct ValidationError {
    pub field: String,
    pub message: String,
}

impl fmt::Display for ValidationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Validation error in field '{}': {}",
            self.field, self.message
        )
    }
}

impl std::error::Error for ValidationError {}

/// Schema for validating JSON state values.
///
/// With typed state, compile-time checks replace most of this.
/// Kept for config-driven validation and runtime schema checks.
#[derive(Debug)]
pub struct StateSchema {
    fields: HashMap<String, FieldValidator>,
    immutable_fields: Vec<String>,
}

impl Default for StateSchema {
    fn default() -> Self {
        Self::new()
    }
}

impl StateSchema {
    pub fn new() -> Self {
        Self {
            fields: HashMap::new(),
            immutable_fields: Vec::new(),
        }
    }

    pub fn require_field(mut self, name: impl Into<String>, field_type: FieldType) -> Self {
        self.fields
            .insert(name.into(), FieldValidator::new(field_type).required());
        self
    }

    pub fn optional_field(mut self, name: impl Into<String>, field_type: FieldType) -> Self {
        self.fields
            .insert(name.into(), FieldValidator::new(field_type));
        self
    }

    pub fn add_field(mut self, name: impl Into<String>, validator: FieldValidator) -> Self {
        self.fields.insert(name.into(), validator);
        self
    }

    pub fn immutable(mut self, field: impl Into<String>) -> Self {
        let field_name = field.into();
        self.immutable_fields.push(field_name.clone());
        if let Some(validator) = self.fields.get_mut(&field_name) {
            validator.immutable = true;
        }
        self
    }

    /// Validate a JSON value against this schema.
    pub fn validate_value(&self, value: &Value) -> Result<(), Vec<ValidationError>> {
        let mut errors = Vec::new();
        let obj = value.as_object();

        for (name, validator) in &self.fields {
            let field_value = obj.and_then(|o| o.get(name));
            if validator.required {
                match field_value {
                    None => {
                        errors.push(ValidationError {
                            field: name.clone(),
                            message: "Required field is missing".to_string(),
                        });
                    }
                    Some(v) => {
                        if let Err(msg) = validator.validate(v) {
                            errors.push(ValidationError {
                                field: name.clone(),
                                message: msg,
                            });
                        }
                    }
                }
            } else if let Some(v) = field_value {
                if let Err(msg) = validator.validate(v) {
                    errors.push(ValidationError {
                        field: name.clone(),
                        message: msg,
                    });
                }
            }
        }

        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }

    /// Validate a typed state by serializing to JSON first.
    pub fn validate_state<T: serde::Serialize>(
        &self,
        state: &T,
    ) -> Result<(), Vec<ValidationError>> {
        let value = serde_json::to_value(state).map_err(|e| {
            vec![ValidationError {
                field: "<root>".to_string(),
                message: format!("Failed to serialize state: {}", e),
            }]
        })?;
        self.validate_value(&value)
    }

    pub fn is_immutable(&self, field: &str) -> bool {
        self.immutable_fields.contains(&field.to_string())
    }

    pub fn get_field(&self, name: &str) -> Option<&FieldValidator> {
        self.fields.get(name)
    }

    pub fn fields(&self) -> &HashMap<String, FieldValidator> {
        &self.fields
    }
}

/// Helper function to create common validators
pub mod validators {
    use super::*;

    pub fn non_empty_string() -> FieldValidatorFn {
        Box::new(|value| {
            if let Some(s) = value.as_str() {
                if s.is_empty() {
                    Err("String must not be empty".to_string())
                } else {
                    Ok(())
                }
            } else {
                Err("Expected a string".to_string())
            }
        })
    }

    pub fn email() -> FieldValidatorFn {
        Box::new(|value| {
            if let Some(email) = value.as_str() {
                if email.contains('@') && email.contains('.') {
                    Ok(())
                } else {
                    Err("Invalid email format".to_string())
                }
            } else {
                Err("Expected a string".to_string())
            }
        })
    }

    pub fn number_range(min: i64, max: i64) -> FieldValidatorFn {
        Box::new(move |value| {
            if let Some(n) = value.as_i64() {
                if n >= min && n <= max {
                    Ok(())
                } else {
                    Err(format!("Number must be between {} and {}", min, max))
                }
            } else {
                Err("Expected a number".to_string())
            }
        })
    }

    pub fn string_length(min: usize, max: usize) -> FieldValidatorFn {
        Box::new(move |value| {
            if let Some(s) = value.as_str() {
                if s.len() >= min && s.len() <= max {
                    Ok(())
                } else {
                    Err(format!("String length must be between {} and {}", min, max))
                }
            } else {
                Err("Expected a string".to_string())
            }
        })
    }

    pub fn enum_values(allowed: Vec<&'static str>) -> FieldValidatorFn {
        Box::new(move |value| {
            if let Some(s) = value.as_str() {
                if allowed.contains(&s) {
                    Ok(())
                } else {
                    Err(format!("Value must be one of: {}", allowed.join(", ")))
                }
            } else {
                Err("Expected a string".to_string())
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn field_type_matching() {
        assert!(FieldType::String.matches(&Value::String("test".to_string())));
        assert!(FieldType::Integer.matches(&json!(42)));
        assert!(FieldType::Boolean.matches(&json!(true)));
        assert!(!FieldType::String.matches(&json!(42)));
    }

    #[test]
    fn basic_schema_validation() {
        let schema = StateSchema::new()
            .require_field("name", FieldType::String)
            .require_field("age", FieldType::Integer);

        let value = json!({"name": "Alice", "age": 30});
        assert!(schema.validate_value(&value).is_ok());
    }

    #[test]
    fn schema_missing_required_field() {
        let schema = StateSchema::new()
            .require_field("name", FieldType::String)
            .require_field("age", FieldType::Integer);

        let value = json!({"name": "Alice"});
        let result = schema.validate_value(&value);
        assert!(result.is_err());
    }

    #[test]
    fn immutable_fields() {
        let schema = StateSchema::new()
            .require_field("id", FieldType::String)
            .immutable("id");

        assert!(schema.is_immutable("id"));
        assert!(!schema.is_immutable("name"));
    }
}
