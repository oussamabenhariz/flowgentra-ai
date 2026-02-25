//! # State Validation & Schema
//!
//! Provides schema-based validation and constraints for state objects.
//!
//! ## Features
//!
//! - **Field Requirements** - Mark fields as required/optional
//! - **Type Checking** - Validate field types
//! - **Custom Validators** - Add custom validation logic
//! - **Schema Definition** - Define what valid state looks like
//! - **Immutability** - Mark fields as read-only
//!
//! ## Example
//!
//! ```ignore
//! use erenflow_ai::core::state_validation::{StateSchema, FieldValidator, FieldType};
//!
//! let schema = StateSchema::new()
//!     .require_field("user_id", FieldType::Integer)
//!     .require_field("email", FieldType::String)
//!     .optional_field("preferences", FieldType::Object)
//!     .immutable("user_id");
//! ```

use crate::core::state::State;
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
    Any, // Accept any type
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

    /// Get string representation
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
    /// Expected type of the field
    pub field_type: FieldType,

    /// Whether the field is required
    pub required: bool,

    /// Whether the field can be modified
    pub immutable: bool,

    /// Custom validation function
    pub custom: Option<FieldValidatorFn>,

    /// Description of the field
    pub description: Option<String>,
}

impl FieldValidator {
    /// Create a new field validator
    pub fn new(field_type: FieldType) -> Self {
        Self {
            field_type,
            required: false,
            immutable: false,
            custom: None,
            description: None,
        }
    }

    /// Mark field as required
    pub fn required(mut self) -> Self {
        self.required = true;
        self
    }

    /// Mark field as immutable
    pub fn immutable(mut self) -> Self {
        self.immutable = true;
        self
    }

    /// Add custom validation
    pub fn with_validator(mut self, validator: FieldValidatorFn) -> Self {
        self.custom = Some(validator);
        self
    }

    /// Add description
    pub fn with_description(mut self, desc: impl Into<String>) -> Self {
        self.description = Some(desc.into());
        self
    }

    /// Validate a value
    pub fn validate(&self, value: &Value) -> Result<(), String> {
        // Check type
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

        // Run custom validation
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

/// Schema for validating state objects
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
    /// Create a new empty schema
    pub fn new() -> Self {
        Self {
            fields: HashMap::new(),
            immutable_fields: Vec::new(),
        }
    }

    /// Add a required field with specified type
    pub fn require_field(mut self, name: impl Into<String>, field_type: FieldType) -> Self {
        self.fields
            .insert(name.into(), FieldValidator::new(field_type).required());
        self
    }

    /// Add an optional field with specified type
    pub fn optional_field(mut self, name: impl Into<String>, field_type: FieldType) -> Self {
        self.fields
            .insert(name.into(), FieldValidator::new(field_type));
        self
    }

    /// Add a custom field validator
    pub fn add_field(mut self, name: impl Into<String>, validator: FieldValidator) -> Self {
        self.fields.insert(name.into(), validator);
        self
    }

    /// Mark a field as immutable
    pub fn immutable(mut self, field: impl Into<String>) -> Self {
        let field_name = field.into();
        self.immutable_fields.push(field_name.clone());
        if let Some(validator) = self.fields.get_mut(&field_name) {
            validator.immutable = true;
        }
        self
    }

    /// Validate a state object
    pub fn validate(&self, state: &State) -> Result<(), Vec<ValidationError>> {
        let mut errors = Vec::new();

        // Check all required fields exist
        for (name, validator) in &self.fields {
            if validator.required {
                match state.get(name) {
                    None => {
                        errors.push(ValidationError {
                            field: name.clone(),
                            message: "Required field is missing".to_string(),
                        });
                    }
                    Some(value) => {
                        if let Err(msg) = validator.validate(value) {
                            errors.push(ValidationError {
                                field: name.clone(),
                                message: msg,
                            });
                        }
                    }
                }
            } else {
                // Optional fields: validate if present
                if let Some(value) = state.get(name) {
                    if let Err(msg) = validator.validate(value) {
                        errors.push(ValidationError {
                            field: name.clone(),
                            message: msg,
                        });
                    }
                }
            }
        }

        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }

    /// Check if a field is immutable
    pub fn is_immutable(&self, field: &str) -> bool {
        self.immutable_fields.contains(&field.to_string())
    }

    /// Get field validator
    pub fn get_field(&self, name: &str) -> Option<&FieldValidator> {
        self.fields.get(name)
    }

    /// Get all defined fields
    pub fn fields(&self) -> &HashMap<String, FieldValidator> {
        &self.fields
    }
}

/// Helper function to create common validators
pub mod validators {
    use super::*;

    /// Non-empty string validator
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

    /// Email validator (basic)
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

    /// Number range validator
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

    /// Length validator for strings
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

    /// Enum validator
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
    fn field_validator_creation() {
        let validator = FieldValidator::new(FieldType::String).required();
        assert!(validator.required);
        assert!(!validator.immutable);
    }

    #[test]
    fn basic_schema_validation() {
        let schema = StateSchema::new()
            .require_field("name", FieldType::String)
            .require_field("age", FieldType::Integer);

        let mut state = State::new();
        state.set("name", json!("Alice"));
        state.set("age", json!(30));

        assert!(schema.validate(&state).is_ok());
    }

    #[test]
    fn schema_missing_required_field() {
        let schema = StateSchema::new()
            .require_field("name", FieldType::String)
            .require_field("age", FieldType::Integer);

        let mut state = State::new();
        state.set("name", json!("Alice"));
        // age is missing

        let result = schema.validate(&state);
        assert!(result.is_err());
        let errors = result.unwrap_err();
        assert_eq!(errors.len(), 1);
        assert_eq!(errors[0].field, "age");
    }

    #[test]
    fn schema_type_mismatch() {
        let schema = StateSchema::new().require_field("age", FieldType::Integer);

        let mut state = State::new();
        state.set("age", json!("thirty")); // Wrong type

        let result = schema.validate(&state);
        assert!(result.is_err());
    }

    #[test]
    fn optional_fields() {
        let schema = StateSchema::new()
            .require_field("name", FieldType::String)
            .optional_field("nickname", FieldType::String);

        let mut state = State::new();
        state.set("name", json!("Alice"));
        // nickname is optional and not provided

        assert!(schema.validate(&state).is_ok());
    }

    #[test]
    fn immutable_fields() {
        let schema = StateSchema::new()
            .require_field("id", FieldType::String)
            .immutable("id");

        assert!(schema.is_immutable("id"));
        assert!(!schema.is_immutable("name"));
    }

    #[test]
    fn custom_validator() {
        let schema = StateSchema::new().add_field(
            "email",
            FieldValidator::new(FieldType::String).with_validator(validators::email()),
        );

        let mut state = State::new();
        state.set("email", json!("test@example.com"));
        assert!(schema.validate(&state).is_ok());

        let mut state = State::new();
        state.set("email", json!("invalid-email"));
        assert!(schema.validate(&state).is_err());
    }
}
