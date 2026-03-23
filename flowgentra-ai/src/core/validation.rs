//! JSON Schema Validation Errors
//!
//! Provides error types for state validation operations.
//! Schema validation is handled by StateSchema in the state module.

/// Errors from schema validation
#[derive(Debug, Clone)]
pub enum ValidationError {
    /// Required field is missing
    MissingRequired { field: String },

    /// Type mismatch (expected, got)
    TypeMismatch {
        field: String,
        expected: String,
        got: String,
    },

    /// Value out of valid range
    OutOfRange { field: String, message: String },

    /// Array validation failed
    ArrayValidationFailed {
        field: String,
        index: usize,
        message: String,
    },

    /// Custom validation failed
    CustomValidationFailed { message: String },

    /// Schema is invalid
    InvalidSchema { message: String },
}

impl std::fmt::Display for ValidationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ValidationError::MissingRequired { field } => {
                write!(f, "Missing required field: '{}'", field)
            }
            ValidationError::TypeMismatch {
                field,
                expected,
                got,
            } => {
                write!(
                    f,
                    "Type mismatch in '{}': expected {}, got {}",
                    field, expected, got
                )
            }
            ValidationError::OutOfRange { field, message } => {
                write!(f, "Out of range for '{}': {}", field, message)
            }
            ValidationError::ArrayValidationFailed {
                field,
                index,
                message,
            } => {
                write!(
                    f,
                    "Array validation failed for '{}[{}]': {}",
                    field, index, message
                )
            }
            ValidationError::CustomValidationFailed { message } => {
                write!(f, "Custom validation failed: {}", message)
            }
            ValidationError::InvalidSchema { message } => {
                write!(f, "Invalid schema: {}", message)
            }
        }
    }
}

impl std::error::Error for ValidationError {}
