//! # State Helper Trait
//!
//! Convenience methods for accessing state values without repetitive chains.

use crate::core::error::{FlowgentraError, Result};
use crate::core::state::SharedState;
use serde_json::Value;

/// Convenience methods for SharedState
pub trait StateExtMethods {
    /// Get a string value with default fallback
    fn get_string(&self, key: &str, default: &str) -> String;

    /// Get a string value, error if not found
    fn require_string(&self, key: &str) -> Result<String>;

    /// Get a number (i32) with default
    fn get_int(&self, key: &str, default: i32) -> i32;

    /// Get a number (f64) with default
    fn get_float(&self, key: &str, default: f64) -> f64;

    /// Get a boolean with default
    fn get_bool(&self, key: &str, default: bool) -> bool;

    /// Get a number, error if not found
    fn require_number(&self, key: &str) -> Result<f64>;

    /// Check if key exists
    fn has(&self, key: &str) -> bool;

    /// Get array of strings
    fn get_string_array(&self, key: &str) -> Vec<String>;

    /// Update multiple values at once
    fn set_all(&self, updates: &[(&str, Value)]);
}

impl StateExtMethods for SharedState {
    fn get_string(&self, key: &str, default: &str) -> String {
        self.get(key)
            .and_then(|v| v.as_str().map(|s| s.to_string()))
            .unwrap_or_else(|| default.to_string())
    }

    fn require_string(&self, key: &str) -> Result<String> {
        self.get(key)
            .and_then(|v| v.as_str().map(|s| s.to_string()))
            .ok_or_else(|| {
                FlowgentraError::ValidationError(format!(
                    "Required string field '{}' not found",
                    key
                ))
            })
    }

    fn get_int(&self, key: &str, default: i32) -> i32 {
        self.get(key)
            .and_then(|v| v.as_i64().map(|n| n as i32))
            .unwrap_or(default)
    }

    fn get_float(&self, key: &str, default: f64) -> f64 {
        self.get(key).and_then(|v| v.as_f64()).unwrap_or(default)
    }

    fn get_bool(&self, key: &str, default: bool) -> bool {
        self.get(key).and_then(|v| v.as_bool()).unwrap_or(default)
    }

    fn require_number(&self, key: &str) -> Result<f64> {
        self.get(key).and_then(|v| v.as_f64()).ok_or_else(|| {
            FlowgentraError::ValidationError(format!("Required numeric field '{}' not found", key))
        })
    }

    fn has(&self, key: &str) -> bool {
        self.get(key).is_some()
    }

    fn get_string_array(&self, key: &str) -> Vec<String> {
        self.get(key)
            .map(|v| {
                if let Value::Array(arr) = v {
                    arr.into_iter()
                        .filter_map(|val| val.as_str().map(|s| s.to_string()))
                        .collect()
                } else {
                    Vec::new()
                }
            })
            .unwrap_or_default()
    }

    fn set_all(&self, updates: &[(&str, Value)]) {
        for (key, value) in updates {
            self.set(*key, value.clone());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_get_string_with_default() {
        let state = SharedState::default();
        state.set("name", json!("Alice"));
        assert_eq!(state.get_string("name", "Bob"), "Alice");
        assert_eq!(state.get_string("missing", "Bob"), "Bob");
    }

    #[test]
    fn test_get_int() {
        let state = SharedState::default();
        state.set("count", json!(42));
        assert_eq!(state.get_int("count", 0), 42);
        assert_eq!(state.get_int("missing", 10), 10);
    }

    #[test]
    fn test_has() {
        let state = SharedState::default();
        state.set("exists", json!("yes"));
        assert!(state.has("exists"));
        assert!(!state.has("missing"));
    }
}
