//! # State Management
//!
//! The `State` is the central data structure that flows through your agent's execution.
//! It's passed between nodes in the graph, allowing data to accumulate and be transformed
//! as it moves through your workflow.
//!
//! ## Overview
//!
//! - Represents mutable data shared across all nodes
//! - Built on JSON values for maximum flexibility
//! - Supports typed access patterns
//! - Thread-safe via cloning
//!
//! ## Example
//!
//! ```no_run
//! use erenflow_ai::core::state::State;
//! use serde_json::json;
//!
//! // Create new state
//! let mut state = State::new();
//!
//! // Add data
//! state.set("user_input", json!("Hello, world!"));
//! state.set("count", json!(42));
//!
//! // Retrieve data
//! if let Some(input) = state.get("user_input") {
//!     println!("Input: {}", input);
//! }
//!
//! // Convert to JSON string
//! let json_string = state.to_json_string()?;
//! println!("Full state: {}", json_string);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

use serde_json::Value;
use std::collections::HashMap;

/// Represents the mutable state shared across all nodes in graph execution.
///
/// The State acts as a communication channel between nodes. Each node receives
/// the current state, performs operations, and returns an updated state that
/// flows to the next node. Think of it as a pipeline where data is gradually
/// enriched as it flows through your agent.
///
/// Internally, it's a JSON object that provides flexible, untyped storage.
#[derive(Clone, Debug)]
pub struct State {
    data: serde_json::Map<String, Value>,
}

// =============================================================================
// Constructor Methods
// =============================================================================

impl State {
    /// Create a new, empty state.
    ///
    /// # Example
    /// ```
    /// use erenflow_ai::core::state::State;
    /// let state = State::new();
    /// assert_eq!(state.keys().count(), 0);
    /// ```
    pub fn new() -> Self {
        State {
            data: serde_json::Map::new(),
        }
    }

    /// Create a state from an existing JSON object.
    ///
    /// # Errors
    /// Returns an error if the provided value is not a JSON object.
    ///
    /// # Example
    /// ```
    /// use erenflow_ai::core::state::State;
    /// use serde_json::json;
    ///
    /// let json = json!({"key": "value"});
    /// let state = State::from_json(json)?;
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn from_json(value: Value) -> crate::core::error::Result<Self> {
        match value {
            Value::Object(map) => Ok(State { data: map }),
            _ => Err(crate::core::error::ErenFlowError::StateError(
                "State must be a JSON object".to_string(),
            )),
        }
    }
}

// =============================================================================
// Data Access Methods
// =============================================================================

impl State {
    /// Insert or replace a value in the state.
    ///
    /// # Example
    /// ```
    /// use erenflow_ai::core::state::State;
    /// use serde_json::json;
    ///
    /// let mut state = State::new();
    /// state.set("name", json!("Alice"));
    /// ```
    pub fn set(&mut self, key: impl Into<String>, value: Value) {
        self.data.insert(key.into(), value);
    }

    /// Retrieve an immutable reference to a value.
    ///
    /// # Example
    /// ```
    /// use erenflow_ai::core::state::State;
    /// use serde_json::json;
    ///
    /// let mut state = State::new();
    /// state.set("name", json!("Alice"));
    /// assert_eq!(state.get("name").unwrap().as_str(), Some("Alice"));
    /// ```
    pub fn get(&self, key: &str) -> Option<&Value> {
        self.data.get(key)
    }

    /// Retrieve a mutable reference to a value.
    ///
    /// Useful when you want to modify a value in place.
    pub fn get_mut(&mut self, key: &str) -> Option<&mut Value> {
        self.data.get_mut(key)
    }

    /// Check if a key exists in the state.
    pub fn contains_key(&self, key: &str) -> bool {
        self.data.contains_key(key)
    }

    /// Remove a value from the state and return it.
    ///
    /// # Example
    /// ```
    /// use erenflow_ai::core::state::State;
    /// use serde_json::json;
    ///
    /// let mut state = State::new();
    /// state.set("temporary", json!(42));
    /// let removed = state.remove("temporary");
    /// assert!(removed.is_some());
    /// ```
    pub fn remove(&mut self, key: &str) -> Option<Value> {
        self.data.remove(key)
    }
}

// =============================================================================
// Inspection Methods
// =============================================================================

impl State {
    /// Get an iterator over all keys in the state.
    ///
    /// # Example
    /// ```
    /// use erenflow_ai::core::state::State;
    /// use serde_json::json;
    ///
    /// let mut state = State::new();
    /// state.set("a", json!(1));
    /// state.set("b", json!(2));
    /// assert_eq!(state.keys().count(), 2);
    /// ```
    pub fn keys(&self) -> impl Iterator<Item = &String> {
        self.data.keys()
    }

    /// Get the underlying JSON map reference.
    pub fn as_map(&self) -> &serde_json::Map<String, Value> {
        &self.data
    }

    /// Get a mutable reference to the underlying JSON map.
    ///
    /// For advanced usage when you need direct map manipulation.
    pub fn as_map_mut(&mut self) -> &mut serde_json::Map<String, Value> {
        &mut self.data
    }
}

// =============================================================================
// Type-Safe Access Methods
// =============================================================================

impl State {
    /// Get a typed value from state, automatically deserializing from JSON.
    ///
    /// # Example
    /// ```
    /// use erenflow_ai::core::state::State;
    /// use serde_json::json;
    ///
    /// let mut state = State::new();
    /// state.set("count", json!(42));
    /// let count: i64 = state.get_typed("count")?;
    /// assert_eq!(count, 42);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn get_typed<T: serde::de::DeserializeOwned>(
        &self,
        key: &str,
    ) -> crate::core::error::Result<T> {
        self.get(key)
            .ok_or_else(|| {
                crate::core::error::ErenFlowError::StateError(format!(
                    "Key '{}' not found in state",
                    key
                ))
            })
            .and_then(|v| {
                serde_json::from_value(v.clone()).map_err(|e| {
                    crate::core::error::ErenFlowError::SerializationError(e.to_string())
                })
            })
    }

    /// Try to get a typed value, returning None if key doesn't exist or deserialization fails.
    pub fn try_get_typed<T: serde::de::DeserializeOwned>(&self, key: &str) -> Option<T> {
        self.get(key)
            .and_then(|v| serde_json::from_value(v.clone()).ok())
    }

    /// Set a typed value, automatically serializing to JSON.
    ///
    /// # Example
    /// ```
    /// use erenflow_ai::core::state::State;
    ///
    /// #[derive(serde::Serialize)]
    /// struct User {
    ///     name: String,
    ///     age: u32,
    /// }
    ///
    /// let mut state = State::new();
    /// state.set_typed("user", &User {
    ///     name: "Alice".to_string(),
    ///     age: 30,
    /// })?;
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn set_typed<T: serde::Serialize>(
        &mut self,
        key: impl Into<String>,
        value: &T,
    ) -> crate::core::error::Result<()> {
        let json_value = serde_json::to_value(value)
            .map_err(|e| crate::core::error::ErenFlowError::SerializationError(e.to_string()))?;
        self.set(key, json_value);
        Ok(())
    }

    /// Get a value as a string reference, or `None` if missing/not a string.
    /// Prefer this over `get_string` when you only need to read and don't need ownership.
    ///
    /// # Example
    /// ```
    /// use erenflow_ai::core::state::State;
    /// use serde_json::json;
    /// let mut state = State::new();
    /// state.set("input", json!("hello"));
    /// assert_eq!(state.get_str("input"), Some("hello"));
    /// ```
    pub fn get_str(&self, key: &str) -> Option<&str> {
        self.get(key).and_then(Value::as_str)
    }

    /// Get a required string or return an error. Use when the key must be present and be a string.
    ///
    /// # Example
    /// ```
    /// use erenflow_ai::core::state::State;
    /// use serde_json::json;
    /// let mut state = State::new();
    /// state.set("input", json!("hello"));
    /// assert_eq!(state.require_str("input").unwrap(), "hello");
    /// ```
    pub fn require_str(&self, key: &str) -> crate::core::error::Result<&str> {
        self.get_str(key).ok_or_else(|| {
            crate::core::error::ErenFlowError::StateError(format!(
                "Missing or non-string state key: '{}'",
                key
            ))
        })
    }

    /// Get a value as a string, converting if necessary.
    pub fn get_string(&self, key: &str) -> Option<String> {
        self.get(key).and_then(|v| match v {
            Value::String(s) => Some(s.clone()),
            _ => v.as_str().map(|s| s.to_string()),
        })
    }

    /// Get a value as an integer (`i64`), or `None` if missing/not a number.
    pub fn get_int(&self, key: &str) -> Option<i64> {
        self.get(key).and_then(|v| v.as_i64())
    }

    /// Alias for `get_int` for consistency with `get_str` / `get_bool` naming.
    pub fn get_i64(&self, key: &str) -> Option<i64> {
        self.get_int(key)
    }

    /// Get a value as a float, converting if necessary.
    pub fn get_float(&self, key: &str) -> Option<f64> {
        self.get(key).and_then(|v| v.as_f64())
    }

    /// Get a value as a boolean.
    pub fn get_bool(&self, key: &str) -> Option<bool> {
        self.get(key).and_then(|v| v.as_bool())
    }
}

// =============================================================================
// Conversion Methods
// =============================================================================

impl State {
    /// Convert the state to a JSON value.
    pub fn to_value(&self) -> Value {
        Value::Object(self.data.clone())
    }

    /// Convert the state to a JSON string.
    ///
    /// # Example
    /// ```
    /// use erenflow_ai::core::state::State;
    /// use serde_json::json;
    ///
    /// let mut state = State::new();
    /// state.set("key", json!("value"));
    /// let json_str = state.to_json_string()?;
    /// println!("{}", json_str);  // {"key":"value"}
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn to_json_string(&self) -> crate::core::error::Result<String> {
        serde_json::to_string(&self.to_value())
            .map_err(|e| crate::core::error::ErenFlowError::SerializationError(e.to_string()))
    }

    /// Merge another state into this one.
    ///
    /// All key-value pairs from the other state are copied into this state.
    /// If keys overlap, the other state's values take precedence.
    ///
    /// # Example
    /// ```
    /// use erenflow_ai::core::state::State;
    /// use serde_json::json;
    ///
    /// let mut state1 = State::new();
    /// state1.set("a", json!(1));
    ///
    /// let mut state2 = State::new();
    /// state2.set("b", json!(2));
    ///
    /// state1.merge(state2);
    /// assert!(state1.contains_key("a"));
    /// assert!(state1.contains_key("b"));
    /// ```
    pub fn merge(&mut self, other: State) {
        for (k, v) in other.data {
            self.data.insert(k, v);
        }
    }
}

// =============================================================================
// Trait Implementations
// =============================================================================

impl Default for State {
    fn default() -> Self {
        Self::new()
    }
}

impl From<HashMap<String, Value>> for State {
    fn from(map: HashMap<String, Value>) -> Self {
        let mut data = serde_json::Map::new();
        for (k, v) in map {
            data.insert(k, v);
        }
        State { data }
    }
}

// Sub-module for state validation
pub mod state_validation;
pub use state_validation::*;
