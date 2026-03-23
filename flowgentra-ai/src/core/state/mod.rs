//! Typed state system for FlowgentraAI
//!
//! Provides `PlainState` (owned JSON map) and `SharedState` (Arc<RwLock> wrapper).
//! The `State` trait defines the common interface for both.

/// Optional trait for validating state structs
pub trait ValidateState {
    fn validate(&self) -> Result<(), String>;
}

use serde::{Serialize, Deserialize};

/// Trait for state types used throughout the framework.
///
/// Implementors **must** override the methods they support. The defaults
/// exist only so that custom state types can opt out of features they
/// don't need (e.g. `set_evaluation`), but core operations like `get`,
/// `set`, `to_value`, etc. should always be implemented.
pub trait State: Clone + Serialize + for<'de> Deserialize<'de> + Send + Sync + 'static {
    /// Get a value from state by key
    fn get(&self, key: &str) -> Option<Value> {
        let _ = key;
        None
    }

    /// Set a value in state.
    ///
    /// Uses `&self` (not `&mut self`) to support interior-mutability
    /// wrappers like `SharedState`. For `PlainState`, prefer the
    /// inherent `set(&mut self, ...)` method instead.
    fn set(&self, key: impl Into<String>, value: Value) {
        let _ = (key, value);
    }

    /// Check if key exists
    fn contains_key(&self, key: &str) -> bool {
        let _ = key;
        false
    }

    /// Remove a value
    fn remove(&self, key: &str) -> Option<Value> {
        let _ = key;
        None
    }

    /// Get keys iterator
    fn keys(&self) -> Box<dyn Iterator<Item = String> + '_> {
        Box::new(std::iter::empty())
    }

    /// Convert to JSON value
    fn to_value(&self) -> Value {
        Value::Null
    }

    /// Convert from JSON value
    fn from_json(value: Value) -> crate::core::error::Result<Self> {
        let _ = value;
        Err(crate::core::error::FlowgentraError::StateError("from_json not implemented".to_string()))
    }

    /// Merge another state into this one
    fn merge(&self, _other: Self) {
        // Default: do nothing
    }

    /// Create an empty state
    fn empty() -> Self
    where
        Self: Default,
    {
        Self::default()
    }

    /// Get string value (borrowed — only works for non-locking state like PlainState)
    fn get_str(&self, key: &str) -> Option<&str> {
        let _ = key;
        None
    }

    /// Get string value (owned — works for all State implementations including SharedState)
    fn get_string(&self, key: &str) -> Option<String> {
        self.get(key).and_then(|v| match v {
            serde_json::Value::String(s) => Some(s),
            _ => None,
        })
    }

    /// Convert to JSON string
    fn to_json_string(&self) -> crate::core::error::Result<String> {
        Err(crate::core::error::FlowgentraError::StateError("to_json_string not implemented".to_string()))
    }

    /// Get all key-value pairs as a map - for iteration/merging
    fn as_map(&self) -> Vec<(String, Value)> {
        Vec::new()
    }

    /// Store evaluation for a node (default no-op for generic states)
    fn set_evaluation(&self, _node: &str, _eval: crate::core::evaluation::EvaluationResult) {
        // Default: do nothing - specific implementations can override
    }
}

/// Partial state update, generated per user state struct by macro
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct StateUpdate<T: State> {
    #[serde(skip)]
    _phantom: std::marker::PhantomData<T>,
}

impl<T: State> StateUpdate<T> {
    pub fn new() -> Self {
        Self { _phantom: std::marker::PhantomData }
    }
}

use serde_json::Value;
use std::collections::HashMap;

/// Inner state structure holding the actual JSON data.
///
/// Use `SharedState` for normal operations (thread-safe, Arc-wrapped).
/// `PlainState` is exported for advanced users who need direct, owned access.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct PlainState {
    pub(crate) data: serde_json::Map<String, Value>,
}

// =============================================================================
// Constructor Methods
// =============================================================================

impl PlainState {
    /// Create a new, empty state.
    ///
    /// # Example
    /// ```
    /// use flowgentra_ai::core::state::PlainState;
    /// let state = PlainState::new();
    /// assert_eq!(state.keys().count(), 0);
    /// ```
    pub fn new() -> Self {
        PlainState {
            data: serde_json::Map::new(),
        }
    }

    /// Create a state from an existing JSON object.
    ///
    /// # Errors
    /// Returns an error if the provided value is not a JSON object.
    pub fn from_json(value: Value) -> crate::core::error::Result<Self> {
        match value {
            Value::Object(map) => Ok(PlainState { data: map }),
            _ => Err(crate::core::error::FlowgentraError::StateError(
                "State must be a JSON object".to_string(),
            )),
        }
    }
}

// =============================================================================
// Data Access Methods
// =============================================================================

impl PlainState {
    /// Insert or replace a value in the state.
    pub fn set(&mut self, key: impl Into<String>, value: Value) {
        self.data.insert(key.into(), value);
    }

    /// Retrieve an immutable reference to a value.
    pub fn get(&self, key: &str) -> Option<&Value> {
        self.data.get(key)
    }

    /// Retrieve a mutable reference to a value.
    pub fn get_mut(&mut self, key: &str) -> Option<&mut Value> {
        self.data.get_mut(key)
    }

    /// Check if a key exists in the state.
    pub fn contains_key(&self, key: &str) -> bool {
        self.data.contains_key(key)
    }

    /// Remove a value from the state and return it.
    pub fn remove(&mut self, key: &str) -> Option<Value> {
        self.data.remove(key)
    }
}

// =============================================================================
// Inspection Methods
// =============================================================================

impl PlainState {
    /// Get an iterator over all keys in the state.
    pub fn keys(&self) -> impl Iterator<Item = &String> {
        self.data.keys()
    }

    /// Get the underlying JSON map reference.
    pub fn as_map(&self) -> &serde_json::Map<String, Value> {
        &self.data
    }

    /// Get a mutable reference to the underlying JSON map.
    pub fn as_map_mut(&mut self) -> &mut serde_json::Map<String, Value> {
        &mut self.data
    }
}

// =============================================================================
// Type-Safe Access Methods
// =============================================================================

impl PlainState {
    /// Get a typed value from state, automatically deserializing from JSON.
    pub fn get_typed<T: serde::de::DeserializeOwned>(
        &self,
        key: &str,
    ) -> crate::core::error::Result<T> {
        self.get(key)
            .ok_or_else(|| {
                crate::core::error::FlowgentraError::StateError(format!(
                    "Key '{}' not found in state",
                    key
                ))
            })
            .and_then(|v| {
                serde_json::from_value(v.clone())
                    .map_err(crate::core::error::FlowgentraError::from)
            })
    }

    /// Try to get a typed value, returning None if key doesn't exist or deserialization fails.
    pub fn try_get_typed<T: serde::de::DeserializeOwned>(&self, key: &str) -> Option<T> {
        self.get(key)
            .and_then(|v| serde_json::from_value(v.clone()).ok())
    }

    /// Set a typed value, automatically serializing to JSON.
    pub fn set_typed<T: serde::Serialize>(
        &mut self,
        key: impl Into<String>,
        value: &T,
    ) -> crate::core::error::Result<()> {
        let json_value =
            serde_json::to_value(value).map_err(crate::core::error::FlowgentraError::from)?;
        self.set(key, json_value);
        Ok(())
    }

    /// Get a value as a string reference, or `None` if missing/not a string.
    pub fn get_str(&self, key: &str) -> Option<&str> {
        self.get(key).and_then(Value::as_str)
    }

    /// Get a required string or return an error.
    pub fn require_str(&self, key: &str) -> crate::core::error::Result<&str> {
        self.get_str(key).ok_or_else(|| {
            crate::core::error::FlowgentraError::StateError(format!(
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

    /// Get a value as an integer (`i64`).
    pub fn get_int(&self, key: &str) -> Option<i64> {
        self.get(key).and_then(|v| v.as_i64())
    }

    /// Alias for `get_int`.
    pub fn get_i64(&self, key: &str) -> Option<i64> {
        self.get_int(key)
    }

    /// Get a value as a float.
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

impl PlainState {
    /// Convert the state to a JSON value.
    pub fn to_value(&self) -> Value {
        Value::Object(self.data.clone())
    }

    /// Convert the state to a JSON string.
    pub fn to_json_string(&self) -> crate::core::error::Result<String> {
        serde_json::to_string(&self.to_value()).map_err(crate::core::error::FlowgentraError::from)
    }

    /// Merge another state into this one.
    ///
    /// All key-value pairs from the other state are copied into this state.
    /// If keys overlap, the other state's values take precedence.
    pub fn merge(&mut self, other: PlainState) {
        for (k, v) in other.data {
            self.data.insert(k, v);
        }
    }

    /// Get the configured LLM client from state.
    ///
    /// The LLM config is automatically injected into state by the Agent.
    pub fn get_llm_client(
        &self,
    ) -> crate::core::error::Result<std::sync::Arc<dyn crate::core::llm::LLMClient>> {
        let config: crate::core::llm::LLMConfig = self.get_typed("_llm_config").map_err(|_| {
            crate::core::error::FlowgentraError::ConfigError(
                "LLM config not found in state. Make sure LLM is configured in config.yaml"
                    .to_string(),
            )
        })?;
        config.create_client()
    }
}

// =============================================================================
// Trait Implementations
// =============================================================================

impl Default for PlainState {
    fn default() -> Self {
        Self::new()
    }
}

impl From<HashMap<String, Value>> for PlainState {
    fn from(map: HashMap<String, Value>) -> Self {
        let mut data = serde_json::Map::new();
        for (k, v) in map {
            data.insert(k, v);
        }
        PlainState { data }
    }
}

/// Implement the `State` trait for `PlainState`, delegating to inherent methods.
///
/// Note: `PlainState` inherent methods use `&mut self` for mutation, but the
/// `State` trait uses `&self`. The trait `set()` is a no-op for `PlainState`
/// because it cannot mutate through `&self` without interior mutability.
/// Use `SharedState` (which wraps `PlainState` in `Arc<RwLock>`) for the
/// full `State` trait experience, or call `PlainState::set(&mut self, ...)` directly.
impl State for PlainState {
    fn get(&self, key: &str) -> Option<Value> {
        self.data.get(key).cloned()
    }

    fn contains_key(&self, key: &str) -> bool {
        self.data.contains_key(key)
    }

    fn keys(&self) -> Box<dyn Iterator<Item = String> + '_> {
        Box::new(self.data.keys().cloned())
    }

    fn to_value(&self) -> Value {
        Value::Object(self.data.clone())
    }

    fn from_json(value: Value) -> crate::core::error::Result<Self> {
        PlainState::from_json(value)
    }

    fn to_json_string(&self) -> crate::core::error::Result<String> {
        PlainState::to_json_string(self)
    }

    fn as_map(&self) -> Vec<(String, Value)> {
        self.data.iter().map(|(k, v)| (k.clone(), v.clone())).collect()
    }

    fn get_str(&self, key: &str) -> Option<&str> {
        self.data.get(key).and_then(Value::as_str)
    }

    fn get_string(&self, key: &str) -> Option<String> {
        PlainState::get_string(self, key)
    }

    fn empty() -> Self {
        PlainState::new()
    }
}

// Sub-module for state validation

pub mod state_validation;
pub(crate) use state_validation::*;

// Shared state wrapper for zero-copy state management
pub mod shared;
pub use shared::SharedState;

// LangGraph-style memory management
pub mod state_management;
pub use state_management::{CompressionManager, CustomState, MessageHistory};

// New modules and re-exports
pub mod typed;
pub mod state_ext;

// Re-export only necessary types
pub use typed::{StateExt, TypedState};
pub use state_ext::StateExtMethods;

// =============================================================================
// Type Alias: State = SharedState (Default)
// =============================================================================
