//! Scoped/namespaced state access per node.
//!
//! `ScopedState` wraps a `SharedState` and prefixes all key operations
//! with a node namespace, preventing accidental key collisions between nodes.
//!
//! # Example
//! ```ignore
//! let shared = SharedState::new(PlainState::new());
//! let scoped = ScopedState::new(shared.clone(), "summarizer");
//! scoped.set("result", json!("done")); // stored as "summarizer.result"
//! scoped.get("result"); // reads "summarizer.result"
//! ```

use crate::core::state::SharedState;
use serde_json::Value;

/// A namespaced view over a `SharedState`.
///
/// All keys are automatically prefixed with `{namespace}.` so that
/// different nodes can use the same logical key names without collision.
#[derive(Clone, Debug)]
pub struct ScopedState {
    inner: SharedState,
    namespace: String,
}

impl ScopedState {
    /// Create a new scoped view with the given namespace (typically the node name).
    pub fn new(inner: SharedState, namespace: impl Into<String>) -> Self {
        Self {
            inner,
            namespace: namespace.into(),
        }
    }

    /// Get the fully-qualified key.
    fn scoped_key(&self, key: &str) -> String {
        format!("{}.{}", self.namespace, key)
    }

    /// Set a value in the scoped namespace.
    pub fn set(&self, key: impl Into<String>, value: Value) {
        let scoped = self.scoped_key(&key.into());
        self.inner.set(scoped, value);
    }

    /// Get a value from the scoped namespace.
    pub fn get(&self, key: &str) -> Option<Value> {
        self.inner.get(&self.scoped_key(key))
    }

    /// Check if a key exists in the scoped namespace.
    pub fn contains_key(&self, key: &str) -> bool {
        self.inner.contains_key(&self.scoped_key(key))
    }

    /// Remove a value from the scoped namespace.
    pub fn remove(&self, key: &str) -> Option<Value> {
        self.inner.remove(&self.scoped_key(key))
    }

    /// Get all keys in this namespace (without the namespace prefix).
    pub fn keys(&self) -> Vec<String> {
        let prefix = format!("{}.", self.namespace);
        use crate::core::state::State;
        self.inner
            .keys()
            .filter_map(|k| k.strip_prefix(&prefix).map(|s| s.to_string()))
            .collect()
    }

    /// Get the underlying shared state (for cross-node reads).
    pub fn shared(&self) -> &SharedState {
        &self.inner
    }

    /// Get the namespace name.
    pub fn namespace(&self) -> &str {
        &self.namespace
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::state::PlainState;
    use serde_json::json;

    #[test]
    fn test_scoped_state_isolation() {
        let shared = SharedState::new(PlainState::new());
        let scope_a = ScopedState::new(shared.clone(), "node_a");
        let scope_b = ScopedState::new(shared.clone(), "node_b");

        scope_a.set("result", json!("from_a"));
        scope_b.set("result", json!("from_b"));

        assert_eq!(scope_a.get("result"), Some(json!("from_a")));
        assert_eq!(scope_b.get("result"), Some(json!("from_b")));

        // Verify they're stored with namespaced keys
        use crate::core::state::State;
        assert_eq!(shared.get("node_a.result"), Some(json!("from_a")));
        assert_eq!(shared.get("node_b.result"), Some(json!("from_b")));
    }

    #[test]
    fn test_scoped_keys() {
        let shared = SharedState::new(PlainState::new());
        let scoped = ScopedState::new(shared.clone(), "mynode");

        scoped.set("x", json!(1));
        scoped.set("y", json!(2));

        let mut keys = scoped.keys();
        keys.sort();
        assert_eq!(keys, vec!["x".to_string(), "y".to_string()]);
    }
}
