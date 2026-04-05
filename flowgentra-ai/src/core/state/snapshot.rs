//! # StateSnapshot — immutable capture of state at a single step.
//!
//! Snapshots are the foundation for time-travel debugging, checkpointing,
//! and interrupt/resume workflows.  Each snapshot records:
//!
//! - `step_id`    — caller-supplied label (e.g. `"before-node-b"`, `"step-3"`)
//! - `state`      — a flat copy of every field value at the moment of capture
//! - `created_at` — Unix timestamp in seconds (UTC)
//! - `metadata`   — arbitrary JSON key-value pairs for caller annotation
//!
//! # Example
//!
//! ```ignore
//! use flowgentra_ai::core::state::{DynState, StateSnapshot};
//! use serde_json::json;
//!
//! let state = DynState::new();
//! state.set("step", json!(0));
//!
//! let snap = state.snapshot("before-agent");
//! state.set("step", json!(5));
//!
//! state.restore(&snap);
//! assert_eq!(state.get("step"), Some(json!(0)));
//! ```

use serde_json::Value;
use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};

// ── StateSnapshot ─────────────────────────────────────────────────────────────

/// Immutable snapshot of all state fields at a single point in time.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct StateSnapshot {
    /// Caller-supplied step identifier.
    pub step_id: String,

    /// Field values at the moment of capture (a flat key-value map).
    pub state: HashMap<String, Value>,

    /// Unix timestamp (seconds since epoch, UTC) when the snapshot was taken.
    pub created_at: u64,

    /// Optional caller annotations (e.g. `{"node": "agent", "thread": "abc"}`).
    #[serde(default)]
    pub metadata: HashMap<String, Value>,
}

impl StateSnapshot {
    /// Create a new snapshot.
    pub fn new(step_id: impl Into<String>, state: HashMap<String, Value>) -> Self {
        let created_at = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        StateSnapshot {
            step_id: step_id.into(),
            state,
            created_at,
            metadata: HashMap::new(),
        }
    }

    /// Attach a metadata entry (builder pattern).
    pub fn with_metadata(mut self, key: impl Into<String>, value: Value) -> Self {
        self.metadata.insert(key.into(), value);
        self
    }

    /// Number of fields captured in this snapshot.
    pub fn field_count(&self) -> usize {
        self.state.len()
    }

    /// Retrieve a field value from this snapshot.
    pub fn get(&self, key: &str) -> Option<&Value> {
        self.state.get(key)
    }
}

impl std::fmt::Display for StateSnapshot {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "StateSnapshot(step_id={:?}, fields={}, created_at={})",
            self.step_id,
            self.state.len(),
            self.created_at
        )
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn snapshot_stores_state() {
        let mut state = HashMap::new();
        state.insert("x".to_string(), json!(42));

        let snap = StateSnapshot::new("test", state);
        assert_eq!(snap.step_id, "test");
        assert_eq!(snap.get("x"), Some(&json!(42)));
        assert_eq!(snap.field_count(), 1);
        assert!(snap.created_at > 0);
    }

    #[test]
    fn snapshot_with_metadata() {
        let snap = StateSnapshot::new("s1", HashMap::new())
            .with_metadata("node", json!("agent"))
            .with_metadata("thread", json!("abc-123"));
        assert_eq!(snap.metadata.get("node"), Some(&json!("agent")));
        assert_eq!(snap.metadata.get("thread"), Some(&json!("abc-123")));
    }

    #[test]
    fn snapshot_serialize_roundtrip() {
        let mut state = HashMap::new();
        state.insert("name".to_string(), json!("alice"));
        let snap = StateSnapshot::new("step-0", state);
        let json_str = serde_json::to_string(&snap).unwrap();
        let restored: StateSnapshot = serde_json::from_str(&json_str).unwrap();
        assert_eq!(restored.step_id, "step-0");
        assert_eq!(restored.get("name"), Some(&json!("alice")));
    }
}
