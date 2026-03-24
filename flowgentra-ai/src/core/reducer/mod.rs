//! ## Extensibility
//!
//! - Implement `Reducer<T>` for your own types for custom merge logic.
//! - Register custom reducers with the macro using #[state(reducer = "...")].
//! - See example in docs and tests.
//! # Reducer system for flowgentra
//!
//! ## Error Handling
//!
//! - Reducers should never panic; always handle errors gracefully.
//! - If a reducer cannot merge, return a Result or log an error (future extension).
//! - Use clear error messages for debugging.
//! Reducer system for flowgentra

/// Trait for merging state field updates
pub trait Reducer<T> {
    fn merge(current: &mut T, update: T);
}

/// Overwrite reducer: replaces the field value
pub struct Overwrite;
impl<T> Reducer<T> for Overwrite {
    fn merge(current: &mut T, update: T) {
        *current = update;
    }
}

/// Append reducer: appends to a Vec
pub struct Append;
impl<T> Reducer<Vec<T>> for Append {
    fn merge(current: &mut Vec<T>, update: Vec<T>) {
        current.extend(update);
    }
}

/// Sum reducer: sums numeric types
pub struct Sum;
impl Reducer<i32> for Sum {
    fn merge(current: &mut i32, update: i32) {
        *current += update;
    }
}
impl Reducer<f64> for Sum {
    fn merge(current: &mut f64, update: f64) {
        *current += update;
    }
}

/// MergeMap reducer: merges HashMaps
use std::collections::HashMap;
pub struct MergeMap;
impl<K, V> Reducer<HashMap<K, V>> for MergeMap
where
    K: std::cmp::Eq + std::hash::Hash,
    V: Clone,
{
    fn merge(current: &mut HashMap<K, V>, update: HashMap<K, V>) {
        current.extend(update);
    }
}

// Users can implement Reducer<T> for custom strategies

// =============================================================================
// JSON-level reducers for dynamic state
// =============================================================================

use serde_json::Value;

/// JSON-level reducer that operates on `serde_json::Value` fields.
///
/// These work with the dynamic JSON state (`PlainState` / `SharedState`)
/// and are applied per-field during `UpdateNode` execution.
#[derive(Debug, Clone)]
pub enum JsonReducer {
    /// Replace the field entirely (default behavior)
    Overwrite,
    /// Append array items (both current and update must be arrays)
    Append,
    /// Sum numeric values
    Sum,
    /// Deep-merge objects (update keys overwrite current keys)
    DeepMerge,
    /// Keep the maximum numeric value
    Max,
    /// Keep the minimum numeric value
    Min,
    /// Deduplicated append (only add items not already present)
    AppendUnique,
}

impl JsonReducer {
    /// Apply this reducer to merge `update` into `current`, returning the merged value.
    pub fn apply(&self, current: &Value, update: &Value) -> Value {
        match self {
            JsonReducer::Overwrite => update.clone(),
            JsonReducer::Append => {
                let mut arr = match current {
                    Value::Array(a) => a.clone(),
                    _ => vec![current.clone()],
                };
                match update {
                    Value::Array(u) => arr.extend(u.iter().cloned()),
                    other => arr.push(other.clone()),
                }
                Value::Array(arr)
            }
            JsonReducer::Sum => {
                let c = current.as_f64().unwrap_or(0.0);
                let u = update.as_f64().unwrap_or(0.0);
                serde_json::json!(c + u)
            }
            JsonReducer::DeepMerge => deep_merge_values(current, update),
            JsonReducer::Max => {
                let c = current.as_f64().unwrap_or(f64::NEG_INFINITY);
                let u = update.as_f64().unwrap_or(f64::NEG_INFINITY);
                if u >= c { update.clone() } else { current.clone() }
            }
            JsonReducer::Min => {
                let c = current.as_f64().unwrap_or(f64::INFINITY);
                let u = update.as_f64().unwrap_or(f64::INFINITY);
                if u <= c { update.clone() } else { current.clone() }
            }
            JsonReducer::AppendUnique => {
                let mut arr = match current {
                    Value::Array(a) => a.clone(),
                    _ => vec![current.clone()],
                };
                let items = match update {
                    Value::Array(u) => u.clone(),
                    other => vec![other.clone()],
                };
                for item in items {
                    if !arr.contains(&item) {
                        arr.push(item);
                    }
                }
                Value::Array(arr)
            }
        }
    }
}

/// Per-field reducer configuration for a state graph.
///
/// Maps field names to their `JsonReducer`. Fields not in the map
/// use `JsonReducer::Overwrite` by default.
#[derive(Debug, Clone, Default)]
pub struct ReducerConfig {
    pub reducers: HashMap<String, JsonReducer>,
}

impl ReducerConfig {
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the reducer for a specific field.
    pub fn field(mut self, name: impl Into<String>, reducer: JsonReducer) -> Self {
        self.reducers.insert(name.into(), reducer);
        self
    }

    /// Get the reducer for a field, defaulting to Overwrite.
    pub fn get(&self, field: &str) -> &JsonReducer {
        self.reducers.get(field).unwrap_or(&JsonReducer::Overwrite)
    }

    /// Merge a partial state update into the current state using configured reducers.
    ///
    /// Returns a new `Value::Object` with merged fields.
    pub fn merge_values(&self, current: &Value, update: &Value) -> Value {
        let mut result = match current {
            Value::Object(m) => m.clone(),
            _ => serde_json::Map::new(),
        };

        if let Value::Object(update_map) = update {
            for (key, update_val) in update_map {
                let reducer = self.get(key);
                let merged = if let Some(current_val) = result.get(key) {
                    reducer.apply(current_val, update_val)
                } else {
                    update_val.clone()
                };
                result.insert(key.clone(), merged);
            }
        }

        Value::Object(result)
    }
}

/// Deep merge two JSON values. Objects are merged recursively;
/// all other types are overwritten by `update`.
pub fn deep_merge_values(current: &Value, update: &Value) -> Value {
    match (current, update) {
        (Value::Object(c), Value::Object(u)) => {
            let mut merged = c.clone();
            for (key, u_val) in u {
                let merged_val = if let Some(c_val) = c.get(key) {
                    deep_merge_values(c_val, u_val)
                } else {
                    u_val.clone()
                };
                merged.insert(key.clone(), merged_val);
            }
            Value::Object(merged)
        }
        _ => update.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_json_overwrite() {
        let r = JsonReducer::Overwrite;
        assert_eq!(r.apply(&json!(1), &json!(2)), json!(2));
    }

    #[test]
    fn test_json_append() {
        let r = JsonReducer::Append;
        assert_eq!(
            r.apply(&json!([1, 2]), &json!([3, 4])),
            json!([1, 2, 3, 4])
        );
    }

    #[test]
    fn test_json_sum() {
        let r = JsonReducer::Sum;
        assert_eq!(r.apply(&json!(10), &json!(5)), json!(15.0));
    }

    #[test]
    fn test_json_deep_merge() {
        let r = JsonReducer::DeepMerge;
        let current = json!({"a": 1, "b": {"x": 10}});
        let update = json!({"b": {"y": 20}, "c": 3});
        let result = r.apply(&current, &update);
        assert_eq!(result, json!({"a": 1, "b": {"x": 10, "y": 20}, "c": 3}));
    }

    #[test]
    fn test_json_append_unique() {
        let r = JsonReducer::AppendUnique;
        assert_eq!(
            r.apply(&json!([1, 2, 3]), &json!([2, 3, 4])),
            json!([1, 2, 3, 4])
        );
    }

    #[test]
    fn test_json_max() {
        let r = JsonReducer::Max;
        assert_eq!(r.apply(&json!(10), &json!(5)), json!(10));
        assert_eq!(r.apply(&json!(5), &json!(10)), json!(10));
    }

    #[test]
    fn test_reducer_config_merge() {
        let config = ReducerConfig::new()
            .field("messages", JsonReducer::Append)
            .field("count", JsonReducer::Sum);

        let current = json!({"messages": ["hello"], "count": 5, "name": "old"});
        let update = json!({"messages": ["world"], "count": 3, "name": "new"});
        let result = config.merge_values(&current, &update);

        assert_eq!(result["messages"], json!(["hello", "world"]));
        assert_eq!(result["count"], json!(8.0));
        assert_eq!(result["name"], json!("new")); // default Overwrite
    }
}
