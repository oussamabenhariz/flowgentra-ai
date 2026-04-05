//! Channel types — the backbone of LangGraph-style state management.
//!
//! Each state field maps to a Channel.  The channel stores the current value
//! and knows how to merge incoming updates via its `ChannelType` (reducer strategy).
//!
//! # Overview
//!
//! | ChannelType        | Semantics                                            |
//! |--------------------|------------------------------------------------------|
//! | `LastValue`        | Overwrite — latest value wins (default)              |
//! | `Topic`            | Accumulate — new items are appended to a list        |
//! | `BinaryOperator`   | Custom — merge via `f(current, new) -> merged`       |
//!
//! # Usage (Rust)
//!
//! ```ignore
//! use flowgentra_ai::core::state::{Channel, ChannelType, FieldSchema};
//! use serde_json::json;
//!
//! // Build a schema with one append field and one replace field
//! let msgs_schema = FieldSchema::topic("messages");
//! let step_schema = FieldSchema::last_value("steps");
//!
//! // Runtime channels
//! let mut msgs_ch  = Channel::from_schema(&msgs_schema);
//! let mut steps_ch = Channel::from_schema(&step_schema);
//!
//! msgs_ch.apply(json!(["hello"]));
//! msgs_ch.apply(json!(["world"]));
//! assert_eq!(msgs_ch.value, json!(["hello", "world"]));
//!
//! steps_ch.apply(json!(0));
//! steps_ch.apply(json!(1));
//! assert_eq!(steps_ch.value, json!(1));
//! ```

use serde_json::Value;
use std::sync::Arc;

// ── ChannelType ───────────────────────────────────────────────────────────────

/// Reducer strategy for a state field channel.
#[derive(Clone)]
pub enum ChannelType {
    /// **LastValue** — overwrite the current value with the new one (default).
    ///
    /// Equivalent to a plain variable assignment.  This is the default channel
    /// type when no reducer annotation is specified.
    LastValue,

    /// **Topic** — accumulate new items into an array.
    ///
    /// Both the existing value and the incoming value are coerced to arrays
    /// before being concatenated.  Mirrors `Annotated[list, operator.add]` in
    /// LangGraph.
    Topic,

    /// **BinaryOperator** — merge via a custom function `f(current, new) -> merged`.
    ///
    /// The function receives both values as `serde_json::Value` and must return
    /// the merged result.  It is stored as an `Arc<dyn Fn>` so it can be cloned
    /// cheaply across nodes.
    BinaryOperator(Arc<dyn Fn(Value, Value) -> Value + Send + Sync>),
}

impl std::fmt::Debug for ChannelType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ChannelType::LastValue => write!(f, "LastValue"),
            ChannelType::Topic => write!(f, "Topic"),
            ChannelType::BinaryOperator(_) => write!(f, "BinaryOperator(<fn>)"),
        }
    }
}

// ── FieldSchema ───────────────────────────────────────────────────────────────

/// Descriptor for a single field in a state schema.
///
/// Used to construct `DynState` with pre-configured channels.
#[derive(Clone, Debug)]
pub struct FieldSchema {
    /// Field name (must match the key used in state maps).
    pub name: String,
    /// Reducer strategy.
    pub channel_type: ChannelType,
    /// Value to use when the field has not been set yet.
    pub default: Value,
}

impl FieldSchema {
    /// Construct a `LastValue` (replace) field.
    pub fn last_value(name: impl Into<String>) -> Self {
        FieldSchema {
            name: name.into(),
            channel_type: ChannelType::LastValue,
            default: Value::Null,
        }
    }

    /// Construct a `Topic` (append) field whose default is an empty array.
    pub fn topic(name: impl Into<String>) -> Self {
        FieldSchema {
            name: name.into(),
            channel_type: ChannelType::Topic,
            default: Value::Array(vec![]),
        }
    }

    /// Construct a field merged by a custom binary function.
    pub fn binary_operator(
        name: impl Into<String>,
        f: impl Fn(Value, Value) -> Value + Send + Sync + 'static,
    ) -> Self {
        FieldSchema {
            name: name.into(),
            channel_type: ChannelType::BinaryOperator(Arc::new(f)),
            default: Value::Null,
        }
    }

    /// Override the default value.
    pub fn with_default(mut self, default: Value) -> Self {
        self.default = default;
        self
    }
}

// ── Channel ───────────────────────────────────────────────────────────────────

/// A single runtime channel — holds one field's current value and its reducer.
#[derive(Clone)]
pub struct Channel {
    /// The current stored value.
    pub value: Value,
    /// Reducer strategy.
    pub channel_type: ChannelType,
}

impl std::fmt::Debug for Channel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Channel")
            .field("value", &self.value)
            .field("channel_type", &self.channel_type)
            .finish()
    }
}

impl Channel {
    /// Construct a channel from its schema descriptor.
    pub fn from_schema(schema: &FieldSchema) -> Self {
        Channel {
            value: schema.default.clone(),
            channel_type: schema.channel_type.clone(),
        }
    }

    /// Construct a plain `LastValue` channel with a given initial value.
    pub fn last_value(value: Value) -> Self {
        Channel {
            value,
            channel_type: ChannelType::LastValue,
        }
    }

    /// Apply an incoming update using this channel's reducer. Mutates `self.value`.
    pub fn apply(&mut self, new_val: Value) {
        self.value = apply_channel_reducer(
            std::mem::replace(&mut self.value, Value::Null),
            new_val,
            &self.channel_type,
        );
    }
}

// ── Standalone reducer helper ─────────────────────────────────────────────────

/// Apply a reducer to merge `current` with `new_val` and return the result.
///
/// This is a pure function; it does not mutate any state.  It is the core
/// merge primitive used by both `Channel::apply` and the graph executor's
/// partial-update merge loop.
pub fn apply_channel_reducer(current: Value, new_val: Value, channel_type: &ChannelType) -> Value {
    match channel_type {
        ChannelType::LastValue => new_val,

        ChannelType::Topic => {
            // Both sides are coerced to arrays.
            let mut base: Vec<Value> = match current {
                Value::Array(arr) => arr,
                Value::Null => vec![],
                other => vec![other],
            };
            match new_val {
                Value::Array(inc) => base.extend(inc),
                Value::Null => {}
                other => base.push(other),
            }
            Value::Array(base)
        }

        ChannelType::BinaryOperator(f) => f(current, new_val),
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn last_value_replaces() {
        let mut ch = Channel::last_value(json!(1));
        ch.apply(json!(42));
        assert_eq!(ch.value, json!(42));
    }

    #[test]
    fn topic_appends_arrays() {
        let mut ch = Channel::from_schema(&FieldSchema::topic("msgs"));
        ch.apply(json!(["a", "b"]));
        ch.apply(json!(["c"]));
        assert_eq!(ch.value, json!(["a", "b", "c"]));
    }

    #[test]
    fn topic_coerces_scalars() {
        let mut ch = Channel::from_schema(&FieldSchema::topic("items"));
        ch.apply(json!("hello"));
        ch.apply(json!("world"));
        assert_eq!(ch.value, json!(["hello", "world"]));
    }

    #[test]
    fn binary_operator_sums() {
        let schema = FieldSchema::binary_operator("count", |a, b| {
            json!(a.as_i64().unwrap_or(0) + b.as_i64().unwrap_or(0))
        });
        let mut ch = Channel::from_schema(&schema);
        ch.value = json!(10);
        ch.apply(json!(5));
        assert_eq!(ch.value, json!(15));
    }

    #[test]
    fn topic_null_noop() {
        let mut ch = Channel::from_schema(&FieldSchema::topic("items"));
        ch.value = json!(["x"]);
        ch.apply(Value::Null);
        assert_eq!(ch.value, json!(["x"]));
    }
}
