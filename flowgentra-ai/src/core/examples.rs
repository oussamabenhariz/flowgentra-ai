//! Example usage of the reducer-based state merge system

use crate::core::reducer::{JsonReducer, ReducerConfig};
use crate::core::runtime::merge_state;
use crate::core::state::PlainState;
use serde_json::json;

/// Example: merging partial updates from multiple nodes with per-field reducers
pub fn merge_example() {
    let mut state = PlainState::new();
    state.set("input", json!("hello"));
    state.set("messages", json!([]));
    state.set("count", json!(0));

    let mut update1 = PlainState::new();
    update1.set("messages", json!(["msg1"]));
    update1.set("count", json!(1));

    let mut update2 = PlainState::new();
    update2.set("messages", json!(["msg2"]));
    update2.set("count", json!(2));

    let reducers = ReducerConfig::new()
        .field("messages", JsonReducer::Append)
        .field("count", JsonReducer::Sum);

    let after_first = merge_state(&state, &update1, &reducers).unwrap();
    let after_both = merge_state(&after_first, &update2, &reducers).unwrap();

    // messages: ["msg1", "msg2"], count: 3, input: "hello"
    println!("Merged state: {:?}", after_both.to_value());
}
