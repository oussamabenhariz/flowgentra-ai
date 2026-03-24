# State Management Guide

Understand how data flows through your workflow and how to control merging behavior.

## State Types

FlowgentraAI provides three state containers for different needs:

### PlainState (Owned, Single-Threaded)

A JSON-backed key-value map. Fast, simple, no locking overhead.

```rust
use flowgentra_ai::core::state::PlainState;

let mut state = PlainState::new();
state.set("user_input", json!("What is Rust?"));
state.set("score", json!(85));

let input = state.get("user_input")
    .and_then(|v| v.as_str())
    .unwrap_or("");
```

Use this in StateGraph nodes where each node gets ownership of the state.

### SharedState (Thread-Safe)

Wraps `PlainState` in `Arc<RwLock<...>>` for concurrent access:

```rust
use flowgentra_ai::core::state::SharedState;

let state = SharedState::new();
state.set("counter", json!(0));

// Safe to clone and share across threads
let state_clone = state.clone();
tokio::spawn(async move {
    state_clone.set("counter", json!(1));
});
```

Use this when multiple tasks or agents need to read/write the same state.

### ScopedState (Namespaced)

Wraps a SharedState with a namespace prefix to prevent key collisions between nodes:

```rust
use flowgentra_ai::core::state::ScopedState;

let shared = SharedState::new();

let node_a = ScopedState::new(shared.clone(), "node_a");
let node_b = ScopedState::new(shared.clone(), "node_b");

node_a.set("result", json!("from A"));
node_b.set("result", json!("from B"));

// Stored as "node_a.result" and "node_b.result" -- no collision
assert!(node_a.get("result") != node_b.get("result"));
```

Methods: `set()`, `get()`, `contains_key()`, `remove()`, `keys()` -- all namespace-aware.

---

## JsonReducer and ReducerConfig

When multiple nodes update the same state fields, reducers control how values merge.

### Available Reducers

| Reducer | Behavior | Example |
|---------|----------|---------|
| `Overwrite` | Replace the old value (default) | `5` + `3` = `3` |
| `Append` | Append to array | `[1,2]` + `[3]` = `[1,2,3]` |
| `Sum` | Add numbers | `5` + `3` = `8` |
| `DeepMerge` | Recursively merge objects | `{a:1}` + `{b:2}` = `{a:1,b:2}` |
| `Max` | Keep the larger value | `5` + `3` = `5` |
| `Min` | Keep the smaller value | `5` + `3` = `3` |
| `AppendUnique` | Append only if not already present | `[1,2]` + `[2,3]` = `[1,2,3]` |

### Setting Up Reducers

```rust
use flowgentra_ai::core::reducer::{ReducerConfig, JsonReducer};

let config = ReducerConfig::default()
    .field("messages", JsonReducer::Append)        // Chat history grows
    .field("total_cost", JsonReducer::Sum)          // Costs accumulate
    .field("config", JsonReducer::DeepMerge)        // Configs merge
    .field("best_score", JsonReducer::Max)           // Track best
    .field("tags", JsonReducer::AppendUnique);       // No duplicate tags
```

### Using Reducers with State Merging

```rust
use flowgentra_ai::core::runtime::merge_state;

let merged = merge_state(&current_state, &update, &reducer_config)?;
```

Fields not listed in the config use `Overwrite` by default.

### MergeStrategy

Control the overall merge behavior:

| Strategy | Behavior |
|----------|----------|
| `Default` | Use per-field reducers from ReducerConfig |
| `Replace` | Completely replace the old state |
| `Merge` | Deep merge all fields |

---

## Reading and Writing State

### Basic Operations

```rust
async fn my_handler(mut state: PlainState) -> Result<PlainState> {
    // Read
    let input = state.get("user_input")
        .and_then(|v| v.as_str())
        .unwrap_or("");

    let count = state.get("count")
        .and_then(|v| v.as_i64())
        .unwrap_or(0);

    // Write
    state.set("processed", json!(true));
    state.set("result", json!({
        "answer": "42",
        "confidence": 0.95
    }));

    // Remove
    state.remove("temporary_field");

    Ok(state)
}
```

### Common Patterns

**Accumulation:**
```rust
let mut items = state.get("results")
    .and_then(|v| v.as_array().cloned())
    .unwrap_or_default();
items.push(json!("new_item"));
state.set("results", json!(items));
```

**Conditional processing:**
```rust
let is_complex = state.get("complexity")
    .and_then(|v| v.as_i64())
    .map(|s| s > 70)
    .unwrap_or(false);

if is_complex {
    // heavy processing
}
```

**Chaining handler outputs:**
```rust
// Handler 1 writes
state.set("step1_output", json!(result));

// Handler 2 reads
let input = state.get("step1_output").unwrap();
```

---

## State in StateGraph

In a StateGraph, each node receives the full state, transforms it, and returns the updated state:

```
START --> greet --> format --> END
         state0   state1    state2
```

With reducers, parallel branches can independently update the same fields and have them merged correctly when branches rejoin.

---

## Config-Driven State Schema

Document your state structure in YAML:

```yaml
state_schema:
  user_input:
    type: string
    description: "The user's question"
  analysis:
    type: object
    description: "Analysis results from the LLM"
  response:
    type: string
    description: "Final response to return"
```

This enables runtime validation:

```rust
state.validate()?;  // Checks against the schema
```

---

## Best Practices

1. **Use PlainState for graph nodes** -- simpler, no locking overhead
2. **Use SharedState when concurrent access is needed** -- agents, parallel branches
3. **Use ScopedState to avoid collisions** -- especially in multi-agent setups
4. **Set up reducers early** -- define ReducerConfig before running the graph
5. **Keep state lean** -- remove temporary fields after use
6. **Use consistent naming** -- `snake_case` for all keys

---

See [graph/README.md](../graph/README.md) for graph building.
See [FEATURES.md](../FEATURES.md) for the complete feature list.
