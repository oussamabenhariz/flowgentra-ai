# Macros Guide

FlowgentraAI provides three proc macros to reduce boilerplate.

## #[node] -- Generate Node Factory Functions

The `#[node]` attribute macro turns a plain async function into a node factory that works with the StateGraph builder.

### Before (manual)

```rust
use flowgentra_ai::core::state_graph::UpdateNode;

let node = UpdateNode::new("my_processor", |state| {
    Box::pin(async move {
        // processing logic
        Ok(state)
    })
});

builder.add_node("my_processor", node)
```

### After (with macro)

```rust
use flowgentra_ai_macros::node;
use flowgentra_ai::core::state::PlainState;
use flowgentra_ai::core::error::Result;

#[node]
async fn my_processor(mut state: PlainState) -> Result<PlainState> {
    state.set("processed", json!(true));
    Ok(state)
}

// Generates: fn my_processor_node() -> impl Node<PlainState>

let graph = StateGraphBuilder::new()
    .add_node("processor", my_processor_node())
    .set_entry_point("processor")
    .add_edge("processor", "__end__")
    .compile()?;
```

### What It Generates

For a function named `my_processor`, the macro generates `my_processor_node()` which returns an `UpdateNode` wrapping your function. The original function remains available for direct calls and testing.

---

## #[register_handler] -- Auto-Register for Config Discovery

Automatically registers handler functions so the config-driven runtime can find them by name.

### Without the macro

```rust
// Manual registration (tedious)
let mut registry = HandlerRegistry::new();
registry.register("validate", Arc::new(|s| Box::pin(validate(s))));
registry.register("process", Arc::new(|s| Box::pin(process(s))));
registry.register("output", Arc::new(|s| Box::pin(output(s))));
```

### With the macro

```rust
use flowgentra_ai::prelude::*;

#[register_handler]
pub async fn validate(state: State) -> Result<State> {
    Ok(state)
}

#[register_handler]
pub async fn process(mut state: State) -> Result<State> {
    state.set("done", json!(true));
    Ok(state)
}
```

Then reference by function name in `config.yaml`:

```yaml
graph:
  nodes:
    - name: step1
      handler: "validate"
    - name: step2
      handler: "process"
```

Load the agent and all handlers are discovered automatically:

```rust
let agent = from_config_path("config.yaml")?;
```

### Requirements

| Rule | Details |
|------|---------|
| Visibility | Must be `pub` |
| Async | Must be `async fn` |
| Parameter | Exactly one `State` parameter |
| Return | Must return `Result<State>` |

### How It Works

Uses Rust's `inventory` crate to register handlers at compile time:

```
#[register_handler]
pub async fn my_handler(state: State) -> Result<State> { ... }

// Expands to:
pub async fn my_handler(state: State) -> Result<State> { ... }

inventory::submit! {
    HandlerEntry::new("my_handler", Arc::new(|s| Box::pin(my_handler(s))))
}
```

Zero runtime overhead -- registration happens at link time.

---

## #[derive(State)] -- Derive State Trait

Automatically implement the State trait for custom structs:

```rust
use flowgentra_ai_macros::State;

#[derive(State)]
struct MyState {
    input: String,
    score: f64,
    tags: Vec<String>,
}
```

This generates the serialization/deserialization needed to convert between your struct and JSON state.

---

## When to Use Each Macro

| Macro | Use When |
|-------|----------|
| `#[node]` | Building graphs with StateGraphBuilder |
| `#[register_handler]` | Using config-driven (YAML) graph loading |
| `#[derive(State)]` | Defining custom typed state structs |

### You DON'T need macros when...

- Using `add_fn()` on StateGraphBuilder (accepts plain async functions directly)
- Using predefined agents (AgentBuilder handles everything internally)
- Writing one-off scripts or experiments

---

## Organizing Macros Across Modules

All three macros work with Rust's module system. Handlers in separate files are discovered automatically as long as the module is imported:

```
src/
  main.rs          -- mod handlers;
  handlers/
    mod.rs         -- pub mod validation; pub mod processing;
    validation.rs  -- #[register_handler] pub async fn validate(...)
    processing.rs  -- #[register_handler] pub async fn process(...)
```

---

## Troubleshooting

**"Handler not found"**
- Check the function is `pub` and decorated with `#[register_handler]`
- Check the module is imported in `main.rs` (`mod handlers;`)
- Check the name in `config.yaml` matches the function name exactly

**"expected async fn"**
- Add `async` to the function signature

**"cannot find type in this scope"**
- Add `use flowgentra_ai::prelude::*;` at the top of the file

---

See [handlers/README.md](../handlers/README.md) for handler development patterns.
See [graph/README.md](../graph/README.md) for StateGraph builder usage.
