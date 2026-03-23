# StateGraph Quick Reference

## Basic Usage

### 1. Create a Node

```rust
use std::sync::Arc;
use flowgentra_ai::core::{FunctionNode, state::PlainState};
use serde_json::json;

// Simple function node
let node = Arc::new(FunctionNode::new("my_node", |state: &PlainState| {
    Box::pin(async {
        let mut new_state = state.clone();
        new_state.set("result", json!("success"));
        Ok(new_state)
    })
}));
```

### 2. Build a Graph

```rust
use flowgentra_ai::core::{StateGraph, END};

let graph = StateGraph::builder()
    .add_node("step1", node1)
    .add_node("step2", node2)
    .set_entry_point("step1")
    .add_edge("step1", "step2")
    .add_edge("step2", END)
    .compile()?;
```

### 3. Execute

```rust
let initial_state = PlainState::new();
let final_state = graph.invoke(initial_state).await?;
```

### 4. Conditional Routing

```rust
graph
    .add_conditional_edge("step1", Box::new(|state| {
        if state.get("done").as_bool().unwrap_or(false) {
            Ok(END.to_string())
        } else {
            Ok("step2".to_string())
        }
    }))
```

## Common Patterns

### Pattern: Simple Pipeline

```
node1 → node2 → node3 → END
```

```rust
graph
    .add_edge("node1", "node2")
    .add_edge("node2", "node3")
    .add_edge("node3", END)
```

### Pattern: ReAct Agent Loop

```
    ┌─→ tool_executor ─┐
    │                   │
  agent ──router────────┘
    │
   END
```

```rust
graph
    .add_node("agent", agent_node)
    .add_node("tool_executor", tool_node)
    .set_entry_point("agent")
    .add_conditional_edge("agent", Box::new(|state| {
        if needs_tool(state) { Ok("tool_executor".into()) }
        else { Ok(END.into()) }
    }))
    .add_edge("tool_executor", "agent")
```

### Pattern: Parallel Branches (Future)

```
        ┌─→ branch1 ─┐
start ──┤             ├─→ merge → END
        └─→ branch2 ─┘
```

Currently chains sequentially. Parallel execution planned.

### Pattern: Error Handling (Future)

```
node1 ──(success)──→ node2 → END
  │
  └──(error)──→ error_handler → END
```

Currently not supported. Error recovery via checkpoint resumption.

## Checkpointing

### Save and Resume

```rust
// Invoke automatically saves checkpoints
let result = graph.invoke(state).await?;

// Later: resume from latest checkpoint
let recovered = graph.resume(&thread_id).await?;

// List all checkpoints for a thread
let checkpoints = graph.history(&thread_id).await?;

// Clear history
graph.clear_history(&thread_id).await?;
```

### Custom Checkpointer

```rust
use flowgentra_ai::core::{Checkpointer, Checkpoint};
use async_trait::async_trait;

struct MyCheckpointer;

#[async_trait]
impl<S: State> Checkpointer<S> for MyCheckpointer {
    async fn save(&self, checkpoint: &Checkpoint<S>) -> Result<()> {
        // Save to database, Redis, etc.
        Ok(())
    }
    
    async fn load(&self, thread_id: &str, step: usize) -> Result<Option<Checkpoint<S>>> {
        // Load from persistent store
        Ok(None)
    }
    
    // ... implement other methods
}

graph
    .set_checkpointer(Arc::new(MyCheckpointer))
```

## State Management

### Access State in Node

```rust
let node = FunctionNode::new("read_state", |state| {
    Box::pin(async move {
        // Read
        let value: Option<json::Value> = state.get("key");
        
        // Write
        let mut new_state = state.clone();
        new_state.set("new_key", json!("new_value"));
        
        Ok(new_state)
    })
});
```

### Typed State (Future)

```rust
#[derive(State)]
struct MyState {
    #[reducer = "Append"]
    messages: Vec<Message>,
    
    #[reducer = "Overwrite"]
    current_tool: String,
}

// Then use StateGraph<MyState>
let graph = StateGraph::<MyState>::builder()...
```

## Interrupts (Future)

```rust
graph
    .interrupt_before("approval_node")  // Pause before
    .interrupt_after("decision_node")   // Pause after
```

Then resume with human feedback:

```rust
match graph.invoke(state).await {
    Err(StateGraphError::InterruptedAtBreakpoint { node }) => {
        println!("Paused at: {}", node);
        // Let human intervene
        graph.resume(&thread_id).await?
    }
    _ => { }
}
```

## Debugging

### Verbose Logging

```rust
// Enable debug logging
tracing_subscriber::fmt()
    .with_max_level(tracing::Level::DEBUG)
    .init();

// Then run graph — logs will show:
// - Which node is executing
// - State before/after
// - Execution time
// - Routing decisions
```

### Visualize Graph (Future)

```rust
graph.visualize()?;  // ASCII diagram

// →  outputs:
// START
//   ↓
// [agent] ←──┐
//   │        │
//   ├─(router)
//   │        │
//   ├─→ [tool_executor] ──┘
//   │
//   └─→ END
```

## Error Handling

### Handle Execution Errors

```rust
match graph.invoke(state).await {
    Ok(result) => { /* success */ }
    Err(StateGraphError::NodeNotFound(name)) => {
        eprintln!("Node not registered: {}", name);
    }
    Err(StateGraphError::ExecutionError { node, reason }) => {
        eprintln!("Error in {}: {}", node, reason);
    }
    Err(StateGraphError::Timeout(_)) => {
        eprintln!("Execution took too long");
    }
    Err(e) => {
        eprintln!("Other error: {}", e);
    }
}
```

### Resumefrom Checkpoint on Error

```rust
let thread_id = "my-session-id";

loop {
    match graph.invoke_with_id(thread_id.into(), initial_state).await {
        Ok(result) => break Ok(result),
        Err(StateGraphError::ExecutionError { .. }) => {
            eprintln!("Error, resuming from checkpoint...");
            // Checkpoint auto-saved, resume will continue from where it left off
            continue;
        }
        Err(e) => break Err(e),
    }
}
```

---

## API Checklist

### StateGraphBuilder<S>

- ✅ `new()` — Create builder
- ✅ `add_node(name, Arc<dyn Node<S>>)` — Register node
- ✅ `add_edge(from, to)` — Fixed edge
- ✅ `add_conditional_edge(from, router)` — Conditional routing
- ✅ `set_entry_point(name)` — Graph start
- ✅ `set_checkpointer(Arc<dyn Checkpointer<S>>)` — Persistence
- ✅ `set_max_steps(usize)` — Safety limit
- ✅ `interrupt_before(name)` — Debug breakpoint
- ✅ `interrupt_after(name)` — Debug breakpoint
- ✅ `compile()` → `StateGraph<S>` — Build & validate

### StateGraph<S>

- ✅ `invoke(state)` — Run with new thread_id
- ✅ `invoke_with_id(thread_id, state)` — Run or resume
- ✅ `resume(thread_id)` — Continue from checkpoint
- ✅ `history(thread_id)` — List checkpoints
- ✅ `clear_history(thread_id)` — Delete checkpoints

### Future APIs

- ⏳ `stream()` — Async iterator over states
- ⏳ `batch(states)` — Multiple inputs
- ⏳ `visualize()` — Mermaid/ASCII diagram
- ⏳ `validate()` — Check for unreachable nodes
- ⏳ Parallel nodes — Concurrent execution
- ⏳ Error handlers — Node-specific error routing
- ⏳ Custom reducers — Per-field merge logic
- ⏳ Middleware — Logging, auth, observability

---

## Full Example: Search Agent

```rust
use serde_json::json;
use std::sync::Arc;
use flowgentra_ai::core::{FunctionNode, StateGraph, END};
use flowgentra_ai::core::state::PlainState;

// Agent decides: search or done?
async fn agent_node(state: &PlainState) {
    let mut new = state.clone();
    if new.get("queries_left").as_u64().unwrap_or(1) > 0 {
        new.set("action", json!("search"));
    } else {
        new.set("action", json!("done"));
    }
    Ok(new)
}

// Search tool
async fn search_node(state: &PlainState) {
    let mut new = state.clone();
    let results = vec![ json!("Result 1"), json!("Result 2") ];
    new.set("search_results", json!(results));
    let queries = new.get("queries_left").as_u64().unwrap_or(1);
    new.set("queries_left", json!(queries - 1));
    Ok(new)
}

#[tokio::main]
async fn main() -> flowgentra_ai::core::Result<()> {
    let agent = Arc::new(FunctionNode::new("agent", agent_node));
    let search = Arc::new(FunctionNode::new("search", search_node));

    let graph = StateGraph::builder()
        .add_node("agent", agent)
        .add_node("search", search)
        .set_entry_point("agent")
        .add_conditional_edge("agent", Box::new(|state| {
            match state.get("action").as_str() {
                Some("search") => Ok("search".into()),
                _ => Ok(END.into()),
            }
        }))
        .add_edge("search", "agent")
        .compile()?;

    let mut state = PlainState::new();
    state.set("queries_left", json!(2));

    let result = graph.invoke(state).await?;
    println!("Final results: {:?}", result.get("search_results"));

    Ok(())
}
```

---

See `examples/state_graph_react_agent.rs` for a complete working example.
