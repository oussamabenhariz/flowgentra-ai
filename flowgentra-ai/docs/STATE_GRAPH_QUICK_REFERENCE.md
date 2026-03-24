# StateGraph Quick Reference

## Basic Usage

### 1. Create Nodes

```rust
// Option A: Use add_fn (simplest)
builder.add_fn("my_node", |mut state: PlainState| async move {
    state.set("result", json!("success"));
    Ok(state)
})

// Option B: Use #[node] macro
#[node]
async fn my_node(mut state: PlainState) -> Result<PlainState> {
    state.set("result", json!("success"));
    Ok(state)
}
// Generates: my_node_node() factory function

// Option C: Arc<dyn Node<S>> (manual)
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
let graph = StateGraphBuilder::new()
    .add_fn("step1", step1_fn)
    .add_fn("step2", step2_fn)
    .set_entry_point("step1")
    .add_edge("step1", "step2")
    .add_edge("step2", "__end__")
    .compile()?;
```

### 3. Execute

```rust
let initial_state = PlainState::new();
let final_state = graph.run(initial_state).await?;
```

### 4. Conditional Routing

```rust
builder.add_conditional_edge("step1", |state| {
    if state.get("done").and_then(|v| v.as_bool()).unwrap_or(false) {
        Ok("__end__".to_string())
    } else {
        Ok("step2".to_string())
    }
})
```

### 5. Async Conditional Routing

```rust
builder.add_async_conditional_edge("step1", Box::new(|state| {
    Box::pin(async move {
        let result = call_api(&state).await?;
        Ok(result.next_step)
    })
}))
```

---

## Common Patterns

### Simple Pipeline

```
step1 -> step2 -> step3 -> END
```

```rust
builder
    .add_edge("step1", "step2")
    .add_edge("step2", "step3")
    .add_edge("step3", "__end__")
```

### ReAct Agent Loop

```
    +-> tool_executor -+
    |                   |
  agent ---router------+
    |
   END
```

```rust
builder
    .add_fn("agent", agent_fn)
    .add_fn("tool_executor", tool_fn)
    .set_entry_point("agent")
    .add_conditional_edge("agent", |state| {
        if needs_tool(state) { Ok("tool_executor".into()) }
        else { Ok("__end__".into()) }
    })
    .add_edge("tool_executor", "agent")
```

### Parallel Branches

```
        +-> branch1 -+
start --+             +-> merge -> END
        +-> branch2 -+
```

```rust
let executor = ParallelExecutor::new()
    .with_join_type(JoinType::WaitAll)
    .with_merge_strategy(MergeStrategy::Merge);

let result = executor.execute(branches, state).await?;
```

### Subgraph Composition

```rust
let inner = StateGraphBuilder::new()
    .add_fn("a", step_a)
    .add_fn("b", step_b)
    .set_entry_point("a")
    .add_edge("a", "b")
    .add_edge("b", "__end__")
    .compile()?;

builder.add_subgraph("pipeline", inner)
```

---

## Checkpointing

### InMemoryCheckpointer (default)

```rust
let checkpointer = InMemoryCheckpointer::new();
```

### FileCheckpointer (durable)

```rust
let checkpointer = FileCheckpointer::new("./checkpoints");
```

### Save and Resume

```rust
// Checkpoints are saved automatically during graph execution

// Resume from latest checkpoint
let recovered = graph.resume(&thread_id).await?;

// Resume with modified state (human-in-the-loop)
let result = graph.resume_with_state(&thread_id, edited_state).await?;
```

---

## Interrupts (Human-in-the-Loop)

```rust
builder
    .interrupt_before("approval_node")  // Pause before
    .interrupt_after("decision_node")   // Pause after
```

```rust
match graph.invoke_with_id(thread_id, state).await {
    Err(StateGraphError::InterruptedAtBreakpoint { node }) => {
        println!("Paused at: {}", node);
        // Human reviews and edits state...
        graph.resume_with_state(&thread_id, modified_state).await?
    }
    Ok(result) => result,
    Err(e) => return Err(e),
}
```

---

## State Reducers

```rust
let config = ReducerConfig::default()
    .field("messages", JsonReducer::Append)
    .field("score", JsonReducer::Sum)
    .field("config", JsonReducer::DeepMerge)
    .field("best", JsonReducer::Max)
    .field("tags", JsonReducer::AppendUnique);
```

Available: `Overwrite`, `Append`, `Sum`, `DeepMerge`, `Max`, `Min`, `AppendUnique`.

---

## Graph Export

```rust
let graph = builder.compile()?;

let dot = graph.to_dot();         // Graphviz DOT
let mermaid = graph.to_mermaid(); // Mermaid diagram
let json = graph.to_json();       // JSON structure
```

---

## Error Handling

```rust
match graph.run(state).await {
    Ok(result) => { /* success */ }
    Err(StateGraphError::NodeNotFound(name)) => { /* missing node */ }
    Err(StateGraphError::ExecutionError { node, reason }) => { /* node failed */ }
    Err(StateGraphError::InterruptedAtBreakpoint { node }) => { /* paused */ }
    Err(StateGraphError::Timeout(_)) => { /* took too long */ }
    Err(e) => { /* other */ }
}
```

---

## API Checklist

### StateGraphBuilder

| Method | Status | Description |
|--------|--------|-------------|
| `new()` | Done | Create builder |
| `add_node(name, node)` | Done | Register node (Arc) |
| `add_fn(name, fn)` | Done | Register node (closure) |
| `add_subgraph(name, graph)` | Done | Embed subgraph as node |
| `add_edge(from, to)` | Done | Fixed edge |
| `add_conditional_edge(from, router)` | Done | Sync routing |
| `add_async_conditional_edge(from, router)` | Done | Async routing |
| `set_entry_point(name)` | Done | Graph start |
| `set_checkpointer(checkpointer)` | Done | Persistence backend |
| `set_max_steps(n)` | Done | Safety limit |
| `interrupt_before(name)` | Done | Pause before node |
| `interrupt_after(name)` | Done | Pause after node |
| `compile()` | Done | Build and validate |

### StateGraph

| Method | Status | Description |
|--------|--------|-------------|
| `run(state)` | Done | Execute graph |
| `invoke_with_id(id, state)` | Done | Execute with thread ID |
| `resume(thread_id)` | Done | Resume from checkpoint |
| `resume_with_state(id, state)` | Done | Resume with modified state |
| `node_names()` | Done | List all nodes |
| `entry_point()` | Done | Get entry point name |
| `to_dot()` | Done | Export Graphviz DOT |
| `to_mermaid()` | Done | Export Mermaid diagram |
| `to_json()` | Done | Export JSON structure |

---

See [STATE_GRAPH_DESIGN.md](STATE_GRAPH_DESIGN.md) for architecture details.
See [graph/README.md](graph/README.md) for the user guide.
