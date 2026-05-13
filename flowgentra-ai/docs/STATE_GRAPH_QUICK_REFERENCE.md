# StateGraph Quick Reference

## Basic Usage

### 1. Create Nodes

```rust
// Option A: Use add_fn (simplest)
builder.add_fn("my_node", |state: &mut MyState| async move {
    state.result = "success".into();
    Ok(())
})

// Option B: Use #[node] macro
#[node]
async fn my_node(state: &mut MyState) -> Result<()> {
    state.result = "success".into();
    Ok(())
}

```

### 2. Build a Graph

```rust
// #[derive(State, Default, Clone)] struct MyState { ... }
let graph = StateGraph::<MyState>::builder()
    .add_fn("step1", step1_fn)
    .add_fn("step2", step2_fn)
    .set_entry("step1")
    .add_edge("step1", "step2")
    .set_finish("step2")
    .build()?;
```

### 3. Execute

```rust
let initial_state = MyState::default();
let final_state = graph.invoke(initial_state).await?;
```

### 4. Conditional Routing

```rust
builder.conditional_edge("step1", |state: &MyState| {
    if state.done {
        Ok("__end__".to_string())
    } else {
        Ok("step2".to_string())
    }
})
```

### 5. Async Conditional Routing

```rust
builder.add_async_conditional_edge("step1", Box::new(|state: &MyState| {
    Box::pin(async move {
        let result = call_api(state).await?;
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
    .set_finish("step3")
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
    .set_entry("agent")
    .conditional_edge("agent", |state: &AgentState| {
        if state.needs_tool { Ok("tool_executor".into()) }
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
// #[derive(State, Default, Clone)] struct MyState { ... }
let inner = StateGraph::<MyState>::builder()
    .add_fn("a", step_a)
    .add_fn("b", step_b)
    .set_entry("a")
    .add_edge("a", "b")
    .set_finish("b")
    .build()?;

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
let graph = builder.build()?;

let dot = graph.to_dot();         // Graphviz DOT
let mermaid = graph.to_mermaid(); // Mermaid diagram
let json = graph.to_json();       // JSON structure
```

---

## Error Handling

```rust
match graph.invoke(state).await {
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
| `StateGraph::<S>::builder()` | Done | Create typed builder |
| `add_node(name, node)` | Done | Register node (Arc) |
| `add_fn(name, fn)` | Done | Register node (closure) |
| `add_subgraph(name, graph)` | Done | Embed subgraph as node |
| `add_edge(from, to)` | Done | Fixed edge |
| `set_finish(name)` | Done | Terminal edge to END |
| `conditional_edge(from, router)` | Done | Sync routing |
| `add_async_conditional_edge(from, router)` | Done | Async routing |
| `set_entry(name)` | Done | Graph start |
| `set_checkpointer(checkpointer)` | Done | Persistence backend |
| `set_max_steps(n)` | Done | Safety limit |
| `interrupt_before(name)` | Done | Pause before node |
| `interrupt_after(name)` | Done | Pause after node |
| `build()` | Done | Build and validate |

### StateGraph

| Method | Status | Description |
|--------|--------|-------------|
| `invoke(state)` | Done | Execute graph |
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
