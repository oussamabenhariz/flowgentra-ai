# Flowgentra StateGraph Architecture

## Overview

The StateGraph system is a **LangGraph-inspired, Rust-idiomatic graph execution engine** designed for building AI agent workflows. It prioritizes:

1. **Type Safety** — State schema enforced at compile time via Rust generics
2. **Async-First** — All operations are async-compatible via `async_trait` and `tokio`
3. **Zero-Copy** — Shared ownership via `Arc<>` minimizes allocations
4. **Fault Tolerance** — Built-in checkpointing and recovery
5. **Ergonomics** — Builder pattern for intuitive graph construction

---

## Module Structure

```
state_graph/
├── error.rs           # Error types
├── node.rs           # Node trait, implementations (FunctionNode, UpdateNode)
├── edge.rs           # Edge definitions (fixed, conditional)
├── checkpoint.rs     # Checkpointing: trait + InMemoryCheckpointer
├── executor.rs       # StateGraph<S>, StateGraphBuilder<S>
└── mod.rs            # Module exports
```

---

## Core Traits & Types

### 1. **Node<S: State>** (node.rs)

```rust
#[async_trait]
pub trait Node<S: State>: Send + Sync {
    async fn execute(&self, state: &S) -> Result<S>;
    fn name(&self) -> &str;
    fn is_parallelizable(&self) -> bool { true }
}
```

**Design Decisions:**

- **`async` by default** — All nodes are inherently async, enabling natural integration with tokio-based agents
- **Immutability** — Nodes receive `&S` (immutable reference) and return a new `S` (not mutated in-place)
  - **Trade-off**: More allocations (cloning state), but prevents mutation bugs and enables checkpointing
  - **Mitigation**: Use `Arc<>` for large payloads; Rust's copy elision optimizes simple cases
- **No `dyn` in hot paths** — Use generic `<S>` instead of trait objects to avoid vtable overhead
- **`Send + Sync`** — Required for safe execution across thread boundaries in tokio

**Implementations:**

1. **`FunctionNode<S, F>`** — Wraps an async closure
   ```rust
   let node = FunctionNode::new("name", |state| {
       Box::pin(async { Ok(new_state) })
   });
   ```

2. **`UpdateNode<S, F>`** — Returns a partial `StateUpdate<S>` that gets merged
   - Useful for nodes that only modify specific fields
   - Allows natural reducer integration in future versions

---

### 2. **Edge<S: State>** (edge.rs)

```rust
pub enum Edge<S: State> {
    Fixed(FixedEdge),
    Conditional {
        from: String,
        router: Box<dyn Fn(&S) -> Result<String> + Send + Sync>,
    },
}
```

**Design Decisions:**

- **Fixed edges** — Simple A→B transitions (zero overhead)
- **Conditional edges** — Router function decides next node at runtime
  - Router is `Fn(&S) -> Result<String>` (not async to keep state queries fast)
  - Returns next node name as string (loose coupling)
- **Special sentinels** — `START` and `END` markers for clear graph bounds
- **Supports cycles** — No DAG restriction; allows loops for ReAct agents
- **No compiled routing** — Edges are determined at runtime, enabling dynamic behavior

**Router Pattern Example:**

```rust
.add_conditional_edge("agent", Box::new(|state| {
    if state.get("done").as_bool().unwrap_or(false) {
        Ok(END.to_string())
    } else {
        Ok("tool_executor".to_string())
    }
}))
```

---

### 3. **Checkpointer<S: State>** (checkpoint.rs)

```rust
#[async_trait]
pub trait Checkpointer<S: State>: Send + Sync {
    async fn save(&self, checkpoint: &Checkpoint<S>) -> Result<()>;
    async fn load(&self, thread_id: &str, step: usize) -> Result<Option<Checkpoint<S>>>;
    async fn load_latest(&self, thread_id: &str) -> Result<Option<Checkpoint<S>>>;
    async fn list_checkpoints(&self, thread_id: &str) -> Result<Vec<(usize, i64)>>;
    async fn delete(&self, thread_id: &str, step: usize) -> Result<()>;
    async fn delete_thread(&self, thread_id: &str) -> Result<()>;
}
```

**Checkpoint Structure:**

```rust
pub struct Checkpoint<S: State> {
    pub thread_id: String,          // UUID for this execution session
    pub step: usize,                 // Super-step number
    pub node_name: String,           // Node that just executed
    pub state: S,                    // Full state snapshot
    pub timestamp: i64,              // Unix timestamp
    pub metadata: HashMap<String, String>,
}
```

**Implementations:**

1. **`InMemoryCheckpointer<S>`** (default)
   - `HashMap<thread_id -> HashMap<step -> Checkpoint>>`
   - Cloned via `Arc<RwLock<...>>` for concurrent access
   - **Pros**: Zero overhead, suitable for testing
   - **Cons**: Lost on process exit

2. **Planned Implementations:**
   - `FileCheckpointer` — JSON/MessagePack to disk
   - `SqliteCheckpointer` — Database persistence
   - `RedisCheckpointer` — Distributed checkpointing

**Key Design:**

- **After every node** → save checkpoint (super-step completion)
- **Enables recovery** → resume from `load_latest()` on error
- **Time-travel debugging** → inspect state at any past step
- **Multi-session** → each `thread_id` is independent

---

### 4. **StateGraph<S: State>** (executor.rs)

```rust
pub struct StateGraph<S: State> {
    nodes: HashMap<String, Arc<dyn Node<S>>>,
    edges: Vec<Edge<S>>,
    entry_point: String,
    checkpointer: Arc<dyn Checkpointer<S>>,
    max_steps: usize,
    // ... interrupt_before, interrupt_after
}

impl<S: State + Send + Sync + 'static> StateGraph<S> {
    pub async fn invoke(&self, initial_state: S) -> Result<S> { }
    pub async fn invoke_with_id(&self, thread_id: String, initial_state: S) -> Result<S> { }
    pub async fn resume(&self, thread_id: &str) -> Result<S> { }
}
```

**Builder Pattern:**

```rust
StateGraph::builder()
    .add_node("node1", node1_arc)
    .add_node("node2", node2_arc)
    .add_edge("node1", "node2")
    .add_conditional_edge("node2", router_fn)
    .set_entry_point("node1")
    .set_checkpointer(Arc::new(checkpointer))
    .compile()?
```

**Execution Model (Pregel-style super-steps):**

1. **Activation** — Start at entry point
2. **Execution** — Call `node.execute(state)` concurrently if possible
3. **Checkpoint** — Save state after each node
4. **Routing** — Evaluate edges to find next active nodes
5. **Repeat** — Until `END` sentinel reached or error

**Code Flow:**

```rust
async fn invoke_with_id(&self, thread_id, initial_state) {
    let mut current_state = initial_state;
    let mut current_node = self.entry_point.clone();
    let mut step = 0;

    loop {
        // Execute node
        let new_state = self.nodes[&current_node].execute(&current_state).await?;
        current_state = new_state;

        // Checkpoint
        let checkpoint = Checkpoint::new(thread_id, step, current_node, current_state.clone());
        self.checkpointer.save(&checkpoint).await?;

        // Route to next
        let next = self.get_next_node(&current_node, &current_state).await?;
        if next == END { return Ok(current_state); }

        current_node = next;
        step += 1;
    }
}
```

---

## Rust-Specific Design Tradeoffs

### 1. **Ownership & Lifetime**

**Why use `Arc<dyn Node<S>>`?**

```rust
nodes: HashMap<String, Arc<dyn Node<S>>>
```

- ✅ Allows multiple references to same node (shared ownership)
- ✅ Can build graph incrementally and reuse nodes
- ❌ One level of indirection (vtable lookup per execution)
- **Mitigation**: Hot path is per-super-step, not per-statement

**Alternative: Generic wrapper?**

```rust
struct StateGraph<S, N1, N2, ...> { ... }  // Not scalable
```

Rejected: Too verbose, not practical for graphs with many node types.

---

### 2. **State Mutation vs. Immutability**

**Current: Immutable (nodes return new state)**

```rust
async fn execute(&self, state: &S) -> Result<S>
```

✅ Enables checkpointing (no side effects)
✅ Prevents data races in parallel nodes
❌ Allocates new state per node

**Alternative: Mutable in-place**

```rust
async fn execute(&mut self) -> Result<()>
```

❌ Can't checkpoint (need immutable snapshot)
❌ Prevents parallel execution (borrow checker)
✅ Single allocation

**Decision**: Immutable wins for a general-purpose framework. For perf-critical agents, users can implement state pooling/reuse.

---

### 3. **Router Function Type**

**Why `Fn(&S) -> Result<String>` (not async)?**

```rust
router: Box<dyn Fn(&S) -> Result<String> + Send + Sync>
```

✅ Simple branching logic is typically synchronous (pattern matching on state)
✅ No need for `async_trait` overhead
❌ Can't do async I/O in router (rare use case)

**Alternative: `async fn(&S) -> Result<String>`**

Would require `async_trait`, but 99% of routers are simple conditional logic. Premature optimization.

---

### 4. **Generic Types vs. Trait Objects**

**Node trait with generic state:**

```rust
trait Node<S: State>: Send + Sync  // Generic
impl<S> Node<S> for FunctionNode<S, F>
```

✅ Compile-time type checking of state
✅ No vtable lookup in hot paths (if monomorphized)
❌ StateGraph<S> must know S at compile time
❌ Can't store `Node<PlainState>` and `Node<MyState>` in same Vec

**Alternative: Erased state with `dyn Any`**

```rust
trait Node: Send + Sync  // No generic
fn execute(&self, state: &dyn Any) -> Result<Box<dyn Any>>
```

❌ Lost compile-time checking
❌ Downcast errors at runtime
✅ Can mix different state types

**Decision**: Generic approach. Type safety > flexibility. If needed, users can use trait objects on top.

---

### 5. **Concurrency Model**

**Design: Async-first with tokio**

```rust
#[async_trait]
pub trait Node<S>: Send + Sync {
    async fn execute(&self, state: &S) -> Result<S>;
}
```

✅ Single runtime (tokio)
✅ Efficient context switching
✅ Non-blocking I/O (LLM calls, tools)
✅ Future-compatible with parallelization

**Future: Parallel super-steps**

```rust
// Spawn multiple independent nodes in parallel
let handles: Vec<_> = active_nodes.into_iter().map(|node| {
    tokio::spawn(node.execute(state.clone()))
}).collect();

// Merge results via reducers
```

Requires state to be cheap to clone (`Arc<>` for data) and merge strategies to be associative.

---

## Example: ReAct Agent Loop

See `examples/state_graph_react_agent.rs` for full code.

**Graph Structure:**

```
START
  ↓
[agent] ← decide: use tool or finish?
  ↓
[router] — conditional edge
  ├→ [tool_executor] → (back to agent)
  └→ END
```

**State:**

```rust
{
  "turn_count": 2,
  "last_response": { "role": "assistant", "content": "..." },
  "has_tool_call": false,
  "tool_results": [ { "tool_name": "search", "result": "..." } ]
}
```

**Execution Trace:**

```
Step 0: agent node executes
  → LLM response: "I'll search for..."
  → has_tool_call = true
  → Checkpoint 0 saved

Step 1: router evaluates state
  → Sees has_tool_call = true
  → Routes to tool_executor

Step 1: tool_executor executes
  → Calls search tool
  → Appends result to tool_results
  → Checkpoint 1 saved

Step 2: router evaluates state
  → Sees has_tool_call = false
  → Routes to END

Done. Final state returned.
```

---

## Error Handling

Custom error type with recovery:

```rust
pub enum StateGraphError {
    NodeNotFound(String),
    ExecutionError { node: String, reason: String },
    InterruptedAtBreakpoint { node: String },
    Timeout(String),
    CheckpointError(String),
    // ...
}
```

**Recovery pattern (pseudo-code):**

```rust
match graph.invoke(state).await {
    Err(StateGraphError::ExecutionError { .. }) => {
        // Resume from last checkpoint
        let recovered = graph.resume(thread_id).await?;
        // Continue execution
    }
    Err(StateGraphError::InterruptedAtBreakpoint { node }) => {
        // Human approved, resume
        graph.resume(thread_id).await?;
    }
    _ => { /* handle other errors */ }
}
```

---

## Performance Characteristics

| Operation | Complexity | Notes |
|-----------|-----------|-------|
| Node execution | O(1) | Async, no hot-path vtable overhead if monomorphized |
| State cloning | O(state_size) | Use `Arc<>` for large payloads |
| Router evaluation | O(1) | Simple conditional logic |
| Checkpoint save | O(state_size) | Depends on Checkpointer impl |
| Graph lookup (edges) | O(edges) | Currently linear; could be optimized with btree |

---

## Future Enhancements

1. **Parallel Super-Steps** — Execute independent nodes concurrently
2. **Stream Output** — `stream()` method yielding state after each step
3. **Error Recovery Nodes** — Dedicated error handling paths
4. **State Validation** — Per-node pre/post conditions
5. **Visualization** — Generate Mermaid diagrams from graph structure
6. **Persistence Backends** — SQLite, Redis, PostgreSQL
7. **Middleware** — Hooks for logging, observability, auth
8. **Custom Reducers** — Per-field merge strategies (append, sum, etc.)

---

## Comparison with LangGraph (Python)

| Feature | LangGraph | Flowgentra StateGraph |
|---------|-----------|-------------------|
| Type Safety | Runtime (Pydantic) | Compile-time (Rust generics) |
| State | `StateGraph(MessageState)` | `StateGraph<S: State>` |
| Nodes | `.add_node("name", func)` | `.add_node("name", Arc::new(node))` |
| Edges | `.add_edge("a", "b")` | `.add_edge("a", "b")` |
| Routers | Dict of node → router | Single router function |
| Checkpointer | `InMemoryCheckpointer`, `SqliteCheckpointer` | `Arc<dyn Checkpointer<S>>` |
| Parallel Nodes | `parallelize=True` | Future enhancement |

---

## Testing Strategy

All components have unit tests. Example test:

```rust
#[tokio::test]
async fn test_graph_with_checkpointing() {
    let node1 = Arc::new(FunctionNode::new("n1", ...));
    let graph = StateGraph::builder()
        .add_node("n1", node1)
        .set_entry_point("n1")
        .add_edge("n1", END)
        .compile()?;

    let result = graph.invoke(initial_state).await?;
    assert_eq!(result.get("counter"), 1);

    // Verify checkpoint was saved
    let checkpoint = graph.checkpointer
        .load_latest("thread1")
        .await?
        .expect("Checkpoint not found");
    assert_eq!(checkpoint.step, 0);
}
```

---

## Conclusion

The StateGraph system combines **Rust's type safety** with **LangGraph's intuitive graph semantics**, enabling production-grade AI agents with compile-time guarantees and zero-cost execution.

Key innovations:
- **Generic state system** enforces schema at compile time
- **Async-first architecture** for modern I/O patterns
- **Checkpointing trait** allows pluggable persistence
- **Builder API** balances flexibility with type safety
