# Flowgentra StateGraph Architecture

## Overview

The StateGraph system is a **LangGraph-inspired, Rust-idiomatic graph execution engine** designed for building AI agent workflows. It prioritizes:

1. **Type Safety** -- State schema enforced at compile time via Rust generics
2. **Async-First** -- All operations are async-compatible via `async_trait` and `tokio`
3. **Zero-Copy** -- Shared ownership via `Arc<>` minimizes allocations
4. **Fault Tolerance** -- Built-in checkpointing, recovery, and human-in-the-loop interrupts
5. **Ergonomics** -- Builder pattern, proc macros, and convenience methods for intuitive graph construction
6. **Composability** -- Subgraph composition and parallel execution for complex workflows

---

## Module Structure

```
state_graph/
├── error.rs              # Error types
├── node.rs               # Node trait, FunctionNode, UpdateNode (with ReducerConfig), SubgraphNode
├── edge.rs               # Edge definitions (Fixed, Conditional, AsyncConditional)
├── checkpoint.rs         # Checkpointer trait + InMemoryCheckpointer
├── file_checkpointer.rs  # FileCheckpointer (JSON persistence to disk)
├── executor.rs           # StateGraph<S>, StateGraphBuilder<S>, ParallelExecutor
├── reducer.rs            # JsonReducer, ReducerConfig, per-field merge strategies
├── scoped_state.rs       # ScopedState for namespaced per-node state
├── macros.rs             # #[node] proc macro for clean node definitions
└── mod.rs                # Module exports
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

- **`async` by default** -- All nodes are inherently async, enabling natural integration with tokio-based agents
- **Immutability** -- Nodes receive `&S` (immutable reference) and return a new `S` (not mutated in-place)
  - **Trade-off**: More allocations (cloning state), but prevents mutation bugs and enables checkpointing
  - **Mitigation**: Use `Arc<>` for large payloads; Rust's copy elision optimizes simple cases
- **No `dyn` in hot paths** -- Use generic `<S>` instead of trait objects to avoid vtable overhead
- **`Send + Sync`** -- Required for safe execution across thread boundaries in tokio

**Implementations:**

1. **`FunctionNode<S, F>`** -- Wraps an async closure
   ```rust
   let node = FunctionNode::new("name", |state| {
       Box::pin(async { Ok(new_state) })
   });
   ```

2. **`UpdateNode<S, F>`** -- Returns a partial `StateUpdate<S>` that gets merged using a `ReducerConfig`
   - Useful for nodes that only modify specific fields
   - Supports per-field merge strategies via `ReducerConfig`
   ```rust
   let node = UpdateNode::new("merge_results", reducer_config, |state| {
       Box::pin(async {
           let mut update = StateUpdate::new();
           update.set("messages", json!(["new message"]));
           Ok(update)
       })
   });
   ```

3. **`SubgraphNode<S>`** -- Embeds a compiled `StateGraph<S>` as a single node within a parent graph
   - Enables hierarchical graph composition
   - The inner graph runs to completion and returns its final state
   ```rust
   let inner_graph = StateGraph::builder()
       .add_node("a", node_a)
       .add_node("b", node_b)
       .set_entry_point("a")
       .add_edge("a", "b")
       .add_edge("b", END)
       .compile()?;

   let subgraph_node = SubgraphNode::new("sub_workflow", inner_graph);
   ```

**#[node] Proc Macro:**

The `#[node]` proc macro provides a clean way to define nodes without boilerplate:

```rust
#[node]
async fn classify(state: &PlainState) -> Result<PlainState> {
    let mut new_state = state.clone();
    new_state.set("category", json!("urgent"));
    Ok(new_state)
}

// Produces an Arc<dyn Node<PlainState>> ready for .add_node()
```

---

### 2. **Edge<S: State>** (edge.rs)

```rust
pub enum Edge<S: State> {
    Fixed(FixedEdge),
    Conditional {
        from: String,
        router: Box<dyn Fn(&S) -> Result<String> + Send + Sync>,
    },
    AsyncConditional {
        from: String,
        router: Box<dyn Fn(&S) -> BoxFuture<'_, Result<String>> + Send + Sync>,
    },
}
```

**Design Decisions:**

- **Fixed edges** -- Simple A->B transitions (zero overhead)
- **Conditional edges** -- Synchronous router function decides next node at runtime
  - Router is `Fn(&S) -> Result<String>` (fast for pattern matching on state)
  - Returns next node name as string (loose coupling)
- **AsyncConditional edges** -- Async router function for cases requiring external service calls
  - Router is `Fn(&S) -> BoxFuture<Result<String>>` (supports I/O-bound decisions)
  - Use when routing depends on an API call, database lookup, or other async operation
- **Special sentinels** -- `START` and `END` markers for clear graph bounds
- **Supports cycles** -- No DAG restriction; allows loops for ReAct agents
- **No compiled routing** -- Edges are determined at runtime, enabling dynamic behavior

**Synchronous Router Example:**

```rust
.add_conditional_edge("agent", Box::new(|state| {
    if state.get("done").as_bool().unwrap_or(false) {
        Ok(END.to_string())
    } else {
        Ok("tool_executor".to_string())
    }
}))
```

**Async Router Example:**

```rust
.add_async_conditional_edge("classifier", Box::new(|state| {
    Box::pin(async move {
        // Call an external classification service
        let category = external_api::classify(&state.get("text")).await?;
        match category.as_str() {
            "urgent" => Ok("priority_handler".to_string()),
            "spam" => Ok(END.to_string()),
            _ => Ok("default_handler".to_string()),
        }
    })
}))
```

---

### 3. **Checkpointer<S: State>** (checkpoint.rs, file_checkpointer.rs)

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
   - **Pros**: Zero overhead, suitable for testing and short-lived workflows
   - **Cons**: Lost on process exit

2. **`FileCheckpointer<S>`** -- Persists checkpoints to disk as JSON files
   - Stores one JSON file per checkpoint in a configurable directory
   - File naming convention: `{thread_id}_{step}.json`
   - **Pros**: Survives process restarts, human-readable, easy to debug
   - **Cons**: Slower than in-memory, not suitable for high-throughput distributed systems
   ```rust
   let checkpointer = FileCheckpointer::new("./checkpoints")?;
   let graph = StateGraph::builder()
       .set_checkpointer(Arc::new(checkpointer))
       // ...
       .compile()?;
   ```

3. **Planned Implementations:**
   - `SqliteCheckpointer` -- Database persistence
   - `RedisCheckpointer` -- Distributed checkpointing

**Key Design:**

- **After every node** -- save checkpoint (super-step completion)
- **Enables recovery** -- resume from `load_latest()` on error
- **Time-travel debugging** -- inspect state at any past step
- **Multi-session** -- each `thread_id` is independent

---

### 4. **Custom Reducers** (reducer.rs)

Reducers define how state fields are merged when multiple updates target the same field. This is critical for parallel execution and UpdateNode workflows.

**JsonReducer Variants:**

```rust
pub enum JsonReducer {
    Overwrite,     // Last write wins (default)
    Append,        // Append to array
    AppendUnique,  // Append only if value not already present
    Sum,           // Add numeric values
    Max,           // Keep the maximum numeric value
    Min,           // Keep the minimum numeric value
    DeepMerge,     // Recursively merge JSON objects
}
```

**ReducerConfig:**

Per-field merge strategies for fine-grained control:

```rust
let config = ReducerConfig::new()
    .field("messages", JsonReducer::Append)
    .field("score", JsonReducer::Sum)
    .field("metadata", JsonReducer::DeepMerge)
    .field("tags", JsonReducer::AppendUnique)
    .default(JsonReducer::Overwrite);
```

Used with `UpdateNode` for partial state updates that merge predictably:

```rust
let update_node = UpdateNode::new("scorer", config, |state| {
    Box::pin(async {
        let mut update = StateUpdate::new();
        update.set("score", json!(10));
        update.set("tags", json!(["reviewed"]));
        Ok(update)
    })
});
```

---

### 5. **SubgraphNode** (node.rs)

SubgraphNode enables composing graphs within graphs. A compiled `StateGraph<S>` can be embedded as a single node in a parent graph using `add_subgraph`:

```rust
let inner = StateGraph::builder()
    .add_node("fetch", fetch_node)
    .add_node("parse", parse_node)
    .set_entry_point("fetch")
    .add_edge("fetch", "parse")
    .add_edge("parse", END)
    .compile()?;

let outer = StateGraph::builder()
    .add_node("prepare", prepare_node)
    .add_subgraph("data_pipeline", inner)
    .add_node("summarize", summarize_node)
    .set_entry_point("prepare")
    .add_edge("prepare", "data_pipeline")
    .add_edge("data_pipeline", "summarize")
    .add_edge("summarize", END)
    .compile()?;
```

**Design:**
- The inner graph runs to completion when the subgraph node executes
- State flows in and out seamlessly (same `S` type required)
- Inner graph checkpoints are scoped to avoid collision with outer graph
- Enables reusable workflow components

---

### 6. **ParallelExecutor** (executor.rs)

The ParallelExecutor enables concurrent execution of independent nodes within a single super-step.

```rust
pub struct ParallelExecutor {
    pub join_type: JoinType,
    pub merge_strategy: MergeStrategy,
}

pub enum JoinType {
    WaitAll,    // Wait for all branches to complete
    WaitAny,   // Proceed as soon as one branch completes
}

pub enum MergeStrategy {
    ReducerBased(ReducerConfig),  // Merge via per-field reducers
    LastWriteWins,                 // Simple overwrite
}
```

**Usage in graph construction:**

```rust
let graph = StateGraph::builder()
    .add_node("branch_a", node_a)
    .add_node("branch_b", node_b)
    .add_node("merge", merge_node)
    .set_entry_point("start")
    .add_parallel(
        "start",
        vec!["branch_a", "branch_b"],
        ParallelExecutor {
            join_type: JoinType::WaitAll,
            merge_strategy: MergeStrategy::ReducerBased(reducer_config),
        },
    )
    .add_edge("branch_a", "merge")
    .add_edge("branch_b", "merge")
    .add_edge("merge", END)
    .compile()?;
```

Nodes `branch_a` and `branch_b` execute concurrently via `tokio::spawn`. Their outputs are merged according to the specified `MergeStrategy` before proceeding to `merge`.

---

### 7. **Human-in-the-Loop** (executor.rs)

The StateGraph supports interrupt-based human-in-the-loop workflows. Execution can be paused before or after specific nodes, allowing external review, approval, or modification of state.

**Setting interrupts:**

```rust
let graph = StateGraph::builder()
    .add_node("draft", draft_node)
    .add_node("publish", publish_node)
    .set_entry_point("draft")
    .add_edge("draft", "publish")
    .add_edge("publish", END)
    .interrupt_before("publish")   // Pause before publishing
    .compile()?;
```

**Handling the interrupt:**

```rust
match graph.invoke_with_id(thread_id.clone(), state).await {
    Err(StateGraphError::InterruptedAtBreakpoint { node }) => {
        println!("Paused before node: {}", node);

        // Human reviews and modifies state
        let mut modified_state = graph
            .checkpointer
            .load_latest(&thread_id)
            .await?
            .unwrap()
            .state;
        modified_state.set("approved", json!(true));

        // Resume with the modified state
        let result = graph.resume_with_state(&thread_id, modified_state).await?;
    }
    Ok(result) => { /* completed without interrupt */ }
    Err(e) => { /* handle other errors */ }
}
```

**`interrupt_after`** works the same way but pauses after the node has executed, allowing review of its output before proceeding.

**`resume_with_state`** allows injecting modified state when resuming, enabling the human to alter the workflow's trajectory.

---

### 8. **ScopedState** (scoped_state.rs)

ScopedState provides namespaced, per-node state isolation. Each node can read and write to its own namespace without colliding with other nodes' state:

```rust
let scoped = ScopedState::new("classifier", &global_state);
let confidence = scoped.get("confidence"); // reads "classifier.confidence"
scoped.set("result", json!("positive"));   // writes "classifier.result"
```

Useful for parallel branches that operate on overlapping field names.

---

### 9. **StateGraph<S: State>** (executor.rs)

```rust
pub struct StateGraph<S: State> {
    nodes: HashMap<String, Arc<dyn Node<S>>>,
    edges: Vec<Edge<S>>,
    entry_point: String,
    checkpointer: Arc<dyn Checkpointer<S>>,
    max_steps: usize,
    interrupt_before: HashSet<String>,
    interrupt_after: HashSet<String>,
}

impl<S: State + Send + Sync + 'static> StateGraph<S> {
    pub async fn invoke(&self, initial_state: S) -> Result<S> { }
    pub async fn invoke_with_id(&self, thread_id: String, initial_state: S) -> Result<S> { }
    pub async fn resume(&self, thread_id: &str) -> Result<S> { }
    pub async fn resume_with_state(&self, thread_id: &str, state: S) -> Result<S> { }

    // Accessors
    pub fn node_names(&self) -> Vec<&str> { }
    pub fn entry_point(&self) -> &str { }

    // Graph export
    pub fn to_dot(&self) -> String { }
    pub fn to_mermaid(&self) -> String { }
    pub fn to_json(&self) -> serde_json::Value { }
}
```

**Builder Pattern:**

```rust
StateGraph::builder()
    .add_node("node1", node1_arc)
    .add_node("node2", node2_arc)
    .add_fn("node3", |state| Box::pin(async { Ok(new_state) }))  // convenience method
    .add_subgraph("sub", compiled_subgraph)
    .add_edge("node1", "node2")
    .add_conditional_edge("node2", router_fn)
    .add_async_conditional_edge("node3", async_router_fn)
    .set_entry_point("node1")
    .set_checkpointer(Arc::new(checkpointer))
    .interrupt_before("node2")
    .interrupt_after("node3")
    .compile()?
```

**Graph Export:**

```rust
let graph = StateGraph::builder()
    // ... build graph ...
    .compile()?;

// Graphviz DOT format
let dot = graph.to_dot();
std::fs::write("graph.dot", &dot)?;

// Mermaid diagram (for Markdown docs)
let mermaid = graph.to_mermaid();
println!("{}", mermaid);

// JSON representation (for tooling and serialization)
let json = graph.to_json();
```

**Execution Model (Pregel-style super-steps):**

1. **Activation** -- Start at entry point
2. **Execution** -- Call `node.execute(state)`, concurrently if using ParallelExecutor
3. **Checkpoint** -- Save state after each node
4. **Interrupt Check** -- Pause if node is in interrupt_before or interrupt_after set
5. **Routing** -- Evaluate edges (sync or async) to find next active nodes
6. **Repeat** -- Until `END` sentinel reached, interrupt triggered, or error

**Code Flow:**

```rust
async fn invoke_with_id(&self, thread_id, initial_state) {
    let mut current_state = initial_state;
    let mut current_node = self.entry_point.clone();
    let mut step = 0;

    loop {
        // Check interrupt_before
        if self.interrupt_before.contains(&current_node) {
            return Err(StateGraphError::InterruptedAtBreakpoint { node: current_node });
        }

        // Execute node
        let new_state = self.nodes[&current_node].execute(&current_state).await?;
        current_state = new_state;

        // Checkpoint
        let checkpoint = Checkpoint::new(thread_id, step, current_node, current_state.clone());
        self.checkpointer.save(&checkpoint).await?;

        // Check interrupt_after
        if self.interrupt_after.contains(&current_node) {
            return Err(StateGraphError::InterruptedAtBreakpoint { node: current_node });
        }

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

- Allows multiple references to same node (shared ownership)
- Can build graph incrementally and reuse nodes
- One level of indirection (vtable lookup per execution)
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

Enables checkpointing (no side effects), prevents data races in parallel nodes, but allocates new state per node.

**Alternative: Mutable in-place**

```rust
async fn execute(&mut self) -> Result<()>
```

Prevents checkpointing (need immutable snapshot), prevents parallel execution (borrow checker), but uses a single allocation.

**Decision**: Immutable wins for a general-purpose framework. For perf-critical agents, users can implement state pooling/reuse.

---

### 3. **Router Function Type**

**Synchronous routers: `Fn(&S) -> Result<String>`**

Simple branching logic is typically synchronous (pattern matching on state). No need for `async_trait` overhead.

**Async routers: `Fn(&S) -> BoxFuture<Result<String>>`**

Added via the `AsyncConditional` edge variant for cases where routing depends on external service calls, database lookups, or other async operations. This keeps the common case fast while supporting the less common async routing path.

---

### 4. **Generic Types vs. Trait Objects**

**Node trait with generic state:**

```rust
trait Node<S: State>: Send + Sync  // Generic
impl<S> Node<S> for FunctionNode<S, F>
```

Compile-time type checking of state. No vtable lookup in hot paths (if monomorphized). StateGraph<S> must know S at compile time. Cannot store `Node<PlainState>` and `Node<MyState>` in same Vec.

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

Single runtime (tokio). Efficient context switching. Non-blocking I/O (LLM calls, tools). ParallelExecutor leverages `tokio::spawn` for concurrent branch execution.

**Parallel super-steps (implemented via ParallelExecutor):**

```rust
// Spawn multiple independent nodes in parallel
let handles: Vec<_> = active_nodes.into_iter().map(|node| {
    tokio::spawn(node.execute(state.clone()))
}).collect();

// Merge results via ReducerConfig
let merged = reducer_config.merge(results)?;
```

State must be cheap to clone (`Arc<>` for large data). Merge strategies must be associative and commutative for deterministic results.

---

## Example: ReAct Agent Loop

See `examples/state_graph_react_agent.rs` for full code.

**Graph Structure:**

```
START
  |
[agent] <-- decide: use tool or finish?
  |
[router] -- conditional edge
  |-> [tool_executor] -> (back to agent)
  |-> END
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
  -> LLM response: "I'll search for..."
  -> has_tool_call = true
  -> Checkpoint 0 saved

Step 1: router evaluates state
  -> Sees has_tool_call = true
  -> Routes to tool_executor

Step 1: tool_executor executes
  -> Calls search tool
  -> Appends result to tool_results
  -> Checkpoint 1 saved

Step 2: router evaluates state
  -> Sees has_tool_call = false
  -> Routes to END

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
    ReducerError(String),
    SubgraphError { subgraph: String, reason: String },
    // ...
}
```

**Recovery pattern:**

```rust
match graph.invoke(state).await {
    Err(StateGraphError::ExecutionError { .. }) => {
        // Resume from last checkpoint
        let recovered = graph.resume(thread_id).await?;
    }
    Err(StateGraphError::InterruptedAtBreakpoint { node }) => {
        // Human reviews, modifies state, resumes
        let modified = get_human_approval(thread_id).await;
        graph.resume_with_state(thread_id, modified).await?;
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
| Router evaluation (sync) | O(1) | Simple conditional logic |
| Router evaluation (async) | O(1) + I/O | Depends on external call latency |
| Checkpoint save (memory) | O(state_size) | InMemoryCheckpointer |
| Checkpoint save (file) | O(state_size) | FileCheckpointer, includes disk I/O |
| Graph lookup (edges) | O(edges) | Currently linear; could be optimized with btree |
| Parallel execution | O(slowest_branch) | Bounded by longest-running parallel node |
| Reducer merge | O(fields) | Per-field merge via ReducerConfig |
| Graph export (dot/mermaid) | O(nodes + edges) | Single-pass traversal |

---

## Future Enhancements

1. **Stream Output** -- `stream()` method yielding state after each step
2. **Batch Execution** -- `batch(states)` for multiple inputs
3. **Error Recovery Nodes** -- Dedicated error handling paths in the graph
4. **State Validation** -- Per-node pre/post conditions
5. **Persistence Backends** -- SQLite, Redis, PostgreSQL checkpointers
6. **Middleware** -- Hooks for logging, observability, auth

---

## Comparison with LangGraph (Python)

| Feature | LangGraph | Flowgentra StateGraph |
|---------|-----------|-------------------|
| Type Safety | Runtime (Pydantic) | Compile-time (Rust generics) |
| State | `StateGraph(MessageState)` | `StateGraph<S: State>` |
| Nodes | `.add_node("name", func)` | `.add_node("name", Arc::new(node))` or `.add_fn("name", fn)` |
| Proc Macro | N/A (Python decorators) | `#[node]` for zero-boilerplate node definitions |
| Edges | `.add_edge("a", "b")` | `.add_edge("a", "b")` |
| Conditional Routing | Dict of node -> router | Sync router fn or async router fn |
| Async Routing | Built-in (Python async) | `AsyncConditional` edge variant |
| Checkpointer | `MemorySaver`, `SqliteSaver` | `InMemoryCheckpointer`, `FileCheckpointer` |
| Parallel Nodes | `parallelize=True` | `ParallelExecutor` with `JoinType` and `MergeStrategy` |
| Subgraphs | `add_node("sub", subgraph)` | `add_subgraph("sub", compiled_graph)` |
| Reducers | `Annotated[list, add]` | `JsonReducer` (Overwrite, Append, Sum, DeepMerge, Max, Min, AppendUnique) |
| Human-in-the-Loop | `interrupt_before`, `interrupt_after` | `interrupt_before()`, `interrupt_after()`, `resume_with_state()` |
| Visualization | `.get_graph().draw_mermaid()` | `to_mermaid()`, `to_dot()`, `to_json()` |
| Scoped State | Channels | `ScopedState` namespaces |

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

Key capabilities:
- **Generic state system** enforces schema at compile time
- **Async-first architecture** for modern I/O patterns
- **Checkpointing** with in-memory and file-based persistence
- **Builder API** with convenience methods (`add_fn`, `add_subgraph`) and proc macros (`#[node]`)
- **Parallel execution** via `ParallelExecutor` with configurable join and merge strategies
- **Human-in-the-loop** via interrupts and `resume_with_state`
- **Graph composition** via `SubgraphNode` for reusable workflow components
- **Custom reducers** for deterministic per-field state merging
- **Graph export** to DOT, Mermaid, and JSON for visualization and tooling
