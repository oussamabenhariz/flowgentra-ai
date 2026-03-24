# Graph Engine Guide

Build custom workflows as directed graphs with nodes, edges, conditional routing, subgraphs, and parallel execution.

## Two Graph Systems

FlowgentraAI provides two ways to build graphs:

| System | API | Best For |
|--------|-----|----------|
| **StateGraph** | Programmatic Rust builder | Custom workflows with full type safety |
| **Config-driven** | YAML config files | Declarative workflows loaded at runtime |

This guide covers both. The StateGraph API is recommended for most use cases.

---

## StateGraph Builder

### Basic Graph

```rust
use flowgentra_ai::core::state_graph::StateGraphBuilder;
use flowgentra_ai::core::state::PlainState;

async fn step_a(mut state: PlainState) -> Result<PlainState> {
    state.set("processed", json!(true));
    Ok(state)
}

async fn step_b(mut state: PlainState) -> Result<PlainState> {
    state.set("output", json!("done"));
    Ok(state)
}

let graph = StateGraphBuilder::new()
    .add_fn("step_a", step_a)
    .add_fn("step_b", step_b)
    .set_entry_point("step_a")
    .add_edge("step_a", "step_b")
    .add_edge("step_b", "__end__")
    .compile()?;

let result = graph.run(PlainState::new()).await?;
```

### Conditional Routing

Route to different nodes based on state:

```rust
let graph = StateGraphBuilder::new()
    .add_fn("classify", classify)
    .add_fn("simple", handle_simple)
    .add_fn("complex", handle_complex)
    .set_entry_point("classify")
    .add_conditional_edge("classify", |state| {
        let score = state.get("score")
            .and_then(|v| v.as_i64())
            .unwrap_or(0);
        if score > 70 {
            Ok("complex".into())
        } else {
            Ok("simple".into())
        }
    })
    .add_edge("simple", "__end__")
    .add_edge("complex", "__end__")
    .compile()?;
```

### Async Conditional Edges

When routing decisions need async operations (API calls, DB lookups):

```rust
builder.add_async_conditional_edge(
    "check_status",
    Box::new(|state| {
        Box::pin(async move {
            // Call an external service to decide routing
            let status = check_external_api(&state).await?;
            Ok(status.next_step)
        })
    })
)
```

---

## Subgraph Composition

Nest an entire compiled graph as a single node inside a parent graph:

```rust
use flowgentra_ai::core::state_graph::SubgraphNode;

// Build the inner graph
let validation_graph = StateGraphBuilder::new()
    .add_fn("check_format", check_format)
    .add_fn("check_content", check_content)
    .set_entry_point("check_format")
    .add_edge("check_format", "check_content")
    .add_edge("check_content", "__end__")
    .compile()?;

// Use it as a node in a parent graph
let pipeline = StateGraphBuilder::new()
    .add_subgraph("validate", validation_graph)
    .add_fn("process", process_data)
    .add_fn("output", format_output)
    .set_entry_point("validate")
    .add_edge("validate", "process")
    .add_edge("process", "output")
    .add_edge("output", "__end__")
    .compile()?;
```

This keeps complex logic encapsulated. Each subgraph is a self-contained unit.

---

## Parallel Execution

Run multiple branches concurrently and merge results:

```rust
use flowgentra_ai::core::runtime::parallel::{ParallelExecutor, JoinType, MergeStrategy};
use std::time::Duration;

let executor = ParallelExecutor::new()
    .with_join_type(JoinType::WaitAll)         // Wait for all branches
    .with_merge_strategy(MergeStrategy::Merge) // Deep merge results
    .with_timeout(Duration::from_secs(30))     // Timeout per branch
    .with_continue_on_error(true);             // Don't fail if one branch errors

let result = executor.execute(branches, initial_state).await?;
```

### Join Strategies

| Strategy | Behavior |
|----------|----------|
| `WaitAll` | Wait for every branch to complete |
| `WaitAny` | Return as soon as any branch finishes |
| `WaitCount(n)` | Return after `n` branches complete |
| `WaitTimeout` | Return whatever completes within the timeout |

---

## Graph Export

Visualize your graph structure in multiple formats:

```rust
let graph = builder.compile()?;

// Graphviz DOT format
let dot = graph.to_dot();
println!("{}", dot);
// digraph { "step_a" -> "step_b"; "step_b" -> "__end__"; }

// Mermaid format (paste into GitHub Markdown)
let mermaid = graph.to_mermaid();
println!("{}", mermaid);
// graph TD; step_a --> step_b; step_b --> __end__;

// JSON format (for custom tooling)
let json = graph.to_json();
```

Render DOT output with Graphviz (`dot -Tpng graph.dot -o graph.png`) or paste Mermaid into any Markdown renderer.

---

## Human-in-the-Loop

Pause execution before or after a node for human review:

```rust
let graph = StateGraphBuilder::new()
    .add_fn("draft", draft_email)
    .add_fn("send", send_email)
    .set_entry_point("draft")
    .interrupt_before("send")  // Pause before sending
    .add_edge("draft", "send")
    .add_edge("send", "__end__")
    .compile()?;

// First run: executes "draft", then pauses
let partial = graph.run(state).await?;

// Human reviews and edits the draft...
let mut edited = partial.clone();
edited.set("draft_body", json!("Revised email body"));

// Resume with the edited state
let final_result = graph.resume_with_state("thread-1", edited).await?;
```

You can also use `interrupt_after("node_name")` to pause after a node completes.

---

## Config-Driven Graphs

Define workflows in YAML and load them at runtime:

```yaml
graph:
  nodes:
    - name: validate
      handler: handlers::validate_input
    - name: process
      handler: handlers::process_data
    - name: output
      handler: handlers::format_output

  edges:
    - from: START
      to: validate
    - from: validate
      to: process
    - from: process
      to: output
    - from: output
      to: END
```

```rust
let agent = from_config_path("config.yaml")?;
agent.run().await?;
```

### Conditional Edges in YAML

```yaml
edges:
  - from: classify
    to: simple_handler
    condition:
      field: "complexity"
      operator: "<"
      value: 5

  - from: classify
    to: complex_handler
    condition:
      field: "complexity"
      operator: ">="
      value: 5
```

---

## MessageGraphBuilder

A convenience wrapper for chat-focused workflows. Automatically manages a `"messages"` array in state with append semantics.

```rust
use flowgentra_ai::prelude::*;

let graph = MessageGraphBuilder::new()
    .add_fn("echo", |state: &PlainState| {
        let messages = MessageGraphBuilder::get_messages(state);
        let last = messages.last().map(|m| m.content.clone()).unwrap_or_default();
        let mut s = state.clone();
        s = MessageGraphBuilder::add_message(s, Message::assistant(format!("Echo: {}", last)));
        Box::pin(async move { Ok(s) })
    })
    .set_entry_point("echo")
    .add_edge("echo", "__end__")
    .compile()?;

// Create state with initial messages
let state = MessageGraphBuilder::initial_state(vec![Message::user("Hello")]);
let result = graph.invoke(state).await?;

let messages = MessageGraphBuilder::get_messages(&result);
assert_eq!(messages[1].content, "Echo: Hello");
```

Helper functions:
- `MessageGraphBuilder::initial_state(messages)` -- create state with initial messages
- `MessageGraphBuilder::get_messages(state)` -- extract messages from state
- `MessageGraphBuilder::add_message(state, message)` -- append a message to state
- `MessageGraphBuilder::last_message(state)` -- get the most recent message

---

## ToolNode

Prebuilt components for ReAct-style tool execution patterns.

### create_tool_node

Creates a graph node that reads `tool_calls` from state, executes each one, writes results to `tool_results`, and appends tool result messages:

```rust
use flowgentra_ai::prelude::*;
use std::sync::Arc;

let tool_node = create_tool_node(Arc::new(|name, args| {
    Box::pin(async move {
        match name.as_str() {
            "calculator" => {
                let a = args.get("a").and_then(|v| v.as_i64()).unwrap_or(0);
                let b = args.get("b").and_then(|v| v.as_i64()).unwrap_or(0);
                Ok(format!("{}", a + b))
            }
            "search" => Ok("Search results here".to_string()),
            _ => Err(format!("Unknown tool: {}", name)),
        }
    })
}));

builder.add_node("tools", tool_node);
```

### tools_condition

Router function for conditional edges. Routes to the tool node when tool calls are present, or to `"__end__"` when there are none:

```rust
builder.add_conditional_edge("agent", tools_condition("tools"));
// If state has tool_calls → route to "tools"
// If no tool_calls → route to "__end__"
```

### store_tool_calls

Helper to extract tool calls from an LLM response and store them in state:

```rust
let response = llm.chat_with_tools(messages, &tools).await?;
let state = store_tool_calls(state, &response);
// state now has "tool_calls" and "last_response" keys
```

### Typical ReAct Loop

```rust
let graph = StateGraphBuilder::new()
    .add_fn("agent", agent_fn)         // LLM generates response + tool calls
    .add_node("tools", tool_node)       // Execute tool calls
    .set_entry_point("agent")
    .add_conditional_edge("agent", tools_condition("tools"))
    .add_edge("tools", "agent")         // Loop back after tool execution
    .compile()?;
```

---

## Placeholder Nodes

Config-driven nodes that haven't been bound to a handler are marked as placeholders. If execution reaches a placeholder node, it returns a clear error instead of silently passing through:

```
ExecutionError: Node 'unregistered_handler' is a placeholder and cannot be executed.
Register a handler for this node before running the graph.
```

---

## Best Practices

1. **Name nodes clearly** -- `validate_input` not `step1`
2. **Always terminate paths** -- Every path should reach `__end__` (StateGraph) or `END` (config)
3. **Keep conditions simple** -- Complex routing logic belongs in a dedicated classifier node
4. **Test each path** -- Write tests with state that triggers each branch
5. **Use subgraphs for reuse** -- Extract repeated patterns into subgraphs
6. **Export for debugging** -- Use `to_dot()` or `to_mermaid()` to visualize complex graphs

---

See [state/README.md](../state/README.md) for state management details.
See [FEATURES.md](../FEATURES.md) for the complete feature list.
