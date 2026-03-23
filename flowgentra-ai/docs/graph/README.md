# Graph Building and Conditional Routing Guide

Learn how to build custom agent workflows using graphs and conditional routing.

## Overview

Instead of hardcoding `Node A → Node B → Node C`, FlowgentraAI lets you:
- **Build graphs programmatically** using `GraphBuilder`
- **Define conditional routing** to make dynamic decisions
- **Route based on state** to create intelligent workflows

## Building Graphs with GraphBuilder

### The Simplest Graph

```rust
use flowgentra_ai::prelude::*;

#[register_handler]
pub async fn handler1(state: State) -> Result<State> {
    Ok(state)
}

#[tokio::main]
async fn main() -> Result<()> {
    let graph = GraphBuilder::new()
        .add_node("step1", handler1)
        .add_edge("START", "step1")
        .add_edge("step1", "END")
        .build()?;
    
    Ok(())
}
```

This creates:
```
START → step1 → END
```

### Adding Multiple Nodes

```rust
#[register_handler]
pub async fn validate(state: State) -> Result<State> {
    Ok(state)
}

#[register_handler]
pub async fn process(state: State) -> Result<State> {
    Ok(state)
}

#[register_handler]
pub async fn output(state: State) -> Result<State> {
    Ok(state)
}

let graph = GraphBuilder::new()
    .add_node("validate", validate)
    .add_node("process", process)
    .add_node("output", output)
    .add_edge("START", "validate")
    .add_edge("validate", "process")
    .add_edge("process", "output")
    .add_edge("output", "END")
    .build()?;
```

This creates:
```
START → validate → process → output → END
```

### Understanding Edges

An edge connects two nodes. Use these special node names:
- `START` - Entry point (start of the graph)
- `END` - Exit point (end of the graph)
- Any other name - User-defined handler node

```rust
// Valid edges
.add_edge("START", "validate")      // Start to first node
.add_edge("validate", "process")    // Node to node
.add_edge("process", "output")      // Last node to another
.add_edge("output", "END")          // Last node to end

// Invalid - wouldn't work as expected:
.add_edge("handler1", "handler2")
.add_edge("handler1", "handler2")   // No END!
```

---

## Conditional Routing

### Basic Conditional Routing

Route to different nodes based on state:

```rust
use flowgentra_ai::prelude::*;

let graph = GraphBuilder::new()
    .add_node("validate", validate_handler)
    .add_node("simple_process", simple_handler)
    .add_node("complex_process", complex_handler)
    .add_node("output", output_handler)
    // Normal path: validate → output (no condition needed)
    .add_edge("START", "validate")
    
    // Conditional: from validate, choose based on state
    .add_edge_with_condition(
        "validate",
        "simple_process",
        Condition::compare("complexity", ComparisonOp::LessThan, 5)
    )
    .add_edge_with_condition(
        "validate",
        "complex_process",
        Condition::compare("complexity", ComparisonOp::GreaterOrEqual, 5)
    )
    
    // Both paths converge at output
    .add_edge("simple_process", "output")
    .add_edge("complex_process", "output")
    .add_edge("output", "END")
    .build()?;
```

This creates:
```
                     ┌─→ simple_process ─┐
                     │                    ↓
START → validate ←──┤                   output → END
                     │                    ↑
                     └─→ complex_process ┘
```

### How It Works

1. When execution reaches a node with conditional edges
2. Check each condition against current state
3. Take the edge where condition is `true`
4. If multiple conditions are `true`, first one is used
5. If no conditions are `true`, execution stops (error)

---

## Condition Types

### 1. Compare Values

```rust
// Equal to
Condition::compare("status", ComparisonOp::Equal, "approved")

// Not equal to
Condition::compare("status", ComparisonOp::NotEqual, "rejected")

// Greater than (numbers)
Condition::compare("score", ComparisonOp::GreaterThan, 80)

// Less than
Condition::compare("confidence", ComparisonOp::LessThan, 0.5)

// Greater or equal
Condition::compare("priority", ComparisonOp::GreaterOrEqual, 3)

// Less or equal
Condition::compare("level", ComparisonOp::LessOrEqual, 5)
```

### 2. Field Existence

```rust
// Check if field exists
Condition::field_exists("result")

// Can be combined with others
```

### 3. Field Type Checks

```rust
Condition::field_type("value", FieldTypeCheck::String)
Condition::field_type("count", FieldTypeCheck::Number)
Condition::field_type("items", FieldTypeCheck::Array)
Condition::field_type("config", FieldTypeCheck::Object)
```

### 4. Combine Conditions

```rust
// AND - all must be true
Condition::and(vec![
    Condition::compare("status", ComparisonOp::Equal, "ready"),
    Condition::compare("score", ComparisonOp::GreaterThan, 75),
])

// OR - at least one must be true
Condition::or(vec![
    Condition::compare("expedited", ComparisonOp::Equal, true),
    Condition::compare("priority", ComparisonOp::GreaterOrEqual, "high"),
])

// NOT - negation
Condition::not_condition(
    Condition::compare("status", ComparisonOp::Equal, "error")
)

// Complex: (A AND B) OR C
Condition::or(vec![
    Condition::and(vec![
        Condition::compare("method", ComparisonOp::Equal, "auto"),
        Condition::compare("approved", ComparisonOp::Equal, true),
    ]),
    Condition::compare("manual_override", ComparisonOp::Equal, true),
])
```

### 5. Custom Conditions

```rust
// Custom predicate function
Condition::custom("is_weekend", |state| {
    if let Some(day) = state.get("day_of_week").and_then(|v| v.as_str()) {
        day == "Saturday" || day == "Sunday"
    } else {
        false
    }
})
```

---

## Real-World Routing Examples

### Example 1: Support Ticket Routing

Route support tickets based on priority:

```rust
let graph = GraphBuilder::new()
    // All tickets enter here
    .add_node("categorize", categorize_handler)
    
    // Route by priority
    .add_node("urgent_team", urgent_handler)
    .add_node("standard_team", standard_handler)
    .add_node("feedback_team", feedback_handler)
    
    // Acknowledge regardless of path
    .add_node("send_confirmation", confirm_handler)
    
    .add_edge("START", "categorize")
    
    // Route to appropriate team
    .add_edge_with_condition(
        "categorize",
        "urgent_team",
        Condition::compare("priority", ComparisonOp::Equal, "critical")
    )
    .add_edge_with_condition(
        "categorize",
        "standard_team",
        Condition::compare("priority", ComparisonOp::Equal, "normal")
    )
    .add_edge_with_condition(
        "categorize",
        "feedback_team",
        Condition::compare("priority", ComparisonOp::Equal, "low")
    )
    
    // All paths converge
    .add_edge("urgent_team", "send_confirmation")
    .add_edge("standard_team", "send_confirmation")
    .add_edge("feedback_team", "send_confirmation")
    
    .add_edge("send_confirmation", "END")
    .build()?;
```

### Example 2: Data Validation Pipeline

Validate input and route based on errors:

```rust
let graph = GraphBuilder::new()
    .add_node("validate_schema", validate_schema_handler)
    .add_node("validate_business_rules", validate_rules_handler)
    .add_node("normalize", normalize_handler)
    .add_node("send_error", error_handler)
    .add_node("process_data", process_handler)
    
    .add_edge("START", "validate_schema")
    
    // If schema invalid, send error
    .add_edge_with_condition(
        "validate_schema",
        "send_error",
        Condition::compare("valid", ComparisonOp::Equal, false)
    )
    
    // If schema valid, check business rules
    .add_edge_with_condition(
        "validate_schema",
        "validate_business_rules",
        Condition::compare("valid", ComparisonOp::Equal, true)
    )
    
    // If business rules invalid, send error
    .add_edge_with_condition(
        "validate_business_rules",
        "send_error",
        Condition::compare("valid", ComparisonOp::Equal, false)
    )
    
    // If business rules valid, normalize
    .add_edge_with_condition(
        "validate_business_rules",
        "normalize",
        Condition::compare("valid", ComparisonOp::Equal, true)
    )
    
    // Both error and success paths go to END
    .add_edge("send_error", "END")
    .add_edge("normalize", "process_data")
    .add_edge("process_data", "END")
    .build()?;
```

### Example 3: Branching Based on User Type

```rust
let graph = GraphBuilder::new()
    .add_node("identify_user", identify_handler)
    
    // Different flows for different user types
    .add_node("admin_dashboard", admin_handler)
    .add_node("user_dashboard", user_handler)
    .add_node("guest_dashboard", guest_handler)
    
    .add_node("finalize", finalize_handler)
    
    .add_edge("START", "identify_user")
    
    // Route by user type
    .add_edge_with_condition(
        "identify_user",
        "admin_dashboard",
        Condition::compare("role", ComparisonOp::Equal, "admin")
    )
    .add_edge_with_condition(
        "identify_user",
        "user_dashboard",
        Condition::compare("role", ComparisonOp::Equal, "user")
    )
    .add_edge_with_condition(
        "identify_user",
        "guest_dashboard",
        Condition::compare("role", ComparisonOp::Equal, "guest")
    )
    
    // All paths converge
    .add_edge("admin_dashboard", "finalize")
    .add_edge("user_dashboard", "finalize")
    .add_edge("guest_dashboard", "finalize")
    
    .add_edge("finalize", "END")
    .build()?;
```

---

## Using GraphBuilder with Config Files

You can mix both approaches:

### Programmatic (GraphBuilder)

```rust
let graph = GraphBuilder::new()
    .add_node("step1", handler1)
    .add_node("step2", handler2)
    .add_edge("START", "step1")
    .add_edge("step1", "step2")
    .add_edge("step2", "END")
    .build()?;

let agent = Agent::new(graph)?;
agent.run().await?;
```

### Config File (YAML)

Create `config.yaml`:

```yaml
graph:
  nodes:
    - name: "step1"
      handler: "handler1"
    - name: "step2"
      handler: "handler2"
  
  edges:
    - from: START
      to: step1
    - from: step1
      to: step2
    - from: step2
      to: END
```

Then load:

```rust
let agent = from_config_path("config.yaml")?;
agent.run().await?;
```

---

## Config File Conditional Routing

In `config.yaml`, use conditions in edges:

```yaml
graph:
  nodes:
    - name: "validate"
      handler: "validate_input"
    - name: "process_simple"
      handler: "process_simple"
    - name: "process_complex"
      handler: "process_complex"
    - name: "output"
      handler: "format_output"
  
  edges:
    - from: START
      to: validate
    
    # Conditional routing based on complexity
    - from: validate
      to: process_simple
      condition:
        field: "complexity"
        operator: "<"
        value: 5
    
    - from: validate
      to: process_complex
      condition:
        field: "complexity"
        operator: ">="
        value: 5
    
    - from: process_simple
      to: output
    
    - from: process_complex
      to: output
    
    - from: output
      to: END
```

---

## Best Practices

### ✅ Do's

1. **Name nodes clearly** - `validate_input` not `step1`
   ```rust
   .add_node("validate_input", validate_handler)   // ✅ Good
   .add_node("s1", handler)                          // ❌ Bad
   ```

2. **Make conditions clear and testable**
   ```rust
   // ✅ Clear and maintainable
   Condition::compare("status", ComparisonOp::Equal, "approved")
   
   // ❌ Complex and hard to debug
   Condition::and(vec![...multiple nested conditions...])
   ```

3. **Always route all paths to END**
   ```rust
   // ✅ All paths lead to END
   .add_edge("handler1", "END")
   .add_edge("handler2", "END")
   
   // ❌ Missing path - graph is incomplete
   .add_edge("handler1", "END")
   // handler2 doesn't lead anywhere!
   ```

4. **Test each path independently**
   ```rust
   #[tokio::test]
   async fn test_valid_path() {
       let mut state = State::new();
       state.set("complexity", json!(3));
       // Should take simple path
   }
   
   #[tokio::test]
   async fn test_complex_path() {
       let mut state = State::new();
       state.set("complexity", json!(10));
       // Should take complex path
   }
   ```

### ❌ Don'ts

1. **Don't forget to add edges**
   ```rust
   let graph = GraphBuilder::new()
       .add_node("step1", handler1)
       // Missing: add_edge("START", "step1")
       // Missing: add_edge("step1", "END")
       .build()?;  // Will fail!
   ```

2. **Don't create disconnected nodes**
   ```rust
   // ❌ "orphaned_node" is unreachable
   .add_node("main_flow", handler1)
   .add_node("orphaned_node", handler2)
   .add_edge("START", "main_flow")
   .add_edge("main_flow", "END")
   ```

3. **Don't use vague condition names**
   ```rust
   // ❌ What does "flag" mean?
   Condition::compare("flag", ComparisonOp::Equal, true)
   
   // ✅ Clear
   Condition::compare("needs_approval", ComparisonOp::Equal, true)
   ```

4. **Don't create circular flows without explicit intent**
   ```rust
   // ❌ Accidental infinite loop
   .add_edge("handler1", "handler2")
   .add_edge("handler2", "handler1")  // Oops!
   
   // ✅ Explicit retry logic
   .add_edge_with_condition(
       "handler1",
       "handler1",  // Intentional retry
       Condition::compare("retry_needed", ComparisonOp::Equal, true)
   )
   ```

---

## Debugging Graphs

### Inspect Graph Structure

```rust
let graph = GraphBuilder::new()
    .add_node("step1", handler1)
    .add_node("step2", handler2)
    .add_edge("START", "step1")
    .add_edge("step1", "step2")
    .add_edge("step2", "END")
    .build()?;

// Print nodes
for node in graph.nodes() {
    println!("Node: {}", node.name);
}

// Print edges
for edge in graph.edges() {
    println!("Edge: {} → {}", edge.from, edge.to);
}
```

### Print Execution Path

```rust
#[register_handler]
pub async fn debug_handler(mut state: State) -> Result<State> {
    println!("📍 Current node: {}", state.get("current_node")?);
    println!("State: {:#?}", state);
    Ok(state)
}

// Add debug handler at key points
let graph = GraphBuilder::new()
    .add_node("step1", handler1)
    .add_node("debug1", debug_handler)
    .add_node("step2", handler2)
    .add_edge("START", "step1")
    .add_edge("step1", "debug1")
    .add_edge("debug1", "step2")
    .add_edge("step2", "END")
    .build()?;
```

---

## Common Patterns

### Pattern: Multi-way Routing

Routes to 3+ different paths based on conditions:

```rust
.add_edge_with_condition(
    "categorize",
    "path_a",
    Condition::compare("type", ComparisonOp::Equal, "A")
)
.add_edge_with_condition(
    "categorize",
    "path_b",
    Condition::compare("type", ComparisonOp::Equal, "B")
)
.add_edge_with_condition(
    "categorize",
    "path_c",
    Condition::compare("type", ComparisonOp::Equal, "C")
)
```

### Pattern: Diamond (Split and Merge)

One node → many nodes → one node:

```rust
.add_edge("validate", "path_a")
.add_edge("validate", "path_b")
.add_edge("path_a", "merge")
.add_edge("path_b", "merge")
```

### Pattern: Sequential with Error Exit

Normal path + error path:

```rust
.add_edge_with_condition(
    "process",
    "output",
    Condition::compare("success", ComparisonOp::Equal, true)
)
.add_edge_with_condition(
    "process",
    "error_handler",
    Condition::compare("success", ComparisonOp::Equal, false)
)
.add_edge("output", "END")
.add_edge("error_handler", "END")
```

---

## Summary

| Feature | Example |
|---------|---------|
| **Add node** | `.add_node("name", handler)` |
| **Add edge** | `.add_edge("from", "to")` |
| **Conditional edge** | `.add_edge_with_condition("from", "to", condition)` |
| **Compare values** | `Condition::compare("field", Op::Equal, value)` |
| **Combine conditions** | `Condition::and(vec![...])` or `Condition::or(vec![...])` |
| **Field exists** | `Condition::field_exists("field")` |
| **Custom condition** | `Condition::custom("name", predicate_fn)` |

Now you can build sophisticated, dynamic workflows with conditional routing! Start with simple graphs and gradually add complexity.
