# Dynamic Planning Guide

Let the LLM decide what to do next instead of hardcoding the workflow path.

## Problem with Hardcoded Workflows

```
Traditional approach:
START → check_power → check_network → check_software → END

What if power is fine but network is offline?
The workflow doesn't adapt!
```

## Solution: Dynamic Planning

```
Adaptive approach:
START → [LLM decides] → check_power → [LLM decides] → [more checks] → END

The LLM sees the state and decides:
"Power is fine, let me check network next"
```

## Setup

### Basic Configuration

```yaml
graph:
  planner:
    enabled: true
    max_plan_steps: 5        # Max replanning iterations
```

### Full Configuration

```yaml
graph:
  planner:
    enabled: true
    max_plan_steps: 10
    prompt_template: |
      You are a troubleshooting expert.
      
      Current state: {current_state}
      Available actions: {available_nodes}
      
      What should we do next?
      Respond with ONLY the node name.
```

## How It Works

```
1. Agent in state: {issue: "device won't start"}
2. Planner: "Let's check power first" → check_power
3. Result: {issue: "...", power: "OK"}
4. Planner: "Power's fine, check network" → check_network
5. Result: {issue: "...", power: "OK", network: "DOWN"}
6. Planner: "Found the problem!" → fix_network
7. Result: {issue: "...", power: "OK", network: "OK"} → END
```

## Configuration

### Nodes Available to Planner

```yaml
graph:
  nodes:
    - name: check_power
      handler: handlers::check_power
    
    - name: check_network
      handler: handlers::check_network
    
    - name: check_storage
      handler: handlers::check_storage
    
    - name: run_diagnostics
      handler: handlers::run_diagnostics
  
  edges:
    - from: START
      to: planner
    
    - from: planner
      to: [check_power, check_network, check_storage, run_diagnostics]
    
    - from: [check_power, check_network, check_storage, run_diagnostics]
      to: planner
    
    - from: planner
      to: END
```

## Custom Planner Prompts

```yaml
graph:
  planner:
    enabled: true
    max_plan_steps: 5
    prompt_template: |
      You are a customer support specialist.
      
      Customer issue: {current_state.customer_issue}
      Already tried: {current_state.tried}
      
      What support action should we try next?
      - escalate_to_specialist
      - send_documentation
      - schedule_callback
      - provide_workaround
      
      Choose one node:
```

## Use Cases

### Troubleshooting

```yaml
nodes:
  - name: check_hardware
  - name: check_drivers
  - name: check_software
  - name: run_diagnostics
  - name: escalate
```

### Customer Support

```yaml
nodes:
  - name: check_faq
  - name: search_knowledge_base
  - name: suggest_workaround
  - name: escalate_to_specialist
```

### Data Processing

```yaml
nodes:
  - name: validate_data
  - name: transform_data
  - name: enrich_data
  - name: store_results
  - name: notify_user
```

## Controlling Planner Behavior

### Limit Planning Steps

```yaml
planner:
  max_plan_steps: 3  # Max 3 planning decisions
```

### Control Available Options

```yaml
planner:
  available_nodes: [check_power, check_network]  # Limit choices
```

### Custom Decision Logic

```rust
pub async fn planner_wrapper(mut state: State) -> Result<State> {
    let current_state = state.get("current_state")?;
    
    // Custom logic to select next node
    let next_node = my_logic(&current_state);
    
    state.set("next_node", json!(next_node));
    Ok(state)
}
```

## Best Practices

1. Use clear, descriptive node names
2. Document what each node does
3. Don't give too many options (keeps planner focused)
4. Show progress in state so planner can see what happened
5. Always have an escalate or abort option
6. Test locally first

## Debugging Planner

```rust
// See what planner decided
let plan = agent.get_plan()?;
println!("Selected node: {}", plan.next_node);
println!("Reason: {}", plan.reasoning);
```

---

See [configuration/CONFIG_GUIDE.md](../configuration/CONFIG_GUIDE.md) for complete reference.
