# Dynamic Planning Guide

Let the LLM decide what step to execute next instead of hardcoding the workflow path.

## The Problem

Traditional workflows are rigid:

```
START -> check_power -> check_network -> check_software -> END
```

What if power is fine but the problem is the network? The workflow runs unnecessary steps and can't adapt.

## The Solution

With dynamic planning, the LLM inspects the current state and decides what to do next:

```
START -> [LLM picks] -> check_power -> [LLM picks] -> check_network -> [LLM picks] -> END
```

The LLM sees "power is fine" and skips ahead to "check network." When it finds the issue, it goes straight to fixing it.

---

## Setup

### Basic

```yaml
graph:
  planner:
    enabled: true
    max_plan_steps: 5
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

      What should we do next? Respond with ONLY the node name.
```

## How It Works

1. Agent enters the planner with current state: `{issue: "device won't start"}`
2. LLM decides: "Let's check power first" -> routes to `check_power`
3. After `check_power`: `{issue: "...", power: "OK"}`
4. LLM decides: "Power is fine, check network" -> routes to `check_network`
5. After `check_network`: `{issue: "...", power: "OK", network: "DOWN"}`
6. LLM decides: "Found it! Fix the network" -> routes to `fix_network`
7. Done.

---

## Example: Support Agent

```yaml
graph:
  planner:
    enabled: true
    max_plan_steps: 5

  nodes:
    - name: check_faq
      handler: handlers::check_faq
    - name: search_kb
      handler: handlers::search_knowledge_base
    - name: suggest_fix
      handler: handlers::suggest_workaround
    - name: escalate
      handler: handlers::escalate_to_specialist

  edges:
    - from: START
      to: planner
    - from: planner
      to: [check_faq, search_kb, suggest_fix, escalate]
    - from: [check_faq, search_kb, suggest_fix, escalate]
      to: planner
    - from: planner
      to: END
```

The planner can try the FAQ first, then search the knowledge base, suggest a fix, or escalate -- adapting to each situation.

---

## Combining with StateGraph

You can also implement planning logic directly in a StateGraph using conditional edges:

```rust
let graph = StateGraphBuilder::new()
    .add_fn("plan", plan_next_step)
    .add_fn("check_power", check_power)
    .add_fn("check_network", check_network)
    .add_fn("fix", apply_fix)
    .set_entry_point("plan")
    .add_conditional_edge("plan", |state| {
        Ok(state.get("next_action")
            .and_then(|v| v.as_str())
            .unwrap_or("__end__")
            .to_string())
    })
    .add_edge("check_power", "plan")   // Loop back to planner
    .add_edge("check_network", "plan")
    .add_edge("fix", "__end__")
    .compile()?;
```

For async planning decisions (e.g., calling an LLM to decide), use `add_async_conditional_edge`.

---

## Best Practices

1. **Use descriptive node names** -- the LLM reads them to decide routing
2. **Limit choices** -- don't give the planner too many options (5-7 max)
3. **Set max_plan_steps** -- prevents infinite loops
4. **Include an escape hatch** -- always have an "escalate" or "finish" option
5. **Show progress in state** -- the planner needs to see what's already been tried
6. **Test with deterministic inputs** -- verify the planner picks sensible routes

---

See [graph/README.md](../graph/README.md) for conditional edge details.
See [FEATURES.md](../FEATURES.md) for the complete feature list.
