# Agents Guide

Use ready-made agent types for common patterns, or orchestrate multiple agents with a Supervisor.

## Predefined Agent Types

### ZeroShotReAct -- General Reasoning

Thinks through problems and uses tools as needed, without requiring examples.

```rust
use flowgentra_ai::core::agents::{AgentBuilder, AgentType, ToolSpec};

let search = ToolSpec::new("search", "Search the web")
    .with_parameter("query", "string")
    .required("query");

let agent = AgentBuilder::new(AgentType::ZeroShotReAct)
    .with_name("researcher")
    .with_llm_config("gpt-4")
    .with_tool(search)
    .build()?;

let response = agent.process("What are the latest AI trends?", &state)?;
```

Best for: open-ended questions, research, analysis, problem-solving.

### FewShotReAct -- Learning from Examples

Shows the LLM examples before asking it to solve new inputs.

```rust
let agent = AgentBuilder::new(AgentType::FewShotReAct)
    .with_name("classifier")
    .with_llm_config("gpt-4")
    .build()?;

agent.add_example("urgent bug report", "Priority: HIGH - escalate");
agent.add_example("feature request", "Priority: LOW - add to roadmap");

let response = agent.process("App crashes on login", &state)?;
```

Best for: classification, pattern-matching, structured responses.

### Conversational -- Chat with Memory

Remembers conversation history across turns.

```rust
let agent = AgentBuilder::new(AgentType::Conversational)
    .with_name("support_bot")
    .with_llm_config("gpt-4")
    .with_memory_steps(20)  // Remember last 20 messages
    .build()?;

agent.process("Hi, my app is crashing", &state)?;
agent.process("What version are you on?", &state)?;  // Remembers context
```

Best for: chatbots, customer support, assistants.

## Choosing the Right Agent

| Need | Agent | Why |
|------|-------|-----|
| General problem-solving | ZeroShotReAct | Flexible, tool-aware |
| Classification tasks | FewShotReAct | Learns from your examples |
| Multi-turn chat | Conversational | Remembers history |
| Complex multi-step | StateGraph | Full control over workflow |

---

## Supervisor (Multi-Agent Orchestration)

The Supervisor pattern routes tasks to specialized sub-agents based on a routing function.

### How It Works

```
User request
     |
     v
  Supervisor
     |-- Router decides which agent handles this
     v
  Agent A / Agent B / Agent C
     |
     v
  Result merged back
     |
     v
  Router decides next step (or finish)
```

### Usage

```rust
use flowgentra_ai::core::agents::Supervisor;
use std::collections::HashMap;

// Build specialized agent graphs
let researcher = build_research_graph()?;   // Good at finding info
let writer = build_writing_graph()?;        // Good at drafting text
let reviewer = build_review_graph()?;       // Good at quality checks

let mut agents = HashMap::new();
agents.insert("researcher".to_string(), researcher);
agents.insert("writer".to_string(), writer);
agents.insert("reviewer".to_string(), reviewer);

// Router function decides which agent handles each step
let router = |state: &PlainState| -> Result<String> {
    let phase = state.get("phase")
        .and_then(|v| v.as_str())
        .unwrap_or("research");
    Ok(phase.to_string())
};

let supervisor = Supervisor::new(router, agents, 10); // max 10 rounds
let result = supervisor.run(initial_state).await?;
```

### Key Parameters

| Parameter | Description |
|-----------|-------------|
| `router` | Function that inspects state and returns an agent name |
| `agents` | Map of agent name to compiled StateGraph |
| `max_rounds` | Maximum dispatch rounds to prevent infinite loops |

### When to Use Supervisor vs StateGraph

| Use Supervisor when... | Use StateGraph when... |
|----------------------|---------------------|
| Different sub-tasks need different "expert" agents | All steps share the same processing logic |
| The number of steps is dynamic | The workflow is fixed or predictable |
| Agents have different tool sets | All nodes share the same tools |

---

## Building Custom Agent Graphs

When predefined agents aren't enough, build a custom workflow with the StateGraph API:

```rust
use flowgentra_ai::core::state_graph::StateGraphBuilder;

let graph = StateGraphBuilder::new()
    .add_fn("intake", intake_handler)
    .add_fn("research", research_handler)
    .add_fn("draft", draft_handler)
    .add_fn("review", review_handler)
    .set_entry_point("intake")
    .add_edge("intake", "research")
    .add_edge("research", "draft")
    .add_conditional_edge("draft", |state| {
        let quality = state.get("quality_score")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0);
        if quality > 0.8 { Ok("__end__".into()) }
        else { Ok("review".into()) }
    })
    .add_edge("review", "draft")  // Loop back for improvement
    .compile()?;
```

---

## Best Practices

1. **Start with a predefined agent** -- only build custom graphs when needed
2. **Use Supervisor for multi-agent systems** -- don't reinvent routing
3. **Set max_rounds on Supervisor** -- prevent runaway loops
4. **Keep router logic simple** -- complex routing belongs in a dedicated classification node
5. **Monitor token usage** -- multi-agent systems can consume tokens quickly

---

See [graph/README.md](../graph/README.md) for StateGraph details.
See [llm/README.md](../llm/README.md) for LLM provider setup.
