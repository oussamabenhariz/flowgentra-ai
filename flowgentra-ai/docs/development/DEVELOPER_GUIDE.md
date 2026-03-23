````markdown
# FlowgentraAI Developer Guide - Build Advanced Agents

Go beyond the basics. Learn patterns for building sophisticated agents.

## 🎯 What You'll Learn

- How memory and checkpointing actually work
- Building evaluation criteria for agent outputs
- Creating custom tools
- Dynamic planning patterns
- Advanced error handling

---

## 📚 Table of Contents

1. [Memory & Checkpointing](#memory--checkpointing)
2. [Auto-Evaluation & Self-Correction](#auto-evaluation--self-correction)
3. [Dynamic Planning](#dynamic-planning)
4. [Custom Tools](#custom-tools)
5. [Error Handling Patterns](#error-handling-patterns)
6. [Testing Strategies](#testing-strategies)

---

## 💾 Memory & Checkpointing

### How Conversation Memory Works

Agents remember what was said:

```rust
// Enable memory with buffer window
let agent = AgentBuilder::new(AgentType::Conversational)
    .with_memory_steps(10)  // Keep last 10 messages
    .build()?;

// Turn 1
agent.process("My favorite color is blue", &state)?;

// Turn 2
agent.process("What did I say?", &state)?;
// Automatically remembers "blue" from turn 1
```

### How Checkpointing Works

Save state between steps. Resume anytime:

```rust
// Create checkpoint after important work
let checkpoint = agent.save_checkpoint("processing_batch_1")?;

// Later, if crash happens, resume:
agent.load_checkpoint("processing_batch_1")?;
agent.process("Continue from where we left off", &state)?;
```

---

## 🎓 Auto-Evaluation & Self-Correction

### How It Works

1. Agent generates answer
2. System scores the answer (relevance, completeness, etc.)
3. LLM grades quality
4. If score < threshold, agent retries with feedback
5. Returns when confident

### Custom Evaluation Criteria

```yaml
evaluation:
  enabled: true
  
  # What to measure
  scoring:
    metrics: [relevance, completeness, accuracy]
    weights: [0.5, 0.3, 0.2]  # Relevance most important
  
  # Quality thresholds
  min_confidence: 0.85
  max_retries: 3
```

---

## 🧠 Dynamic Planning

### How Planner Works

Instead of "always go Node A then Node B", ask LLM what to do:

```yaml
graph:
  planner:
    enabled: true
    max_plan_steps: 10
    prompt_template: |
      You are a troubleshooting expert.
      Given: {current_state}
      Available actions: {available_nodes}
      
      What should we do next?
```

---

## 🔧 Custom Tools

### Define a Tool

```rust
let calculator = ToolSpec::new("math", "Calculate expressions")
    .with_parameter("expression", "string")
    .required("expression");
```

### Add to Agent

```rust
let agent = AgentBuilder::new(AgentType::ZeroShotReAct)
    .with_tool(calculator)
    .build()?;
```

---

## 🚨 Error Handling Patterns

### Pattern 1: Graceful Degradation

```rust
let agent = AgentBuilder::new(AgentType::ZeroShotReAct)
    .with_llm_config("gpt-4")
    .with_tool(primary_tool)
    .with_tool(fallback_tool)
    .build()?;
```

### Pattern 2: Fallback Providers

```yaml
llm:
  provider: openai
  model: gpt-4
  fallbacks:
    - provider: anthropic
      model: claude-3-opus
    - provider: mistral
      model: mistral-large
```

---

## 🧪 Testing Strategies

### Test 1: Agent Responds

```rust
#[tokio::test]
async fn test_agent_responds() {
    let agent = AgentBuilder::new(AgentType::ZeroShotReAct)
        .with_llm_config("gpt-4")
        .build()?;

    let response = agent.process("Hello", &state)?;
    assert!(!response.is_empty());
}
```

---

## 🏆 Best Practices

### ✅ Do's

- ✅ Use predefined agents for common patterns
- ✅ Enable memory for multi-turn interactions
- ✅ Use evaluation to improve quality
- ✅ Test memory and evaluation separately
- ✅ Use checkpoints in long workflows

### ❌ Don'ts

- ❌ Don't create agents without LLM config
- ❌ Don't create too many tools (< 10 recommended)
- ❌ Don't forget to initialize agents
- ❌ Don't set evaluation threshold too high (>0.95)

---

## 📖 Next Steps

- [See all features explained](../FEATURES.md)
- [Configuration reference](../configuration/CONFIG_GUIDE.md)
- [Example code](../../examples/)

````
