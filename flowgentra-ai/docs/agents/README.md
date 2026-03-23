# Predefined Agents Guide

Use ready-made agent types for common patterns instead of building from scratch.

## Three Agent Types

### 1. ZeroShotReAct - General Reasoning

Best for thinking through problems and using tools as needed.

```rust
let agent = AgentBuilder::new(AgentType::ZeroShotReAct)
    .with_name("researcher")
    .with_tool(search_tool)
    .build()?;

let response = agent.process("Analyze this market trend", &state)?;
```

How it works:
1. Takes your question or task
2. Thinks through the problem
3. Decides what tools to use
4. Uses them and reasons with the results
5. Gives a final answer

Use when:
- You need general problem-solving
- The agent should decide what steps to take
- You have tools available (search, calculator, etc.)

### 2. FewShotReAct - Learning from Examples

Learns from examples you show it, then applies the pattern.

```rust
let agent = AgentBuilder::new(AgentType::FewShotReAct)
    .with_name("classifier")
    .build()?;

// Show examples
agent.add_example(
    "urgent bug report",
    "Priority: HIGH - escalate_to_specialist"
);

agent.add_example(
    "feature request",
    "Priority: LOW - add_to_roadmap"
);

// Now classify new items like the examples
let response = agent.process("User reports app crashes", &state)?;
```

How it works:
1. You provide examples of input → output pairs
2. Agent learns the pattern from examples
3. Applies that pattern to new inputs
4. Classifies or generates like the examples

Use when:
- You have specific patterns to match
- Classification or structured responses needed
- Examples exist for training the agent

### 3. Conversational - Chat with Memory

Remembers what was said in previous turns.

```rust
let agent = AgentBuilder::new(AgentType::Conversational)
    .with_name("support_bot")
    .with_memory_steps(20)  // Remember last 20 messages
    .build()?;

// Turn 1
agent.process("Hi, my app is crashing", &state)?;
// Agent stores this in memory

// Turn 2 - agent remembers the context
agent.process("How can you help?", &state)?;
// Agent knows you were talking about the crash
```

How it works:
1. Stores all messages in conversation history
2. Includes recent history in every prompt
3. Remembers context across turns
4. Maintains specific memory window (e.g., last 20 messages)

Use when:
- Building chatbots or assistants
- Multi-turn conversations needed
- Context from previous messages matters

## Choosing Your Agent

| Need | Agent | Why |
|------|-------|-----|
| General problem-solving | ZeroShotReAct | Flexible, uses tools smartly |
| Classification | FewShotReAct | Learns patterns from examples |
| Chat or support | Conversational | Remembers conversation history |
| Long workflows | ZeroShotReAct | Can make complex decisions |

## Building Agents

### Basic Setup

Every agent needs at minimum:

```rust
let agent = AgentBuilder::new(AgentType::ZeroShotReAct)
    .with_name("my_agent")
    .build()?;
```

### Add Tools

```rust
let agent = AgentBuilder::new(AgentType::ZeroShotReAct)
    .with_name("researcher")
    .with_tool(search_tool)
    .with_tool(math_tool)
    .build()?;
```

### Add LLM Config

```rust
let agent = AgentBuilder::new(AgentType::Conversational)
    .with_llm_config("gpt-4")
    .build()?;

// Or use environment variable
let agent = AgentBuilder::new(AgentType::ZeroShotReAct)
    .with_llm_provider("openai")
    .build()?;
```

### Customize Behavior

```rust
let agent = AgentBuilder::new(AgentType::ZeroShotReAct)
    .with_temperature(0.7)
    .with_max_tokens(2000)
    .with_timeout_secs(30)
    .build()?;
```

## Memory Settings

For Conversational agents specifically:

```rust
let agent = AgentBuilder::new(AgentType::Conversational)
    .with_memory_steps(20)      // Remember last 20 messages
    .with_memory_tokens(10000)  // Or limit by token count
    .build()?;
```

## Common Patterns

### Research Agent

```rust
let agent = AgentBuilder::new(AgentType::ZeroShotReAct)
    .with_name("researcher")
    .with_tool(web_search)
    .with_tool(data_fetch)
    .with_temperature(0.7)
    .build()?;

agent.process("Research the latest AI trends and summarize", &state)?;
```

### Support Chatbot

```rust
let agent = AgentBuilder::new(AgentType::Conversational)
    .with_name("support_bot")
    .with_memory_steps(50)
    .with_tool(knowledge_base)
    .build()?;

// Handles multiple turns with context
```

### Document Classifier

```rust
let agent = AgentBuilder::new(AgentType::FewShotReAct)
    .with_name("classifier")
    .build()?;

// Add training examples
for (text, label) in training_data {
    agent.add_example(text, label);
}

// Classify new documents
```

## When to Go Custom

Predefined agents work well for most cases, but consider building a custom graph when:

- You need very specific workflow steps
- The order of operations is fixed
- Multiple handlers need specific sequencing
- Complex branching logic

See [handlers/README.md](../handlers/README.md) to build custom handlers.

## Best Practices

1. Start simple with a predefined agent
2. Add tools gradually to test behavior
3. Tune temperature for your use case
4. Monitor token usage for costs
5. Use memory for continuity in chat scenarios

---

See [configuration/CONFIG_GUIDE.md](../configuration/CONFIG_GUIDE.md) for complete reference.
