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

**Under the hood:**
```
Turn 1: "My favorite color is blue"
  ↓
Add to memory buffer
  ↓
Memory: ["My favorite color is blue"]

Turn 2: "What did I say?"
  ↓
Include memory in LLM prompt:
"User previously said: My favorite color is blue"
  ↓
LLM can reference earlier context
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

**Real example: Document processing**
```
Checkpoint 1: Processed documents 1-20
Checkpoint 2: Processed documents 21-40
[✓ Safe to resume from here]
Checkpoint 3: Processed documents 41-60
[✓ Safe to resume from here]
```

### Choosing the Right Strategy

```rust
// Use conversation memory for chat
let chat_agent = AgentBuilder::new(AgentType::Conversational)
    .with_memory_steps(20)
    .build()?;

// Use checkpointing for long workflows
let workflow_agent = AgentBuilder::new(AgentType::ZeroShotReAct)
    .build()?;
// Save checkpoints after each phase
```

---

## 🎓 Auto-Evaluation & Self-Correction

### How It Works

1. Agent generates answer
2. System scores the answer (relevance, completeness, etc.)
3. LLM grades quality
4. If score < threshold, agent retries with feedback
5. Returns when confident

### Setting Up Evaluation

```rust
// Build agent with evaluation
let agent = AgentBuilder::new(AgentType::ZeroShotReAct)
    .with_llm_config("gpt-4")
    // Evaluation settings go here
    .build()?;

// With evaluation enabled:
// - Every answer auto-evaluated
// - Retries automatically if quality low
// - No code changes needed
```

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

### Real-World Examples

**Email Generation (High Completeness)**
```yaml
evaluation:
  scoring:
    metrics: [completeness, professionalism, clarity]
    weights: [0.5, 0.3, 0.2]
  min_confidence: 0.9  # Require nearly perfect emails
```

**Customer Search (High Relevance)**
```yaml
evaluation:
  scoring:
    metrics: [relevance, recency]
    weights: [0.8, 0.2]
  min_confidence: 0.85
```

**Content Analysis (High Accuracy)**
```yaml
evaluation:
  scoring:
    metrics: [accuracy, completeness, nuance]
    weights: [0.6, 0.2, 0.2]
  min_confidence: 0.9  # Factual accuracy critical
```

---

## 🧠 Dynamic Planning

### How Planner Works

Instead of "always go Node A then Node B", ask LLM what to do:

```yaml
# Traditional (hardcoded):
START → check_power → check_network → END

# Dynamic (planner-based):
START → [LLM decides] → various nodes → [LLM decides] → [more nodes] → END
```

### Implementing Custom Planner Logic

```rust
// With planner enabled, LLM sees:
// 1. Current state
// 2. Available next nodes
// 3. Previous findings

// Then decides: "Based on what we know, the next step should be..."
```

### Planner Configuration

```yaml
graph:
  planner:
    enabled: true
    max_plan_steps: 10  # Max times to replan
    prompt_template: |
      You are a troubleshooting expert.
      Given: {current_state}
      Available actions: {available_nodes}
      
      What should we do next? (reply with node name only)
```

### Real-World Troubleshooting Example

```
System malfunction detected
  ↓
Planner: "Let me check the basics first" → check_power
  ↓
Finding: Power OK
  ↓
Planner: "Power's fine, check connectivity" → check_network
  ↓
Finding: Network down! (CAUSE FOUND)
  ↓
Planner: "Found it! Restart network" → restart_network
  ↓
Resolution
```

---

## 🔧 Custom Tools

### Define a Tool

```rust
let calculator = ToolSpec::new("math", "Calculate expressions")
    .with_parameter("expression", "string")
    .required("expression");

let web_search = ToolSpec::new("search", "Search the web")
    .with_parameter("query", "string")
    .required("query")
    .with_parameter("max_results", "number");

let translator = ToolSpec::new("translate", "Translate text")
    .with_parameter("text", "string")
    .with_parameter("language", "string")
    .required("text")
    .required("language");
```

### Add to Agent

```rust
let agent = AgentBuilder::new(AgentType::ZeroShotReAct)
    .with_tool(calculator)
    .with_tool(web_search)
    .with_tool(translator)
    .build()?;
```

### Tool Best Practices

1. **Clear names** - `calculator` not `calc`, `web_search` not `search`
2. **Good descriptions** - "Calculate mathematical expressions" not "Do math"
3. **Required fields** - Mark which parameters are required
4. **Sensible defaults** - When possible, provide defaults
5. **One job** - Each tool does one thing well

---

## 🚨 Error Handling Patterns

### Pattern 1: Graceful Degradation

```rust
let agent = AgentBuilder::new(AgentType::ZeroShotReAct)
    .with_llm_config("gpt-4")
    .with_tool(primary_tool)      // Primary tool
    .with_tool(fallback_tool)     // Fallback if primary fails
    .build()?;

// Agent automatically tries tools in order
```

### Pattern 2: Retry with Backoff

```rust
let agent = AgentBuilder::new(AgentType::ZeroShotReAct)
    .with_llm_config("gpt-4")
    .build()?;

// With evaluation enabled:
// - First attempt fails
// - Auto-retry with backoff
// - LLM learns from failure
```

### Pattern 3: Fallback Providers

```rust
llm:
  provider: openai
  model: gpt-4
  fallbacks:
    - provider: anthropic
      model: claude-3-opus
    - provider: mistral
      model: mistral-large
```

**Flow:**
```
Try OpenAI
  ↓ [fails]
Try Anthropic
  ↓ [fails]
Try Mistral
  ↓ [success]
Return result
```

---

## 🧪 Testing Strategies

### Test 1: Agent Responds

```rust
#[tokio::test]
async fn test_agent_responds() {
    let agent = AgentBuilder::new(AgentType::ZeroShotReAct)
        .with_llm_config("gpt-4")
        .build()
        .expect("Failed to build agent");

    let mut state = State::new();
    let mut agent = agent;
    agent.initialize(&mut state).expect("Failed to init");

    let response = agent.process("Hello", &state)
        .expect("Failed to process");

    assert!(!response.is_empty(), "Agent returned empty response");
}
```

### Test 2: Memory Works

```rust
#[tokio::test]
async fn test_memory_works() {
    let agent = AgentBuilder::new(AgentType::Conversational)
        .with_memory_steps(10)
        .build()
        .expect("Failed to build");

    let mut state = State::new();
    let mut agent = agent;
    agent.initialize(&mut state).expect("Failed to init");

    // First turn
    agent.process("My name is Alice", &state)
        .expect("Turn 1 failed");

    // Second turn - should remember Alice
    let response = agent.process("What's my name?", &state)
        .expect("Turn 2 failed");

    assert!(response.to_lowercase().contains("alice"), 
            "Agent didn't remember name");
}
```

### Test 3: Tool Integration

```rust
#[tokio::test]
async fn test_agent_uses_tools() {
    let tool = ToolSpec::new("test_tool", "A test tool");
    
    let agent = AgentBuilder::new(AgentType::ZeroShotReAct)
        .with_tool(tool)
        .build()
        .expect("Failed to build");

    let tools = agent.tools();
    assert_eq!(tools.len(), 1, "Tool not added");
    assert_eq!(tools[0].name, "test_tool", "Wrong tool added");
}
```

---

## 📊 Debugging Tips

### Enable Logging

```rust
// During development
let agent = AgentBuilder::new(AgentType::ZeroShotReAct)
    .with_llm_config("gpt-4")
    .build()?;

// Check memory state
println!("Memory: {:?}", agent.config().memory_enabled);

// Check tools
for tool in agent.tools() {
    println!("Tool: {}", tool.name);
}
```

### Inspect State

```rust
// Print full state for debugging
println!("State: {:#?}", state);

// Check specific field
if let Some(value) = state.get("input") {
    println!("Input: {}", value);
}
```

### Trace Execution

```
[Agent Init] Name: "my_agent", Type: "zero-shot-react"
[LLM Config] Provider: "openai", Model: "gpt-4"
[Tools] 2 tools registered
[Memory] Enabled: false
[Process] Input: "What is Rust?"
[LLM Call] Generating response...
[Response] "Rust is a systems programming language..."
```

---

## 🏆 Best Practices

### ✅ Do's

- ✅ Use predefined agents for common patterns
- ✅ Enable memory for multi-turn interactions
- ✅ Use evaluation to improve quality
- ✅ Provide clear tool descriptions
- ✅ Test memory and evaluation separately
- ✅ Log important state transitions
- ✅ Use checkpoints in long workflows

### ❌ Don'ts

- ❌ Don't create agents without LLM config
- ❌ Don't create too many tools (< 10 recommended)
- ❌ Don't forget to initialize agents
- ❌ Don't assume memory without enabling it
- ❌ Don't set evaluation threshold too high (>0.95)
- ❌ Don't ignore error messages

---

## 📖 Next Steps

- [See all features explained](FEATURES.md)
- [Configuration reference](CONFIG_FEATURES.md)
- [Example code](examples/)

---

## Table of Contents

1. [Handler Development](#handler-development)
2. [State Management](#state-management)
3. [Conditions & Routing](#conditions--routing)
4. [Custom Middleware](#custom-middleware)
5. [Error Handling](#error-handling)
6. [Testing Agents](#testing-agents)
7. [Performance Optimization](#performance-optimization)
8. [Debugging](#debugging)

---

## Handler Development

### Handler Signature

All handlers follow this async signature:

```rust
pub async fn my_handler(mut state: State) -> Result<State> {
    // Process state
    Ok(state)
}
```

**Requirements:**
- Function must be `pub` and `async`
- Input: `State` (mutable)
````
