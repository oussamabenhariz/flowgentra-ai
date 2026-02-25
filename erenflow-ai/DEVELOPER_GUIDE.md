# ErenFlowAI Developer Guide - Build Advanced Agents

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
- Output: `Result<State>`
- Must not take other parameters
- Should not store references (to support distributed execution)

### Example Handlers

#### Simple Processing

```rust
pub async fn extract_fields(mut state: State) -> Result<State> {
    // Get input
    let text = state.get("input")
        .and_then(|v| v.as_str())
        .ok_or_else(|| ErenFlowAIError::StateError("Missing input".into()))?;
    
    // Process
    let words: Vec<&str> = text.split_whitespace().collect();
    
    // Update state
    state.set("word_count", json!(words.len()));
    state.set("words", json!(words));
    
    Ok(state)
}
```

#### With Validation

```rust
pub async fn validate_input(mut state: State) -> Result<State> {
    let input = state.get("input")
        .and_then(|v| v.as_str())
        .ok_or(ErenFlowAIError::StateError("No input".into()))?;
    
    // Validate
    if input.is_empty() {
        return Err(ErenFlowAIError::StateError("Input cannot be empty".into()));
    }
    
    if input.len() > 10000 {
        return Err(ErenFlowAIError::StateError("Input too long".into()));
    }
    
    state.set("input_valid", json!(true));
    Ok(state)
}
```

#### With LLM Integration

```rust
pub async fn llm_handler(mut state: State) -> Result<State> {
    let input = state.get("input")
        .and_then(|v| v.as_str())
        .unwrap_or("default");
    
    // Create message for LLM
    let message = Message {
        role: "user".to_string(),
        content: input.to_string(),
    };
    
    // Call LLM (in real code, use the LLM client from context)
    let response = "Generated response".to_string();
    
    state.set("response", json!(response));
    Ok(state)
}
```

#### With Error Recovery

```rust
pub async fn resilient_handler(mut state: State) -> Result<State> {
    match fetch_data().await {
        Ok(data) => {
            state.set("data", json!(data));
            Ok(state)
        }
        Err(e) => {
            // Log error but don't fail
            eprintln!("Failed to fetch: {}", e);
            
            // Set fallback
            state.set("data", json!({"fallback": true}));
            Ok(state)
        }
    }
}

async fn fetch_data() -> Result<serde_json::Value> {
    // Could fail
    Ok(json!({}))
}
```

#### Conditional State Updates

```rust
pub async fn conditional_update(mut state: State) -> Result<State> {
    if let Some(value) = state.get("flag") {
        if value.as_bool().unwrap_or(false) {
            state.set("result", json!("flag enabled"));
        } else {
            state.set("result", json!("flag disabled"));
        }
    }
    
    Ok(state)
}
```

---

## State Management

### Understanding State

State is a wrapper around `serde_json::Value`. It flows through your graph:

```
START → Handler1 → Handler2 → Handler3 → END
  ↓         ↓         ↓         ↓       ↓
State0 → State1 → State2 → State3 → State4
```

Each handler receives the previous handler's output state.

### State Operations

#### Get Values

```rust
// Simple get
let value = state.get("key");

// Get with type conversion
let string = state.get("key").and_then(|v| v.as_str());
let number = state.get("key").and_then(|v| v.as_i64());
let array = state.get("key").and_then(|v| v.as_array());
let object = state.get("key").and_then(|v| v.as_object());
```

#### Set Values

```rust
// Simple set
state.set("key", json!("value"));

// JSON convenience
state.set("count", json!(42));
state.set("items", json!(vec!["a", "b", "c"]));
state.set("data", json!({
    "name": "John",
    "age": 30
}));
```

#### Mutable Access

```rust
// Get mutable reference
if let Some(obj) = state.get_mut("data").and_then(|v| v.as_object_mut()) {
    obj.insert("modified".to_string(), json!(true));
}
```

#### Merge States

```rust
pub async fn merge_results(mut state: State) -> Result<State> {
    let results1 = state.get("search_results_a")
        .unwrap_or(json!([]));
    let results2 = state.get("search_results_b")
        .unwrap_or(json!([]));
    
    let mut merged = Vec::new();
    
    if let Some(arr1) = results1.as_array() {
        merged.extend(arr1.clone());
    }
    if let Some(arr2) = results2.as_array() {
        merged.extend(arr2.clone());
    }
    
    state.set("merged_results", json!(merged));
    Ok(state)
}
```

### State Schema

Document expected state in config.yaml:

```yaml
state_schema:
  input:
    type: string
    description: "User query"
  documents:
    type: Array<Document>
    description: "Retrieved documents"
  processed:
    type: boolean
    description: "Whether processed"
  score:
    type: number
    description: "Analysis score"
  metadata:
    type: object
    description: "Additional metadata"
```

### State Validation

Validate state in handlers:

```rust
pub async fn safe_handler(mut state: State) -> Result<State> {
    // Check required fields exist
    if state.get("input").is_none() {
        return Err(ErenFlowAIError::StateError(
            "Missing required field: input".into()
        ));
    }
    
    // Validate types
    if let Some(val) = state.get("count") {
        if !val.is_number() {
            return Err(ErenFlowAIError::StateError(
                "count must be a number".into()
            ));
        }
    }
    
    Ok(state)
}
```

---

## Conditions & Routing

### Writing Conditions

Conditions are functions that take `&State` and return `bool`:

```rust
pub fn is_simple_query(state: &State) -> bool {
    state.get("input")
        .and_then(|v| v.as_str())
        .map(|s| s.len() < 50)  // Simple if less than 50 chars
        .unwrap_or(false)
}

pub fn is_complex_query(state: &State) -> bool {
    !is_simple_query(state)
}

pub fn has_required_fields(state: &State) -> bool {
    state.get("input").is_some()
        && state.get("context").is_some()
}

pub fn should_retry(state: &State) -> bool {
    state.get("attempt_count")
        .and_then(|v| v.as_i64())
        .map(|count| count < 3)
        .unwrap_or(true)
}
```

### Using Conditions in Config

```yaml
edges:
  - from: analyze
    to: complex_branch
    condition: is_complex_query

  - from: analyze
    to: simple_branch
    condition: is_simple_query

  - from: error_branch
    to: retry_handler
    condition: should_retry

  - from: error_branch
    to: fallback_handler
    condition: "!should_retry"  # NOT condition
```

### Complex Conditions

```rust
pub fn should_use_rag(state: &State) -> bool {
    // Check multiple factors
    let has_query = state.get("input").is_some();
    let not_cached = state.get("cached_response").is_none();
    let needs_context = state.get("needs_context")
        .and_then(|v| v.as_bool())
        .unwrap_or(true);
    
    has_query && not_cached && needs_context
}

pub fn can_parallelize(state: &State) -> bool {
    state.get("tasks")
        .and_then(|v| v.as_array())
        .map(|arr| arr.len() > 1)
        .unwrap_or(false)
}
```

---

## Custom Middleware

Middleware intercepts requests and responses. While ErenFlowAI provides built-in middleware (logging, caching, rate-limiting), you can extend with custom middleware.

### Built-in Middleware

```yaml
middleware:
  - name: logging
    enabled: true
    level: debug
    include_state: true

  - name: rate_limiting
    enabled: true
    rpm: 60
    burst_size: 10

  - name: cache
    enabled: true
    ttl: 3600
    strategy: content_hash
```

### Custom Middleware Pattern

Create a custom middleware handler:

```rust
pub async fn timing_middleware(mut state: State) -> Result<State> {
    let start = std::time::Instant::now();
    
    // Simulate work
    tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
    
    let elapsed = start.elapsed();
    state.set("_timing", json!({
        "elapsed_ms": elapsed.as_millis()
    }));
    
    Ok(state)
}

pub async fn audit_middleware(mut state: State) -> Result<State> {
    // Log all state changes
    if let Ok(state_str) = serde_json::to_string(&state) {
        println!("[AUDIT] State: {}", state_str);
    }
    
    Ok(state)
}
```

---

## Error Handling

### Error Types

ErenFlowAI provides `ErenFlowAIError`:

```rust
use erenflow_ai::prelude::*;

// Various error types
ErenFlowAIError::ConfigError("Invalid config".into())
ErenFlowAIError::StateError("State error".into())
ErenFlowAIError::HandlerError("Handler failed".into())
ErenFlowAIError::LLMError("LLM failure".into())
ErenFlowAIError::RAGError("Vector store error".into())
ErenFlowAIError::MCPError("Tool error".into())
```

### Error Propagation

```rust
pub async fn with_error_handling(mut state: State) -> Result<State> {
    // Convert Result types
    let data = fetch_data()
        .map_err(|e| ErenFlowAIError::HandlerError(e.to_string()))?;
    
    state.set("data", json!(data));
    Ok(state)
}

async fn fetch_data() -> std::result::Result<String, Box<dyn std::error::Error>> {
    Ok("data".to_string())
}
```

### Recovery Patterns

```rust
pub async fn with_fallback(mut state: State) -> Result<State> {
    match primary_operation(&mut state).await {
        Ok(state) => Ok(state),
        Err(e) => {
            eprintln!("Primary failed: {}, trying fallback", e);
            fallback_operation(state).await
        }
    }
}

pub async fn primary_operation(state: &mut State) -> Result<State> {
    state.set("method", json!("primary"));
    Ok(state.clone())
}

pub async fn fallback_operation(mut state: State) -> Result<State> {
    state.set("method", json!("fallback"));
    Ok(state)
}

impl Clone for State {
    fn clone(&self) -> Self {
        // State cloning logic
        State::new()
    }
}
```

### Context-Aware Errors

```rust
pub async fn contextual_error(mut state: State) -> Result<State> {
    let input = state.get("input")
        .and_then(|v| v.as_str())
        .unwrap_or("unknown");
    
    Err(ErenFlowAIError::HandlerError(
        format!("Failed processing input: '{}'", input)
    ))
}
```

---

## Testing Agents

### Unit Testing Handlers

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_simple_handler() {
        let mut state = State::new();
        state.set("input", json!("test"));

        let result = extract_fields(state).await;
        
        assert!(result.is_ok());
        let state = result.unwrap();
        assert_eq!(state.get("word_count"), Some(json!(1)));
    }

    #[tokio::test]
    async fn test_error_handling() {
        let state = State::new();
        
        let result = validate_input(state).await;
        assert!(result.is_err());
    }
}
```

### Integration Testing

```rust
#[cfg(test)]
mod integration_tests {
    use super::*;

    #[tokio::test]
    async fn test_agent_workflow() {
        let mut agent = Agent::from_config_with_handlers(
            "test_config.yaml",
            &handlers,
        ).expect("Failed to load config");

        let mut state = State::new();
        state.set("input", json!("test query"));

        let result = agent.run(state).await;
        assert!(result.is_ok());

        let final_state = result.unwrap();
        assert!(final_state.get("response").is_some());
    }
}
```

### Mock Handlers

```rust
#[cfg(test)]
mod mock_handlers {
    use super::*;

    pub async fn mock_llm_handler(mut state: State) -> Result<State> {
        // Mock LLM response
        state.set("response", json!("mocked response"));
        Ok(state)
    }

    pub async fn mock_rag_handler(mut state: State) -> Result<State> {
        // Mock RAG retrieval
        state.set("docs", json!(vec!["doc1", "doc2"]));
        Ok(state)
    }
}
```

---

## Performance Optimization

### Async Best Practices

```rust
// ✅ Good: Use tokio for concurrent operations
pub async fn parallel_fetch(mut state: State) -> Result<State> {
    let futures = vec![
        fetch_source_a(),
        fetch_source_b(),
        fetch_source_c(),
    ];
    
    let results = futures::future::join_all(futures).await;
    state.set("results", json!(results));
    Ok(state)
}

async fn fetch_source_a() -> String { "a".into() }
async fn fetch_source_b() -> String { "b".into() }
async fn fetch_source_c() -> String { "c".into() }
```

### Memory Efficiency

```rust
// ✅ Good: Stream large data
pub async fn process_large_file(mut state: State) -> Result<State> {
    let chunks = vec!["chunk1", "chunk2", "chunk3"];
    
    let processed: Vec<_> = chunks.iter()
        .map(|chunk| process_chunk(chunk))
        .collect();
    
    state.set("processed", json!(processed));
    Ok(state)
}

fn process_chunk(chunk: &str) -> String {
    format!("processed: {}", chunk)
}
```

### Caching

```rust
pub async fn cached_operation(mut state: State) -> Result<State> {
    let key = state.get("query")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    
    // Check cache
    if let Some(cached) = get_cache(key) {
        state.set("result", cached);
        return Ok(state);
    }
    
    // Compute
    let result = expensive_operation(key).await?;
    set_cache(key, &result);
    
    state.set("result", result);
    Ok(state)
}

fn get_cache(key: &str) -> Option<serde_json::Value> {
    None // Implement actual caching
}

fn set_cache(key: &str, value: &serde_json::Value) {
    // Implement actual caching
}

async fn expensive_operation(key: &str) -> Result<serde_json::Value> {
    Ok(json!({"result": key}))
}
```

---

## Debugging

### Enable Debug Logging

```rust
// In main.rs
use tracing_subscriber;

#[tokio::main]
async fn main() -> Result<()> {
    // Enable debug logging
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::DEBUG)
        .init();

    // ... rest of code
    Ok(())
}
```

### Log State Changes

```rust
pub async fn debug_handler(mut state: State) -> Result<State> {
    println!("[DEBUG] State before: {:?}", state);
    
    // Do work
    state.set("processed", json!(true));
    
    println!("[DEBUG] State after: {:?}", state);
    
    Ok(state)
}
```

### Trace Execution

```python
# Use observability config in YAML
observability:
  tracing_enabled: true
  trace_level: debug
  exporters:
    - type: jaeger
      endpoint: "http://localhost:14268/api/traces"
```

### Inspect State at Breakpoints

Use conditional logging to inspect specific scenarios:

```rust
pub async fn conditional_debug(mut state: State) -> Result<State> {
    if let Some(error_flag) = state.get("_debug") {
        if error_flag.as_bool().unwrap_or(false) {
            eprintln!("[STATE DUMP] {:#?}", state);
        }
    }
    
    Ok(state)
}
```

---

## Advanced Patterns

### Chain of Responsibility

```rust
pub async fn handler_chain(mut state: State) -> Result<State> {
    // Chain multiple operations
    state = validate(state).await?;
    state = enrich(state).await?;
    state = transform(state).await?;
    Ok(state)
}

async fn validate(mut state: State) -> Result<State> {
    state.set("validated", json!(true));
    Ok(state)
}

async fn enrich(mut state: State) -> Result<State> {
    state.set("enriched", json!(true));
    Ok(state)
}

async fn transform(mut state: State) -> Result<State> {
    state.set("transformed", json!(true));
    Ok(state)
}
```

### Fan-out/Fan-in Pattern

```rust
pub async fn map_reduce(mut state: State) -> Result<State> {
    // Map: split work
    if let Some(items) = state.get("items").and_then(|v| v.as_array()) {
        let results: Vec<_> = futures::future::join_all(
            items.iter().map(|item| process_async(item.clone()))
        ).await;
        
        // Reduce: combine results
        state.set("results", json!(results));
    }
    
    Ok(state)
}

async fn process_async(item: serde_json::Value) -> String {
    "processed".into()
}
```

---

## Best Practices

✅ **Do:**
- Keep handlers small and focused
- Use descriptive names for state fields
- Document handler behavior in comments
- Test handlers with various inputs
- Use type hints in state_schema
- Handle errors gracefully
- Log important state transitions

❌ **Don't:**
- Store references in state (not serializable)
- Mix async/sync code without care
- Ignore error results
- Create unbounded loops
- Store secrets in state
- Make handlers dependent on execution order
- Use unwrap() without fallbacks

---

**Happy developing! 🚀**
