# ErenFlowAI Features Guide

A practical guide to all the amazing features that make ErenFlowAI powerful.

## 🎯 Quick Feature Overview

### Core Features (You'll Use These First)

| Feature | What It Does | When to Use |
|---------|-------------|-----------|
| **Predefined Agents** | Ready-to-use agent templates (ZeroShotReAct, FewShotReAct, Conversational) | When you want to get started quickly without building custom graphs |
| **Memory & Checkpointing** | Save conversation history and agent state | For multi-turn conversations or resumable workflows |
| **Auto-Evaluation** | LLM grades outputs, retries if quality is low | When you need high-quality results from your agent |
| **Dynamic Planning** | LLM decides what to do next | For complex workflows that need flexibility |
| **Tools & MCP** | Connect external services and APIs | When you need to access databases, search engines, etc. |
| **RAG** | Semantic search over your documents | When you need context from your own knowledge base |

---

## 🤖 Predefined Agents

**What's This?**  
Instead of building a graph from scratch, ErenFlowAI provides ready-made agent types for common patterns. Think of them as templates.

**Three Built-In Types:**

### 1. ZeroShotReAct Agent
- **Best for:** General-purpose reasoning tasks
- **How it works:** "Think through this problem, use tools as needed, give answer"
- **Example use case:** Research, analysis, problem-solving

```rust
let agent = AgentBuilder::new(AgentType::ZeroShotReAct)
    .with_name("researcher")
    .with_llm_config("gpt-4")
    .with_tool(search_tool)
    .with_mcp(web_search_mcp)
    .build()?;

let response = agent.process("What are the latest AI breakthroughs?", &state)?;
```

### 2. FewShotReAct Agent
- **Best for:** Tasks where examples help
- **How it works:** Shows the LLM examples before asking it to solve
- **Example use case:** Classification, pattern-based tasks

```rust
let agent = AgentBuilder::new(AgentType::FewShotReAct)
    .with_name("classifier")
    .with_llm_config("gpt-4")
    .build()?;

// Add examples showing what you want
agent.add_example(
    "urgent bug",
    "This should go to priority support",
    "Actions: escalate_to_priority_support"
);
```

### 3. Conversational Agent
- **Best for:** Chat-like interactions with memory
- **How it works:** Remembers conversation history across turns
- **Example use case:** Chatbots, customer support, assistants

```rust
let agent = AgentBuilder::new(AgentType::Conversational)
    .with_name("support_bot")
    .with_llm_config("gpt-4")
    .with_memory_steps(20)  // Remember last 20 messages
    .build()?;

// First turn
agent.process("Hi, I have a bug", &state)?;

// Second turn - agent remembers the context
agent.process("Can you help?", &state)?;
```

---

## 💾 Memory & Checkpointing

**What's This?**  
Save your agent's state so it can:
- Resume multi-step processes
- Remember conversations across sessions
- Maintain context in long-running tasks

**Two Types:**

### 1. Checkpointing (State Persistence)

Save the full state at each step. Resume from any checkpoint.

```yaml
# In config.yaml
memory:
  checkpointer:
    enabled: true
    checkpoint_dir: "./checkpoints"
```

```rust
// Save state
let checkpoint = agent.create_checkpoint("user_123")?;

// Later, load and resume
agent.load_checkpoint("user_123")?;
agent.run(state).await?;
```

**Real-world example:** An agent analyzing a 100-page document
- Page 1-10: Creates checkpoint
- Page 11-20: Creates checkpoint (can resume here if it fails)
- Page 21-30: Creates checkpoint
- Etc.

### 2. Conversation Memory (Message History)

Remember what was said in a conversation.

```yaml
# In config.yaml
memory:
  conversation:
    enabled: true
    buffer_window: 10  # Remember last 10 messages
```

```rust
// Automatically memories messages
agent.process("What's my name?", &state)?;
agent.process("My name is Alice", &state)?;
agent.process("What did I just tell you?", &state)?;  // Remembers "Alice"
```

**Real-world example:** Customer support chatbot
- Customer: "I bought a laptop yesterday"
- Support bot creates a checkpoint
- Customer: "It doesn't turn on"
- Support bot remembers the laptop purchase from earlier
- Support bot: "Hi! I see you bought a laptop. Let me help with the power issue..."

---

## 🎓 Auto-Evaluation & Self-Correction

**What's This?**  
Your agent automatically grades its own work and retries if quality is low.

**Why it matters:** Get better results without human supervision.

### Three-Step Process

1. **Output Scoring** - Rate answer quality (0.0 to 1.0)
2. **LLM Grading** - Use another LLM to evaluate
3. **Auto-Retry** - If score < threshold, try again with corrections

```yaml
# In config.yaml
evaluation:
  enabled: true
  min_confidence: 0.8  # Retry if confidence < 80%
  max_retries: 3
  scoring:
    - metrics: [relevance, completeness]
      weights: [0.5, 0.5]
  grading:
    enabled: true
    rubric: "Is the answer accurate and complete?"
```

```rust
// This automatically happens in your nodes
pub async fn generate_answer(mut state: State) -> Result<State> {
    // Your logic generates an answer
    let answer = llm.generate("Here's my answer...").await?;
    state.set("answer", json!(answer));
    
    // Auto-evaluation middleware:
    // 1. Scores the answer
    // 2. If score is low, retries with "Improve your answer..."
    // 3. Returns only when confident or max retries reached
    
    Ok(state)
}
```

**Real-world example:** Customer email response generator
- Agent drafts response: "OK" (score: 0.3 - too short!)
- Auto-retry triggers
- Agent drafts response: "Thank you for contacting us. I see your issue..." (score: 0.9 - good!)
- Response sent

---

## 🧠 Dynamic Planning (Planner Node)

**What's This?**  
Instead of hardcoding which node runs next, ask the LLM to decide.

**Why it matters:** Handle complex, unpredictable workflows

### How It Works

```
State (current situation)
    ↓
LLM: "Based on this state, what should I do next?"
    ↓
Planner decides next node (can be anything)
    ↓
Execute that node
    ↓
Repeat (dynamic planning!)
```

### Example: Troubleshooting Agent

```yaml
# config.yaml
graph:
  planner:
    enabled: true
    max_plan_steps: 5
    prompt_template: "You are a troubleshooting expert. Decide what to test next."
  
  nodes:
    - name: power_check
      handler: handlers::check_power
    - name: connection_check
      handler: handlers::check_connection
    - name: software_check
      handler: handlers::check_software
  
  edges:
    - from: START
      to: planner
    - from: planner
      to: [power_check, connection_check, software_check]
    - from: power_check
      to: planner
    - from: connection_check
      to: planner
    - from: software_check
      to: planner
```

**Agent reasoning:**
1. "The device doesn't turn on..."
2. "First, let me check power" → power_check
3. "Power is fine. Let me check connection" → connection_check
4. "Connection is fine. Let me check software" → software_check
5. "Found the bug!  → END

**No hardcoding needed!** The LLM adapts based on findings.

---

## 🔧 Tools & MCP (Model Context Protocol)

**What's This?**  
Connect your agent to external tools, APIs, and services.

**Two types:**

### Local Tools
Quick functions your agent calls directly.

```rust
let calculator = ToolSpec::new("calc", "Do math")
    .with_parameter("expression", "string")
    .required("expression");

let agent = AgentBuilder::new(AgentType::ZeroShotReAct)
    .with_tool(calculator)
    .build()?;
```

### MCP Services (Remote)
Connect to external services via HTTP, stdio, or Docker.

```yaml
# In config.yaml
graph:
  mcps:
    web_search:
      type: sse
      url: "http://search-api.local:8000"
      timeout: 30
    
    email_service:
      type: stdio
      command: "/usr/bin/email-handler"
      timeout: 15
    
    nlp_api:
      type: docker
      image: "nlp-service:latest"
      timeout: 60
```

```rust
let agent = AgentBuilder::new(AgentType::ZeroShotReAct)
    .with_mcp(web_search_mcp)
    .with_mcp(email_service_mcp)
    .build()?;

// Agent automatically sees all available MCPs
let response = agent.process(
    "Search for Python 3.12 and email the results to me",
    &state
)?;
// Agent calls both MCPs automatically!
```

---

## 🔍 RAG (Retrieval-Augmented Generation)

**What's This?**  
Your agent searches your documents for context before answering.

**Why it matters:** Give LLM access to your company data without fine-tuning.

### Three Steps

1. **Upload** - Index your documents into vector store
2. **Search** - Find relevant documents for each query
3. **Augment** - Pass documents to LLM for better answers

```yaml
# In config.yaml
rag:
  enabled: true
  vector_store:
    type: pinecone
    index: "my-docs"
    api_key: ${PINECONE_API_KEY}
  embedding_model: openai
  retrieval:
    top_k: 5
    min_similarity: 0.7
```

```rust
pub async fn answer_with_context(mut state: State) -> Result<State> {
    let query = state.get_str("user_question")?;
    
    // Automatically retrieves relevant docs
    let docs = rag.retrieve(query, 5).await?;
    
    // Build context
    let context = docs.iter()
        .map(|d| d.content.clone())
        .collect::<Vec<_>>()
        .join("\n---\n");
    
    // LLM uses context for better answer
    let answer = llm.generate(format!(
        "Context:\n{}\n\nQuestion: {}",
        context, query
    )).await?;
    
    state.set("answer", json!(answer));
    Ok(state)
}
```

**Real-world example:** Internal knowledge base search
- Employee: "What's our PTO policy?"
- Agent retrieves policy document from RAG
- Agent answers: "According to our policy, you get 20 days..."

---

## 🏗️ Graph Compiler

**What's This?**  
Converts your YAML workflow into an optimized execution graph.

**Why it matters:** Catches errors early, optimizes execution.

### What It Does

1. **Validates** - Checks all nodes and edges exist
2. **Optimizes** - Finds parallel paths, removes redundancy
3. **Compiles** - Builds execution plan

```yaml
# config.yaml - compiler validates this
graph:
  nodes:
    - name: user_input
      handler: handlers::capture_input
    - name: analyze
      handler: handlers::analyze
    - name: generate
      handler: handlers::generate_response
  
  edges:
    - from: START
      to: user_input
    - from: user_input
      to: analyze
    - from: analyze
      to: generate
    - from: generate
      to: END
```

Compiler checks:
- ✅ All nodes referenced in edges exist
- ✅ No orphaned nodes (nodes nothing points to)
- ✅ START and END properly connected
- ✅ No circular paths (unless intentional)
- ✅ Parallel execution can be optimized

---

## 📊 State Management

**What's This?**  
Pass data through your workflow. Think of it as the conversation context.

**Three Ways to Use:**

### 1. Document Your State Schema

```yaml
# In config.yaml
state_schema:
  user_input:
    type: string
    description: "What the user asked"
  analysis:
    type: object
    description: "Results of analysis"
  response:
    type: string
    description: "Final answer"
```

### 2. Access in Handlers

```rust
pub async fn my_handler(mut state: State) -> Result<State> {
    // Read
    let input = state.get_str("user_input")?;
    let analysis = state.get("analysis")?;
    
    // Process
    let result = process(&input);
    
    // Write
    state.set("response", json!(result));
    
    Ok(state)
}
```

### 3. Validate at Runtime

State schema enables validation to catch bugs early.

```rust
// Ensure required fields exist before executing
state.validate()?;  // Checks against schema
```

---

## 📝 Complete Feature Checklist

- ✅ Predefined agents (ZeroShotReAct, FewShotReAct, Conversational)
- ✅ Memory & checkpointing
- ✅ Auto-evaluation & self-correction
- ✅ Dynamic planning
- ✅ Local tools
- ✅ MCP services (SSE, Stdio, Docker)
- ✅ RAG (vector stores)
- ✅ Graph compiler
- ✅ State management & validation
- ✅ Multi-LLM support
- ✅ Error handling & retry policies
- ✅ Middleware pipeline
- ✅ Health monitoring
- ✅ Distributed tracing

---

## 🚀 Next Steps

1. **Start simple:** Try a predefined agent first
2. **Add memory:** Enable checkpointing for resumable workflows
3. **Get smart:** Enable auto-evaluation for quality
4. **Go dynamic:** Use planner for complex workflows
5. **Connect:** Add tools and MCPs

See the other guides for implementation details!
