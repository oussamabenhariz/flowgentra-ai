# Config Guide: Setting Up Memory, Evaluation & More

A practical guide to configuring all ErenFlowAI features.

## TL;DR - Common Configurations

### I want my agent to remember conversations
```yaml
memory:
  conversation:
    enabled: true
    buffer_window: 10  # Remember last 10 messages
```

### I want auto-correction (retry on low quality)
```yaml
evaluation:
  enabled: true
  min_confidence: 0.8  # Retry if < 80% confident
  max_retries: 3
```

### I want the agent to decide what to do next
```yaml
graph:
  planner:
    enabled: true
    max_plan_steps: 5
```

### I want to save and resume workflows
```yaml
memory:
  checkpointer:
    enabled: true
    checkpoint_dir: "./checkpoints"
```

---

## Memory Configuration

### What It Does

Your agent can remember:
1. **Conversation history** - What was said before
2. **Execution checkpoints** - Save progress and resume

### Conversation Memory (Chat History)

```yaml
memory:
  conversation:
    enabled: true
    buffer_window: 10  # Keep last 10 messages
```

**Why?** Multi-turn conversations need context.

**Example:**
```
User: "What's 2 + 2?"
Agent: "4"
User: "Double that"
Agent: [remembers 4, returns 8] ✅
```

### Checkpointing (Save/Resume)

```yaml
memory:
  checkpointer:
    enabled: true
    checkpoint_dir: "./checkpoints"
    storage_type: "file"  # or "redis", "database"
```

**Why?** Resume long workflows if they fail.

**Example:** Processing 100 documents
```
Documents 1-20: Checkpoint ✓
Documents 21-40: Checkpoint ✓
Documents 41-60: Checkpoint ✓
[crash happens]
Resume from checkpoint at #41 ✓
```

### Full Memory Configuration

```yaml
memory:
  # Conversation memory (chat history)
  conversation:
    enabled: true
    buffer_window: 20        # Remember last 20 messages
    max_tokens: 10000       # Or limit by tokens
    strategy: "buffer"      # "buffer" or "summary"
  
  # Checkpointing (save/resume state)
  checkpointer:
    enabled: true
    checkpoint_dir: "./checkpoints"
    storage_type: "file"    # "file", "redis", "database"
    auto_save: true         # Save after each step
    save_interval: 5        # Save every 5 steps
```

---

## Evaluation Configuration

### What It Does

Your agent automatically:
1. Scores its output
2. Grades quality with LLM
3. Fixes low-quality answers
4. Retries until confident

### Simple Setup

```yaml
evaluation:
  enabled: true
  min_confidence: 0.8  # Retry if < 80%
  max_retries: 3
```

**What happens:**
1. Agent generates answer (score 0.6)
2. "Too low! Try again..." 
3. Agent improves answer (score 0.85)
4. "Good enough!" → Return

### Full Configuration

```yaml
evaluation:
  enabled: true
  
  # When to retry
  min_confidence: 0.8
  max_retries: 3
  
  # What to score
  scoring:
    metrics: [relevance, completeness, accuracy]
    weights: [0.4, 0.3, 0.3]   # Sum should be 1.0
  
  # How to grade
  grading:
    enabled: true
    rubric: "Is the answer correct and helpful?"
    grade_on: "output"
  
  # Options for retries
  retry_policy: "exponential"  # or "linear", "fixed"
  retry_delay_ms: 500
```

### Scoring Metrics

Choose what to evaluate:

| Metric | Meaning | Good For |
|--------|---------|----------|
| `relevance` | Does answer match the question? | Q&A tasks |
| `completeness` | Is it thorough and complete? | Research tasks |
| `accuracy` | Is it factually correct? | Knowledge-based tasks |
| `clarity` | Is it well-written? | content generation |
| `safety` | Is it safe and appropriate? | All public-facing tasks |

---

## Planner Configuration

### What It Does

Instead of hardcoded "next step", ask the LLM to decide.

**Why?** Handle complex workflows that branch unpredictably.

### Simple Setup

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
    max_plan_steps: 5        # Max replanning iterations
    prompt_template: |       # Custom prompt for LLM
      You are a planning expert.
      Given the current state, decide the next node.
      Available nodes: {nodes}
      Current state: {state}
      Next node:
```

### How Planner Works

```
Flow with hardcoded steps (BAD):
START → check_power → check_connection → END
(What if power is fine but connection fails?)

Flow with planner (GOOD):
START → planner(LLM decides) → check_power → planner → check_connection → planner → END
(LLM adapts to actual situation)
```

### Real Example: Troubleshooting

```yaml
graph:
  nodes:
    - name: check_power
      handler: handlers::check_power
    - name: check_connection
      handler: handlers::check_connection
    - name: check_software
      handler: handlers::check_software
    - name: planner
      handler: "builtin::planner"
  
  planner:
    enabled: true
    max_plan_steps: 5
    prompt_template: |
      You are diagnosing a laptop issue.
      
      Recent findings: {state}
      
      What should you check next?
      - check_power (is it powered on?)
      - check_connection (network working?)
      - check_software (updates needed?)
      
      Respond with ONLY the node name.
```

**How it runs:**
1. "Device won't turn on..."
2. LLM: "check_power"
3. Power: OK
4. LLM: "check_connection"
5. Network: Down!
6. "Found it - reconnect to network"

---

## LLM Configuration

### Multiple Providers

```yaml
llm:
  provider: openai
  model: gpt-4
  api_key: ${OPENAI_API_KEY}
  temperature: 0.7
  max_tokens: 2000
```

**Supported Providers:**
- `openai` - GPT-4, GPT-3.5
- `anthropic` - Claude
- `mistral` - Mistral models
- `groq` - Fast LLM
- `ollama` - Local models
- `azure` - Azure OpenAI

### Fallback Chains

```yaml
llm:
  provider: openai
  model: gpt-4
  fallbacks:
    - provider: anthropic
      model: claude-opus
    - provider: mistral
      model: mistral-large
```

**How it works:**
1. Try OpenAI GPT-4
2. If fails, try Claude
3. If fails, try Mistral
4. If all fail, error

---

## Tools Configuration

### Local Tools

```yaml
graph:
  nodes:
    - name: my_node
      tools:
        - calculator
        - web_search
```

### Remote Tools (MCP)

```yaml
graph:
  mcps:
    web_search:
      type: sse
      url: "http://api.local:8000"
      timeout: 30
      auth:
        type: "bearer"
        token: ${SEARCH_API_KEY}
    
    email:
      type: stdio
      command: "/usr/bin/email-handler"
      timeout: 15
```

---

## Complete Example Config

```yaml
name: "intelligent_agent"
description: "An agent with all features enabled"

# LLM
llm:
  provider: openai
  model: gpt-4
  api_key: ${OPENAI_API_KEY}
  temperature: 0.7
  fallbacks:
    - provider: anthropic
      model: claude-opus

# Memory & Checkpointing
memory:
  conversation:
    enabled: true
    buffer_window: 20
  checkpointer:
    enabled: true
    checkpoint_dir: "./checkpoints"

# Evaluation (auto-correction)
evaluation:
  enabled: true
  min_confidence: 0.8
  max_retries: 3

# Tools
graph:
  # Planner (dynamic routing)
  planner:
    enabled: true
    max_plan_steps: 5
  
  # Nodes
  nodes:
    - name: input_handler
      handler: handlers::handle_input
    - name: analyze
      handler: handlers::analyze
    - name: respond
      handler: handlers::generate_response
      uses_llm: true

  # Edges
  edges:
    - from: START
      to: input_handler
    - from: input_handler
      to: planner
    - from: planner
      to: [analyze, respond]
    - from: analyze
      to: planner
    - from: respond
      to: END

  # External services
  mcps:
    web_search:
      type: sse
      url: "http://search-api.local:8000"
      timeout: 30
    email:
      type: stdio
      command: "/usr/bin/email-handler"
      timeout: 15

# Validation
state_schema:
  input:
    type: string
    description: "User input"
  analysis:
    type: object
    description: "Analysis results"
  response:
    type: string
    description: "Final response"
```

---

## Configuration Tips

### 1. Use Environment Variables for Secrets
```yaml
api_key: ${OPENAI_API_KEY}
database_url: ${DATABASE_URL}
```

Then set before running:
```bash
export OPENAI_API_KEY="sk-..."
export DATABASE_URL="postgresql://..."
cargo run
```

### 2. Memory for Better Performance
```yaml
memory:
  checkpointer:
    enabled: true
    storage_type: "redis"  # Faster than file
```

### 3. Evaluation for Quality
```yaml
evaluation:
  enabled: true
  min_confidence: 0.85  # Higher = more retries
```

### 4. Planner for Complex Workflows
```yaml
graph:
  planner:
    enabled: true
    max_plan_steps: 10
```

---

## Troubleshooting

**Q: Agent is slow?**
A: Enable checkpointing to resume faster:
```yaml
memory:
  checkpointer:
    enabled: true
```

**Q: Agent returns low-quality answers?**
A: Enable evaluation for auto-correction:
```yaml
evaluation:
  enabled: true
  min_confidence: 0.9  # Stricter threshold
```

**Q: Workflow branching is hardcoded?**
A: Use planner for dynamic routing:
```yaml
graph:
  planner:
    enabled: true
```

---

## Next Steps

- [See full examples](examples/)
- [Learn about predefined agents](FEATURES.md)
- [Advanced patterns](DEVELOPER_GUIDE.md)
