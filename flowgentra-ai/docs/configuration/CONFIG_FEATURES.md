````markdown
# Config Guide: Setting Up Memory, Evaluation & More

A practical guide to configuring all FlowgentraAI features.

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

### Checkpointing (Save/Resume)

```yaml
memory:
  checkpointer:
    enabled: true
    checkpoint_dir: "./checkpoints"
    storage_type: "file"  # or "redis", "database"
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

---

## Planner Configuration

### What It Does

Instead of hardcoded "next step", ask the LLM to decide.

```yaml
graph:
  planner:
    enabled: true
    max_plan_steps: 5
```

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
```

---

## Configuration Tips

1. Use environment variables for secrets
2. Enable memory for better performance
3. Use evaluation for quality results
4. Use planner for complex workflows

---

## Troubleshooting

**Q: Agent is slow?**
Enable checkpointing to resume faster.

**Q: Agent returns low-quality answers?**
Enable evaluation for auto-correction.

**Q: Workflow branching is hardcoded?**
Use planner for dynamic routing.

````
