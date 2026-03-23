# Memory Guide

Remember conversations and save workflow progress so your agent can resume where it left off.

## Two Types of Memory

### Conversation Memory
Remember what was said in previous turns - perfect for chatbots and multi-turn interactions.

### Checkpointing
Save workflow progress and pick up where you left off - great for long-running tasks.

## Configuration Examples

### Chat Agent with Memory

```yaml
memory:
  conversation:
    enabled: true
    buffer_window: 20       # Remember last 20 messages
```

```rust
// User can have multi-turn conversations
agent.process("Hi, I'm Alice", &state)?;
agent.process("Tell me a joke", &state)?;
agent.process("Remember my name?", &state)?;  // Yes!
```

### Workflow with Checkpointing

```yaml
memory:
  checkpointer:
    enabled: true
    checkpoint_dir: "./data/checkpoints"
    auto_save: true
    save_interval: 10
```

```rust
// Workflow auto-saves progress
for batch in batches {
    agent.process(format!("Process: {}", batch), &state)?;
    // Checkpoint saved after each batch
}
```

### Both Enabled

```yaml
memory:
  conversation:
    enabled: true
    buffer_window: 20
  checkpointer:
    enabled: true
    checkpoint_dir: "./checkpoints"
```

## Storage Types

### File (Default)
- Simple, no dependencies
- Creates checkpoint files on disk
- Good for development

### Redis
- Fast, in-memory
- Distributed across machines
- Best for production

### Database
- Persistent
- Via external database
- Enterprise-ready

## Making a Choice

1. Chat interactions: Use conversation memory
2. Long workflows: Use checkpointing
3. Both: Enable both features
4. Production: Use Redis or database
5. Development: File storage is fine

## Memory Limits

```yaml
memory:
  conversation:
    # Option 1: Remember last N messages
    buffer_window: 20
    
    # Option 2: Limit total tokens
    max_tokens: 10000
```

## Debugging Memory

```rust
// Check what's in memory
let memory = agent.get_memory()?;
println!("Previous messages: {}", memory.messages.len());
println!("Total tokens: {}", memory.token_count);
```

---

See [configuration/CONFIG_GUIDE.md](../configuration/CONFIG_GUIDE.md) for complete reference.
