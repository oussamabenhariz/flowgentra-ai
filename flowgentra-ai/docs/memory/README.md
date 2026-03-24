# Memory and Checkpointing Guide

Save conversation history and workflow progress so your agent can resume where it left off.

## Types of Memory

| Type | Purpose | Persistence |
|------|---------|-------------|
| **Conversation Memory** | Remember chat messages across turns | In-memory (session) |
| **TokenBufferMemory** | Sliding window by token budget | In-memory (session) |
| **SummaryMemory** | LLM-summarized conversation history | In-memory (session) |
| **Checkpointing** | Save full workflow state to disk | Durable (survives restart) |

---

## Conversation Memory

Remember what was said in previous turns -- essential for chatbots and multi-turn interactions.

```yaml
memory:
  conversation:
    enabled: true
    buffer_window: 20    # Remember last 20 messages
```

```rust
agent.process("Hi, I'm Alice", &state)?;
agent.process("What's my name?", &state)?;  // Knows "Alice"
```

### Memory Limits

```yaml
memory:
  conversation:
    buffer_window: 20       # Limit by message count
    max_tokens: 10000       # Or limit by token count
```

When the limit is reached, oldest messages are dropped first (system messages are always kept).

---

## TokenBufferMemory

A sliding window that manages memory by token count rather than message count. System messages are always preserved.

```rust
use flowgentra_ai::prelude::*;

let mut memory = TokenBufferMemory::new(4000); // 4000 token budget

memory.add_message(Message::system("You are a helpful assistant."));
memory.add_message(Message::user("Tell me about Rust."));
memory.add_message(Message::assistant("Rust is a systems programming language..."));

// When total tokens exceed budget, oldest non-system messages are dropped
let messages = memory.messages();
```

Token estimation uses ~4 characters per token as an approximation.

---

## SummaryMemory

When conversation history grows too large, older messages are summarized by an LLM to maintain context in a compact form.

```rust
use flowgentra_ai::prelude::*;

let config = SummaryConfig {
    buffer_size: 10,        // Keep last 10 messages in full
    max_summary_tokens: 500, // Max tokens for the summary
};

let memory = SummaryMemory::new(config, |messages| {
    Box::pin(async move {
        // Call your LLM here to generate a summary
        let summary = my_llm.summarize(&messages).await?;
        Ok(summary)
    })
});

memory.add_message(Message::user("Hello"));
// After buffer_size messages, older ones get summarized
let messages = memory.messages(); // Includes summary as system message + recent messages
```

---

## Checkpointing

Save the full state at each step so workflows can resume after interruptions.

### InMemoryCheckpointer

Fast, ephemeral storage. Good for development and testing.

```rust
use flowgentra_ai::core::state_graph::InMemoryCheckpointer;

let checkpointer = InMemoryCheckpointer::new();

// Save state
checkpointer.save("thread-1", &state)?;

// Load state
let restored = checkpointer.load("thread-1")?;
```

Data is lost when the process exits.

### FileCheckpointer

Persist checkpoints as JSON files on disk. Survives process restarts.

```rust
use flowgentra_ai::core::state_graph::FileCheckpointer;

let checkpointer = FileCheckpointer::new("./checkpoints");

// Save -- writes to ./checkpoints/thread-1/checkpoint.json
checkpointer.save("thread-1", &state)?;

// Load -- reads from disk
let restored = checkpointer.load("thread-1")?;
```

File structure:
```
checkpoints/
  thread-1/
    checkpoint.json
  thread-2/
    checkpoint.json
```

### Config-Based Setup

```yaml
memory:
  checkpointer:
    enabled: true
    checkpoint_dir: "./data/checkpoints"
    auto_save: true        # Save after each node
    save_interval: 10      # Or save every N steps
```

---

## Using Both Together

```yaml
memory:
  conversation:
    enabled: true
    buffer_window: 20
  checkpointer:
    enabled: true
    checkpoint_dir: "./checkpoints"
```

Conversation memory handles the chat context. Checkpointing handles the full workflow state. They serve different purposes and work well together.

---

## Checkpointing with Human-in-the-Loop

Checkpointing pairs naturally with the human-in-the-loop pattern:

```rust
let graph = StateGraphBuilder::new()
    .add_fn("draft", draft_response)
    .add_fn("send", send_response)
    .set_entry_point("draft")
    .interrupt_before("send")
    .add_edge("draft", "send")
    .add_edge("send", "__end__")
    .compile()?;

// Run until interrupt -- state is checkpointed
let partial = graph.run(state).await?;

// Human reviews, edits, approves...

// Resume from checkpoint with modifications
let result = graph.resume_with_state("thread-1", edited_state).await?;
```

---

## Best Practices

1. **Use InMemoryCheckpointer for tests** -- fast, no cleanup needed
2. **Use FileCheckpointer for production** -- durable, survives restarts
3. **Set reasonable buffer_window** -- 20 messages is a good default for chat
4. **Clean up old checkpoints** -- they accumulate on disk over time
5. **Use thread IDs** -- each conversation or workflow run gets its own thread ID

---

See [graph/README.md](../graph/README.md) for human-in-the-loop details.
See [FEATURES.md](../FEATURES.md) for the complete feature list.
