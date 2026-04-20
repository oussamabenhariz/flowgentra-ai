# Conversational

Multi-turn dialogue agent with persistent conversation history. Unlike the ReAct agents, it does not loop through a Thought/Action/Observation cycle — it makes a single LLM call per turn and appends the exchange to an in-memory history buffer.

## How it works

```
START
  └─► conversation node ──► END
```

The graph is a single node with no conditional routing. Each call to `execute_input` appends the user message and assistant response to `GraphBasedAgent.conversation_history`, which is injected into state on the next call.

1. `conversation` node calls `ConversationalNode`.
2. `ConversationalNode` builds the message list: system prompt → conversation history (as alternating user/assistant messages) → current user input.
3. The full message list is sent to `llm.chat()`.
4. The response content is stored in state under `"response"`.
5. `execute_input` extracts `state["response"]` as the return value.
6. The `(user, response)` pair is appended to `conversation_history` for the next turn.

## Memory trimming

`ConversationalAgent` (internal struct) uses a `VecDeque<Message>` with a sliding window. When the history exceeds `memory_steps` messages, the oldest messages are dropped from the front:

```rust
while self.history.len() > self.config.memory_steps {
    self.history.pop_front();
}
```

Memory is **enabled by default** for `Conversational` agents with a window of 10 messages.

## State keys

| Key | Set by | Meaning |
|-----|--------|---------|
| `input` | caller | current user message |
| `conversation_history` | GraphBasedAgent | JSON array of `{role, content}` pairs |
| `response` | conversation node | LLM reply text |
| `__agent_name` | initialize() | agent name |
| `__agent_type` | initialize() | `"conversational"` |
| `__agent_memory_steps` | initialize() | configured window size |

## Python usage

```python
from flowgentra_ai.agent import Conversational
from flowgentra_ai.llm import LLM

llm = LLM(provider="openai", model="gpt-4o")

agent = Conversational(
    name="chat-bot",
    llm=llm,
    system_prompt="You are a friendly customer support agent for Acme Corp.",
    memory_steps=20,
)

# Multi-turn conversation — history is maintained automatically
print(agent.execute_input("Hello, I need help with my order."))
print(agent.execute_input("It was order number 12345."))
print(agent.execute_input("When will it arrive?"))
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | str | required | Agent identifier |
| `llm` | LLM | required | Language model |
| `system_prompt` | str | None | Persona / instructions override |
| `tools` | list[ToolSpec] | [] | Tools (visible in prompt but not auto-called) |
| `retries` | int | 3 | Max LLM retries |
| `memory_steps` | int | None | History window (default 10 when None) |

## When to use

- Chatbots and assistants with multi-turn context
- Customer support flows
- Any scenario requiring the agent to remember earlier parts of the conversation
- When you do NOT need the agent to call external tools autonomously

## Notes

- Tools listed in `tools` are included in the system prompt description but the `Conversational` graph has no `tool_executor` node — tool calls require manual handling or switching to `ToolCalling`.
- History survives across multiple `execute_input` calls on the same agent instance.
- To reset the conversation, create a new agent instance.
