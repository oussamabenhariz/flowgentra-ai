# StructuredChat

ReAct agent that communicates tool calls as **JSON blobs** instead of free-text `<action>` tags. The LLM is instructed to always output a JSON object with `"action"` and `"action_input"` keys. The final answer is signaled by `"action": "Final Answer"`.

Equivalent to LangChain's `structured-chat-zero-shot-react-description` agent.

## How it works

```
START
  └─► agent node ──(needs_tool=true)──► tool_executor node ──┐
        ▲                                                      │
        └──────────────────────────────────────────────────────┘
        │(needs_tool=false)
        ▼
       END
```

The graph structure mirrors `ZeroShotReAct`. The difference is in the node implementation:

1. `agent` node calls `StructuredChatNode`.
2. `StructuredChatNode` sends the system prompt (which specifies the JSON blob format) + tools + input to the LLM.
3. The LLM response is parsed for a JSON blob:
   - If `action` ≠ `"Final Answer"` → `needs_tool = true`, tool name and input stored in state.
   - If `action` = `"Final Answer"` → `structured_final_answer` is set, `needs_tool = false`.
4. `reasoning_router` routes to `tool_executor` or `END`.
5. `tool_executor` runs the tool and loops back to `agent`.
6. `execute_input` returns `state["structured_final_answer"]`, falling back to `state["llm_response"]`.

## JSON action format

```json
{
  "action": "tool_name",
  "action_input": "argument string or object"
}
```

Final answer:

```json
{
  "action": "Final Answer",
  "action_input": "The answer is 42"
}
```

## System prompt structure

The default prompt embeds `{tools}` as a placeholder (filled at runtime) and shows the full Thought/Action/Observation cycle with examples:

```
Question: input question
Thought: consider previous and subsequent steps
Action:
```json
{"action": $TOOL_NAME, "action_input": $INPUT}
```
Observation: action result
... (repeat)
Thought: I know what to respond
Action:
```json
{"action": "Final Answer", "action_input": "Final response"}
```
```

## State keys

| Key | Set by | Meaning |
|-----|--------|---------|
| `input` | caller | user question |
| `llm_response` | agent node | raw LLM output text |
| `needs_tool` | agent node | JSON action detected (not Final Answer) |
| `pending_tool_name` | agent node | action value from JSON |
| `pending_tool_args` | agent node | action_input value from JSON |
| `tool_result` | tool_executor | executor return value |
| `structured_final_answer` | agent node | extracted Final Answer content |
| `__agent_type` | initialize() | `"structured-chat-zero-shot-react"` |

## Python usage

```python
from flowgentra_ai.agent import StructuredChat, ToolSpec
from flowgentra_ai.llm import LLM

llm = LLM(provider="openai", model="gpt-4o")

search = ToolSpec("web_search", "Search the web")
search.add_parameter("query", "string")
search.set_required("query")

agent = StructuredChat(
    name="structured-researcher",
    llm=llm,
    tools=[search],
)

result = agent.execute_input("What is the population of Tokyo?")
print(result)
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | str | required | Agent identifier |
| `llm` | LLM | required | Language model |
| `system_prompt` | str | None | Override default JSON-format prompt |
| `tools` | list[ToolSpec] | [] | Available tools |
| `retries` | int | 3 | Max LLM retries |
| `memory_steps` | int | None | Conversation window size |

## When to use vs ZeroShotReAct

| Scenario | Prefer |
|----------|--------|
| LLM reliably outputs `<action>` text tags | `ZeroShotReAct` |
| LLM is better at JSON output | `StructuredChat` |
| Tool arguments are complex objects | `StructuredChat` |
| Simpler prompt overhead | `ZeroShotReAct` |
| Provider supports native tool calling | `ToolCalling` (best option) |

## Notes

- The `{tools}` placeholder in the default system prompt is filled at node execution time with the formatted tool list.
- If the LLM outputs malformed JSON, parsing falls back to returning the raw `llm_response`.
