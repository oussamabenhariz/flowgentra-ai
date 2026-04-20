# ToolCalling

Uses the LLM provider's **native function-calling API** (`chat_with_tools`) instead of text-based `<action>` tags. Tool calls are returned as structured `ToolCall` objects in the response, eliminating the need for action-tag parsing.

Supported providers: OpenAI, Anthropic, Mistral, Groq, and any OpenAI-compatible endpoint.

## How it works

```
START
  └─► agent node ──(has native tool_calls)──► tool_executor node ──┐
        ▲                                                            │
        └────────────────────────────────────────────────────────────┘
        │(no tool_calls in response)
        ▼
       END
```

1. `agent` node calls `ToolCallingNode`.
2. `ToolCallingNode` converts `ToolSpec` entries to `ToolDefinition` objects and calls `llm.chat_with_tools(messages, tools)`.
3. If the LLM response contains native `tool_calls`, they are stored in state as `pending_tool_calls` and `needs_tool` is set to `true`.
4. `tool_calling_router` reads `needs_tool`: routes to `tool_executor` if true, else to `END`.
5. `tool_executor` node iterates over `pending_tool_calls`, calls the user executor function for each, and appends results as tool messages back into the conversation.
6. The loop continues until the LLM responds with plain text (no tool calls).
7. `execute_input` returns `state["llm_response"]`.

## Key difference from ZeroShotReAct

| Aspect | ZeroShotReAct | ToolCalling |
|--------|---------------|-------------|
| Tool invocation format | Text `<action>` tags | Native API `tool_calls` objects |
| Parsing | String search + regex | Structured deserialization |
| Provider requirement | Any | Must support function calling |
| Multi-tool per turn | Sequential only | Provider-dependent |

## State keys

| Key | Set by | Meaning |
|-----|--------|---------|
| `input` | caller | user question |
| `llm_response` | agent node | final LLM text after all tool calls |
| `needs_tool` | agent node | native tool calls detected |
| `pending_tool_calls` | agent node | list of tool call objects |
| `tool_results` | tool_executor | results mapped by call ID |
| `__agent_type` | initialize() | `"tool-calling"` |
| `__agent_tools_count` | initialize() | number of registered tools |

## Python usage

```python
from flowgentra_ai.agent import ToolCalling, ToolSpec
from flowgentra_ai.llm import LLM

llm = LLM(provider="openai", model="gpt-4o")

weather = ToolSpec("get_weather", "Get current weather for a city")
weather.add_parameter("city", "string")
weather.set_required("city")

def my_executor(tool_name: str, args: str) -> str:
    if tool_name == "get_weather":
        return f"Weather in {args}: 22°C, sunny"
    return "Unknown tool"

agent = ToolCalling(
    name="weather-agent",
    llm=llm,
    tools=[weather],
)

# Tool executor must be wired via AgentConfig.tool_executor in Rust;
# Python agents register it through the ToolSpec + executor pattern.
result = agent.execute_input("What's the weather in Paris?")
print(result)
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | str | required | Agent identifier |
| `llm` | LLM | required | LLM with function-calling support |
| `system_prompt` | str | None | Override default instructions |
| `tools` | list[ToolSpec] | [] | Tools exposed to the provider |
| `retries` | int | 3 | Max LLM retries |
| `memory_steps` | int | None | Conversation window size |

## When to use

- Provider natively supports function calling (OpenAI, Anthropic, Mistral, Groq)
- You want structured, reliable tool dispatch without text parsing
- Multiple tools may need to be called in a single turn
- You need tool arguments as typed objects rather than raw strings

## Notes

- If the provider does not support `chat_with_tools`, the call will fail at the HTTP level. Use `ZeroShotReAct` for providers without function calling.
- `ToolDefinition` is generated from `ToolSpec` automatically — no JSON schema authoring required.
