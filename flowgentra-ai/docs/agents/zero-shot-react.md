# ZeroShotReAct

General-purpose reasoning + action agent. Solves problems without any example demonstrations by following the Thought → Action → Observation loop until it reaches a final answer.

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

1. `agent` node calls `AgentReasoningNode`, which sends the system prompt + available tools + user input to the LLM.
2. The LLM responds. If the response contains `<action>tool_name(args)</action>`, `needs_tool` is set to `true` in state and `pending_tool_name`/`pending_tool_args` are extracted.
3. `reasoning_router` reads `needs_tool`: routes to `tool_executor` if true, else to `END`.
4. `tool_executor` node calls the user-supplied executor function with `(tool_name, args)`, stores the result as `tool_result` in state, then loops back to `agent`.
5. When the LLM emits `<answer>...</answer>`, the agent extracts that content as the final response.

## State keys

| Key | Set by | Meaning |
|-----|--------|---------|
| `input` | caller | user question |
| `llm_response` | agent node | raw LLM text |
| `needs_tool` | agent node | whether a tool call was parsed |
| `pending_tool_name` | agent node | tool to call |
| `pending_tool_args` | agent node | arguments for the tool |
| `tool_result` | tool_executor node | result returned by the tool |
| `__agent_name` | initialize() | agent name |
| `__agent_type` | initialize() | `"zero-shot-react"` |
| `__agent_tools_count` | initialize() | number of registered tools |

## System prompt format

```
<action>tool_name(arguments)</action>   ← trigger a tool call
<answer>your final answer</answer>      ← terminate with answer
```

The default system prompt instructs the LLM to:
1. Analyze the problem
2. Call tools using `<action>` tags when needed
3. Wrap the final answer in `<answer>` tags

## Python usage

```python
from flowgentra_ai.agent import ZeroShotReAct, ToolSpec
from flowgentra_ai.llm import LLM

llm = LLM(provider="openai", model="gpt-4o")

search = ToolSpec("search", "Search the web for information")
search.add_parameter("query", "string")
search.set_required("query")

agent = ZeroShotReAct(
    name="researcher",
    llm=llm,
    tools=[search],
    retries=3,
)

result = agent.execute_input("What year was Python created?")
print(result)
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | str | required | Agent identifier |
| `llm` | LLM | required | Language model to use |
| `system_prompt` | str | None | Override default system prompt |
| `tools` | list[ToolSpec] | [] | Tools the agent can call |
| `retries` | int | 3 | Max LLM retry attempts on failure |
| `memory_steps` | int | None | If set, retains last N conversation turns |

## When to use

- Open-ended questions requiring dynamic tool selection
- Tasks where the solution path is unknown upfront
- Any single-turn Q&A where tools may or may not be needed

## Notes

- `FewShotReAct` uses the identical graph; the difference is only the system prompt (which includes worked examples).
- If no tool executor is registered, the agent continues but tool calls return a placeholder message.
- Conversation history is tracked in `GraphBasedAgent.conversation_history` across `execute_input` calls, providing implicit multi-turn context.
