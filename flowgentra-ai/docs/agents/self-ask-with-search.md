# SelfAskWithSearch

Decomposes complex, multi-hop questions into a chain of simpler sub-questions, answers each with a **search** tool, and accumulates intermediate answers until it can state a final answer.

Requires exactly one tool named `"search"` (case-insensitive). All other tools are ignored.

Equivalent to LangChain's `self-ask-with-search` agent type.

## How it works

```
START
  └─► agent node ──(has follow-up)──► tool_executor node ──┐
        ▲                              (search only)         │
        └─────────────────────────────────────────────────────┘
        │("So the final answer is:")
        ▼
       END
```

1. `agent` node calls `SelfAskNode`.
2. `SelfAskNode` builds the prompt with the four built-in few-shot examples and appends `Question: <input>`.
3. The LLM responds following the pattern:
   - `Are follow up questions needed here: Yes.` → sub-question loop begins
   - `Follow up: <query>` → search query extracted
4. `self_ask_router` routes to `tool_executor` if a follow-up was detected, else to `END`.
5. `tool_executor` calls the search executor with the follow-up query, stores `Intermediate answer: <result>` in the scratchpad.
6. The loop continues until the LLM emits `So the final answer is: <answer>`.
7. `execute_input` returns `state["sa_final_answer"]`, falling back to `state["llm_response"]`.

## Output format

The LLM is expected to follow this pattern (from the built-in few-shot examples in the system prompt):

```
Question: Who lived longer, Muhammad Ali or Alan Turing?
Are follow up questions needed here: Yes.
Follow up: How old was Muhammad Ali when he died?
Intermediate answer: Muhammad Ali was 74 years old when he died.
Follow up: How old was Alan Turing when he died?
Intermediate answer: Alan Turing was 41 years old when he died.
So the final answer is: Muhammad Ali
```

## Built-in search tool stub

The default `SelfAskWithSearchAgent` pre-registers a `"search"` tool stub so users see it in `.tools()`:

```rust
ToolSpec::new("search", "Search for information to answer follow-up questions")
    .with_parameter("query", "string")
    .required("query")
```

## State keys

| Key | Set by | Meaning |
|-----|--------|---------|
| `input` | caller | original multi-hop question |
| `llm_response` | agent node | raw LLM output |
| `sa_follow_up` | agent node | extracted follow-up query |
| `needs_tool` | agent node | follow-up detected |
| `scratchpad` | initialize() / agent node | accumulated Thought/Answer trace |
| `sa_final_answer` | agent node | extracted final answer |
| `__agent_type` | initialize() | `"self-ask-with-search"` |

## Python usage

```python
from flowgentra_ai.agent import SelfAskWithSearch, ToolSpec
from flowgentra_ai.llm import LLM

llm = LLM(provider="openai", model="gpt-4o")

# The search tool must be named "search"
search = ToolSpec("search", "Search the web for facts")
search.add_parameter("query", "string")
search.set_required("query")

agent = SelfAskWithSearch(
    name="multi-hop",
    llm=llm,
    tools=[search],
)

result = agent.execute_input(
    "Are both the directors of Jaws and Casino Royale from the same country?"
)
print(result)
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | str | required | Agent identifier |
| `llm` | LLM | required | Language model |
| `system_prompt` | str | None | Override default few-shot prompt |
| `tools` | list[ToolSpec] | [] | Must include a tool named `"search"` |
| `retries` | int | 3 | Max LLM retries |
| `memory_steps` | int | None | Conversation window size |

## When to use

- Questions requiring multiple sequential lookups (multi-hop reasoning)
- Fact-verification tasks where each fact needs a separate search
- Research assistants that must cite intermediate steps

## When NOT to use

- Simple single-step questions (use `ZeroShotReAct` or `ToolCalling`)
- Tasks requiring tools other than search
- Latency-sensitive applications (each sub-question adds a round-trip)

## Notes

- The system prompt is a pure few-shot prompt (no instruction header), consisting entirely of the four worked examples. Overriding it with `system_prompt` replaces those examples.
- If no search executor is registered, sub-question answers will be placeholder strings.
