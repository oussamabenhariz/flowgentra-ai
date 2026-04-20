# FewShotReAct

Same Thought → Action → Observation loop as `ZeroShotReAct`, but the default system prompt includes four worked examples that demonstrate the expected reasoning pattern. This gives the LLM stronger guidance on format and style.

## How it works

The graph structure is **identical** to `ZeroShotReAct`:

```
START
  └─► agent node ──(needs_tool=true)──► tool_executor node ──┐
        ▲                                                      │
        └──────────────────────────────────────────────────────┘
        │(needs_tool=false)
        ▼
       END
```

`builders.rs` wires `FewShotReAct` through `build_few_shot_react_graph`, which currently delegates to `build_zero_shot_react_graph`. The only behavioral difference is the system prompt.

## System prompt format

The default `FEW_SHOT_REACT` prompt walks the LLM through a five-step process:

```
Step 1: Think  — Analyze what's being asked
Step 2: Act    — Call tools with <action>tool_name(arguments)</action>
Step 3: Observe — Review the result
Step 4: Refine — Decide if more info is needed
Step 5: Answer — Output <answer>final answer</answer>
```

It also embeds an explicit example pattern so the LLM has a concrete template to follow.

## ExampleDemonstration (Rust internals)

The `FewShotReActAgent` struct (in `few_shot_react.rs`) holds a `Vec<ExampleDemonstration>`:

```rust
pub struct ExampleDemonstration {
    pub input: String,
    pub thought: String,
    pub action: String,
    pub observation: String,
    pub output: String,
}
```

These are formatted and injected into the prompt as:

```
EXAMPLES:

Example 1:
Input: What is 2+2?
Thought: I need to add two numbers
Action: calculate(2 + 2)
Observation: Result is 4
Output: The answer is 4
---
```

> **Note:** The `FewShotReActAgent` struct is the older implementation layer. The Python-facing `FewShotReAct` class uses `GraphBasedAgent` and the built-in system prompt, not `FewShotReActAgent` directly.

## State keys

Same as `ZeroShotReAct`:

| Key | Set by | Meaning |
|-----|--------|---------|
| `input` | caller | user question |
| `llm_response` | agent node | raw LLM text |
| `needs_tool` | agent node | tool call detected |
| `pending_tool_name` | agent node | tool to call |
| `pending_tool_args` | agent node | tool arguments |
| `tool_result` | tool_executor | result string |
| `__agent_type` | initialize() | `"few-shot-react"` |
| `__agent_examples_count` | initialize() | number of examples stored |

## Python usage

```python
from flowgentra_ai.agent import FewShotReAct, ToolSpec
from flowgentra_ai.llm import LLM

llm = LLM(provider="anthropic", model="claude-opus-4-6")

calc = ToolSpec("calculator", "Evaluate a math expression")
calc.add_parameter("expression", "string")
calc.set_required("expression")

agent = FewShotReAct(
    name="math-agent",
    llm=llm,
    tools=[calc],
    retries=2,
)

result = agent.execute_input("What is 17 * 43?")
print(result)
```

## Parameters

Same as `ZeroShotReAct`:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | str | required | Agent identifier |
| `llm` | LLM | required | Language model |
| `system_prompt` | str | None | Override default few-shot prompt |
| `tools` | list[ToolSpec] | [] | Available tools |
| `retries` | int | 3 | Max LLM retries |
| `memory_steps` | int | None | Conversation window size |

## When to use vs ZeroShotReAct

| Scenario | Prefer |
|----------|--------|
| LLM reliably follows instructions | `ZeroShotReAct` |
| LLM needs format guidance | `FewShotReAct` |
| Smaller/weaker models | `FewShotReAct` |
| Minimizing prompt tokens | `ZeroShotReAct` |
| Custom domain with specific patterns | `FewShotReAct` with `system_prompt` override |
