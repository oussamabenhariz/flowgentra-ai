# ReactDocstore

ReAct loop specialized for document-store retrieval. The agent uses exactly three operations — `Search`, `Lookup`, and `Finish` — to navigate a document store and answer questions.

Requires two tools: **`"search"`** and **`"lookup"`**. A single executor function dispatches both by tool name.

Equivalent to LangChain's `react-docstore` agent type.

## How it works

```
START
  └─► agent node ──(Search or Lookup)──► tool_executor node ──┐
        ▲                                                       │
        └───────────────────────────────────────────────────────┘
        │(Finish)
        ▼
       END
```

1. `agent` node calls `DocstoreNode`.
2. `DocstoreNode` builds the prompt with the two built-in examples and sends it to the LLM.
3. The LLM responds with a Thought + one of:
   - `Action: Search[query]` → search the doc store
   - `Action: Lookup[term]` → look up a term in the last returned document
   - `Action: Finish[answer]` → terminate with final answer
4. `docstore_router` parses the action:
   - `Search` or `Lookup` → routes to `tool_executor`, sets `pending_tool_name` and `pending_tool_args`
   - `Finish` → stores `ds_final_answer`, routes to `END`
5. `tool_executor` calls the user executor with `("search", query)` or `("lookup", term)` and stores the result as `Observation:` in the scratchpad.
6. The loop continues until `Finish` is emitted.
7. `execute_input` returns `state["ds_final_answer"]`, falling back to `state["llm_response"]`.

## Output format

The LLM follows this Thought/Action/Observation pattern (from built-in examples):

```
Thought: I need to search for Colorado orogeny.
Action: Search[Colorado orogeny]
Observation: The Colorado orogeny was an episode of mountain building...
Thought: It does not mention the eastern sector. So I need to look up eastern sector.
Action: Lookup[eastern sector]
Observation: The eastern sector extends into the High Plains.
Thought: High Plains rise in elevation from 1,800 to 7,000 ft.
Action: Finish[1,800 to 7,000 ft (550 to 2,130 m).]
```

## Built-in tool stubs

The default `ReactDocstoreAgent` pre-registers both required tool stubs:

```rust
// search tool
ToolSpec::new("search", "Search the document store for a query")
    .with_parameter("query", "string").required("query")

// lookup tool
ToolSpec::new("lookup", "Look up a term in the most recently found document")
    .with_parameter("term", "string").required("term")
```

## State keys

| Key | Set by | Meaning |
|-----|--------|---------|
| `input` | caller | question to answer |
| `llm_response` | agent node | raw LLM output |
| `pending_tool_name` | agent node | `"search"` or `"lookup"` |
| `pending_tool_args` | agent node | query or term string |
| `tool_result` | tool_executor | document or passage returned |
| `scratchpad` | initialize() / agent node | accumulated Thought/Action/Observation |
| `ds_final_answer` | agent node | content of `Finish[...]` |
| `__agent_type` | initialize() | `"react-docstore"` |

## Python usage

```python
from flowgentra_ai.agent import ReactDocstore, ToolSpec
from flowgentra_ai.llm import LLM

llm = LLM(provider="openai", model="gpt-4o")

# Both tools must be named exactly "search" and "lookup"
search = ToolSpec("search", "Search the document store")
search.add_parameter("query", "string")
search.set_required("query")

lookup = ToolSpec("lookup", "Look up a term in the last document")
lookup.add_parameter("term", "string")
lookup.set_required("term")

# In-memory docstore for illustration
docstore = {"Colorado orogeny": "The Colorado orogeny ... eastern sector extends into High Plains."}
last_doc = {}

def executor(tool_name: str, args: str) -> str:
    if tool_name == "search":
        return docstore.get(args, "Not found")
    if tool_name == "lookup":
        # Look up term in last document returned
        doc = last_doc.get("content", "")
        return doc if args.lower() in doc.lower() else "Not found in document"
    return "Unknown tool"

agent = ReactDocstore(
    name="doc-agent",
    llm=llm,
    tools=[search, lookup],
)

result = agent.execute_input(
    "What is the elevation range for the area the eastern sector of the Colorado orogeny extends into?"
)
print(result)
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | str | required | Agent identifier |
| `llm` | LLM | required | Language model |
| `system_prompt` | str | None | Override default docstore examples |
| `tools` | list[ToolSpec] | [] | Must include `"search"` and `"lookup"` |
| `retries` | int | 3 | Max LLM retries |
| `memory_steps` | int | None | Conversation window size |

## When to use

- Structured document retrieval (Wikipedia-style, knowledge bases)
- Tasks requiring iterative zooming: broad search → specific lookup
- Multi-step factual Q&A grounded in a document corpus

## When NOT to use

- Web search (no persistent "last document" concept) — use `SelfAskWithSearch`
- Tasks requiring tools beyond search and lookup — use `ZeroShotReAct` or `ToolCalling`
- Real-time streaming document sources

## Notes

- The executor function receives the tool name as a string, so a single function can dispatch both `"search"` and `"lookup"` calls.
- The system prompt consists entirely of two worked examples. Overriding `system_prompt` replaces both.
- `Lookup` semantics depend entirely on the user's executor — the agent itself does not track which document was last returned.
