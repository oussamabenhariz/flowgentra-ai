# LLM Configuration Guide

Set up any language model provider, add retries, track costs, and get structured output.

## Supported Providers

| Provider | Auth | Streaming | Tool Calling | Local |
|----------|------|-----------|--------------|-------|
| **OpenAI** | Bearer token | SSE | Function calling | No |
| **Anthropic** | x-api-key | SSE | input_schema | No |
| **Mistral** | Bearer token | SSE | Function calling | No |
| **Groq** | Bearer token | SSE | Function calling | No |
| **Azure OpenAI** | api-key header | SSE | Function calling | No |
| **HuggingFace** | Bearer token | Real SSE (TGI) | -- | Optional |
| **Ollama** | None | NDJSON | -- | Yes |

## Quick Setup

### OpenAI

```yaml
llm:
  provider: openai
  model: gpt-4
  api_key: ${OPENAI_API_KEY}
  temperature: 0.7
  max_tokens: 2000
```

### Anthropic (Claude)

```yaml
llm:
  provider: anthropic
  model: claude-3-opus-20240229
  api_key: ${ANTHROPIC_API_KEY}
```

### Mistral / Groq

```yaml
llm:
  provider: mistral  # or groq
  model: mistral-large  # or mixtral-8x7b-32768
  api_key: ${MISTRAL_API_KEY}
```

### Azure OpenAI

```yaml
llm:
  provider: azure
  model: gpt-4
  api_key: ${AZURE_API_KEY}
  extra_params:
    resource_name: "my-resource"
    api_version: "2024-02-15-preview"
```

### HuggingFace

```yaml
# Cloud API
llm:
  provider: huggingface
  model: mistralai/Mistral-7B-Instruct-v0.1
  api_key: ${HF_API_TOKEN}

# Local TGI server
llm:
  provider: huggingface
  model: mistralai/Mistral-7B-Instruct-v0.1
  api_key: ""
  extra_params:
    mode: "local"
    endpoint: "http://localhost:80"
```

HuggingFace streaming uses real SSE with TGI's `token.text` format and also supports OpenAI-compatible `choices[0].delta.content`.

### Ollama (Local)

```yaml
llm:
  provider: ollama
  model: mistral
  base_url: http://localhost:11434
```

---

## RetryLLMClient

Wrap any LLM client with automatic retry on transient failures:

```rust
use flowgentra_ai::core::llm::RetryLLMClient;
use std::time::Duration;

let retry_client = RetryLLMClient::new(inner_client)
    .with_max_retries(3)
    .with_initial_delay(Duration::from_millis(500));

// Retries automatically on network errors, rate limits, etc.
let response = retry_client.chat(messages).await?;
```

Uses exponential backoff between retries.

---

## Token Counting and Context Window

### Estimate Tokens

```rust
use flowgentra_ai::core::llm::token_counter::estimate_tokens;

let count = estimate_tokens("Hello, how are you today?");
// Roughly text.len() / 3.5
```

### Check Context Window Limits

```rust
use flowgentra_ai::core::llm::token_counter::context_window;

let max = context_window("gpt-4");          // Some(8192)
let max = context_window("claude-3-opus");  // Some(200000)
let max = context_window("mistral-large");  // Some(32768)
```

### Truncate Messages to Fit

```rust
use flowgentra_ai::core::llm::token_counter::ContextWindow;

let ctx = ContextWindow {
    max_tokens: 8192,
    reserve_for_completion: 1024,
};

// Keeps system message + most recent messages that fit
let trimmed = ctx.truncate(&messages);
```

---

## Cost Tracking

### Model Pricing

```rust
use flowgentra_ai::core::llm::model_pricing;

// Returns (input_cost_per_million, output_cost_per_million) in USD
let (input_price, output_price) = model_pricing("gpt-4").unwrap();
// (30.0, 60.0) = $30/M input, $60/M output
```

### Estimate Cost Per Call

```rust
// After an LLM call with usage tracking:
let (response, usage) = client.chat_with_usage(messages).await?;

if let Some(usage) = usage {
    println!("Input tokens: {}", usage.prompt_tokens);
    println!("Output tokens: {}", usage.completion_tokens);

    if let Some(cost) = usage.estimated_cost("gpt-4") {
        println!("Estimated cost: ${:.4}", cost);
    }
}
```

---

## Structured Output (ResponseFormat)

Get deterministic JSON responses instead of free-form text:

### JSON Mode

Forces the LLM to output valid JSON:

```rust
use flowgentra_ai::core::llm::{LLMConfig, ResponseFormat};

let config = LLMConfig::new(provider, model, key)
    .with_response_format(ResponseFormat::Json);
```

For Anthropic, this appends a system instruction: "You must respond with valid JSON only."

### JSON Schema Mode (OpenAI)

Enforce a specific structure:

```rust
let config = LLMConfig::new(provider, model, key)
    .with_response_format(ResponseFormat::JsonSchema {
        name: "sentiment_analysis".into(),
        schema: json!({
            "type": "object",
            "properties": {
                "sentiment": { "type": "string", "enum": ["positive", "negative", "neutral"] },
                "confidence": { "type": "number" }
            },
            "required": ["sentiment", "confidence"]
        }),
    });
```

---

## Anthropic Tool Calling

Anthropic uses a different format than OpenAI for tool definitions and responses:

```rust
// Tool definitions use `input_schema` (not `parameters`)
// {
//   "name": "search",
//   "description": "Search the web",
//   "input_schema": { "type": "object", ... }
// }

// Response tool calls come as `tool_use` content blocks:
// { "type": "tool_use", "id": "...", "name": "search", "input": {...} }

// All handled automatically by the adapter:
let response = client.chat_with_tools(messages, &tools).await?;
if let Some(tool_calls) = response.tool_calls {
    for call in tool_calls {
        println!("Tool: {}, Args: {}", call.name, call.arguments);
    }
}
```

---

## CachedLLMClient

Cache LLM responses to avoid redundant API calls. Uses hash-based caching on message role and content:

```rust
use flowgentra_ai::prelude::*;
use std::sync::Arc;

let cached = CachedLLMClient::new(inner_client)
    .with_max_entries(1000);

let response = cached.chat(messages.clone()).await?;
let again = cached.chat(messages).await?; // cache hit, no API call

println!("Cache entries: {}", cached.cache_size());
cached.clear_cache();
```

Note: Tool calls and streaming are **not** cached (they have side effects).

---

## FallbackLLMClient

Try multiple LLM providers in sequence until one succeeds:

```rust
use flowgentra_ai::prelude::*;

let client = FallbackLLMClient::new(primary_client)
    .with_fallback(secondary_client)
    .with_fallback(tertiary_client);

// Tries primary → secondary → tertiary
let response = client.chat(messages).await?;
```

Implements the full `LLMClient` trait (chat, chat_with_usage, chat_with_tools, chat_stream). Failed providers are logged via `tracing::warn!`.

### Config-Based Fallback

```yaml
llm:
  provider: openai
  model: gpt-4
  api_key: ${OPENAI_API_KEY}
  fallbacks:
    - provider: anthropic
      model: claude-3-opus-20240229
      api_key: ${ANTHROPIC_API_KEY}
    - provider: ollama
      model: mistral
      base_url: http://localhost:11434
```

---

## PromptTemplate

Template strings with `{variable}` interpolation and auto-extraction of variable names:

```rust
use flowgentra_ai::core::llm::prompt_template::PromptTemplate;

let template = PromptTemplate::new("Translate '{text}' to {language}");

// Check required variables
assert_eq!(template.variables(), &["language", "text"]);

// Format with all variables
let result = template.format(&[
    ("text", "Hello"),
    ("language", "French"),
])?;
// "Translate 'Hello' to French"

// Partial formatting (fill some variables, leave others)
let partial = template.partial(&[("language", "Spanish")])?;
let result = partial.format(&[("text", "Goodbye")])?;
```

### ChatPromptTemplate

Build multi-message prompts:

```rust
use flowgentra_ai::core::llm::prompt_template::ChatPromptTemplate;

let prompt = ChatPromptTemplate::new()
    .system("You are a {domain} expert.")
    .user("Explain {concept} in simple terms.");

let messages = prompt.format_messages(&[
    ("domain", "Rust"),
    ("concept", "lifetimes"),
])?;
// [Message::system("You are a Rust expert."), Message::user("Explain lifetimes in simple terms.")]
```

### FewShotPromptTemplate

Build prompts with examples:

```rust
use flowgentra_ai::core::llm::prompt_template::FewShotPromptTemplate;

let template = FewShotPromptTemplate::new(
    "Classify the sentiment:",
    "Input: {input}\nSentiment: {output}",
    "Input: {input}\nSentiment:",
);

let examples = vec![
    vec![("input", "I love it"), ("output", "positive")],
    vec![("input", "Terrible"), ("output", "negative")],
];

let prompt = template.format(&examples, &[("input", "Pretty good")])?;
```

---

## OutputParser

Parse structured data from LLM text responses. All parsers implement the `OutputParser` trait with `parse()` and `format_instructions()`.

### JsonOutputParser

Extracts JSON from freeform text, including markdown code fences:

```rust
use flowgentra_ai::core::llm::output_parser::JsonOutputParser;

let parser = JsonOutputParser::new();

// Handles markdown fences
let value = parser.parse("Here's the result:\n```json\n{\"score\": 95}\n```")?;

// Handles raw JSON in text
let value = parser.parse("The answer is {\"name\": \"Alice\", \"age\": 30}")?;
```

### ListOutputParser

Parse lists from various formats:

```rust
use flowgentra_ai::core::llm::output_parser::ListOutputParser;

let parser = ListOutputParser::comma_separated();
let items = parser.parse("apples, oranges, bananas")?;

let parser = ListOutputParser::newline_separated();
let items = parser.parse("first\nsecond\nthird")?;

let parser = ListOutputParser::numbered();
let items = parser.parse("1. alpha\n2. beta\n3. gamma")?;
```

### StructuredOutputParser

Validate required fields in JSON output:

```rust
use flowgentra_ai::core::llm::output_parser::{StructuredOutputParser, FieldSpec};

let parser = StructuredOutputParser::new(vec![
    FieldSpec { name: "name".into(), field_type: "string".into(), description: "The name".into() },
    FieldSpec { name: "score".into(), field_type: "number".into(), description: "Quality score".into() },
]);

// Validates that "name" and "score" are present
let value = parser.parse("{\"name\": \"test\", \"score\": 42}")?;
```

---

## Provider Comparison

| Provider | Cost | Speed | Quality | Best For |
|----------|------|-------|---------|----------|
| **OpenAI** | Paid | Fast | Excellent | Production, high quality |
| **Anthropic** | Paid | Medium | Excellent | Long context, safety |
| **Groq** | Free tier | Very fast | Good | Low-latency, prototyping |
| **HuggingFace** | Free/Paid | Medium | Good | Open models, privacy (local TGI) |
| **Ollama** | Free | Medium | Good | Local development |
| **Azure** | Paid | Fast | Excellent | Enterprise |

## Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `temperature` | Randomness (0.0 = deterministic, 1.0 = creative) | 0.7 |
| `max_tokens` | Maximum response length | 2048 |
| `top_p` | Nucleus sampling threshold | 1.0 |
| `timeout` | Request timeout in seconds | 30 |

---

See [FEATURES.md](../FEATURES.md) for the complete feature list.
