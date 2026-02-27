# QA Agent Example

A complete, runnable example of an ErenFlowAI agent with:
- **config.yaml** - Workflow configuration
- **handlers** - Multi-step handlers with auto-registration
- **main.rs** - Entry point that loads config and runs the workflow

## Structure

```
qa_agent/
├── config.yaml              # Workflow definition
├── src/
│   ├── main.rs             # Entry point
│   └── handlers.rs         # Handler implementations
├── .env.example            # Environment variables template
└── README.md               # This file
```

## Setup

### 1. Get a Mistral API Key

- Visit [console.mistral.ai](https://console.mistral.ai/)
- Create an account (free tier available)
- Generate an API key

### 2. Set Environment Variable

```bash
export MISTRAL_API_KEY=your_key_here
```

Or create a `.env` file:
```bash
echo "MISTRAL_API_KEY=your_key_here" > .env
```

## Running the Example

From the project root (`erenflow-ai/`):

```bash
cargo run --example qa_agent
```

Or from the example directory:

```bash
cd examples/qa_agent
cargo run
```

## Expected Output

```
======================================================================
🤖 QA Agent Example - Loading from config.yaml
======================================================================

📋 Loading configuration from config.yaml...

✅ Configuration loaded successfully!
   Agent: qa_agent

📝 User Question:
   "What is Rust and why is it important?"

🔄 Executing workflow:

   START
   ✓ Handler: validate_input
   ✓ Handler: get_context
   ✓ Handler: generate_answer
   ✓ Handler: format_response
   END

======================================================================
✅ Workflow completed successfully!

📤 Final Response:

{
  "question": "What is Rust and why is it important?",
  "answer": "Based on the provided context and question: 'What is Rust and why is it important?', this is a comprehensive answer...",
  "timestamp": "2024-02-27T10:30:45Z",
  "status": "success"
}

======================================================================
✨ Example completed successfully!
======================================================================
```

## What It Does

### Workflow Steps:

1. **validate_input** - Checks that the user question is valid
2. **get_context** - Retrieves relevant context for the question
3. **generate_answer** - Uses Mistral LLM to generate an answer
4. **format_response** - Formats the response nicely

### State Flow:

```
User Input
    ↓
[validate_input] → validated_question
    ↓
[get_context] → context
    ↓
[generate_answer] → llm_answer (from Mistral)
    ↓
[format_response] → final_response (JSON)
    ↓
Output
```

## Customization

### Change the Question

Edit `src/main.rs` line 44:
```rust
let user_question = "Your new question here?";
```

### Change the LLM Model

Edit `config.yaml` line 9:
```yaml
model: "mistral-large"  # or "mistral-medium"
```

### Add More Handlers

1. Add a new handler function in `src/handlers.rs`:
```rust
#[register_handler]
pub async fn my_handler(mut state: State) -> Result<State> {
    // Your logic here
    Ok(state)
}
```

2. Add a new node to `config.yaml`:
```yaml
- name: my_node
  handler: handlers::my_handler
  description: "My custom handler"
```

3. Add edges to connect it:
```yaml
- from: some_node
  to: my_node
```

## Troubleshooting

### "MISTRAL_API_KEY environment variable not set"

```bash
export MISTRAL_API_KEY=your_key_here
```

### "Failed to load config"

Make sure you're running from the project root:
```bash
cd erenflow-ai
cargo run --example qa_agent
```

### "API request failed"

1. Verify your API key is correct at [console.mistral.ai](https://console.mistral.ai/)
2. Check internet connectivity
3. Ensure your Mistral account has API credits

### "Handler not found"

Ensure handlers are decorated with `#[register_handler]` macro.

## Learn More

- [ErenFlowAI Documentation](../DOCUMENTATION.md)
- [Configuration Guide](../CONFIG_GUIDE.md)
- [Feature Documentation](../FEATURES.md)
- [Mistral API Docs](https://docs.mistral.ai/)
