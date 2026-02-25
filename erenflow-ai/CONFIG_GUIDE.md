# ErenFlowAI Config.yaml Complete Guide

## Overview

The `config.yaml` file is the declarative definition of your entire ErenFlowAI agent. It defines:
- **What LLM provider to use** (OpenAI, Anthropic, etc.)
- **What nodes (steps) your workflow has**
- **How nodes connect** (edges with optional conditions)
- **What data flows through** the agent (state schema)
- **Middleware, RAG, MCP tools, and monitoring**

**Without config.yaml, you cannot run an agent.** It's the single source of truth for your agent's structure.

---

## Minimal Configuration

Here's the absolute minimum config.yaml to get started:

```yaml
name: "my_simple_agent"
description: "A basic agent"

llm:
  provider: openai
  model: gpt-4
  api_key: ${OPENAI_API_KEY}

graph:
  nodes:
    - name: process
      handler: handlers::process_input

  edges:
    - from: START
      to: process
    - from: process
      to: END
```

This creates an agent with one node that receives input and returns output.

---

## Complete Configuration Structure

Here's a fully-featured config.yaml with all available options:

```yaml
# ============================================================================
# 1. BASIC AGENT METADATA
# ============================================================================
name: "comprehensive_agent"
description: "A comprehensive agent showcasing all ErenFlowAI features"
version: "1.0.0"      # Optional: Agent version
author: "Your Name"   # Optional: Author info

# ============================================================================
# 2. LLM CONFIGURATION (REQUIRED)
# ============================================================================
# This determines which language model powers your agent
llm:
  # Primary provider
  provider: openai                    # Required
  model: gpt-4                        # Required: specific model version
  
  # API authentication
  api_key: ${OPENAI_API_KEY}          # Use env vars for secrets
  
  # Behavior parameters
  temperature: 0.7                    # 0.0 (deterministic) - 1.0 (creative)
  max_tokens: 2000                    # Optional: max response length
  top_p: 0.9                          # Optional: nucleus sampling
  
  # Timeout and retry policy
  timeout: 30                         # Seconds before timeout
  max_retries: 3                      # Auto-retry failed requests
  
  # Fallback providers (automatic failover)
  fallbacks:
    - provider: anthropic
      model: claude-3-opus-20240229
      api_key: ${ANTHROPIC_API_KEY}
      temperature: 0.7
      
    - provider: mistral
      model: mistral-large
      api_key: ${MISTRAL_API_KEY}

# ============================================================================
# 3. GRAPH CONFIGURATION (REQUIRED)
# ============================================================================
# This defines the workflow structure: nodes and how they connect
graph:
  # NODES: Individual computational steps
  nodes:
    # Simple node with just a handler
    - name: input_processor
      description: "Extract and validate user input"
      handler: handlers::input_processor
    
    # Node with timeout and retry settings
    - name: retrieve_context
      description: "Retrieve relevant context from vector store"
      handler: handlers::retrieve_context
      retries: 3                      # Retry this node up to 3 times
      timeout: 15                     # This node must complete in 15s
    
    # Node that uses LLM
    - name: plan_query
      description: "Plan the query execution strategy"
      handler: handlers::plan_query
      uses_llm: true                  # This node will call the LLM
      timeout: 20
    
    # Node that uses multiple features
    - name: gather_information
      description: "Gather info using tools"
      handler: handlers::gather_information
      uses_llm: true                  # Uses LLM
      uses_rag: true                  # Uses vector store
      uses_mcp: true                  # Uses external tools
      mcp_tools:                      # Which tools to use
        - web_search
        - calculator
        - file_reader
      timeout: 30
    
    # Node that uses RAG
    - name: analyze_data
      description: "Analyze gathered data"
      handler: handlers::analyze_data
      uses_llm: true
      uses_rag: true
      timeout: 25
    
    # Final response generation
    - name: generate_response
      description: "Generate final response"
      handler: handlers::generate_response
      uses_llm: true
      timeout: 20
    
    # Output formatting
    - name: format_output
      description: "Format output for user"
      handler: handlers::format_output
      timeout: 10

  # EDGES: Connections between nodes
  edges:
    # Unconditional edges (always follow)
    - from: START
      to: input_processor
      description: "Begin workflow"

    - from: input_processor
      to: retrieve_context
      description: "Retrieve context after input"

    - from: retrieve_context
      to: plan_query
      description: "Plan strategy after context"

    # Conditional edges (follow only if condition is true)
    - from: plan_query
      to: gather_information
      condition: is_complex_query     # Only if this condition returns true
      description: "Complex queries need more info gathering"

    - from: plan_query
      to: analyze_data
      condition: is_simple_query      # Only if this condition returns true
      description: "Simple queries go straight to analysis"

    # Converging edges
    - from: gather_information
      to: analyze_data
      description: "Combine gathered info with analysis"

    - from: analyze_data
      to: generate_response
      description: "Generate response from analysis"

    - from: generate_response
      to: format_output
      description: "Format final output"

    - from: format_output
      to: END
      description: "Complete workflow"

# ============================================================================
# 4. CONDITIONAL ROUTING
# ============================================================================
# Define conditions used in edges above
conditions:
  - name: is_complex_query
    description: "Check if query complexity > 70"
  
  - name: is_simple_query
    description: "Check if query is straightforward"

# ============================================================================
# 5. STATE SCHEMA (Optional but Recommended)
# ============================================================================
# Document what data flows through your agent
# Format: Each field has a type and description
state_schema:
  # Input/Output
  input:
    type: string
    description: "User input query"
  output:
    type: object
    description: "Formatted final output"
  
  # Processing stages
  input_validated:
    type: boolean
    description: "Whether input was validated"
  context_docs:
    type: Array<Document>
    description: "Retrieved context from RAG"
  query_plan:
    type: object
    description: "Execution plan for query"
  is_complex:
    type: boolean
    description: "Whether query is complex"
  
  # Intermediate results
  search_results:
    type: Array<object>
    description: "Web search results"
  calculation_results:
    type: Array<string>
    description: "Math calculation results"
  analysis:
    type: object
    description: "Analysis of gathered data"
  
  # Final results
  response:
    type: string
    description: "Generated response content"
  citations:
    type: Array<Citation>
    description: "Sources and citations"
  metadata:
    type: object
    description: "Additional metadata"

# ============================================================================
# 6. RAG CONFIGURATION (Optional)
# ============================================================================
# Enable semantic search and retrieval-augmented generation
rag:
  enabled: true
  
  # Vector store selection and config
  vector_store:
    type: pinecone                        # Options: pinecone, weaviate, chroma
    index_name: "erenflow-index"         # Pinecone: index name
    environment: "us-west-2-aws"          # Pinecone: region
    api_key: ${PINECONE_API_KEY}          # Pinecone: API key
    
    # Alternative: Weaviate config
    # type: weaviate
    # url: "http://localhost:8080"
    # api_key: ${WEAVIATE_API_KEY}
    
    # Alternative: Chroma config
    # type: chroma
    # collection_name: "documents"
    # host: "localhost"
    # port: 8000
  
  # Embedding configuration
  embedding_model: openai                 # Options: openai, huggingface
  embedding_dimension: 1536               # Must match model output
  embedding_batch_size: 100               # Batch process embeddings
  
  # Retrieval behavior
  retrieval_strategy: semantic            # Options: semantic, bm25, hybrid
  top_k: 5                                # Return top 5 most similar docs
  similarity_threshold: 0.7               # Min score to include doc
  
  # Chunking strategy
  chunk_size: 1024                        # Characters per chunk
  chunk_overlap: 200                      # Overlap between chunks
  separator: "\n\n"                       # Chunk boundary

# ============================================================================
# 7. MCP CONFIGURATION (Optional)
# ============================================================================
# Model Context Protocol: Connect to external tools and services
mcp:
  enabled: true
  
  # Available tools
  tools:
    # External HTTP tools
    - name: web_search
      description: "Search the web for information"
      type: external                      # Options: external, builtin
      endpoint: "http://localhost:3000/web_search"
      method: POST                        # HTTP method
      auth:
        type: bearer                      # Options: none, bearer, api_key
        token: ${WEB_SEARCH_TOKEN}
      timeout: 10
    
    # Built-in tools
    - name: calculator
      description: "Perform mathematical calculations"
      type: builtin
      capabilities:
        - basic_math
        - trigonometry
        - statistics
    
    # Another external tool
    - name: file_reader
      description: "Read and process files"
      type: external
      endpoint: "http://localhost:3001/files"
      method: POST
      auth:
        type: api_key
        key: ${FILE_READER_API_KEY}
      timeout: 15
  
  # Tool execution settings
  execution:
    parallel: true                        # Run multiple tools concurrently
    max_parallel: 3                       # Max concurrent executions
    timeout: 30                           # Tool execution timeout
    error_handling: "continue"            # Continue on tool error

# ============================================================================
# 8. MIDDLEWARE (Optional)
# ============================================================================
# Add middleware for request/response processing
middleware:
  # Logging middleware
  - name: logging
    enabled: true
    level: info                           # debug, info, warn, error
    include_state: true                   # Log state contents
  
  # Rate limiting
  - name: rate_limiting
    enabled: true
    rpm: 60                               # 60 requests per minute
    burst_size: 10                        # Allow 10 in rapid succession
  
  # Response caching
  - name: cache
    enabled: true
    ttl: 3600                             # 1 hour cache TTL
    strategy: content_hash                # Cache based on input hash

# ============================================================================
# 9. HEALTH MONITORING (Optional)
# ============================================================================
# Monitor agent and service health
health:
  enabled: true
  check_interval: 30                      # Check every 30 seconds
  
  checks:
    - llm_connectivity                    # Can reach LLM provider
    - rag_availability                    # Vector store is accessible
    - mcp_tools                           # External tools respond
    - memory_usage                        # Memory not exhausted
    - response_time                       # Response time acceptable
  
  # Alert thresholds
  memory_threshold: 80                    # % memory used
  response_time_threshold: 5000           # ms max response time
  
  # Failure policies
  on_failure: "alert"                     # Options: alert, fallback, stop

# ============================================================================
# 10. OBSERVABILITY & TRACING (Optional)
# ============================================================================
# Enable distributed tracing and monitoring
observability:
  tracing_enabled: true
  trace_level: debug                      # debug, info, warn, error
  
  # Which spans to trace and at what sample rate
  spans:
    - name: handler_execution
      sample_rate: 1.0                    # Always sample this
    
    - name: llm_calls
      sample_rate: 0.5                    # Sample 50% of LLM calls
    
    - name: rag_queries
      sample_rate: 0.5                    # Sample 50% of RAG queries
    
    - name: mcp_tool_execution
      sample_rate: 1.0                    # Always sample tools
  
  # Where to export traces
  exporters:
    - type: jaeger
      endpoint: "http://localhost:14268/api/traces"
    
    - type: otlp_http
      endpoint: "http://localhost:4318/v1/traces"
    
    - type: datadog
      api_key: ${DATADOG_API_KEY}
      service: "erenflow-agent"

# ============================================================================
# 11. ERROR HANDLING (Optional)
# ============================================================================
# Configure error handling and recovery
error_handling:
  default_strategy: "retry"               # retry, fallback, abort
  
  # Retry configuration
  retries:
    max_attempts: 3                       # Max retry attempts
    backoff_strategy: exponential          # linear, exponential, fixed
    initial_delay: 1                      # Starting delay in seconds
    max_delay: 60                         # Max delay between retries
  
  # Fallback strategies
  fallback:
    enabled: true
    cache_responses: true                 # Cache successful responses
    cache_ttl: 3600

# ============================================================================
# 12. ENVIRONMENT & DEPLOYMENT (Optional)
# ============================================================================
# Environment-specific settings
environment:
  name: development                       # development, staging, production
  debug_mode: true                        # Enable debug output
  
  # Resource limits
  resources:
    memory_limit: "512Mi"                 # Max memory allowed
    timeout_global: 300                   # Global timeout in seconds
    
  # Logging to file
  log_file:
    enabled: true
    path: "./logs/agent.log"
    max_size: "100Mi"
    max_backups: 5
    max_age_days: 7
```

---

## Step-by-Step: Creating Your First config.yaml

### Step 1: Start with the Structure

Create `config.yaml` in your project root:

```yaml
name: "my_agent"
description: "My first agent"
```

### Step 2: Add LLM Configuration

Choose your provider and add credentials:

```yaml
llm:
  provider: openai
  model: gpt-4
  api_key: ${OPENAI_API_KEY}
```

### Step 3: Create Your Handlers in Rust

Create `src/handlers.rs`:

```rust
use erenflow_ai::prelude::*;
use serde_json::json;

pub async fn input_processor(mut state: State) -> Result<State> {
    println!("Processing input...");
    // Your logic here
    Ok(state)
}

pub async fn generate_response(mut state: State) -> Result<State> {
    println!("Generating response...");
    // Your logic here
    Ok(state)
}
```

### Step 4: Define Your Workflow

Add graph nodes and edges:

```yaml
graph:
  nodes:
    - name: input_processor
      handler: handlers::input_processor

    - name: generate_response
      handler: handlers::generate_response
      uses_llm: true

  edges:
    - from: START
      to: input_processor
    
    - from: input_processor
      to: generate_response
    
    - from: generate_response
      to: END
```

### Step 5: Document Your State

Add state_schema:

```yaml
state_schema:
  input:
    type: string
    description: "User input"
  response:
    type: string
    description: "Agent response"
```

### Step 6: Run Your Agent

In `src/main.rs`:

```rust
use erenflow_ai::prelude::*;
use serde_json::json;

mod handlers;

#[tokio::main]
async fn main() -> Result<()> {
    let mut agent = Agent::from_config_with_handlers(
        "config.yaml",
        &handlers,
    )?;

    let mut state = State::new();
    state.set("input", json!("Hello!"));

    let result = agent.run(state).await?;
    Ok(())
}
```

---

## Environment Variables

ErenFlowAI supports `${VARIABLE_NAME}` substitution in config.yaml.

### Setting Environment Variables

**Option 1: Command line**
```bash
export OPENAI_API_KEY="sk-..."
export PINECONE_API_KEY="your-key"
cargo run
```

**Option 2: .env file**

Create `.env` in your project root:

```
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-...
PINECONE_API_KEY=your-key
MISTRAL_API_KEY=your-key
```

Then create a `.env.example` for sharing:

```
OPENAI_API_KEY=your-key-here
ANTHROPIC_API_KEY=your-key-here
PINECONE_API_KEY=your-key-here
```

**Option 3: Docker/Container**

```dockerfile
ENV OPENAI_API_KEY="sk-..."
ENV PINECONE_API_KEY="your-key"
```

---

## Provider Configuration Examples

### OpenAI

```yaml
llm:
  provider: openai
  model: gpt-4                    # or gpt-4-turbo, gpt-3.5-turbo
  api_key: ${OPENAI_API_KEY}
  temperature: 0.7
  max_tokens: 2000
  timeout: 30
```

### Anthropic (Claude)

```yaml
llm:
  provider: anthropic
  model: claude-3-opus-20240229   # or claude-3-sonnet, claude-3-haiku
  api_key: ${ANTHROPIC_API_KEY}
  temperature: 0.7
```

### Azure OpenAI

```yaml
llm:
  provider: azure
  model: gpt-4
  api_key: ${AZURE_OPENAI_KEY}
  endpoint: ${AZURE_OPENAI_ENDPOINT}
  api_version: "2024-02-15-preview"
```

### Mistral

```yaml
llm:
  provider: mistral
  model: mistral-large
  api_key: ${MISTRAL_API_KEY}
  temperature: 0.7
```

### Groq

```yaml
llm:
  provider: groq
  model: mixtral-8x7b-32768
  api_key: ${GROQ_API_KEY}
  temperature: 0.7
```

### Ollama (Local)

```yaml
llm:
  provider: ollama
  model: mistral                  # or llama2, neural-chat, etc.
  base_url: "http://localhost:11434"
  timeout: 60                     # Local can be slower
```

---

## Vector Store Configuration Examples

### Pinecone

```yaml
rag:
  enabled: true
  vector_store:
    type: pinecone
    index_name: "my-index"
    environment: "us-west-2-aws"  # or your region
    api_key: ${PINECONE_API_KEY}
  embedding_model: openai
  embedding_dimension: 1536
```

### Weaviate

```yaml
rag:
  enabled: true
  vector_store:
    type: weaviate
    url: "http://localhost:8080"
    api_key: ${WEAVIATE_API_KEY}
  embedding_model: openai
  embedding_dimension: 1536
```

### Chroma

```yaml
rag:
  enabled: true
  vector_store:
    type: chroma
    collection_name: "documents"
    host: "localhost"
    port: 8000
  embedding_model: openai
  embedding_dimension: 1536
```

---

## Common Patterns

### Simple Sequential Workflow

```yaml
graph:
  nodes:
    - name: step1
      handler: handlers::step1
    - name: step2
      handler: handlers::step2
    - name: step3
      handler: handlers::step3

  edges:
    - from: START
      to: step1
    - from: step1
      to: step2
    - from: step2
      to: step3
    - from: step3
      to: END
```

### Conditional Branching

```yaml
edges:
  - from: route_node
    to: path_a
    condition: condition_a        # Takes path A if true

  - from: route_node
    to: path_b
    condition: condition_b        # Takes path B if true
```

### Parallel Execution

```yaml
edges:
  # Both execute after source
  - from: source
    to: branch1

  - from: source
    to: branch2

  # Both converge at destination
  - from: branch1
    to: destination

  - from: branch2
    to: destination
```

---

## Validation Checklist

Before running your agent, verify:

- ✅ `config.yaml` is in correct location
- ✅ YAML syntax is valid (proper indentation, no hard tabs)
- ✅ Handler paths match actual Rust functions
- ✅ All handlers are declared `pub async fn`
- ✅ Environment variables are set (e.g., `OPENAI_API_KEY`)
- ✅ All node names in edges exist
- ✅ No circular paths (unless intentional)
- ✅ START connects to first nodes, nodes connect to END
- ✅ Condition function names are valid

---

## Troubleshooting Config Issues

### "Failed to load config"
- Check file path is correct
- Verify YAML is valid: use YAML linter
- Ensure file exists and is readable

### "Handler not found: handlers::my_func"
- Verify function is `pub async fn`
- Check module path in config matches Rust module structure
- Ensure handler is exported from handlers module

### "Invalid API key"
- Check `echo $OPENAI_API_KEY` to verify env var is set
- Verify correct environment variable name matches config
- If using .env file, ensure it's loaded before running

### "State validation failed"
- Verify handlers set required state fields
- Check handler return type is `Result<State>`
- Ensure state schema matches what handlers create

---

**You now have everything needed to create powerful ErenFlowAI agents! 🚀**
