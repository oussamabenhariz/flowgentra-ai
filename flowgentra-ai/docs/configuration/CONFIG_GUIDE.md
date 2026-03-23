````markdown
# FlowgentraAI Config.yaml Complete Guide

## Overview

The `config.yaml` file is the declarative definition of your entire FlowgentraAI agent. It defines:
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
description: "A comprehensive agent showcasing all FlowgentraAI features"
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
      timeout: 20
    
    # Node that uses multiple features
    - name: gather_information
      description: "Gather info using tools"
      handler: handlers::gather_information
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
      uses_rag: true
      timeout: 25
    
    # Final response generation
    - name: generate_response
      description: "Generate final response"
      handler: handlers::generate_response
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
```

See full guide for all options.
````
