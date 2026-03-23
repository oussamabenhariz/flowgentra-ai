# LLM Configuration Guide

Set up any language model provider you prefer - OpenAI, Anthropic, Mistral, Groq, Azure, or even run Ollama locally.

## Supported Providers

- OpenAI (GPT-4, GPT-3.5-turbo)
- Anthropic (Claude)
- Mistral
- Groq
- Azure OpenAI
- HuggingFace (cloud API & local TGI server)
- Ollama (local models)

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

Set environment variable:
```bash
export OPENAI_API_KEY="sk-..."
```

### Anthropic (Claude)

```yaml
llm:
  provider: anthropic
  model: claude-3-opus-20240229
  api_key: ${ANTHROPIC_API_KEY}
  temperature: 0.7
```

### Mistral

```yaml
llm:
  provider: mistral
  model: mistral-large
  api_key: ${MISTRAL_API_KEY}
  temperature: 0.7
```

### Groq (Fast inference)

```yaml
llm:
  provider: groq
  model: mixtral-8x7b-32768
  api_key: ${GROQ_API_KEY}
```

### Azure OpenAI

```yaml
llm:
  provider: azure
  model: gpt-4
  api_key: ${AZURE_API_KEY}
  endpoint: ${AZURE_ENDPOINT}
  deployment: my-deployment
```

### HuggingFace (Cloud API)

Use any model from the HuggingFace Model Hub with the cloud API:

```yaml
llm:
  provider: huggingface
  model: mistralai/Mistral-7B-Instruct-v0.1
  api_key: ${HF_API_TOKEN}
```

Get your API token from [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens):

```bash
export HF_API_TOKEN="hf_..."
```

**Available models:**
- Text generation: `mistralai/Mistral-7B-Instruct-v0.1`, `meta-llama/Llama-2-7b-chat`, etc.
- Check [HuggingFace Model Hub](https://huggingface.co/models) for all available models

### HuggingFace Local (Text Generation Inference)

Deploy models locally using HuggingFace Text Generation Inference (TGI):

```bash
# Start TGI server with your model
docker run --gpus all -p 80:80 -v /tmp/data:/data \
    ghcr.io/huggingface/text-generation-inference:latest \
    --model-id mistralai/Mistral-7B-Instruct-v0.1
```

Then configure:

```yaml
llm:
  provider: huggingface
  model: mistralai/Mistral-7B-Instruct-v0.1
  api_key: ""
  extra_params:
    mode: "local"
    endpoint: "http://localhost:80"
```

**Benefits of local deployment:**
- No API costs
- Lower latency
- Full privacy (data stays on your machine)
- Use any HuggingFace model

**System requirements:**
- GPU (NVIDIA recommended with CUDA)
- At least 12GB VRAM for 7B models
- Docker installed

### Ollama (Local)

```yaml
llm:
  provider: ollama
  model: mistral
  base_url: http://localhost:11434
```

## Fallback Providers

Automatically try alternative providers if the main one fails:

```yaml
llm:
  provider: openai
  model: gpt-4
  api_key: ${OPENAI_API_KEY}
  fallbacks:
    - provider: anthropic
      model: claude-3-opus-20240229
      api_key: ${ANTHROPIC_API_KEY}
    - provider: mistral
      model: mistral-large
      api_key: ${MISTRAL_API_KEY}
    - provider: huggingface
      model: mistralai/Mistral-7B-Instruct-v0.1
      api_key: ${HF_API_TOKEN}
```

Flow:
1. Try OpenAI GPT-4
2. If fails → try Claude
3. If fails → try Mistral
4. If fails → try HuggingFace
5. If all fail → error

**Example: Cost-optimized fallback chain**
```yaml
llm:
  provider: groq              # Free tier, very fast
  model: mixtral-8x7b-32768
  api_key: ${GROQ_API_KEY}
  fallbacks:
    - provider: huggingface   # Free cloud API
      model: mistralai/Mistral-7B-Instruct-v0.1
      api_key: ${HF_API_TOKEN}
    - provider: ollama        # Free local (if available)
      model: mistral
      base_url: http://localhost:11434
```

## Parameters Explained

**temperature** - Controls randomness in responses
- 0.0: Deterministic (always the same answer)
- 0.7: Balanced (recommended)
- 1.0: Creative (lots of variation)

**max_tokens** - Maximum response length (default: 2000)

**timeout** - Request timeout in seconds (default: 30)

**top_p** - Nucleus sampling (0.0 - 1.0)

## Provider Comparison

| Provider | Cost | Speed | Quality | Setup | Best For |
|----------|------|-------|---------|-------|----------|
| **OpenAI** | Paid | Fast | Excellent | 5 min | Production, high quality |
| **Anthropic** | Paid | Medium | Excellent | 5 min | Long context, safeguard |
| **Mistral** | Paid | Fast | Good | 5 min | Cost-effective production |
| **Groq** | Free tier | Very Fast | Good | 5 min | Fast inference, prototyping |
| **HuggingFace Cloud** | Free | Medium | Good | 5 min | Prototyping, open models |
| **HuggingFace Local** | Free | Fast | Good | 30 min | Privacy, no costs, custom models |
| **Ollama** | Free | Medium | Good | 15 min | Local development |
| **Azure** | Paid | Fast | Excellent | 10 min | Enterprise, private |

### When to Use Each Provider

**OpenAI (GPT-4)**
- When: You need the best possible quality
- Cost: ~$0.03-0.06 per 1K tokens
- Use case: Customer-facing applications, complex reasoning

**HuggingFace (Cloud API)**
- When: Free tier, prototyping, open-source models
- Cost: Free (rate-limited)
- Use case: Early development, testing, open models
- Best models: Mistral, Llama, Zephyr

**HuggingFace (Local TGI)**
- When: Need privacy, zero cost, full control
- Cost: $0 (just GPU/compute)
- Use case: Private deployments, custom models, no-internet
- Requirement: Must have a GPU

**Groq**
- When: Need speed and free tier
- Cost: Free tier + paid
- Use case: Fast inference, real-time applications
- Best for: Latency-critical applications

**Ollama**
- When: Laptop/development environment
- Cost: $0
- Use case: Local development, testing
- Limitation: CPU-only or limited VRAM

## Usage in Code

```rust
let llm = state.get_llm_client()?;
let response = llm.chat(messages).await?;
```

## HuggingFace Advanced Configuration

### Cloud API with Custom Parameters

```yaml
llm:
  provider: huggingface
  model: mistralai/Mistral-7B-Instruct-v0.1
  api_key: ${HF_API_TOKEN}
  temperature: 0.7
  max_tokens: 1024
  top_p: 0.95
```

### Multiple Models (via fallbacks)

```yaml
llm:
  provider: huggingface
  model: mistralai/Mistral-7B-Large      # Larger, slower
  api_key: ${HF_API_TOKEN}
  fallbacks:
    - provider: huggingface
      model: mistralai/Mistral-7B        # Smaller, faster fallback
      api_key: ${HF_API_TOKEN}
```

### Local TGI with Custom Endpoint

```yaml
llm:
  provider: huggingface
  model: meta-llama/Llama-2-7b-chat
  api_key: ""
  extra_params:
    mode: "local"
    endpoint: "http://your-server.com:8080"
```

### Deploying Your Own TGI Server

```bash
# With GPU (CUDA)
docker run --gpus all -p 8080:80 \
  ghcr.io/huggingface/text-generation-inference:latest \
  --model-id meta-llama/Llama-2-7b-chat

# With specific GPU memory
docker run --gpus all -e CUDA_VISIBLE_DEVICES=0 -p 8080:80 \
  ghcr.io/huggingface/text-generation-inference:latest \
  --model-id meta-llama/Llama-2-7b-chat \
  --max-total-tokens 2048

# On CPU (slower)
docker run -p 8080:80 \
  ghcr.io/huggingface/text-generation-inference:latest \
  --model-id mistralai/Mistral-7B-Instruct-v0.1
```

### Recommended Models for Different Use Cases

**Fast & Efficient (7B parameters)**
- `mistralai/Mistral-7B-Instruct-v0.1` - Best balance
- `meta-llama/Llama-2-7b-chat` - Good alternative

**High Quality (13B+ parameters)**
- `mistralai/Mistral-7B-Instruct-v0.2` - Latest version
- `meta-llama/Llama-2-13b-chat` - Larger model

**Specialized Models**
- Code: `codellama/CodeLlama-7b-Instruct-hf`
- Multilingual: `allenai/Llama-2-7b-hf`
- Long context: `mistralai/Mistral-7B-Instruct-v0.1` (can handle 32k)

## Best Practices

1. **Always use environment variables for API keys**
   ```bash
   export OPENAI_API_KEY="sk-..."
   export HF_API_TOKEN="hf_..."
   ```

2. **Pick the right temperature for your task**
   - Factual: 0.0-0.3
   - Balanced: 0.5-0.7
   - Creative: 0.8-1.0

3. **Add fallback providers for reliability**
   ```yaml
   provider: openai
   fallbacks:
     - provider: anthropic
     - provider: huggingface
   ```

4. **Monitor token usage and costs**
   - Use HuggingFace cloud for free prototyping
   - Use local TGI for production to avoid API costs
   - Set reasonable max_tokens limits

5. **HuggingFace-specific tips**
   - **Cloud API**: Great for free/testing, rate-limited, good for small projects
   - **Local TGI**: Perfect for private/sensitive data, zero cost, full control
   - **Choose models wisely**: Smaller models (7B) are faster, larger (13B+) are smarter
   - **Stream responses**: For better UX (available in later versions)
   - **Use fallbacks**: Combine cloud and local for robustness

6. **Test locally first**
   - Develop with Ollama (easy setup)
   - Test with HuggingFace cloud (free)
   - Deploy with HuggingFace local TGI (production-ready)

---

See [configuration/CONFIG_GUIDE.md](../configuration/CONFIG_GUIDE.md) for complete reference.
