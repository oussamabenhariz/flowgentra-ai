# Evaluation and Self-Correction Guide

Automatically grade agent output and retry when quality is low.

## How It Works

```
Agent generates response
  |
  v
System scores the output (0.0 - 1.0)
  |
  v
Score >= threshold? --YES--> Return response
  |
  NO
  v
Retries left? --YES--> Retry with feedback ("improve your answer")
  |
  NO
  v
Return best attempt so far
```

---

## Setup

### Basic

```yaml
evaluation:
  enabled: true
  min_confidence: 0.8    # Retry if score < 80%
  max_retries: 3
```

### Full Configuration

```yaml
evaluation:
  enabled: true

  min_confidence: 0.8
  max_retries: 3

  scoring:
    metrics: [relevance, completeness, accuracy]
    weights: [0.5, 0.3, 0.2]    # Must sum to 1.0

  grading:
    enabled: true
    rubric: "Is the answer correct, complete, and well-written?"

  retry_policy: exponential    # or linear, fixed
  retry_delay_ms: 500
```

---

## Scoring Metrics

| Metric | Measures | Use For |
|--------|----------|---------|
| `relevance` | Does the answer match the question? | Q&A, search |
| `completeness` | Is the answer thorough? | Reports, analysis |
| `accuracy` | Is it factually correct? | Knowledge tasks |
| `clarity` | Is it well-written and clear? | Content generation |
| `safety` | Is it appropriate? | Public-facing output |

### Weight Profiles

| Profile | Weights | Best For |
|---------|---------|----------|
| Relevance-focused | `[0.7, 0.2, 0.1]` | Search, Q&A |
| Balanced | `[0.4, 0.3, 0.3]` | General purpose |
| Accuracy-focused | `[0.2, 0.2, 0.6]` | Factual tasks |

---

## Threshold Guidelines

| Use Case | Suggested Threshold |
|----------|-------------------|
| Quick answers | 0.6 |
| Customer-facing Q&A | 0.8 |
| Content generation | 0.85 |
| Critical/compliance | 0.95 |

Higher thresholds mean more retries and better quality, but higher latency and cost.

---

## How Evaluation Integrates

Evaluation runs as middleware -- it wraps your handler output transparently:

```rust
pub async fn generate_answer(mut state: State) -> Result<State> {
    let answer = llm.generate("...").await?;
    state.set("answer", json!(answer));
    Ok(state)
    // Evaluation middleware automatically:
    // 1. Scores the output
    // 2. Retries if below threshold
    // 3. Returns only when confident (or max retries reached)
}
```

You don't need to write scoring logic yourself. The middleware handles it based on your YAML config.

---

## Best Practices

1. **Start with a moderate threshold** -- 0.8 is a good default
2. **Keep max_retries reasonable** -- 3 retries balances quality vs cost
3. **Tune weights for your task** -- relevance matters most for Q&A, accuracy for factual tasks
4. **Monitor retry rates** -- high retry rates mean your prompts need improvement
5. **Use exponential retry delay** -- avoids hammering the LLM on rate limits

---

See [FEATURES.md](../FEATURES.md) for the complete feature list.
