# Evaluation Guide

Let your agent check its own work - score the output, detect low quality, and retry automatically when needed.

## What Happens

With evaluation enabled:
1. Agent generates a response
2. System scores it automatically
3. If the score is high enough, return it
4. If not, retry with feedback (up to max_retries)
5. Return the best result

## Setup

### Basic Setup

```yaml
evaluation:
  enabled: true
  min_confidence: 0.8      # Retry if < 80%
  max_retries: 3
```

### Full Configuration

```yaml
evaluation:
  enabled: true
  
  # Quality threshold
  min_confidence: 0.8
  max_retries: 3
  
  # What to measure
  scoring:
    metrics: [relevance, completeness, accuracy]
    weights: [0.5, 0.3, 0.2]   # Sum = 1.0
  
  # How to grade
  grading:
    enabled: true
    rubric: "Is the answer correct and helpful?"
  
  # Retry strategy
  retry_policy: "exponential"   # or "linear", "fixed"
  retry_delay_ms: 500
```

## Metrics

Pick what matters for your task:

| Metric | Use For |
|--------|----------|
| relevance | Does answer match the question? |
| completeness | Is it thorough and complete? |
| accuracy | Is it factually correct? |
| clarity | Is it well-written? |
| safety | Is it safe and appropriate? |

## How It Works

```
Agent generates response
    ↓
System scores response
    ↓
Is score >= min_confidence?
    ├─ YES → Return response ✓
    ├─ NO  → Retries < max_retries?
    │           ├─ YES → Try again with feedback
    │           └─ NO  → Return best attempt
```

## In Code

Evaluation happens automatically:

```rust
pub async fn generate_response(mut state: State) -> Result<State> {
    // Your logic
    let response = llm.generate("...").await?;
    state.set("response", json!(response));
    
    // Evaluation middleware:
    // 1. Scores automatically
    // 2. Retries if needed
    // 3. Returns confident response
    
    Ok(state)
}
```

## Configuration Tips

1. **Different thresholds by task**:
   - Q&A: 0.80 confidence
   - Content: 0.85 confidence
   - Critical: 0.95 confidence

2. **Adjust weights for priorities**:
   - Relevance-heavy: [0.7, 0.2, 0.1]
   - Balanced: [0.5, 0.3, 0.2]
   - Accuracy-heavy: [0.2, 0.2, 0.6]

3. **Control retries**:
   - Fast response: max_retries: 1
   - Balanced: max_retries: 3
   - High quality: max_retries: 5

## Monitoring

```rust
// Check evaluation metrics
let metrics = agent.get_evaluation_metrics()?;
println!("Avg confidence: {}", metrics.avg_confidence);
println!("Total retries: {}", metrics.total_retries);
```

---

See [configuration/CONFIG_GUIDE.md](../configuration/CONFIG_GUIDE.md) for complete reference.
