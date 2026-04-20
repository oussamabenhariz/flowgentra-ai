//! # Evaluation Node
//!
//! Iteratively evaluates and refines node outputs until reaching quality confidence thresholds.
//!
//! ## Features
//! - **Automatic Retry**: Executes handler up to max_retries times
//! - **Confidence-Based**: Stops when score >= min_confidence
//! - **Field-Specific**: Evaluates any state field via field_state parameter
//! - **Feedback Loop**: Provides feedback for handler improvement
//! - **Metadata Tracking**: Records all attempts and scores
//!
//! ## Example (YAML)
//! ```yaml
//! - name: refine_content
//!   type: evaluation
//!   handler: refine_handler
//!   timeout: 20
//!   config:
//!     field_state: llm_output
//!     min_confidence: 0.80
//!     max_retries: 3
//!     rubric: "Is the content clear and well-structured?"
//! ```
//!
//! The evaluation node directly references a `#[register_handler]` function.
//! It calls the handler in a loop, scoring the output each time, until
//! confidence >= min_confidence or max_retries is reached.

use crate::core::error::{FlowgentraError, Result};
use crate::core::node::nodes_trait::PluggableNode;
use crate::core::state::DynState;
use crate::core::NodeOutput;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime};
use tracing::{info, warn};

/// An async scorer: receives `(output, attempt)`, returns `(score 0.0–1.0, feedback)`.
///
/// Build one with the factory helpers in this module:
/// - [`default_scorer`] — built-in heuristic (no setup needed)
/// - [`scorer_from_sync`] — wrap any synchronous `Fn(&Value, u32) -> (f64, String)`
/// - [`scorer_from_node_scorer`] — use the built-in [`NodeScorer`](crate::core::evaluation::NodeScorer)
/// - [`scorer_from_confidence`] — use the built-in [`ConfidenceScorer`](crate::core::evaluation::ConfidenceScorer)
/// - [`scorer_from_llm_grader`] — use the built-in [`LLMGrader`](crate::core::evaluation::LLMGrader)
/// - [`scorer_combine`] — weighted mix of multiple scorers
pub type ScorerFn =
    Arc<dyn Fn(Value, u32) -> futures::future::BoxFuture<'static, (f64, String)> + Send + Sync>;

/// Configuration for an evaluation node (from YAML or programmatic)
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EvaluationNodeConfig {
    /// Node identifier
    pub name: String,

    /// Handler function name (resolved at runtime from #[register_handler])
    pub handler: String,

    /// State field to evaluate/refine (e.g., "llm_output")
    #[serde(alias = "field_state")]
    pub field_state: Option<String>,

    /// Minimum confidence threshold (0.0-1.0)
    #[serde(default = "default_min_confidence")]
    pub min_confidence: f64,

    /// Maximum retry attempts
    #[serde(default = "default_max_retries")]
    pub max_retries: u32,

    /// Evaluation rubric/guidance
    #[serde(default)]
    pub rubric: Option<String>,

    /// Optional node-specific configuration
    #[serde(default)]
    pub config: HashMap<String, Value>,
}

fn default_min_confidence() -> f64 {
    0.80
}

fn default_max_retries() -> u32 {
    3
}

impl EvaluationNodeConfig {
    /// Validate configuration
    pub fn validate(config: &EvaluationNodeConfig) -> Result<()> {
        if config.name.is_empty() {
            return Err(FlowgentraError::ConfigError(
                "Node name cannot be empty".to_string(),
            ));
        }
        if config.handler.is_empty() {
            return Err(FlowgentraError::ConfigError(
                "Handler name cannot be empty".to_string(),
            ));
        }
        if config.min_confidence < 0.0 || config.min_confidence > 1.0 {
            return Err(FlowgentraError::ConfigError(format!(
                "min_confidence must be 0.0-1.0, got {}",
                config.min_confidence
            )));
        }
        Ok(())
    }

    /// Build an EvaluationNodeConfig from a NodeConfig
    ///
    /// Extracts evaluation-specific fields (field_state, min_confidence, max_retries, rubric)
    /// from the node's config map.
    pub fn from_node_config(node: &crate::core::node::NodeConfig) -> Result<Self> {
        let field_state = node
            .config
            .get("field_state")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());

        let min_confidence = node
            .config
            .get("min_confidence")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.80);

        let max_retries = node
            .config
            .get("max_retries")
            .and_then(|v| v.as_u64())
            .unwrap_or(3) as u32;

        let rubric = node
            .config
            .get("rubric")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());

        let config = Self {
            name: node.name.clone(),
            handler: node.handler.clone(),
            field_state,
            min_confidence,
            max_retries,
            rubric,
            config: node.config.clone(),
        };

        Self::validate(&config)?;
        Ok(config)
    }

    /// Get effective field name (handles aliases)
    pub fn get_field_name(&self) -> Option<String> {
        self.field_state.clone().or_else(|| {
            self.config
                .get("field_state")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string())
        })
    }

    // ── NodeFunction builders ─────────────────────────────────────────────────
    //
    // All code paths (YAML config, programmatic add_node) go through these.
    // The `_with_scorer` variants are the canonical implementations.
    // The plain variants use the built-in heuristic scorer as the default.

    /// Build a `NodeFunction` that scores the current state field on every visit,
    /// using the **built-in heuristic scorer**.
    ///
    /// For a custom scorer use [`into_standalone_node_fn_with_scorer`].
    pub fn into_standalone_node_fn(self) -> super::NodeFunction<DynState> {
        self.into_standalone_node_fn_with_scorer(default_scorer())
    }

    /// Build a `NodeFunction` that scores the current state field on every visit,
    /// using a **custom scorer function**.
    ///
    /// The scorer receives `(output: &Value, attempt: u32)` and returns
    /// `(score: f64, feedback: String)` where `score` must be in `0.0..=1.0`.
    ///
    /// Wire a back-edge conditioned on `__eval_needs_retry__<name>` to create a
    /// graph-level retry loop.
    ///
    /// # Example
    /// ```ignore
    /// use std::sync::Arc;
    ///
    /// let cfg = EvaluationNodeConfig { name: "score".into(), handler: "".into(),
    ///     field_state: Some("output".into()), min_confidence: 0.9,
    ///     max_retries: 5, rubric: None, config: HashMap::new() };
    ///
    /// // Custom scorer: count JSON array length as quality signal
    /// let scorer = Arc::new(|output: &Value, _attempt: u32| -> (f64, String) {
    ///     let len = output.as_array().map(|a| a.len()).unwrap_or(0);
    ///     let score = (len as f64 / 10.0).min(1.0);
    ///     let feedback = if score < 0.5 { "Need more items".into() } else { "Good".into() };
    ///     (score, feedback)
    /// });
    ///
    /// let node = Node::new("score", cfg.into_standalone_node_fn_with_scorer(scorer), vec![], HashMap::new());
    /// agent.runtime_mut().graph.add_node(node);
    /// ```
    pub fn into_standalone_node_fn_with_scorer(
        self,
        scorer: ScorerFn,
    ) -> super::NodeFunction<DynState> {
        Box::new(move |state| {
            let config = self.clone();
            let scorer = scorer.clone();
            Box::pin(async move {
                let field_name = config.get_field_name();

                let output = match &field_name {
                    Some(field) => state.get(field).unwrap_or(Value::Null),
                    None => state
                        .get("output")
                        .or_else(|| state.get("llm_output"))
                        .or_else(|| state.get("result"))
                        .unwrap_or(Value::Null),
                };

                let attempt_key = format!("__eval_attempt__{}", config.name);
                let attempt = state
                    .get(&attempt_key)
                    .and_then(|v| v.as_u64())
                    .unwrap_or(0) as u32
                    + 1;

                let (score, feedback) = scorer(output.clone(), attempt).await;
                let needs_retry = score < config.min_confidence && attempt < config.max_retries;

                state.set(&attempt_key, json!(attempt));
                state.set(format!("__eval_score__{}", config.name), json!(score));
                state.set(
                    format!("__eval_feedback__{}", config.name),
                    json!(&feedback),
                );
                state.set(
                    format!("__eval_needs_retry__{}", config.name),
                    json!(needs_retry),
                );
                state.set(
                    format!("__eval_meta__{}", config.name),
                    json!({
                        "score": score,
                        "attempt": attempt,
                        "max_retries": config.max_retries,
                        "min_confidence": config.min_confidence,
                        "needs_retry": needs_retry,
                        "feedback": feedback,
                        "exit_reason": if score >= config.min_confidence {
                            "HighConfidence"
                        } else if attempt >= config.max_retries {
                            "MaxRetriesReached"
                        } else {
                            "Retrying"
                        },
                    }),
                );

                tracing::info!(
                    "Standalone eval '{}' attempt {}/{}: score={:.2}, needs_retry={}",
                    config.name,
                    attempt,
                    config.max_retries,
                    score,
                    needs_retry
                );

                Ok(state)
            })
        })
    }

    /// Build a `NodeFunction` that calls `inner` in a loop using the **built-in heuristic scorer**.
    ///
    /// For a custom scorer use [`into_wrapping_node_fn_with_scorer`].
    pub fn into_wrapping_node_fn(
        self,
        inner: std::sync::Arc<
            dyn Fn(DynState) -> futures::future::BoxFuture<'static, Result<DynState>> + Send + Sync,
        >,
    ) -> super::NodeFunction<DynState> {
        self.into_wrapping_node_fn_with_scorer(inner, default_scorer())
    }

    /// Build a `NodeFunction` that calls `inner` in a loop, scoring each output
    /// with a **custom scorer**, until `min_confidence` is reached or `max_retries` exhausted.
    ///
    /// The best result (highest score) is kept if max retries are reached.
    /// Feedback from each attempt is injected into state as `__eval_feedback__<name>`
    /// so `inner` can read and act on it.
    ///
    /// # Example
    /// ```ignore
    /// use std::sync::Arc;
    ///
    /// let cfg = EvaluationNodeConfig { name: "refine".into(), handler: "".into(),
    ///     field_state: Some("llm_output".into()), min_confidence: 0.9,
    ///     max_retries: 3, rubric: None, config: HashMap::new() };
    ///
    /// // Custom LLM-based scorer
    /// let scorer: ScorerFn = Arc::new(|output: &Value, attempt: u32| -> (f64, String) {
    ///     let text = output.as_str().unwrap_or("");
    ///     // ... call your grader here ...
    ///     (0.95, "Looks great".into())
    /// });
    ///
    /// let inner = Arc::new(my_refine_handler);
    /// let node = Node::new("refine", cfg.into_wrapping_node_fn_with_scorer(inner, scorer), vec![], HashMap::new());
    /// ```
    pub fn into_wrapping_node_fn_with_scorer(
        self,
        inner: std::sync::Arc<
            dyn Fn(DynState) -> futures::future::BoxFuture<'static, Result<DynState>> + Send + Sync,
        >,
        scorer: ScorerFn,
    ) -> super::NodeFunction<DynState> {
        Box::new(move |state| {
            let config = self.clone();
            let inner = inner.clone();
            let scorer = scorer.clone();
            Box::pin(async move {
                let field_name = config.get_field_name();
                let start_time = Instant::now();
                let mut state = state;
                let mut best_score: f64 = 0.0;
                let mut best_state: Option<DynState> = None;
                let mut all_attempts = Vec::new();

                for attempt_num in 1..=config.max_retries {
                    tracing::info!(
                        "Evaluation attempt {}/{} for '{}'",
                        attempt_num,
                        config.max_retries,
                        config.name
                    );

                    state = inner(state).await?;

                    let output = match &field_name {
                        Some(field) => state.get(field).unwrap_or(Value::Null),
                        None => Value::Null,
                    };

                    let (score, feedback) = scorer(output.clone(), attempt_num).await;

                    all_attempts.push(json!({
                        "attempt": attempt_num,
                        "score": score,
                        "feedback": feedback,
                        "duration_ms": start_time.elapsed().as_millis(),
                    }));

                    if score > best_score {
                        best_score = score;
                        best_state = Some(state.clone());
                    }

                    if score >= config.min_confidence {
                        state.set(
                            format!("__eval_meta__{}", config.name),
                            json!({
                                "final_score": score,
                                "attempts": attempt_num,
                                "total_attempts": config.max_retries,
                                "duration_ms": start_time.elapsed().as_millis(),
                                "exit_reason": "HighConfidence",
                                "all_attempts": all_attempts,
                            }),
                        );
                        tracing::info!(
                            "Evaluation '{}' reached confidence {:.2} after {} attempts",
                            config.name,
                            score,
                            attempt_num
                        );
                        return Ok(state);
                    }

                    state.set(format!("__eval_feedback__{}", config.name), json!(feedback));
                    tracing::info!(
                        "Evaluation '{}' attempt {} scored {:.2}, retrying...",
                        config.name,
                        attempt_num,
                        score
                    );
                }

                if let Some(best) = best_state {
                    state = best;
                }

                state.set(
                    format!("__eval_meta__{}", config.name),
                    json!({
                        "final_score": best_score,
                        "attempts": config.max_retries,
                        "total_attempts": config.max_retries,
                        "duration_ms": start_time.elapsed().as_millis(),
                        "exit_reason": "MaxRetriesReached",
                        "all_attempts": all_attempts,
                    }),
                );

                tracing::warn!(
                    "Evaluation '{}' reached max retries with best score {:.2}",
                    config.name,
                    best_score
                );
                Ok(state)
            })
        })
    }
}

// =============================================================================
// Standalone scoring (used by both EvaluationNode and wrap_handler_with_evaluation)
// =============================================================================

/// Score output quality and generate feedback.
///
/// In production, this would call an LLM evaluator. Currently uses
/// heuristic scoring based on output length and complexity.
pub fn evaluate_output_score(output: &Value, attempt: u32) -> (f64, String) {
    let text = match output {
        Value::String(s) => s.clone(),
        other => other.to_string(),
    };

    let length_score = ((text.len() as f64).min(1000.0)) / 1000.0;
    let complexity_score = estimate_text_complexity(&text);
    let base_score = (length_score * 0.4 + complexity_score * 0.6).min(0.99);

    // Slight bonus per attempt (simulating iterative improvement)
    let attempt_bonus = ((attempt as f64) - 1.0) * 0.05;
    let score = (base_score + attempt_bonus).min(0.99);

    let feedback = if score < 0.6 {
        "Output too short or simple. Provide more detailed and comprehensive content."
    } else if score < 0.8 {
        "Good foundation. Add more depth and structured information."
    } else {
        "Output meets quality standards."
    };

    (score, feedback.to_string())
}

fn estimate_text_complexity(text: &str) -> f64 {
    let word_count = text.split_whitespace().count();
    let sentence_count = text.matches(&['.', '!', '?'][..]).count().max(1);
    let avg_sentence_length = word_count as f64 / sentence_count as f64;

    let complexity: f64 = if avg_sentence_length < 5.0 {
        0.3
    } else if avg_sentence_length < 20.0 {
        0.7
    } else {
        0.9
    };

    complexity.min(0.99)
}

// =============================================================================
// Single attempt record
// =============================================================================

/// Single execution attempt record
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Attempt {
    pub number: u32,
    pub output: Value,
    pub score: f64,
    pub feedback: String,
    pub duration: Duration,
    pub timestamp: SystemTime,
}

/// Reason for evaluation completion
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum ExitReason {
    HighConfidence,
    MaxRetriesReached,
    FatalError(String),
}

/// Result of evaluation execution
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EvaluationResult {
    pub success: bool,
    pub final_attempt: Attempt,
    pub all_attempts: Vec<Attempt>,
    pub best_attempt: Attempt,
    pub total_duration: Duration,
    pub exit_reason: ExitReason,
    pub message: String,
}

// =============================================================================
// EvaluationNode (PluggableNode-based, for programmatic use)
// =============================================================================

/// Evaluation node - iteratively evaluates and refines output
pub struct EvaluationNode {
    pub config: EvaluationNodeConfig,
    pub inner_node: Box<dyn PluggableNode<DynState>>,
}

impl EvaluationNode {
    /// Create a new evaluation node with validation
    pub fn new(
        config: EvaluationNodeConfig,
        inner_node: Box<dyn PluggableNode<DynState>>,
    ) -> Result<Self> {
        EvaluationNodeConfig::validate(&config)?;
        Ok(EvaluationNode { config, inner_node })
    }

    /// Create without validation (useful for dynamic wrapping)
    pub fn new_unchecked(
        config: EvaluationNodeConfig,
        inner_node: Box<dyn PluggableNode<DynState>>,
    ) -> Self {
        EvaluationNode { config, inner_node }
    }
}

impl Clone for EvaluationNode {
    fn clone(&self) -> Self {
        EvaluationNode {
            config: self.config.clone(),
            inner_node: self.inner_node.clone_box(),
        }
    }
}

#[async_trait]
impl PluggableNode<DynState> for EvaluationNode {
    fn node_type(&self) -> &'static str {
        "evaluation"
    }

    fn name(&self) -> &str {
        &self.config.name
    }

    fn config(&self) -> &HashMap<String, Value> {
        &self.config.config
    }

    fn clone_box(&self) -> Box<dyn PluggableNode<DynState>> {
        Box::new(self.clone())
    }

    async fn run(&self, state: DynState) -> Result<NodeOutput<DynState>> {
        let config = &self.config;
        let start_time = Instant::now();
        let field_name = config.get_field_name();

        let mut all_attempts = Vec::new();
        let mut best_attempt: Option<Attempt> = None;

        for attempt_num in 1..=config.max_retries {
            info!(
                "Evaluation attempt {}/{} for node '{}'",
                attempt_num, config.max_retries, config.name
            );

            let attempt_start = Instant::now();
            let inner_state = state.clone();

            // Inject previous feedback
            if let Some(ref best) = best_attempt {
                inner_state.set(
                    format!("__eval_feedback__{}", config.name),
                    Value::String(best.feedback.clone()),
                );
            }

            let inner_result = self.inner_node.run(inner_state).await?;

            let inner_state = inner_result.state;
            let output = match field_name.as_ref() {
                Some(field) => inner_state.get(field).unwrap_or(Value::Null),
                None => Value::String(
                    inner_result
                        .metadata
                        .get("output")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string(),
                ),
            };

            let duration = attempt_start.elapsed();
            let (score, feedback) = evaluate_output_score(&output, attempt_num);

            let attempt = Attempt {
                number: attempt_num,
                output: output.clone(),
                score,
                feedback: feedback.clone(),
                duration,
                timestamp: SystemTime::now(),
            };

            all_attempts.push(attempt.clone());

            if best_attempt.is_none() || attempt.score > best_attempt.as_ref().unwrap().score {
                best_attempt = Some(attempt);
            }

            if best_attempt.as_ref().unwrap().score >= config.min_confidence {
                let best = best_attempt.unwrap();

                if let Some(ref field) = field_name {
                    state.set(field.clone(), best.output.clone());
                }

                state.set(
                    format!("__eval_meta__{}", config.name),
                    json!({
                        "final_score": best.score,
                        "attempts": attempt_num,
                        "total_attempts": config.max_retries,
                        "duration_ms": start_time.elapsed().as_millis(),
                        "exit_reason": "HighConfidence",
                        "all_attempts": all_attempts,
                    }),
                );

                info!(
                    "Evaluation node '{}' reached high confidence ({:.2}) after {} attempts",
                    config.name, best.score, attempt_num
                );

                return Ok(NodeOutput::success(state));
            }
        }

        // Max retries reached
        let best = best_attempt.ok_or_else(|| {
            FlowgentraError::RuntimeError("No evaluation attempts completed".to_string())
        })?;

        if let Some(ref field) = field_name {
            state.set(field.clone(), best.output.clone());
        }

        state.set(
            format!("__eval_meta__{}", config.name),
            json!({
                "final_score": best.score,
                "attempts": config.max_retries,
                "total_attempts": config.max_retries,
                "duration_ms": start_time.elapsed().as_millis(),
                "exit_reason": "MaxRetriesReached",
                "all_attempts": all_attempts,
            }),
        );

        warn!(
            "Evaluation node '{}' reached max retries with best score {:.2}",
            config.name, best.score
        );

        Ok(NodeOutput::success(state))
    }
}

// =============================================================================
// ScorerFn factory helpers
//
// These let users build a ScorerFn from the built-in evaluation components
// (or their own logic) without writing async boilerplate.
// =============================================================================

/// The default built-in heuristic scorer (text length + complexity).
/// Used automatically when you call `into_standalone_node_fn` / `into_wrapping_node_fn`.
pub fn default_scorer() -> ScorerFn {
    Arc::new(|output: Value, attempt: u32| {
        Box::pin(async move { evaluate_output_score(&output, attempt) })
    })
}

/// Wrap any synchronous `Fn(&Value, u32) -> (f64, String)` as a `ScorerFn`.
///
/// Use this to keep simple heuristics without writing async boilerplate.
///
/// # Example
/// ```ignore
/// let scorer = scorer_from_sync(|output, _attempt| {
///     let score = if output.as_str().map(|s| s.len()).unwrap_or(0) > 100 { 0.9 } else { 0.4 };
///     (score, "Length check".into())
/// });
/// ```
pub fn scorer_from_sync(
    f: impl Fn(&Value, u32) -> (f64, String) + Send + Sync + 'static,
) -> ScorerFn {
    Arc::new(move |output: Value, attempt: u32| {
        let result = f(&output, attempt);
        Box::pin(async move { result })
    })
}

/// Use the built-in [`NodeScorer`](crate::core::evaluation::NodeScorer) as a `ScorerFn`.
///
/// Scores on: completeness, validity, usefulness, and (if history is in state) consistency.
///
/// # Example
/// ```ignore
/// use flowgentra_ai::core::evaluation::ScoringCriteria;
///
/// let scorer = scorer_from_node_scorer(ScoringCriteria {
///     min_length: 50,
///     check_consistency: false,
///     ..Default::default()
/// });
/// cfg.into_standalone_node_fn_with_scorer::<DynState>(scorer)
/// ```
pub fn scorer_from_node_scorer(criteria: crate::core::evaluation::ScoringCriteria) -> ScorerFn {
    Arc::new(move |output: Value, _attempt: u32| {
        let criteria = criteria.clone();
        Box::pin(async move {
            use crate::core::evaluation::NodeScorer;
            // Consistency check needs state history; use a fresh state as fallback.
            let dummy = DynState::new();
            let score = NodeScorer::score(&output, &criteria, &dummy, "");
            (score.overall, score.explanation)
        })
    })
}

/// Use the built-in [`ConfidenceScorer`](crate::core::evaluation::ConfidenceScorer) as a `ScorerFn`.
///
/// Scores on: clarity, relevance (optional task hint), and completeness.
///
/// # Example
/// ```ignore
/// use flowgentra_ai::core::evaluation::ConfidenceConfig;
///
/// let scorer = scorer_from_confidence(
///     ConfidenceConfig { high_threshold: 0.9, ..Default::default() },
///     Some("Summarise the article in 3 bullet points".into()),
/// );
/// ```
pub fn scorer_from_confidence(
    config: crate::core::evaluation::ConfidenceConfig,
    task: Option<String>,
) -> ScorerFn {
    Arc::new(move |output: Value, _attempt: u32| {
        let config = config.clone();
        let task = task.clone();
        Box::pin(async move {
            use crate::core::evaluation::ConfidenceScorer;
            let dummy = DynState::new();
            let score = ConfidenceScorer::score(&output, task.as_deref(), &dummy, "", &config);
            let feedback = format!(
                "clarity={:.2} relevance={:.2} completeness={:.2} ({})",
                score.clarity,
                score.relevance,
                score.completeness,
                score.indicators.join(", "),
            );
            (score.overall, feedback)
        })
    })
}

/// Use the built-in [`LLMGrader`](crate::core::evaluation::LLMGrader) as a `ScorerFn`.
///
/// Makes an async LLM call to evaluate the output against `task_description`.
/// On LLM error, returns score `0.5` with the error message as feedback so the
/// pipeline continues rather than panicking.
///
/// # Example
/// ```ignore
/// use std::sync::Arc;
///
/// // llm comes from agent.llm() or create_llm(...)
/// let scorer = scorer_from_llm_grader(
///     "Summarise the article in 3 bullet points".into(),
///     "The user wants a concise summary".into(),
///     llm.clone(),
/// );
/// cfg.into_wrapping_node_fn_with_scorer(inner_handler, scorer)
/// ```
pub fn scorer_from_llm_grader(
    task_description: String,
    context: String,
    llm: Arc<dyn crate::core::llm::LLM>,
) -> ScorerFn {
    Arc::new(move |output: Value, _attempt: u32| {
        let task = task_description.clone();
        let ctx = context.clone();
        let llm = llm.clone();
        Box::pin(async move {
            use crate::core::evaluation::LLMGrader;
            match LLMGrader::grade(&output, &task, &ctx, llm).await {
                Ok(grade) => (grade.score, grade.feedback),
                Err(e) => (0.5, format!("LLM grader error: {e}")),
            }
        })
    })
}

/// Combine multiple scorers into one using weighted averaging.
///
/// `scorers` is a list of `(weight, ScorerFn)` pairs.  Weights do not need to
/// sum to 1.0 — they are normalised automatically.  Feedback messages from all
/// scorers are joined with `" | "`.
///
/// # Example
/// ```ignore
/// let scorer = scorer_combine(vec![
///     (0.4, scorer_from_node_scorer(ScoringCriteria::default())),
///     (0.6, scorer_from_llm_grader(task, ctx, llm)),
/// ]);
/// ```
pub fn scorer_combine(scorers: Vec<(f64, ScorerFn)>) -> ScorerFn {
    Arc::new(move |output: Value, attempt: u32| {
        let scorers = scorers.clone();
        let output = output.clone();
        Box::pin(async move {
            let mut total_weight = 0.0_f64;
            let mut weighted_score = 0.0_f64;
            let mut feedbacks = Vec::new();

            for (weight, scorer) in &scorers {
                let (score, feedback) = scorer(output.clone(), attempt).await;
                weighted_score += weight * score;
                total_weight += weight;
                feedbacks.push(format!("[w={weight:.1}] {feedback}"));
            }

            let final_score = if total_weight > 0.0 {
                weighted_score / total_weight
            } else {
                0.0
            };

            (final_score, feedbacks.join(" | "))
        })
    })
}
