//! # Retry LLM Client
//!
//! Wraps any `LLMClient` with exponential backoff retry, circuit breaker,
//! and automatic rate-limit handling.

use std::sync::Arc;
use std::time::Duration;

use super::{LLMClient, Message, TokenUsage, ToolDefinition};
use crate::core::error::{FlowgentraError, Result};

/// Circuit breaker state for tracking consecutive failures.
struct CircuitBreakerState {
    consecutive_failures: u32,
    opened_at: Option<std::time::Instant>,
}

/// LLM client wrapper that adds:
/// - Retry with exponential backoff (rate-limit and transport errors)
/// - Circuit breaker (opens after N consecutive failures, resets after cooldown)
///
/// # Example
/// ```ignore
/// use flowgentra_ai::core::llm::{RetryLLMClient, create_llm_client};
///
/// let base_client = create_llm_client(&config)?;
/// let client = RetryLLMClient::new(base_client, 3); // 3 retries
/// ```
pub struct RetryLLMClient {
    inner: Arc<dyn LLMClient>,
    max_retries: u32,
    /// Base delay for exponential backoff (doubles each attempt).
    base_delay: Duration,
    /// Circuit breaker state.
    circuit: tokio::sync::Mutex<CircuitBreakerState>,
    /// Number of consecutive failures before the circuit opens.
    circuit_threshold: u32,
    /// How long the circuit stays open before allowing a probe request.
    circuit_cooldown: Duration,
}

impl RetryLLMClient {
    /// Wrap an existing LLM client with retry logic.
    pub fn new(inner: Arc<dyn LLMClient>, max_retries: u32) -> Self {
        Self {
            inner,
            max_retries,
            base_delay: Duration::from_millis(500),
            circuit: tokio::sync::Mutex::new(CircuitBreakerState {
                consecutive_failures: 0,
                opened_at: None,
            }),
            circuit_threshold: 5,
            circuit_cooldown: Duration::from_secs(30),
        }
    }

    /// Create with custom settings.
    pub fn with_settings(
        inner: Arc<dyn LLMClient>,
        max_retries: u32,
        base_delay: Duration,
        circuit_threshold: u32,
        circuit_cooldown: Duration,
    ) -> Self {
        Self {
            inner,
            max_retries,
            base_delay,
            circuit: tokio::sync::Mutex::new(CircuitBreakerState {
                consecutive_failures: 0,
                opened_at: None,
            }),
            circuit_threshold,
            circuit_cooldown,
        }
    }

    /// Check circuit breaker state. Returns Err if the circuit is open.
    async fn check_circuit(&self) -> Result<()> {
        let mut cb = self.circuit.lock().await;
        if let Some(opened_at) = cb.opened_at {
            if opened_at.elapsed() < self.circuit_cooldown {
                return Err(FlowgentraError::LLMError(
                    "Circuit breaker is open — LLM provider appears down".into(),
                ));
            }
            // Cooldown elapsed — allow one probe (half-open)
            tracing::info!("LLM circuit breaker half-open, allowing probe request");
            cb.opened_at = None;
            cb.consecutive_failures = 0;
        }
        Ok(())
    }

    async fn record_success(&self) {
        let mut cb = self.circuit.lock().await;
        cb.consecutive_failures = 0;
        cb.opened_at = None;
    }

    async fn record_failure(&self) {
        let mut cb = self.circuit.lock().await;
        cb.consecutive_failures += 1;
        if cb.consecutive_failures >= self.circuit_threshold {
            tracing::error!(
                failures = cb.consecutive_failures,
                "LLM circuit breaker opened after {} consecutive failures",
                cb.consecutive_failures
            );
            cb.opened_at = Some(std::time::Instant::now());
        }
    }

    /// Execute an async operation with exponential backoff retry.
    async fn with_retry<F, Fut, T>(&self, op_name: &str, f: F) -> Result<T>
    where
        F: Fn() -> Fut,
        Fut: std::future::Future<Output = Result<T>>,
    {
        self.check_circuit().await?;

        let mut last_err = None;
        for attempt in 0..=self.max_retries {
            match f().await {
                Ok(val) => {
                    self.record_success().await;
                    return Ok(val);
                }
                Err(e) => {
                    if !is_llm_retryable(&e) || attempt == self.max_retries {
                        self.record_failure().await;
                        return Err(e);
                    }
                    last_err = Some(e);
                    let delay = self.base_delay * 2u32.pow(attempt);
                    tracing::warn!(
                        op = %op_name,
                        attempt = attempt + 1,
                        max = self.max_retries,
                        delay_ms = delay.as_millis() as u64,
                        "LLM call failed, retrying"
                    );
                    tokio::time::sleep(delay).await;
                }
            }
        }
        self.record_failure().await;
        Err(last_err.unwrap())
    }
}

/// Determine if an LLM error is safe to retry.
fn is_llm_retryable(err: &FlowgentraError) -> bool {
    match err {
        FlowgentraError::LLMError(msg) => {
            // Rate limit (429), server errors (5xx), timeouts, connection failures.
            // Status codes are matched as standalone numbers to avoid false positives
            // from model names like "gpt-3500" or error IDs like "ERR_5001".
            let lower = msg.to_lowercase();
            http_status_present(&lower, 429)
                || http_status_present(&lower, 500)
                || http_status_present(&lower, 502)
                || http_status_present(&lower, 503)
                || http_status_present(&lower, 504)
                || lower.contains("rate limit")
                || lower.contains("too many requests")
                || lower.contains("server error")
                || lower.contains("timeout")
                || lower.contains("timed out")
                || lower.contains("connection")
                || lower.contains("request failed")
        }
        FlowgentraError::TimeoutError | FlowgentraError::ExecutionTimeout(_) => true,
        _ => false,
    }
}

/// Returns `true` if `msg` contains `code` as a standalone number
/// (not embedded inside a longer digit sequence such as a model name or ID).
fn http_status_present(msg: &str, code: u16) -> bool {
    let code_str = code.to_string();
    let code_bytes = code_str.as_bytes();
    let code_len = code_bytes.len();
    let bytes = msg.as_bytes();

    if bytes.len() < code_len {
        return false;
    }

    for i in 0..=(bytes.len() - code_len) {
        if &bytes[i..i + code_len] == code_bytes {
            let before_ok = i == 0 || !bytes[i - 1].is_ascii_digit();
            let after_ok = i + code_len >= bytes.len() || !bytes[i + code_len].is_ascii_digit();
            if before_ok && after_ok {
                return true;
            }
        }
    }
    false
}

#[async_trait::async_trait]
impl LLMClient for RetryLLMClient {
    async fn chat(&self, messages: Vec<Message>) -> Result<Message> {
        let inner = self.inner.clone();
        let msgs = messages.clone();
        self.with_retry("chat", move || {
            let inner = inner.clone();
            let msgs = msgs.clone();
            async move { inner.chat(msgs).await }
        })
        .await
    }

    async fn chat_with_usage(
        &self,
        messages: Vec<Message>,
    ) -> Result<(Message, Option<TokenUsage>)> {
        let inner = self.inner.clone();
        let msgs = messages.clone();
        self.with_retry("chat_with_usage", move || {
            let inner = inner.clone();
            let msgs = msgs.clone();
            async move { inner.chat_with_usage(msgs).await }
        })
        .await
    }

    async fn chat_with_tools(
        &self,
        messages: Vec<Message>,
        tools: &[ToolDefinition],
    ) -> Result<Message> {
        let inner = self.inner.clone();
        let msgs = messages.clone();
        let tools = tools.to_vec();
        self.with_retry("chat_with_tools", move || {
            let inner = inner.clone();
            let msgs = msgs.clone();
            let tools = tools.clone();
            async move { inner.chat_with_tools(msgs, &tools).await }
        })
        .await
    }

    async fn chat_stream(
        &self,
        messages: Vec<Message>,
    ) -> Result<tokio::sync::mpsc::Receiver<String>> {
        // Streaming can't be retried mid-stream; retry the initial connection only
        let inner = self.inner.clone();
        let msgs = messages.clone();
        self.with_retry("chat_stream", move || {
            let inner = inner.clone();
            let msgs = msgs.clone();
            async move { inner.chat_stream(msgs).await }
        })
        .await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_retryable_detection() {
        assert!(is_llm_retryable(&FlowgentraError::LLMError(
            "OpenAI API error: 429 Too Many Requests".into()
        )));
        assert!(is_llm_retryable(&FlowgentraError::LLMError(
            "OpenAI API request failed: connection refused".into()
        )));
        assert!(is_llm_retryable(&FlowgentraError::LLMError(
            "Anthropic API error: 503 Service Unavailable".into()
        )));
        assert!(!is_llm_retryable(&FlowgentraError::LLMError(
            "Invalid API key".into()
        )));
        assert!(!is_llm_retryable(&FlowgentraError::ConfigError(
            "bad config".into()
        )));
    }

    #[test]
    fn test_http_status_no_false_positive_from_model_name() {
        // "gpt-3500" contains "500" but it is embedded in a longer digit sequence
        assert!(!http_status_present("model gpt-3500 not found", 500));
        // "ERR_5001" also embeds "500"
        assert!(!http_status_present("err_5001 occurred", 500));
        // Standalone "500" should still match
        assert!(http_status_present("http 500 internal server error", 500));
        assert!(http_status_present("status: 500", 500));
        // 429 at end of string
        assert!(http_status_present("rate limited: 429", 429));
        // Surrounded by non-digit punctuation
        assert!(http_status_present("(503)", 503));
    }
}
