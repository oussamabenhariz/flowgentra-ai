//! Control-flow node wrappers for the state_graph executor: retry and timeout.
//!
//! These wrap an inner [`Node`] and are used by the config→state_graph bridge
//! to port `type: retry` and `type: timeout` config nodes. Semantics match the
//! legacy runtime's `RetryNode`/`TimeoutNode` (bug-for-bug parity).

use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;

use super::error::{Result, StateGraphError};
use super::node::Node;
use crate::core::state::{Context, State};

/// Retries an inner node on error with exponential backoff.
///
/// Matches the legacy `RetryNode`: the first retry waits `backoff_ms`, and each
/// subsequent wait is `previous * multiplier`, capped at `max_backoff_ms`. Up
/// to `max_retries` retries are attempted; the last error propagates.
pub struct RetryNode<S: State> {
    name: String,
    inner: Arc<dyn Node<S>>,
    max_retries: usize,
    backoff_ms: u64,
    multiplier: f32,
    max_backoff_ms: u64,
}

impl<S: State> RetryNode<S> {
    pub fn new(name: impl Into<String>, inner: Arc<dyn Node<S>>) -> Self {
        Self {
            name: name.into(),
            inner,
            max_retries: 3,
            backoff_ms: 100,
            multiplier: 2.0,
            max_backoff_ms: 30_000,
        }
    }

    pub fn with_max_retries(mut self, n: usize) -> Self {
        self.max_retries = n;
        self
    }
    pub fn with_backoff_ms(mut self, ms: u64) -> Self {
        self.backoff_ms = ms;
        self
    }
    pub fn with_multiplier(mut self, m: f32) -> Self {
        self.multiplier = m;
        self
    }
    pub fn with_max_backoff_ms(mut self, ms: u64) -> Self {
        self.max_backoff_ms = ms;
        self
    }
}

#[async_trait]
impl<S: State + Send + Sync + 'static> Node<S> for RetryNode<S> {
    async fn execute(&self, state: &S, ctx: &Context) -> Result<S::Update> {
        let mut attempt = 0usize;
        let mut backoff = self.backoff_ms;
        loop {
            match self.inner.execute(state, ctx).await {
                Ok(update) => return Ok(update),
                Err(e) => {
                    if attempt >= self.max_retries {
                        return Err(e);
                    }
                    tracing::warn!(
                        node = %self.name,
                        attempt = attempt + 1,
                        max = self.max_retries,
                        "retrying after error: {e}"
                    );
                    if backoff > 0 {
                        tokio::time::sleep(Duration::from_millis(backoff)).await;
                    }
                    attempt += 1;
                    backoff = ((backoff as f32 * self.multiplier) as u64).min(self.max_backoff_ms);
                }
            }
        }
    }

    fn name(&self) -> &str {
        &self.name
    }
}

/// What to do when the inner node exceeds the timeout.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum OnTimeout {
    /// Return a `StateGraphError::Timeout`.
    Error,
    /// Return an empty update (state unchanged), continuing execution.
    Skip,
}

/// Runs an inner node under a wall-clock timeout.
///
/// Matches the legacy `TimeoutNode` for the `error` and `skip` actions. The
/// `default_value` action is not representable generically (it needs a target
/// field in an untyped state) and is treated as `skip`; use a handler for that.
pub struct TimeoutNode<S: State> {
    name: String,
    inner: Arc<dyn Node<S>>,
    timeout: Duration,
    on_timeout: OnTimeout,
}

impl<S: State> TimeoutNode<S> {
    pub fn new(name: impl Into<String>, inner: Arc<dyn Node<S>>, timeout_ms: u64) -> Self {
        Self {
            name: name.into(),
            inner,
            timeout: Duration::from_millis(timeout_ms),
            on_timeout: OnTimeout::Error,
        }
    }

    pub fn with_on_timeout(mut self, action: OnTimeout) -> Self {
        self.on_timeout = action;
        self
    }
}

#[async_trait]
impl<S: State + Send + Sync + 'static> Node<S> for TimeoutNode<S> {
    async fn execute(&self, state: &S, ctx: &Context) -> Result<S::Update> {
        match tokio::time::timeout(self.timeout, self.inner.execute(state, ctx)).await {
            Ok(result) => result,
            Err(_) => match self.on_timeout {
                OnTimeout::Error => Err(StateGraphError::Timeout(format!(
                    "node '{}' exceeded {} ms",
                    self.name,
                    self.timeout.as_millis()
                ))),
                OnTimeout::Skip => {
                    tracing::warn!(node = %self.name, "timed out; skipping (state unchanged)");
                    Ok(S::Update::default())
                }
            },
        }
    }

    fn name(&self) -> &str {
        &self.name
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::llm::Message;
    use crate::core::state_graph::message_graph::{MessageState, MessageStateUpdate};
    use crate::core::state_graph::node::FunctionNode;
    use std::sync::atomic::{AtomicUsize, Ordering};

    fn fail_n_times(n: usize, counter: Arc<AtomicUsize>) -> Arc<dyn Node<MessageState>> {
        Arc::new(FunctionNode::new(
            "flaky",
            move |_s: &MessageState, _c: &Context| {
                let counter = Arc::clone(&counter);
                Box::pin(async move {
                    let attempt = counter.fetch_add(1, Ordering::Relaxed);
                    if attempt < n {
                        Err(StateGraphError::ExecutionError {
                            node: "flaky".into(),
                            reason: format!("fail {attempt}"),
                        })
                    } else {
                        Ok(MessageStateUpdate {
                            messages: Some(vec![Message::assistant("ok")]),
                        })
                    }
                })
            },
        ))
    }

    #[tokio::test]
    async fn retry_succeeds_within_budget() {
        let calls = Arc::new(AtomicUsize::new(0));
        let node = RetryNode::new("r", fail_n_times(2, Arc::clone(&calls)))
            .with_max_retries(3)
            .with_backoff_ms(0);
        let update = node
            .execute(&MessageState::empty(), &Context::new())
            .await
            .unwrap();
        assert!(update.messages.is_some());
        assert_eq!(calls.load(Ordering::Relaxed), 3); // 2 fails + 1 success
    }

    #[tokio::test]
    async fn retry_exhausts_and_propagates() {
        let calls = Arc::new(AtomicUsize::new(0));
        let node = RetryNode::new("r", fail_n_times(10, Arc::clone(&calls)))
            .with_max_retries(2)
            .with_backoff_ms(0);
        let err = node
            .execute(&MessageState::empty(), &Context::new())
            .await
            .unwrap_err();
        assert!(err.to_string().contains("fail"));
        assert_eq!(calls.load(Ordering::Relaxed), 3); // 1 initial + 2 retries
    }

    #[tokio::test]
    async fn timeout_error_action() {
        let slow: Arc<dyn Node<MessageState>> = Arc::new(FunctionNode::new(
            "slow",
            |_s: &MessageState, _c: &Context| {
                Box::pin(async move {
                    tokio::time::sleep(Duration::from_millis(200)).await;
                    Ok(MessageStateUpdate::default())
                })
            },
        ));
        let node = TimeoutNode::new("t", slow, 30);
        let err = node
            .execute(&MessageState::empty(), &Context::new())
            .await
            .unwrap_err();
        assert!(matches!(err, StateGraphError::Timeout(_)));
    }

    #[tokio::test]
    async fn timeout_skip_action_returns_empty() {
        let slow: Arc<dyn Node<MessageState>> = Arc::new(FunctionNode::new(
            "slow",
            |_s: &MessageState, _c: &Context| {
                Box::pin(async move {
                    tokio::time::sleep(Duration::from_millis(200)).await;
                    Ok(MessageStateUpdate {
                        messages: Some(vec![Message::assistant("late")]),
                    })
                })
            },
        ));
        let node = TimeoutNode::new("t", slow, 30).with_on_timeout(OnTimeout::Skip);
        let update = node
            .execute(&MessageState::empty(), &Context::new())
            .await
            .unwrap();
        assert!(update.messages.is_none()); // empty update
    }
}
