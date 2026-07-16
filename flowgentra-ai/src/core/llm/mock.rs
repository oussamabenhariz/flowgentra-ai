//! Scripted mock LLM for offline, deterministic tests.
//!
//! Implements the [`LLM`] trait with responses you script up front, so tests
//! that exercise LLM-driven paths (planner routing, agents, evaluation) run in
//! CI with no network or credentials.
//!
//! ```no_run
//! use flowgentra_ai::core::llm::{MockLLM, Message, LLM};
//!
//! # async fn demo() {
//! // Fixed reply to everything:
//! let llm = MockLLM::always("hello");
//! let reply = llm.chat(vec![Message::user("hi")]).await.unwrap();
//! assert_eq!(reply.content, "hello");
//!
//! // Reply chosen by a substring in the latest user message:
//! let llm = MockLLM::new()
//!     .when_contains("weather", "It is sunny")
//!     .when_contains("time", "It is noon")
//!     .otherwise("I don't know");
//! # }
//! ```

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use super::{Message, TokenUsage, ToolDefinition, LLM};
use crate::core::error::Result;

type Matcher = Arc<dyn Fn(&[Message]) -> Option<String> + Send + Sync>;

/// A deterministic, scripted [`LLM`] implementation for tests.
#[derive(Clone)]
pub struct MockLLM {
    matchers: Vec<Matcher>,
    default: String,
    /// Fixed sequence of replies returned in order, ignoring input, when set.
    sequence: Option<Arc<Vec<String>>>,
    call_index: Arc<AtomicUsize>,
    /// Reports usage on chat_with_usage (prompt+completion token estimates).
    report_usage: bool,
}

impl Default for MockLLM {
    fn default() -> Self {
        Self::new()
    }
}

impl MockLLM {
    /// Empty mock; falls back to an empty string until `otherwise`/matchers are set.
    pub fn new() -> Self {
        Self {
            matchers: Vec::new(),
            default: String::new(),
            sequence: None,
            call_index: Arc::new(AtomicUsize::new(0)),
            report_usage: false,
        }
    }

    /// Always return the same reply, regardless of input.
    pub fn always(reply: impl Into<String>) -> Self {
        Self::new().otherwise(reply)
    }

    /// Return `replies` in order, one per call (repeats the last once exhausted).
    pub fn sequence(replies: Vec<impl Into<String>>) -> Self {
        let seq: Vec<String> = replies.into_iter().map(Into::into).collect();
        Self {
            sequence: Some(Arc::new(seq)),
            ..Self::new()
        }
    }

    /// Reply with `reply` when the latest user message contains `needle`.
    pub fn when_contains(mut self, needle: impl Into<String>, reply: impl Into<String>) -> Self {
        let needle = needle.into();
        let reply = reply.into();
        self.matchers.push(Arc::new(move |msgs| {
            let latest = msgs.iter().rev().find(|m| m.content.contains(&needle));
            latest.map(|_| reply.clone())
        }));
        self
    }

    /// Reply chosen by a custom predicate over the full message history.
    pub fn when(
        mut self,
        f: impl Fn(&[Message]) -> Option<String> + Send + Sync + 'static,
    ) -> Self {
        self.matchers.push(Arc::new(f));
        self
    }

    /// Fallback reply when no matcher fires.
    pub fn otherwise(mut self, reply: impl Into<String>) -> Self {
        self.default = reply.into();
        self
    }

    /// Report token usage from `chat_with_usage` (rough word-count estimate).
    pub fn with_usage(mut self) -> Self {
        self.report_usage = true;
        self
    }

    /// Number of times `chat*` has been invoked.
    pub fn call_count(&self) -> usize {
        self.call_index.load(Ordering::Relaxed)
    }

    fn resolve(&self, messages: &[Message]) -> String {
        let n = self.call_index.fetch_add(1, Ordering::Relaxed);
        if let Some(seq) = &self.sequence {
            if seq.is_empty() {
                return self.default.clone();
            }
            let idx = n.min(seq.len() - 1);
            return seq[idx].clone();
        }
        for m in &self.matchers {
            if let Some(reply) = m(messages) {
                return reply;
            }
        }
        self.default.clone()
    }
}

#[async_trait::async_trait]
impl LLM for MockLLM {
    async fn chat(&self, messages: Vec<Message>) -> Result<Message> {
        Ok(Message::assistant(self.resolve(&messages)))
    }

    async fn chat_with_usage(
        &self,
        messages: Vec<Message>,
    ) -> Result<(Message, Option<TokenUsage>)> {
        let prompt_tokens: u64 = messages
            .iter()
            .map(|m| m.content.split_whitespace().count() as u64)
            .sum();
        let reply = self.resolve(&messages);
        let completion_tokens = reply.split_whitespace().count() as u64;
        let usage = self
            .report_usage
            .then(|| TokenUsage::new(prompt_tokens, completion_tokens));
        Ok((Message::assistant(reply), usage))
    }

    async fn chat_with_tools(
        &self,
        messages: Vec<Message>,
        _tools: &[ToolDefinition],
    ) -> Result<Message> {
        self.chat(messages).await
    }

    async fn chat_stream(
        &self,
        messages: Vec<Message>,
    ) -> Result<tokio::sync::mpsc::Receiver<String>> {
        let reply = self.resolve(&messages);
        let (tx, rx) = tokio::sync::mpsc::channel(16);
        tokio::spawn(async move {
            for word in reply.split_inclusive(' ') {
                if tx.send(word.to_string()).await.is_err() {
                    break;
                }
            }
        });
        Ok(rx)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn always_returns_fixed() {
        let llm = MockLLM::always("fixed");
        assert_eq!(
            llm.chat(vec![Message::user("x")]).await.unwrap().content,
            "fixed"
        );
        assert_eq!(
            llm.chat(vec![Message::user("y")]).await.unwrap().content,
            "fixed"
        );
        assert_eq!(llm.call_count(), 2);
    }

    #[tokio::test]
    async fn when_contains_matches_latest_user_message() {
        let llm = MockLLM::new()
            .when_contains("weather", "sunny")
            .otherwise("dunno");
        assert_eq!(
            llm.chat(vec![Message::user("what's the weather?")])
                .await
                .unwrap()
                .content,
            "sunny"
        );
        assert_eq!(
            llm.chat(vec![Message::user("hello")])
                .await
                .unwrap()
                .content,
            "dunno"
        );
    }

    #[tokio::test]
    async fn sequence_returns_in_order_then_repeats() {
        let llm = MockLLM::sequence(vec!["a", "b"]);
        assert_eq!(llm.chat(vec![]).await.unwrap().content, "a");
        assert_eq!(llm.chat(vec![]).await.unwrap().content, "b");
        assert_eq!(llm.chat(vec![]).await.unwrap().content, "b"); // clamps to last
    }

    #[tokio::test]
    async fn usage_reported_when_enabled() {
        let llm = MockLLM::always("one two three").with_usage();
        let (_, usage) = llm
            .chat_with_usage(vec![Message::user("hi there")])
            .await
            .unwrap();
        let usage = usage.expect("usage reported");
        assert_eq!(usage.prompt_tokens, 2);
        assert_eq!(usage.completion_tokens, 3);
    }

    #[tokio::test]
    async fn stream_emits_words() {
        let llm = MockLLM::always("a b c");
        let mut rx = llm.chat_stream(vec![]).await.unwrap();
        let mut got = String::new();
        while let Some(chunk) = rx.recv().await {
            got.push_str(&chunk);
        }
        assert_eq!(got, "a b c");
    }
}
