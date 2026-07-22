//! `Command` — a single object to resume an interrupted run.
//!
//! Unifies the three things a caller may want to do when resuming a graph
//! paused by [`interrupt`](super::error::interrupt) or an `interrupt_before`/
//! `interrupt_after` breakpoint:
//!
//! - **`resume`** — hand a value to the paused node. The node reads it back
//!   via [`Context::resume_value`](crate::core::state::Context::resume_value)
//!   instead of you having to know which state field to overwrite.
//! - **`update`** — merge a partial state update before resuming (equivalent
//!   to [`StateGraph::resume_with_update`](super::executor::StateGraph::resume_with_update)).
//! - **`goto`** — resume at an arbitrary node instead of the checkpoint's
//!   natural successor(s).
//!
//! All three are optional and composable. Mirrors LangGraph's
//! `Command(resume=, update=, goto=)`.
//!
//! # Example
//!
//! ```ignore
//! use flowgentra_ai::core::state_graph::Command;
//!
//! // Hand the human's answer to the paused node — it reads ctx.resume_value().
//! let state = graph.resume_with_command(thread_id, Command::resume(json!("yes"))).await?;
//!
//! // Jump straight to a different node on resume.
//! let state = graph
//!     .resume_with_command(thread_id, Command::default().with_goto("cleanup"))
//!     .await?;
//! ```

use crate::core::state::State;
use serde_json::Value;

/// Resume instructions for [`StateGraph::resume_with_command`](super::executor::StateGraph::resume_with_command).
///
/// Construct with [`Command::resume`], [`Command::update`], or
/// [`Command::default`], then chain `.with_*` to combine them.
pub struct Command<S: State> {
    /// Value handed to the paused node via `Context::resume_value()`.
    pub resume: Option<Value>,
    /// Partial state update merged into the checkpointed state before resuming.
    pub update: Option<S::Update>,
    /// Node to resume at, overriding the checkpoint's computed successor(s).
    pub goto: Option<String>,
}

impl<S: State> Command<S> {
    /// An empty command — equivalent to plain [`StateGraph::resume`](super::executor::StateGraph::resume).
    pub fn new() -> Self {
        Self {
            resume: None,
            update: None,
            goto: None,
        }
    }

    /// Resume, handing `value` to the paused node via `ctx.resume_value()`.
    ///
    /// Only reaches the node that was executing when the run paused (single-node
    /// resume target); it is not injected into a parallel superstep's branches.
    pub fn resume(value: impl Into<Value>) -> Self {
        Self {
            resume: Some(value.into()),
            update: None,
            goto: None,
        }
    }

    /// Resume, merging `update` into the checkpointed state first.
    pub fn update(update: S::Update) -> Self {
        Self {
            resume: None,
            update: Some(update),
            goto: None,
        }
    }

    /// Attach a state update (builder form).
    pub fn with_update(mut self, update: S::Update) -> Self {
        self.update = Some(update);
        self
    }

    /// Attach a resume value (builder form).
    pub fn with_resume(mut self, value: impl Into<Value>) -> Self {
        self.resume = Some(value.into());
        self
    }

    /// Resume execution at `node` instead of the checkpoint's natural successor(s).
    pub fn with_goto(mut self, node: impl Into<String>) -> Self {
        self.goto = Some(node.into());
        self
    }
}

impl<S: State> Default for Command<S> {
    fn default() -> Self {
        Self::new()
    }
}
