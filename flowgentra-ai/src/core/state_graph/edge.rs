//! Edge definitions for state graph connectivity

use super::error::Result;
use crate::core::state::State;
use std::fmt;

/// A fixed edge always routes to a specific node
#[derive(Debug, Clone)]
pub struct FixedEdge {
    pub from: String,
    pub to: String,
}

impl FixedEdge {
    pub fn new(from: impl Into<String>, to: impl Into<String>) -> Self {
        Self {
            from: from.into(),
            to: to.into(),
        }
    }
}

/// Special node markers
pub const START: &str = "__start__";
pub const END: &str = "__end__";

/// Async router function type.
pub type AsyncRouterFn<S> = Box<
    dyn Fn(&S) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<String>> + Send>>
        + Send
        + Sync,
>;

/// Edge definition in the graph
#[allow(clippy::type_complexity)]
pub enum Edge<S: State> {
    /// Fixed edge that always takes the same path
    Fixed(FixedEdge),

    /// Conditional edge that routes based on state (sync)
    Conditional {
        from: String,
        router: Box<dyn Fn(&S) -> Result<String> + Send + Sync>,
    },

    /// Conditional edge with an async router
    AsyncConditional {
        from: String,
        router: AsyncRouterFn<S>,
    },
}

impl<S: State> fmt::Debug for Edge<S> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Edge::Fixed(e) => f.debug_tuple("Edge::Fixed").field(e).finish(),
            Edge::Conditional { from, .. } => f
                .debug_struct("Edge::Conditional")
                .field("from", from)
                .field("router", &"<router_fn>")
                .finish(),
            Edge::AsyncConditional { from, .. } => f
                .debug_struct("Edge::AsyncConditional")
                .field("from", from)
                .field("router", &"<async_router_fn>")
                .finish(),
        }
    }
}

impl<S: State> Edge<S> {
    pub fn fixed(from: impl Into<String>, to: impl Into<String>) -> Self {
        Edge::Fixed(FixedEdge::new(from, to))
    }

    #[allow(clippy::type_complexity)]
    pub fn conditional(
        from: impl Into<String>,
        router: Box<dyn Fn(&S) -> Result<String> + Send + Sync>,
    ) -> Self {
        Edge::Conditional {
            from: from.into(),
            router,
        }
    }

    pub fn async_conditional(from: impl Into<String>, router: AsyncRouterFn<S>) -> Self {
        Edge::AsyncConditional {
            from: from.into(),
            router,
        }
    }

    pub fn from(&self) -> &str {
        match self {
            Edge::Fixed(e) => &e.from,
            Edge::Conditional { from, .. } => from,
            Edge::AsyncConditional { from, .. } => from,
        }
    }

    /// Determine the next node given current state
    pub async fn next_node(&self, state: &S) -> Result<String> {
        match self {
            Edge::Fixed(e) => Ok(e.to.clone()),
            Edge::Conditional { router, .. } => router(state),
            Edge::AsyncConditional { router, .. } => router(state).await,
        }
    }
}
