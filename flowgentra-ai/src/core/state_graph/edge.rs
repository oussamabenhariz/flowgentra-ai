//! Edge definitions for state graph connectivity

use crate::core::state::State;
use super::error::Result;
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

/// Edge definition in the graph
pub enum Edge<S: State> {
    /// Fixed edge that always takes the same path
    Fixed(FixedEdge),
    
    /// Conditional edge that routes based on state
    Conditional {
        from: String,
        router: Box<dyn Fn(&S) -> Result<String> + Send + Sync>,
    },
}

impl<S: State> fmt::Debug for Edge<S> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Edge::Fixed(e) => f.debug_tuple("Edge::Fixed").field(e).finish(),
            Edge::Conditional { from, .. } => {
                f.debug_struct("Edge::Conditional")
                    .field("from", from)
                    .field("router", &"<router_fn>")
                    .finish()
            }
        }
    }
}

impl<S: State> Edge<S> {
    pub fn fixed(from: impl Into<String>, to: impl Into<String>) -> Self {
        Edge::Fixed(FixedEdge::new(from, to))
    }

    pub fn conditional(
        from: impl Into<String>,
        router: Box<dyn Fn(&S) -> Result<String> + Send + Sync>,
    ) -> Self {
        Edge::Conditional {
            from: from.into(),
            router,
        }
    }

    pub fn from(&self) -> &str {
        match self {
            Edge::Fixed(e) => &e.from,
            Edge::Conditional { from, .. } => from,
        }
    }

    /// Determine the next node given current state
    pub async fn next_node(&self, state: &S) -> Result<String> {
        match self {
            Edge::Fixed(e) => Ok(e.to.clone()),
            Edge::Conditional { router, .. } => router(state),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::state::PlainState;

    #[test]
    fn test_fixed_edge() {
        let edge = Edge::<PlainState>::fixed("node1", "node2");
        assert_eq!(edge.from(), "node1");
    }
}
