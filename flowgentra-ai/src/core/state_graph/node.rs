//! Node definitions for state graph execution

use async_trait::async_trait;

use super::error::Result;
use crate::core::state::{Context, State};

/// A node in the state graph.
///
/// Nodes receive the current state and a framework context, and return
/// a partial update (`S::Update`) indicating which fields changed.
///
/// # Example
///
/// ```ignore
/// use flowgentra_ai::core::state_graph::node::*;
/// use std::sync::Arc;
///
/// let node = Arc::new(FunctionNode::new("greet", |state: &MyState, _ctx: &Context| {
///     Box::pin(async move {
///         Ok(MyStateUpdate::new().greeting(format!("Hello, {}!", state.name)))
///     })
/// }));
/// ```
#[async_trait]
pub trait Node<S: State>: Send + Sync {
    /// Execute this node with the current state and context.
    /// Returns a partial update — only the fields that changed.
    async fn execute(&self, state: &S, ctx: &Context) -> Result<S::Update>;

    /// Name of this node (for debugging and routing)
    fn name(&self) -> &str;

    /// Whether this node can run in parallel with others
    fn is_parallelizable(&self) -> bool {
        true
    }
}

/// A node backed by an async closure that returns a partial update.
pub struct FunctionNode<S: State, F>
where
    F: Fn(
            &S,
            &Context,
        ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<S::Update>> + Send>>
        + Send
        + Sync,
{
    name: String,
    func: F,
    _phantom: std::marker::PhantomData<S>,
}

impl<S: State, F> FunctionNode<S, F>
where
    F: Fn(
            &S,
            &Context,
        ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<S::Update>> + Send>>
        + Send
        + Sync,
{
    pub fn new(name: impl Into<String>, func: F) -> Self {
        Self {
            name: name.into(),
            func,
            _phantom: std::marker::PhantomData,
        }
    }
}

#[async_trait]
impl<S: State, F> Node<S> for FunctionNode<S, F>
where
    F: Fn(
            &S,
            &Context,
        ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<S::Update>> + Send>>
        + Send
        + Sync,
{
    async fn execute(&self, state: &S, ctx: &Context) -> Result<S::Update> {
        (self.func)(state, ctx).await
    }

    fn name(&self) -> &str {
        &self.name
    }
}

/// Router function: given current state, decide which node to go to next
pub type RouterFn<S> = Box<dyn Fn(&S) -> Result<String> + Send + Sync>;

/// Conditional edge definition
pub struct ConditionalEdge<S: State> {
    pub from: String,
    pub router: RouterFn<S>,
}

impl<S: State> std::fmt::Debug for ConditionalEdge<S> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ConditionalEdge")
            .field("from", &self.from)
            .field("router", &"<router_fn>")
            .finish()
    }
}

impl<S: State> ConditionalEdge<S> {
    pub fn new(from: impl Into<String>, router: RouterFn<S>) -> Self {
        Self {
            from: from.into(),
            router,
        }
    }
}
