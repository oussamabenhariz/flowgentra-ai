//! Node definitions for state graph execution

use async_trait::async_trait;

use crate::core::state::State;
use super::error::Result;

/// Represents a partial state update (only the fields that changed)
/// Used when a node doesn't return the full state, just the diff
pub struct StateUpdate<S: State> {
    pub partial_state: S,
    pub merge_strategy: MergeStrategy,
}

impl<S: State> StateUpdate<S> {
    pub fn new(partial_state: S) -> Self {
        Self {
            partial_state,
            merge_strategy: MergeStrategy::Default,
        }
    }

    pub fn with_strategy(partial_state: S, strategy: MergeStrategy) -> Self {
        Self {
            partial_state,
            merge_strategy: strategy,
        }
    }
}

/// How to merge a node's output into the current state
#[derive(Debug, Clone, Copy)]
pub enum MergeStrategy {
    /// Use default reducers from state schema
    Default,
    /// Replace entire state (no merge)
    Replace,
    /// Deep merge (for nested objects)
    Merge,
}

/// A node in the state graph
#[async_trait]
pub trait Node<S: State>: Send + Sync {
    /// Execute this node with the current state
    /// Returns the updated state or an error
    async fn execute(&self, state: &S) -> Result<S>;

    /// Name of this node (for debugging and routing)
    fn name(&self) -> &str;

    /// Optional: whether this node can run in parallel with others
    fn is_parallelizable(&self) -> bool {
        true
    }
}

/// A stateless node backed by an async function
pub struct FunctionNode<S: State, F>
where
    F: Fn(&S) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<S>> + Send>> + Send + Sync,
{
    name: String,
    func: F,
    _phantom: std::marker::PhantomData<S>,
}

impl<S: State, F> FunctionNode<S, F>
where
    F: Fn(&S) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<S>> + Send>> + Send + Sync,
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
    F: Fn(&S) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<S>> + Send>> + Send + Sync + Send,
{
    async fn execute(&self, state: &S) -> Result<S> {
        (self.func)(state).await
    }

    fn name(&self) -> &str {
        &self.name
    }
}

/// A simpler node backed by an async function that returns a state update
pub struct UpdateNode<S: State, F>
where
    F: Fn(&S) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<StateUpdate<S>>> + Send>> + Send + Sync,
{
    name: String,
    func: F,
    _phantom: std::marker::PhantomData<S>,
}

impl<S: State, F> UpdateNode<S, F>
where
    F: Fn(&S) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<StateUpdate<S>>> + Send>> + Send + Sync,
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
impl<S: State, F> Node<S> for UpdateNode<S, F>
where
    F: Fn(&S) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<StateUpdate<S>>> + Send>> + Send + Sync + Send,
{
    async fn execute(&self, state: &S) -> Result<S> {
        let update = (self.func)(state).await?;
        // For now, just merge the update into the state
        // In reality, you'd apply reducers here per the state schema
        let merged_state = state.clone();
        // TODO: Apply merge strategy and per-field reducers
        merged_state.merge(update.partial_state);
        Ok(merged_state)
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

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use crate::core::state::PlainState;

    #[tokio::test]
    async fn test_function_node() {
        let node = FunctionNode::new("test", |state: &PlainState| {
            let cloned = state.clone();
            Box::pin(async move {
                let mut new_state = cloned;
                PlainState::set(&mut new_state, "result", json!("success"));
                Ok(new_state)
            })
        });

        let state = PlainState::new();
        let result = node.execute(&state).await.unwrap();
        assert_eq!(result.get("result"), Some(&json!("success")));
    }
}
