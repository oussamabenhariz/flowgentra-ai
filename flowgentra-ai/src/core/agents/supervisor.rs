//! Supervisor pattern for multi-agent orchestration.
//!
//! A `Supervisor` manages a set of named agents (each a `StateGraph`) and
//! routes work between them based on a routing function. This enables
//! composable multi-agent systems where a coordinator decides which
//! specialist agent should handle the current task.

use std::collections::HashMap;
use std::sync::Arc;

use crate::core::state::State;
use crate::core::state_graph::{StateGraph, StateGraphError};

/// A supervisor that orchestrates multiple agent graphs.
///
/// # Example
/// ```ignore
/// let research_graph = StateGraph::builder()./* ... */.compile()?;
/// let writing_graph = StateGraph::builder()./* ... */.compile()?;
///
/// let supervisor = Supervisor::new(|state: &DynState| {
///     let task = state.get_string("task").unwrap_or_default();
///     if task.contains("research") { Ok("researcher".to_string()) }
///     else { Ok("writer".to_string()) }
/// })
/// .add_agent("researcher", research_graph)
/// .add_agent("writer", writing_graph)
/// .max_rounds(5);
///
/// let result = supervisor.run(initial_state).await?;
/// ```
pub struct Supervisor<S: State> {
    agents: HashMap<String, Arc<StateGraph<S>>>,
    #[allow(clippy::type_complexity)]
    router: Box<dyn Fn(&S) -> super::super::state_graph::error::Result<String> + Send + Sync>,
    max_rounds: usize,
    finish_marker: String,
}

impl<S: State + Send + Sync + 'static> Supervisor<S> {
    /// Create a new supervisor with a routing function.
    ///
    /// The router inspects the current state and returns the name of the
    /// agent to dispatch to, or `"FINISH"` to stop.
    pub fn new(
        router: impl Fn(&S) -> super::super::state_graph::error::Result<String> + Send + Sync + 'static,
    ) -> Self {
        Self {
            agents: HashMap::new(),
            router: Box::new(router),
            max_rounds: 10,
            finish_marker: "FINISH".to_string(),
        }
    }

    /// Add a named agent graph.
    pub fn add_agent(mut self, name: impl Into<String>, graph: StateGraph<S>) -> Self {
        self.agents.insert(name.into(), Arc::new(graph));
        self
    }

    /// Set the maximum number of routing rounds.
    pub fn max_rounds(mut self, rounds: usize) -> Self {
        self.max_rounds = rounds;
        self
    }

    /// Set the finish marker string (default: "FINISH").
    pub fn finish_marker(mut self, marker: impl Into<String>) -> Self {
        self.finish_marker = marker.into();
        self
    }

    /// Run the supervisor loop.
    ///
    /// Each round: call the router, dispatch to the selected agent,
    /// update state with the result. Repeat until the router returns
    /// the finish marker or max rounds is reached.
    pub async fn run(&self, initial_state: S) -> super::super::state_graph::error::Result<S> {
        let mut state = initial_state;

        for round in 0..self.max_rounds {
            let next_agent = (self.router)(&state)?;

            if next_agent == self.finish_marker {
                tracing::info!(round, "Supervisor finished");
                return Ok(state);
            }

            let agent = self.agents.get(&next_agent).ok_or_else(|| {
                StateGraphError::NodeNotFound(format!(
                    "Supervisor: agent '{}' not found. Available: {:?}",
                    next_agent,
                    self.agents.keys().collect::<Vec<_>>()
                ))
            })?;

            tracing::info!(round, agent = %next_agent, "Supervisor dispatching to agent");
            state = agent.invoke(state).await?;
        }

        Err(StateGraphError::Timeout(format!(
            "Supervisor exceeded max rounds ({})",
            self.max_rounds
        )))
    }

    /// Get the list of registered agent names.
    pub fn agent_names(&self) -> Vec<&str> {
        self.agents.keys().map(|s| s.as_str()).collect()
    }
}
