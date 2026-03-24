//! # State Optimization Utilities
//!
//! Provides utilities to optimize state handling and reduce cloning during execution.
//!
//! ## Strategies
//!
//! 1. **Lazy Clone** - Only clone state when mutations occur
//! 2. **Reference Counting** - Use Arc<Mutex> for shared state
//! 3. **Parallel Execution** - Efficient state distribution to parallel nodes

use crate::core::state::State;
use std::sync::Arc;

/// Lazy-clone state wrapper that avoids cloning until actual mutations
///
/// This implements a simple copy-on-write pattern for state.
/// Great for cases where most nodes only read from state.
pub struct OptimizedState<T: State> {
    inner: Arc<T>,
}

impl<T: State> Clone for OptimizedState<T> {
    fn clone(&self) -> Self {
        OptimizedState {
            inner: Arc::clone(&self.inner),
        }
    }
}

impl<T: State> OptimizedState<T> {
    /// Create a new optimized state from a base state
    pub fn new(state: T) -> OptimizedState<T> {
        OptimizedState {
            inner: Arc::new(state),
        }
    }

    /// Convert to owned state
    pub fn into_owned(self) -> T {
        Arc::try_unwrap(self.inner).unwrap_or_else(|arc| arc.as_ref().clone())
    }

    /// Returns true if this is the only owner of the state
    pub fn is_owned(&self) -> bool {
        Arc::strong_count(&self.inner) == 1
    }

    /// Returns the strong count of the inner Arc
    pub fn strong_count(&self) -> usize {
        Arc::strong_count(&self.inner)
    }
}

impl<T: State> AsRef<T> for OptimizedState<T> {
    fn as_ref(&self) -> &T {
        &self.inner
    }
}

impl<T: State> AsMut<T> for OptimizedState<T> {
    fn as_mut(&mut self) -> &mut T {
        if Arc::strong_count(&self.inner) > 1 {
            // Clone-on-write: make a private copy before mutating
            let state = self.inner.as_ref().clone();
            self.inner = Arc::new(state);
        }
        Arc::get_mut(&mut self.inner).expect("BUG: Arc strong_count should be 1 after CoW clone")
    }
}

/// Statistics for tracking cloning efficiency
#[derive(Debug, Clone, Default)]
pub struct CloneStats {
    /// Number of state clones performed
    pub clone_count: u64,
    /// Number of state mutations
    pub mutation_count: u64,
    /// Estimated bytes cloned (rough estimation)
    pub estimated_bytes_cloned: u64,
}

impl CloneStats {
    /// Calculate efficiency ratio (mutations / clones)
    pub fn efficiency_ratio(&self) -> f64 {
        if self.clone_count == 0 {
            return 0.0;
        }
        self.mutation_count as f64 / self.clone_count as f64
    }

    /// Format nice summary
    pub fn summary(&self) -> String {
        format!(
            "Clones: {}, Mutations: {}, Efficiency: {:.2}%, Est. Data: {} bytes",
            self.clone_count,
            self.mutation_count,
            self.efficiency_ratio() * 100.0,
            self.estimated_bytes_cloned
        )
    }
}

/// Guidelines for when to use different state strategies
pub mod strategies {
    /// Use standard State (with Clone) when:
    /// - Nodes are few and state updates are frequent
    /// - Small JSON payloads (< 1MB)
    /// - Sequential execution (no parallelism needed)
    /// - Default behavior, always safe
    pub struct StandardStateStrategy;

    /// Use SharedState when:
    /// - State is very large (multiple MB of JSON)
    /// - Many nodes read without modifying
    /// - Checkpointing/resumability is critical
    /// - You accept slight overhead from mutex locks
    pub struct SharedStateStrategy;

    /// Use OptimizedState when:
    /// - Mixed read/write patterns
    /// - Want lazy cloning only on mutation
    /// - Performance-sensitive applications
    pub struct OptimizedStateStrategy;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::state::SharedState;

    #[test]
    fn test_optimized_state_no_clone_on_read() {
        let state = SharedState::new(Default::default());
        let opt = OptimizedState::new(state);

        // Reading doesn't require ownership
        let opt2 = opt.clone();
        assert_eq!(opt.strong_count(), 2);
        assert_eq!(opt2.strong_count(), 2);
    }

    #[test]
    fn test_clone_stats_efficiency() {
        let stats = CloneStats {
            clone_count: 10,
            mutation_count: 8,
            estimated_bytes_cloned: 1024,
        };
        assert!(stats.efficiency_ratio() > 0.7);
        assert!(stats.efficiency_ratio() < 0.9);
    }
}
