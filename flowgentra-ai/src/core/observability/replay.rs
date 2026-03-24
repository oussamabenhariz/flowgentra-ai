//! # Replay Mode
//!
//! Load saved execution traces and step through them for debugging.

use super::trace::ExecutionTrace;

/// Replay controller for stepping through a saved trace
#[derive(Debug, Clone)]
pub struct ReplayMode {
    trace: ExecutionTrace,
    current_step: usize,
}

impl ReplayMode {
    pub fn from_trace(trace: ExecutionTrace) -> Self {
        Self {
            trace,
            current_step: 0,
        }
    }

    pub fn from_json(json: &str) -> serde_json::Result<Self> {
        let trace = ExecutionTrace::from_json(json)?;
        Ok(Self::from_trace(trace))
    }

    pub fn trace(&self) -> &ExecutionTrace {
        &self.trace
    }

    pub fn current_step(&self) -> usize {
        self.current_step
    }

    pub fn total_steps(&self) -> usize {
        self.trace.node_timings.len()
    }

    pub fn step_forward(&mut self) -> bool {
        if self.current_step < self.trace.node_timings.len() {
            self.current_step += 1;
            true
        } else {
            false
        }
    }

    pub fn step_back(&mut self) -> bool {
        if self.current_step > 0 {
            self.current_step -= 1;
            true
        } else {
            false
        }
    }

    pub fn go_to_step(&mut self, step: usize) {
        self.current_step = step.min(self.trace.node_timings.len());
    }

    pub fn current_node(&self) -> Option<&str> {
        self.trace
            .node_timings
            .get(self.current_step)
            .map(|t| t.node_name.as_str())
    }

    pub fn execution_path(&self) -> Vec<String> {
        self.trace.execution_path()
    }

    pub fn is_complete(&self) -> bool {
        self.current_step >= self.trace.node_timings.len()
    }

    /// Get the state snapshot at the current step (if captured).
    pub fn current_state(&self) -> Option<&serde_json::Value> {
        self.trace
            .node_timings
            .get(self.current_step)
            .and_then(|t| t.state_snapshot.as_ref())
    }

    /// Get the state snapshot at a specific step.
    pub fn state_at(&self, step: usize) -> Option<&serde_json::Value> {
        self.trace
            .node_timings
            .get(step)
            .and_then(|t| t.state_snapshot.as_ref())
    }

    /// Compare state between two steps (returns changed keys).
    pub fn diff_states(&self, step_a: usize, step_b: usize) -> Option<Vec<String>> {
        let a = self.state_at(step_a)?;
        let b = self.state_at(step_b)?;
        let a_obj = a.as_object()?;
        let b_obj = b.as_object()?;

        let mut changed = Vec::new();
        for key in a_obj
            .keys()
            .chain(b_obj.keys())
            .collect::<std::collections::HashSet<_>>()
        {
            if a_obj.get(key) != b_obj.get(key) {
                changed.push(key.clone());
            }
        }
        Some(changed)
    }
}
