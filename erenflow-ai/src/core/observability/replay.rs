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
}
