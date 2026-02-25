//! # Execution Debugging Module
//!
//! Provides debugging utilities for tracking and analyzing agent execution flow.
//!
//! ## Features
//!
//! - **Execution snapshots** - Capture state at each node
//! - **Timing analysis** - Measure performance bottlenecks
//! - **Path tracking** - See which edges were taken
//! - **Variable tracking** - Monitor state changes
//! - **Error context** - Rich error information

use crate::core::state::State;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Debug configuration for execution
#[derive(Debug, Clone, Default)]
pub struct DebugConfig {
    /// Enable execution snapshots at each node
    pub capture_snapshots: bool,
    /// Enable timing measurements
    pub measure_timing: bool,
    /// Enable path tracking
    pub track_paths: bool,
    /// Maximum snapshot size in bytes (0 = unlimited)
    pub max_snapshot_size: usize,
}

impl DebugConfig {
    /// Create a basic debug config with snapshots and timing
    pub fn basic() -> Self {
        Self {
            capture_snapshots: true,
            measure_timing: true,
            track_paths: true,
            max_snapshot_size: 1_000_000,
        }
    }

    /// Create with all debugging disabled
    pub fn disabled() -> Self {
        Self::default()
    }
}

/// Snapshot of execution state at a specific point
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionSnapshot {
    /// Node name where snapshot was taken
    pub node: String,
    /// Current execution step number
    pub step: usize,
    /// Timestamp of snapshot
    pub timestamp: String,
    /// State at this point (may be truncated for large states)
    pub state: Option<String>,
    /// Size of actual state in bytes
    pub state_size_bytes: usize,
}

/// Information about an edge taken during execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PathSegment {
    /// Source node
    pub from: String,
    /// Target node
    pub to: String,
    /// Condition that was evaluated (if any)
    pub condition: Option<String>,
    /// Whether condition was satisfied
    pub satisfied: bool,
    /// Time taken (in milliseconds)
    pub duration_ms: u64,
}

/// Debug information for an execution run
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionDebugInfo {
    /// Snapshots captured during execution
    pub snapshots: Vec<ExecutionSnapshot>,
    /// Path segments taken
    pub paths: Vec<PathSegment>,
    /// Node execution timings (node_name -> duration_ms)
    pub node_timings: HashMap<String, u64>,
    /// Total execution time in milliseconds
    pub total_time_ms: u64,
    /// Whether execution completed successfully
    pub success: bool,
    /// Error message if execution failed
    pub error: Option<String>,
}

impl ExecutionDebugInfo {
    /// Create new debug info
    pub fn new() -> Self {
        Self {
            snapshots: Vec::new(),
            paths: Vec::new(),
            node_timings: HashMap::new(),
            total_time_ms: 0,
            success: true,
            error: None,
        }
    }

    /// Record a snapshot
    pub fn record_snapshot(&mut self, node: &str, step: usize, state: &State) {
        let state_value = state.to_value();
        let state_size = serde_json::to_string(&state_value)
            .map(|s| s.len())
            .unwrap_or(0);

        let state_str = if state_size < 10_000 {
            serde_json::to_string_pretty(&state_value).ok()
        } else {
            None
        };

        self.snapshots.push(ExecutionSnapshot {
            node: node.to_string(),
            step,
            timestamp: chrono::Utc::now().to_rfc3339(),
            state: state_str,
            state_size_bytes: state_size,
        });
    }

    /// Record a path segment
    pub fn record_path(&mut self, from: &str, to: &str, condition: Option<&str>, duration_ms: u64) {
        self.paths.push(PathSegment {
            from: from.to_string(),
            to: to.to_string(),
            condition: condition.map(|s| s.to_string()),
            satisfied: true,
            duration_ms,
        });
    }

    /// Record node timing
    pub fn record_node_timing(&mut self, node: &str, duration_ms: u64) {
        self.node_timings.insert(node.to_string(), duration_ms);
    }

    /// Get slowest node
    pub fn slowest_node(&self) -> Option<(String, u64)> {
        self.node_timings
            .iter()
            .max_by_key(|(_, &duration)| duration)
            .map(|(node, &duration)| (node.clone(), duration))
    }

    /// Get execution path as string
    pub fn path_as_string(&self) -> String {
        let mut result = String::from("START");
        for segment in &self.paths {
            result.push_str(" -> ");
            result.push_str(&segment.to);
        }
        result
    }

    /// Get summary of execution
    pub fn summary(&self) -> String {
        format!(
            "Execution Summary:\n  Status: {}\n  Total Time: {}ms\n  Nodes: {}\n  Path: {}",
            if self.success { "SUCCESS" } else { "FAILED" },
            self.total_time_ms,
            self.node_timings.len(),
            self.path_as_string()
        )
    }
}

impl Default for ExecutionDebugInfo {
    fn default() -> Self {
        Self::new()
    }
}

/// Builder for debug configurations
pub struct DebugConfigBuilder {
    capture_snapshots: bool,
    measure_timing: bool,
    track_paths: bool,
    max_snapshot_size: usize,
}

impl DebugConfigBuilder {
    /// Create new builder
    pub fn new() -> Self {
        Self {
            capture_snapshots: false,
            measure_timing: false,
            track_paths: false,
            max_snapshot_size: 0,
        }
    }

    /// Enable snapshot capture
    pub fn with_snapshots(mut self) -> Self {
        self.capture_snapshots = true;
        self
    }

    /// Enable timing measurements
    pub fn with_timing(mut self) -> Self {
        self.measure_timing = true;
        self
    }

    /// Enable path tracking
    pub fn with_paths(mut self) -> Self {
        self.track_paths = true;
        self
    }

    /// Set max snapshot size
    pub fn max_snapshot_size(mut self, size: usize) -> Self {
        self.max_snapshot_size = size;
        self
    }

    /// Build config
    pub fn build(self) -> DebugConfig {
        DebugConfig {
            capture_snapshots: self.capture_snapshots,
            measure_timing: self.measure_timing,
            track_paths: self.track_paths,
            max_snapshot_size: self.max_snapshot_size,
        }
    }
}

impl Default for DebugConfigBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn debug_config_builder() {
        let config = DebugConfigBuilder::new()
            .with_snapshots()
            .with_timing()
            .build();

        assert!(config.capture_snapshots);
        assert!(config.measure_timing);
        assert!(!config.track_paths);
    }

    #[test]
    fn execution_debug_info_path() {
        let mut debug = ExecutionDebugInfo::new();
        debug.record_path("START", "process", None, 10);
        debug.record_path("process", "END", None, 5);

        let path = debug.path_as_string();
        assert!(path.contains("process"));
        assert!(path.contains("END"));
    }

    #[test]
    fn execution_debug_slowest_node() {
        let mut debug = ExecutionDebugInfo::new();
        debug.record_node_timing("node_a", 100);
        debug.record_node_timing("node_b", 50);
        debug.record_node_timing("node_c", 150);

        let (name, duration) = debug.slowest_node().unwrap();
        assert_eq!(name, "node_c");
        assert_eq!(duration, 150);
    }
}
