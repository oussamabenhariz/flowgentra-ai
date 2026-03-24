//! # Execution Tracing
//!

use super::types::{NodeTiming, PathSegment, TokenUsage};
use chrono::Utc;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Complete execution trace for replay and analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionTrace {
    pub trace_id: String,
    pub agent_name: Option<String>,
    pub start_time: chrono::DateTime<Utc>,
    pub end_time: Option<chrono::DateTime<Utc>>,
    pub status: TraceStatus,
    pub node_timings: Vec<NodeTiming>,
    pub path_segments: Vec<PathSegment>,
    pub token_usage: TokenUsage,
    pub metadata: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TraceStatus {
    Completed,
    Failed { node: String, error: String },
}

impl ExecutionTrace {
    pub fn new(agent_name: Option<String>) -> Self {
        Self {
            trace_id: uuid::Uuid::new_v4().to_string(),
            agent_name,
            start_time: Utc::now(),
            end_time: None,
            status: TraceStatus::Completed,
            node_timings: Vec::new(),
            path_segments: Vec::new(),
            token_usage: TokenUsage::default(),
            metadata: HashMap::new(),
        }
    }

    pub fn record_node(
        &mut self,
        node_name: &str,
        duration_ms: u64,
        start_time: chrono::DateTime<Utc>,
    ) {
        self.node_timings.push(NodeTiming {
            node_name: node_name.to_string(),
            duration_ms,
            start_time,
            state_snapshot: None,
        });
    }

    /// Record a node execution with a state snapshot for replay.
    pub fn record_node_with_state(
        &mut self,
        node_name: &str,
        duration_ms: u64,
        start_time: chrono::DateTime<Utc>,
        state_snapshot: serde_json::Value,
    ) {
        self.node_timings.push(NodeTiming {
            node_name: node_name.to_string(),
            duration_ms,
            start_time,
            state_snapshot: Some(state_snapshot),
        });
    }

    pub fn record_path(&mut self, from: &str, to: &str, condition: Option<&str>, duration_ms: u64) {
        self.path_segments.push(PathSegment {
            from: from.to_string(),
            to: to.to_string(),
            condition: condition.map(String::from),
            duration_ms,
        });
    }

    pub fn add_token_usage(&mut self, usage: &TokenUsage) {
        self.token_usage = self.token_usage.add(usage);
    }

    pub fn mark_failed(&mut self, node: impl Into<String>, error: impl Into<String>) {
        self.status = TraceStatus::Failed {
            node: node.into(),
            error: error.into(),
        };
    }

    pub fn complete(&mut self) {
        self.end_time = Some(Utc::now());
    }

    pub fn total_duration_ms(&self) -> Option<u64> {
        self.end_time.map(|end| {
            end.signed_duration_since(self.start_time)
                .num_milliseconds()
                .max(0) as u64
        })
    }

    pub fn execution_path(&self) -> Vec<String> {
        let mut path = vec!["START".to_string()];
        if !self.path_segments.is_empty() {
            for seg in &self.path_segments {
                if seg.to != "END" {
                    path.push(seg.to.clone());
                }
            }
        } else {
            for t in &self.node_timings {
                path.push(t.node_name.clone());
            }
        }
        path
    }

    /// Serialize trace for replay
    pub fn to_json(&self) -> serde_json::Result<String> {
        serde_json::to_string_pretty(self)
    }

    /// Deserialize trace from JSON (for replay)
    pub fn from_json(s: &str) -> serde_json::Result<Self> {
        serde_json::from_str(s)
    }
}
