//! # Observability Types
//!
//! Core data structures for execution tracing, metrics, token usage,
//! and failure snapshots.

use crate::core::llm::TokenUsage as LlmTokenUsage;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

pub type TokenUsage = LlmTokenUsage;

/// Node timing metric with optional state snapshot.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeTiming {
    pub node_name: String,
    pub duration_ms: u64,
    pub start_time: DateTime<Utc>,
    /// JSON state snapshot taken after this node executed.
    /// Populated when observability middleware has `capture_state: true`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub state_snapshot: Option<serde_json::Value>,
}

/// Failure snapshot for debugging
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FailureSnapshot {
    pub trace_id: String,
    pub failed_node: String,
    pub error_message: String,
    pub state_snapshot: Option<String>,
    pub execution_path: Vec<String>,
    pub node_timings: HashMap<String, u64>,
    pub token_usage: Option<TokenUsage>,
    pub timestamp: DateTime<Utc>,
}

/// Path segment for DAG execution overlay
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PathSegment {
    pub from: String,
    pub to: String,
    pub condition: Option<String>,
    pub duration_ms: u64,
}
