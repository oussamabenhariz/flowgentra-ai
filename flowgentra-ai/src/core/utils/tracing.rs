//! # Tracing & Observability Module
//!
//! for the FlowgentraAI runtime. Enables detailed insights into agent execution flow,
//! performance metrics, and debugging information.
//!
//! ## Features
//!
//! - **Structured Logging**: JSON-formatted logs with context spans
//! - **Execution Tracing**: Track node execution, state changes, and performance
//! - **Performance Metrics**: Measure execution time for all operations
//! - **Debug Context**: Automatic context propagation through async operations

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};

/// Initializes the tracing subscriber with structured logging.
///
/// # Example
///
/// ```rust
/// use flowgentra_ai::core::tracing::init_tracing;
///
/// #[tokio::main]
/// async fn main() {
///     init_tracing("info");
///     // All operations now properly logged
/// }
/// ```
pub fn init_tracing(log_level: &str) {
    use tracing_subscriber::{fmt, prelude::*, EnvFilter};

    let env_filter = EnvFilter::try_from_default_env()
        .or_else(|_| EnvFilter::try_new(log_level))
        .unwrap_or_else(|_| EnvFilter::new("info"));

    tracing_subscriber::registry()
        .with(env_filter)
        .with(fmt::layer().with_writer(std::io::stdout).json())
        .init();

    tracing::info!("Tracing initialized at {} level", log_level);
}

/// Execution trace record for a single node execution event.
///
/// Captures timing, state transitions, and contextual information
/// for performance analysis and debugging.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionEvent {
    /// Unique identifier for this event
    pub event_id: String,
    /// Node or operation name
    pub name: String,
    /// Event type (Started, Completed, Error, etc)
    pub event_type: String,
    /// When this event occurred
    pub timestamp: DateTime<Utc>,
    /// Duration for operations that track time
    pub duration_ms: Option<u64>,
    /// Associated error message if applicable
    pub error: Option<String>,
    /// Contextual metadata
    pub metadata: std::collections::HashMap<String, String>,
}

impl ExecutionEvent {
    /// Creates a new execution event.
    pub fn new(name: impl Into<String>, event_type: impl Into<String>) -> Self {
        Self {
            event_id: uuid::Uuid::new_v4().to_string(),
            name: name.into(),
            event_type: event_type.into(),
            timestamp: Utc::now(),
            duration_ms: None,
            error: None,
            metadata: Default::default(),
        }
    }

    /// Records duration for a completed operation.
    pub fn with_duration(mut self, duration: Duration) -> Self {
        self.duration_ms = Some(duration.as_millis() as u64);
        self
    }

    /// Adds an error message.
    pub fn with_error(mut self, error: impl Into<String>) -> Self {
        self.error = Some(error.into());
        self
    }

    /// Adds metadata.
    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }
}

/// Complete execution trace for an agent run.
///
/// Records all events that occurred during agent execution,
/// useful for debugging and performance analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionTrace {
    /// Trace identifier
    pub trace_id: String,
    /// When execution started
    pub start_time: DateTime<Utc>,
    /// When execution finished
    pub end_time: Option<DateTime<Utc>>,
    /// All recorded events
    pub events: Vec<ExecutionEvent>,
    /// Total execution time if completed
    pub total_duration_ms: Option<u64>,
}

impl ExecutionTrace {
    /// Creates a new execution trace.
    pub fn new() -> Self {
        Self {
            trace_id: uuid::Uuid::new_v4().to_string(),
            start_time: Utc::now(),
            end_time: None,
            events: Vec::new(),
            total_duration_ms: None,
        }
    }

    /// Adds an event to the trace.
    pub fn record(&mut self, event: ExecutionEvent) {
        tracing::debug!(
            event_id = %event.event_id,
            event_type = %event.event_type,
            name = %event.name,
            "Recording execution event"
        );
        self.events.push(event);
    }

    /// Marks trace as complete and calculates total duration.
    pub fn complete(&mut self) {
        self.end_time = Some(Utc::now());
        if let Some(end) = self.end_time {
            let duration = end.signed_duration_since(self.start_time);
            self.total_duration_ms = Some(duration.num_milliseconds() as u64);
        }
        tracing::info!(
            trace_id = %self.trace_id,
            event_count = self.events.len(),
            duration_ms = ?self.total_duration_ms,
            "Execution trace completed"
        );
    }

    /// Gets summary statistics for the trace.
    pub fn summary(&self) -> TraceSummary {
        let error_count = self.events.iter().filter(|e| e.error.is_some()).count();
        let total_time: u64 = self.events.iter().filter_map(|e| e.duration_ms).sum();

        TraceSummary {
            event_count: self.events.len(),
            error_count,
            total_duration_ms: total_time,
        }
    }
}

impl Default for ExecutionTrace {
    fn default() -> Self {
        Self::new()
    }
}

/// Summary statistics for an execution trace.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceSummary {
    /// Total number of events recorded
    pub event_count: usize,
    /// Number of error events
    pub error_count: usize,
    /// Total time spent in timed operations
    pub total_duration_ms: u64,
}

/// Timer guard for automatic duration measurement.
///
/// # Example
///
/// ```rust
/// use flowgentra_ai::core::tracing::TimerGuard;
///
/// async fn my_operation() {
///     let _timer = TimerGuard::start("operation");
///     // operation code
/// }
/// ```
pub struct TimerGuard {
    name: String,
    start: Instant,
}

impl TimerGuard {
    /// Starts a new timer.
    pub fn start(name: impl Into<String>) -> Self {
        let name = name.into();
        tracing::trace!(name = %name, "Timer started");
        Self {
            name,
            start: Instant::now(),
        }
    }

    /// Gets elapsed time.
    pub fn elapsed(&self) -> Duration {
        self.start.elapsed()
    }
}

impl Drop for TimerGuard {
    fn drop(&mut self) {
        let elapsed = self.start.elapsed();
        tracing::trace!(
            name = %self.name,
            duration_ms = elapsed.as_millis(),
            "Timer completed"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn execution_event_creation() {
        let event = ExecutionEvent::new("test_node", "Started");
        assert_eq!(event.name, "test_node");
        assert_eq!(event.event_type, "Started");
        assert!(event.error.is_none());
    }

    #[test]
    fn execution_event_with_metadata() {
        let event = ExecutionEvent::new("test", "Started")
            .with_metadata("key1", "value1")
            .with_error("test error");
        assert_eq!(event.metadata.get("key1").unwrap(), "value1");
        assert_eq!(event.error.unwrap(), "test error");
    }

    #[test]
    fn execution_trace_recording() {
        let mut trace = ExecutionTrace::new();
        trace.record(ExecutionEvent::new("node1", "Started"));
        trace.record(ExecutionEvent::new("node2", "Started"));
        trace.complete();

        let summary = trace.summary();
        assert_eq!(summary.event_count, 2);
        assert_eq!(summary.error_count, 0);
    }

    #[test]
    fn timer_guard_measures_time() {
        let timer = TimerGuard::start("test");
        std::thread::sleep(std::time::Duration::from_millis(10));
        let elapsed = timer.elapsed();
        assert!(elapsed.as_millis() >= 10);
    }
}
