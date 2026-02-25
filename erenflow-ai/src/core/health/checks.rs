//! # Health Check and Monitoring
//!
//! Provides health checking and monitoring capabilities for agents.
//!
//! ## Features
//!
//! - **Agent health status** - Track overall agent health
//! - **Component checks** - Individual checks for each component
//! - **Availability tracking** - Monitor uptime
//! - **Metrics collection** - Gather performance metrics

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};

/// Health status enumeration
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum HealthStatus {
    /// Completely healthy
    Healthy,
    /// Degraded but functional
    Degraded,
    /// Unhealthy but starting
    Starting,
    /// Completely unhealthy
    Unhealthy,
}

impl HealthStatus {
    /// Check if status is ok (healthy or degraded)
    pub fn is_ok(&self) -> bool {
        matches!(self, HealthStatus::Healthy | HealthStatus::Degraded)
    }
}

impl std::fmt::Display for HealthStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            HealthStatus::Healthy => write!(f, "healthy"),
            HealthStatus::Degraded => write!(f, "degraded"),
            HealthStatus::Starting => write!(f, "starting"),
            HealthStatus::Unhealthy => write!(f, "unhealthy"),
        }
    }
}

/// Health check result for a component
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComponentHealth {
    /// Component name
    pub name: String,
    /// Current status
    pub status: HealthStatus,
    /// Message providing details
    pub message: String,
    /// Response time in milliseconds
    pub response_time_ms: u64,
    /// Timestamp of last check
    pub last_checked: u64,
}

impl ComponentHealth {
    /// Create new component health
    pub fn new(name: impl Into<String>, status: HealthStatus) -> Self {
        Self {
            name: name.into(),
            status,
            message: String::new(),
            response_time_ms: 0,
            last_checked: current_timestamp(),
        }
    }

    /// Add message
    pub fn with_message(mut self, msg: impl Into<String>) -> Self {
        self.message = msg.into();
        self
    }

    /// Set response time
    pub fn with_response_time(mut self, ms: u64) -> Self {
        self.response_time_ms = ms;
        self
    }
}

/// Overall agent health status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentHealth {
    /// Overall status
    pub status: HealthStatus,
    /// Component-level health checks
    pub components: HashMap<String, ComponentHealth>,
    /// Uptime in seconds
    pub uptime_seconds: u64,
    /// Total requests processed
    pub total_requests: u64,
    /// Total errors occurred
    pub total_errors: u64,
    /// Average response time in milliseconds
    pub avg_response_time_ms: f64,
    /// Last updated timestamp
    pub last_updated: u64,
}

impl AgentHealth {
    /// Create new agent health
    pub fn new() -> Self {
        Self {
            status: HealthStatus::Starting,
            components: HashMap::new(),
            uptime_seconds: 0,
            total_requests: 0,
            total_errors: 0,
            avg_response_time_ms: 0.0,
            last_updated: current_timestamp(),
        }
    }

    /// Add component health check
    pub fn add_component(&mut self, component: ComponentHealth) {
        self.components.insert(component.name.clone(), component);
    }

    /// Update overall status based on components
    pub fn update_status(&mut self) {
        let all_healthy = self
            .components
            .values()
            .all(|c| c.status == HealthStatus::Healthy);
        let any_unhealthy = self
            .components
            .values()
            .any(|c| c.status == HealthStatus::Unhealthy);

        self.status = if any_unhealthy {
            HealthStatus::Unhealthy
        } else if all_healthy {
            HealthStatus::Healthy
        } else {
            HealthStatus::Degraded
        };

        self.last_updated = current_timestamp();
    }

    /// Record successful request
    pub fn record_request(&mut self, response_time_ms: u64) {
        self.total_requests += 1;
        self.avg_response_time_ms = (self.avg_response_time_ms
            * (self.total_requests as f64 - 1.0)
            + response_time_ms as f64)
            / self.total_requests as f64;
    }

    /// Record error
    pub fn record_error(&mut self) {
        self.total_requests += 1;
        self.total_errors += 1;
    }

    /// Get health percentage (0-100)
    pub fn health_percentage(&self) -> u32 {
        if self.total_requests == 0 {
            return 100;
        }
        let success_rate = 1.0 - (self.total_errors as f64 / self.total_requests as f64);
        (success_rate * 100.0) as u32
    }

    /// Check if agent is healthy
    pub fn is_healthy(&self) -> bool {
        self.status.is_ok()
    }
}

impl Default for AgentHealth {
    fn default() -> Self {
        Self::new()
    }
}

/// Get current timestamp in seconds
fn current_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn health_status_ok() {
        assert!(HealthStatus::Healthy.is_ok());
        assert!(HealthStatus::Degraded.is_ok());
        assert!(!HealthStatus::Unhealthy.is_ok());
    }

    #[test]
    fn component_health_creation() {
        let comp = ComponentHealth::new("llm", HealthStatus::Healthy)
            .with_message("OpenAI API responding normally")
            .with_response_time(150);

        assert_eq!(comp.name, "llm");
        assert_eq!(comp.status, HealthStatus::Healthy);
        assert_eq!(comp.response_time_ms, 150);
    }

    #[test]
    fn agent_health_tracking() {
        let mut health = AgentHealth::new();
        health.record_request(100);
        health.record_request(150);
        health.record_error();

        assert_eq!(health.total_requests, 3);
        assert_eq!(health.total_errors, 1);
        assert_eq!(health.health_percentage(), 66);
    }

    #[test]
    fn agent_health_status_update() {
        let mut health = AgentHealth::new();
        health.add_component(ComponentHealth::new("llm", HealthStatus::Healthy));
        health.add_component(ComponentHealth::new("mcp", HealthStatus::Unhealthy));
        health.update_status();

        assert_eq!(health.status, HealthStatus::Unhealthy);
    }
}
