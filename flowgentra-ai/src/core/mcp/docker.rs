//! Docker Protocol Handler for MCP
//!
//! The actual Docker MCP client (`DockerMCPClient`) lives in `mod.rs` and
//! manages containers via the `docker` CLI. This module is reserved for
//! future Docker API integration (e.g. via bollard).
//!
//! The types below are kept minimal and unused — they exist only so the
//! public re-export in `mod.rs` doesn't break downstream code that may
//! reference them by name.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Docker configuration for MCP container (placeholder — not used by DockerMCPClient).
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct DockerConfig {
    pub image: String,
    pub container_name: Option<String>,
    pub port_mappings: HashMap<u16, u16>,
    pub environment: HashMap<String, String>,
    pub volumes: HashMap<String, String>,
    pub working_dir: Option<String>,
    pub command: Option<Vec<String>>,
    pub network: Option<String>,
}

/// Placeholder — not used by DockerMCPClient.
pub struct DockerConnection {
    pub config: DockerConfig,
}

impl DockerConnection {
    pub fn new(config: DockerConfig) -> Self {
        Self { config }
    }
}

/// Container execution status
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum ContainerState {
    Created,
    Running,
    Paused,
    Stopped,
    Exited,
    Removed,
}

/// Container status information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContainerStatus {
    pub container_id: String,
    pub state: ContainerState,
    pub image: String,
    pub created_at: String,
    pub ports: HashMap<u16, u16>,
}

/// Builder for DockerConnection (placeholder).
pub struct DockerConnectionBuilder {
    config: DockerConfig,
}

impl DockerConnectionBuilder {
    pub fn new(image: impl Into<String>) -> Self {
        Self {
            config: DockerConfig {
                image: image.into(),
                ..Default::default()
            },
        }
    }

    pub fn build(self) -> DockerConnection {
        DockerConnection::new(self.config)
    }
}
