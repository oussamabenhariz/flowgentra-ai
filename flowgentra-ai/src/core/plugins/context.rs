/// Plugin context providing access to runtime state and utilities
use crate::prelude::*;
use dashmap::DashMap;
use std::sync::Arc;

/// Context passed to plugin lifecycle hooks
///
/// Provides plugins with access to runtime state, graph information,
/// and registration utilities.
#[derive(Clone)]
pub struct PluginContext {
    /// Current execution state (use SharedState or PlainState wrapper)
    pub state: Arc<crate::core::state::SharedState>,
    /// Graph being executed
    pub graph_info: Arc<GraphInfo>,
    /// Custom plugin data storage
    data: Arc<DashMap<String, Arc<dyn std::any::Any + Send + Sync>>>,
}

/// Information about the graph being executed
#[derive(Clone, Debug)]
pub struct GraphInfo {
    pub node_count: usize,
    pub edge_count: usize,
    pub description: String,
}

impl PluginContext {
    /// Create a new plugin context
    pub fn new() -> Self {
        let plain_state = crate::core::state::PlainState::new();
        Self {
            state: Arc::new(crate::core::state::SharedState::new(plain_state)),
            graph_info: Arc::new(GraphInfo {
                node_count: 0,
                edge_count: 0,
                description: String::new(),
            }),
            data: Arc::new(DashMap::new()),
        }
    }

    /// Store custom data in plugin context
    pub fn set_data<T: 'static + Send + Sync>(&self, key: &str, value: T) -> Result<()> {
        self.data.insert(key.to_string(), Arc::new(value));
        Ok(())
    }

    /// Retrieve custom data from plugin context
    pub fn get_data<T: 'static + Send + Sync>(&self, key: &str) -> Result<Option<Arc<T>>> {
        Ok(self.data.get(key).and_then(|entry| {
            let arc = entry.value();
            arc.clone().downcast::<T>().ok()
        }))
    }

    /// Get current state
    pub fn get_state(&self) -> Arc<crate::core::state::SharedState> {
        Arc::clone(&self.state)
    }

    /// Get graph information
    pub fn graph_info(&self) -> &GraphInfo {
        &self.graph_info
    }
}

impl Default for PluginContext {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_plugin_context_creation() {
        let ctx = PluginContext::new();
        assert_eq!(ctx.graph_info.node_count, 0);
    }

    #[test]
    fn test_plugin_context_data_storage() {
        let ctx = PluginContext::new();
        ctx.set_data("key1", "value1".to_string()).unwrap();
        let data: Option<Arc<String>> = ctx.get_data("key1").unwrap();
        assert!(data.is_some());
        assert_eq!(*data.unwrap(), "value1");
    }

    #[test]
    fn test_plugin_context_data_missing() {
        let ctx = PluginContext::new();
        let data: Option<Arc<String>> = ctx.get_data("nonexistent").unwrap();
        assert!(data.is_none());
    }
}
