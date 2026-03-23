/// Plugin registry for managing loaded plugins
use super::{Plugin, PluginContext};
use crate::prelude::*;
use std::collections::HashMap;
use std::sync::Arc;

/// Unique identifier for a loaded plugin
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct PluginId(String);

impl PluginId {
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }
}

impl std::fmt::Display for PluginId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Registry for loading and managing plugins
pub struct PluginRegistry {
    plugins: Arc<std::sync::RwLock<HashMap<PluginId, Arc<dyn Plugin>>>>,
    initialized: Arc<std::sync::RwLock<bool>>,
}

impl PluginRegistry {
    /// Create a new plugin registry
    pub fn new() -> Self {
        Self {
            plugins: Arc::new(std::sync::RwLock::new(HashMap::new())),
            initialized: Arc::new(std::sync::RwLock::new(false)),
        }
    }

    /// Register a plugin
    pub fn register(&self, id: PluginId, plugin: Arc<dyn Plugin>) -> Result<()> {
        let mut plugins = self.plugins.write().map_err(|_| {
            FlowgentraError::RuntimeError("Plugin registry lock poisoned".to_string())
        })?;

        if plugins.contains_key(&id) {
            return Err(FlowgentraError::ConfigError(format!(
                "Plugin with id '{}' already registered",
                id
            )));
        }

        plugins.insert(id, plugin);
        Ok(())
    }

    /// Unregister a plugin
    pub fn unregister(&self, id: &PluginId) -> Result<Option<Arc<dyn Plugin>>> {
        let mut plugins = self.plugins.write().map_err(|_| {
            FlowgentraError::RuntimeError("Plugin registry lock poisoned".to_string())
        })?;

        Ok(plugins.remove(id))
    }

    /// Get a registered plugin
    pub fn get(&self, id: &PluginId) -> Result<Option<Arc<dyn Plugin>>> {
        let plugins = self.plugins.read().map_err(|_| {
            FlowgentraError::RuntimeError("Plugin registry lock poisoned".to_string())
        })?;

        Ok(plugins.get(id).cloned())
    }

    /// Get all registered plugins
    pub fn all(&self) -> Result<Vec<Arc<dyn Plugin>>> {
        let plugins = self.plugins.read().map_err(|_| {
            FlowgentraError::RuntimeError("Plugin registry lock poisoned".to_string())
        })?;

        Ok(plugins.values().cloned().collect())
    }

    /// Initialize all registered plugins
    pub async fn initialize(&self, context: &mut PluginContext) -> Result<()> {
        let plugins = self.all()?;

        for plugin in plugins {
            plugin.initialize(context).await?;
        }

        let mut initialized = self.initialized.write().map_err(|_| {
            FlowgentraError::RuntimeError("Plugin registry lock poisoned".to_string())
        })?;
        *initialized = true;

        Ok(())
    }

    /// Shutdown all plugins
    pub async fn shutdown(&self, context: &PluginContext) -> Result<()> {
        let plugins = self.all()?;

        for plugin in plugins {
            plugin.shutdown(context).await?;
        }

        let mut initialized = self.initialized.write().map_err(|_| {
            FlowgentraError::RuntimeError("Plugin registry lock poisoned".to_string())
        })?;
        *initialized = false;

        Ok(())
    }

    /// Check if plugins are initialized
    pub fn is_initialized(&self) -> Result<bool> {
        self.initialized
            .read()
            .map_err(|_| FlowgentraError::RuntimeError("Plugin registry lock poisoned".to_string()))
            .map(|flag| *flag)
    }

    /// Call a lifecycle hook on all plugins that implement handler start
    pub async fn on_handler_start(
        &self,
        context: &PluginContext,
        handler_name: &str,
    ) -> Result<()> {
        let plugins = self.all()?;

        for plugin in plugins {
            plugin.on_handler_start(context, handler_name).await?;
        }

        Ok(())
    }

    /// Call a lifecycle hook on all plugins that implement handler complete
    pub async fn on_handler_complete(
        &self,
        context: &PluginContext,
        handler_name: &str,
        duration_ms: u64,
    ) -> Result<()> {
        let plugins = self.all()?;

        for plugin in plugins {
            plugin
                .on_handler_complete(context, handler_name, duration_ms)
                .await?;
        }

        Ok(())
    }

    /// Call a lifecycle hook on all plugins that implement handler error
    pub async fn on_handler_error(
        &self,
        context: &PluginContext,
        handler_name: &str,
        error: &FlowgentraError,
    ) -> Result<()> {
        let plugins = self.all()?;

        for plugin in plugins {
            plugin
                .on_handler_error(context, handler_name, error)
                .await?;
        }

        Ok(())
    }

    /// Call a lifecycle hook on all plugins that implement node execute
    pub async fn on_node_execute(&self, context: &PluginContext, node_id: &str) -> Result<()> {
        let plugins = self.all()?;

        for plugin in plugins {
            plugin.on_node_execute(context, node_id).await?;
        }

        Ok(())
    }

    /// Call a lifecycle hook on all plugins that implement node complete
    pub async fn on_node_complete(
        &self,
        context: &PluginContext,
        node_id: &str,
        success: bool,
    ) -> Result<()> {
        let plugins = self.all()?;

        for plugin in plugins {
            plugin.on_node_complete(context, node_id, success).await?;
        }

        Ok(())
    }

    /// Call a lifecycle hook on all plugins that implement tool invoke
    pub async fn on_tool_invoke(
        &self,
        context: &PluginContext,
        tool_name: &str,
        args: &serde_json::Value,
    ) -> Result<()> {
        let plugins = self.all()?;

        for plugin in plugins {
            plugin.on_tool_invoke(context, tool_name, args).await?;
        }

        Ok(())
    }
}

impl Default for PluginRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;

    struct TestPlugin;

    #[async_trait]
    impl Plugin for TestPlugin {
        fn name(&self) -> &str {
            "test-plugin"
        }

        fn version(&self) -> &str {
            "0.1.0"
        }
    }

    #[tokio::test]
    async fn test_register_plugin() {
        let registry = PluginRegistry::new();
        let plugin = Arc::new(TestPlugin);
        let id = PluginId::new("test");

        registry.register(id.clone(), plugin.clone()).unwrap();
        assert!(registry.get(&id).unwrap().is_some());
    }

    #[tokio::test]
    async fn test_duplicate_registration() {
        let registry = PluginRegistry::new();
        let plugin = Arc::new(TestPlugin);
        let id = PluginId::new("test");

        registry.register(id.clone(), plugin.clone()).unwrap();
        assert!(registry.register(id, plugin).is_err());
    }

    #[tokio::test]
    async fn test_unregister_plugin() {
        let registry = PluginRegistry::new();
        let plugin = Arc::new(TestPlugin);
        let id = PluginId::new("test");

        registry.register(id.clone(), plugin).unwrap();
        assert!(registry.unregister(&id).unwrap().is_some());
        assert!(registry.get(&id).unwrap().is_none());
    }

    #[tokio::test]
    async fn test_get_all_plugins() {
        let registry = PluginRegistry::new();
        let plugin1 = Arc::new(TestPlugin);
        let plugin2 = Arc::new(TestPlugin);

        registry.register(PluginId::new("test1"), plugin1).unwrap();
        registry.register(PluginId::new("test2"), plugin2).unwrap();

        let all = registry.all().unwrap();
        assert_eq!(all.len(), 2);
    }

    #[tokio::test]
    async fn test_initialize_shutdown() {
        let registry = PluginRegistry::new();
        let plugin = Arc::new(TestPlugin);
        registry.register(PluginId::new("test"), plugin).unwrap();

        let mut ctx = PluginContext::new();
        assert!(!registry.is_initialized().unwrap());

        registry.initialize(&mut ctx).await.unwrap();
        assert!(registry.is_initialized().unwrap());

        registry.shutdown(&ctx).await.unwrap();
        assert!(!registry.is_initialized().unwrap());
    }
}
