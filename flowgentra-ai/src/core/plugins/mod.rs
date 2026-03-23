/// Plugin system for FlowgentraAI extensibility
///
/// This module provides a plugin architecture for extending FlowgentraAI with
/// custom functionality at runtime. Plugins can hook into handler execution,
/// error handling, and tool registration.
use crate::prelude::*;

pub mod context;
pub mod registry;

pub use context::PluginContext;
pub use registry::{PluginId, PluginRegistry};

/// The main plugin trait that all plugins must implement
///
/// Plugins can hook into various lifecycle events and extend FlowgentraAI's
/// functionality at runtime.
///
/// # Example
///
/// ```ignore
/// use flowgentra_ai::prelude::*;
/// use flowgentra_ai::plugins::Plugin;
///
/// struct MyPlugin;
///
/// #[async_trait]
/// impl Plugin for MyPlugin {
///     fn name(&self) -> &str { "my-plugin" }
///     fn version(&self) -> &str { "0.1.0" }
///
///     async fn initialize(&self, context: &mut PluginContext) -> Result<()> {
///         println!("Plugin initialized!");
///         Ok(())
///     }
///
///     async fn on_handler_start(
///         &self,
///         _context: &PluginContext,
///         handler_name: &str,
///     ) -> Result<()> {
///         println!("Handler starting: {}", handler_name);
///         Ok(())
///     }
/// }
/// ```
#[async_trait::async_trait]
pub trait Plugin: Send + Sync {
    /// Plugin name (unique identifier)
    fn name(&self) -> &str;

    /// Plugin semantic version (e.g., "0.1.0")
    fn version(&self) -> &str;

    /// Called when plugin is loaded and initialized
    async fn initialize(&self, context: &mut PluginContext) -> Result<()> {
        let _context = context;
        Ok(())
    }

    /// Called when a handler is about to execute
    async fn on_handler_start(&self, _context: &PluginContext, _handler_name: &str) -> Result<()> {
        Ok(())
    }

    /// Called after a handler completes successfully
    async fn on_handler_complete(
        &self,
        _context: &PluginContext,
        _handler_name: &str,
        _duration_ms: u64,
    ) -> Result<()> {
        Ok(())
    }

    /// Called when a handler fails
    async fn on_handler_error(
        &self,
        _context: &PluginContext,
        _handler_name: &str,
        _error: &FlowgentraError,
    ) -> Result<()> {
        Ok(())
    }

    /// Called before node execution
    async fn on_node_execute(&self, _context: &PluginContext, _node_id: &str) -> Result<()> {
        Ok(())
    }

    /// Called after node execution completes
    async fn on_node_complete(
        &self,
        _context: &PluginContext,
        _node_id: &str,
        _success: bool,
    ) -> Result<()> {
        Ok(())
    }

    /// Called when a tool is invoked
    async fn on_tool_invoke(
        &self,
        _context: &PluginContext,
        _tool_name: &str,
        _args: &serde_json::Value,
    ) -> Result<()> {
        Ok(())
    }

    /// Called when plugin is being unloaded
    async fn shutdown(&self, _context: &PluginContext) -> Result<()> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct TestPlugin;

    #[async_trait::async_trait]
    impl Plugin for TestPlugin {
        fn name(&self) -> &str {
            "test-plugin"
        }

        fn version(&self) -> &str {
            "0.1.0"
        }
    }

    #[tokio::test]
    async fn test_plugin_trait() {
        let plugin = TestPlugin;
        assert_eq!(plugin.name(), "test-plugin");
        assert_eq!(plugin.version(), "0.1.0");
        assert!(plugin.initialize(&mut PluginContext::new()).await.is_ok());
    }
}
