use tracing::info;
// # Agent API - High-Level Interface
//
// The `Agent` struct provides a simple API for creating and running agents.
// Most users interact with this module through the `from_config_path()` function
// which uses automatic handler discovery.
//
// ## Quick Start
//
// 1. **Decorate your handlers** with `#[register_handler]`
// 2. **Create a config.yaml** with your agent graph
// 3. **Use `from_config_path()`** to create and run the agent
//
// ```ignore
// use flowgentra_ai::prelude::*;
// use serde_json::json;
//
// #[register_handler]
// pub async fn my_handler(mut state: State) -> Result<State> {
//     let input = state.get("input").and_then(|v| v.as_str()).unwrap_or("");
//     state.set("result", json!(input.to_uppercase()));
//     Ok(state)
// }
//
// #[tokio::main]
// async fn main() -> Result<()> {
//     // Auto-discovers all #[register_handler] functions
//     let mut agent = from_config_path("config.yaml")?;
//
//     let mut state = State::new(Default::default());
//     state.set("input", json!("hello world"));
//
//     let result = agent.run(state).await?;
//     println!("Done: {}", result.to_json_string()?);
//     Ok(())
// }
// ```
//
// ## Handler Registration
//
// Handlers are automatically registered via the `#[register_handler]` attribute macro.
/// Handler names must match the function name and be referenced by that name in your config.yaml.
pub(crate) use crate::core::config::AgentConfig;
pub(crate) use crate::core::error::{FlowgentraError, Result};
pub(crate) use crate::core::llm::{create_llm_client, LLMClient};
pub(crate) use crate::core::memory::{
    ConversationMemory, InMemoryConversationMemory, MemoryCheckpointer,
};
pub(crate) use crate::core::runtime::AgentRuntime;
pub(crate) use crate::core::state::DynState;
use std::collections::HashMap;
use std::sync::Arc;
// Use inventory for auto-registration of handlers - collected dynamically at runtime
inventory::collect!(HandlerEntry);

// =============================================================================
// Auto-Registration via Inventory
// =============================================================================

/// Entry for a handler in the global inventory
/// Handlers submit themselves to this list for auto-registration
pub struct HandlerEntry {
    /// Name of the handler (matches config node names)
    pub name: String,
    /// The handler function (always uses DynState)
    pub handler: ArcHandler<DynState>,
}

impl HandlerEntry {
    /// Create a new handler entry for auto-registration
    pub fn new(name: impl Into<String>, handler: ArcHandler<DynState>) -> HandlerEntry {
        HandlerEntry {
            name: name.into(),
            handler,
        }
    }
}

// Type for Arc-wrapped handlers (used by inventory auto-registration)
pub type ArcHandler<T> = Arc<
    dyn Fn(T) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<T>> + Send>>
        + Send
        + Sync,
>;

// =============================================================================
// Type Aliases
// =============================================================================

/// Type signature for handler functions
///
/// A handler receives the current state and returns the updated state.
/// Handlers are async and can perform I/O operations.
///
/// # Example
/// ```no_run
/// use flowgentra_ai::core::agent::Handler;
/// use flowgentra_ai::core::state::DynState;
/// use serde_json::json;
///
/// let my_handler: Handler<DynState> = Box::new(|state| {
///     Box::pin(async move {
///         let input = state.get("input");
///         state.set("output", json!("processed"));
///         Ok(state)
///     })
/// });
/// ```
pub type Handler<T> = Box<
    dyn Fn(T) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<T>> + Send>>
        + Send
        + Sync,
>;

/// Type signature for condition functions
///
/// A condition function evaluates the current state and returns a boolean.
/// Used for branching logic in the graph.
///
/// # Example
/// ```no_run
/// use flowgentra_ai::core::agent::Condition;
/// use flowgentra_ai::core::state::DynState;
///
/// let is_complex: Condition<DynState> = Box::new(|state: &DynState| {
///     state.get("complexity_score")
///         .and_then(|v| v.as_i64())
///         .map(|score| score > 50)
///         .unwrap_or(false)
/// });
/// ```
pub type Condition<T> = Box<dyn Fn(&T) -> bool + Send + Sync>;

/// Registry mapping handler names to handler functions
pub type HandlerRegistry<T> = HashMap<String, Handler<T>>;

/// Registry mapping condition names to condition functions
pub type ConditionRegistry<T> = HashMap<String, Condition<T>>;

// =============================================================================
// Agent - Main API
// =============================================================================

/// The Agent - your main interface to FlowgentraAI
///
/// Create an agent from a YAML config and handler implementations,
/// then run it with a state to get results.
///
/// The Agent handles:
/// - Loading and validating configuration
/// - Registering handlers and conditions
/// - Managing the execution runtime
/// - Orchestrating node execution
/// - Optional checkpointer and conversation memory (from config or programmatic)
pub struct Agent {
    runtime: AgentRuntime,
    llm_client: Arc<dyn LLMClient>,
    config: AgentConfig,
    /// Current state of the agent (initialized from state_schema)
    pub state: DynState,
    /// Optional conversation memory (message history per thread). Set via config or with_conversation_memory().
    conversation_memory: Option<Arc<dyn ConversationMemory>>,
}

impl Agent {
    /// Log agent startup and configuration for observability
    pub fn log_startup(&self) {
        info!(
            "Agent '{}' starting with config: {:?}",
            self.config.name, self.config
        );
    }
    /// Create an agent from a YAML config file
    ///
    /// This is the main entry point for users. It:
    /// 1. Loads the YAML config file
    /// 2. Validates the graph structure
    /// 3. Creates the runtime with your handlers and conditions
    /// 4. Sets up the LLM client
    ///
    /// # Arguments
    /// - `config_path`: Path to your `config.yaml` file
    /// - `handlers`: Registry of handler functions
    /// - `conditions`: Registry of condition functions
    ///
    /// # Example
    /// ```no_run
    /// use flowgentra_ai::prelude::*;
    /// use std::collections::HashMap;
    ///
    /// #[tokio::main]
    /// async fn main() -> Result<()> {
    ///     let handlers = HashMap::new();
    ///     let conditions = HashMap::new();
    ///     let mut agent = Agent::from_config("config.yaml", handlers, conditions)?;
    ///     Ok(())
    /// }
    /// ```
    pub fn from_config(
        config_path: &str,
        handlers: HandlerRegistry<DynState>,
        conditions: ConditionRegistry<DynState>,
    ) -> Result<Self> {
        let config = AgentConfig::from_file(config_path)?;
        config.validate()?;
        Self::from_config_inner(config, handlers, conditions)
    }

    /// Create an agent from an already-loaded [`AgentConfig`].
    ///
    /// Use this when you have a config object in hand (e.g. from `from_config_path`
    /// which already loaded the file) to avoid reading the file a second time.
    pub fn from_config_inner(
        config: AgentConfig,
        handlers: HandlerRegistry<DynState>,
        conditions: ConditionRegistry<DynState>,
    ) -> Result<Self> {
        // Create runtime
        let mut runtime = AgentRuntime::from_config(config.clone())?;

        // Create LLM client
        let llm_client = create_llm_client(&config.llm)?;

        // Convert all handlers to Arc (cloneable) so multiple nodes can share a handler
        let arc_handlers: HashMap<String, ArcHandler<DynState>> = handlers
            .into_iter()
            .map(|(name, handler)| {
                let arc: ArcHandler<DynState> = Arc::new(move |state| handler(state));
                (name, arc)
            })
            .collect();

        // Register handlers by mapping each node config to its handler function.
        // Looks up by node.name first (from from_config_path, which pre-resolves by node name),
        // then by node.handler (from manual Agent::from_config, which uses handler names).
        let mut missing_handlers = Vec::new();

        // Built-in node types that don't require a user-supplied handler
        const BUILTIN_TYPES: &[&str] = &[
            "evaluation",
            "retry",
            "timeout",
            "loop",
            "planner",
            "human_in_the_loop",
            "memory",
            // supervisor (+ backwards-compat alias)
            "supervisor",
            "orchestrator",
            // subgraph (+ backwards-compat aliases)
            "subgraph",
            "agent",
            "agent_or_graph",
        ];

        // Supervisor-managed children are not in the runtime graph — skip registration.
        let supervisor_children: std::collections::HashSet<String> = config
            .graph
            .nodes
            .iter()
            .filter(|n| {
                matches!(
                    n.node_type.as_deref(),
                    Some("supervisor") | Some("orchestrator")
                )
            })
            .flat_map(|n| {
                n.config
                    .get("children")
                    .and_then(|v| v.as_array())
                    .map(|arr| {
                        arr.iter()
                            .filter_map(|v| v.as_str().map(String::from))
                            .collect::<Vec<_>>()
                    })
                    .unwrap_or_default()
            })
            .collect();

        for node in &config.graph.nodes {
            if node.name == "START" || node.name == "END" {
                continue;
            }
            if supervisor_children.contains(&node.name) {
                continue;
            }
            if node.handler.starts_with("builtin::") {
                continue;
            }
            let arc_handler = arc_handlers
                .get(&node.name)
                .or_else(|| arc_handlers.get(&node.handler))
                .cloned();

            // Standalone built-in type with no user-supplied handler — only skip when
            // from_config_path did NOT pre-build a handler for this node.
            // If a handler IS present (e.g. the planner built by from_config_path), always
            // register it so the placeholder function is replaced.
            let is_builtin_standalone = node.handler.is_empty()
                && node
                    .node_type
                    .as_deref()
                    .is_some_and(|t| BUILTIN_TYPES.contains(&t))
                && arc_handler.is_none();
            if is_builtin_standalone {
                continue;
            }

            match arc_handler {
                Some(h) => {
                    runtime
                        .register_node(&node.name, Box::new(move |state| h(state)))
                        .map_err(|e| {
                            if matches!(e, FlowgentraError::NodeNotFound(_)) {
                                FlowgentraError::NodeNotFound(format!(
                                    "Node '{}' not found in graph. Check your config.",
                                    node.name
                                ))
                            } else {
                                e
                            }
                        })?;
                }
                None => {
                    missing_handlers.push(node.handler.clone());
                }
            }
        }

        if !missing_handlers.is_empty() {
            let available_list = if arc_handlers.is_empty() {
                "(none registered)".to_string()
            } else {
                arc_handlers
                    .keys()
                    .map(|k| format!("'{}'", k))
                    .collect::<Vec<_>>()
                    .join(", ")
            };

            return Err(FlowgentraError::ConfigError(format!(
                "Configuration references unknown handler(s): {}.\nAvailable handlers: {}\nMake sure to #[register_handler] these functions in your code.",
                missing_handlers.iter()
                    .map(|h| format!("'{}'", h))
                    .collect::<Vec<_>>()
                    .join(", "),
                available_list
            )));
        }

        // Register all conditions
        type EdgeConditionFn = std::sync::Arc<
            dyn Fn(
                    &DynState,
                )
                    -> std::result::Result<Option<String>, crate::core::error::FlowgentraError>
                + Send
                + Sync,
        >;
        for (condition_name, condition_fn) in conditions {
            let cond_name_clone = condition_name.clone();
            let edge_condition: EdgeConditionFn = std::sync::Arc::new(move |state: &DynState| {
                if condition_fn(state) {
                    Ok(Some(cond_name_clone.clone()))
                } else {
                    Ok(None)
                }
            });

            let edges_to_register: Vec<String> = runtime
                .graph()
                .edges
                .iter()
                .filter(|e| e.condition_name.as_deref() == Some(condition_name.as_str()))
                .map(|e| e.from.clone())
                .collect();

            for from_node in edges_to_register {
                runtime.register_edge_condition(
                    &from_node,
                    &condition_name,
                    edge_condition.clone(),
                )?;
            }
        }

        let initial_state = config.create_initial_state();
        Ok(Agent {
            runtime,
            llm_client,
            config,
            state: initial_state,
            conversation_memory: None,
        })
    }

    /// Set the checkpointer (e.g. for thread-scoped state persistence). Can also be set via config.yaml `memory.checkpointer`.
    pub fn set_checkpointer(&mut self, checkpointer: Arc<MemoryCheckpointer>) -> &mut Self {
        self.runtime.set_checkpointer(checkpointer);
        self
    }

    /// Builder-style: set the checkpointer.
    pub fn with_checkpointer(mut self, checkpointer: Arc<MemoryCheckpointer>) -> Self {
        self.runtime.set_checkpointer(checkpointer);
        self
    }

    /// Set conversation memory (message history per thread). Can also be set via config.yaml `memory.conversation`.
    pub fn set_conversation_memory(&mut self, memory: Arc<dyn ConversationMemory>) -> &mut Self {
        self.conversation_memory = Some(memory);
        self
    }

    /// Builder-style: set conversation memory.
    pub fn with_conversation_memory(mut self, memory: Arc<dyn ConversationMemory>) -> Self {
        self.conversation_memory = Some(memory);
        self
    }

    /// Get conversation memory if set (for use in handlers to add/get messages).
    pub fn conversation_memory(&self) -> Option<Arc<dyn ConversationMemory>> {
        self.conversation_memory.clone()
    }

    /// Execute the agent with its current state
    ///
    /// Runs the agent through all nodes following the edges until completion.
    /// Uses the agent's built-in state (initialized from state_schema).
    /// Automatically injects the LLM configuration into state so handlers can access it.
    ///
    /// Set state values before calling:
    /// ```no_run
    /// use flowgentra_ai::prelude::*;
    /// use serde_json::json;
    ///
    /// #[tokio::main]
    /// async fn main() -> Result<()> {
    ///     let mut agent = from_config_path("config.yaml")?;
    ///     
    ///     // Set initial values on agent.state
    ///     agent.state.set("input", json!("Say hello"));
    ///     
    ///     let result_state = agent.run().await?;
    ///     println!("Done! Result: {}", result_state.to_json_string()?);
    ///     
    ///     Ok(())
    /// }
    /// ```
    pub async fn run(&mut self) -> Result<DynState> {
        // Automatically inject LLM config into state for handler access
        let llm_config_json =
            serde_json::to_value(&self.config.llm).unwrap_or_else(|_| serde_json::json!({}));
        self.state.set("_llm_config", llm_config_json); // Ensure set() is defined in State

        // Automatically inject MCP configs into state for handler access
        if !self.config.graph.mcps.is_empty() {
            let mcp_configs_json = serde_json::to_value(&self.config.graph.mcps)
                .unwrap_or_else(|_| serde_json::json!({}));
            self.state.set("_mcp_configs", mcp_configs_json);
        }

        // Automatically inject RAG config into state for handler access
        if let Some(ref rag_config) = self.config.graph.rag {
            let mut resolved = rag_config.clone();
            resolved.resolve_env_vars();
            let rag_config_json =
                serde_json::to_value(&resolved).unwrap_or_else(|_| serde_json::json!({}));
            self.state.set("_rag_config", rag_config_json);
        }

        self.runtime.execute(self.state.clone()).await
    }

    /// Run with a thread id for checkpointing and conversation memory. When a checkpointer is set,
    /// state is loaded from the last checkpoint for this thread (if any) and saved after each node.
    /// Use the same thread_id with conversation_memory to get/add messages for this conversation.
    /// Automatically injects the LLM configuration into state so handlers can access it.
    pub async fn run_with_thread(&mut self, thread_id: &str) -> Result<DynState> {
        // Automatically inject LLM config into state for handler access
        let llm_config_json =
            serde_json::to_value(&self.config.llm).unwrap_or_else(|_| serde_json::json!({}));
        self.state.set("_llm_config", llm_config_json);

        // Automatically inject MCP configs into state for handler access
        if !self.config.graph.mcps.is_empty() {
            let mcp_configs_json = serde_json::to_value(&self.config.graph.mcps)
                .unwrap_or_else(|_| serde_json::json!({}));
            self.state.set("_mcp_configs", mcp_configs_json);
        }

        // Automatically inject RAG config into state for handler access
        if let Some(ref rag_config) = self.config.graph.rag {
            let mut resolved = rag_config.clone();
            resolved.resolve_env_vars();
            let rag_config_json =
                serde_json::to_value(&resolved).unwrap_or_else(|_| serde_json::json!({}));
            self.state.set("_rag_config", rag_config_json);
        }

        self.runtime
            .execute_with_thread(thread_id, self.state.clone())
            .await
    }

    /// Get the LLM client for use in handlers
    ///
    /// Handlers can use this to access the configured LLM provider.
    pub fn llm_client(&self) -> Arc<dyn LLMClient> {
        Arc::clone(&self.llm_client)
    }

    /// Get a reference to the agent's configuration
    pub fn config(&self) -> &AgentConfig {
        &self.config
    }

    // Visualize the agent's execution graph
    //
    // Generates a text-based or graphical representation of your agent's workflow.
    // Useful for debugging and documentation.
    //
    // visualize_graph requires type parameter T to be known, which is not available in Agent context
    // pub async fn visualize_graph(&self, output_path: &str) -> Result<()> {
    //     self.runtime.visualize_graph(output_path)
    // }

    /// Get mutable access to the underlying runtime
    ///
    /// For advanced users who need direct runtime access.
    pub fn runtime_mut(&mut self) -> &mut AgentRuntime {
        &mut self.runtime
    }

    // ==============================================
    // Memory Management API
    // ==============================================

    /// Initialize message history in state for memory
    ///
    /// Creates an empty message history that handlers can append to.
    ///
    /// # Example (Code API)
    /// ```ignore
    /// let mut agent = from_config_path("config.yaml")?;
    /// agent.enable_message_history()?;  // Just add this line!
    /// ```
    ///
    /// # Example (Config approach - just add to state_schema)
    /// ```yaml
    /// state_schema:
    ///   messages:
    ///     type: array
    ///     description: "Message history"
    /// ```
    pub fn enable_message_history(&mut self) -> Result<()> {
        use crate::core::state::MessageHistory;
        let history = MessageHistory::new();
        history.save_to_state(&self.state)?;
        Ok(())
    }

    /// Add a memory handler node to the graph
    ///
    /// Allows programmatic addition of memory handlers without editing config.
    /// Handlers: "memory::append_message", "memory::compress_history", "memory::clear_history"
    ///
    /// # Example
    /// ```ignore
    /// agent.add_memory_handler("append", "memory::append_message")?;
    /// agent.add_memory_handler("compress", "memory::compress_history")?;
    /// ```
    pub fn add_memory_handler(&mut self, node_name: &str, handler_type: &str) -> Result<()> {
        // Register the memory handler in the runtime
        let handler: Handler<DynState> = match handler_type {
            "memory::append_message" => Box::new(|state| {
                Box::pin(crate::core::node::memory_handlers::append_message_handler(
                    state,
                ))
            }),
            "memory::compress_history" => Box::new(|state| {
                Box::pin(crate::core::node::memory_handlers::compress_history_handler(state))
            }),
            "memory::clear_history" => Box::new(|state| {
                Box::pin(crate::core::node::memory_handlers::clear_history_handler(
                    state,
                ))
            }),
            "memory::get_message_count" => Box::new(|state| {
                Box::pin(crate::core::node::memory_handlers::get_message_count_handler(state))
            }),
            "memory::format_history_for_context" => Box::new(|state| {
                Box::pin(
                    crate::core::node::memory_handlers::format_history_for_context_handler(state),
                )
            }),
            _ => {
                return Err(FlowgentraError::ValidationError(format!(
                    "Unknown memory handler: {}",
                    handler_type
                )))
            }
        };

        self.runtime.register_node(node_name, handler)?;

        Ok(())
    }

    /// Get message history helper
    ///
    /// Convenience method to work with message history
    ///
    /// # Example
    /// ```ignore
    /// let mut history = agent.get_message_history()?;
    /// history.add_user_message("Hello!");
    /// history.save_to_state(&agent.state)?;
    /// ```
    pub fn get_message_history(&self) -> Result<crate::core::state::MessageHistory> {
        crate::core::state::MessageHistory::from_state(&self.state)
    }

    /// Set message history from user messages
    ///
    /// Useful for loading existing conversation or initializing with context
    ///
    /// # Example
    /// ```ignore
    /// let messages = vec![
    ///     ("user", "What is Rust?"),
    ///     ("assistant", "Rust is a systems programming language..."),
    /// ];
    /// for (role, content) in &messages {
    ///     agent.add_message(role, content)?;
    /// }
    /// ```
    pub fn add_message(&mut self, role: &str, content: &str) -> Result<()> {
        let mut history = self.get_message_history()?;
        match role {
            "user" => history.add_user_message(content),
            "assistant" => history.add_assistant_message(content),
            "system" => history.add_system_message(content),
            _ => {
                return Err(FlowgentraError::ValidationError(format!(
                    "Invalid role: {}",
                    role
                )))
            }
        }
        history.save_to_state(&self.state)?;
        Ok(())
    }

    /// Clear all messages from history
    ///
    /// Useful for resetting conversation state
    pub fn clear_messages(&mut self) -> Result<()> {
        let history = crate::core::state::MessageHistory::new();
        history.save_to_state(&self.state)?;
        Ok(())
    }

    /// Get custom state field helper
    ///
    /// For Pattern 4: Custom State Fields
    pub fn custom_state(&self) -> Result<crate::core::state::CustomState> {
        crate::core::state::CustomState::from_state(&self.state)
    }

    /// Set a custom state field
    pub fn set_custom_field(&mut self, key: &str, value: serde_json::Value) -> Result<()> {
        let mut custom = self.custom_state()?;
        custom.set(key, value);
        custom.save_to_state(&self.state)?;
        Ok(())
    }
}

/// Create an agent from config path only - handlers are auto-discovered!
///
/// Handlers registered via `#[register_handler]` attribute are automatically collected
/// and available to the agent builder. No manual registration needed!
///
/// # Arguments
/// - `config_path`: Path to your `config.yaml` file
///
/// # Returns
/// Result with Agent if successful, error if any handlers are missing
///
/// # Error Messages
/// The function will provide helpful error messages if handlers are missing,
/// showing which handlers are required by the config but not registered.
///
/// # Example
/// ```ignore
/// use flowgentra_ai::prelude::*;
///
/// #[register_handler]
/// pub async fn my_handler(state: State) -> Result<State> {
///     state.set("output", json!("done"));
///     Ok(state)
/// }
///
/// #[tokio::main]
/// async fn main() -> Result<()> {
///     // That's it! Just create the agent from config
///     let mut agent = from_config_path("config.yaml")?;
///     
///     let state = State::new(Default::default());
///     let result = agent.run(state).await?;
///     Ok(())
/// }
/// ```
/// Create an [`Agent`] from a YAML config file, auto-discovering all `#[register_handler]`
/// functions plus any extra handlers provided (e.g. Python callables wrapped as Rust handlers).
///
/// Extra handlers take priority over inventory-registered handlers with the same name.
pub fn from_config_path_with_extra_handlers(
    config_path: &str,
    extra_handlers: HashMap<String, ArcHandler<DynState>>,
) -> Result<Agent> {
    from_config_path_impl(config_path, extra_handlers)
}

pub fn from_config_path(config_path: &str) -> Result<Agent> {
    from_config_path_impl(config_path, HashMap::new())
}

fn from_config_path_impl(
    config_path: &str,
    extra_handlers: HashMap<String, ArcHandler<DynState>>,
) -> Result<Agent> {
    // Load config to get required node names
    let mut config = AgentConfig::from_file(config_path)?;
    config.validate()?;

    // Resolve relative paths in subgraph configs against the parent config file's directory.
    // This ensures `path: agents/researcher.yaml` in config.yaml correctly resolves to
    // `<config_dir>/agents/researcher.yaml` regardless of the process CWD.
    let config_dir = std::path::Path::new(config_path)
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(std::path::Path::new("."))
        .to_path_buf();

    // For stdio MCPs: resolve working_dir and command path.
    // - Default working_dir to config file's directory so relative script paths work
    // - Resolve command via PATH if it's not an absolute path
    for mcp in config.graph.mcps.values_mut() {
        if mcp.connection_type == crate::core::mcp::MCPConnectionType::Stdio {
            if mcp.connection_settings.working_dir.is_none() {
                mcp.connection_settings.working_dir =
                    Some(config_dir.to_string_lossy().to_string());
            }
            // Resolve command to absolute path if needed (e.g. "python" → "/usr/bin/python")
            let cmd = mcp.stdio_command().to_string();
            if !std::path::Path::new(&cmd).is_absolute() {
                let mut resolved = String::new();

                // Try system lookup (where on Windows, which on Unix)
                let lookup_cmd = if cfg!(windows) { "where" } else { "which" };
                if let Ok(output) = std::process::Command::new(lookup_cmd).arg(&cmd).output() {
                    if output.status.success() {
                        resolved = String::from_utf8_lossy(&output.stdout)
                            .lines()
                            .next()
                            .unwrap_or("")
                            .trim()
                            .to_string();
                    }
                }

                // On Windows, also search common installation paths
                #[cfg(windows)]
                if resolved.is_empty() && (cmd == "python" || cmd == "python3") {
                    let candidates = [
                        // Standard Python installer paths
                        "C:\\Python312\\python.exe".to_string(),
                        "C:\\Python311\\python.exe".to_string(),
                        "C:\\Python310\\python.exe".to_string(),
                        "C:\\Python39\\python.exe".to_string(),
                        // Program Files
                        "C:\\Program Files\\Python312\\python.exe".to_string(),
                        "C:\\Program Files\\Python311\\python.exe".to_string(),
                        "C:\\Program Files\\Python310\\python.exe".to_string(),
                        "C:\\Program Files\\Python39\\python.exe".to_string(),
                    ];
                    if let Ok(home) = std::env::var("USERPROFILE") {
                        let user_candidates = [
                            format!(
                                "{}\\AppData\\Local\\Programs\\Python\\Python312\\python.exe",
                                home
                            ),
                            format!(
                                "{}\\AppData\\Local\\Programs\\Python\\Python311\\python.exe",
                                home
                            ),
                            format!(
                                "{}\\AppData\\Local\\Programs\\Python\\Python310\\python.exe",
                                home
                            ),
                        ];
                        for c in user_candidates.iter().chain(candidates.iter()) {
                            if std::path::Path::new(c).exists() {
                                resolved = c.clone();
                                break;
                            }
                        }
                    } else {
                        for c in &candidates {
                            if std::path::Path::new(c).exists() {
                                resolved = c.clone();
                                break;
                            }
                        }
                    }
                }

                if !resolved.is_empty() {
                    tracing::info!(command = %cmd, resolved = %resolved, "Resolved stdio command path");
                    if mcp.command.is_some() {
                        mcp.command = Some(resolved);
                    } else {
                        mcp.uri = resolved;
                    }
                }
            }
        }
    }

    // Collect all registered handlers from inventory (ArcHandler = cloneable)
    let mut handlers_map: HashMap<String, ArcHandler<DynState>> = HashMap::new();
    for entry in inventory::iter::<HandlerEntry> {
        handlers_map.insert(entry.name.clone(), entry.handler.clone());
    }
    // Extra handlers (e.g. Python callables) take priority over inventory handlers
    handlers_map.extend(extra_handlers);

    // Inject builtin::planner into handlers_map for backward compatibility
    // (when a node uses handler: "builtin::planner" instead of type: "planner")
    let has_legacy_planner = config
        .graph
        .nodes
        .iter()
        .any(|n| n.handler == "builtin::planner");
    let has_planner_type = config
        .graph
        .nodes
        .iter()
        .any(|n| n.node_type.as_deref() == Some("planner"));
    if has_legacy_planner || has_planner_type {
        let llm_client = config.create_llm_client()?;
        let prompt_template = config.graph.planner.prompt_template.clone();
        let planner_fn = Arc::new(crate::core::node::planner::create_planner_handler(
            llm_client,
            prompt_template,
        ));
        let arc_handler: ArcHandler<DynState> = Arc::new(move |state| planner_fn.as_ref()(state));
        handlers_map.insert("__builtin_planner__".to_string(), arc_handler);
    }

    // Build handler registry keyed by NODE NAME.
    // Dispatch based on node type — every built-in type is detected here.
    let mut node_handlers: HandlerRegistry<DynState> = HashMap::new();
    let mut missing_handlers: Vec<String> = Vec::new();

    // Helper: look up a handler by name, or record it as missing
    macro_rules! require_handler {
        ($name:expr) => {
            match handlers_map.get($name) {
                Some(h) => h.clone(),
                None => {
                    missing_handlers.push($name.to_string());
                    continue;
                }
            }
        };
    }

    for node_config in &config.graph.nodes {
        if node_config.name == "START" || node_config.name == "END" {
            continue;
        }
        // Supervisor nodes are built in a second pass (need child handlers to exist first).
        // Subgraph nodes are self-contained and built in the second pass too.
        match node_config.node_type.as_deref() {
            Some("supervisor") | Some("orchestrator") => continue,
            Some("subgraph") | Some("agent") | Some("agent_or_graph") => continue,
            _ => {}
        }

        let node_name = node_config.name.clone();
        let handler = match node_config.node_type.as_deref() {
            // ── Evaluation: loop until confident ───────────────────────────
            // Standalone (no handler): scores the current state field and writes
            // evaluation metadata. Designed to be used as a graph node with back-edges.
            // Wrapping mode (handler provided): calls the handler repeatedly until
            // min_confidence is reached or max_retries is exhausted.
            Some("evaluation") => {
                use crate::core::node::evaluation_node::EvaluationNodeConfig;
                let cfg = EvaluationNodeConfig::from_node_config(node_config)?;
                if node_config.handler.is_empty() {
                    create_evaluation_standalone_handler(cfg)
                } else {
                    let arc = require_handler!(&node_config.handler);
                    wrap_handler_with_evaluation(arc, cfg)
                }
            }

            // ── Retry: exponential-backoff retry management ─────────────────
            // Standalone: manages __retry_count__ and __retry_should_retry__ in
            // state. Pair with a back-edge and condition on __retry_should_retry__.
            // Wrapping mode: re-calls the handler on error automatically.
            Some("retry") => {
                use crate::core::node::nodes_trait::RetryNodeConfig;
                let cfg = RetryNodeConfig::from_node_config(node_config)?;
                if node_config.handler.is_empty() {
                    create_retry_standalone_handler(cfg)
                } else {
                    let arc = require_handler!(&node_config.handler);
                    wrap_handler_with_retry(arc, cfg)
                }
            }

            // ── Timeout: wall-clock deadline tracking ───────────────────────
            // Standalone: sets __timeout_deadline__ on first visit, checks it on
            // subsequent visits. Sets __timeout_timed_out__ for routing.
            // Wrapping mode: aborts the handler if it exceeds the duration.
            Some("timeout") => {
                use crate::core::node::nodes_trait::TimeoutNodeConfig;
                let cfg = TimeoutNodeConfig::from_node_config(node_config)?;
                if node_config.handler.is_empty() {
                    create_timeout_standalone_handler(cfg)
                } else {
                    let arc = require_handler!(&node_config.handler);
                    wrap_handler_with_timeout(arc, cfg)
                }
            }

            // ── Loop: iteration counter management ──────────────────────────
            // Standalone: tracks __loop_iteration__ and __loop_continue__ in state.
            // Pair with a back-edge conditioned on __loop_continue__.
            // Wrapping mode: runs the handler up to max_iterations times inline.
            Some("loop") => {
                use crate::core::node::advanced_nodes::LoopNodeConfig;
                let cfg = LoopNodeConfig::from_node_config(node_config)?;
                if node_config.handler.is_empty() {
                    create_loop_standalone_handler(cfg)
                } else {
                    let arc = require_handler!(&node_config.handler);
                    wrap_handler_with_loop(arc, cfg)
                }
            }

            // ── Planner: LLM-driven next-node selection (no user handler) ───
            // `type: planner` always uses the builtin planner regardless of the handler field.
            // `handler: "builtin::planner"` on a typeless node is the legacy spelling.
            // NOTE: the guard `if` on a `|` pattern applies to ALL alternatives, so these
            // two cases must be separate arms to avoid the guard blocking `Some("planner")`.
            Some("planner") => {
                let arc = handlers_map
                    .get("__builtin_planner__")
                    .cloned()
                    .expect("planner was pre-injected");
                Box::new(move |state| arc(state))
            }
            None if node_config.handler == "builtin::planner" => {
                let arc = handlers_map
                    .get("__builtin_planner__")
                    .cloned()
                    .expect("planner was pre-injected");
                Box::new(move |state| arc(state))
            }

            // ── Human-in-the-loop: pause for human approval/edit ───────────
            Some("human_in_the_loop") => {
                use crate::core::node::nodes_trait::HumanInTheLoopConfig;
                let cfg = HumanInTheLoopConfig::from_node_config(node_config)?;
                create_human_in_loop_handler(cfg)
            }

            // ── Memory: built-in memory operations ─────────────────────────
            Some("memory") => {
                let op = node_config
                    .config
                    .get("operation")
                    .and_then(|v| v.as_str())
                    .unwrap_or("");
                match create_memory_handler(op) {
                    Some(h) => h,
                    None => {
                        return Err(FlowgentraError::ConfigError(format!(
                            "Unknown memory operation '{}' for node '{}'. \
                         Valid operations: append_message, compress_history, \
                         clear_history, get_message_count, format_history_for_context",
                            op, node_config.name
                        )))
                    }
                }
            }

            // ── Plain handler (or unrecognized type: falls back to handler) ─
            _ => {
                let arc = require_handler!(&node_config.handler);
                Box::new(move |state| arc(state))
            }
        };

        node_handlers.insert(node_name, handler);
    }

    // ── Second pass: supervisor + subgraph nodes ────────────────────────────────
    // Convert first-pass handlers to Arc so supervisor children can be shared
    // without being consumed (each child must remain a standalone graph node too).
    let node_handlers_arc: HashMap<String, ArcHandler<DynState>> = node_handlers
        .into_iter()
        .map(|(name, h)| {
            let arc: ArcHandler<DynState> = Arc::new(move |state| h(state));
            (name, arc)
        })
        .collect();

    let mut second_pass_handlers: HandlerRegistry<DynState> = HashMap::new();

    // ── Sub-pass A: index all subgraph configs by node name ───────────────────
    // create_subgraph_handler is cheap (YAML loads only at execution time),
    // so we build on demand in sub-pass B rather than pre-building.
    use crate::core::node::agent_or_graph_node::SubgraphNodeConfig;
    let subgraph_configs: HashMap<String, SubgraphNodeConfig> = config
        .graph
        .nodes
        .iter()
        .filter(|n| {
            matches!(
                n.node_type.as_deref(),
                Some("subgraph") | Some("agent") | Some("agent_or_graph")
            )
        })
        .map(|n| {
            let mut cfg = SubgraphNodeConfig::from_node_config(n)?;
            // Resolve subgraph path relative to the parent config file's directory
            if !std::path::Path::new(&cfg.path).is_absolute() {
                cfg.path = config_dir.join(&cfg.path).to_string_lossy().to_string();
            }
            Ok((n.name.clone(), cfg))
        })
        .collect::<Result<_>>()?;

    // Register all subgraph nodes as standalone graph nodes too
    for (name, cfg) in &subgraph_configs {
        second_pass_handlers.insert(name.clone(), create_subgraph_handler(cfg.clone()));
    }

    // ── Sub-pass B: build supervisor handlers ─────────────────────────────────
    // Child resolution order:
    //   1. first-pass plain handlers  (node_handlers_arc)
    //   2. subgraph nodes             (subgraph_configs — built fresh per supervisor)
    //   3. already-built supervisor handlers (built_supervisor_arcs — enables nesting)
    //   4. inventory handlers         (handlers_map)
    //
    // Supervisors are built in multiple passes so that parent supervisors can
    // reference child supervisors that were built in an earlier pass.
    let supervisor_nodes: Vec<_> = config
        .graph
        .nodes
        .iter()
        .filter(|n| {
            matches!(
                n.node_type.as_deref(),
                Some("supervisor") | Some("orchestrator")
            )
        })
        .collect();

    let mut built_supervisor_arcs: HashMap<String, ArcHandler<DynState>> = HashMap::new();
    let mut remaining: Vec<_> = supervisor_nodes.iter().map(|n| n.name.clone()).collect();
    let max_passes = remaining.len() + 1; // guard against infinite loops

    for _pass in 0..max_passes {
        if remaining.is_empty() {
            break;
        }
        let mut still_remaining = Vec::new();

        for sup_name in &remaining {
            let node_config = supervisor_nodes
                .iter()
                .find(|n| &n.name == sup_name)
                .unwrap();
            use crate::core::node::orchestrator_node::SupervisorNodeConfig;
            let cfg = SupervisorNodeConfig::from_node_config(node_config)?;

            let mut child_arcs: Vec<(String, ArcHandler<DynState>)> = Vec::new();
            let mut all_resolved = true;

            for child_name in &cfg.children {
                if let Some(arc) = node_handlers_arc.get(child_name).cloned() {
                    child_arcs.push((child_name.clone(), arc));
                } else if let Some(sub_cfg) = subgraph_configs.get(child_name) {
                    let handler = create_subgraph_handler(sub_cfg.clone());
                    child_arcs.push((child_name.clone(), Arc::new(move |state| handler(state))));
                } else if let Some(arc) = built_supervisor_arcs.get(child_name).cloned() {
                    // child is a supervisor built in a previous pass
                    child_arcs.push((child_name.clone(), arc));
                } else if let Some(arc) = handlers_map.get(child_name).cloned() {
                    child_arcs.push((child_name.clone(), arc));
                } else {
                    // child not yet available — might be built in a later pass
                    all_resolved = false;
                    break;
                }
            }

            if all_resolved {
                // Build per-child MCP map from node configs
                let child_mcps: HashMap<String, Vec<String>> = cfg
                    .children
                    .iter()
                    .filter_map(|name| {
                        let node = config.graph.nodes.iter().find(|n| &n.name == name)?;
                        if node.mcps.is_empty() {
                            None
                        } else {
                            Some((name.clone(), node.mcps.clone()))
                        }
                    })
                    .collect();

                let handler = if matches!(
                    cfg.strategy,
                    crate::core::node::orchestrator_node::OrchestrationStrategy::Dynamic
                ) {
                    let llm = config
                        .create_llm_client()
                        .ok()
                        .map(|c| c as Arc<dyn LLMClient>);
                    create_supervisor_handler_with_llm(cfg, child_arcs, llm, child_mcps)
                } else {
                    create_supervisor_handler(cfg, child_arcs, child_mcps)
                };
                let arc: ArcHandler<DynState> = Arc::new(move |state| handler(state));
                built_supervisor_arcs.insert(sup_name.clone(), arc);
                second_pass_handlers.insert(sup_name.clone(), {
                    let arc = built_supervisor_arcs.get(sup_name).unwrap().clone();
                    Box::new(move |state| arc(state))
                });
            } else {
                still_remaining.push(sup_name.clone());
            }
        }

        if still_remaining.len() == remaining.len() {
            // No progress — remaining supervisors have unresolvable children
            for sup_name in &still_remaining {
                let node_config = supervisor_nodes
                    .iter()
                    .find(|n| &n.name == sup_name)
                    .unwrap();
                use crate::core::node::orchestrator_node::SupervisorNodeConfig;
                let cfg = SupervisorNodeConfig::from_node_config(node_config)?;
                for child_name in &cfg.children {
                    if !node_handlers_arc.contains_key(child_name)
                        && !subgraph_configs.contains_key(child_name)
                        && !built_supervisor_arcs.contains_key(child_name)
                        && !handlers_map.contains_key(child_name)
                    {
                        missing_handlers.push(child_name.clone());
                    }
                }
            }
            break;
        }

        remaining = still_remaining;
    }

    // Merge both passes back into a single HandlerRegistry
    let mut node_handlers: HandlerRegistry<DynState> = node_handlers_arc
        .into_iter()
        .map(|(name, arc)| {
            let h: Handler<DynState> = Box::new(move |state| arc(state));
            (name, h)
        })
        .collect();
    node_handlers.extend(second_pass_handlers);

    if !missing_handlers.is_empty() {
        missing_handlers.dedup();
        let missing = missing_handlers.join(", ");
        let registered = handlers_map
            .keys()
            .filter(|k| !k.starts_with("__builtin"))
            .map(|k: &String| k.as_str())
            .collect::<Vec<_>>()
            .join(", ");

        let msg = if registered.is_empty() {
            format!(
                "Missing handlers: {}. No handlers registered yet.\n\
                 Use #[register_handler] attribute on your handler functions to register them.\n\
                 Example:\n  #[register_handler]\n  pub async fn {} (state: State) -> Result<State> {{ ... }}",
                missing,
                missing_handlers[0]
            )
        } else {
            format!(
                "Missing handlers: {}\n\
                 Registered handlers: {}\n\
                 Use #[register_handler] attribute to register missing handlers.",
                missing, registered
            )
        };

        return Err(crate::core::error::FlowgentraError::ConfigError(msg));
    }

    // Pass handlers keyed by node name — from_config_inner will match by node.name
    let mut agent = Agent::from_config_inner(config.clone(), node_handlers, HashMap::new())?;

    // Apply memory from config (checkpointer, conversation memory, buffer/window)
    if config
        .memory
        .checkpointer
        .type_
        .eq_ignore_ascii_case("memory")
    {
        agent.set_checkpointer(Arc::new(MemoryCheckpointer::new()));
    }
    if config.memory.conversation.enabled {
        let conv = InMemoryConversationMemory::with_config(&config.memory.buffer);
        agent.set_conversation_memory(Arc::new(conv));
    }

    // Apply evaluation from config (auto-add AutoEvaluationMiddleware)
    if let Some(eval_config) = &config.evaluation {
        if eval_config.enabled {
            use crate::core::evaluation::{
                AutoEvaluationMiddleware, ConfidenceConfig, LegacyEvaluationPolicy, RetryConfig,
                ScoringCriteria,
            };

            let policy = LegacyEvaluationPolicy {
                enable_scoring: true,
                enable_grading: eval_config
                    .grading
                    .as_ref()
                    .map(|g| g.enabled)
                    .unwrap_or(false),
                enable_confidence_scoring: true,
                confidence_threshold: eval_config.min_confidence,
                max_retries: eval_config.max_retries,
                enable_self_correction: true,
                store_evaluation_history: true,
            };

            let scoring_criteria = ScoringCriteria::default();

            let confidence_config = ConfidenceConfig {
                low_threshold: eval_config.min_confidence * 0.6,
                high_threshold: eval_config.min_confidence,
                ..Default::default()
            };

            let retry_config = RetryConfig {
                max_retries: eval_config.max_retries,
                confidence_threshold: eval_config.min_confidence,
                ..Default::default()
            };

            let middleware = AutoEvaluationMiddleware::new()
                .with_policy(policy)
                .with_scoring_criteria(scoring_criteria)
                .with_confidence_config(confidence_config)
                .with_retry_config(retry_config);

            agent.runtime_mut().add_middleware(Arc::new(middleware));
        }
    }

    Ok(agent)
}

// Memory-aware agent wrapper for simplified memory handling
mod memory_aware;
pub use memory_aware::{MemoryAwareAgent, MemoryStats};

// =============================================================================
// Built-in Node Handler Wrappers
//
// Each function converts a built-in node config into a Handler<DynState>
// so every node type is uniformly represented as a NodeFunction in the runtime.
// =============================================================================

/// Wraps a handler in an evaluation retry loop.
/// Delegates to `EvaluationNodeConfig::into_wrapping_node_fn` — single source of truth.
fn wrap_handler_with_evaluation(
    handler: ArcHandler<DynState>,
    eval_config: crate::core::node::evaluation_node::EvaluationNodeConfig,
) -> Handler<DynState> {
    eval_config.into_wrapping_node_fn(handler)
}

// ── Retry ──────────────────────────────────────────────────────────────────
/// Wraps a handler in an exponential-backoff retry loop.
/// Re-calls the handler on error up to `max_retries` times.
///
/// YAML:
/// ```yaml
/// - name: call_api
///   type: retry
///   handler: call_api_handler
///   config:
///     max_retries: 3
///     backoff_ms: 1000
///     backoff_multiplier: 2.0
///     max_backoff_ms: 30000
/// ```
fn wrap_handler_with_retry(
    handler: ArcHandler<DynState>,
    config: crate::core::node::nodes_trait::RetryNodeConfig,
) -> Handler<DynState> {
    Box::new(move |state| {
        let handler = handler.clone();
        let config = config.clone();
        Box::pin(async move {
            let mut last_err: Option<FlowgentraError> = None;

            for attempt in 0..=config.max_retries {
                match handler(state.clone()).await {
                    Ok(new_state) => return Ok(new_state),
                    Err(e) => {
                        tracing::warn!(
                            "Retry '{}' attempt {}/{} failed: {}",
                            config.name,
                            attempt + 1,
                            config.max_retries,
                            e
                        );
                        last_err = Some(e);

                        if attempt < config.max_retries {
                            let backoff = if attempt == 0 {
                                config.backoff_ms
                            } else {
                                let exp = config.backoff_ms as f64
                                    * (config.backoff_multiplier as f64).powi(attempt as i32);
                                (exp as u64).min(config.max_backoff_ms)
                            };
                            tokio::time::sleep(std::time::Duration::from_millis(backoff)).await;
                        }
                    }
                }
            }

            Err(last_err.unwrap_or_else(|| {
                FlowgentraError::RuntimeError(format!(
                    "Retry '{}' exhausted {} attempts",
                    config.name, config.max_retries
                ))
            }))
        })
    })
}

// ── Timeout ────────────────────────────────────────────────────────────────
/// Wraps a handler with a wall-clock timeout.
/// On timeout, behaviour is controlled by `on_timeout`:
/// - `"error"` (default) → return an error
/// - `"skip"` → return state unchanged
/// - `"default_value"` → set `_timeout_default` in state and return ok
///
/// YAML:
/// ```yaml
/// - name: slow_op
///   type: timeout
///   handler: slow_handler
///   config:
///     timeout_ms: 5000
///     on_timeout: "error"   # or "skip" or "default_value"
/// ```
fn wrap_handler_with_timeout(
    handler: ArcHandler<DynState>,
    config: crate::core::node::nodes_trait::TimeoutNodeConfig,
) -> Handler<DynState> {
    Box::new(move |state| {
        let handler = handler.clone();
        let config = config.clone();
        Box::pin(async move {
            let duration = std::time::Duration::from_millis(config.timeout_ms);
            match tokio::time::timeout(duration, handler(state.clone())).await {
                Ok(Ok(new_state)) => Ok(new_state),
                Ok(Err(e)) => Err(e),
                Err(_elapsed) => match config.on_timeout.as_str() {
                    "skip" => Ok(state),
                    "default_value" => {
                        if let Some(default) = &config.default_value {
                            state.set("_timeout_default", default.clone());
                        }
                        Ok(state)
                    }
                    _ => Err(FlowgentraError::ExecutionTimeout(format!(
                        "Node '{}' timed out after {}ms",
                        config.name, config.timeout_ms
                    ))),
                },
            }
        })
    })
}

// ── Loop ───────────────────────────────────────────────────────────────────
/// Wraps a handler in a fixed-iteration loop.
/// Exits early when the state field named `break_condition` is true.
///
/// YAML:
/// ```yaml
/// - name: iterate_process
///   type: loop
///   handler: process_step
///   config:
///     max_iterations: 5
///     break_condition: "is_done"   # name of a bool state field
/// ```
fn wrap_handler_with_loop(
    handler: ArcHandler<DynState>,
    config: crate::core::node::advanced_nodes::LoopNodeConfig,
) -> Handler<DynState> {
    Box::new(move |state| {
        let handler = handler.clone();
        let config = config.clone();
        Box::pin(async move {
            let mut state = state;

            for iteration in 0..config.max_iterations {
                tracing::info!(
                    "Loop '{}' iteration {}/{}",
                    config.handler,
                    iteration + 1,
                    config.max_iterations
                );
                state = handler(state).await?;

                if let Some(ref cond) = config.break_condition {
                    if state.get(cond).and_then(|v| v.as_bool()).unwrap_or(false) {
                        tracing::info!(
                            "Loop '{}' break condition '{}' met at iteration {}",
                            config.handler,
                            cond,
                            iteration + 1
                        );
                        break;
                    }
                }
            }

            Ok(state)
        })
    })
}

// ── Human-in-the-Loop ─────────────────────────────────────────────────────
/// Simulates a human approval checkpoint.
/// Sets `_human_approved = true` and `_human_node` in state.
/// (In production, replace with a real interactive or webhook mechanism.)
///
/// YAML:
/// ```yaml
/// - name: approval
///   type: human_in_the_loop
///   config:
///     prompt: "Please approve this action"
///     require_approval: true
///     editable_fields: ["amount", "recipient"]
/// ```
fn create_human_in_loop_handler(
    config: crate::core::node::nodes_trait::HumanInTheLoopConfig,
) -> Handler<DynState> {
    Box::new(move |state| {
        let config = config.clone();
        Box::pin(async move {
            tracing::info!("Human-in-the-loop '{}': {}", config.name, config.prompt);
            state.set("_human_approved", serde_json::json!(true));
            state.set("_human_node", serde_json::json!(config.name));
            if !config.editable_fields.is_empty() {
                state.set(
                    "_human_editable_fields",
                    serde_json::json!(config.editable_fields),
                );
            }
            Ok(state)
        })
    })
}

// ── Memory ─────────────────────────────────────────────────────────────────
/// Creates a memory operation handler by operation name.
///
/// YAML:
/// ```yaml
/// - name: append_msg
///   type: memory
///   config:
///     operation: append_message   # or compress_history, clear_history,
///                                 #    get_message_count, format_history_for_context
/// ```
fn create_memory_handler(operation: &str) -> Option<Handler<DynState>> {
    use crate::core::node::memory_handlers;
    match operation {
        "append_message" => Some(Box::new(|state| {
            Box::pin(memory_handlers::append_message_handler(state))
        })),
        "compress_history" => Some(Box::new(|state| {
            Box::pin(memory_handlers::compress_history_handler(state))
        })),
        "clear_history" => Some(Box::new(|state| {
            Box::pin(memory_handlers::clear_history_handler(state))
        })),
        "get_message_count" => Some(Box::new(|state| {
            Box::pin(memory_handlers::get_message_count_handler(state))
        })),
        "format_history_for_context" => Some(Box::new(|state| {
            Box::pin(memory_handlers::format_history_for_context_handler(state))
        })),
        _ => None,
    }
}

// =============================================================================
// Standalone Built-in Node Handlers
//
// These create self-contained handlers that manage state flags so the caller
// can route the graph based on those flags (e.g. with conditional back-edges).
// No user handler is required — they are fully implemented by the library.
// =============================================================================

// ── Standalone Evaluation ─────────────────────────────────────────────────────
/// Reads the configured `field_state` from the current state, scores it with
/// the built-in heuristic scorer, and writes evaluation metadata to state.
///
/// State keys written:
/// - `__eval_score__<name>`        – current numeric score (0.0–1.0)
/// - `__eval_feedback__<name>`     – textual feedback for the next handler
/// - `__eval_needs_retry__<name>`  – bool, true when score < min_confidence
///   AND attempt < max_retries
/// - `__eval_attempt__<name>`      – current attempt counter (starts at 1)
/// - `__eval_meta__<name>`         – full JSON object with all metadata
///
/// Pair with a conditional back-edge on `__eval_needs_retry__<name>` to retry.
///
/// YAML:
/// ```yaml
/// - name: score_output
///   type: evaluation
///   config:
///     field_state: llm_output
///     min_confidence: 0.80
///     max_retries: 3
///     rubric: "Is the output clear and accurate?"
/// ```
/// Delegates to `EvaluationNodeConfig::into_standalone_node_fn` — single source of truth.
fn create_evaluation_standalone_handler(
    config: crate::core::node::evaluation_node::EvaluationNodeConfig,
) -> Handler<DynState> {
    config.into_standalone_node_fn()
}

// ── Standalone Retry ──────────────────────────────────────────────────────────
/// Manages a retry counter and computes exponential backoff in state.
/// Does NOT call any user handler — use this as a gate node in your graph.
///
/// State keys written:
/// - `__retry_count__<name>`        – current attempt number (starts at 1)
/// - `__retry_should_retry__<name>` – bool, true while count < max_retries
/// - `__retry_meta__<name>`         – JSON with attempt / backoff / should_retry
///
/// Pair with a conditional back-edge on `__retry_should_retry__<name>`.
///
/// YAML:
/// ```yaml
/// - name: retry_gate
///   type: retry
///   config:
///     max_retries: 3
///     backoff_ms: 500
///     backoff_multiplier: 2.0
///     max_backoff_ms: 10000
/// ```
fn create_retry_standalone_handler(
    config: crate::core::node::nodes_trait::RetryNodeConfig,
) -> Handler<DynState> {
    use serde_json::json;

    Box::new(move |state| {
        let config = config.clone();
        Box::pin(async move {
            let count_key = format!("__retry_count__{}", config.name);
            let should_retry_key = format!("__retry_should_retry__{}", config.name);

            let current = state.get(&count_key).and_then(|v| v.as_u64()).unwrap_or(0) as usize;

            let next = current + 1;
            let should_retry = current < config.max_retries;

            // Exponential backoff for the current attempt
            let backoff_ms = if current == 0 {
                config.backoff_ms
            } else {
                let exp = config.backoff_ms as f64
                    * (config.backoff_multiplier as f64).powi(current as i32);
                (exp as u64).min(config.max_backoff_ms)
            };

            state.set(&count_key, json!(next));
            state.set(&should_retry_key, json!(should_retry));
            state.set(
                format!("__retry_meta__{}", config.name),
                json!({
                    "attempt": next,
                    "max_retries": config.max_retries,
                    "should_retry": should_retry,
                    "backoff_ms": backoff_ms,
                }),
            );

            if should_retry {
                tracing::info!(
                    "Standalone retry '{}' attempt {}/{}, sleeping {}ms",
                    config.name,
                    next,
                    config.max_retries,
                    backoff_ms
                );
                tokio::time::sleep(std::time::Duration::from_millis(backoff_ms)).await;
            } else {
                tracing::info!(
                    "Standalone retry '{}' exhausted {} attempts — no more retries",
                    config.name,
                    config.max_retries
                );
            }

            Ok(state)
        })
    })
}

// ── Standalone Timeout ────────────────────────────────────────────────────────
/// Tracks a wall-clock deadline in state.
/// On first visit, records `now + timeout_ms` as the deadline.
/// On each subsequent visit, checks whether the deadline has passed.
///
/// State keys written:
/// - `__timeout_deadline__<name>` – UNIX-ms deadline (set once on first visit)
/// - `__timeout_timed_out__<name>` – bool, true once deadline passes
/// - `__timeout_meta__<name>`      – JSON with deadline / elapsed info
///
/// On timeout, behaviour mirrors the wrapping variant:
/// - `on_timeout: "error"` → returns an error
/// - `on_timeout: "skip"`  → continues silently
/// - `on_timeout: "default_value"` → writes `_timeout_default` to state
///
/// YAML:
/// ```yaml
/// - name: check_deadline
///   type: timeout
///   config:
///     timeout_ms: 30000
///     on_timeout: "skip"
/// ```
fn create_timeout_standalone_handler(
    config: crate::core::node::nodes_trait::TimeoutNodeConfig,
) -> Handler<DynState> {
    use serde_json::json;

    Box::new(move |state| {
        let config = config.clone();
        Box::pin(async move {
            let deadline_key = format!("__timeout_deadline__{}", config.name);
            let timed_out_key = format!("__timeout_timed_out__{}", config.name);

            let now_ms = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as u64;

            let deadline_ms = match state.get(&deadline_key).and_then(|v| v.as_u64()) {
                Some(d) => d,
                None => {
                    // First visit: record the deadline, not yet timed out
                    let d = now_ms + config.timeout_ms;
                    state.set(&deadline_key, json!(d));
                    state.set(&timed_out_key, json!(false));
                    state.set(
                        format!("__timeout_meta__{}", config.name),
                        json!({ "timed_out": false, "deadline_ms": d, "current_ms": now_ms }),
                    );
                    return Ok(state);
                }
            };

            let timed_out = now_ms > deadline_ms;
            state.set(&timed_out_key, json!(timed_out));
            state.set(
                format!("__timeout_meta__{}", config.name),
                json!({
                    "timed_out": timed_out,
                    "timeout_ms": config.timeout_ms,
                    "deadline_ms": deadline_ms,
                    "current_ms": now_ms,
                    "elapsed_ms": now_ms.saturating_sub(deadline_ms - config.timeout_ms),
                }),
            );

            if timed_out {
                tracing::warn!(
                    "Standalone timeout '{}' expired (deadline={}ms, now={}ms)",
                    config.name,
                    deadline_ms,
                    now_ms
                );
                match config.on_timeout.as_str() {
                    "error" => {
                        return Err(FlowgentraError::ExecutionTimeout(format!(
                            "Node '{}' timed out after {}ms",
                            config.name, config.timeout_ms
                        )))
                    }
                    "default_value" => {
                        if let Some(default) = &config.default_value {
                            state.set("_timeout_default", default.clone());
                        }
                    }
                    _ => {} // "skip" — continue silently
                }
            }

            Ok(state)
        })
    })
}

// ── Standalone Loop ───────────────────────────────────────────────────────────
/// Manages an iteration counter in state.
/// Use this as a loop-gate node with a conditional back-edge on
/// `__loop_continue__<name>` to build explicit loop subgraphs.
///
/// State keys written:
/// - `__loop_iteration__<name>` – current iteration (starts at 1)
/// - `__loop_continue__<name>`  – bool, true while within max_iterations
///   AND break_condition is not set in state
/// - `__loop_meta__<name>`      – JSON with full iteration metadata
///
/// YAML:
/// ```yaml
/// - name: loop_gate
///   type: loop
///   config:
///     max_iterations: 5
///     break_condition: "is_done"   # optional bool state field
/// ```
// ── Supervisor ────────────────────────────────────────────────────────────────
/// Coordinates multiple child handlers (or subgraphs) as a single node.
///
/// This is the canonical multi-agent orchestration pattern:
/// a Supervisor delegates to sub-agents, manages their execution, and aggregates results.
///
/// Sequential: calls children one after another; state flows through each.
/// Parallel: calls all children concurrently; merges results per `merge_strategy`.
///
/// State written:
/// - `__supervisor_meta__<name>` — per-child results, errors, timing
///
/// YAML:
/// ```yaml
/// - name: research_coordinator
///   type: supervisor           # alias: orchestrator
///   config:
///     strategy: sequential     # or "parallel"
///     children: [research_agent, writer_agent, critic_agent]
///     fail_fast: true
///     child_timeout_ms: 30000
///     collect_stats: true
///     merge_strategy: latest   # parallel only — or "first_success", "deep_merge"
///     parallel_aggregation: all  # or "majority", "first_success"
/// ```
fn create_supervisor_handler(
    config: crate::core::node::orchestrator_node::SupervisorNodeConfig,
    children: Vec<(String, ArcHandler<DynState>)>,
    child_mcps: HashMap<String, Vec<String>>,
) -> Handler<DynState> {
    create_supervisor_handler_with_llm(config, children, None, child_mcps)
}

fn create_supervisor_handler_with_llm(
    config: crate::core::node::orchestrator_node::SupervisorNodeConfig,
    children: Vec<(String, ArcHandler<DynState>)>,
    llm_client: Option<Arc<dyn LLMClient>>,
    child_mcps: HashMap<String, Vec<String>>,
) -> Handler<DynState> {
    use crate::core::node::orchestrator_node::{OrchestrationStrategy, ParallelMergeStrategy};
    use serde_json::json;

    // Evaluate a skip condition expression against state.
    // Supported forms:
    //   "key"            — skip if state[key] is non-null and non-false
    //   "key != null"    — skip if state[key] is non-null
    //   "key == null"    — skip if state[key] is null/absent
    //   "key == value"   — skip if state[key].to_string() == value
    fn should_skip(condition: &str, state: &DynState) -> bool {
        let condition = condition.trim();
        if condition.contains("!=") {
            let parts: Vec<&str> = condition.splitn(2, "!=").collect();
            let key = parts[0].trim();
            let rhs = parts[1].trim();
            let val = state.get(key);
            if rhs == "null" {
                return val.is_some() && !val.unwrap().is_null();
            }
            return val
                .map(|v| v.to_string().trim_matches('"') != rhs)
                .unwrap_or(false);
        }
        if condition.contains("==") {
            let parts: Vec<&str> = condition.splitn(2, "==").collect();
            let key = parts[0].trim();
            let rhs = parts[1].trim();
            let val = state.get(key);
            if rhs == "null" {
                return val.is_none() || val.unwrap().is_null();
            }
            return val
                .map(|v| v.to_string().trim_matches('"') == rhs)
                .unwrap_or(false);
        }
        // bare key: skip if truthy
        state
            .get(condition)
            .map(|v| !v.is_null() && v.as_bool() != Some(false))
            .unwrap_or(false)
    }

    // Run a single child with optional timeout, returning (result, duration_ms).
    async fn run_child(
        name: &str,
        handler: &ArcHandler<DynState>,
        state: DynState,
        timeout_ms: Option<u64>,
    ) -> (crate::core::error::Result<DynState>, u128) {
        let child_start = std::time::Instant::now();
        let result = if let Some(ms) = timeout_ms {
            tokio::time::timeout(std::time::Duration::from_millis(ms), handler(state))
                .await
                .unwrap_or_else(|_| {
                    Err(FlowgentraError::ExecutionTimeout(format!(
                        "Child '{}' timed out after {}ms",
                        name, ms
                    )))
                })
        } else {
            handler(state).await
        };
        (result, child_start.elapsed().as_millis())
    }

    Box::new(move |state| {
        let config = config.clone();
        let children = children.clone();
        let llm_client = llm_client.clone();
        let child_mcps = child_mcps.clone();
        Box::pin(async move {
            let start = std::time::Instant::now();

            // Inject per-node MCP assignments into state before running a child
            let inject_mcps = |child_name: &str, state: &DynState| {
                if let Some(mcps) = child_mcps.get(child_name) {
                    state.set("_node_mcps", serde_json::json!(mcps));
                } else {
                    state.remove("_node_mcps");
                }
            };

            match &config.strategy {
                // ── Sequential ──────────────────────────────────────────────
                OrchestrationStrategy::Sequential => {
                    let mut current = state;
                    let mut child_results = Vec::new();
                    let mut errors: Vec<String> = Vec::new();

                    for (name, handler) in &children {
                        // ── Skip condition ──────────────────────────────────
                        if let Some(cond) = config.skip_conditions.get(name.as_str()) {
                            if should_skip(cond, &current) {
                                tracing::info!(
                                    "Supervisor '{}': skipping '{}' (condition: {cond})",
                                    config.name,
                                    name
                                );
                                child_results.push(json!({
                                    "name": name, "skipped": true, "condition": cond
                                }));
                                continue;
                            }
                        }

                        // ── Run with retry ──────────────────────────────────
                        let max_attempts = config.max_retries_per_child + 1;
                        let mut last_err = String::new();
                        let mut succeeded = false;
                        let mut total_ms = 0u128;

                        inject_mcps(name, &current);
                        for attempt in 1..=max_attempts {
                            let (result, ms) =
                                run_child(name, handler, current.clone(), config.child_timeout_ms)
                                    .await;
                            total_ms += ms;
                            match result {
                                Ok(new_state) => {
                                    current = new_state;
                                    succeeded = true;
                                    if attempt > 1 {
                                        tracing::info!(
                                            "Supervisor '{}': child '{}' succeeded on attempt {attempt}",
                                            config.name, name
                                        );
                                    }
                                    break;
                                }
                                Err(e) => {
                                    last_err = e.to_string();
                                    if attempt < max_attempts {
                                        tracing::warn!(
                                            "Supervisor '{}': child '{}' failed (attempt {attempt}/{max_attempts}): {e}",
                                            config.name, name
                                        );
                                    }
                                }
                            }
                        }

                        if succeeded {
                            child_results.push(json!({
                                "name": name,
                                "success": true,
                                "duration_ms": total_ms,
                                "attempts": (total_ms > 0) as u8,  // always 1+ if here
                            }));
                        } else {
                            child_results.push(json!({
                                "name": name,
                                "success": false,
                                "error": last_err,
                                "duration_ms": total_ms,
                                "attempts": max_attempts,
                            }));
                            errors.push(format!("'{}': {}", name, last_err));
                            if config.fail_fast {
                                tracing::warn!(
                                    "Supervisor '{}' fail_fast: stopped after '{}' failed after {max_attempts} attempt(s)",
                                    config.name, name
                                );
                                break;
                            }
                        }
                    }

                    current.set(
                        format!("__supervisor_meta__{}", config.name),
                        json!({
                            "strategy": "sequential",
                            "children": child_results,
                            "errors": errors,
                            "success": errors.is_empty(),
                            "duration_ms": start.elapsed().as_millis(),
                        }),
                    );
                    tracing::info!(
                        "Supervisor '{}' sequential done in {}ms, errors={}",
                        config.name,
                        start.elapsed().as_millis(),
                        errors.len()
                    );
                    Ok(current)
                }

                // ── Parallel ────────────────────────────────────────────────
                OrchestrationStrategy::Parallel => {
                    let base_state = Arc::new(state);
                    // Semaphore limits how many children run at the same time.
                    // All futures are spawned immediately; only `max_concurrent` can hold a permit.
                    let semaphore = config
                        .max_concurrent
                        .map(|n| Arc::new(tokio::sync::Semaphore::new(n)));

                    // Filter children by skip conditions, cloning so futures own their data
                    let active_children: Vec<(String, ArcHandler<DynState>)> = children
                        .iter()
                        .filter(|(name, _)| {
                            if let Some(cond) = config.skip_conditions.get(name.as_str()) {
                                if should_skip(cond, &base_state) {
                                    tracing::info!(
                                        "Supervisor '{}': skipping '{}' in parallel (condition: {cond})",
                                        config.name, name
                                    );
                                    return false;
                                }
                            }
                            true
                        })
                        .map(|(n, h)| (n.clone(), h.clone()))
                        .collect();

                    let futures: Vec<_> = active_children
                        .into_iter()
                        .map(|(name, handler)| {
                            let state_copy = base_state.deep_clone();
                            inject_mcps(&name, &state_copy);
                            let timeout_ms = config.child_timeout_ms;
                            let max_attempts = config.max_retries_per_child + 1;
                            let sem = semaphore.clone();
                            async move {
                                // Acquire semaphore permit — blocks until a slot is free
                                let _permit = if let Some(ref sem) = sem {
                                    Some(sem.acquire().await.expect("semaphore closed"))
                                } else {
                                    None
                                };

                                let mut total_ms = 0u128;
                                let mut last_err = String::new();
                                for attempt in 1..=max_attempts {
                                    let (result, ms) = run_child(&name, &handler, state_copy.clone(), timeout_ms).await;
                                    total_ms += ms;
                                    match result {
                                        Ok(s) => {
                                            if attempt > 1 {
                                                tracing::info!(
                                                    "Supervisor parallel: '{}' succeeded on attempt {attempt}",
                                                    name
                                                );
                                            }
                                            return (name, Ok(s), total_ms, attempt);
                                        }
                                        Err(e) => {
                                            last_err = e.to_string();
                                            if attempt < max_attempts {
                                                tracing::warn!(
                                                    "Supervisor parallel: '{}' failed attempt {attempt}/{max_attempts}: {e}",
                                                    name
                                                );
                                            }
                                        }
                                    }
                                }
                                (name, Err(FlowgentraError::ExecutionError(last_err)), total_ms, max_attempts)
                            }
                        })
                        .collect();

                    let results = futures::future::join_all(futures).await;

                    let mut successes: Vec<DynState> = Vec::new();
                    let mut child_results = Vec::new();
                    let mut errors: Vec<String> = Vec::new();

                    for (name, result, duration_ms, attempts) in results {
                        match result {
                            Ok(s) => {
                                child_results.push(json!({
                                    "name": name, "success": true,
                                    "duration_ms": duration_ms, "attempts": attempts,
                                }));
                                successes.push(s);
                            }
                            Err(e) => {
                                child_results.push(json!({
                                    "name": name, "success": false,
                                    "error": e.to_string(),
                                    "duration_ms": duration_ms, "attempts": attempts,
                                }));
                                errors.push(format!("'{}': {}", name, e));
                            }
                        }
                    }

                    // ── Merge ───────────────────────────────────────────────
                    let final_state = match &config.merge_strategy {
                        ParallelMergeStrategy::Latest => successes
                            .into_iter()
                            .last()
                            .unwrap_or_else(|| (*base_state).clone()),

                        ParallelMergeStrategy::FirstSuccess => successes
                            .into_iter()
                            .next()
                            .unwrap_or_else(|| (*base_state).clone()),

                        ParallelMergeStrategy::DeepMerge => {
                            // Start from a fresh clone of base state, then overlay
                            // only keys that each child actually changed.
                            // Each child starts with a deep_clone of base_state, so we
                            // compare each child's values against the base to detect
                            // real modifications — avoiding null overwrites from
                            // schema-initialized keys the child never touched.
                            let base_snapshot: Vec<(String, serde_json::Value)> =
                                base_state.as_map();
                            let merged = (*base_state).clone();
                            for child_state in successes {
                                for (key, value) in child_state.as_map() {
                                    // Skip internal supervisor/planner metadata keys
                                    if key.starts_with("__supervisor_meta__")
                                        || key.starts_with("__eval_")
                                        || key.starts_with("_next_node")
                                    {
                                        continue;
                                    }
                                    // Only merge if the child actually changed this key
                                    let changed =
                                        match base_snapshot.iter().find(|(k, _)| k == &key) {
                                            Some((_, base_val)) => value != *base_val,
                                            None => true, // new key not in base — always merge
                                        };
                                    if changed {
                                        merged.set(key, value);
                                    }
                                }
                            }
                            merged
                        }

                        ParallelMergeStrategy::Custom(_) => successes
                            .into_iter()
                            .last()
                            .unwrap_or_else(|| (*base_state).clone()),
                    };

                    final_state.set(
                        format!("__supervisor_meta__{}", config.name),
                        json!({
                            "strategy": "parallel",
                            "children": child_results,
                            "errors": errors,
                            "success": errors.is_empty(),
                            "duration_ms": start.elapsed().as_millis(),
                        }),
                    );
                    tracing::info!(
                        "Supervisor '{}' parallel done in {}ms, errors={}",
                        config.name,
                        start.elapsed().as_millis(),
                        errors.len()
                    );
                    Ok(final_state)
                }

                // ── Autonomous ──────────────────────────────────────────────
                // The supervisor loops, calling whichever agent owns each missing
                // required output, until all are present or max_iterations is hit.
                OrchestrationStrategy::Autonomous => {
                    let mut current = state;

                    if let Some(goal) = &config.goal {
                        tracing::info!(
                            "Supervisor '{}' autonomous start, goal: {}",
                            config.name,
                            goal
                        );
                    }

                    // Build O(1) lookup: child name → handler
                    let child_map: HashMap<String, ArcHandler<DynState>> = children
                        .iter()
                        .map(|(n, h)| (n.clone(), h.clone()))
                        .collect();

                    let max_iter = if config.max_iterations == 0 {
                        10
                    } else {
                        config.max_iterations
                    };
                    let mut iteration_log = Vec::new();

                    'outer: for iteration in 1..=max_iter {
                        // Which required outputs are still missing?
                        let missing: Vec<String> = config
                            .required_outputs
                            .iter()
                            .filter(|key| {
                                current
                                    .get(key.as_str())
                                    .map(|v| v.is_null())
                                    .unwrap_or(true)
                            })
                            .cloned()
                            .collect();

                        if missing.is_empty() {
                            tracing::info!(
                                "Supervisor '{}' autonomous: all outputs satisfied after {} iterations",
                                config.name, iteration - 1
                            );
                            break;
                        }

                        tracing::info!(
                            "Supervisor '{}' autonomous iteration {}/{}: missing {:?}",
                            config.name,
                            iteration,
                            max_iter,
                            missing
                        );

                        // Collect agents responsible for the missing outputs (deduplicated)
                        let mut agents_to_call: Vec<String> = Vec::new();
                        for key in &missing {
                            if let Some(owner) = config.output_owners.get(key.as_str()) {
                                if !agents_to_call.contains(owner) {
                                    agents_to_call.push(owner.clone());
                                }
                            }
                        }

                        if agents_to_call.is_empty() {
                            tracing::warn!(
                                "Supervisor '{}' autonomous: no owners for missing {:?}, stopping",
                                config.name,
                                missing
                            );
                            break;
                        }

                        let mut iter_child_results = Vec::new();
                        for agent_name in &agents_to_call {
                            if let Some(handler) = child_map.get(agent_name.as_str()) {
                                inject_mcps(agent_name, &current);
                                let max_attempts = config.max_retries_per_child + 1;
                                let mut succeeded = false;
                                let mut total_ms = 0u128;
                                let mut last_err = String::new();

                                for attempt in 1..=max_attempts {
                                    let (result, ms) = run_child(
                                        agent_name,
                                        handler,
                                        current.clone(),
                                        config.child_timeout_ms,
                                    )
                                    .await;
                                    total_ms += ms;
                                    match result {
                                        Ok(new_state) => {
                                            current = new_state;
                                            succeeded = true;
                                            break;
                                        }
                                        Err(e) => {
                                            last_err = e.to_string();
                                            if attempt < max_attempts {
                                                tracing::warn!(
                                                    "Supervisor '{}' autonomous: '{}' failed attempt {}/{}: {}",
                                                    config.name, agent_name, attempt, max_attempts, e
                                                );
                                            }
                                        }
                                    }
                                }

                                iter_child_results.push(json!({
                                    "name": agent_name,
                                    "success": succeeded,
                                    "duration_ms": total_ms,
                                    "error": if succeeded { serde_json::Value::Null } else { json!(last_err) },
                                }));

                                if !succeeded && config.fail_fast {
                                    break 'outer;
                                }
                            } else {
                                tracing::warn!(
                                    "Supervisor '{}' autonomous: agent '{}' not found",
                                    config.name,
                                    agent_name
                                );
                            }
                        }

                        iteration_log.push(json!({
                            "iteration": iteration,
                            "missing_before": missing,
                            "agents_called": agents_to_call,
                            "results": iter_child_results,
                        }));
                    }

                    // Final completeness check
                    let final_missing: Vec<String> = config
                        .required_outputs
                        .iter()
                        .filter(|key| {
                            current
                                .get(key.as_str())
                                .map(|v| v.is_null())
                                .unwrap_or(true)
                        })
                        .cloned()
                        .collect();

                    let success = final_missing.is_empty();
                    current.set(
                        format!("__supervisor_meta__{}", config.name),
                        json!({
                            "strategy": "autonomous",
                            "goal": config.goal,
                            "required_outputs": config.required_outputs,
                            "iterations": iteration_log,
                            "missing_outputs": final_missing,
                            "success": success,
                            "duration_ms": start.elapsed().as_millis(),
                        }),
                    );
                    tracing::info!(
                        "Supervisor '{}' autonomous done in {}ms, success={}",
                        config.name,
                        start.elapsed().as_millis(),
                        success
                    );
                    Ok(current)
                }

                // ── Dynamic (LLM-driven) ────────────────────────────────────
                // The supervisor asks an LLM which agents to call, in what order.
                OrchestrationStrategy::Dynamic => {
                    let mut current = state;

                    let child_map: HashMap<String, ArcHandler<DynState>> = children
                        .iter()
                        .map(|(n, h)| (n.clone(), h.clone()))
                        .collect();

                    let child_names: Vec<String> =
                        children.iter().map(|(n, _)| n.clone()).collect();
                    let max_iter = if config.max_iterations == 0 {
                        10
                    } else {
                        config.max_iterations
                    };
                    let mut iteration_log = Vec::new();

                    let llm = llm_client.clone();

                    'dynamic_outer: for iteration in 1..=max_iter {
                        println!("\n  ── Dynamic iteration {iteration}/{max_iter} ──");

                        // Build a filtered view: separate populated vs null keys, hide internals
                        let all_keys: Vec<String> = current.keys();
                        let populated_keys: Vec<&String> = all_keys
                            .iter()
                            .filter(|k| !k.starts_with('_'))
                            .filter(|k| {
                                current
                                    .get(k.as_str())
                                    .map(|v| !v.is_null())
                                    .unwrap_or(false)
                            })
                            .collect();
                        let null_keys: Vec<&String> = all_keys
                            .iter()
                            .filter(|k| !k.starts_with('_'))
                            .filter(|k| {
                                current.get(k.as_str()).map(|v| v.is_null()).unwrap_or(true)
                            })
                            .collect();

                        println!("  [dynamic]  Completed: {populated_keys:?}");
                        println!("  [dynamic]  Missing:   {null_keys:?}");

                        // Ask LLM which agents to call
                        let agents_to_call = if let Some(ref llm) = llm {
                            let system_prompt = config.selector_prompt.as_deref().unwrap_or(
                                "You pick which agents to run. Reply with ONLY a JSON array of agent name strings. \
                                 Example: [\"agent_a\", \"agent_b\"]. If no agents needed, reply []."
                            );
                            let user_prompt = if null_keys.is_empty() {
                                "All outputs are complete. Reply with [].".to_string()
                            } else {
                                format!(
                                    "Goal: {}\n\
                                     Agents: {:?}\n\
                                     Done: {:?}\n\
                                     Still needed: {:?}\n\
                                     Pick the agents to run next. Reply with ONLY a JSON array.",
                                    config.goal.as_deref().unwrap_or("Complete the task"),
                                    child_names,
                                    populated_keys,
                                    null_keys,
                                )
                            };

                            println!("  [dynamic]  Asking LLM...");

                            let messages = vec![
                                crate::core::llm::Message::system(system_prompt),
                                crate::core::llm::Message::user(user_prompt),
                            ];

                            match llm.chat(messages).await {
                                Ok(response) => {
                                    println!(
                                        "  [dynamic]  LLM raw response: {:?}",
                                        response.content
                                    );
                                    let content = response.content.trim();
                                    // Try to extract JSON array from the response
                                    let json_str = if let Some(start) = content.find('[') {
                                        if let Some(end) = content.rfind(']') {
                                            &content[start..=end]
                                        } else {
                                            content
                                        }
                                    } else {
                                        content
                                    };
                                    let parsed = serde_json::from_str::<Vec<String>>(json_str)
                                        .unwrap_or_default();
                                    if parsed.is_empty() && !content.contains("[]") {
                                        println!("  [dynamic]  WARNING: could not parse LLM response as JSON array");
                                    }
                                    parsed
                                }
                                Err(e) => {
                                    println!("  [dynamic]  LLM error: {e}");
                                    child_names.clone()
                                }
                            }
                        } else {
                            // No LLM — fall back to output_owners
                            let missing: Vec<String> = config
                                .required_outputs
                                .iter()
                                .filter(|key| {
                                    current
                                        .get(key.as_str())
                                        .map(|v| v.is_null())
                                        .unwrap_or(true)
                                })
                                .cloned()
                                .collect();

                            if missing.is_empty() {
                                break;
                            }

                            let mut agents = Vec::new();
                            for key in &missing {
                                if let Some(owner) = config.output_owners.get(key.as_str()) {
                                    if !agents.contains(owner) {
                                        agents.push(owner.clone());
                                    }
                                }
                            }
                            if agents.is_empty() {
                                break;
                            }
                            agents
                        };

                        // If LLM returned empty but outputs still missing, use fallback
                        if agents_to_call.is_empty() {
                            let still_missing: Vec<String> = config
                                .required_outputs
                                .iter()
                                .filter(|key| {
                                    current
                                        .get(key.as_str())
                                        .map(|v| v.is_null())
                                        .unwrap_or(true)
                                })
                                .cloned()
                                .collect();

                            if still_missing.is_empty() {
                                println!("  [dynamic]  All required outputs present → done");
                                break;
                            }

                            println!("  [dynamic]  LLM returned [] but still missing: {still_missing:?}, using fallback");
                            let mut fallback = Vec::new();
                            for key in &still_missing {
                                if let Some(owner) = config.output_owners.get(key.as_str()) {
                                    if !fallback.contains(owner)
                                        && child_map.contains_key(owner.as_str())
                                    {
                                        fallback.push(owner.clone());
                                    }
                                }
                            }
                            if fallback.is_empty() {
                                println!("  [dynamic]  No fallback agents, stopping");
                                break;
                            }

                            // Run fallback agents
                            let mut iter_results = Vec::new();
                            for agent_name in &fallback {
                                if let Some(handler) = child_map.get(agent_name.as_str()) {
                                    inject_mcps(agent_name, &current);
                                    println!("  [dynamic]  Running {agent_name}...");
                                    let (result, ms) = run_child(
                                        agent_name,
                                        handler,
                                        current.clone(),
                                        config.child_timeout_ms,
                                    )
                                    .await;
                                    match result {
                                        Ok(new_state) => {
                                            current = new_state;
                                            println!("  [dynamic]  {agent_name} ✓ ({ms}ms)");
                                            iter_results.push(json!({
                                                "name": agent_name, "success": true, "duration_ms": ms,
                                            }));
                                        }
                                        Err(e) => {
                                            println!("  [dynamic]  {agent_name} ✗ ({ms}ms): {e}");
                                            iter_results.push(json!({
                                                "name": agent_name, "success": false,
                                                "error": e.to_string(), "duration_ms": ms,
                                            }));
                                            if config.fail_fast {
                                                break 'dynamic_outer;
                                            }
                                        }
                                    }
                                }
                            }
                            iteration_log.push(json!({
                                "iteration": iteration,
                                "agents_called": fallback,
                                "results": iter_results,
                            }));
                            continue;
                        }

                        println!("  [dynamic]  Calling: {agents_to_call:?}");

                        let mut iter_results = Vec::new();
                        for agent_name in &agents_to_call {
                            if let Some(handler) = child_map.get(agent_name.as_str()) {
                                inject_mcps(agent_name, &current);
                                println!("  [dynamic]  Running {agent_name}...");
                                let (result, ms) = run_child(
                                    agent_name,
                                    handler,
                                    current.clone(),
                                    config.child_timeout_ms,
                                )
                                .await;
                                match result {
                                    Ok(new_state) => {
                                        current = new_state;
                                        println!("  [dynamic]  {agent_name} ✓ ({ms}ms)");
                                        iter_results.push(json!({
                                            "name": agent_name, "success": true, "duration_ms": ms,
                                        }));
                                    }
                                    Err(e) => {
                                        println!("  [dynamic]  {agent_name} ✗ ({ms}ms): {e}");
                                        iter_results.push(json!({
                                            "name": agent_name, "success": false,
                                            "error": e.to_string(), "duration_ms": ms,
                                        }));
                                        if config.fail_fast {
                                            break 'dynamic_outer;
                                        }
                                    }
                                }
                            } else {
                                println!("  [dynamic]  WARNING: agent '{agent_name}' not found");
                            }
                        }

                        iteration_log.push(json!({
                            "iteration": iteration,
                            "agents_called": agents_to_call,
                            "results": iter_results,
                        }));
                    }

                    current.set(
                        format!("__supervisor_meta__{}", config.name),
                        json!({
                            "strategy": "dynamic",
                            "goal": config.goal,
                            "iterations": iteration_log,
                            "duration_ms": start.elapsed().as_millis(),
                        }),
                    );
                    Ok(current)
                }

                // ── RoundRobin ─────────────────────────────────────────────────
                // Tasks from a state array are distributed across agents in rotation.
                OrchestrationStrategy::RoundRobin => {
                    let tasks_key = config.tasks_key.as_deref().unwrap_or("tasks");
                    let tasks: Vec<serde_json::Value> = state
                        .get(tasks_key)
                        .and_then(|v| v.as_array().cloned())
                        .unwrap_or_default();

                    if tasks.is_empty() {
                        tracing::warn!(
                            "Supervisor '{}' round_robin: no tasks at key '{}'",
                            config.name,
                            tasks_key
                        );
                        state.set(
                            format!("__supervisor_meta__{}", config.name),
                            json!({"strategy": "round_robin", "tasks": 0, "success": true}),
                        );
                        return Ok(state);
                    }

                    let mut current = state;
                    let num_children = children.len();
                    let mut all_results: Vec<serde_json::Value> = Vec::new();
                    let mut child_results = Vec::new();
                    let mut errors: Vec<String> = Vec::new();

                    for (i, task) in tasks.iter().enumerate() {
                        let child_idx = i % num_children;
                        let (name, handler) = &children[child_idx];

                        current.set("__current_task__".to_string(), task.clone());
                        current.set("__task_index__".to_string(), json!(i));
                        inject_mcps(name, &current);

                        let (result, ms) =
                            run_child(name, handler, current.clone(), config.child_timeout_ms)
                                .await;
                        match result {
                            Ok(new_state) => {
                                if let Some(r) = new_state.get("__task_result__") {
                                    all_results.push(r);
                                }
                                current = new_state;
                                child_results.push(json!({
                                    "task_index": i, "agent": name, "success": true, "duration_ms": ms,
                                }));
                            }
                            Err(e) => {
                                errors.push(format!("Task {}: {}", i, e));
                                child_results.push(json!({
                                    "task_index": i, "agent": name, "success": false,
                                    "error": e.to_string(), "duration_ms": ms,
                                }));
                                if config.fail_fast {
                                    break;
                                }
                            }
                        }
                    }

                    current.set("__round_robin_results__".to_string(), json!(all_results));
                    current.set(
                        format!("__supervisor_meta__{}", config.name),
                        json!({
                            "strategy": "round_robin",
                            "tasks": tasks.len(),
                            "children": child_results,
                            "errors": errors,
                            "success": errors.is_empty(),
                            "duration_ms": start.elapsed().as_millis(),
                        }),
                    );
                    Ok(current)
                }

                // ── Hierarchical ───────────────────────────────────────────────
                // Delegates to sub-supervisors. Each child is expected to be a
                // supervisor or subgraph managing its own agent group.
                OrchestrationStrategy::Hierarchical => {
                    let mut current = state;
                    let mut child_results = Vec::new();
                    let mut errors: Vec<String> = Vec::new();

                    for (name, handler) in &children {
                        tracing::info!(
                            "Supervisor '{}' hierarchical: delegating to sub-supervisor '{}'",
                            config.name,
                            name
                        );

                        // Skip condition
                        if let Some(cond) = config.skip_conditions.get(name.as_str()) {
                            if should_skip(cond, &current) {
                                child_results.push(json!({"name": name, "skipped": true}));
                                continue;
                            }
                        }

                        inject_mcps(name, &current);
                        let max_attempts = config.max_retries_per_child + 1;
                        let mut succeeded = false;
                        let mut total_ms = 0u128;
                        let mut last_err = String::new();

                        for attempt in 1..=max_attempts {
                            let (result, ms) =
                                run_child(name, handler, current.clone(), config.child_timeout_ms)
                                    .await;
                            total_ms += ms;
                            match result {
                                Ok(new_state) => {
                                    current = new_state;
                                    succeeded = true;
                                    break;
                                }
                                Err(e) => {
                                    last_err = e.to_string();
                                    if attempt < max_attempts {
                                        tracing::warn!(
                                            "Supervisor '{}' hierarchical: sub-supervisor '{}' failed attempt {}/{}",
                                            config.name, name, attempt, max_attempts
                                        );
                                    }
                                }
                            }
                        }

                        child_results.push(json!({
                            "name": name, "success": succeeded,
                            "duration_ms": total_ms,
                            "error": if succeeded { serde_json::Value::Null } else { json!(last_err) },
                        }));

                        if !succeeded {
                            errors.push(format!("Sub-supervisor '{}': {}", name, last_err));
                            if config.fail_fast {
                                break;
                            }
                        }
                    }

                    current.set(
                        format!("__supervisor_meta__{}", config.name),
                        json!({
                            "strategy": "hierarchical",
                            "children": child_results,
                            "errors": errors,
                            "success": errors.is_empty(),
                            "duration_ms": start.elapsed().as_millis(),
                        }),
                    );
                    Ok(current)
                }

                // ── Broadcast (Fan-out) ────────────────────────────────────────
                // Same task sent to all agents; best result selected.
                // Each child gets a deep_clone so writes are isolated.
                OrchestrationStrategy::Broadcast => {
                    let base_state = Arc::new(state);
                    let semaphore = config
                        .max_concurrent
                        .map(|n| Arc::new(tokio::sync::Semaphore::new(n)));

                    let futures: Vec<_> = children
                        .iter()
                        .map(|(name, handler)| {
                            let state_copy = base_state.deep_clone();
                            inject_mcps(name, &state_copy);
                            let timeout_ms = config.child_timeout_ms;
                            let sem = semaphore.clone();
                            let name = name.clone();
                            let handler = handler.clone();
                            async move {
                                let _permit = if let Some(ref sem) = sem {
                                    Some(sem.acquire().await.expect("semaphore closed"))
                                } else {
                                    None
                                };
                                let (result, ms) =
                                    run_child(&name, &handler, state_copy, timeout_ms).await;
                                (name, result, ms)
                            }
                        })
                        .collect();

                    let results = futures::future::join_all(futures).await;

                    let score_key = config.score_key.as_deref().unwrap_or("__score__");
                    let criteria = config
                        .selection_criteria
                        .as_deref()
                        .unwrap_or("first_success");

                    let mut successes: Vec<(String, DynState, u128)> = Vec::new();
                    let mut child_results = Vec::new();
                    let mut errors: Vec<String> = Vec::new();

                    for (name, result, ms) in results {
                        match result {
                            Ok(s) => {
                                child_results.push(json!({
                                    "name": name, "success": true, "duration_ms": ms,
                                }));
                                successes.push((name, s, ms));
                            }
                            Err(e) => {
                                child_results.push(json!({
                                    "name": name, "success": false,
                                    "error": e.to_string(), "duration_ms": ms,
                                }));
                                errors.push(format!("'{}': {}", name, e));
                            }
                        }
                    }

                    if successes.is_empty() {
                        let final_state = (*base_state).clone();
                        final_state.set(
                            format!("__supervisor_meta__{}", config.name),
                            json!({
                                "strategy": "broadcast",
                                "children": child_results,
                                "errors": errors,
                                "success": false,
                                "duration_ms": start.elapsed().as_millis(),
                            }),
                        );
                        return Ok(final_state);
                    }

                    let (winner_name, winner_state) = match criteria {
                        "highest_score" => {
                            let best = successes.into_iter().max_by(|(_, a, _), (_, b, _)| {
                                let sa = a.get(score_key).and_then(|v| v.as_f64()).unwrap_or(0.0);
                                let sb = b.get(score_key).and_then(|v| v.as_f64()).unwrap_or(0.0);
                                sa.partial_cmp(&sb).unwrap_or(std::cmp::Ordering::Equal)
                            });
                            best.map(|(n, s, _)| (n, s))
                                .unwrap_or_else(|| ("unknown".to_string(), (*base_state).clone()))
                        }
                        _ => {
                            // first_success
                            successes
                                .into_iter()
                                .next()
                                .map(|(n, s, _)| (n, s))
                                .unwrap_or_else(|| ("unknown".to_string(), (*base_state).clone()))
                        }
                    };

                    winner_state.set(
                        format!("__supervisor_meta__{}", config.name),
                        json!({
                            "strategy": "broadcast",
                            "winner": winner_name,
                            "selection_criteria": criteria,
                            "children": child_results,
                            "success": true,
                            "duration_ms": start.elapsed().as_millis(),
                        }),
                    );
                    Ok(winner_state)
                }

                // ── MapReduce ──────────────────────────────────────────────────
                // Input split into chunks, each processed by a child in parallel, then merged.
                OrchestrationStrategy::MapReduce => {
                    let map_key = config.map_key.as_deref().unwrap_or("input_chunks");
                    let reduce_key = config.reduce_key.as_deref().unwrap_or("reduced_output");

                    let chunks: Vec<serde_json::Value> = state
                        .get(map_key)
                        .and_then(|v| v.as_array().cloned())
                        .unwrap_or_default();

                    if chunks.is_empty() {
                        tracing::warn!(
                            "Supervisor '{}' map_reduce: no data at key '{}'",
                            config.name,
                            map_key
                        );
                        state.set(
                            format!("__supervisor_meta__{}", config.name),
                            json!({"strategy": "map_reduce", "chunks": 0, "success": true}),
                        );
                        state.set(reduce_key.to_string(), json!([]));
                        return Ok(state);
                    }

                    let base_state = Arc::new(state);
                    let num_children = children.len();
                    let semaphore = config
                        .max_concurrent
                        .map(|n| Arc::new(tokio::sync::Semaphore::new(n)));

                    // Map phase: distribute chunks across children
                    let futures: Vec<_> = chunks
                        .iter()
                        .enumerate()
                        .map(|(i, chunk)| {
                            let child_idx = i % num_children;
                            let (name, handler) = children[child_idx].clone();
                            let chunk_state = base_state.deep_clone();
                            chunk_state.set("__map_chunk__".to_string(), chunk.clone());
                            chunk_state.set("__chunk_index__".to_string(), json!(i));
                            let timeout_ms = config.child_timeout_ms;
                            let sem = semaphore.clone();
                            async move {
                                let _permit = if let Some(ref sem) = sem {
                                    Some(sem.acquire().await.expect("semaphore closed"))
                                } else {
                                    None
                                };
                                let (result, ms) =
                                    run_child(&name, &handler, chunk_state, timeout_ms).await;
                                (i, name, result, ms)
                            }
                        })
                        .collect();

                    let results = futures::future::join_all(futures).await;

                    // Reduce phase: collect results in order
                    let mut reduced: Vec<serde_json::Value> = vec![json!(null); chunks.len()];
                    let mut child_results = Vec::new();
                    let mut errors: Vec<String> = Vec::new();

                    for (i, name, result, ms) in results {
                        match result {
                            Ok(s) => {
                                let chunk_result = s.get("__map_result__").unwrap_or(json!(null));
                                reduced[i] = chunk_result;
                                child_results.push(json!({
                                    "chunk_index": i, "agent": name, "success": true, "duration_ms": ms,
                                }));
                            }
                            Err(e) => {
                                errors.push(format!("Chunk {}: {}", i, e));
                                child_results.push(json!({
                                    "chunk_index": i, "agent": name, "success": false,
                                    "error": e.to_string(), "duration_ms": ms,
                                }));
                            }
                        }
                    }

                    let final_state = (*base_state).clone();
                    final_state.set(reduce_key.to_string(), json!(reduced));
                    final_state.set(
                        format!("__supervisor_meta__{}", config.name),
                        json!({
                            "strategy": "map_reduce",
                            "chunks": chunks.len(),
                            "children": child_results,
                            "errors": errors,
                            "success": errors.is_empty(),
                            "duration_ms": start.elapsed().as_millis(),
                        }),
                    );
                    Ok(final_state)
                }

                // ── ConditionalRouting ─────────────────────────────────────────
                // Routes to the most appropriate agent based on state-driven rules.
                OrchestrationStrategy::ConditionalRouting => {
                    let child_map: HashMap<String, ArcHandler<DynState>> = children
                        .iter()
                        .map(|(n, h)| (n.clone(), h.clone()))
                        .collect();

                    let mut selected_name: Option<String> = None;

                    for (condition, child_name) in &config.routing_rules {
                        if should_skip(condition, &state) {
                            // Condition is truthy → route to this agent
                            if child_map.contains_key(child_name.as_str()) {
                                selected_name = Some(child_name.clone());
                                break;
                            }
                        }
                    }

                    // Default to first child
                    let agent_name = selected_name.unwrap_or_else(|| {
                        let default = children.first().map(|(n, _)| n.clone()).unwrap_or_default();
                        tracing::info!(
                            "Supervisor '{}' conditional_routing: no rule matched, using default '{}'",
                            config.name, default
                        );
                        default
                    });

                    if let Some(handler) = child_map.get(&agent_name) {
                        let (result, ms) =
                            run_child(&agent_name, handler, state, config.child_timeout_ms).await;
                        match result {
                            Ok(final_state) => {
                                final_state.set(
                                    format!("__supervisor_meta__{}", config.name),
                                    json!({
                                        "strategy": "conditional_routing",
                                        "routed_to": agent_name,
                                        "duration_ms": ms,
                                        "success": true,
                                    }),
                                );
                                Ok(final_state)
                            }
                            Err(e) => Err(e),
                        }
                    } else {
                        Err(FlowgentraError::ConfigError(format!(
                            "Supervisor '{}': conditional routing target '{}' not found",
                            config.name, agent_name
                        )))
                    }
                }

                // ── RetryFallback ──────────────────────────────────────────────
                // Agents tried in order until one succeeds.
                OrchestrationStrategy::RetryFallback => {
                    let order: Vec<String> = if !config.fallback_order.is_empty() {
                        config.fallback_order.clone()
                    } else {
                        children.iter().map(|(n, _)| n.clone()).collect()
                    };

                    let child_map: HashMap<String, ArcHandler<DynState>> = children
                        .iter()
                        .map(|(n, h)| (n.clone(), h.clone()))
                        .collect();

                    let mut child_results = Vec::new();
                    let mut last_error = String::new();

                    for agent_name in &order {
                        if let Some(handler) = child_map.get(agent_name.as_str()) {
                            let max_attempts = config.max_retries_per_child + 1;
                            let mut total_ms = 0u128;

                            for attempt in 1..=max_attempts {
                                let (result, ms) = run_child(
                                    agent_name,
                                    handler,
                                    state.clone(),
                                    config.child_timeout_ms,
                                )
                                .await;
                                total_ms += ms;

                                match result {
                                    Ok(final_state) => {
                                        child_results.push(json!({
                                            "name": agent_name, "success": true,
                                            "duration_ms": total_ms, "attempts": attempt,
                                        }));
                                        final_state.set(
                                            format!("__supervisor_meta__{}", config.name),
                                            json!({
                                                "strategy": "retry_fallback",
                                                "succeeded_agent": agent_name,
                                                "children": child_results,
                                                "success": true,
                                                "duration_ms": start.elapsed().as_millis(),
                                            }),
                                        );
                                        tracing::info!(
                                            "Supervisor '{}' retry_fallback: '{}' succeeded",
                                            config.name,
                                            agent_name
                                        );
                                        return Ok(final_state);
                                    }
                                    Err(e) => {
                                        last_error = e.to_string();
                                        if attempt < max_attempts {
                                            tracing::warn!(
                                                "Supervisor '{}' retry_fallback: '{}' failed attempt {}/{}",
                                                config.name, agent_name, attempt, max_attempts
                                            );
                                        }
                                    }
                                }
                            }

                            child_results.push(json!({
                                "name": agent_name, "success": false,
                                "error": last_error, "duration_ms": total_ms,
                                "attempts": max_attempts,
                            }));
                            tracing::info!(
                                "Supervisor '{}' retry_fallback: '{}' exhausted, trying next fallback",
                                config.name, agent_name
                            );
                        }
                    }

                    // All agents failed
                    state.set(
                        format!("__supervisor_meta__{}", config.name),
                        json!({
                            "strategy": "retry_fallback",
                            "children": child_results,
                            "success": false,
                            "error": format!("All fallback agents failed. Last: {}", last_error),
                            "duration_ms": start.elapsed().as_millis(),
                        }),
                    );
                    Err(FlowgentraError::ExecutionError(format!(
                        "Supervisor '{}': all fallback agents failed. Last error: {}",
                        config.name, last_error
                    )))
                }

                // ── Debate / Critique ──────────────────────────────────────────
                // Agents generate responses and critique each other across rounds.
                OrchestrationStrategy::Debate => {
                    let mut current = state;
                    let rounds = if config.debate_rounds == 0 {
                        2
                    } else {
                        config.debate_rounds
                    };
                    let mut debate_log: Vec<serde_json::Value> = Vec::new();

                    let child_map: HashMap<String, ArcHandler<DynState>> = children
                        .iter()
                        .map(|(n, h)| (n.clone(), h.clone()))
                        .collect();
                    let child_names: Vec<String> =
                        children.iter().map(|(n, _)| n.clone()).collect();

                    for round in 0..rounds {
                        tracing::info!(
                            "Supervisor '{}' debate: round {}/{}",
                            config.name,
                            round + 1,
                            rounds
                        );
                        let mut round_responses: Vec<serde_json::Value> = Vec::new();

                        // Provide debate context to each agent
                        if !debate_log.is_empty() {
                            current.set("__debate_history__".to_string(), json!(debate_log));
                        }
                        current.set("__debate_round__".to_string(), json!(round));

                        for agent_name in &child_names {
                            if let Some(handler) = child_map.get(agent_name.as_str()) {
                                // Set previous round's responses so agent can critique
                                if !round_responses.is_empty() {
                                    current.set(
                                        "__debate_current_responses__".to_string(),
                                        json!(round_responses),
                                    );
                                }

                                let (result, ms) = run_child(
                                    agent_name,
                                    handler,
                                    current.clone(),
                                    config.child_timeout_ms,
                                )
                                .await;

                                match result {
                                    Ok(new_state) => {
                                        let response = new_state
                                            .get("__debate_response__")
                                            .unwrap_or(json!(null));
                                        round_responses.push(json!({
                                            "agent": agent_name,
                                            "response": response,
                                            "duration_ms": ms,
                                        }));
                                        current = new_state;
                                    }
                                    Err(e) => {
                                        round_responses.push(json!({
                                            "agent": agent_name,
                                            "error": e.to_string(),
                                            "duration_ms": ms,
                                        }));
                                    }
                                }
                            }
                        }

                        current.set(
                            "__debate_current_responses__".to_string(),
                            json!(round_responses.clone()),
                        );
                        debate_log.push(json!({
                            "round": round + 1,
                            "responses": round_responses,
                        }));
                    }

                    current.set("__debate_log__".to_string(), json!(debate_log));
                    current.set(
                        format!("__supervisor_meta__{}", config.name),
                        json!({
                            "strategy": "debate",
                            "rounds": rounds,
                            "participants": child_names,
                            "success": true,
                            "duration_ms": start.elapsed().as_millis(),
                        }),
                    );
                    Ok(current)
                }

                OrchestrationStrategy::Custom(strategy_name) => {
                    Err(FlowgentraError::ConfigError(format!(
                        "Orchestrator '{}': custom strategy '{}' is not implemented.",
                        config.name, strategy_name
                    )))
                }
            }
        })
    })
}

fn create_loop_standalone_handler(
    config: crate::core::node::advanced_nodes::LoopNodeConfig,
) -> Handler<DynState> {
    use serde_json::json;

    Box::new(move |state| {
        let config = config.clone();
        Box::pin(async move {
            // Use handler name as the key base (may be empty — fall back to "loop")
            let key_base = if config.handler.is_empty() {
                "loop".to_string()
            } else {
                config.handler.clone()
            };
            let iteration_key = format!("__loop_iteration__{}", key_base);
            let continue_key = format!("__loop_continue__{}", key_base);

            let iteration = state
                .get(&iteration_key)
                .and_then(|v| v.as_u64())
                .unwrap_or(0) as usize
                + 1;

            let break_now = config
                .break_condition
                .as_ref()
                .map(|cond| state.get(cond).and_then(|v| v.as_bool()).unwrap_or(false))
                .unwrap_or(false);

            let should_continue = iteration < config.max_iterations && !break_now;

            state.set(&iteration_key, json!(iteration));
            state.set(&continue_key, json!(should_continue));
            state.set(
                format!("__loop_meta__{}", key_base),
                json!({
                    "iteration": iteration,
                    "max_iterations": config.max_iterations,
                    "should_continue": should_continue,
                    "break_condition_met": break_now,
                }),
            );

            tracing::info!(
                "Standalone loop '{}' iteration={}/{}, continue={}",
                key_base,
                iteration,
                config.max_iterations,
                should_continue
            );

            Ok(state)
        })
    })
}

// ── Subgraph ──────────────────────────────────────────────────────────────────
/// Creates a handler that loads and runs a nested agent from a YAML config file.
///
/// The subgraph receives the parent's current state, executes its own full graph,
/// and returns the final state back to the parent graph. This enables true
/// hierarchical multi-agent systems where each subgraph is an independent agent.
///
/// Handlers are auto-discovered from the shared inventory (#[register_handler] pool).
///
/// YAML:
/// ```yaml
/// - name: research_agent
///   type: subgraph           # aliases: agent, agent_or_graph
///   config:
///     path: agents/research_agent.yaml
///
/// - name: coordinator
///   type: supervisor
///   config:
///     strategy: sequential
///     children: [research_agent, writer_agent]
/// ```
///
/// State: parent state flows in → subgraph executes fully → result flows back to parent
fn create_subgraph_handler(
    config: crate::core::node::agent_or_graph_node::SubgraphNodeConfig,
) -> Handler<DynState> {
    Box::new(move |state| {
        let path = config.path.clone();
        let name = config.name.clone();
        Box::pin(async move {
            tracing::info!("SubgraphNode '{}': loading agent from '{}'", name, path);

            // Compile the subgraph from its YAML using the shared handler inventory.
            // This is synchronous (YAML parse + hashmap build) — safe inside async.
            let mut sub_agent = from_config_path(&path).map_err(|e| {
                FlowgentraError::ConfigError(format!(
                    "SubgraphNode '{}': failed to load '{}': {}",
                    name, path, e
                ))
            })?;

            // Inject parent state into the subgraph
            sub_agent.state = state;

            // Execute the subgraph and return its final state to the parent graph
            let result = sub_agent.run().await?;
            tracing::info!("SubgraphNode '{}': completed", name);
            Ok(result)
        })
    })
}
