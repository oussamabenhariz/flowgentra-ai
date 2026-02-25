//! # Agent API - High-Level Interface
//!
//! The `Agent` struct provides a simple API for creating and running agents.
//! Most users interact with this module through the `from_config_path()` function
//! which uses automatic handler discovery.
//!
//! ## Quick Start
//!
//! 1. **Decorate your handlers** with `#[register_handler]`
//! 2. **Create a config.yaml** with your agent graph
//! 3. **Use `from_config_path()`** to create and run the agent
//!
//! ```ignore
//! use erenflow_ai::prelude::*;
//! use serde_json::json;
//!
//! #[register_handler]
//! pub async fn my_handler(mut state: State) -> Result<State> {
//!     let input = state.get("input").and_then(|v| v.as_str()).unwrap_or("");
//!     state.set("result", json!(input.to_uppercase()));
//!     Ok(state)
//! }
//!
//! #[tokio::main]
//! async fn main() -> Result<()> {
//!     // Auto-discovers all #[register_handler] functions
//!     let mut agent = from_config_path("config.yaml")?;
//!
//!     let mut state = State::new();
//!     state.set("input", json!("hello world"));
//!
//!     let result = agent.run(state).await?;
//!     println!("Done: {}", result.to_json_string()?);
//!     Ok(())
//! }
//! ```
//!
//! ## Handler Registration
//!
//! Handlers are automatically registered via the `#[register_handler]` attribute macro.
//! Handler names must match the function name and be referenced by that name in your config.yaml.

use crate::core::config::AgentConfig;
use crate::core::error::Result;
use crate::core::llm::{create_llm_client, LLMClient};
use crate::core::memory::{
    Checkpointer, ConversationMemory, InMemoryConversationMemory, MemoryCheckpointer,
};
use crate::core::runtime::AgentRuntime;
use crate::core::state::State;
use std::collections::HashMap;
use std::sync::Arc;
// Use inventory for auto-registration
inventory::collect!(HandlerEntry);

// =============================================================================
// Auto-Registration via Inventory
// =============================================================================

/// Entry for a handler in the global inventory
/// Handlers submit themselves to this list for auto-registration
pub struct HandlerEntry {
    /// Name of the handler (matches config node names)
    pub name: String,
    /// The handler function
    pub handler: ArcHandler,
}

impl HandlerEntry {
    /// Create a new handler entry for auto-registration
    pub fn new(name: impl Into<String>, handler: ArcHandler) -> Self {
        HandlerEntry {
            name: name.into(),
            handler,
        }
    }
}

// Type for Arc-wrapped handlers (used by inventory auto-registration)
pub type ArcHandler = Arc<
    dyn Fn(State) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<State>> + Send>>
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
/// use erenflow_ai::core::agent::Handler;
/// use erenflow_ai::core::state::State;
/// use serde_json::json;
///
/// let my_handler: Handler = Box::new(|mut state| {
///     Box::pin(async move {
///         let input = state.get("input");
///         state.set("output", json!("processed"));
///         Ok(state)
///     })
/// });
/// ```
pub type Handler = Box<
    dyn Fn(State) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<State>> + Send>>
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
/// use erenflow_ai::core::agent::Condition;
/// use erenflow_ai::core::state::State;
///
/// let is_complex: Condition = Box::new(|state: &State| {
///     state.get("complexity_score")
///         .and_then(|v| v.as_i64())
///         .map(|score| score > 50)
///         .unwrap_or(false)
/// });
/// ```
pub type Condition = Box<dyn Fn(&State) -> bool + Send + Sync>;

/// Registry mapping handler names to handler functions
pub type HandlerRegistry = HashMap<String, Handler>;

/// Registry mapping condition names to condition functions
pub type ConditionRegistry = HashMap<String, Condition>;

// =============================================================================
// Agent - Main API
// =============================================================================

/// The Agent - your main interface to ErenFlowAI
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
    /// Optional conversation memory (message history per thread). Set via config or with_conversation_memory().
    conversation_memory: Option<Arc<dyn ConversationMemory>>,
}

impl Agent {
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
    /// use erenflow_ai::prelude::*;
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
        handlers: HandlerRegistry,
        conditions: ConditionRegistry,
    ) -> Result<Self> {
        // Load and validate configuration
        let config = AgentConfig::from_file(config_path)?;
        config.validate()?;

        // Create runtime
        let mut runtime = AgentRuntime::from_config(config.clone())?;

        // Create LLM client
        let llm_client = create_llm_client(&config.llm)?;

        // Register all handlers
        for (name, handler) in handlers {
            runtime.register_node(&name, handler)?;
        }

        // Register all conditions
        type EdgeConditionFn = std::sync::Arc<
            dyn Fn(&State) -> std::result::Result<Option<String>, crate::core::error::ErenFlowError>
                + Send
                + Sync,
        >;
        for (condition_name, condition_fn) in conditions {
            // Convert bool condition to EdgeCondition format
            let edge_condition: EdgeConditionFn = std::sync::Arc::new(move |state: &State| {
                condition_fn(state);
                Ok(None)
            });

            // Find all edges with this condition_name and register
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

        Ok(Agent {
            runtime,
            llm_client,
            conversation_memory: None,
        })
    }

    /// Set the checkpointer (e.g. for thread-scoped state persistence). Can also be set via config.yaml `memory.checkpointer`.
    pub fn set_checkpointer(&mut self, checkpointer: Arc<dyn Checkpointer>) -> &mut Self {
        self.runtime.set_checkpointer(checkpointer);
        self
    }

    /// Builder-style: set the checkpointer.
    pub fn with_checkpointer(mut self, checkpointer: Arc<dyn Checkpointer>) -> Self {
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

    /// Execute the agent with initial state
    ///
    /// Runs the agent through all nodes following the edges until completion.
    ///
    /// # Example
    /// ```no_run
    /// use erenflow_ai::prelude::*;
    /// use serde_json::json;
    /// use std::collections::HashMap;
    ///
    /// #[tokio::main]
    /// async fn main() -> Result<()> {
    ///     let mut agent = Agent::from_config("config.yaml", HashMap::new(), HashMap::new())?;
    ///     
    ///     let mut state = State::new();
    ///     state.set("input", json!("Say hello"));
    ///     
    ///     let result = agent.run(state).await?;
    ///     println!("Done! Result: {}", result.to_json_string()?);
    ///     
    ///     Ok(())
    /// }
    /// ```
    pub async fn run(&mut self, initial_state: State) -> Result<State> {
        self.runtime.execute(initial_state).await
    }

    /// Run with a thread id for checkpointing and conversation memory. When a checkpointer is set,
    /// state is loaded from the last checkpoint for this thread (if any) and saved after each node.
    /// Use the same thread_id with conversation_memory to get/add messages for this conversation.
    pub async fn run_with_thread(
        &mut self,
        thread_id: &str,
        initial_state: State,
    ) -> Result<State> {
        self.runtime
            .execute_with_thread(thread_id, initial_state)
            .await
    }

    /// Get the LLM client for use in handlers
    ///
    /// Handlers can use this to access the configured LLM provider.
    pub fn llm_client(&self) -> Arc<dyn LLMClient> {
        Arc::clone(&self.llm_client)
    }

    /// Visualize the agent's execution graph
    ///
    /// Generates a text-based or graphical representation of your agent's workflow.
    /// Useful for debugging and documentation.
    ///
    /// # Arguments
    /// - `output_path`: Where to save the visualization
    ///
    /// # Example
    /// ```no_run
    /// # use erenflow_ai::prelude::*;
    /// # use std::collections::HashMap;
    /// # #[tokio::main]
    /// # async fn main() -> Result<()> {
    /// # let mut agent = Agent::from_config("config.yaml", HashMap::new(), HashMap::new())?;
    /// // Save the graph visualization
    /// #[cfg(feature = "visualization")]
    /// agent.visualize_graph("agent_graph.txt")?;
    /// # Ok(())
    /// # }
    /// ```
    #[cfg(feature = "visualization")]
    pub fn visualize_graph(&self, output_path: &str) -> Result<()> {
        self.runtime.visualize_graph(output_path)
    }

    /// Get mutable access to the underlying runtime
    ///
    /// For advanced users who need direct runtime access.
    pub fn runtime_mut(&mut self) -> &mut AgentRuntime {
        &mut self.runtime
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
/// use erenflow_ai::prelude::*;
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
///     let state = State::new();
///     let result = agent.run(state).await?;
///     Ok(())
/// }
/// ```
pub fn from_config_path(config_path: &str) -> Result<Agent> {
    // Load config to get required node names
    let config = AgentConfig::from_file(config_path)?;
    config.validate()?;

    // Get list of required handlers from config nodes
    let required_handlers: Vec<String> = config
        .graph
        .nodes
        .iter()
        .filter(|node| node.name != "START" && node.name != "END")
        .map(|node| node.handler.clone())
        .collect();

    // Collect all registered handlers from inventory
    let mut handlers_map: HashMap<String, ArcHandler> = HashMap::new();
    for entry in inventory::iter::<HandlerEntry> {
        handlers_map.insert(entry.name.clone(), entry.handler.clone());
    }

    // Inject built-in planner handler if any node uses it
    if required_handlers.iter().any(|h| h == "builtin::planner") {
        let llm_client = config.create_llm_client()?;
        let prompt_template = config.graph.planner.prompt_template.clone();
        let planner_fn = Arc::new(crate::core::node::planner::create_planner_handler(
            llm_client,
            prompt_template,
        ));
        let arc_handler: ArcHandler = Arc::new(move |state| planner_fn.as_ref()(state));
        handlers_map.insert("builtin::planner".to_string(), arc_handler);
    }

    // Check if all required handlers are registered
    let missing_handlers: Vec<&str> = required_handlers
        .iter()
        .filter(|handler| !handlers_map.contains_key(*handler))
        .map(|h| h.as_str())
        .collect();

    if !missing_handlers.is_empty() {
        let missing = missing_handlers.join(", ");
        let registered = handlers_map
            .keys()
            .map(|k| k.as_str())
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

        let error = crate::core::error::ErenFlowError::ConfigError(msg);
        return Err(error);
    }

    // Convert Arc handlers to Box handlers for Agent::from_config
    let handlers: HandlerRegistry = handlers_map
        .into_iter()
        .map(|(name, arc_handler)| {
            let handler: Handler = Box::new(move |state| arc_handler(state));
            (name, handler)
        })
        .collect();

    let mut agent = Agent::from_config(config_path, handlers, HashMap::new())?;

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

    Ok(agent)
}
