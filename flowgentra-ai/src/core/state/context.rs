//! # Node Execution Context
//!
//! `Context` carries framework-managed resources (LLM, MCP, RAG) that nodes need
//! but which are NOT part of the user's state schema.
//!
//! This replaces the old pattern of injecting `_llm_config`, `_mcp_configs`, etc.
//! into state as hidden keys. With typed state, dynamic injection is impossible,
//! so these resources live in a separate `Context` passed alongside state.
//!
//! # Example
//!
//! ```ignore
//! async fn my_node(state: &MyState, ctx: &Context) -> Result<MyStateUpdate> {
//!     let llm = ctx.get_llm()?;
//!     let response = llm.chat(vec![Message::user(&state.query)]).await?;
//!     Ok(MyStateUpdate::new().result(Some(response.content)))
//! }
//! ```

use std::collections::HashMap;
use std::sync::Arc;

/// Framework-managed resources available to every node during execution.
///
/// Passed as the second argument to `Node::execute(state, ctx)`.
/// Created by the graph executor from the agent configuration.
#[derive(Clone, Debug)]
pub struct Context {
    /// LLM configuration (if configured)
    llm_config: Option<crate::core::llm::LLMConfig>,

    /// MCP server configurations keyed by name
    mcp_configs: HashMap<String, crate::core::mcp::MCPConfig>,

    /// RAG configuration (if configured)
    rag_config: Option<crate::core::config::RAGGraphConfig>,

    /// MCP names assigned to the current node via `mcps: [...]` in config
    node_mcps: Vec<String>,

    /// Current node name
    node_name: String,

    /// Arbitrary metadata
    metadata: HashMap<String, serde_json::Value>,

    /// Event broadcaster for streaming execution events to subscribers.
    ///
    /// Injected by the graph executor on every node call. Use
    /// `ctx.event_broadcaster()` to emit LLM chunks, tool calls, etc.
    event_broadcaster: Option<Arc<crate::core::observability::events::EventBroadcaster>>,

    /// Runtime tool registry for dynamic tool binding.
    ///
    /// Nodes can read or update this registry to add/remove tools mid-execution.
    tool_registry: Option<Arc<tokio::sync::RwLock<crate::core::tools::ToolRegistry>>>,
}

impl Default for Context {
    fn default() -> Self {
        Self::new()
    }
}

impl Context {
    /// Create an empty context (no LLM, no MCP, no RAG).
    pub fn new() -> Self {
        Self {
            llm_config: None,
            mcp_configs: HashMap::new(),
            rag_config: None,
            node_mcps: Vec::new(),
            node_name: String::new(),
            metadata: HashMap::new(),
            event_broadcaster: None,
            tool_registry: None,
        }
    }

    // ── Builder methods ──

    /// Set the LLM configuration.
    pub fn with_llm_config(mut self, config: crate::core::llm::LLMConfig) -> Self {
        self.llm_config = Some(config);
        self
    }

    /// Set MCP configurations.
    pub fn with_mcp_configs(
        mut self,
        configs: HashMap<String, crate::core::mcp::MCPConfig>,
    ) -> Self {
        self.mcp_configs = configs;
        self
    }

    /// Set RAG configuration.
    pub fn with_rag_config(mut self, config: crate::core::config::RAGGraphConfig) -> Self {
        self.rag_config = Some(config);
        self
    }

    /// Set the MCPs assigned to the current node.
    pub fn with_node_mcps(mut self, mcps: Vec<String>) -> Self {
        self.node_mcps = mcps;
        self
    }

    /// Set the current node name.
    pub fn with_node_name(mut self, name: impl Into<String>) -> Self {
        self.node_name = name.into();
        self
    }

    /// Set arbitrary metadata.
    pub fn with_metadata(mut self, metadata: HashMap<String, serde_json::Value>) -> Self {
        self.metadata = metadata;
        self
    }

    /// Attach an event broadcaster (builder form).
    pub fn with_event_broadcaster(
        mut self,
        broadcaster: Arc<crate::core::observability::events::EventBroadcaster>,
    ) -> Self {
        self.event_broadcaster = Some(broadcaster);
        self
    }

    /// Attach a tool registry for dynamic tool binding (builder form).
    pub fn with_tool_registry(
        mut self,
        registry: Arc<tokio::sync::RwLock<crate::core::tools::ToolRegistry>>,
    ) -> Self {
        self.tool_registry = Some(registry);
        self
    }

    // ── Setters (for mutation after creation) ──

    /// Update the node MCPs for a new node execution.
    pub fn set_node_mcps(&mut self, mcps: Vec<String>) {
        self.node_mcps = mcps;
    }

    /// Update the current node name.
    pub fn set_node_name(&mut self, name: impl Into<String>) {
        self.node_name = name.into();
    }

    /// Set a metadata value.
    pub fn set_metadata(&mut self, key: impl Into<String>, value: serde_json::Value) {
        self.metadata.insert(key.into(), value);
    }

    /// Inject the event broadcaster (called by the graph executor before each node).
    pub fn set_event_broadcaster(
        &mut self,
        broadcaster: Arc<crate::core::observability::events::EventBroadcaster>,
    ) {
        self.event_broadcaster = Some(broadcaster);
    }

    /// Inject a tool registry for dynamic tool binding.
    pub fn set_tool_registry(
        &mut self,
        registry: Arc<tokio::sync::RwLock<crate::core::tools::ToolRegistry>>,
    ) {
        self.tool_registry = Some(registry);
    }

    // ── Accessors ──

    /// Get the current node name.
    pub fn node_name(&self) -> &str {
        &self.node_name
    }

    /// Get metadata value.
    pub fn get_metadata(&self, key: &str) -> Option<&serde_json::Value> {
        self.metadata.get(key)
    }

    /// Get the MCP names assigned to the current node.
    pub fn node_mcps(&self) -> &[String] {
        &self.node_mcps
    }

    /// Get the event broadcaster, if one was injected by the executor.
    ///
    /// Use this to emit streaming LLM chunks or tool call events from inside a node:
    ///
    /// ```ignore
    /// if let Some(b) = ctx.event_broadcaster() {
    ///     b.emit_llm_chunk(ctx.node_name(), token, chunk_index);
    /// }
    /// ```
    pub fn event_broadcaster(
        &self,
    ) -> Option<&Arc<crate::core::observability::events::EventBroadcaster>> {
        self.event_broadcaster.as_ref()
    }

    /// Get the tool registry for dynamic tool binding at runtime.
    ///
    /// Returns `None` if no registry was attached to the graph builder.
    ///
    /// ```ignore
    /// if let Some(registry) = ctx.tool_registry() {
    ///     let guard = registry.read().await;
    ///     let tools = guard.list_definitions();
    /// }
    /// ```
    pub fn tool_registry(
        &self,
    ) -> Option<&Arc<tokio::sync::RwLock<crate::core::tools::ToolRegistry>>> {
        self.tool_registry.as_ref()
    }

    // ── LLM ──

    /// Get the configured LLM.
    pub fn get_llm(&self) -> crate::core::error::Result<Arc<dyn crate::core::llm::LLM>> {
        let config = self.llm_config.as_ref().ok_or_else(|| {
            crate::core::error::FlowgentraError::ConfigError(
                "LLM config not found in context. Make sure LLM is configured in config.yaml"
                    .to_string(),
            )
        })?;
        config.create_client()
    }

    /// Get the raw LLM config (if set).
    pub fn llm_config(&self) -> Option<&crate::core::llm::LLMConfig> {
        self.llm_config.as_ref()
    }

    // ── MCP ──

    /// Get an MCP client by name.
    pub fn get_mcp_client(
        &self,
        name: &str,
    ) -> crate::core::error::Result<Arc<dyn crate::core::mcp::MCPClient>> {
        let config = self.mcp_configs.get(name).ok_or_else(|| {
            crate::core::error::FlowgentraError::ConfigError(format!(
                "MCP '{}' not found in context. Available: {:?}",
                name,
                self.mcp_configs.keys().collect::<Vec<_>>()
            ))
        })?;
        crate::core::mcp::MCPClientFactory::create(config.clone())
    }

    /// Get the MCP client for this node's assigned MCP (first one).
    pub fn get_node_mcp_client(
        &self,
    ) -> crate::core::error::Result<Arc<dyn crate::core::mcp::MCPClient>> {
        let name = self.node_mcps.first().ok_or_else(|| {
            crate::core::error::FlowgentraError::ConfigError(
                "No MCPs assigned to this node. Add mcps: [mcp_name] to the node config."
                    .to_string(),
            )
        })?;
        self.get_mcp_client(name)
    }

    // ── RAG ──

    /// Get the RAG configuration.
    pub fn get_rag_config(
        &self,
    ) -> crate::core::error::Result<&crate::core::config::RAGGraphConfig> {
        self.rag_config.as_ref().ok_or_else(|| {
            crate::core::error::FlowgentraError::ConfigError(
                "RAG config not found in context. Make sure `graph.rag` is defined in config.yaml"
                    .to_string(),
            )
        })
    }

    /// Get an embeddings provider from the RAG config.
    pub fn get_rag_embeddings(
        &self,
    ) -> crate::core::error::Result<Arc<crate::core::rag::Embeddings>> {
        let config = self.get_rag_config()?;

        let embeddings: Arc<crate::core::rag::Embeddings> =
            match config.embeddings.provider.as_str() {
                "mistral" => {
                    let api_key = config.embeddings.api_key.as_deref().unwrap_or_default();
                    let model = config.embeddings.model.clone();
                    Arc::new(crate::core::rag::Embeddings::new(Arc::new(
                        crate::core::rag::MistralEmbeddings::new(api_key, model),
                    )))
                }
                "ollama" => {
                    let model = config
                        .embeddings
                        .model
                        .as_deref()
                        .unwrap_or("nomic-embed-text");
                    let base_url = config
                        .embeddings
                        .api_key
                        .clone()
                        .or_else(|| Some("http://localhost:11434".to_string()));
                    Arc::new(crate::core::rag::Embeddings::new(Arc::new(
                        crate::core::rag::OllamaEmbeddings::new(model, base_url),
                    )))
                }
                "openai" => {
                    let api_key = config.embeddings.api_key.as_deref().unwrap_or_default();
                    let model = config
                        .embeddings
                        .model
                        .as_deref()
                        .unwrap_or("text-embedding-3-small");
                    Arc::new(crate::core::rag::Embeddings::new(Arc::new(
                        crate::core::rag::OpenAIEmbeddings::new(api_key, model),
                    )))
                }
                _ => Arc::new(crate::core::rag::Embeddings::mock(
                    config.embedding_dimension(),
                )),
            };

        Ok(embeddings)
    }

    /// Get a ChromaDB vector store from the RAG config.
    pub async fn get_rag_store(&self) -> crate::core::error::Result<crate::core::rag::ChromaStore> {
        let config = self.get_rag_config()?;

        let endpoint = config
            .vector_store
            .endpoint
            .as_deref()
            .unwrap_or("http://localhost:8000");

        let rag_db_config = crate::core::rag::RAGConfig {
            store_type: crate::core::rag::VectorStoreType::Chroma,
            api_key: config.vector_store.api_key.clone(),
            endpoint: Some(endpoint.to_string()),
            index_name: config.vector_store.collection.clone(),
            embedding_dim: config.embedding_dimension(),
        };

        crate::core::rag::ChromaStore::new(&rag_db_config)
            .await
            .map_err(|e| {
                crate::core::error::FlowgentraError::ToolError(format!(
                    "Failed to connect to ChromaDB: {}",
                    e
                ))
            })
    }

    // ── Chat with MCP Tools ──

    /// Run an LLM chat with MCP tools in a loop until the LLM produces a final text response.
    ///
    /// 1. Lists available tools from the MCP server
    /// 2. Sends the conversation + tool definitions to the LLM
    /// 3. If the LLM requests tool calls → executes them via MCP → feeds results back
    /// 4. Repeats until the LLM returns a text response (or max iterations reached)
    pub async fn chat_with_mcp_tools(
        &self,
        mcp_name: &str,
        messages: Vec<crate::core::llm::Message>,
        max_iterations: usize,
    ) -> crate::core::error::Result<crate::core::llm::Message> {
        let llm = self.get_llm()?;
        let mcp = self.get_mcp_client(mcp_name)?;

        let mcp_tools = mcp.list_tools().await?;
        let tool_defs: Vec<crate::core::llm::ToolDefinition> = mcp_tools
            .iter()
            .map(|t| {
                crate::core::llm::ToolDefinition::new(
                    &t.name,
                    t.description.as_deref().unwrap_or(""),
                    t.input_schema.clone(),
                )
            })
            .collect();

        let mut conversation = messages;

        for iteration in 0..max_iterations {
            let response = llm
                .chat_with_tools(conversation.clone(), &tool_defs)
                .await?;

            if !response.has_tool_calls() {
                return Ok(response);
            }

            let tool_calls = response.tool_calls.clone().unwrap_or_default();
            conversation.push(response);

            for tc in &tool_calls {
                tracing::debug!(tool = %tc.name, iteration, "LLM tool call");

                let result = match mcp.call_tool(&tc.name, tc.arguments.clone()).await {
                    Ok(val) => serde_json::to_string(&val).unwrap_or_else(|_| "{}".to_string()),
                    Err(e) => format!("{{\"error\": \"{}\"}}", e),
                };

                conversation.push(crate::core::llm::Message::tool_result(&tc.id, result));
            }
        }

        Err(crate::core::error::FlowgentraError::ExecutionError(
            format!(
                "chat_with_mcp_tools: max iterations ({}) reached without final response",
                max_iterations
            ),
        ))
    }

    /// Run an LLM chat using MCP tools assigned to this node.
    pub async fn chat_with_node_mcp_tools(
        &self,
        messages: Vec<crate::core::llm::Message>,
        max_iterations: usize,
    ) -> crate::core::error::Result<crate::core::llm::Message> {
        let name = self.node_mcps.first().ok_or_else(|| {
            crate::core::error::FlowgentraError::ConfigError(
                "No MCPs assigned to this node. Add mcps: [mcp_name] to the node config."
                    .to_string(),
            )
        })?;
        self.chat_with_mcp_tools(name, messages, max_iterations)
            .await
    }
}
