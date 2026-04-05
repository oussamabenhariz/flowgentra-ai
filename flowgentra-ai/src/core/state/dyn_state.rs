//! # DynState — channel-based, reducer-aware state implementation
//!
//! `DynState` is the channel-backed JSON `State` implementation.
//! Every field is backed by a `Channel` which carries both the current value and its
//! reducer strategy (`LastValue`, `Topic`, or `BinaryOperator`).
//!
//! ## Key features
//!
//! | Capability                        | API                                          |
//! |-----------------------------------|----------------------------------------------|
//! | Set a field                       | `state.set("k", v)` — applies reducer        |
//! | Read a field                      | `state.get("k")`                             |
//! | Per-field reducer strategies      | `ChannelType::LastValue / Topic / BinaryOp`  |
//! | Snapshot / restore                | `snapshot(step_id)` / `restore(snap)`        |
//!
//! ## Example
//!
//! ```ignore
//! use flowgentra_ai::core::state::{DynState, FieldSchema};
//! use serde_json::json;
//!
//! let state = DynState::with_schema(vec![
//!     FieldSchema::topic("messages"),
//!     FieldSchema::last_value("query"),
//! ]);
//!
//! state.set("messages", json!(["hello"]));
//! state.set("messages", json!(["world"]));   // appended, not replaced
//! assert_eq!(state.get("messages"), Some(json!(["hello", "world"])));
//! ```

use crate::core::error::{FlowgentraError, Result};
use crate::core::state::channel::{Channel, FieldSchema};
use crate::core::state::{snapshot::StateSnapshot, State};
use serde_json::Value;
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

// ── DynStateUpdate ────────────────────────────────────────────────────────────

/// Partial update for `DynState` — a set of key-value patches.
///
/// Only keys present in `fields` are applied; other fields remain unchanged.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct DynStateUpdate {
    /// Fields to update.
    pub fields: serde_json::Map<String, Value>,
}

impl DynStateUpdate {
    pub fn new() -> Self {
        Self {
            fields: serde_json::Map::new(),
        }
    }

    /// Set a field in the update (builder pattern).
    pub fn set(mut self, key: impl Into<String>, value: Value) -> Self {
        self.fields.insert(key.into(), value);
        self
    }

    /// Set a field in the update (mutable ref).
    pub fn insert(&mut self, key: impl Into<String>, value: Value) {
        self.fields.insert(key.into(), value);
    }
}

// ── DynState ──────────────────────────────────────────────────────────────────

/// Thread-safe, channel-backed state implementing the `State` trait.
///
/// Provides a key-value API with per-field reducer strategies and snapshot/restore support.
///
/// A "cheap clone" (`Clone`) shares the same `Arc` — mutations are visible to
/// all handles.  Use `deep_clone()` for an isolated copy.
#[derive(Clone)]
pub struct DynState {
    pub(crate) inner: Arc<RwLock<HashMap<String, Channel>>>,
}

// ── State trait impl ──────────────────────────────────────────────────────────

impl State for DynState {
    type Update = DynStateUpdate;

    /// Apply a partial update: each field in `update.fields` is merged using
    /// its channel's reducer strategy (LastValue / Topic / BinaryOperator).
    fn apply_update(&mut self, update: Self::Update) {
        match self.inner.write() {
            Ok(mut guard) => {
                for (k, v) in update.fields {
                    guard
                        .entry(k)
                        .and_modify(|c| c.apply(v.clone()))
                        .or_insert_with(|| Channel::last_value(v));
                }
            }
            Err(poisoned) => {
                tracing::warn!("DynState::apply_update: lock poisoned, recovering");
                let mut guard = poisoned.into_inner();
                for (k, v) in update.fields {
                    guard
                        .entry(k)
                        .and_modify(|c| c.apply(v.clone()))
                        .or_insert_with(|| Channel::last_value(v));
                }
            }
        }
    }
}

// ── Serialize / Deserialize ───────────────────────────────────────────────────

impl serde::Serialize for DynState {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeMap;
        let guard = self
            .inner
            .read()
            .map_err(|_| serde::ser::Error::custom("Failed to acquire read lock"))?;
        let mut map = serializer.serialize_map(Some(guard.len()))?;
        for (k, c) in guard.iter() {
            map.serialize_entry(k, &c.value)?;
        }
        map.end()
    }
}

impl<'de> serde::Deserialize<'de> for DynState {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let map: serde_json::Map<String, Value> = serde_json::Map::deserialize(deserializer)?;
        let channels: HashMap<String, Channel> = map
            .into_iter()
            .map(|(k, v)| (k, Channel::last_value(v)))
            .collect();
        Ok(DynState {
            inner: Arc::new(RwLock::new(channels)),
        })
    }
}

// ── Constructors ──────────────────────────────────────────────────────────────

impl DynState {
    /// Empty state with no channels pre-defined.
    pub fn new() -> Self {
        DynState {
            inner: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Alias for `new()`.
    pub fn empty() -> Self {
        Self::new()
    }

    /// Pre-populate channels from a schema.
    ///
    /// Each `FieldSchema` describes a field name, reducer, and default value.
    pub fn with_schema(fields: Vec<FieldSchema>) -> Self {
        let channels: HashMap<String, Channel> = fields
            .into_iter()
            .map(|f| (f.name.clone(), Channel::from_schema(&f)))
            .collect();
        DynState {
            inner: Arc::new(RwLock::new(channels)),
        }
    }

    /// Build from a plain key-value map (all fields become `LastValue` channels).
    pub fn from_map(map: HashMap<String, Value>) -> Self {
        let channels = map
            .into_iter()
            .map(|(k, v)| (k, Channel::last_value(v)))
            .collect();
        DynState {
            inner: Arc::new(RwLock::new(channels)),
        }
    }

    /// Create from a JSON `Value::Object`.
    pub fn from_json(value: Value) -> Result<Self> {
        match value {
            Value::Object(map) => {
                let channels: HashMap<String, Channel> = map
                    .into_iter()
                    .map(|(k, v)| (k, Channel::last_value(v)))
                    .collect();
                Ok(DynState {
                    inner: Arc::new(RwLock::new(channels)),
                })
            }
            _ => Err(FlowgentraError::StateError(
                "State must be a JSON object".to_string(),
            )),
        }
    }
}

// ── Core access ───────────────────────────────────────────────────────────────

impl DynState {
    /// Set a field value using its channel's reducer.
    ///
    /// For `LastValue` channels this is a plain overwrite.
    /// For `Topic` channels the value is appended.
    /// Unknown fields are auto-created as `LastValue`.
    pub fn set(&self, key: impl Into<String>, value: Value) {
        match self.inner.write() {
            Ok(mut guard) => {
                let k = key.into();
                guard
                    .entry(k)
                    .and_modify(|c| c.apply(value.clone()))
                    .or_insert_with(|| Channel::last_value(value));
            }
            Err(poisoned) => {
                tracing::warn!("DynState::set: lock poisoned, recovering");
                let k = key.into();
                poisoned
                    .into_inner()
                    .entry(k)
                    .and_modify(|c| c.apply(value.clone()))
                    .or_insert_with(|| Channel::last_value(value));
            }
        }
    }

    /// Set a field value **without** applying the reducer (raw overwrite).
    ///
    /// Used for initial population and snapshot restoration.
    pub fn set_raw(&self, key: impl Into<String>, value: Value) {
        match self.inner.write() {
            Ok(mut guard) => {
                let k = key.into();
                match guard.get_mut(&k) {
                    Some(c) => c.value = value,
                    None => {
                        guard.insert(k, Channel::last_value(value));
                    }
                }
            }
            Err(poisoned) => {
                tracing::warn!("DynState::set_raw: lock poisoned, recovering");
                let k = key.into();
                let mut g = poisoned.into_inner();
                match g.get_mut(&k) {
                    Some(c) => c.value = value,
                    None => {
                        g.insert(k, Channel::last_value(value));
                    }
                }
            }
        }
    }

    /// Register a channel reducer for a field (after construction).
    ///
    /// If the field already has a value, it is preserved.
    pub fn register_channel(&self, schema: &FieldSchema) {
        if let Ok(mut guard) = self.inner.write() {
            let entry = guard
                .entry(schema.name.clone())
                .or_insert_with(|| Channel::from_schema(schema));
            // Update the reducer without disturbing the existing value.
            entry.channel_type = schema.channel_type.clone();
        }
    }

    /// Get a field value; returns `None` if absent.
    pub fn get(&self, key: &str) -> Option<Value> {
        match self.inner.read() {
            Ok(guard) => guard.get(key).map(|c| c.value.clone()),
            Err(poisoned) => {
                tracing::warn!("DynState::get: lock poisoned, recovering");
                poisoned.into_inner().get(key).map(|c| c.value.clone())
            }
        }
    }

    /// Get a field and deserialize it to type `T`.
    pub fn get_typed<T: serde::de::DeserializeOwned>(&self, key: &str) -> Result<T> {
        self.get(key)
            .ok_or_else(|| FlowgentraError::StateError(format!("Key '{}' not found in state", key)))
            .and_then(|v| serde_json::from_value(v).map_err(FlowgentraError::from))
    }

    pub fn contains_key(&self, key: &str) -> bool {
        self.inner
            .read()
            .ok()
            .map(|g| g.contains_key(key))
            .unwrap_or(false)
    }

    pub fn remove(&self, key: &str) -> Option<Value> {
        match self.inner.write() {
            Ok(mut guard) => guard.remove(key).map(|c| c.value),
            Err(poisoned) => {
                tracing::warn!("DynState::remove: lock poisoned, recovering");
                poisoned.into_inner().remove(key).map(|c| c.value)
            }
        }
    }

    pub fn keys(&self) -> Vec<String> {
        self.inner
            .read()
            .ok()
            .map(|g| g.keys().cloned().collect())
            .unwrap_or_default()
    }

    pub fn len(&self) -> usize {
        self.inner.read().ok().map(|g| g.len()).unwrap_or(0)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

// ── Typed helpers ─────────────────────────────────────────────────────────────

impl DynState {
    pub fn get_string(&self, key: &str) -> Option<String> {
        self.get(key)
            .and_then(|v| v.as_str().map(|s| s.to_string()))
    }

    /// Alias for `get_string`.
    pub fn get_str(&self, key: &str) -> Option<String> {
        self.get_string(key)
    }

    pub fn get_int(&self, key: &str) -> Option<i64> {
        self.get(key).and_then(|v| v.as_i64())
    }

    pub fn get_float(&self, key: &str) -> Option<f64> {
        self.get(key).and_then(|v| v.as_f64())
    }

    pub fn get_bool(&self, key: &str) -> Option<bool> {
        self.get(key).and_then(|v| v.as_bool())
    }
}

// ── Serialization ─────────────────────────────────────────────────────────────

impl DynState {
    /// Flatten to a `HashMap<String, Value>`.
    pub fn to_map(&self) -> HashMap<String, Value> {
        self.inner
            .read()
            .ok()
            .map(|g| {
                g.iter()
                    .map(|(k, c)| (k.clone(), c.value.clone()))
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Serialize to a `Value::Object`.
    pub fn to_value(&self) -> Value {
        let map: serde_json::Map<String, Value> = self.to_map().into_iter().collect();
        Value::Object(map)
    }

    /// Serialize to a JSON string.
    pub fn to_json_string(&self) -> Result<String> {
        serde_json::to_string(&self.to_value()).map_err(FlowgentraError::from)
    }

    /// Pairs of (field_name, value) — equivalent to `to_map().into_iter()`.
    pub fn as_map(&self) -> Vec<(String, Value)> {
        self.inner
            .read()
            .ok()
            .map(|g| {
                g.iter()
                    .map(|(k, c)| (k.clone(), c.value.clone()))
                    .collect()
            })
            .unwrap_or_default()
    }
}

// ── Merge ─────────────────────────────────────────────────────────────────────

impl DynState {
    /// Merge all fields from `other` into this state using each field's reducer.
    pub fn merge(&self, other: &DynState) {
        let other_map = other.to_map();
        for (k, v) in other_map {
            self.set(k, v);
        }
    }

    /// Apply a partial-update map directly (merges each key through its reducer).
    pub fn apply_map(&self, patch: HashMap<String, Value>) {
        for (k, v) in patch {
            self.set(k, v);
        }
    }
}

// ── Clone semantics ───────────────────────────────────────────────────────────

impl DynState {
    /// Deep clone — new `Arc` wrapping an independent copy of all channels.
    pub fn deep_clone(&self) -> Self {
        let cloned = self.inner.read().unwrap_or_else(|p| p.into_inner()).clone();
        DynState {
            inner: Arc::new(RwLock::new(cloned)),
        }
    }
}

// ── Snapshot / restore ────────────────────────────────────────────────────────

impl DynState {
    /// Capture current field values as a `StateSnapshot`.
    pub fn snapshot(&self, step_id: impl Into<String>) -> StateSnapshot {
        StateSnapshot::new(step_id, self.to_map())
    }

    /// Restore field values from a snapshot (raw overwrite, reducers are bypassed).
    ///
    /// Fields NOT present in the snapshot are left unchanged.
    pub fn restore(&self, snapshot: &StateSnapshot) {
        if let Ok(mut guard) = self.inner.write() {
            for (k, v) in &snapshot.state {
                match guard.get_mut(k) {
                    Some(c) => c.value = v.clone(),
                    None => {
                        guard.insert(k.clone(), Channel::last_value(v.clone()));
                    }
                }
            }
        }
    }
}

// ── Closure-based access ──────────────────────────────────────────────────────

impl DynState {
    /// Execute a closure with mutable access to the raw value map.
    pub fn with_mut<F>(&self, f: F) -> Result<()>
    where
        F: FnOnce(&mut HashMap<String, Value>),
    {
        let mut guard = self
            .inner
            .write()
            .map_err(|_| FlowgentraError::StateError("Failed to acquire write lock".to_string()))?;
        // Expose a temporary HashMap of only the values.
        let mut value_map: HashMap<String, Value> = guard
            .iter()
            .map(|(k, c)| (k.clone(), c.value.clone()))
            .collect();
        f(&mut value_map);
        // Write back (preserving channel type for existing keys).
        for (k, v) in value_map {
            match guard.get_mut(&k) {
                Some(c) => c.value = v,
                None => {
                    guard.insert(k, Channel::last_value(v));
                }
            }
        }
        Ok(())
    }

    /// Execute a closure with read access to the raw value map.
    pub fn with<F, R>(&self, f: F) -> Result<R>
    where
        F: FnOnce(&HashMap<String, Value>) -> R,
    {
        let guard = self
            .inner
            .read()
            .map_err(|_| FlowgentraError::StateError("Failed to acquire read lock".to_string()))?;
        let value_map: HashMap<String, Value> = guard
            .iter()
            .map(|(k, c)| (k.clone(), c.value.clone()))
            .collect();
        Ok(f(&value_map))
    }
}

// ── LLM / MCP / RAG helpers (same as DynState, for drop-in replacement) ──

impl DynState {
    /// Store an evaluation result for `node`.
    pub fn set_evaluation(&self, node: &str, eval: crate::core::evaluation::EvaluationResult) {
        let key = format!("_evaluation_{}", node);
        if let Ok(json_val) = serde_json::to_value(&eval) {
            self.set(key, json_val);
        }
    }

    /// Get the configured LLM client stored in the `_llm_config` field.
    pub fn get_llm_client(&self) -> Result<Arc<dyn crate::core::llm::LLMClient>> {
        let config: crate::core::llm::LLMConfig = self.get_typed("_llm_config").map_err(|_| {
            FlowgentraError::ConfigError(
                "LLM config not found in state. Make sure LLM is configured in config.yaml"
                    .to_string(),
            )
        })?;
        config.create_client()
    }

    /// Get an MCP client by name (config stored in `_mcp_configs`).
    pub fn get_mcp_client(&self, name: &str) -> Result<Arc<dyn crate::core::mcp::MCPClient>> {
        let configs: HashMap<String, crate::core::mcp::MCPConfig> =
            self.get_typed("_mcp_configs").map_err(|_| {
                FlowgentraError::ConfigError("MCP configs not found in state".to_string())
            })?;
        let config = configs
            .get(name)
            .ok_or_else(|| FlowgentraError::ConfigError(format!("MCP '{}' not found", name)))?;
        crate::core::mcp::MCPClientFactory::create(config.clone())
    }

    /// Get the MCP names assigned to the current node (`_node_mcps`).
    pub fn get_node_mcps(&self) -> Vec<String> {
        self.get("_node_mcps")
            .and_then(|v| serde_json::from_value::<Vec<String>>(v).ok())
            .unwrap_or_default()
    }

    /// Get the first MCP client assigned to the current node.
    pub fn get_node_mcp_client(&self) -> Result<Arc<dyn crate::core::mcp::MCPClient>> {
        let name = self.get_node_mcps().into_iter().next().ok_or_else(|| {
            FlowgentraError::ConfigError("No MCPs assigned to this node".to_string())
        })?;
        self.get_mcp_client(&name)
    }

    /// Run LLM chat with MCP tools in an agentic loop.
    pub async fn chat_with_mcp_tools(
        &self,
        mcp_name: &str,
        messages: Vec<crate::core::llm::Message>,
        max_iterations: usize,
    ) -> Result<crate::core::llm::Message> {
        let llm = self.get_llm_client()?;
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

        Err(FlowgentraError::ExecutionError(format!(
            "chat_with_mcp_tools: max iterations ({}) reached",
            max_iterations
        )))
    }

    pub async fn chat_with_node_mcp_tools(
        &self,
        messages: Vec<crate::core::llm::Message>,
        max_iterations: usize,
    ) -> Result<crate::core::llm::Message> {
        let name = self.get_node_mcps().into_iter().next().ok_or_else(|| {
            FlowgentraError::ConfigError("No MCPs assigned to this node".to_string())
        })?;
        self.chat_with_mcp_tools(&name, messages, max_iterations)
            .await
    }

    /// Get RAG config stored in `_rag_config`.
    pub fn get_rag_config(&self) -> Result<crate::core::config::RAGGraphConfig> {
        self.get_typed("_rag_config")
            .map_err(|_| FlowgentraError::ConfigError("RAG config not found in state".to_string()))
    }

    /// Get embeddings provider from RAG config.
    pub fn get_rag_embeddings(&self) -> Result<Arc<crate::core::rag::Embeddings>> {
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

    /// Get ChromaDB vector store from RAG config.
    pub async fn get_rag_store(&self) -> Result<crate::core::rag::ChromaStore> {
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
                FlowgentraError::ToolError(format!("Failed to connect to ChromaDB: {}", e))
            })
    }
}

// ── std trait impls ───────────────────────────────────────────────────────────

impl Default for DynState {
    fn default() -> Self {
        DynState::new()
    }
}

impl std::fmt::Debug for DynState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DynState")
            .field("arc_refs", &Arc::strong_count(&self.inner))
            .finish()
    }
}

impl From<HashMap<String, Value>> for DynState {
    fn from(map: HashMap<String, Value>) -> Self {
        DynState::from_map(map)
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::state::channel::FieldSchema;
    use serde_json::json;

    #[test]
    fn last_value_overwrites() {
        let state = DynState::new();
        state.set("x", json!(1));
        state.set("x", json!(2));
        assert_eq!(state.get("x"), Some(json!(2)));
    }

    #[test]
    fn topic_appends() {
        let state = DynState::with_schema(vec![FieldSchema::topic("msgs")]);
        state.set("msgs", json!(["a"]));
        state.set("msgs", json!(["b"]));
        assert_eq!(state.get("msgs"), Some(json!(["a", "b"])));
    }

    #[test]
    fn apply_update_uses_reducers() {
        let mut state = DynState::with_schema(vec![FieldSchema::topic("log")]);
        state.set_raw("log", json!(["init"]));

        let update = DynStateUpdate::new().set("log", json!(["step-1"]));
        state.apply_update(update);

        assert_eq!(state.get("log"), Some(json!(["init", "step-1"])));
    }

    #[test]
    fn snapshot_and_restore() {
        let state = DynState::new();
        state.set("counter", json!(0));

        let snap = state.snapshot("before");
        state.set("counter", json!(99));
        assert_eq!(state.get("counter"), Some(json!(99)));

        state.restore(&snap);
        assert_eq!(state.get("counter"), Some(json!(0)));
    }

    #[test]
    fn deep_clone_is_independent() {
        let a = DynState::new();
        a.set("v", json!(1));
        let b = a.deep_clone();
        b.set("v", json!(2));
        assert_eq!(a.get("v"), Some(json!(1)));
        assert_eq!(b.get("v"), Some(json!(2)));
    }

    #[test]
    fn serialize_roundtrip() {
        let state = DynState::new();
        state.set("name", json!("alice"));
        let json_str = serde_json::to_string(&state).unwrap();
        let restored: DynState = serde_json::from_str(&json_str).unwrap();
        assert_eq!(restored.get("name"), Some(json!("alice")));
    }
}
