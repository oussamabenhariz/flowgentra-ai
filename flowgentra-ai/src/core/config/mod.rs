use crate::core::state::ValidationError;
use std::sync::Arc;
// # Configuration Best Practices
//
// - Validate config.yaml at startup; provide clear error messages.
// - Encourage documenting the state schema in config.
// - Consider auto-generating Rust state structs from schema (future).
// # Configuration Management
//
// The configuration system allows you to define agents declaratively using YAML.
//
// ## Config File Example
//
// ```yaml
// name: my_agent
// description: "An intelligent agent for task X"
//
// llm:
//   provider: openai
//   model: gpt-4
//   temperature: 0.7
//   api_key: ${OPENAI_API_KEY}
//
// graph:
//   nodes:
//     - name: process_input
//       handler: my_handlers.process_input
//       mcps: [web_search]
//
//     - name: generate_response
//       handler: my_handlers.generate_response
//
//   edges:
//     - from: START
//       to: process_input
//     
//     - from: process_input
//       to: generate_response
//     
//     - from: generate_response
//       to: END
//
// state_schema:
//   input:
//     type: string
//     description: "User input"
//   processed_data:
//     type: string
//     description: "Processed data"
//   response:
//     type: string
//     description: "Agent response"
//
// ## Memory (optional)
//
// Enable checkpointer and conversation memory under `memory:`:
//
// ```yaml
// memory:
//   checkpointer:
//     type: memory    # or "none" to disable (default)
//   conversation:
//     enabled: true
//     state_key: messages   # optional, key to sync messages in state
//   buffer:
//     max_messages: 20     # optional, keep only last N messages (buffer/window)
// ```
//
// Then use `agent.run_with_thread(thread_id, state)` to run with persistence.
//
// ## State Schema Formats
//
// The `state_schema` supports two formats:
//
// ### Structured Format (Recommended)
// ```yaml
// state_schema:
//   field_name:
//     type: string
//     description: "Field description"
// ```
//
// ### Legacy Format
// ```yaml
// state_schema:
//   field_name: "string - Field description"
// ```
//
// Both formats are fully supported and compatible.
//
// ## Features
//
// - **Environment Variables** - Substitute `${VAR_NAME}` in config
// - **Validation** - Checks graph structure for consistency
// - **Flexibility** - Easy to update without recompiling
// - **Dual Format Support** - Works with both structured and legacy state schemas

use crate::core::llm::{LLMConfig, LLMProvider};
use crate::core::mcp::MCPConfig;
use crate::core::memory::MemoryConfig;
use crate::core::node::{EdgeConfig, NodeConfig};
use crate::core::rag::vector_db::VectorStoreType;
use crate::core::state::state_validation::StateSchema;
use once_cell::sync::Lazy;
use regex::Regex;
use serde::{Deserialize, Deserializer, Serialize};
use std::collections::HashMap;
use std::path::Path;

/// Cached regex for environment variable substitution (${ VAR_NAME })
/// Compiled once at first use, reused for all subsequent config loads
static ENV_VAR_PATTERN: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"\$\{([^}]+)\}").expect("Environment variable substitution regex is valid")
});

// =============================================================================
// State Schema Structures
// =============================================================================

/// Represents a single field in the state schema
///
/// Represents a field in the agent's state schema.
///
/// Can be deserialized from either:
/// - Structured format: `{ type: "string", description: "..." }`
/// - Legacy format: `"string - description"`
#[derive(Debug, Clone, Serialize)]
pub struct StateField {
    /// The type of the field (e.g., "string", "object", "Array<T>")
    pub field_type: String,

    /// Human-readable description of the field
    #[serde(default)]
    pub description: String,
}

impl StateField {
    /// Create a new state field
    pub fn new(field_type: impl Into<String>, description: impl Into<String>) -> Self {
        Self {
            field_type: field_type.into(),
            description: description.into(),
        }
    }

    /// Parse from the legacy "type - description" format
    pub fn from_legacy(s: &str) -> Self {
        if let Some((type_part, desc_part)) = s.split_once(" - ") {
            Self {
                field_type: type_part.trim().to_string(),
                description: desc_part.trim().to_string(),
            }
        } else {
            Self {
                field_type: s.to_string(),
                description: String::new(),
            }
        }
    }

    /// Convert to legacy "type - description" format
    pub fn to_legacy(&self) -> String {
        if self.description.is_empty() {
            self.field_type.clone()
        } else {
            format!("{} - {}", self.field_type, self.description)
        }
    }
}

impl<'de> Deserialize<'de> for StateField {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        #[serde(untagged)]
        enum StateFieldRaw {
            Structured {
                #[serde(rename = "type")]
                field_type: String,
                #[serde(default)]
                description: String,
            },
            Legacy(String),
        }

        match StateFieldRaw::deserialize(deserializer)? {
            StateFieldRaw::Structured {
                field_type,
                description,
            } => Ok(StateField {
                field_type,
                description,
            }),
            StateFieldRaw::Legacy(s) => Ok(StateField::from_legacy(&s)),
        }
    }
}

/// Backwards-compatibility type alias
///
/// Use [`StateField`] instead. This is maintained for legacy code.
#[deprecated(since = "0.2.0", note = "Use `StateField` instead")]
pub type StateSchemaField = StateField;

// =============================================================================
// Environment Variable Substitution
// =============================================================================

/// Substitute environment variables in the format `${VAR_NAME}`
///
/// This allows keeping sensitive information like API keys out of the codebase.
/// If a variable is not found, it logs a warning and uses the literal value.
///
/// # Performance
/// Uses a cached regex pattern compiled once at first use.
/// Subsequent calls reuse the compiled pattern.
///
/// # Example
/// In your config: `api_key: ${OPENAI_API_KEY}`
/// Gets replaced with the actual `OPENAI_API_KEY` environment variable.
fn substitute_env_vars(input: &str) -> String {
    ENV_VAR_PATTERN
        .replace_all(input, |caps: &regex::Captures| {
            let var_name = &caps[1];
            std::env::var(var_name).unwrap_or_else(|_| {
                eprintln!(
                    "Warning: Environment variable {} not found, using literal value",
                    var_name
                );
                caps[0].to_string()
            })
        })
        .to_string()
}

// =============================================================================
// Main Configuration
// =============================================================================

/// Complete agent configuration
///
/// Contains everything needed to create and run an agent:
/// - Agent metadata (name, description)
/// - LLM provider settings
/// - Graph structure (nodes and edges)
/// - Optional state schema documentation
/// - Optional state validation schema
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentConfig {
    /// Unique name for this agent
    pub name: String,

    /// Human-readable description of what the agent does
    pub description: Option<String>,

    /// LLM provider configuration
    pub llm: LLMConfig,

    /// Graph structure definition
    pub graph: GraphConfig,

    /// Schema describing the state shape
    /// (Required for proper state initialization and validation)
    ///
    /// Fields can be defined in structured format:
    /// ```yaml
    /// state_schema:
    ///   field_name:
    ///     type: string
    ///     description: "Field description"
    /// ```
    ///
    /// Or legacy format:
    /// ```yaml
    /// state_schema:
    ///   field_name: "string - Field description"
    /// ```
    pub state_schema: HashMap<String, StateField>,

    /// Optional state validation schema for runtime validation
    /// Can be set programmatically via with_validation_schema()
    #[serde(skip)]
    pub validation_schema: Option<std::sync::Arc<StateSchema>>,

    /// Optional memory configuration (checkpointer, conversation memory, buffer/window).
    /// Omitted or empty = memory features disabled.
    #[serde(default)]
    pub memory: MemoryConfig,

    /// Optional evaluation configuration.
    /// When enabled, the agent automatically adds AutoEvaluationMiddleware.
    #[serde(default)]
    pub evaluation: Option<EvaluationConfig>,
}

/// Planner configuration for LLM-driven dynamic routing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlannerConfig {
    /// Enable planner (optional). Default false.
    #[serde(default)]
    pub enabled: bool,

    /// Maximum planning iterations (re-plan after each step). Default 5.
    #[serde(default = "default_max_plan_steps")]
    pub max_plan_steps: usize,

    /// Optional prompt template for the planner LLM call.
    #[serde(default)]
    pub prompt_template: Option<String>,
}

fn default_max_plan_steps() -> usize {
    5
}

impl Default for PlannerConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            max_plan_steps: 5,
            prompt_template: None,
        }
    }
}

/// The execution graph structure
///
/// Defines the workflow: which nodes exist, how they're connected,
/// and what MCPs are available.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphConfig {
    /// All computation nodes in the graph
    pub nodes: Vec<NodeConfig>,

    /// All connections between nodes
    pub edges: Vec<EdgeConfig>,

    /// Optional MCP tool configurations
    /// These are referenced by nodes via the `mcps` field
    #[serde(default)]
    pub mcps: HashMap<String, MCPConfig>,

    /// Optional RAG (vector store + embeddings) configuration
    #[serde(default)]
    pub rag: Option<RAGGraphConfig>,

    /// Optional planner configuration for LLM-driven dynamic routing
    #[serde(default)]
    pub planner: PlannerConfig,

    /// Optional explicit parallel groups. When nodes in a group are in the same frontier, they run concurrently (tokio).
    /// If omitted, all frontier nodes run in parallel (DAG-based scheduler).
    /// Example: `parallel: [[analyze_logs, analyze_pcap], [step_a, step_b]]`
    #[serde(default)]
    pub parallel: Vec<Vec<String>>,

    /// Allow cycles in the graph. Defaults to true.
    /// Set to false to enforce strict DAG mode (cycles will be rejected at build time).
    #[serde(default = "eval_default_true")]
    pub allow_cycles: bool,

    /// Maximum number of execution steps before the runtime aborts with an error.
    /// Prevents infinite loops in cyclic graphs. Defaults to 25.
    /// Raise this for long pipelines; lower it for tighter safety in production.
    #[serde(default = "default_recursion_limit")]
    pub recursion_limit: usize,
}

// =============================================================================
// Evaluation Configuration
// =============================================================================

/// Configuration for automatic evaluation from config.yaml
///
/// Example:
/// ```yaml
/// evaluation:
///   enabled: true
///   min_confidence: 0.8
///   max_retries: 3
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvaluationConfig {
    /// Whether evaluation is enabled
    #[serde(default = "eval_default_true")]
    pub enabled: bool,

    /// Minimum confidence threshold (0.0-1.0). Retry if below this.
    #[serde(default = "eval_default_min_confidence")]
    pub min_confidence: f64,

    /// Maximum number of retry attempts
    #[serde(default = "eval_default_max_retries")]
    pub max_retries: u32,

    /// Optional scoring configuration
    #[serde(default)]
    pub scoring: Option<ScoringConfig>,

    /// Optional grading configuration
    #[serde(default)]
    pub grading: Option<GradingConfig>,

    /// Retry policy: "exponential", "linear", or "fixed"
    #[serde(default)]
    pub retry_policy: Option<String>,

    /// Retry delay in milliseconds
    #[serde(default)]
    pub retry_delay_ms: Option<u64>,
}

/// Scoring sub-configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScoringConfig {
    /// Metrics to measure (e.g. ["relevance", "completeness", "accuracy"])
    #[serde(default)]
    pub metrics: Vec<String>,

    /// Weights for each metric (should sum to 1.0)
    #[serde(default)]
    pub weights: Vec<f64>,
}

/// Grading sub-configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GradingConfig {
    /// Whether grading is enabled
    #[serde(default)]
    pub enabled: bool,

    /// Rubric for grading
    #[serde(default)]
    pub rubric: Option<String>,
}

fn eval_default_true() -> bool { true }
fn default_recursion_limit() -> usize { 25 }
fn eval_default_min_confidence() -> f64 { 0.8 }
fn eval_default_max_retries() -> u32 { 3 }

// =============================================================================
// RAG Configuration
// =============================================================================

/// RAG configuration section in the graph config
///
/// Example:
/// ```yaml
/// graph:
///   rag:
///     vector_store:
///       type: chroma
///       endpoint: "http://localhost:8000"
///       collection: "my_docs"
///     embeddings:
///       provider: openai
///       model: text-embedding-3-small
///       api_key: ${OPENAI_API_KEY}
///     retrieval:
///       top_k: 5
///       similarity_threshold: 0.7
///     pdf:
///       chunk_size: 1000
///       chunk_overlap: 200
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RAGGraphConfig {
    /// Vector store configuration
    pub vector_store: VectorStoreConfig,

    /// Embeddings provider configuration
    pub embeddings: EmbeddingsConfig,

    /// Retrieval settings
    #[serde(default)]
    pub retrieval: RetrievalSettings,

    /// PDF processing settings
    #[serde(default)]
    pub pdf: PdfSettings,
}

/// Vector store configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorStoreConfig {
    /// Store type: "chroma", "pinecone", "memory", etc.
    #[serde(rename = "type")]
    pub store_type: VectorStoreType,

    /// Endpoint URL (for chroma, weaviate, etc.)
    #[serde(default)]
    pub endpoint: Option<String>,

    /// Collection/index name
    #[serde(default = "default_collection_name")]
    pub collection: String,

    /// API key (for pinecone, etc.)
    #[serde(default)]
    pub api_key: Option<String>,

    /// Embedding dimension (auto-detected from embeddings config if omitted)
    #[serde(default)]
    pub embedding_dim: Option<usize>,
}

fn default_collection_name() -> String {
    "documents".to_string()
}

/// Embeddings provider configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingsConfig {
    /// Provider: "openai", "mock"
    pub provider: String,

    /// Model name (e.g. "text-embedding-3-small")
    #[serde(default)]
    pub model: Option<String>,

    /// API key (supports ${ENV_VAR} substitution)
    #[serde(default)]
    pub api_key: Option<String>,

    /// Embedding dimension override
    #[serde(default)]
    pub dimension: Option<usize>,
}

/// Retrieval settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetrievalSettings {
    /// Number of top results to retrieve
    #[serde(default = "default_top_k")]
    pub top_k: usize,

    /// Minimum similarity threshold
    #[serde(default = "default_similarity_threshold")]
    pub similarity_threshold: f32,
}

fn default_top_k() -> usize { 5 }
fn default_similarity_threshold() -> f32 { 0.7 }

impl Default for RetrievalSettings {
    fn default() -> Self {
        Self {
            top_k: 5,
            similarity_threshold: 0.7,
        }
    }
}

/// PDF processing settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PdfSettings {
    /// Target chunk size in characters
    #[serde(default = "default_chunk_size")]
    pub chunk_size: usize,

    /// Overlap between chunks in characters
    #[serde(default = "default_chunk_overlap")]
    pub chunk_overlap: usize,
}

fn default_chunk_size() -> usize { 1000 }
fn default_chunk_overlap() -> usize { 200 }

impl Default for PdfSettings {
    fn default() -> Self {
        Self {
            chunk_size: 1000,
            chunk_overlap: 200,
        }
    }
}

impl RAGGraphConfig {
    /// Resolve environment variables in API keys
    pub fn resolve_env_vars(&mut self) {
        if let Some(ref key) = self.embeddings.api_key {
            self.embeddings.api_key = Some(substitute_env_vars(key));
        }
        if let Some(ref key) = self.vector_store.api_key {
            self.vector_store.api_key = Some(substitute_env_vars(key));
        }
    }

    /// Get the embedding dimension (from explicit config or model defaults)
    pub fn embedding_dimension(&self) -> usize {
        if let Some(dim) = self.embeddings.dimension {
            return dim;
        }
        if let Some(dim) = self.vector_store.embedding_dim {
            return dim;
        }
        // Infer from provider + model
        match self.embeddings.provider.as_str() {
            "mistral" => 1024, // mistral-embed is always 1024
            "ollama" => match self.embeddings.model.as_deref() {
                Some("nomic-embed-text") => 768,
                Some("all-minilm") => 384,
                Some("mxbai-embed-large") => 1024,
                _ => 768,
            },
            _ => match self.embeddings.model.as_deref() {
                Some("text-embedding-3-large") => 3072,
                Some("text-embedding-3-small") => 1536,
                Some("text-embedding-ada-002") => 1536,
                _ => 1536,
            },
        }
    }
}

impl AgentConfig {
    /// Load configuration from a YAML file, validating and providing clear error messages
    pub fn from_file(path: impl AsRef<Path>) -> crate::core::error::Result<Self> {
        let content = std::fs::read_to_string(&path).map_err(|e| {
            crate::core::error::FlowgentraError::ConfigError(format!(
                "Failed to read config file '{}': {}",
                path.as_ref().display(), e
            ))
        })?;
        let config = Self::from_yaml_str(&content).map_err(|e| {
            crate::core::error::FlowgentraError::ConfigError(format!(
                "Failed to parse config file '{}': {}",
                path.as_ref().display(), e
            ))
        })?;
        config.validate().map_err(|e| {
            crate::core::error::FlowgentraError::ConfigError(format!(
                "Invalid configuration in '{}': {}",
                path.as_ref().display(), e
            ))
        })?;
        Ok(config)
    }

    /// Load configuration from a YAML string
    pub fn from_yaml_str(yaml: &str) -> crate::core::error::Result<Self> {
        let config: RawAgentConfig = serde_yml::from_str(yaml)
            .map_err(|e| crate::core::error::FlowgentraError::YamlError(e.to_string()))?;
        config.into_agent_config()
    }

    /// Validate the configuration
    pub fn validate(&self) -> crate::core::error::Result<()> {
        // Validate individual node configurations
        for node in &self.graph.nodes {
            node.validate()?;
        }

        // Check all edges reference existing nodes
        let node_names: std::collections::HashSet<_> =
            self.graph.nodes.iter().map(|n| &n.name).collect();

        for edge in &self.graph.edges {
            if !node_names.contains(&edge.from) && edge.from != "START" {
                return Err(crate::core::error::FlowgentraError::ConfigError(format!(
                    "Edge references non-existent node: {}",
                    edge.from
                )));
            }
            for to in &edge.to {
                if !node_names.contains(to) && *to != "END" {
                    return Err(crate::core::error::FlowgentraError::ConfigError(format!(
                        "Edge references non-existent node: {}",
                        to
                    )));
                }
            }
        }

        // Check for START and END nodes
        let has_start = self.graph.edges.iter().any(|e| e.from == "START");
        let has_end = self
            .graph
            .edges
            .iter()
            .any(|e| e.to.iter().any(|t| t == "END"));

        if !has_start {
            return Err(crate::core::error::FlowgentraError::ConfigError(
                "Graph must have at least one edge from START".to_string(),
            ));
        }

        if !has_end {
            return Err(crate::core::error::FlowgentraError::ConfigError(
                "Graph must have at least one edge to END".to_string(),
            ));
        }

        Ok(())
    }

    /// Set the validation schema for this configuration
    ///
    /// The schema will be used to validate state before and during execution.
    pub fn with_validation_schema(mut self, schema: StateSchema) -> Self {
        self.validation_schema = Some(std::sync::Arc::new(schema));
        self
    }

    /// Get the validation schema if configured
    pub fn get_validation_schema(&self) -> Option<&StateSchema> {
        self.validation_schema.as_ref().map(|arc: &Arc<crate::core::state::state_validation::StateSchema>| arc.as_ref())
    }

    /// Create an initial state from the configured state_schema
    ///
    /// Initializes all fields defined in state_schema with null values.
    /// Users can then set actual values via `state.set()`.
    ///
    /// Create initial state from config
    /// Uses SharedState (the default state type) for state creation
    ///
    /// # Example
    /// ```ignore
    /// let config = AgentConfig::from_file("agent.yaml")?;
    /// let state = config.create_initial_state();
    /// ```
    pub fn create_initial_state(&self) -> crate::core::state::SharedState {
        let state = crate::core::state::SharedState::new(crate::core::state::PlainState::new());

        // Initialize all schema fields with null
        for key in self.state_schema.keys() {
            state.set(key.clone(), serde_json::Value::Null);
        }

        state
    }

    /// Validate a state against the configured schema
    ///
    /// Returns Ok(()) if validation passes, Err if configured schema
    /// validation fails or if schema is not configured.
    pub fn validate_state<T: crate::core::state::State>(
        &self,
        state: &T,
    ) -> crate::core::error::Result<()> {
        if let Some(schema) = &self.validation_schema {
            schema.validate(state).map_err(|errors: Vec<ValidationError>| {
                let error_msgs = errors
                    .iter()
                    .map(|e| format!("{}: {}", e.field, e.message))
                    .collect::<Vec<_>>()
                    .join("; ");
                crate::core::error::FlowgentraError::ValidationError(format!(
                    "State validation failed: {}",
                    error_msgs
                ))
            })?;
        }
        Ok(())
    }

    /// Create an LLM client based on the agent's LLM configuration
    ///
    /// # Example
    /// ```ignore
    /// let config = AgentConfig::from_file("agent.yaml")?;
    /// let llm_client = config.create_llm_client()?;
    /// ```
    pub fn create_llm_client(
        &self,
    ) -> crate::core::error::Result<std::sync::Arc<dyn crate::core::llm::LLMClient>> {
        self.llm.create_client()
    }
}

/// Intermediate structure for YAML deserialization
#[derive(Debug, Deserialize)]
struct RawAgentConfig {
    name: String,
    description: Option<String>,
    llm: RawLLMConfig,
    graph: GraphConfig,
    state_schema: Option<HashMap<String, StateField>>,
    #[serde(default)]
    memory: MemoryConfig,
    #[serde(default)]
    evaluation: Option<EvaluationConfig>,
}

#[derive(Debug, Deserialize)]
struct RawLLMConfig {
    provider: String,
    model: String,
    temperature: Option<f32>,
    max_tokens: Option<usize>,
    top_p: Option<f32>,
    api_key: String,
    #[serde(default)]
    extra_params: Option<HashMap<String, serde_json::Value>>,
}

impl RawAgentConfig {
    fn into_agent_config(self) -> crate::core::error::Result<AgentConfig> {
        let provider_str = self.llm.provider.to_lowercase();

        let provider = match provider_str.as_str() {
            "openai" => LLMProvider::OpenAI,
            "anthropic" => LLMProvider::Anthropic,
            "mistral" => LLMProvider::Mistral,
            "groq" => LLMProvider::Groq,
            "ollama" => LLMProvider::Ollama,
            "azure" => LLMProvider::Azure,
            custom => LLMProvider::Custom(custom.to_string()),
        };

        // Substitute environment variables in the API key
        let api_key = substitute_env_vars(&self.llm.api_key);

        let mut llm_config = LLMConfig::new(provider, self.llm.model, api_key);

        if let Some(temp) = self.llm.temperature {
            llm_config = llm_config.with_temperature(temp);
        }
        if let Some(max) = self.llm.max_tokens {
            llm_config = llm_config.with_max_tokens(max);
        }
        if let Some(top_p) = self.llm.top_p {
            llm_config = llm_config.with_top_p(top_p);
        }

        // Add extra parameters (e.g., for Azure OpenAI)
        if let Some(extra_params) = self.llm.extra_params {
            for (key, value) in extra_params {
                llm_config = llm_config.with_extra_param(key, value);
            }
        }

        let config = AgentConfig {
            name: self.name,
            description: self.description,
            llm: llm_config,
            graph: self.graph,
            state_schema: self.state_schema.unwrap_or_default(),
            validation_schema: None, // Set via with_validation_schema()
            memory: self.memory,
            evaluation: self.evaluation,
        };

        Ok(config)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_parsing() {
        let yaml = r#"
name: coding_agent
description: "Agent that adds placeholder docstrings to Python functions/classes"

llm:
  provider: mistral
  model: mistral-medium
  temperature: 0.2
  api_key: ${MISTRAL_API_KEY}

graph:
  nodes:
    - name: collect_files
      function: agents.coding_agent.tools.add_docstrings.collect_files
  edges:
    - from: START
      to: collect_files
    - from: collect_files
      to: END

state_schema:
  path: str
  files: list
"#;

        let result = AgentConfig::from_yaml_str(yaml);
        assert!(result.is_ok());
        let config = result.unwrap();
        assert_eq!(config.name, "coding_agent");
    }

    #[test]
    fn test_state_schema_structured_format() {
        let yaml = r#"
name: test_agent
description: "Test agent with structured state schema"

llm:
  provider: openai
  model: gpt-4
  api_key: ${OPENAI_API_KEY}

graph:
  nodes:
    - name: test_node
      function: test.handler
  edges:
    - from: START
      to: test_node
    - from: test_node
      to: END

state_schema:
  input:
    type: string
    description: "User input query"
  output:
    type: object
    description: "Formatted final output"
"#;

        let result = AgentConfig::from_yaml_str(yaml);
        assert!(result.is_ok());
        let config = result.unwrap();

        // Verify structured format was parsed correctly
        assert!(config.state_schema.contains_key("input"));
        assert!(config.state_schema.contains_key("output"));

        let input_field = &config.state_schema["input"];
        assert_eq!(input_field.field_type, "string");
        assert_eq!(input_field.description, "User input query");

        let output_field = &config.state_schema["output"];
        assert_eq!(output_field.field_type, "object");
        assert_eq!(output_field.description, "Formatted final output");
    }

    #[test]
    fn test_state_schema_legacy_format() {
        let yaml = r#"
name: test_agent
description: "Test agent with legacy state schema"

llm:
  provider: openai
  model: gpt-4
  api_key: ${OPENAI_API_KEY}

graph:
  nodes:
    - name: test_node
      function: test.handler
  edges:
    - from: START
      to: test_node
    - from: test_node
      to: END

state_schema:
  input: "string - User input query"
  output: "object - Formatted final output"
"#;

        let result = AgentConfig::from_yaml_str(yaml);
        assert!(result.is_ok());
        let config = result.unwrap();

        // Verify legacy format was parsed correctly
        assert!(config.state_schema.contains_key("input"));
        assert!(config.state_schema.contains_key("output"));

        let input_field = &config.state_schema["input"];
        assert_eq!(input_field.field_type, "string");
        assert_eq!(input_field.description, "User input query");

        let output_field = &config.state_schema["output"];
        assert_eq!(output_field.field_type, "object");
        assert_eq!(output_field.description, "Formatted final output");
    }

    #[test]
    fn test_state_schema_field_conversion() {
        // Test legacy parsing and conversion
        let field = StateField::from_legacy("string - User input query");
        assert_eq!(field.field_type, "string");
        assert_eq!(field.description, "User input query");
        assert_eq!(field.to_legacy(), "string - User input query");

        // Test without description
        let field_no_desc = StateField::from_legacy("bool");
        assert_eq!(field_no_desc.field_type, "bool");
        assert_eq!(field_no_desc.description, "");
    }
}
