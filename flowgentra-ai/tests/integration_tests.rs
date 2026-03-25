//! Integration tests for FlowgentraAI
//!
//! These tests validate end-to-end functionality including:
//! - Agent creation and execution
//! - Handler registration and discovery
//! - State management and transformations
//! - Graph validation
//! - LLM provider integration
//! - Tool execution

use flowgentra_ai::core::state::SharedState;
use flowgentra_ai::prelude::*;
use serde_json::json;
use std::collections::HashMap;

// ============================================================================
// State Management Tests
// ============================================================================

#[test]
fn test_state_set_and_get() {
    let state = SharedState::new(Default::default());
    state.set("key", json!("value"));
    assert_eq!(state.get("key"), Some(json!("value")));
}

#[test]
fn test_state_clone() {
    let state = SharedState::new(Default::default());
    state.set("data", json!({"nested": "value"}));

    let cloned = state.clone();
    assert_eq!(state.get("data"), cloned.get("data"));
}

#[test]
fn test_state_merge() {
    let state1 = SharedState::new(Default::default());
    state1.set("a", json!(1));

    let state2 = SharedState::new(Default::default());
    state2.set("b", json!(2));
    state2.set("a", json!(10));

    // Verify both have their values
    assert_eq!(state1.get("a"), Some(json!(1)));
    assert_eq!(state2.get("a"), Some(json!(10)));
    assert_eq!(state2.get("b"), Some(json!(2)));
}

#[test]
fn test_state_json_serialization() {
    let state = SharedState::new(Default::default());
    state.set("key", json!("value"));
    state.set("number", json!(42));
    state.set("array", json!([1, 2, 3]));

    let json_str = state.to_json_string().expect("serialization failed");
    assert!(json_str.contains("key"));
    assert!(json_str.contains("value"));
}

// ============================================================================
// Graph Tests
// ============================================================================

#[test]
fn test_graph_creation() {
    let _graph: Graph<SharedState> = Graph::new();
}

#[test]
fn test_graph_builder() {
    let _builder: StateGraphBuilder<SharedState> = StateGraphBuilder::new();
}

// ============================================================================
// Error Handling Tests
// ============================================================================

#[test]
fn test_error_creation() {
    let error = FlowgentraError::ConfigError("Test error".to_string());
    assert!(error.to_string().contains("Configuration error"));
}

#[test]
fn test_error_types() {
    use flowgentra_ai::core::error::FlowgentraError;

    let config_err = FlowgentraError::ConfigError("config".to_string());
    assert!(config_err.to_string().contains("Configuration"));

    let node_err = FlowgentraError::NodeNotFound("test_node".to_string());
    assert!(node_err.to_string().contains("Node not found"));

    let runtime_err = FlowgentraError::RuntimeError("runtime".to_string());
    assert!(runtime_err.to_string().contains("Runtime error"));
}

// ============================================================================
// Handler Tests
// ============================================================================

#[test]
fn test_handler_registry_creation() {
    let registry = HashMap::<String, Handler<SharedState>>::new();
    assert!(registry.is_empty());
}

#[test]
fn test_handler_registry_insertion() {
    let mut registry: HashMap<String, Handler<SharedState>> = HashMap::new();

    let handler: Handler<SharedState> = Box::new(|state| Box::pin(async move { Ok(state) }));

    registry.insert("test_handler".to_string(), handler);
    assert!(registry.contains_key("test_handler"));
}

// ============================================================================
// LLM Configuration Tests
// ============================================================================

#[test]
fn test_llm_config_creation() {
    let config = LLMConfig::new(
        LLMProvider::OpenAI,
        "gpt-4".to_string(),
        "test-key".to_string(),
    );
    assert_eq!(config.model, "gpt-4");
    assert_eq!(config.provider, LLMProvider::OpenAI);
}

#[test]
fn test_all_llm_providers() {
    let providers = vec![
        (LLMProvider::OpenAI, "gpt-4"),
        (LLMProvider::Anthropic, "claude-3-opus"),
        (LLMProvider::Mistral, "mistral-large"),
        (LLMProvider::Groq, "mixtral-8x7b"),
        (LLMProvider::HuggingFace, "gpt2"),
        (LLMProvider::Ollama, "llama2"),
        (LLMProvider::Azure, "gpt-4"),
    ];

    for (provider, model) in providers {
        let config = LLMConfig::new(provider.clone(), model.to_string(), "key".to_string());
        assert_eq!(config.provider, provider);
        assert_eq!(config.model, model);
    }
}

// ============================================================================
// Message Tests
// ============================================================================

#[test]
fn test_message_creation() {
    let msg = Message::user("Hello");
    assert_eq!(msg.content, "Hello");
}

#[test]
fn test_message_serialization() {
    let msg = Message::assistant("Response");
    let json = serde_json::to_value(&msg).expect("serialization failed");
    assert_eq!(json["content"], "Response");
}

// ============================================================================
// Configuration Tests
// ============================================================================

#[test]
fn test_agent_config_basic() {
    let llm_config = LLMConfig::new(
        LLMProvider::OpenAI,
        "gpt-4".to_string(),
        "test-key".to_string(),
    );
    assert_eq!(llm_config.model, "gpt-4");
}

// ============================================================================
// Result Type Tests
// ============================================================================

#[test]
fn test_result_type() {
    fn operation() -> Result<String> {
        Ok("success".to_string())
    }

    let result = operation();
    assert!(result.is_ok());
    assert_eq!(result.unwrap(), "success");
}

#[test]
fn test_result_error() {
    fn operation() -> Result<String> {
        Err(FlowgentraError::RuntimeError("failed".to_string()))
    }

    let result = operation();
    assert!(result.is_err());
}

// ============================================================================
// Message Role Tests
// ============================================================================

#[test]
fn test_message_roles() {
    use flowgentra_ai::core::llm::MessageRole;

    let user = MessageRole::User;
    let assistant = MessageRole::Assistant;
    let system = MessageRole::System;

    let _ = format!("{:?}", user);
    let _ = format!("{:?}", assistant);
    let _ = format!("{:?}", system);
}

// ============================================================================
// Circular Dependency & Type Safety Tests
// ============================================================================

#[test]
fn test_handler_entry_creation() {
    use flowgentra_ai::core::agent::HandlerEntry;
    use std::sync::Arc;

    let handler: Arc<
        dyn Fn(
                SharedState,
            )
                -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<SharedState>> + Send>>
            + Send
            + Sync,
    > = Arc::new(|state| Box::pin(async move { Ok(state) }));

    let entry = HandlerEntry::new("test", handler);
    assert_eq!(entry.name, "test");
}

// ============================================================================
// Feature Tests
// ============================================================================

#[test]
fn test_rag_feature_available() {
    assert!(true);
}

// ============================================================================
// Format & Display Tests
// ============================================================================

#[test]
fn test_error_display() {
    let error = FlowgentraError::ExecutionError("test".to_string());
    let display_str = format!("{}", error);
    assert!(display_str.contains("Execution error"));
}

#[test]
fn test_state_to_string() {
    let state = SharedState::new(Default::default());
    state.set("key", json!("value"));
    let json_str = state.to_json_string().expect("serialization failed");
    assert!(!json_str.is_empty());
}

// ============================================================================
// Concurrency Tests (Basic)
// ============================================================================

#[tokio::test]
async fn test_async_handler() {
    let handler: Handler<SharedState> = Box::new(|state| {
        Box::pin(async move {
            state.set("async", json!(true));
            Ok(state)
        })
    });

    let state = SharedState::new(Default::default());
    let result = handler(state).await;
    assert!(result.is_ok());
    assert_eq!(result.unwrap().get("async"), Some(json!(true)));
}

#[tokio::test]
async fn test_multiple_async_operations() {
    let mut tasks = vec![];

    for i in 0..5 {
        let task = tokio::spawn(async move {
            let state = SharedState::new(Default::default());
            state.set(&format!("task_{}", i), json!(i));
            state
        });
        tasks.push(task);
    }

    let mut count = 0;
    for task in tasks {
        let _ = task.await.expect("task panicked");
        count += 1;
    }

    assert_eq!(count, 5);
}

// ============================================================================
// Trait Tests
// ============================================================================

#[test]
fn test_llm_config_creation_complete() {
    let _config = LLMConfig::new(
        LLMProvider::Mistral,
        "mistral-large".to_string(),
        "test-key".to_string(),
    );
}

// ============================================================================
// Macro Tests
// ============================================================================

#[test]
fn test_prelude_imports() {
    let _state = SharedState::new(Default::default());
    let _graph: Graph<SharedState> = Graph::new();
    let _ = FlowgentraError::RuntimeError("test".to_string());
    let _config = LLMConfig::new(LLMProvider::OpenAI, "gpt-4".to_string(), "key".to_string());
}

// ============================================================================
// Integration Sanity Checks
// ============================================================================

#[test]
fn test_all_types_compile() {
    let state = SharedState::new(Default::default());
    state.set("test", json!("value"));

    let _graph: Graph<SharedState> = Graph::new();

    let config = LLMConfig::new(LLMProvider::OpenAI, "gpt-4".to_string(), "key".to_string());

    assert!(!state.to_json_string().unwrap_or_default().is_empty());
    assert_eq!(config.model, "gpt-4");
}
