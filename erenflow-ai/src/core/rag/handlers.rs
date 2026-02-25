//! Pre-built RAG Handlers and Nodes
//!
//! ready-to-use handlers for common RAG patterns.

use crate::core::error::Result;
use crate::core::state::State;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::sync::Arc;

use super::embeddings::Embeddings;
use super::retrievers::RetrievalConfig;
use super::vector_db::VectorStore;

/// Configuration for RAG nodes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RAGNodeConfig {
    pub retrieval_config: RetrievalConfig,
    pub context_key: String,
    pub query_key: String,
    pub results_key: String,
}

impl Default for RAGNodeConfig {
    fn default() -> Self {
        Self {
            retrieval_config: RetrievalConfig::default(),
            query_key: "query".to_string(),
            context_key: "context".to_string(),
            results_key: "retrieved_documents".to_string(),
        }
    }
}

/// Pre-built RAG handlers
pub struct RAGHandlers;

impl RAGHandlers {
    /// Handler: Index document in vector store
    pub fn index_handler(
        vector_store: Arc<VectorStore>,
        embeddings: Arc<Embeddings>,
    ) -> impl Fn(State) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<State>> + Send>>
           + Send
           + Sync
           + Clone
           + 'static {
        move |mut state: State| {
            let store = Arc::clone(&vector_store);
            let emb = Arc::clone(&embeddings);

            Box::pin(async move {
                // Get document from state
                let doc_id = state
                    .get("doc_id")
                    .and_then(|v| v.as_str())
                    .unwrap_or("unknown")
                    .to_string();

                let doc_text = state
                    .get("doc_text")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();

                if doc_text.is_empty() {
                    state.set("error", json!("Document text is empty"));
                    return Ok(state);
                }

                // Generate embedding
                match emb.embed(&doc_text).await {
                    Ok(embedding) => {
                        // Index document
                        match store.index_document(&doc_id, &doc_text, json!({})).await {
                            Ok(_) => {
                                state.set("indexed", json!(true));
                                state.set("doc_id", json!(&doc_id));
                                state.set("embedding_dim", json!(embedding.len()));
                            }
                            Err(e) => {
                                state.set("error", json!(format!("Indexing failed: {}", e)));
                            }
                        }
                    }
                    Err(e) => {
                        state.set("error", json!(format!("Embedding failed: {}", e)));
                    }
                }

                Ok(state)
            })
        }
    }

    /// Handler: Retrieve relevant documents
    pub fn retrieval_handler(
        vector_store: Arc<VectorStore>,
        embeddings: Arc<Embeddings>,
        config: RAGNodeConfig,
    ) -> impl Fn(State) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<State>> + Send>>
           + Send
           + Sync
           + Clone
           + 'static {
        move |mut state: State| {
            let store = Arc::clone(&vector_store);
            let emb = Arc::clone(&embeddings);
            let cfg = config.clone();

            Box::pin(async move {
                // Get query from state
                let query = state
                    .get(&cfg.query_key)
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();

                if query.is_empty() {
                    state.set("error", json!("Query is empty"));
                    return Ok(state);
                }

                // Generate query embedding
                match emb.embed(&query).await {
                    Ok(query_embedding) => {
                        // Search vector store
                        match store
                            .search(query_embedding, cfg.retrieval_config.top_k)
                            .await
                        {
                            Ok(results) => {
                                let retrieved: Vec<_> = results
                                    .into_iter()
                                    .filter(|r| {
                                        r.score >= cfg.retrieval_config.similarity_threshold
                                    })
                                    .map(|r| {
                                        json!({
                                            "id": r.id,
                                            "text": r.text,
                                            "score": r.score,
                                            "metadata": r.metadata
                                        })
                                    })
                                    .collect();

                                state.set(&cfg.results_key, json!(retrieved.clone()));
                                state.set("retrieval_count", json!(retrieved.len()));

                                // Create context string
                                let context = retrieved
                                    .iter()
                                    .filter_map(|doc| doc.get("text").and_then(|t| t.as_str()))
                                    .collect::<Vec<_>>()
                                    .join("\n\n");

                                state.set(&cfg.context_key, json!(context));
                            }
                            Err(e) => {
                                state.set("error", json!(format!("Retrieval failed: {}", e)));
                            }
                        }
                    }
                    Err(e) => {
                        state.set("error", json!(format!("Query embedding failed: {}", e)));
                    }
                }

                Ok(state)
            })
        }
    }

    /// Handler: Augment LLM prompt with retrieved context
    pub fn augment_prompt_handler(
        context_key: String,
        prompt_key: String,
    ) -> impl Fn(State) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<State>> + Send>>
           + Send
           + Sync
           + Clone
           + 'static {
        move |mut state: State| {
            let ctx_key = context_key.clone();
            let pmpt_key = prompt_key.clone();

            Box::pin(async move {
                let query = state
                    .get(&pmpt_key)
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();

                let context = state
                    .get(&ctx_key)
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();

                // Augment prompt with context
                let augmented = if !context.is_empty() {
                    format!("Context:\n{}\n\nQuestion: {}\n\nAnswer:", context, query)
                } else {
                    format!("Question: {}\n\nAnswer:", query)
                };

                state.set("augmented_prompt", json!(augmented));
                state.set("context_used", json!(!context.is_empty()));

                Ok(state)
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rag_node_config_default() {
        let config = RAGNodeConfig::default();
        assert_eq!(config.query_key, "query");
        assert_eq!(config.context_key, "context");
        assert_eq!(config.results_key, "retrieved_documents");
    }

    #[test]
    fn test_rag_node_config_custom() {
        let config = RAGNodeConfig {
            query_key: "user_query".to_string(),
            context_key: "retrieval_context".to_string(),
            ..Default::default()
        };

        assert_eq!(config.query_key, "user_query");
        assert_eq!(config.context_key, "retrieval_context");
    }
}
