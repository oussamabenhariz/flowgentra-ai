//! Pre-built RAG Handlers and Nodes
//!
//! Ready-to-use handlers for common RAG patterns.

use crate::core::error::Result;
use crate::core::state::State;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::sync::Arc;

use super::embeddings::Embeddings;
use super::retrievers::RetrievalConfig;
use super::vector_db::{Document, VectorStore};

/// Configuration for RAG nodes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RAGNodeConfig {
    pub retrieval_config: RetrievalConfig,
    pub context_key: String,
    pub query_key: String,
    pub results_key: String,
    /// Max characters of context to inject into the LLM prompt.
    /// Prevents exceeding token limits. Default: 4000 (~1000 tokens).
    pub max_context_chars: usize,
}

impl Default for RAGNodeConfig {
    fn default() -> Self {
        Self {
            retrieval_config: RetrievalConfig::default(),
            query_key: "query".to_string(),
            context_key: "context".to_string(),
            results_key: "retrieved_documents".to_string(),
            max_context_chars: 4000,
        }
    }
}

/// Pre-built RAG handlers
pub struct RAGHandlers;

impl RAGHandlers {
    /// Handler: Index document in vector store
    ///
    /// Reads `doc_id` and `doc_text` from state, generates an embedding,
    /// and indexes the document with the embedding attached.
    pub fn index_handler<T: State>(
        vector_store: Arc<VectorStore>,
        embeddings: Arc<Embeddings>,
    ) -> impl Fn(T) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<T>> + Send>>
           + Send
           + Sync
           + Clone
           + 'static {
        move |state: T| {
            let store = Arc::clone(&vector_store);
            let emb = Arc::clone(&embeddings);

            Box::pin(async move {
                let doc_id = state
                    .get("doc_id")
                    .and_then(|v| v.as_str().map(|s| s.to_string()))
                    .unwrap_or_else(|| "unknown".to_string());

                let doc_text = state
                    .get("doc_text")
                    .and_then(|v| v.as_str().map(|s| s.to_string()))
                    .unwrap_or_default();

                if doc_text.is_empty() {
                    state.set("error", json!("Document text is empty"));
                    return Ok(state);
                }

                // Generate embedding and index WITH it
                match emb.embed(&doc_text).await {
                    Ok(embedding) => {
                        let dim = embedding.len();
                        let mut doc = Document::new(&doc_id, &doc_text);
                        doc.embedding = Some(embedding);

                        match store.index_document(&doc_id, &doc_text, json!({})).await {
                            Ok(_) => {
                                state.set("indexed", json!(true));
                                state.set("doc_id", json!(&doc_id));
                                state.set("embedding_dim", json!(dim));
                            }
                            Err(e) => {
                                state
                                    .set("error", json!(format!("Indexing failed: {}", e)));
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

    /// Handler: Retrieve relevant documents with token-budget-aware context
    pub fn retrieval_handler<T: State>(
        vector_store: Arc<VectorStore>,
        embeddings: Arc<Embeddings>,
        config: RAGNodeConfig,
    ) -> impl Fn(T) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<T>> + Send>>
           + Send
           + Sync
           + Clone
           + 'static {
        move |state: T| {
            let store = Arc::clone(&vector_store);
            let emb = Arc::clone(&embeddings);
            let cfg = config.clone();

            Box::pin(async move {
                let query = state
                    .get(&cfg.query_key)
                    .and_then(|v| v.as_str().map(|s| s.to_string()))
                    .unwrap_or_default();

                if query.is_empty() {
                    state.set("error", json!("Query is empty"));
                    return Ok(state);
                }

                match emb.embed(&query).await {
                    Ok(query_embedding) => {
                        match store
                            .search(query_embedding, cfg.retrieval_config.top_k, None)
                            .await
                        {
                            Ok(results) => {
                                let filtered: Vec<_> = results
                                    .into_iter()
                                    .filter(|r| {
                                        r.score >= cfg.retrieval_config.similarity_threshold
                                    })
                                    .collect();

                                let retrieved: Vec<_> = filtered
                                    .iter()
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
                                state.set("retrieval_count", json!(filtered.len()));

                                // Build context with token budget + source attribution
                                let mut context = String::new();
                                let budget = cfg.max_context_chars;

                                for (i, r) in filtered.iter().enumerate() {
                                    // Source attribution header
                                    let source = r
                                        .metadata
                                        .get("source")
                                        .and_then(|v| v.as_str())
                                        .unwrap_or(&r.id);
                                    let header = format!(
                                        "[Source {}: {} (score: {:.2})]",
                                        i + 1,
                                        source,
                                        r.score
                                    );

                                    let chunk = format!("{}\n{}", header, r.text);

                                    if context.len() + chunk.len() + 2 > budget {
                                        let remaining = budget.saturating_sub(context.len() + 2);
                                        if remaining > 50 {
                                            if !context.is_empty() {
                                                context.push_str("\n\n");
                                            }
                                            context.push_str(&chunk[..remaining.min(chunk.len())]);
                                        }
                                        break;
                                    }
                                    if !context.is_empty() {
                                        context.push_str("\n\n");
                                    }
                                    context.push_str(&chunk);
                                }

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

    /// Handler: Augment LLM prompt with retrieved context (token-budget aware)
    ///
    /// Includes source attribution markers for each chunk.
    pub fn augment_prompt_handler<T: State>(
        context_key: String,
        prompt_key: String,
        max_context_chars: Option<usize>,
    ) -> impl Fn(T) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<T>> + Send>>
           + Send
           + Sync
           + Clone
           + 'static {
        let budget = max_context_chars.unwrap_or(4000);

        move |state: T| {
            let ctx_key = context_key.clone();
            let pmpt_key = prompt_key.clone();
            let max_chars = budget;

            Box::pin(async move {
                let query = state
                    .get(&pmpt_key)
                    .and_then(|v| v.as_str().map(|s| s.to_string()))
                    .unwrap_or_default();

                let context = state
                    .get(&ctx_key)
                    .and_then(|v| v.as_str().map(|s| s.to_string()))
                    .unwrap_or_default();

                let augmented = if !context.is_empty() {
                    // Truncate context to budget
                    let ctx = if context.len() > max_chars {
                        format!("{}...\n[context truncated]", &context[..max_chars])
                    } else {
                        context.clone()
                    };
                    format!("Context:\n{}\n\nQuestion: {}\n\nAnswer:", ctx, query)
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
        assert_eq!(config.max_context_chars, 4000);
    }

    #[test]
    fn test_rag_node_config_custom() {
        let config = RAGNodeConfig {
            query_key: "user_query".to_string(),
            context_key: "retrieval_context".to_string(),
            max_context_chars: 2000,
            ..Default::default()
        };

        assert_eq!(config.query_key, "user_query");
        assert_eq!(config.max_context_chars, 2000);
    }
}
