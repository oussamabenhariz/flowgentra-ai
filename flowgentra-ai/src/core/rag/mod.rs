//! # RAG (Retrieval-Augmented Generation) Integration
//!
//! This module provides seamless integration with vector databases for RAG pipelines.
//!
//! ## Features
//!
//! - **Multiple Vector Stores**: Pinecone, Weaviate, Chroma, and local in-memory
//! - **Embedding Generation**: Support for various embedding models
//! - **Retrieval Strategies**: Semantic search, hybrid search, re-ranking
//! - **Document Management**: Indexing, updating, and deleting documents
//! - **RAG Handlers**: Pre-built nodes for common RAG patterns
//!
//! ## Quick Start
//!
//! ```ignore
//! use flowgentra_ai::core::rag::{VectorStore, Pinecone, RAGConfig};
//! use serde_json::json;
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     // Initialize vector store
//!     let config = RAGConfig::pinecone("your-index", "your-api-key")?;
//!     let store = Pinecone::new(config).await?;
//!
//!     // Index documents
//!     store.index_document(
//!         "doc-1",
//!         "Your document text here",
//!         json!({"source": "example"})
//!     ).await?;
//!
//!     // Retrieve relevant documents
//!     let results = store.retrieve("query text", 5).await?;
//!
//!     Ok(())
//! }
//! ```

pub mod cache;
pub mod chroma;
pub mod dedup;
pub mod document_loader;
pub mod embeddings;
pub mod evaluation;
pub mod handlers;
pub mod hybrid;
pub mod ingestion;
pub mod mistral_embeddings;
pub mod ollama_embeddings;
pub mod openai_embeddings;
pub mod pdf;
pub mod persistence;
pub mod reranker;
pub mod retrievers;
pub mod vector_db;

pub use cache::CachedEmbeddings;
pub use chroma::ChromaStore;
pub use dedup::{dedup_by_id, dedup_by_similarity};
pub use document_loader::{load_directory, load_document, FileType, LoadedDocument};
pub use embeddings::{EmbeddingError, EmbeddingModel, Embeddings};
pub use evaluation::{evaluate, hit_rate, mean_ndcg, mrr, EvalQuery, EvalResults, QueryResult};
pub use handlers::{RAGHandlers, RAGNodeConfig};
pub use hybrid::{bm25_score, hybrid_merge};
pub use ingestion::{IngestionPipeline, IngestionStats};
pub use mistral_embeddings::MistralEmbeddings;
pub use ollama_embeddings::OllamaEmbeddings;
pub use openai_embeddings::OpenAIEmbeddings;
pub use pdf::{chunk_text, chunk_text_by_tokens, estimate_tokens, extract_and_chunk, extract_text, PdfDocument};
pub use reranker::{LLMReranker, NoopReranker, RRFReranker, RerankStrategy, Reranker};
pub use retrievers::{QueryExpander, RetrievalConfig, RetrieverStrategy};
pub use vector_db::{Document, InMemoryVectorStore, MetadataFilter, RAGConfig, SearchResult, VectorStore, VectorStoreError, VectorStoreType};
