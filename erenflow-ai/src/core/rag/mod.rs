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
//! use erenflow_ai::core::rag::{VectorStore, Pinecone, RAGConfig};
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

pub mod embeddings;
pub mod handlers;
pub mod retrievers;
pub mod vector_db;

pub use embeddings::{EmbeddingModel, Embeddings};
pub use handlers::{RAGHandlers, RAGNodeConfig};
pub use retrievers::{RetrievalConfig, RetrieverStrategy};
pub use vector_db::{Document, RAGConfig, SearchResult, VectorStore, VectorStoreError};
