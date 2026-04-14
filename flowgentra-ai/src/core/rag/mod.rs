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
pub mod filter;
pub mod cross_encoder;
pub mod http_helpers;
pub mod milvus;
pub mod weaviate;
pub mod pgvector;
pub mod redis_vector;
pub mod elasticsearch;
pub mod opensearch;
pub mod upstash;
pub mod astra_db;
pub mod mongodb_atlas;
pub mod dedup;
pub mod document_loader;
pub mod embeddings;
pub mod evaluation;
pub mod handlers;
pub mod huggingface_embeddings;
pub mod hybrid;
pub mod ingestion;
pub mod mistral_embeddings;
pub mod ollama_embeddings;
pub mod openai_embeddings;
pub mod pdf;
pub mod persistence;
pub mod reranker;
pub mod retriever;
pub mod retrievers;
pub mod text_splitter;
pub mod vector_db;
// New advanced retrievers and stores
pub mod self_query_retriever;
pub mod parent_doc_retriever;
pub mod multi_vector_retriever;
pub mod time_weighted_retriever;
pub mod hnsw_store;
pub mod bm25_retriever;
pub mod ensemble_retriever;
pub mod reorder;
// Extended document loaders
pub mod loaders;
// Additional retrievers (Group 4)
pub mod compression_retriever;
pub mod score_threshold_retriever;
pub mod multi_query_retriever;
pub mod web_retrievers;
// Additional vector stores (Group 5)
pub mod extra_vector_stores;
// Additional embedding providers (Group 6)
pub mod extra_embeddings;
// SQL database agent toolkit (Group 2)
pub mod sql_database;
// Document store (Group 8)
pub mod doc_store;
// Indexing pipeline (Group 9)
pub mod indexing;

pub use cache::CachedEmbeddings;
pub use chroma::ChromaStore;
pub use milvus::MilvusStore;
pub use weaviate::WeaviateStore;
pub use pgvector::{PgVectorConfig, PgVectorStore};
pub use redis_vector::{RedisVectorConfig, RedisVectorStore};
pub use elasticsearch::{ElasticsearchConfig, ElasticsearchStore};
pub use opensearch::{OpenSearchConfig, OpenSearchStore};
pub use upstash::{UpstashVectorConfig, UpstashVectorStore};
pub use astra_db::{AstraDbConfig, AstraDbStore};
pub use mongodb_atlas::{MongoAtlasConfig, MongoAtlasVectorStore};
pub use http_helpers::resolve_env_vars;
pub use cross_encoder::CrossEncoderReranker;
pub use dedup::{dedup_by_id, dedup_by_similarity};
pub use document_loader::{load_directory, load_document, FileType, LoadedDocument};
pub use embeddings::{
    EmbeddingError, EmbeddingModel, Embeddings, EmbeddingsProvider, MockEmbeddings,
};
pub use evaluation::{evaluate, hit_rate, mean_ndcg, mrr, EvalQuery, EvalResults, QueryResult};
pub use handlers::{RAGHandlers, RAGNodeConfig};
pub use huggingface_embeddings::HuggingFaceEmbeddings;
pub use hybrid::{bm25_score, hybrid_merge};
pub use ingestion::{IngestionPipeline, IngestionStats};
pub use mistral_embeddings::MistralEmbeddings;
pub use ollama_embeddings::OllamaEmbeddings;
pub use openai_embeddings::OpenAIEmbeddings;
pub use pdf::{
    chunk_text, chunk_text_by_tokens, estimate_tokens, extract_and_chunk, extract_text, PdfDocument,
};
pub use reranker::{LLMReranker, NoopReranker, RRFReranker, RerankStrategy, Reranker};
pub use retriever::Retriever;
pub use retrievers::{QueryExpander, RetrievalConfig, RetrieverStrategy};
pub use text_splitter::{
    ChunkMetadata, CodeTextSplitter, HTMLTextSplitter, Language, MarkdownTextSplitter,
    RecursiveCharacterTextSplitter, TextChunk, TextSplitter, TokenTextSplitter,
};
pub use filter::{FilterExpr, MetadataFilter};
pub use vector_db::{
    ChromaConfig, Document, InMemoryVectorStore, MilvusConfig, PineconeConfig, PineconeStore,
    QdrantConfig, QdrantStore, RAGConfig, SearchResult, VectorStore, VectorStoreBackend,
    VectorStoreError, VectorStoreType, WeaviateConfig,
};
// Advanced retrievers
pub use self_query_retriever::{filter_from_json_object, SelfQueryConfig, SelfQueryRetriever};
pub use parent_doc_retriever::{ParentDocConfig, ParentDocument, ParentDocumentRetriever};
pub use multi_vector_retriever::{
    MultiVectorConfig, MultiVectorParent, MultiVectorRetriever, VectorView,
};
pub use time_weighted_retriever::{TimeWeightedConfig, TimeWeightedRetriever};
pub use hnsw_store::HnswVectorStore;
pub use bm25_retriever::{Bm25Config, Bm25Document, Bm25Retriever};
pub use ensemble_retriever::{
    AsyncRetriever, Bm25RetrieverAdapter, EnsembleConfig, EnsembleRetriever, VectorRetriever,
};
pub use reorder::{reorder, reorder_for_long_context, ReorderStrategy};
// Extended loaders
pub use loaders::{
    web::{WebLoader, WebLoaderConfig},
    csv::CsvLoader,
    json_loader::{JsonLoader, JsonlLoader},
    docx::DocxLoader,
    directory::{DirectoryLoader, DirectoryLoaderConfig},
    excel::ExcelLoader,
    epub::EpubLoader,
    sitemap::{SitemapConfig, SitemapLoader},
    recursive_url::{RecursiveUrlConfig, RecursiveUrlLoader},
    wikipedia::WikipediaLoader,
    arxiv::ArxivLoader,
    rss::RssFeedLoader,
    youtube::YouTubeLoader,
    s3::S3Loader,
    dataframe::DataFrameLoader,
    git::{GitLoader, GitLoaderConfig},
};
// Additional retrievers
pub use compression_retriever::{
    ContextualCompressionRetriever, DocumentCompressor, DocumentCompressorPipeline,
    EmbeddingsFilter, LLMCompressor,
};
pub use score_threshold_retriever::ScoreThresholdRetriever;
pub use multi_query_retriever::{MultiQueryConfig, MultiQueryRetriever};
pub use web_retrievers::{
    ArxivRetriever, TavilySearchDepth, TavilySearchRetriever, WikipediaRetriever,
};
// Additional vector stores
pub use extra_vector_stores::{
    AzureAISearchStore, Neo4jVectorStore, SingleStoreVectorStore, TurbopufferStore, VectaraStore,
};
// Additional embedding providers
pub use extra_embeddings::{
    AzureOpenAIEmbeddings, BedrockEmbeddings, CohereEmbeddings, GoogleVertexEmbeddings,
    JinaEmbeddings, NomicEmbeddings, TogetherEmbeddings, VoyageEmbeddings,
};
// SQL database toolkit
pub use sql_database::{
    InfoSQLDatabaseTool, ListSQLDatabaseTool, QuerySQLCheckerTool, QuerySQLDatabaseTool,
    SqlBackend, SqlDatabaseLoader, SqlDatabaseWrapper, SqlDbError,
};
// Document store
pub use doc_store::{DocStore, DocStoreError, InMemoryDocStore, LocalFileDocStore, StoredDocument};
// Indexing
pub use indexing::{
    hash_document, index, CleanupMode, IndexStats, InMemoryRecordManager, RecordEntry,
    RecordManager,
};
