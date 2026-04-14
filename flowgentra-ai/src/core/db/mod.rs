//! # Database Layer
//!
//! Unified interfaces for SQL and NoSQL / document databases.
//!
//! ## Design
//!
//! Every database type is represented by a minimal async trait:
//!
//! | Trait             | Backends                                                               |
//! |-------------------|------------------------------------------------------------------------|
//! | [`SqlDatabase`]   | SQLite, PostgreSQL, MySQL, MSSQL (`sql-*` features), BigQuery, Databricks |
//! | [`DocumentStore`] | MongoDB, Redis, Neo4j, Cassandra/Astra, Elasticsearch                  |
//!
//! All trait methods operate on [`serde_json::Value`] so callers never need to
//! import backend-specific types. Concrete implementations are feature-gated so
//! only the drivers you use end up in your binary.
//!
//! ## Example — SQLite
//! ```rust,ignore
//! use flowgentra_ai::core::db::sql::SqlDatabase;
//! #[cfg(feature = "sqlite")]
//! use flowgentra_ai::core::db::sql::sqlite::SqliteDatabase;
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let db = SqliteDatabase::connect("sqlite::memory:").await?;
//!     db.execute("CREATE TABLE t (id INTEGER, name TEXT)", &[]).await?;
//!     db.execute("INSERT INTO t VALUES (1, 'Alice')", &[]).await?;
//!     let rows = db.query("SELECT * FROM t", &[]).await?;
//!     println!("{rows:?}");
//!     Ok(())
//! }
//! ```
//!
//! ## Example — MongoDB
//! ```rust,ignore
//! use flowgentra_ai::core::db::document::DocumentStore;
//! #[cfg(feature = "mongodb")]
//! use flowgentra_ai::core::db::document::mongodb::MongoDocumentStore;
//! use serde_json::json;
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let store = MongoDocumentStore::connect("mongodb://localhost:27017", "mydb").await?;
//!     let id = store.insert("users", json!({"name": "Alice"})).await?;
//!     let docs = store.find("users", json!({"name": "Alice"})).await?;
//!     store.delete("users", &id).await?;
//!     Ok(())
//! }
//! ```

pub mod document;
pub mod sql;

pub use document::DocumentStore;
pub use sql::SqlDatabase;

/// Unified error type for all database operations.
#[derive(Debug, thiserror::Error)]
pub enum DbError {
    #[error("Connection error: {0}")]
    Connection(String),

    #[error("Query error: {0}")]
    Query(String),

    #[error("Document not found: {0}")]
    NotFound(String),

    #[error("Serialization error: {0}")]
    Serialization(String),

    #[error("Operation not supported by this backend: {0}")]
    NotSupported(String),

    #[error("Configuration error: {0}")]
    Config(String),
}
