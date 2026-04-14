//! NoSQL / document store trait and backend implementations.
//!
//! The [`DocumentStore`] trait provides a minimal, backend-agnostic interface
//! for inserting, querying, and deleting JSON documents. All operations work
//! with [`serde_json::Value`] so callers never import MongoDB or other
//! backend types directly.
//!
//! Backends are feature-gated:
//! - `mongodb-store`        → [`mongodb::MongoDocumentStore`]
//! - `redis-store`          → [`redis::RedisDocumentStore`]
//! - `neo4j-store`          → [`neo4j::Neo4jDocumentStore`]
//! - `cassandra-store`      → [`cassandra::CassandraDocumentStore`]
//! - `elasticsearch-store`  → [`elasticsearch::ElasticsearchDocumentStore`]

use async_trait::async_trait;
use serde_json::Value;

use super::DbError;

#[cfg(feature = "mongodb-store")]
pub mod mongodb;

#[cfg(feature = "redis-store")]
pub mod redis;

#[cfg(feature = "neo4j-store")]
pub mod neo4j;

#[cfg(feature = "cassandra-store")]
pub mod cassandra;

#[cfg(feature = "elasticsearch-store")]
pub mod elasticsearch;

pub mod template;

/// Minimal async interface for document (NoSQL) databases.
///
/// Implement this trait to add a new document store backend (see `template.rs`).
#[async_trait]
pub trait DocumentStore: Send + Sync {
    /// Insert a JSON document into `collection`.
    ///
    /// Returns the inserted document's ID as a string (e.g. MongoDB ObjectId).
    async fn insert(&self, collection: &str, doc: Value) -> Result<String, DbError>;

    /// Find documents in `collection` matching `filter`.
    ///
    /// `filter` is a backend-specific JSON query — e.g. MongoDB query syntax
    /// `{"name": "Alice"}` or `{"age": {"$gt": 25}}`.
    async fn find(&self, collection: &str, filter: Value) -> Result<Vec<Value>, DbError>;

    /// Delete the document with the given `id` from `collection`.
    async fn delete(&self, collection: &str, id: &str) -> Result<(), DbError>;
}
