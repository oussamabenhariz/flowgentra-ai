//! Template for adding a new document store backend.
//!
//! Copy this file, rename `MyDocumentStore`, implement the three trait methods,
//! and add the backend behind a Cargo feature flag.
//!
//! Steps:
//! 1. Copy this file to e.g. `redis.rs`
//! 2. Add `#[cfg(feature = "redis")] pub mod redis;` in `document/mod.rs`
//! 3. Add the crate dependency in `Cargo.toml` under `[dependencies]` (optional)
//! 4. Add the feature in `[features]`
//! 5. Implement `insert`, `find`, and `delete` below
//!
//! Rules:
//! - Do NOT add backend-specific public methods to the struct; keep them private.
//! - Return [`DbError`] — never expose the backend's own error type through the trait.
//! - `filter` in `find()` is a freeform JSON value; interpret it however the
//!   backend's query language requires.

use async_trait::async_trait;
use serde_json::Value;

use super::{DbError, DocumentStore};

/// Replace with your actual client/connection type.
pub struct MyDocumentStore {
    // client: MyClient,
}

impl MyDocumentStore {
    /// Connect to the database and return a new instance.
    pub async fn connect(_uri: &str, _db: &str) -> Result<Self, DbError> {
        // let client = MyClient::connect(uri, db)
        //     .await
        //     .map_err(|e| DbError::Connection(e.to_string()))?;
        // Ok(Self { client })
        Err(DbError::Config("MyDocumentStore: not yet implemented".into()))
    }
}

#[async_trait]
impl DocumentStore for MyDocumentStore {
    async fn insert(&self, _collection: &str, _doc: Value) -> Result<String, DbError> {
        // let result = self.client.insert(collection, doc).await
        //     .map_err(|e| DbError::Query(e.to_string()))?;
        // Ok(result.id.to_string())
        Err(DbError::Config("MyDocumentStore: not yet implemented".into()))
    }

    async fn find(&self, _collection: &str, _filter: Value) -> Result<Vec<Value>, DbError> {
        // let docs = self.client.find(collection, filter).await
        //     .map_err(|e| DbError::Query(e.to_string()))?;
        // Ok(docs.into_iter().map(to_json).collect())
        Err(DbError::Config("MyDocumentStore: not yet implemented".into()))
    }

    async fn delete(&self, _collection: &str, _id: &str) -> Result<(), DbError> {
        // self.client.delete(collection, id).await
        //     .map_err(|e| DbError::Query(e.to_string()))?;
        // Ok(())
        Err(DbError::Config("MyDocumentStore: not yet implemented".into()))
    }
}
