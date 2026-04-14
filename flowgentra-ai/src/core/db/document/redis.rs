//! Redis document store backend for [`DocumentStore`].
//!
//! Stores JSON documents as Redis hashes under the key `{collection}:{id}`.
//! Document IDs are auto-generated UUIDs unless the document already contains
//! an `"id"` field.
//!
//! The `find` method supports a subset of query operators via a scan-and-filter
//! approach (suitable for development/small datasets). For production at scale
//! consider using the `redis-vector` backend with RediSearch, or switching to
//! a dedicated document database.
//!
//! Enabled with the `redis-store` Cargo feature.
//!
//! # Example
//! ```rust,ignore
//! use flowgentra_ai::core::db::document::redis::RedisDocumentStore;
//! use flowgentra_ai::core::db::document::DocumentStore;
//! use serde_json::json;
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let store = RedisDocumentStore::connect("redis://127.0.0.1/").await?;
//!     let id = store.insert("users", json!({"name": "Alice", "age": 30})).await?;
//!     let results = store.find("users", json!({"name": "Alice"})).await?;
//!     store.delete("users", &id).await?;
//!     Ok(())
//! }
//! ```

use async_trait::async_trait;
use redis::AsyncCommands;
use serde_json::Value;
use uuid::Uuid;

use super::super::DbError;
use super::DocumentStore;

/// Redis document store — JSON documents stored as Redis hash fields.
pub struct RedisDocumentStore {
    client: redis::Client,
}

impl RedisDocumentStore {
    /// Connect to Redis.
    ///
    /// URL formats:
    /// - `redis://127.0.0.1/`
    /// - `redis://:password@host:6379/0`
    /// - `rediss://host:6380/` (TLS)
    pub async fn connect(url: &str) -> Result<Self, DbError> {
        let client = redis::Client::open(url)
            .map_err(|e| DbError::Connection(e.to_string()))?;
        // Verify the connection is reachable.
        let mut conn = client.get_async_connection().await
            .map_err(|e| DbError::Connection(e.to_string()))?;
        redis::cmd("PING").query_async::<_, ()>(&mut conn).await
            .map_err(|e| DbError::Connection(e.to_string()))?;
        Ok(Self { client })
    }

    fn doc_key(collection: &str, id: &str) -> String {
        format!("doc:{}:{}", collection, id)
    }

    fn index_key(collection: &str) -> String {
        format!("doc_ids:{}", collection)
    }

    async fn conn(&self) -> Result<redis::aio::Connection, DbError> {
        self.client.get_async_connection().await
            .map_err(|e| DbError::Connection(e.to_string()))
    }
}

#[async_trait]
impl DocumentStore for RedisDocumentStore {
    async fn insert(&self, collection: &str, doc: Value) -> Result<String, DbError> {
        let id = doc.get("id")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
            .unwrap_or_else(|| Uuid::new_v4().to_string());

        let json_str = serde_json::to_string(&doc)
            .map_err(|e| DbError::Serialization(e.to_string()))?;

        let mut conn = self.conn().await?;
        let key = Self::doc_key(collection, &id);
        // Store the full document JSON under a single "data" hash field.
        conn.hset::<_, _, _, ()>(&key, "data", &json_str).await
            .map_err(|e| DbError::Query(e.to_string()))?;
        // Track IDs in a set so `find` can enumerate them.
        conn.sadd::<_, _, ()>(Self::index_key(collection), &id).await
            .map_err(|e| DbError::Query(e.to_string()))?;

        Ok(id)
    }

    async fn find(&self, collection: &str, filter: Value) -> Result<Vec<Value>, DbError> {
        let mut conn = self.conn().await?;

        // Collect all document IDs for this collection.
        let ids: Vec<String> = conn.smembers(Self::index_key(collection)).await
            .map_err(|e| DbError::Query(e.to_string()))?;

        let mut results = Vec::new();
        for id in ids {
            let key = Self::doc_key(collection, &id);
            let raw: Option<String> = conn.hget(&key, "data").await
                .map_err(|e| DbError::Query(e.to_string()))?;
            if let Some(json_str) = raw {
                let doc: Value = serde_json::from_str(&json_str)
                    .map_err(|e| DbError::Serialization(e.to_string()))?;
                if matches_filter(&doc, &filter) {
                    results.push(doc);
                }
            }
        }
        Ok(results)
    }

    async fn delete(&self, collection: &str, id: &str) -> Result<(), DbError> {
        let mut conn = self.conn().await?;
        let key = Self::doc_key(collection, id);
        conn.del::<_, ()>(&key).await
            .map_err(|e| DbError::Query(e.to_string()))?;
        conn.srem::<_, _, ()>(Self::index_key(collection), id).await
            .map_err(|e| DbError::Query(e.to_string()))?;
        Ok(())
    }
}

// ── Filter helpers ───────────────────────────────────────────────────────────

/// Recursively evaluate whether `doc` matches the MongoDB-style `filter`.
///
/// Supported top-level forms:
/// - `{"field": value}` — exact match
/// - `{"field": {"$eq": v, "$ne": v, "$gt": v, "$gte": v, "$lt": v, "$lte": v, "$in": [...]}}` — comparisons
/// - `{"$and": [...]}` / `{"$or": [...]}` — logical operators
fn matches_filter(doc: &Value, filter: &Value) -> bool {
    let obj = match filter.as_object() {
        Some(o) => o,
        None => return true,
    };
    if obj.is_empty() {
        return true;
    }
    for (key, cond) in obj {
        match key.as_str() {
            "$and" => {
                if let Some(arr) = cond.as_array() {
                    if !arr.iter().all(|f| matches_filter(doc, f)) {
                        return false;
                    }
                }
            }
            "$or" => {
                if let Some(arr) = cond.as_array() {
                    if !arr.iter().any(|f| matches_filter(doc, f)) {
                        return false;
                    }
                }
            }
            field => {
                let doc_val = doc_field(doc, field);
                if !matches_condition(doc_val.as_ref(), cond) {
                    return false;
                }
            }
        }
    }
    true
}

fn doc_field<'a>(doc: &'a Value, field: &str) -> Option<&'a Value> {
    // Support nested fields via dot notation: "address.city"
    let mut cur = doc;
    for part in field.split('.') {
        cur = cur.get(part)?;
    }
    Some(cur)
}

fn matches_condition(doc_val: Option<&Value>, cond: &Value) -> bool {
    if let Some(cond_obj) = cond.as_object() {
        // Operator form: {"$eq": ...}
        for (op, expected) in cond_obj {
            let ok = match op.as_str() {
                "$eq"  => doc_val.map_or(false, |v| v == expected),
                "$ne"  => doc_val.map_or(true,  |v| v != expected),
                "$gt"  => cmp_values(doc_val, expected) == Some(std::cmp::Ordering::Greater),
                "$gte" => matches!(cmp_values(doc_val, expected), Some(std::cmp::Ordering::Greater) | Some(std::cmp::Ordering::Equal)),
                "$lt"  => cmp_values(doc_val, expected) == Some(std::cmp::Ordering::Less),
                "$lte" => matches!(cmp_values(doc_val, expected), Some(std::cmp::Ordering::Less) | Some(std::cmp::Ordering::Equal)),
                "$in"  => {
                    if let Some(arr) = expected.as_array() {
                        doc_val.map_or(false, |v| arr.contains(v))
                    } else {
                        false
                    }
                }
                _ => false,
            };
            if !ok {
                return false;
            }
        }
        true
    } else {
        // Direct equality: {"field": value}
        doc_val.map_or(false, |v| v == cond)
    }
}

fn cmp_values(a: Option<&Value>, b: &Value) -> Option<std::cmp::Ordering> {
    match (a?, b) {
        (Value::Number(x), Value::Number(y)) => {
            let xf = x.as_f64()?;
            let yf = y.as_f64()?;
            xf.partial_cmp(&yf)
        }
        (Value::String(x), Value::String(y)) => Some(x.cmp(y)),
        _ => None,
    }
}
