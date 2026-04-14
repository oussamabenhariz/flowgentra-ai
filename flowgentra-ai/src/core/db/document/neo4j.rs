//! Neo4j graph database backend for [`DocumentStore`].
//!
//! Uses the [`neo4rs`] crate (async Bolt driver).
//! Documents are stored as labelled nodes; `collection` maps to the node label.
//! The `id` property is used as the document identifier.
//!
//! Enabled with the `neo4j-store` Cargo feature.
//!
//! # Example
//! ```rust,ignore
//! use flowgentra_ai::core::db::document::neo4j::Neo4jDocumentStore;
//! use flowgentra_ai::core::db::document::DocumentStore;
//! use serde_json::json;
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let store = Neo4jDocumentStore::connect("bolt://localhost:7687", "neo4j", "password").await?;
//!     let id = store.insert("Person", json!({"name": "Alice", "age": 30})).await?;
//!     let docs = store.find("Person", json!({"name": "Alice"})).await?;
//!     store.delete("Person", &id).await?;
//!     Ok(())
//! }
//! ```

use async_trait::async_trait;
use neo4rs::{query, Graph};
use serde_json::{Map, Value};
use uuid::Uuid;

use super::super::DbError;
use super::DocumentStore;

/// Neo4j document store — stores JSON documents as graph nodes.
pub struct Neo4jDocumentStore {
    graph: Graph,
}

impl Neo4jDocumentStore {
    /// Connect to Neo4j via the Bolt protocol.
    ///
    /// - `uri`: e.g. `"bolt://localhost:7687"` or `"bolt+s://host:7687"` (TLS)
    /// - `user`: database user (default `"neo4j"`)
    /// - `password`: database password
    pub async fn connect(uri: &str, user: &str, password: &str) -> Result<Self, DbError> {
        let config = neo4rs::ConfigBuilder::default()
            .uri(uri)
            .user(user)
            .password(password)
            .build()
            .map_err(|e| DbError::Config(e.to_string()))?;
        let graph = Graph::connect(config).await
            .map_err(|e| DbError::Connection(e.to_string()))?;
        Ok(Self { graph })
    }
}

#[async_trait]
impl DocumentStore for Neo4jDocumentStore {
    /// Insert a JSON document as a node labelled `collection`.
    ///
    /// All top-level JSON fields are stored as node properties.
    /// An `id` property is auto-generated (UUID v4) if the document lacks one.
    async fn insert(&self, collection: &str, doc: Value) -> Result<String, DbError> {
        let mut obj = match doc {
            Value::Object(m) => m,
            _ => {
                let mut m = Map::new();
                m.insert("data".into(), doc);
                m
            }
        };

        let id = obj.get("id")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
            .unwrap_or_else(|| Uuid::new_v4().to_string());
        obj.insert("id".into(), Value::String(id.clone()));

        // Build a SET clause from all properties.
        let props: Vec<String> = obj
            .iter()
            .map(|(k, v)| {
                let val = json_val_to_cypher(v);
                format!("n.`{}` = {}", k.replace('`', "\\`"), val)
            })
            .collect();
        let set_clause = props.join(", ");

        let cypher = format!(
            "MERGE (n:`{}` {{id: $id}}) SET {}",
            collection.replace('`', "\\`"),
            set_clause
        );

        self.graph
            .run(query(&cypher).param("id", id.clone()))
            .await
            .map_err(|e| DbError::Query(e.to_string()))?;

        Ok(id)
    }

    /// Find nodes with label `collection` matching all top-level `filter` fields.
    ///
    /// `filter` uses simple equality matching: `{"field": value, ...}`.
    /// All matched nodes are returned as JSON objects.
    async fn find(&self, collection: &str, filter: Value) -> Result<Vec<Value>, DbError> {
        let conditions: Vec<String> = if let Some(obj) = filter.as_object() {
            obj.iter()
                .map(|(k, v)| {
                    format!(
                        "n.`{}` = {}",
                        k.replace('`', "\\`"),
                        json_val_to_cypher(v)
                    )
                })
                .collect()
        } else {
            vec![]
        };

        let where_clause = if conditions.is_empty() {
            String::new()
        } else {
            format!("WHERE {}", conditions.join(" AND "))
        };

        let cypher = format!(
            "MATCH (n:`{}`) {} RETURN n",
            collection.replace('`', "\\`"),
            where_clause
        );

        let mut result = self.graph
            .execute(query(&cypher))
            .await
            .map_err(|e| DbError::Query(e.to_string()))?;

        let mut docs = Vec::new();
        while let Ok(Some(row)) = result.next().await {
            if let Ok(node) = row.get::<neo4rs::Node>("n") {
                let mut map = Map::new();
                for key in node.keys() {
                    if let Ok(v) = node.get::<String>(key) {
                        map.insert(key.to_string(), Value::String(v));
                    } else if let Ok(v) = node.get::<i64>(key) {
                        map.insert(key.to_string(), Value::Number(v.into()));
                    } else if let Ok(v) = node.get::<f64>(key) {
                        if let Some(n) = serde_json::Number::from_f64(v) {
                            map.insert(key.to_string(), Value::Number(n));
                        }
                    } else if let Ok(v) = node.get::<bool>(key) {
                        map.insert(key.to_string(), Value::Bool(v));
                    }
                }
                docs.push(Value::Object(map));
            }
        }
        Ok(docs)
    }

    /// Delete the node with `id` property equal to `id` in label `collection`.
    async fn delete(&self, collection: &str, id: &str) -> Result<(), DbError> {
        let cypher = format!(
            "MATCH (n:`{}` {{id: $id}}) DETACH DELETE n",
            collection.replace('`', "\\`")
        );
        self.graph
            .run(query(&cypher).param("id", id))
            .await
            .map_err(|e| DbError::Query(e.to_string()))?;
        Ok(())
    }
}

// ── Helpers ──────────────────────────────────────────────────────────────────

/// Convert a `serde_json::Value` to a Cypher literal string.
fn json_val_to_cypher(v: &Value) -> String {
    match v {
        Value::Null        => "null".to_string(),
        Value::Bool(b)     => b.to_string(),
        Value::Number(n)   => n.to_string(),
        Value::String(s)   => format!("'{}'", s.replace('\'', "\\'")),
        Value::Array(_)    => "null".to_string(), // nested arrays not supported inline
        Value::Object(_)   => "null".to_string(), // nested objects not supported inline
    }
}
