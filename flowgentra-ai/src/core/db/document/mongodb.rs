//! MongoDB backend for [`DocumentStore`].
//!
//! Enabled with the `mongodb` Cargo feature.
//!
//! Wraps the official [`mongodb`] async driver. Documents are stored and
//! returned as [`serde_json::Value`] objects. The `filter` passed to `find()`
//! uses standard MongoDB query syntax (`{"field": value}`, `{"$gt": n}`, etc.).
//!
//! # Example
//! ```rust,ignore
//! use flowgentra_ai::core::db::document::DocumentStore;
//! use flowgentra_ai::core::db::document::mongodb::MongoDocumentStore;
//! use serde_json::json;
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let store = MongoDocumentStore::connect("mongodb://localhost:27017", "mydb").await?;
//!
//!     // Insert
//!     let id = store.insert("users", json!({"name": "Alice", "age": 30})).await?;
//!     println!("Inserted: {id}");
//!
//!     // Query
//!     let docs = store.find("users", json!({"name": "Alice"})).await?;
//!     println!("{docs:#?}");
//!
//!     // Delete
//!     store.delete("users", &id).await?;
//!     Ok(())
//! }
//! ```

use async_trait::async_trait;
use mongodb::{
    bson::{doc, oid::ObjectId, Bson, Document as BsonDocument},
    options::ClientOptions,
    Client,
};
use serde_json::Value;
use std::str::FromStr;

use super::{DbError, DocumentStore};

/// MongoDB backend — wraps a [`mongodb::Client`] and a chosen database name.
pub struct MongoDocumentStore {
    client: Client,
    database: String,
}

impl MongoDocumentStore {
    /// Connect to a MongoDB instance and select the target database.
    ///
    /// `uri` format: `mongodb://[user:pass@]host[:port]`
    pub async fn connect(uri: &str, database: &str) -> Result<Self, DbError> {
        let options = ClientOptions::parse(uri)
            .await
            .map_err(|e| DbError::Connection(e.to_string()))?;
        let client =
            Client::with_options(options).map_err(|e| DbError::Connection(e.to_string()))?;
        Ok(Self {
            client,
            database: database.to_string(),
        })
    }

    fn collection(&self, name: &str) -> mongodb::Collection<BsonDocument> {
        self.client.database(&self.database).collection(name)
    }

    /// Convert a [`serde_json::Value`] to a [`BsonDocument`].
    fn json_to_bson(v: Value) -> Result<BsonDocument, DbError> {
        mongodb::bson::to_document(&v).map_err(|e| DbError::Serialization(e.to_string()))
    }

    /// Convert a [`BsonDocument`] to a [`serde_json::Value`].
    fn bson_to_json(doc: BsonDocument) -> Value {
        // Convert Bson → serde_json::Value via the mongodb bson crate.
        bson_doc_to_json(doc)
    }
}

#[async_trait]
impl DocumentStore for MongoDocumentStore {
    async fn insert(&self, collection: &str, doc: Value) -> Result<String, DbError> {
        let bson_doc = Self::json_to_bson(doc)?;
        let result = self
            .collection(collection)
            .insert_one(bson_doc, None)
            .await
            .map_err(|e| DbError::Query(e.to_string()))?;

        // Return the inserted _id as a string.
        let id = match result.inserted_id {
            Bson::ObjectId(oid) => oid.to_hex(),
            other => other.to_string(),
        };
        Ok(id)
    }

    async fn find(&self, collection: &str, filter: Value) -> Result<Vec<Value>, DbError> {
        let filter_doc = Self::json_to_bson(filter)?;
        let mut cursor = self
            .collection(collection)
            .find(filter_doc, None)
            .await
            .map_err(|e| DbError::Query(e.to_string()))?;

        let mut results = Vec::new();
        use futures::TryStreamExt;
        use mongodb::error::Result as MongoResult;
        while let Some(doc) = cursor
            .try_next()
            .await
            .map_err(|e| DbError::Query(e.to_string()))?
        {
            results.push(Self::bson_to_json(doc));
        }
        Ok(results)
    }

    async fn delete(&self, collection: &str, id: &str) -> Result<(), DbError> {
        // Try to parse as ObjectId; fall back to string match on `_id`.
        let filter = if let Ok(oid) = ObjectId::from_str(id) {
            doc! { "_id": oid }
        } else {
            doc! { "_id": id }
        };

        self.collection(collection)
            .delete_one(filter, None)
            .await
            .map_err(|e| DbError::Query(e.to_string()))?;
        Ok(())
    }
}

// ─── BSON → JSON conversion ────────────────────────────────────────────────────

fn bson_to_json_value(b: Bson) -> Value {
    match b {
        Bson::Double(f) => serde_json::Number::from_f64(f)
            .map(Value::Number)
            .unwrap_or(Value::Null),
        Bson::String(s) => Value::String(s),
        Bson::Array(arr) => Value::Array(arr.into_iter().map(bson_to_json_value).collect()),
        Bson::Document(d) => bson_doc_to_json(d),
        Bson::Boolean(b) => Value::Bool(b),
        Bson::Null => Value::Null,
        Bson::Int32(n) => Value::Number((n as i64).into()),
        Bson::Int64(n) => Value::Number(n.into()),
        Bson::ObjectId(oid) => Value::String(oid.to_hex()),
        Bson::DateTime(dt) => Value::String(dt.to_string()),
        other => Value::String(other.to_string()),
    }
}

fn bson_doc_to_json(doc: BsonDocument) -> Value {
    let map: serde_json::Map<String, Value> = doc
        .into_iter()
        .map(|(k, v)| (k, bson_to_json_value(v)))
        .collect();
    Value::Object(map)
}
