//! MongoDB Atlas Vector Search — uses the `$vectorSearch` aggregation operator.
//!
//! Enabled with the `mongodb-store` Cargo feature (reuses the same `mongodb` dep).
//! Requires a MongoDB Atlas cluster (M10+) with a vector search index created
//! on the `embedding` field.
//!
//! **Index setup** (create once via Atlas UI or CLI):
//! ```json
//! {
//!   "fields": [{
//!     "type": "vector",
//!     "path": "embedding",
//!     "numDimensions": 1536,
//!     "similarity": "cosine"
//!   }]
//! }
//! ```
//!
//! # Example
//! ```rust,ignore
//! use flowgentra_ai::core::rag::{MongoAtlasConfig, MongoAtlasVectorStore, VectorStoreBackend};
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let store = MongoAtlasVectorStore::connect(MongoAtlasConfig {
//!         uri:        "mongodb+srv://user:pass@cluster.mongodb.net".into(),
//!         database:   "mydb".into(),
//!         collection: "documents".into(),
//!         index_name: "vector_index".into(),
//!         embedding_dim: 1536,
//!     }).await?;
//!     Ok(())
//! }
//! ```

use async_trait::async_trait;
use mongodb::{
    bson::{doc, Bson, Document as BsonDoc},
    options::ClientOptions,
    Client,
};
use serde_json::Value;
use std::collections::HashMap;

use super::filter::FilterExpr;
use super::vector_db::{Document, MetadataFilter, SearchResult, VectorStoreBackend, VectorStoreError};

/// Configuration for [`MongoAtlasVectorStore`].
#[derive(Debug, Clone)]
pub struct MongoAtlasConfig {
    /// MongoDB Atlas connection URI
    pub uri: String,
    /// Database name
    pub database: String,
    /// Collection name
    pub collection: String,
    /// Name of the Atlas vector search index (created in Atlas UI)
    pub index_name: String,
    /// Embedding dimension
    pub embedding_dim: usize,
}

/// MongoDB Atlas Vector Search backend.
pub struct MongoAtlasVectorStore {
    col: mongodb::Collection<BsonDoc>,
    index_name: String,
}

impl MongoAtlasVectorStore {
    pub async fn connect(config: MongoAtlasConfig) -> Result<Self, VectorStoreError> {
        let opts = ClientOptions::parse(&config.uri)
            .await
            .map_err(|e| VectorStoreError::ConnectionError(e.to_string()))?;
        let client = Client::with_options(opts)
            .map_err(|e| VectorStoreError::ConnectionError(e.to_string()))?;
        let col = client
            .database(&config.database)
            .collection::<BsonDoc>(&config.collection);
        Ok(Self { col, index_name: config.index_name })
    }

    fn json_to_bson(v: Value) -> Result<BsonDoc, VectorStoreError> {
        mongodb::bson::to_document(&v)
            .map_err(|e| VectorStoreError::SerializationError(e.to_string()))
    }

    fn bson_to_json(doc: BsonDoc) -> Value {
        let map: serde_json::Map<String, Value> = doc
            .into_iter()
            .map(|(k, v)| (k, bson_val_to_json(v)))
            .collect();
        Value::Object(map)
    }

    /// Convert a `FilterExpr` to a MongoDB BSON filter document.
    fn filter_to_bson(f: &FilterExpr) -> mongodb::bson::Document {
        match f {
            FilterExpr::Eq(k, v)  => doc! { format!("metadata.{}", k): { "$eq":  bson_from_json(v) } },
            FilterExpr::Ne(k, v)  => doc! { format!("metadata.{}", k): { "$ne":  bson_from_json(v) } },
            FilterExpr::Gt(k, v)  => doc! { format!("metadata.{}", k): { "$gt":  bson_from_json(v) } },
            FilterExpr::Lt(k, v)  => doc! { format!("metadata.{}", k): { "$lt":  bson_from_json(v) } },
            FilterExpr::Gte(k, v) => doc! { format!("metadata.{}", k): { "$gte": bson_from_json(v) } },
            FilterExpr::Lte(k, v) => doc! { format!("metadata.{}", k): { "$lte": bson_from_json(v) } },
            FilterExpr::In(k, vs) => {
                let arr: Vec<Bson> = vs.iter().map(bson_from_json).collect();
                doc! { format!("metadata.{}", k): { "$in": arr } }
            }
            FilterExpr::And(exprs) => {
                let parts: Vec<Bson> = exprs.iter()
                    .map(|e| Bson::Document(Self::filter_to_bson(e)))
                    .collect();
                doc! { "$and": parts }
            }
            FilterExpr::Or(exprs) => {
                let parts: Vec<Bson> = exprs.iter()
                    .map(|e| Bson::Document(Self::filter_to_bson(e)))
                    .collect();
                doc! { "$or": parts }
            }
        }
    }
}

#[async_trait]
impl VectorStoreBackend for MongoAtlasVectorStore {
    async fn index(&self, doc: Document) -> Result<(), VectorStoreError> {
        let embedding = doc.embedding.ok_or_else(|| {
            VectorStoreError::EmbeddingError("Document must have an embedding".into())
        })?;
        let bson_vec: Vec<Bson> = embedding.iter().map(|&f| Bson::Double(f as f64)).collect();
        let metadata_bson = mongodb::bson::to_document(
            &serde_json::to_value(&doc.metadata).unwrap_or(Value::Object(Default::default()))
        ).unwrap_or_default();

        let filter   = doc! { "_id": &doc.id };
        let update   = doc! {
            "$set": {
                "_id":       &doc.id,
                "text":      &doc.text,
                "embedding": bson_vec,
                "metadata":  metadata_bson,
            }
        };
        let opts = mongodb::options::UpdateOptions::builder().upsert(true).build();
        self.col.update_one(filter, update, opts)
            .await
            .map_err(|e| VectorStoreError::ApiError(e.to_string()))?;
        Ok(())
    }

    async fn search(
        &self,
        query_embedding: Vec<f32>,
        top_k: usize,
        filter: Option<MetadataFilter>,
    ) -> Result<Vec<SearchResult>, VectorStoreError> {
        let bson_vec: Vec<Bson> = query_embedding.iter().map(|&f| Bson::Double(f as f64)).collect();

        let mut vs_stage = doc! {
            "index":         &self.index_name,
            "path":          "embedding",
            "queryVector":   bson_vec,
            "numCandidates": (top_k * 10) as i64,
            "limit":         top_k as i64,
        };
        if let Some(f) = filter {
            vs_stage.insert("filter", Bson::Document(MongoAtlasVectorStore::filter_to_bson(&f)));
        }

        let pipeline = vec![
            doc! { "$vectorSearch": vs_stage },
            doc! { "$project": { "text": 1, "metadata": 1, "score": { "$meta": "vectorSearchScore" } } },
        ];

        use futures::TryStreamExt;
        let mut cursor = self.col.aggregate(pipeline, None)
            .await
            .map_err(|e| VectorStoreError::QueryError(e.to_string()))?;

        let mut results = Vec::new();
        while let Some(doc) = cursor.try_next().await
            .map_err(|e| VectorStoreError::QueryError(e.to_string()))?
        {
            let id    = doc.get_str("_id").unwrap_or("").to_string();
            let text  = doc.get_str("text").unwrap_or("").to_string();
            let score = doc.get_f64("score").unwrap_or(0.0) as f32;
            let meta_doc = doc.get_document("metadata").cloned().unwrap_or_default();
            let metadata = bson_doc_to_map(meta_doc);
            results.push(SearchResult { id, text, score, metadata });
        }
        Ok(results)
    }

    async fn delete(&self, doc_id: &str) -> Result<(), VectorStoreError> {
        self.col.delete_one(doc! { "_id": doc_id }, None)
            .await
            .map_err(|e| VectorStoreError::ApiError(e.to_string()))?;
        Ok(())
    }

    async fn update(&self, doc: Document) -> Result<(), VectorStoreError> {
        self.index(doc).await
    }

    async fn get(&self, doc_id: &str) -> Result<Document, VectorStoreError> {
        let bson_doc = self.col
            .find_one(doc! { "_id": doc_id }, None)
            .await
            .map_err(|e| VectorStoreError::ApiError(e.to_string()))?
            .ok_or_else(|| VectorStoreError::NotFound(doc_id.to_string()))?;
        let text = bson_doc.get_str("text").unwrap_or("").to_string();
        let metadata = bson_doc.get_document("metadata").cloned()
            .map(bson_doc_to_map).unwrap_or_default();
        Ok(Document { id: doc_id.to_string(), text, embedding: None, metadata })
    }

    async fn list(&self) -> Result<Vec<Document>, VectorStoreError> {
        use futures::TryStreamExt;
        let mut cursor = self.col.find(None, None).await
            .map_err(|e| VectorStoreError::ApiError(e.to_string()))?;
        let mut docs = Vec::new();
        while let Some(d) = cursor.try_next().await
            .map_err(|e| VectorStoreError::ApiError(e.to_string()))?
        {
            let id   = d.get_str("_id").unwrap_or("").to_string();
            let text = d.get_str("text").unwrap_or("").to_string();
            let metadata = d.get_document("metadata").cloned().map(bson_doc_to_map).unwrap_or_default();
            docs.push(Document { id, text, embedding: None, metadata });
        }
        Ok(docs)
    }

    async fn clear(&self) -> Result<(), VectorStoreError> {
        self.col.delete_many(doc! {}, None).await
            .map_err(|e| VectorStoreError::ApiError(e.to_string()))?;
        Ok(())
    }
}

// ── BSON helpers ────────────────────────────────────────────────────────────

fn bson_from_json(v: &Value) -> Bson {
    match v {
        Value::Null        => Bson::Null,
        Value::Bool(b)     => Bson::Boolean(*b),
        Value::Number(n)   => n.as_f64().map(Bson::Double).unwrap_or(Bson::Null),
        Value::String(s)   => Bson::String(s.clone()),
        Value::Array(arr)  => Bson::Array(arr.iter().map(bson_from_json).collect()),
        Value::Object(map) => {
            let doc: BsonDoc = map.iter().map(|(k, v)| (k.clone(), bson_from_json(v))).collect();
            Bson::Document(doc)
        }
    }
}

fn bson_val_to_json(b: Bson) -> Value {
    match b {
        Bson::Double(f)    => serde_json::Number::from_f64(f).map(Value::Number).unwrap_or(Value::Null),
        Bson::String(s)    => Value::String(s),
        Bson::Boolean(b)   => Value::Bool(b),
        Bson::Null         => Value::Null,
        Bson::Int32(n)     => Value::Number((n as i64).into()),
        Bson::Int64(n)     => Value::Number(n.into()),
        Bson::ObjectId(o)  => Value::String(o.to_hex()),
        Bson::Document(d)  => bson_doc_to_json(d),
        Bson::Array(arr)   => Value::Array(arr.into_iter().map(bson_val_to_json).collect()),
        other              => Value::String(other.to_string()),
    }
}

fn bson_doc_to_json(d: BsonDoc) -> Value {
    let map: serde_json::Map<String, Value> = d.into_iter().map(|(k, v)| (k, bson_val_to_json(v))).collect();
    Value::Object(map)
}

fn bson_doc_to_map(d: BsonDoc) -> HashMap<String, Value> {
    d.into_iter().map(|(k, v)| (k, bson_val_to_json(v))).collect()
}
