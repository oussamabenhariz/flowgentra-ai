//! PGVector — PostgreSQL with the `pgvector` extension.
//!
//! Enabled with the `pgvector-store` Cargo feature.
//!
//! Stores documents in a PostgreSQL table with a `vector(N)` column and runs
//! cosine-similarity searches via the `<=>` operator. A `ivfflat` index is
//! created automatically so searches scale beyond a few thousand documents.
//!
//! **Supabase note:** Supabase is hosted PostgreSQL with pgvector enabled by
//! default. Point `endpoint` at your Supabase connection string and this
//! backend works without modification.
//!
//! # Example
//! ```rust,ignore
//! use flowgentra_ai::core::rag::{PgVectorConfig, PgVectorStore, VectorStoreBackend};
//! use flowgentra_ai::core::rag::{Document, FilterExpr};
//! use serde_json::json;
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let store = PgVectorStore::connect(PgVectorConfig {
//!         url: "postgres://user:pass@localhost/mydb".into(),
//!         table: "documents".into(),
//!         embedding_dim: 1536,
//!     }).await?;
//!
//!     let mut doc = Document::new("doc-1", "Hello world");
//!     doc.embedding = Some(vec![0.1_f32; 1536]);
//!     store.index(doc).await?;
//!
//!     let results = store.search(vec![0.1_f32; 1536], 5, None).await?;
//!     Ok(())
//! }
//! ```

use async_trait::async_trait;
use serde_json::Value;
use sqlx::postgres::PgPool;
use std::collections::HashMap;

use super::filter::FilterExpr;
use super::vector_db::{
    Document, MetadataFilter, SearchResult, VectorStoreBackend, VectorStoreError,
};

/// Configuration for [`PgVectorStore`].
#[derive(Debug, Clone)]
pub struct PgVectorConfig {
    /// PostgreSQL connection URL (`postgres://user:pass@host/db`)
    pub url: String,
    /// Table name to store documents in (created automatically if absent)
    pub table: String,
    /// Embedding dimension — must match your embedding model output
    pub embedding_dim: usize,
}

/// PostgreSQL + pgvector vector store.
pub struct PgVectorStore {
    pool: PgPool,
    table: String,
    embedding_dim: usize,
}

impl PgVectorStore {
    /// Connect to PostgreSQL and ensure the documents table and index exist.
    pub async fn connect(config: PgVectorConfig) -> Result<Self, VectorStoreError> {
        let pool = PgPool::connect(&config.url)
            .await
            .map_err(|e| VectorStoreError::ConnectionError(e.to_string()))?;

        let store = Self {
            pool,
            table: config.table,
            embedding_dim: config.embedding_dim,
        };
        store.ensure_table().await?;
        Ok(store)
    }

    async fn ensure_table(&self) -> Result<(), VectorStoreError> {
        let dim = self.embedding_dim as i32;
        let sql = format!(
            r#"
            CREATE EXTENSION IF NOT EXISTS vector;
            CREATE TABLE IF NOT EXISTS {table} (
                id       TEXT PRIMARY KEY,
                text     TEXT NOT NULL,
                embedding vector({dim}),
                metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb
            );
            CREATE INDEX IF NOT EXISTS {table}_embedding_idx
                ON {table} USING ivfflat (embedding vector_cosine_ops)
                WITH (lists = 100);
            "#,
            table = self.table,
            dim = dim,
        );
        sqlx::raw_sql(&sql)
            .execute(&self.pool)
            .await
            .map_err(|e| VectorStoreError::ConnectionError(e.to_string()))?;
        Ok(())
    }

    /// Convert a `FilterExpr` to a SQL WHERE clause fragment and bind values.
    fn filter_to_sql(f: &FilterExpr, params: &mut Vec<Value>, offset: usize) -> String {
        match f {
            FilterExpr::Eq(k, v) => {
                params.push(v.clone());
                format!("metadata->>'{}' = ${}", k, offset + params.len())
            }
            FilterExpr::Ne(k, v) => {
                params.push(v.clone());
                format!("metadata->>'{}' != ${}", k, offset + params.len())
            }
            FilterExpr::Gt(k, v) => {
                params.push(v.clone());
                format!(
                    "(metadata->>'{}')::numeric > (${})::numeric",
                    k,
                    offset + params.len()
                )
            }
            FilterExpr::Lt(k, v) => {
                params.push(v.clone());
                format!(
                    "(metadata->>'{}')::numeric < (${})::numeric",
                    k,
                    offset + params.len()
                )
            }
            FilterExpr::Gte(k, v) => {
                params.push(v.clone());
                format!(
                    "(metadata->>'{}')::numeric >= (${})::numeric",
                    k,
                    offset + params.len()
                )
            }
            FilterExpr::Lte(k, v) => {
                params.push(v.clone());
                format!(
                    "(metadata->>'{}')::numeric <= (${})::numeric",
                    k,
                    offset + params.len()
                )
            }
            FilterExpr::In(k, vs) => {
                let placeholders: Vec<String> = vs
                    .iter()
                    .enumerate()
                    .map(|(_i, v)| {
                        params.push(v.clone());
                        format!("${}", offset + params.len())
                    })
                    .collect();
                format!("metadata->>'{}' IN ({})", k, placeholders.join(", "))
            }
            FilterExpr::And(exprs) => {
                let parts: Vec<String> = exprs
                    .iter()
                    .map(|e| format!("({})", Self::filter_to_sql(e, params, offset)))
                    .collect();
                parts.join(" AND ")
            }
            FilterExpr::Or(exprs) => {
                let parts: Vec<String> = exprs
                    .iter()
                    .map(|e| format!("({})", Self::filter_to_sql(e, params, offset)))
                    .collect();
                parts.join(" OR ")
            }
        }
    }
}

#[async_trait]
impl VectorStoreBackend for PgVectorStore {
    async fn index(&self, doc: Document) -> Result<(), VectorStoreError> {
        let embedding = doc.embedding.ok_or_else(|| {
            VectorStoreError::EmbeddingError("Document must have an embedding".into())
        })?;
        let metadata = serde_json::to_value(&doc.metadata)
            .map_err(|e| VectorStoreError::SerializationError(e.to_string()))?;

        // pgvector stores vectors as '[0.1, 0.2, ...]' text format
        let vec_str = format!(
            "[{}]",
            embedding
                .iter()
                .map(|f| f.to_string())
                .collect::<Vec<_>>()
                .join(",")
        );

        let sql = format!(
            r#"INSERT INTO {table} (id, text, embedding, metadata)
               VALUES ($1, $2, $3::vector, $4)
               ON CONFLICT (id) DO UPDATE
               SET text = EXCLUDED.text,
                   embedding = EXCLUDED.embedding,
                   metadata = EXCLUDED.metadata"#,
            table = self.table
        );
        sqlx::query(&sql)
            .bind(&doc.id)
            .bind(&doc.text)
            .bind(&vec_str)
            .bind(&metadata)
            .execute(&self.pool)
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
        let vec_str = format!(
            "[{}]",
            query_embedding
                .iter()
                .map(|f| f.to_string())
                .collect::<Vec<_>>()
                .join(",")
        );

        let (where_clause, filter_params) = if let Some(f) = filter {
            let mut params = Vec::new();
            // $1 = query vector, $2 = top_k, filter params start at $3
            let clause = Self::filter_to_sql(&f, &mut params, 2);
            (format!("WHERE {}", clause), params)
        } else {
            (String::new(), Vec::new())
        };

        let sql = format!(
            r#"SELECT id, text, metadata,
                      1 - (embedding <=> $1::vector) AS score
               FROM {table}
               {where_clause}
               ORDER BY embedding <=> $1::vector
               LIMIT $2"#,
            table = self.table,
            where_clause = where_clause,
        );

        let mut q = sqlx::query(&sql).bind(&vec_str).bind(top_k as i64);
        for param in &filter_params {
            q = q.bind(param.as_str().unwrap_or(""));
        }

        let rows = q
            .fetch_all(&self.pool)
            .await
            .map_err(|e| VectorStoreError::QueryError(e.to_string()))?;

        let results = rows
            .iter()
            .filter_map(|row| {
                use sqlx::Row;
                let id: String = row.try_get("id").ok()?;
                let text: String = row.try_get("text").ok()?;
                let score: f64 = row.try_get("score").ok()?;
                let metadata: Value = row.try_get("metadata").ok()?;
                let meta_map: HashMap<String, Value> =
                    serde_json::from_value(metadata).unwrap_or_default();
                Some(SearchResult {
                    id,
                    text,
                    score: score as f32,
                    metadata: meta_map,
                })
            })
            .collect();
        Ok(results)
    }

    async fn delete(&self, doc_id: &str) -> Result<(), VectorStoreError> {
        let sql = format!("DELETE FROM {table} WHERE id = $1", table = self.table);
        sqlx::query(&sql)
            .bind(doc_id)
            .execute(&self.pool)
            .await
            .map_err(|e| VectorStoreError::ApiError(e.to_string()))?;
        Ok(())
    }

    async fn update(&self, doc: Document) -> Result<(), VectorStoreError> {
        self.index(doc).await
    }

    async fn get(&self, doc_id: &str) -> Result<Document, VectorStoreError> {
        let sql = format!(
            "SELECT id, text, metadata FROM {table} WHERE id = $1",
            table = self.table
        );
        let row = sqlx::query(&sql)
            .bind(doc_id)
            .fetch_optional(&self.pool)
            .await
            .map_err(|e| VectorStoreError::ApiError(e.to_string()))?
            .ok_or_else(|| VectorStoreError::NotFound(doc_id.to_string()))?;

        use sqlx::Row;
        let id: String = row
            .try_get("id")
            .map_err(|e| VectorStoreError::ApiError(e.to_string()))?;
        let text: String = row
            .try_get("text")
            .map_err(|e| VectorStoreError::ApiError(e.to_string()))?;
        let metadata: Value = row
            .try_get("metadata")
            .map_err(|e| VectorStoreError::ApiError(e.to_string()))?;
        let meta_map: HashMap<String, Value> = serde_json::from_value(metadata).unwrap_or_default();
        Ok(Document {
            id,
            text,
            embedding: None,
            metadata: meta_map,
        })
    }

    async fn list(&self) -> Result<Vec<Document>, VectorStoreError> {
        let sql = format!("SELECT id, text, metadata FROM {table}", table = self.table);
        let rows = sqlx::query(&sql)
            .fetch_all(&self.pool)
            .await
            .map_err(|e| VectorStoreError::ApiError(e.to_string()))?;

        let docs = rows
            .iter()
            .filter_map(|row| {
                use sqlx::Row;
                let id: String = row.try_get("id").ok()?;
                let text: String = row.try_get("text").ok()?;
                let metadata: Value = row.try_get("metadata").ok()?;
                let meta_map: HashMap<String, Value> =
                    serde_json::from_value(metadata).unwrap_or_default();
                Some(Document {
                    id,
                    text,
                    embedding: None,
                    metadata: meta_map,
                })
            })
            .collect();
        Ok(docs)
    }

    async fn clear(&self) -> Result<(), VectorStoreError> {
        let sql = format!("TRUNCATE TABLE {table}", table = self.table);
        sqlx::query(&sql)
            .execute(&self.pool)
            .await
            .map_err(|e| VectorStoreError::ApiError(e.to_string()))?;
        Ok(())
    }
}
