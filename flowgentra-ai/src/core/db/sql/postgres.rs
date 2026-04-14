//! PostgreSQL backend for [`SqlDatabase`].
//!
//! Enabled with the `postgres` Cargo feature.
//!
//! Uses [`sqlx`] with `PgPool`. Supports parameterised queries with `$1`, `$2` … placeholders.
//!
//! # Example
//! ```rust,ignore
//! use flowgentra_ai::core::db::sql::postgres::PostgresDatabase;
//! use flowgentra_ai::core::db::sql::SqlDatabase;
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let db = PostgresDatabase::connect("postgres://user:pass@localhost/mydb").await?;
//!
//!     db.execute(
//!         "CREATE TABLE IF NOT EXISTS users (id SERIAL PRIMARY KEY, name TEXT NOT NULL)",
//!         &[],
//!     ).await?;
//!
//!     db.execute("INSERT INTO users (name) VALUES ($1)", &[serde_json::json!("Alice")]).await?;
//!
//!     let rows = db.query("SELECT * FROM users", &[]).await?;
//!     println!("{rows:#?}");
//!     Ok(())
//! }
//! ```

use async_trait::async_trait;
use serde_json::Value;
use sqlx::postgres::{PgPool, PgRow};
use sqlx::{Column, Row as SqlxRow, TypeInfo};
use std::collections::HashMap;

use super::{DbError, Row, SqlDatabase};

/// PostgreSQL backend — wraps a [`sqlx::PgPool`].
pub struct PostgresDatabase {
    pool: PgPool,
}

impl PostgresDatabase {
    /// Connect to a PostgreSQL database using a connection URL.
    ///
    /// Format: `postgres://user:password@host:port/database`
    pub async fn connect(url: &str) -> Result<Self, DbError> {
        let pool = PgPool::connect(url)
            .await
            .map_err(|e| DbError::Connection(e.to_string()))?;
        Ok(Self { pool })
    }

    /// Convert a [`PgRow`] to a column-name → JSON-value map.
    fn row_to_map(row: &PgRow) -> Row {
        let mut map = HashMap::new();
        for (i, col) in row.columns().iter().enumerate() {
            let name = col.name().to_string();
            let type_name = col.type_info().name().to_uppercase();
            let value = if type_name.starts_with("INT")
                || type_name == "BIGINT"
                || type_name == "SMALLINT"
                || type_name == "SERIAL"
                || type_name == "BIGSERIAL"
            {
                row.try_get::<i64, _>(i)
                    .map(|n| Value::Number(n.into()))
                    .unwrap_or(Value::Null)
            } else if type_name == "FLOAT4"
                || type_name == "FLOAT8"
                || type_name == "REAL"
                || type_name == "DOUBLE PRECISION"
                || type_name.starts_with("NUMERIC")
                || type_name.starts_with("DECIMAL")
            {
                row.try_get::<f64, _>(i)
                    .ok()
                    .and_then(|f| serde_json::Number::from_f64(f).map(Value::Number))
                    .unwrap_or(Value::Null)
            } else if type_name == "BOOL" || type_name == "BOOLEAN" {
                row.try_get::<bool, _>(i)
                    .map(Value::Bool)
                    .unwrap_or(Value::Null)
            } else if type_name == "JSON" || type_name == "JSONB" {
                row.try_get::<serde_json::Value, _>(i)
                    .unwrap_or(Value::Null)
            } else {
                // TEXT, VARCHAR, CHAR, UUID, TIMESTAMP, etc. → String
                row.try_get::<String, _>(i)
                    .map(Value::String)
                    .unwrap_or(Value::Null)
            };
            map.insert(name, value);
        }
        map
    }
}

#[async_trait]
impl SqlDatabase for PostgresDatabase {
    async fn query(&self, sql: &str, params: &[Value]) -> Result<Vec<Row>, DbError> {
        let mut q = sqlx::query(sql);
        for param in params {
            q = bind_value(q, param);
        }
        let rows = q
            .fetch_all(&self.pool)
            .await
            .map_err(|e| DbError::Query(e.to_string()))?;
        Ok(rows.iter().map(Self::row_to_map).collect())
    }

    async fn execute(&self, sql: &str, params: &[Value]) -> Result<u64, DbError> {
        let mut q = sqlx::query(sql);
        for param in params {
            q = bind_value(q, param);
        }
        let result = q
            .execute(&self.pool)
            .await
            .map_err(|e| DbError::Query(e.to_string()))?;
        Ok(result.rows_affected())
    }
}

/// Bind a single [`serde_json::Value`] parameter to a sqlx PostgreSQL query.
fn bind_value<'q>(
    query: sqlx::query::Query<'q, sqlx::Postgres, sqlx::postgres::PgArguments>,
    value: &'q Value,
) -> sqlx::query::Query<'q, sqlx::Postgres, sqlx::postgres::PgArguments> {
    match value {
        Value::Null => query.bind(Option::<String>::None),
        Value::Bool(b) => query.bind(*b),
        Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                query.bind(i)
            } else if let Some(f) = n.as_f64() {
                query.bind(f)
            } else {
                query.bind(n.to_string())
            }
        }
        Value::String(s) => query.bind(s.as_str()),
        other => query.bind(other.to_string()),
    }
}
