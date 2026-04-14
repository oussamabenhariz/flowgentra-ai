//! MySQL backend for [`SqlDatabase`].
//!
//! Enabled with the `mysql` Cargo feature.
//! Uses `sqlx` with the MySQL driver. Bind parameters use `?` placeholders.
//!
//! # Example
//! ```rust,ignore
//! use flowgentra_ai::core::db::sql::mysql::MySqlDatabase;
//! use flowgentra_ai::core::db::sql::SqlDatabase;
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let db = MySqlDatabase::connect("mysql://user:pass@localhost/mydb").await?;
//!     db.execute("CREATE TABLE IF NOT EXISTS t (id INT, name VARCHAR(255))", &[]).await?;
//!     db.execute("INSERT INTO t VALUES (?, ?)", &[serde_json::json!(1), serde_json::json!("Alice")]).await?;
//!     let rows = db.query("SELECT * FROM t WHERE id = ?", &[serde_json::json!(1)]).await?;
//!     println!("{rows:#?}");
//!     Ok(())
//! }
//! ```

use async_trait::async_trait;
use serde_json::Value;
use sqlx::mysql::{MySqlPool, MySqlRow};
use sqlx::{Column, Row as SqlxRow, TypeInfo};
use std::collections::HashMap;

use super::{DbError, Row, SqlDatabase};

/// MySQL backend — wraps a [`sqlx::MySqlPool`].
pub struct MySqlDatabase {
    pool: MySqlPool,
}

impl MySqlDatabase {
    /// Connect to a MySQL database.
    /// URL format: `mysql://user:password@host:port/database`
    pub async fn connect(url: &str) -> Result<Self, DbError> {
        let pool = MySqlPool::connect(url)
            .await
            .map_err(|e| DbError::Connection(e.to_string()))?;
        Ok(Self { pool })
    }

    fn row_to_map(row: &MySqlRow) -> Row {
        let mut map = HashMap::new();
        for (i, col) in row.columns().iter().enumerate() {
            let name = col.name().to_string();
            let type_name = col.type_info().name().to_uppercase();
            let value = if type_name.contains("INT") || type_name == "TINYINT(1)" {
                // TINYINT(1) is MySQL BOOL — check before INT
                if type_name == "TINYINT" || type_name == "TINYINT(1)" {
                    row.try_get::<bool, _>(i)
                        .map(Value::Bool)
                        .unwrap_or_else(|_| {
                            row.try_get::<i64, _>(i)
                                .map(|n| Value::Number(n.into()))
                                .unwrap_or(Value::Null)
                        })
                } else {
                    row.try_get::<i64, _>(i)
                        .map(|n| Value::Number(n.into()))
                        .unwrap_or(Value::Null)
                }
            } else if type_name.contains("FLOAT")
                || type_name.contains("DOUBLE")
                || type_name.contains("DECIMAL")
                || type_name.contains("NUMERIC")
            {
                row.try_get::<f64, _>(i)
                    .ok()
                    .and_then(|f| serde_json::Number::from_f64(f).map(Value::Number))
                    .unwrap_or(Value::Null)
            } else if type_name == "JSON" {
                row.try_get::<serde_json::Value, _>(i)
                    .unwrap_or(Value::Null)
            } else {
                // VARCHAR, TEXT, CHAR, DATE, DATETIME, TIMESTAMP, BLOB, ENUM, etc.
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
impl SqlDatabase for MySqlDatabase {
    async fn query(&self, sql: &str, params: &[Value]) -> Result<Vec<Row>, DbError> {
        let mut q = sqlx::query(sql);
        for p in params {
            q = bind_value(q, p);
        }
        let rows = q
            .fetch_all(&self.pool)
            .await
            .map_err(|e| DbError::Query(e.to_string()))?;
        Ok(rows.iter().map(Self::row_to_map).collect())
    }

    async fn execute(&self, sql: &str, params: &[Value]) -> Result<u64, DbError> {
        let mut q = sqlx::query(sql);
        for p in params {
            q = bind_value(q, p);
        }
        let result = q
            .execute(&self.pool)
            .await
            .map_err(|e| DbError::Query(e.to_string()))?;
        Ok(result.rows_affected())
    }
}

fn bind_value<'q>(
    query: sqlx::query::Query<'q, sqlx::MySql, sqlx::mysql::MySqlArguments>,
    value: &'q Value,
) -> sqlx::query::Query<'q, sqlx::MySql, sqlx::mysql::MySqlArguments> {
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
