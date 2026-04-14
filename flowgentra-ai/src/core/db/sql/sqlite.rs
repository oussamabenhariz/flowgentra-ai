//! SQLite backend for [`SqlDatabase`].
//!
//! Enabled with the `sqlite` Cargo feature.
//!
//! Uses [`sqlx`] with the `SqlitePool` connection pool.
//!
//! # Example
//! ```rust,ignore
//! use flowgentra_ai::core::db::sql::sqlite::SqliteDatabase;
//! use flowgentra_ai::core::db::sql::SqlDatabase;
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let db = SqliteDatabase::connect("sqlite::memory:").await?;
//!
//!     db.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT NOT NULL)", &[]).await?;
//!     db.execute("INSERT INTO users (id, name) VALUES (1, 'Alice')", &[]).await?;
//!
//!     let rows = db.query("SELECT * FROM users", &[]).await?;
//!     println!("{rows:#?}");
//!     Ok(())
//! }
//! ```

use async_trait::async_trait;
use serde_json::Value;
use sqlx::{Column, Row as SqlxRow, TypeInfo};
use sqlx::sqlite::{SqlitePool, SqliteRow};
use std::collections::HashMap;

use super::{DbError, Row, SqlDatabase};

/// SQLite backend — wraps a [`sqlx::SqlitePool`].
pub struct SqliteDatabase {
    pool: SqlitePool,
}

impl SqliteDatabase {
    /// Open (or create) a SQLite database at the given URL.
    ///
    /// Use `"sqlite::memory:"` for an in-process in-memory database.
    /// Use `"sqlite:///path/to/file.db"` for a file-backed database.
    pub async fn connect(url: &str) -> Result<Self, DbError> {
        let pool = SqlitePool::connect(url)
            .await
            .map_err(|e| DbError::Connection(e.to_string()))?;
        Ok(Self { pool })
    }

    /// Convert a [`SqliteRow`] to a column-name → JSON-value map.
    fn row_to_map(row: &SqliteRow) -> Row {
        let mut map = HashMap::new();
        for (i, col) in row.columns().iter().enumerate() {
            let name = col.name().to_string();
            let type_name = col.type_info().name().to_uppercase();
            let value = if type_name.contains("INT") {
                row.try_get::<i64, _>(i)
                    .map(|n| Value::Number(n.into()))
                    .unwrap_or(Value::Null)
            } else if type_name.contains("REAL")
                || type_name.contains("FLOAT")
                || type_name.contains("DOUBLE")
                || type_name.contains("NUMERIC")
            {
                row.try_get::<f64, _>(i)
                    .ok()
                    .and_then(|f| serde_json::Number::from_f64(f).map(Value::Number))
                    .unwrap_or(Value::Null)
            } else if type_name.contains("BOOL") {
                row.try_get::<bool, _>(i)
                    .map(Value::Bool)
                    .unwrap_or(Value::Null)
            } else if type_name == "NULL" {
                Value::Null
            } else {
                // TEXT, BLOB (as UTF-8 string), or unknown type — try String
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
impl SqlDatabase for SqliteDatabase {
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

/// Bind a single [`serde_json::Value`] parameter to a sqlx SQLite query.
fn bind_value<'q>(
    query: sqlx::query::Query<'q, sqlx::Sqlite, sqlx::sqlite::SqliteArguments<'q>>,
    value: &'q Value,
) -> sqlx::query::Query<'q, sqlx::Sqlite, sqlx::sqlite::SqliteArguments<'q>> {
    match value {
        Value::Null        => query.bind(Option::<String>::None),
        Value::Bool(b)     => query.bind(*b),
        Value::Number(n)   => {
            if let Some(i) = n.as_i64() {
                query.bind(i)
            } else if let Some(f) = n.as_f64() {
                query.bind(f)
            } else {
                query.bind(n.to_string())
            }
        }
        Value::String(s)   => query.bind(s.as_str()),
        other              => query.bind(other.to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_sqlite_basic_crud() {
        let db = SqliteDatabase::connect("sqlite::memory:").await.unwrap();

        db.execute(
            "CREATE TABLE t (id INTEGER PRIMARY KEY, name TEXT NOT NULL)",
            &[],
        )
        .await
        .unwrap();

        let affected = db
            .execute("INSERT INTO t (id, name) VALUES (1, 'Alice')", &[])
            .await
            .unwrap();
        assert_eq!(affected, 1);

        let rows = db.query("SELECT * FROM t", &[]).await.unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].get("name"), Some(&Value::String("Alice".into())));
    }

    #[tokio::test]
    async fn test_sqlite_parameterised() {
        let db = SqliteDatabase::connect("sqlite::memory:").await.unwrap();
        db.execute("CREATE TABLE t (id INTEGER, val TEXT)", &[])
            .await
            .unwrap();
        db.execute(
            "INSERT INTO t VALUES (?, ?)",
            &[Value::Number(42.into()), Value::String("hi".into())],
        )
        .await
        .unwrap();

        let rows = db
            .query("SELECT * FROM t WHERE id = ?", &[Value::Number(42.into())])
            .await
            .unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].get("val"), Some(&Value::String("hi".into())));
    }
}
