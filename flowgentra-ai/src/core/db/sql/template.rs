//! Template for adding a new SQL backend.
//!
//! Copy this file, rename `MyDatabase`, implement the two trait methods,
//! and add the backend behind a Cargo feature flag.
//!
//! Steps:
//! 1. Copy this file to e.g. `mysql.rs`
//! 2. Add `#[cfg(feature = "mysql")] pub mod mysql;` in `sql/mod.rs`
//! 3. Add the crate dependency in `Cargo.toml` under `[dependencies]` (optional)
//! 4. Add the feature in `[features]`
//! 5. Implement `query` and `execute` below
//!
//! Rules:
//! - Do NOT add backend-specific public methods to the struct; keep them private.
//! - Return [`DbError`] — never expose the backend's own error type through the trait.
//! - `params` are positional; translate `?` / `$N` placeholders as needed.

use async_trait::async_trait;
use serde_json::Value;

use super::{DbError, Row, SqlDatabase};

/// Replace with your actual connection-pool type.
pub struct MyDatabase {
    // pool: MyPool,
}

impl MyDatabase {
    /// Connect to the database and return a new instance.
    pub async fn connect(_url: &str) -> Result<Self, DbError> {
        // let pool = MyPool::connect(url)
        //     .await
        //     .map_err(|e| DbError::Connection(e.to_string()))?;
        // Ok(Self { pool })
        Err(DbError::Config("MyDatabase: not yet implemented".into()))
    }

    /// Convert a backend row to the shared `Row` type.
    #[allow(dead_code)]
    fn row_to_map(_row: &()) -> Row {
        // Iterate columns, extract values as serde_json::Value.
        std::collections::HashMap::new()
    }
}

#[async_trait]
impl SqlDatabase for MyDatabase {
    async fn query(&self, _sql: &str, _params: &[Value]) -> Result<Vec<Row>, DbError> {
        // let rows = sqlx::query(sql).fetch_all(&self.pool).await
        //     .map_err(|e| DbError::Query(e.to_string()))?;
        // Ok(rows.iter().map(Self::row_to_map).collect())
        Err(DbError::Config("MyDatabase: not yet implemented".into()))
    }

    async fn execute(&self, _sql: &str, _params: &[Value]) -> Result<u64, DbError> {
        // let r = sqlx::query(sql).execute(&self.pool).await
        //     .map_err(|e| DbError::Query(e.to_string()))?;
        // Ok(r.rows_affected())
        Err(DbError::Config("MyDatabase: not yet implemented".into()))
    }
}
