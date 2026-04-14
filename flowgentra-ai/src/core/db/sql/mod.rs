//! SQL database trait and backend implementations.
//!
//! The [`SqlDatabase`] trait provides a minimal, backend-agnostic interface
//! for running SQL queries. Rows are returned as `Vec<HashMap<String, Value>>`
//! so callers never import sqlx types directly.
//!
//! Backends are feature-gated:
//! - `sqlite`          → [`sqlite::SqliteDatabase`]
//! - `postgres`        → [`postgres::PostgresDatabase`]
//! - `mysql`           → [`mysql::MySqlDatabase`]
//! - `mssql`           → [`mssql::MssqlDatabase`] (tiberius + bb8, not sqlx)
//! - `bigquery-sql`    → [`bigquery::BigQueryDatabase`]
//! - `databricks-sql`  → [`databricks::DatabricksDatabase`]

use async_trait::async_trait;
use serde_json::Value;
use std::collections::HashMap;

use super::DbError;

#[cfg(feature = "sqlite")]
pub mod sqlite;

#[cfg(feature = "postgres")]
pub mod postgres;

#[cfg(feature = "mysql")]
pub mod mysql;

#[cfg(feature = "mssql")]
pub mod mssql;

#[cfg(feature = "bigquery-sql")]
pub mod bigquery;

#[cfg(feature = "databricks-sql")]
pub mod databricks;

pub mod template;

/// A row returned from a SQL query: column name → JSON value.
pub type Row = HashMap<String, Value>;

/// Minimal async interface for SQL databases.
///
/// Implement this trait to add a new SQL backend (see `template.rs`).
/// All backends must be `Send + Sync` so they can be shared across async tasks.
#[async_trait]
pub trait SqlDatabase: Send + Sync {
    /// Run a `SELECT` (or any query that returns rows).
    ///
    /// `params` are positional bind parameters serialised as JSON values.
    /// Returns one [`Row`] per result row.
    async fn query(&self, sql: &str, params: &[Value]) -> Result<Vec<Row>, DbError>;

    /// Run an `INSERT`, `UPDATE`, `DELETE`, or DDL statement.
    ///
    /// Returns the number of rows affected (0 for DDL).
    async fn execute(&self, sql: &str, params: &[Value]) -> Result<u64, DbError>;
}
