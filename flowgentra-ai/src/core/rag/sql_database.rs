//! SQL Database Wrapper & Agent Tools
//!
//! Mirrors LangChain's SQL agent toolkit. Provides:
//!
//! - [`SqlDatabaseWrapper`] — thin async wrapper over a SQL connection that
//!   exposes schema introspection and safe query execution.
//! - **SQL Agent Tools** — ready-to-use tool implementations for an LLM agent:
//!   - [`ListSQLDatabaseTool`] — lists usable table names
//!   - [`InfoSQLDatabaseTool`] — returns DDL + sample rows for requested tables
//!   - [`QuerySQLDatabaseTool`] — executes a SELECT and returns results as JSON
//!   - [`QuerySQLCheckerTool`] — validates / reformats a query before execution
//! - [`SqlDatabaseLoader`] — loads SQL table rows as [`LoadedDocument`]s for RAG.
//!
//! ## Example
//!
//! ```rust,ignore
//! use flowgentra_ai::core::rag::sql_database::{SqlDatabaseWrapper, QuerySQLDatabaseTool};
//!
//! let wrapper = SqlDatabaseWrapper::connect("sqlite::memory:").await?;
//! wrapper.execute("CREATE TABLE users (id INT, name TEXT)", &[]).await?;
//!
//! let tool = QuerySQLDatabaseTool::new(wrapper.clone());
//! let result = tool.run("SELECT * FROM users").await?;
//! println!("{result}");
//! ```

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use serde_json::{json, Value};

use crate::core::rag::document_loader::{FileType, LoadedDocument};

// ── Error ─────────────────────────────────────────────────────────────────────

#[derive(Debug, thiserror::Error)]
pub enum SqlDbError {
    #[error("Connection error: {0}")]
    Connection(String),
    #[error("Query error: {0}")]
    Query(String),
    #[error("Schema error: {0}")]
    Schema(String),
    #[error("Unsupported feature: {0}")]
    Unsupported(String),
}

pub type SqlResult<T> = std::result::Result<T, SqlDbError>;

// ── SqlBackend async trait ────────────────────────────────────────────────────

/// Low-level async backend. Implement this for each SQL engine.
#[async_trait]
pub trait SqlBackend: Send + Sync {
    /// Execute a non-SELECT statement. Returns number of rows affected.
    async fn execute(&self, sql: &str) -> SqlResult<u64>;

    /// Execute a SELECT and return rows as `Vec<HashMap<column, Value>>`.
    async fn query(&self, sql: &str) -> SqlResult<Vec<HashMap<String, Value>>>;

    /// Return all user table names.
    async fn table_names(&self) -> SqlResult<Vec<String>>;

    /// Return CREATE TABLE DDL for a table (or equivalent schema text).
    async fn table_ddl(&self, table: &str) -> SqlResult<String>;

    /// Return a few sample rows for a table as JSON.
    async fn sample_rows(&self, table: &str, n: usize) -> SqlResult<Vec<HashMap<String, Value>>> {
        self.query(&format!("SELECT * FROM {table} LIMIT {n}")).await
    }
}

// ── SqlDatabaseWrapper ────────────────────────────────────────────────────────

/// High-level wrapper adding schema helpers and safety guards on top of a
/// [`SqlBackend`].
#[derive(Clone)]
pub struct SqlDatabaseWrapper {
    backend: Arc<dyn SqlBackend>,
    /// Subset of tables visible to the LLM. Empty = all tables.
    include_tables: Vec<String>,
    /// Tables always excluded (e.g. internal migration tables).
    exclude_tables: Vec<String>,
    /// Rows returned by InfoSQLDatabaseTool.
    sample_rows_in_table_info: usize,
}

impl SqlDatabaseWrapper {
    pub fn new(backend: Arc<dyn SqlBackend>) -> Self {
        Self {
            backend,
            include_tables: vec![],
            exclude_tables: vec![],
            sample_rows_in_table_info: 3,
        }
    }

    pub fn with_include_tables(mut self, tables: Vec<String>) -> Self {
        self.include_tables = tables;
        self
    }

    pub fn with_exclude_tables(mut self, tables: Vec<String>) -> Self {
        self.exclude_tables = tables;
        self
    }

    pub fn with_sample_rows(mut self, n: usize) -> Self {
        self.sample_rows_in_table_info = n;
        self
    }

    /// Usable table names after applying include/exclude filters.
    pub async fn get_usable_table_names(&self) -> SqlResult<Vec<String>> {
        let all = self.backend.table_names().await?;
        let names: Vec<String> = all
            .into_iter()
            .filter(|t| {
                if !self.include_tables.is_empty() {
                    self.include_tables.contains(t)
                } else {
                    !self.exclude_tables.contains(t)
                }
            })
            .collect();
        Ok(names)
    }

    /// Return schema description (DDL + sample rows) for the given tables.
    pub async fn get_table_info(&self, tables: &[&str]) -> SqlResult<String> {
        let mut parts = Vec::new();
        for &table in tables {
            let ddl = self.backend.table_ddl(table).await?;
            let samples = self.backend.sample_rows(table, self.sample_rows_in_table_info).await?;
            let sample_str = if samples.is_empty() {
                "(no rows)".to_string()
            } else {
                let headers: Vec<_> = samples[0].keys().cloned().collect();
                let mut lines = vec![headers.join(" | ")];
                for row in &samples {
                    let vals: Vec<_> = headers
                        .iter()
                        .map(|h| {
                            row.get(h)
                                .map(|v| v.to_string())
                                .unwrap_or_default()
                        })
                        .collect();
                    lines.push(vals.join(" | "));
                }
                lines.join("\n")
            };
            parts.push(format!(
                "Table: {table}\n{ddl}\n\nSample rows:\n{sample_str}"
            ));
        }
        Ok(parts.join("\n\n---\n\n"))
    }

    /// Execute a SELECT and return JSON-formatted rows.
    pub async fn run(&self, sql: &str) -> SqlResult<String> {
        // Safety: only allow SELECT statements
        let trimmed = sql.trim().to_ascii_uppercase();
        if !trimmed.starts_with("SELECT") && !trimmed.starts_with("WITH") {
            return Err(SqlDbError::Query(
                "Only SELECT / WITH queries are allowed via run()".into(),
            ));
        }
        let rows = self.backend.query(sql).await?;
        Ok(serde_json::to_string_pretty(&rows).unwrap_or_default())
    }

    /// Execute a DDL / DML statement (for admin use, not exposed to LLM tools).
    pub async fn execute(&self, sql: &str) -> SqlResult<u64> {
        self.backend.execute(sql).await
    }
}

// ── SQL Agent Tools ───────────────────────────────────────────────────────────

/// Tool: list all usable table names.
pub struct ListSQLDatabaseTool {
    db: SqlDatabaseWrapper,
}

impl ListSQLDatabaseTool {
    pub fn new(db: SqlDatabaseWrapper) -> Self {
        Self { db }
    }

    pub fn name(&self) -> &str {
        "list_sql_database"
    }

    pub fn description(&self) -> &str {
        "List the names of all tables in the SQL database. Use this before any other SQL tool."
    }

    /// Run the tool. Input is ignored.
    pub async fn run(&self, _input: &str) -> SqlResult<String> {
        let tables = self.db.get_usable_table_names().await?;
        Ok(tables.join(", "))
    }
}

// ─────────────────────────────────────────────────────────────────────────────

/// Tool: get schema and sample rows for specified tables.
pub struct InfoSQLDatabaseTool {
    db: SqlDatabaseWrapper,
}

impl InfoSQLDatabaseTool {
    pub fn new(db: SqlDatabaseWrapper) -> Self {
        Self { db }
    }

    pub fn name(&self) -> &str {
        "info_sql_database"
    }

    pub fn description(&self) -> &str {
        "Get the schema (CREATE TABLE) and sample rows for the specified tables. \
         Input: comma-separated table names, e.g. 'users, orders'."
    }

    pub async fn run(&self, input: &str) -> SqlResult<String> {
        let tables: Vec<&str> = input.split(',').map(str::trim).collect();
        self.db.get_table_info(&tables).await
    }
}

// ─────────────────────────────────────────────────────────────────────────────

/// Tool: execute a SELECT query and return results.
pub struct QuerySQLDatabaseTool {
    db: SqlDatabaseWrapper,
}

impl QuerySQLDatabaseTool {
    pub fn new(db: SqlDatabaseWrapper) -> Self {
        Self { db }
    }

    pub fn name(&self) -> &str {
        "query_sql_database"
    }

    pub fn description(&self) -> &str {
        "Execute a SELECT SQL query and return the results as JSON. \
         Only SELECT / WITH queries are allowed. \
         Always check the query with the checker tool first if uncertain."
    }

    pub async fn run(&self, sql: &str) -> SqlResult<String> {
        self.db.run(sql).await
    }
}

// ─────────────────────────────────────────────────────────────────────────────

/// Tool: lightweight static SQL checker (catches obvious mistakes before execution).
///
/// For a full LLM-powered checker, pass the query to an LLM with a checker prompt.
/// This static version validates:
/// - Statement is SELECT / WITH (no DML/DDL)
/// - No semicolons mid-query (SQL injection guard)
/// - No comments that could hide injections
pub struct QuerySQLCheckerTool {
    db: SqlDatabaseWrapper,
}

impl QuerySQLCheckerTool {
    pub fn new(db: SqlDatabaseWrapper) -> Self {
        Self { db }
    }

    pub fn name(&self) -> &str {
        "query_sql_checker"
    }

    pub fn description(&self) -> &str {
        "Check a SQL query for correctness before running it. \
         Returns the (possibly corrected) query or an error message."
    }

    pub async fn run(&self, sql: &str) -> SqlResult<String> {
        let trimmed = sql.trim();
        let upper = trimmed.to_ascii_uppercase();

        // Must start with SELECT or WITH
        if !upper.starts_with("SELECT") && !upper.starts_with("WITH") {
            return Ok(format!(
                "ERROR: Only SELECT/WITH queries are allowed. Got: {trimmed}"
            ));
        }

        // Reject inline comments (-- or /*) — common injection vector
        if trimmed.contains("--") || trimmed.contains("/*") {
            return Ok("ERROR: Comments are not allowed in SQL queries.".to_string());
        }

        // Multiple statements guard (semicolon before end of string)
        let without_trailing = trimmed.trim_end_matches(';');
        if without_trailing.contains(';') {
            return Ok("ERROR: Multiple SQL statements are not allowed.".to_string());
        }

        // Basic validation passed; return normalised query
        Ok(format!("{trimmed};"))
    }
}

// ── SqlDatabaseLoader ─────────────────────────────────────────────────────────

/// Loads rows from a SQL table (or arbitrary SELECT) as [`LoadedDocument`]s.
///
/// Each row becomes one document; all column values are stored in `metadata`.
/// The `text_columns` option specifies which columns become the document text
/// (joined with " | "); if empty, all columns are concatenated.
pub struct SqlDatabaseLoader {
    db: SqlDatabaseWrapper,
    query: String,
    text_columns: Vec<String>,
}

impl SqlDatabaseLoader {
    /// Load all rows from a table.
    pub fn from_table(db: SqlDatabaseWrapper, table: &str) -> Self {
        Self {
            db,
            query: format!("SELECT * FROM {table}"),
            text_columns: vec![],
        }
    }

    /// Load rows using an arbitrary SELECT.
    pub fn from_query(db: SqlDatabaseWrapper, query: impl Into<String>) -> Self {
        Self {
            db,
            query: query.into(),
            text_columns: vec![],
        }
    }

    /// Columns whose values form the document text (joined with " | ").
    pub fn with_text_columns(mut self, cols: Vec<String>) -> Self {
        self.text_columns = cols;
        self
    }

    /// Execute the query and produce one [`LoadedDocument`] per row.
    pub async fn load(&self) -> SqlResult<Vec<LoadedDocument>> {
        let rows = self.db.backend.query(&self.query).await?;
        let mut docs = Vec::new();
        for (i, row) in rows.iter().enumerate() {
            let text = if self.text_columns.is_empty() {
                row.values()
                    .map(|v| v.to_string())
                    .collect::<Vec<_>>()
                    .join(" | ")
            } else {
                self.text_columns
                    .iter()
                    .filter_map(|col| row.get(col))
                    .map(|v| v.to_string())
                    .collect::<Vec<_>>()
                    .join(" | ")
            };

            let mut metadata: HashMap<String, Value> = row
                .iter()
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect();
            metadata.insert("row_index".to_string(), json!(i));
            metadata.insert("source_query".to_string(), json!(self.query));

            docs.push(LoadedDocument {
                id: format!("sql_row_{i}"),
                text,
                source: format!("sql_row_{i}"),
                file_type: FileType::PlainText,
                metadata,
            });
        }
        Ok(docs)
    }
}

// ── SQLite backend implementation ─────────────────────────────────────────────

#[cfg(feature = "sqlite")]
pub mod sqlite_backend {
    use super::*;
    use sqlx::SqlitePool;

    pub struct SqliteBackend {
        pool: SqlitePool,
    }

    impl SqliteBackend {
        pub async fn connect(url: &str) -> SqlResult<Self> {
            let pool = SqlitePool::connect(url)
                .await
                .map_err(|e| SqlDbError::Connection(e.to_string()))?;
            Ok(Self { pool })
        }
    }

    #[async_trait]
    impl SqlBackend for SqliteBackend {
        async fn execute(&self, sql: &str) -> SqlResult<u64> {
            let result = sqlx::query(sql)
                .execute(&self.pool)
                .await
                .map_err(|e| SqlDbError::Query(e.to_string()))?;
            Ok(result.rows_affected())
        }

        async fn query(&self, sql: &str) -> SqlResult<Vec<HashMap<String, Value>>> {
            use sqlx::Row;
            let rows = sqlx::query(sql)
                .fetch_all(&self.pool)
                .await
                .map_err(|e| SqlDbError::Query(e.to_string()))?;
            let mut result = Vec::new();
            for row in rows {
                let mut map = HashMap::new();
                for col in row.columns() {
                    let val: Value = row.try_get_raw(col.ordinal())
                        .map(|raw| {
                            if let Ok(s) = row.try_get::<String, _>(col.ordinal()) {
                                json!(s)
                            } else if let Ok(i) = row.try_get::<i64, _>(col.ordinal()) {
                                json!(i)
                            } else if let Ok(f) = row.try_get::<f64, _>(col.ordinal()) {
                                json!(f)
                            } else if let Ok(b) = row.try_get::<bool, _>(col.ordinal()) {
                                json!(b)
                            } else {
                                Value::Null
                            }
                        })
                        .unwrap_or(Value::Null);
                    map.insert(col.name().to_string(), val);
                }
                result.push(map);
            }
            Ok(result)
        }

        async fn table_names(&self) -> SqlResult<Vec<String>> {
            let rows: Vec<(String,)> = sqlx::query_as(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'
                 ORDER BY name",
            )
            .fetch_all(&self.pool)
            .await
            .map_err(|e| SqlDbError::Schema(e.to_string()))?;
            Ok(rows.into_iter().map(|(n,)| n).collect())
        }

        async fn table_ddl(&self, table: &str) -> SqlResult<String> {
            let row: Option<(String,)> = sqlx::query_as(
                "SELECT sql FROM sqlite_master WHERE type='table' AND name = ?",
            )
            .bind(table)
            .fetch_optional(&self.pool)
            .await
            .map_err(|e| SqlDbError::Schema(e.to_string()))?;
            Ok(row.map(|(s,)| s).unwrap_or_default())
        }
    }

    /// Convenience: create a `SqlDatabaseWrapper` backed by SQLite.
    pub async fn sqlite_wrapper(url: &str) -> SqlResult<SqlDatabaseWrapper> {
        let backend = SqliteBackend::connect(url).await?;
        Ok(SqlDatabaseWrapper::new(Arc::new(backend)))
    }
}

// ── PostgreSQL backend ────────────────────────────────────────────────────────

#[cfg(feature = "postgres")]
pub mod postgres_backend {
    use super::*;
    use sqlx::PgPool;

    pub struct PostgresBackend {
        pool: PgPool,
    }

    impl PostgresBackend {
        pub async fn connect(url: &str) -> SqlResult<Self> {
            let pool = PgPool::connect(url)
                .await
                .map_err(|e| SqlDbError::Connection(e.to_string()))?;
            Ok(Self { pool })
        }
    }

    #[async_trait]
    impl SqlBackend for PostgresBackend {
        async fn execute(&self, sql: &str) -> SqlResult<u64> {
            let result = sqlx::query(sql)
                .execute(&self.pool)
                .await
                .map_err(|e| SqlDbError::Query(e.to_string()))?;
            Ok(result.rows_affected())
        }

        async fn query(&self, sql: &str) -> SqlResult<Vec<HashMap<String, Value>>> {
            use sqlx::Row;
            let rows = sqlx::query(sql)
                .fetch_all(&self.pool)
                .await
                .map_err(|e| SqlDbError::Query(e.to_string()))?;
            let mut result = Vec::new();
            for row in rows {
                let mut map = HashMap::new();
                for col in row.columns() {
                    let val: Value = if let Ok(s) = row.try_get::<String, _>(col.ordinal()) {
                        json!(s)
                    } else if let Ok(i) = row.try_get::<i64, _>(col.ordinal()) {
                        json!(i)
                    } else if let Ok(f) = row.try_get::<f64, _>(col.ordinal()) {
                        json!(f)
                    } else if let Ok(b) = row.try_get::<bool, _>(col.ordinal()) {
                        json!(b)
                    } else {
                        Value::Null
                    };
                    map.insert(col.name().to_string(), val);
                }
                result.push(map);
            }
            Ok(result)
        }

        async fn table_names(&self) -> SqlResult<Vec<String>> {
            let rows: Vec<(String,)> = sqlx::query_as(
                "SELECT table_name FROM information_schema.tables
                 WHERE table_schema = 'public' ORDER BY table_name",
            )
            .fetch_all(&self.pool)
            .await
            .map_err(|e| SqlDbError::Schema(e.to_string()))?;
            Ok(rows.into_iter().map(|(n,)| n).collect())
        }

        async fn table_ddl(&self, table: &str) -> SqlResult<String> {
            // Reconstruct DDL from information_schema
            let cols: Vec<(String, String, String)> = sqlx::query_as(
                "SELECT column_name, data_type, is_nullable
                 FROM information_schema.columns
                 WHERE table_schema='public' AND table_name=$1
                 ORDER BY ordinal_position",
            )
            .bind(table)
            .fetch_all(&self.pool)
            .await
            .map_err(|e| SqlDbError::Schema(e.to_string()))?;
            let col_defs: Vec<String> = cols
                .into_iter()
                .map(|(name, dtype, nullable)| {
                    let null_str = if nullable == "NO" { " NOT NULL" } else { "" };
                    format!("  {name} {dtype}{null_str}")
                })
                .collect();
            Ok(format!("CREATE TABLE {table} (\n{}\n)", col_defs.join(",\n")))
        }
    }

    pub async fn postgres_wrapper(url: &str) -> SqlResult<SqlDatabaseWrapper> {
        let backend = PostgresBackend::connect(url).await?;
        Ok(SqlDatabaseWrapper::new(Arc::new(backend)))
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(all(test, feature = "sqlite"))]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_sqlite_wrapper_and_tools() {
        let wrapper =
            sqlite_backend::sqlite_wrapper("sqlite::memory:").await.unwrap();

        wrapper.execute("CREATE TABLE users (id INTEGER, name TEXT)").await.unwrap();
        wrapper.execute("INSERT INTO users VALUES (1, 'Alice')").await.unwrap();
        wrapper.execute("INSERT INTO users VALUES (2, 'Bob')").await.unwrap();

        let tables = wrapper.get_usable_table_names().await.unwrap();
        assert!(tables.contains(&"users".to_string()));

        let info = wrapper.get_table_info(&["users"]).await.unwrap();
        assert!(info.contains("users"));

        let result = wrapper.run("SELECT * FROM users").await.unwrap();
        assert!(result.contains("Alice"));
    }

    #[tokio::test]
    async fn test_sql_checker_rejects_dml() {
        let wrapper =
            sqlite_backend::sqlite_wrapper("sqlite::memory:").await.unwrap();
        let checker = QuerySQLCheckerTool::new(wrapper);
        let out = checker.run("DROP TABLE users").await.unwrap();
        assert!(out.starts_with("ERROR"));
    }

    #[tokio::test]
    async fn test_loader() {
        let wrapper =
            sqlite_backend::sqlite_wrapper("sqlite::memory:").await.unwrap();
        wrapper.execute("CREATE TABLE items (id INTEGER, label TEXT)").await.unwrap();
        wrapper.execute("INSERT INTO items VALUES (1, 'foo')").await.unwrap();
        let loader = SqlDatabaseLoader::from_table(wrapper, "items");
        let docs = loader.load().await.unwrap();
        assert_eq!(docs.len(), 1);
        assert!(docs[0].text.contains("foo"));
    }
}
