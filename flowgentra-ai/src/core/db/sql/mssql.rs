//! Microsoft SQL Server backend for [`SqlDatabase`].
//!
//! Enabled with the `mssql` Cargo feature.
//!
//! Uses the [`tiberius`] async driver (not sqlx — sqlx 0.7 dropped MSSQL support)
//! with a [`bb8`] connection pool for concurrent access.
//!
//! Use `@p1`, `@p2`, ... positional bind placeholders in your SQL statements.
//!
//! # Connection URL format
//!
//! ```text
//! mssql://user:password@host[:port][/database][?option=value]
//! ```
//!
//! Supported query parameters:
//! - `trust_server_certificate=true` — skip TLS certificate validation (useful for dev/local)
//! - `encrypt=false` — disable TLS entirely (not recommended for production)
//!
//! # Example
//! ```rust,ignore
//! use flowgentra_ai::core::db::sql::mssql::MssqlDatabase;
//! use flowgentra_ai::core::db::sql::SqlDatabase;
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let db = MssqlDatabase::connect(
//!         "mssql://sa:Password1@localhost:1433/mydb?trust_server_certificate=true"
//!     ).await?;
//!
//!     db.execute(
//!         "INSERT INTO users (name) VALUES (@p1)",
//!         &[serde_json::json!("Alice")]
//!     ).await?;
//!
//!     let rows = db.query(
//!         "SELECT * FROM users WHERE name = @p1",
//!         &[serde_json::json!("Alice")]
//!     ).await?;
//!     println!("{rows:#?}");
//!     Ok(())
//! }
//! ```

use async_trait::async_trait;
use bb8::Pool;
use bb8_tiberius::ConnectionManager;
use serde_json::Value;
use std::collections::HashMap;
use tiberius::{ColumnType, Config, EncryptionLevel, Query};

use super::{DbError, Row, SqlDatabase};

/// Microsoft SQL Server backend — wraps a [`bb8`] pool of [`tiberius`] connections.
pub struct MssqlDatabase {
    pool: Pool<ConnectionManager>,
}

impl MssqlDatabase {
    /// Connect to SQL Server.
    ///
    /// `url` format: `mssql://user:password@host[:port][/database][?trust_server_certificate=true]`
    pub async fn connect(url: &str) -> Result<Self, DbError> {
        let config = parse_url(url)?;
        let mgr = ConnectionManager::new(config);
        let pool = Pool::builder()
            .max_size(10)
            .build(mgr)
            .await
            .map_err(|e| DbError::Connection(e.to_string()))?;
        Ok(Self { pool })
    }
}

#[async_trait]
impl SqlDatabase for MssqlDatabase {
    async fn query(&self, sql: &str, params: &[Value]) -> Result<Vec<Row>, DbError> {
        let mut conn = self
            .pool
            .get()
            .await
            .map_err(|e| DbError::Connection(e.to_string()))?;

        let mut query = Query::new(sql);
        bind_params(&mut query, params);

        let results = query
            .query(&mut *conn)
            .await
            .map_err(|e| DbError::Query(e.to_string()))?
            .into_results()
            .await
            .map_err(|e| DbError::Query(e.to_string()))?;

        let mut rows = Vec::new();
        for row_set in results {
            for row in row_set {
                rows.push(row_to_map(row));
            }
        }
        Ok(rows)
    }

    async fn execute(&self, sql: &str, params: &[Value]) -> Result<u64, DbError> {
        let mut conn = self
            .pool
            .get()
            .await
            .map_err(|e| DbError::Connection(e.to_string()))?;

        let mut query = Query::new(sql);
        bind_params(&mut query, params);

        let result = query
            .execute(&mut *conn)
            .await
            .map_err(|e| DbError::Query(e.to_string()))?;

        Ok(result.rows_affected().iter().sum())
    }
}

// ── URL parsing ───────────────────────────────────────────────────────────────

/// Parse `mssql://user:pass@host:port/database?opts` into a tiberius [`Config`].
fn parse_url(url: &str) -> Result<Config, DbError> {
    let rest = url
        .strip_prefix("mssql://")
        .or_else(|| url.strip_prefix("sqlserver://"))
        .ok_or_else(|| {
            DbError::Connection("MSSQL URL must start with mssql:// or sqlserver://".into())
        })?;

    let mut config = Config::new();

    // Split off query string
    let (rest, query_str) = match rest.find('?') {
        Some(pos) => (&rest[..pos], Some(&rest[pos + 1..])),
        None => (rest, None),
    };

    // Split off path (database name)
    let (authority, database) = match rest.find('/') {
        Some(pos) => (&rest[..pos], Some(&rest[pos + 1..])),
        None => (rest, None),
    };

    if let Some(db) = database.filter(|s| !s.is_empty()) {
        config.database(db);
    }

    // Split userinfo from host:port
    let (userinfo, hostport) = match authority.rfind('@') {
        Some(pos) => (Some(&authority[..pos]), &authority[pos + 1..]),
        None => (None, authority),
    };

    if let Some(ui) = userinfo {
        let (user, pass) = match ui.find(':') {
            Some(pos) => (&ui[..pos], &ui[pos + 1..]),
            None => (ui, ""),
        };
        config.authentication(tiberius::AuthMethod::sql_server(user, pass));
    } else {
        // No credentials — try Windows Integrated Authentication
        config.authentication(tiberius::AuthMethod::Integrated);
    }

    // Split host from port
    let (host, port) = match hostport.rfind(':') {
        Some(pos) => {
            let port_str = &hostport[pos + 1..];
            match port_str.parse::<u16>() {
                Ok(p) => (&hostport[..pos], p),
                Err(_) => (hostport, 1433),
            }
        }
        None => (hostport, 1433),
    };

    config.host(host);
    config.port(port);

    // Apply query-string options
    if let Some(qs) = query_str {
        for pair in qs.split('&') {
            if let Some(eq_pos) = pair.find('=') {
                let key = &pair[..eq_pos];
                let val = &pair[eq_pos + 1..];
                match key {
                    "trust_server_certificate" | "trustServerCertificate"
                        if val.eq_ignore_ascii_case("true") =>
                    {
                        config.trust_cert();
                    }
                    "encrypt" if val.eq_ignore_ascii_case("false") => {
                        config.encryption(EncryptionLevel::Off);
                    }
                    "encrypt" if val.eq_ignore_ascii_case("true") => {
                        config.encryption(EncryptionLevel::Required);
                    }
                    _ => {} // unknown params silently ignored
                }
            }
        }
    }

    Ok(config)
}

// ── Parameter binding ─────────────────────────────────────────────────────────

/// Bind [`serde_json::Value`] parameters onto a tiberius [`Query`].
///
/// Uses owned `String` values so there are no lifetime issues with the query.
fn bind_params(query: &mut Query<'_>, params: &[Value]) {
    for param in params {
        match param {
            Value::Null => query.bind(Option::<String>::None),
            Value::Bool(b) => query.bind(*b),
            Value::Number(n) => {
                if let Some(i) = n.as_i64() {
                    query.bind(i);
                } else if let Some(f) = n.as_f64() {
                    query.bind(f);
                } else {
                    // Rare: number that is neither i64 nor f64 — bind as string
                    query.bind(n.to_string());
                }
            }
            Value::String(s) => query.bind(s.clone()),
            // Arrays / objects — serialize to JSON string
            other => query.bind(other.to_string()),
        }
    }
}

// ── Row → HashMap conversion ──────────────────────────────────────────────────

fn row_to_map(row: tiberius::Row) -> Row {
    let mut map = HashMap::new();
    let columns = row.columns().to_vec();

    for (i, col) in columns.iter().enumerate() {
        let name = col.name().to_string();
        let value = match col.column_type() {
            // Boolean
            ColumnType::Bit => row
                .get::<bool, _>(i)
                .map(Value::Bool)
                .unwrap_or(Value::Null),

            // Integers
            ColumnType::Int1 => row
                .get::<u8, _>(i)
                .map(|n| Value::Number((n as i64).into()))
                .unwrap_or(Value::Null),
            ColumnType::Int2 => row
                .get::<i16, _>(i)
                .map(|n| Value::Number((n as i64).into()))
                .unwrap_or(Value::Null),
            ColumnType::Int4 => row
                .get::<i32, _>(i)
                .map(|n| Value::Number((n as i64).into()))
                .unwrap_or(Value::Null),
            ColumnType::Int8 => row
                .get::<i64, _>(i)
                .map(|n| Value::Number(n.into()))
                .unwrap_or(Value::Null),

            // Floats / money / numeric — map to f64
            ColumnType::Float4 => row
                .get::<f32, _>(i)
                .and_then(|f| serde_json::Number::from_f64(f as f64))
                .map(Value::Number)
                .unwrap_or(Value::Null),
            ColumnType::Float8 | ColumnType::Money | ColumnType::Money4 => row
                .get::<f64, _>(i)
                .and_then(serde_json::Number::from_f64)
                .map(Value::Number)
                .unwrap_or(Value::Null),

            // Null
            ColumnType::Null => Value::Null,

            // Everything else (VARCHAR, NVARCHAR, TEXT, DATETIME, DATE, GUID, XML, …)
            // Try &str first, then fall back to Null.
            _ => row
                .get::<&str, _>(i)
                .map(|s| Value::String(s.to_owned()))
                .unwrap_or(Value::Null),
        };
        map.insert(name, value);
    }
    map
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_url_full() {
        let cfg = parse_url("mssql://sa:Secret1@localhost:1433/mydb?trust_server_certificate=true")
            .expect("should parse");
        // Verify via debug output (Config doesn't expose getters for all fields)
        let dbg = format!("{:?}", cfg);
        assert!(dbg.contains("mydb") || dbg.contains("localhost"));
    }

    #[test]
    fn test_parse_url_no_port() {
        let cfg = parse_url("mssql://user:pass@db.example.com/sales").expect("should parse");
        let dbg = format!("{:?}", cfg);
        assert!(dbg.contains("sales") || dbg.contains("db.example.com"));
    }

    #[test]
    fn test_parse_url_sqlserver_scheme() {
        parse_url("sqlserver://user:pass@host/db").expect("sqlserver:// should be accepted");
    }

    #[test]
    fn test_parse_url_bad_scheme() {
        assert!(parse_url("postgres://user:pass@host/db").is_err());
    }
}
