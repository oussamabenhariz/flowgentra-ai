//! Databricks SQL backend for [`SqlDatabase`].
//!
//! Uses the Databricks SQL Statement Execution REST API.
//! No extra crate — relies on `reqwest`.
//!
//! # Example
//! ```rust,ignore
//! use flowgentra_ai::core::db::sql::databricks::DatabricksDatabase;
//! use flowgentra_ai::core::db::sql::SqlDatabase;
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let db = DatabricksDatabase::new(DatabricksConfig {
//!         host:         "https://<workspace>.azuredatabricks.net".into(),
//!         warehouse_id: "abc123".into(),
//!         token:        std::env::var("DATABRICKS_TOKEN")?,
//!         catalog:      Some("main".into()),
//!         schema:       Some("default".into()),
//!     });
//!
//!     let rows = db.query("SELECT * FROM my_table LIMIT 10", &[]).await?;
//!     println!("{rows:#?}");
//!     Ok(())
//! }
//! ```

use async_trait::async_trait;
use serde_json::{json, Value};
use std::collections::HashMap;

use super::{DbError, Row, SqlDatabase};

/// Configuration for [`DatabricksDatabase`].
#[derive(Debug, Clone)]
pub struct DatabricksConfig {
    /// Databricks workspace host (e.g. `https://adb-xxx.azuredatabricks.net`)
    pub host: String,
    /// SQL Warehouse ID
    pub warehouse_id: String,
    /// Personal access token or service principal token
    pub token: String,
    /// Optional Unity Catalog name
    pub catalog: Option<String>,
    /// Optional schema name
    pub schema: Option<String>,
}

/// Databricks SQL backend using the Statement Execution API.
pub struct DatabricksDatabase {
    client: reqwest::Client,
    config: DatabricksConfig,
}

impl DatabricksDatabase {
    pub fn new(config: DatabricksConfig) -> Self {
        Self {
            client: reqwest::Client::new(),
            config,
        }
    }

    fn statements_url(&self) -> String {
        format!(
            "{}/api/2.0/sql/statements",
            self.config.host.trim_end_matches('/')
        )
    }

    fn auth(&self, req: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
        req.bearer_auth(&self.config.token)
    }

    async fn execute_statement(&self, sql: &str) -> Result<Value, DbError> {
        let mut body = json!({
            "statement":    sql,
            "warehouse_id": self.config.warehouse_id,
            "wait_timeout": "30s",
            "on_wait_timeout": "CONTINUE",
        });
        if let Some(ref cat) = self.config.catalog {
            body["catalog"] = Value::String(cat.clone());
        }
        if let Some(ref sch) = self.config.schema {
            body["schema"] = Value::String(sch.clone());
        }

        let resp = self
            .auth(self.client.post(&self.statements_url()))
            .json(&body)
            .send()
            .await
            .map_err(|e| DbError::Connection(e.to_string()))?;

        if !resp.status().is_success() {
            let text = resp.text().await.unwrap_or_default();
            return Err(DbError::Query(format!("Databricks error: {text}")));
        }

        let mut data: Value = resp
            .json()
            .await
            .map_err(|e| DbError::Serialization(e.to_string()))?;

        // Poll if not yet done
        let statement_id = data["statement_id"].as_str().unwrap_or("").to_string();
        let poll_url = format!("{}/{}", self.statements_url(), statement_id);
        for _ in 0..60 {
            let state = data["status"]["state"].as_str().unwrap_or("");
            match state {
                "SUCCEEDED" => break,
                "FAILED" | "CANCELED" | "CLOSED" => {
                    let msg = data["status"]["error"]["message"]
                        .as_str()
                        .unwrap_or("unknown");
                    return Err(DbError::Query(format!(
                        "Databricks statement failed: {msg}"
                    )));
                }
                _ => {
                    tokio::time::sleep(std::time::Duration::from_millis(500)).await;
                    let poll_resp = self
                        .auth(self.client.get(&poll_url))
                        .send()
                        .await
                        .map_err(|e| DbError::Connection(e.to_string()))?;
                    data = poll_resp
                        .json()
                        .await
                        .map_err(|e| DbError::Serialization(e.to_string()))?;
                }
            }
        }
        Ok(data)
    }

    fn parse_response(data: &Value) -> Vec<Row> {
        let schema = data["manifest"]["schema"]["columns"].as_array();
        let rows = data["result"]["data_array"].as_array();
        match (schema, rows) {
            (Some(cols), Some(rows)) => rows
                .iter()
                .map(|row| {
                    let cells = row.as_array().map(|v| v.as_slice()).unwrap_or(&[]);
                    cols.iter()
                        .zip(cells.iter())
                        .map(|(col, cell)| {
                            let name = col["name"].as_str().unwrap_or("").to_string();
                            let type_name = col["type_name"].as_str().unwrap_or("STRING");
                            let val = match (type_name, cell) {
                                (_, Value::Null) => Value::Null,
                                ("LONG" | "INT" | "SHORT" | "BYTE", v) => v
                                    .as_str()
                                    .and_then(|s| s.parse::<i64>().ok())
                                    .map(|n| Value::Number(n.into()))
                                    .unwrap_or(Value::Null),
                                ("DOUBLE" | "FLOAT" | "DECIMAL", v) => v
                                    .as_str()
                                    .and_then(|s| s.parse::<f64>().ok())
                                    .and_then(|f| {
                                        serde_json::Number::from_f64(f).map(Value::Number)
                                    })
                                    .unwrap_or(Value::Null),
                                ("BOOLEAN", v) => Value::Bool(v.as_str() == Some("true")),
                                (_, v) => v
                                    .as_str()
                                    .map(|s| Value::String(s.to_string()))
                                    .unwrap_or(Value::Null),
                            };
                            (name, val)
                        })
                        .collect()
                })
                .collect(),
            _ => Vec::new(),
        }
    }
}

#[async_trait]
impl SqlDatabase for DatabricksDatabase {
    async fn query(&self, sql: &str, _params: &[Value]) -> Result<Vec<Row>, DbError> {
        // Note: Databricks Statement API supports named params; for simplicity
        // we pass the SQL as-is. Use `?` markers and pre-fill params yourself,
        // or use the Databricks SDK for full parameterized support.
        let data = self.execute_statement(sql).await?;
        Ok(Self::parse_response(&data))
    }

    async fn execute(&self, sql: &str, _params: &[Value]) -> Result<u64, DbError> {
        let data = self.execute_statement(sql).await?;
        let rows = Self::parse_response(&data);
        Ok(rows.len() as u64)
    }
}
