//! Google BigQuery backend for [`SqlDatabase`].
//!
//! Uses the BigQuery REST API — no extra crate needed beyond `reqwest`.
//! Authentication is via a service account Bearer token.
//!
//! # Example
//! ```rust,ignore
//! use flowgentra_ai::core::db::sql::bigquery::BigQueryDatabase;
//! use flowgentra_ai::core::db::sql::SqlDatabase;
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let db = BigQueryDatabase::new(BigQueryConfig {
//!         project_id: "my-project".into(),
//!         dataset_id: "my_dataset".into(),
//!         access_token: std::env::var("BIGQUERY_TOKEN")?,
//!     });
//!
//!     let rows = db.query(
//!         "SELECT name, age FROM `my-project.my_dataset.users` LIMIT 10",
//!         &[]
//!     ).await?;
//!     println!("{rows:#?}");
//!     Ok(())
//! }
//! ```
//!
//! ## Obtaining an access token
//! Use `gcloud auth print-access-token` in local dev, or a service account
//! JSON key with the `google-cloud-auth` crate for production.

use async_trait::async_trait;
use serde_json::{json, Value};
use std::collections::HashMap;

use super::{DbError, Row, SqlDatabase};

/// Configuration for [`BigQueryDatabase`].
#[derive(Debug, Clone)]
pub struct BigQueryConfig {
    /// GCP project ID
    pub project_id: String,
    /// Default dataset ID (used for DDL operations; SELECT queries are fully qualified)
    pub dataset_id: String,
    /// OAuth2 Bearer access token
    pub access_token: String,
}

/// Google BigQuery SQL backend using the Jobs REST API.
pub struct BigQueryDatabase {
    client: reqwest::Client,
    config: BigQueryConfig,
}

impl BigQueryDatabase {
    pub fn new(config: BigQueryConfig) -> Self {
        Self {
            client: reqwest::Client::new(),
            config,
        }
    }

    fn query_url(&self) -> String {
        format!(
            "https://bigquery.googleapis.com/bigquery/v2/projects/{}/queries",
            self.config.project_id
        )
    }

    fn jobs_url(&self) -> String {
        format!(
            "https://bigquery.googleapis.com/bigquery/v2/projects/{}/jobs",
            self.config.project_id
        )
    }

    fn auth(&self, req: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
        req.bearer_auth(&self.config.access_token)
    }

    /// Convert a BQ schema field type + value to `serde_json::Value`.
    fn bq_val(type_name: &str, raw: Option<&str>) -> Value {
        match raw {
            None | Some("") => Value::Null,
            Some(s) => match type_name {
                "INTEGER" | "INT64" => s
                    .parse::<i64>()
                    .map(|n| Value::Number(n.into()))
                    .unwrap_or(Value::Null),
                "FLOAT" | "FLOAT64" => s
                    .parse::<f64>()
                    .ok()
                    .and_then(|f| serde_json::Number::from_f64(f).map(Value::Number))
                    .unwrap_or(Value::Null),
                "BOOLEAN" | "BOOL" => Value::Bool(s == "true"),
                "JSON" => serde_json::from_str(s).unwrap_or(Value::String(s.to_string())),
                _ => Value::String(s.to_string()),
            },
        }
    }

    async fn run_query(&self, sql: &str, params: &[Value]) -> Result<Vec<Row>, DbError> {
        let mut body = json!({
            "query":            sql,
            "useLegacySql":     false,
            "timeoutMs":        30000,
            "defaultDataset": {
                "projectId": self.config.project_id,
                "datasetId": self.config.dataset_id,
            }
        });

        if !params.is_empty() {
            let query_parameters: Vec<Value> = params
                .iter()
                .enumerate()
                .map(|(i, p)| bq_query_parameter(&format!("param_{}", i), p))
                .collect();
            body["queryParameters"] = json!(query_parameters);
            body["parameterMode"] = json!("NAMED");
        }

        let resp = self
            .auth(self.client.post(&self.query_url()))
            .json(&body)
            .send()
            .await
            .map_err(|e| DbError::Connection(e.to_string()))?;

        if !resp.status().is_success() {
            let text = resp.text().await.unwrap_or_default();
            return Err(DbError::Query(format!("BigQuery error: {text}")));
        }

        let data: Value = resp
            .json()
            .await
            .map_err(|e| DbError::Serialization(e.to_string()))?;

        // Handle async jobs (jobComplete = false)
        if data["jobComplete"].as_bool() == Some(false) {
            let job_id = data["jobReference"]["jobId"].as_str().unwrap_or("");
            return self.wait_for_job(job_id).await;
        }

        Ok(parse_query_response(&data))
    }

    async fn wait_for_job(&self, job_id: &str) -> Result<Vec<Row>, DbError> {
        let results_url = format!(
            "https://bigquery.googleapis.com/bigquery/v2/projects/{}/queries/{}",
            self.config.project_id, job_id
        );
        for _ in 0..30 {
            tokio::time::sleep(std::time::Duration::from_secs(1)).await;
            let resp = self
                .auth(self.client.get(&results_url))
                .send()
                .await
                .map_err(|e| DbError::Connection(e.to_string()))?;
            let data: Value = resp
                .json()
                .await
                .map_err(|e| DbError::Serialization(e.to_string()))?;
            if data["jobComplete"].as_bool() == Some(true) {
                return Ok(parse_query_response(&data));
            }
        }
        Err(DbError::Query("BigQuery job timed out".into()))
    }
}

/// Convert a `serde_json::Value` into a BigQuery named query parameter object.
///
/// This uses the BigQuery Jobs REST API `queryParameters` format so values are
/// never interpolated into the SQL string — eliminating SQL injection risk.
fn bq_query_parameter(name: &str, value: &Value) -> Value {
    let (type_name, value_obj) = match value {
        Value::String(s) => ("STRING", json!({ "value": s })),
        Value::Bool(b) => (
            "BOOL",
            json!({ "value": if *b { "true" } else { "false" } }),
        ),
        Value::Number(n) => {
            if n.is_f64() {
                ("FLOAT64", json!({ "value": n.to_string() }))
            } else {
                ("INT64", json!({ "value": n.to_string() }))
            }
        }
        Value::Null => ("STRING", json!({ "value": null })),
        other => ("STRING", json!({ "value": other.to_string() })),
    };
    json!({
        "name": name,
        "parameterType": { "type": type_name },
        "parameterValue": value_obj
    })
}

fn parse_query_response(data: &Value) -> Vec<Row> {
    let schema_fields = data["schema"]["fields"].as_array();
    let rows = data["rows"].as_array();
    match (schema_fields, rows) {
        (Some(fields), Some(rows)) => rows
            .iter()
            .map(|row| {
                let cells = row["f"].as_array().map(|v| v.as_slice()).unwrap_or(&[]);
                fields
                    .iter()
                    .zip(cells.iter())
                    .map(|(field, cell)| {
                        let col_name = field["name"].as_str().unwrap_or("").to_string();
                        let type_name = field["type"].as_str().unwrap_or("STRING");
                        let raw = cell["v"].as_str();
                        (col_name, BigQueryDatabase::bq_val(type_name, raw))
                    })
                    .collect::<Row>()
            })
            .collect(),
        _ => Vec::new(),
    }
}

#[async_trait]
impl SqlDatabase for BigQueryDatabase {
    /// Run a BigQuery SQL query.
    ///
    /// `params` are passed as `@param_0`, `@param_1` etc. using BigQuery's
    /// native named query parameter API (no string interpolation).
    /// Use `@param_0` in your SQL to reference them.
    async fn query(&self, sql: &str, params: &[Value]) -> Result<Vec<Row>, DbError> {
        self.run_query(sql, params).await
    }

    /// Execute a DDL or DML statement (INSERT, UPDATE, DELETE, CREATE, etc.).
    ///
    /// BigQuery is append-optimised; DML statements work but are slower than
    /// bulk load. Returns 0 for DDL (BigQuery doesn't report rows affected).
    async fn execute(&self, sql: &str, params: &[Value]) -> Result<u64, DbError> {
        self.run_query(sql, params)
            .await
            .map(|rows| rows.len() as u64)
    }
}
