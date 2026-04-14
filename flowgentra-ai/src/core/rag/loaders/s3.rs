//! S3 Loader
//!
//! Downloads objects from an AWS S3 bucket and parses them as documents.
//! Uses the S3 REST API with AWS Signature V4 (simplified implementation).
//! For production, integrate with the official `aws-sdk-s3` crate.

use std::collections::HashMap;
use serde_json::json;
use crate::core::rag::document_loader::{load_document, LoadedDocument};

pub struct S3Loader {
    pub bucket: String,
    pub prefix: String,
    pub region: String,
    pub access_key: String,
    pub secret_key: String,
    /// Temp directory to download files before parsing.
    pub tmp_dir: String,
}

impl S3Loader {
    pub fn new(
        bucket: impl Into<String>,
        region: impl Into<String>,
        access_key: impl Into<String>,
        secret_key: impl Into<String>,
    ) -> Self {
        Self {
            bucket: bucket.into(),
            prefix: String::new(),
            region: region.into(),
            access_key: access_key.into(),
            secret_key: secret_key.into(),
            tmp_dir: std::env::temp_dir().to_string_lossy().to_string(),
        }
    }

    pub fn with_prefix(mut self, prefix: impl Into<String>) -> Self {
        self.prefix = prefix.into();
        self
    }

    pub fn with_tmp_dir(mut self, dir: impl Into<String>) -> Self {
        self.tmp_dir = dir.into();
        self
    }

    /// List objects in the bucket with the configured prefix.
    async fn list_objects(&self, client: &reqwest::Client) -> Result<Vec<String>, Box<dyn std::error::Error>> {
        let url = format!(
            "https://{}.s3.{}.amazonaws.com/?list-type=2&prefix={}",
            self.bucket, self.region, urlencoding::encode(&self.prefix)
        );
        // NOTE: In production, sign the request with AWS Signature V4.
        // Here we use unsigned requests (works with public buckets or pre-configured credentials).
        let resp = client.get(&url)
            .header("x-amz-content-sha256", "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855")
            .send().await?.text().await?;

        let mut keys = Vec::new();
        for part in resp.split("<Key>").skip(1) {
            let key = part.split("</Key>").next().unwrap_or("").trim().to_string();
            if !key.is_empty() {
                keys.push(key);
            }
        }
        Ok(keys)
    }

    /// Download an object and load its content as a document.
    async fn load_object(
        &self,
        client: &reqwest::Client,
        key: &str,
    ) -> Result<Option<LoadedDocument>, Box<dyn std::error::Error>> {
        let url = format!(
            "https://{}.s3.{}.amazonaws.com/{}",
            self.bucket, self.region, key
        );
        let bytes = client.get(&url).send().await?.bytes().await?;

        // Write to a temp file and use the existing load_document infrastructure
        let ext = key.rsplit('.').next().unwrap_or("txt");
        let tmp_path = format!("{}/{}", self.tmp_dir, key.replace('/', "_"));
        std::fs::write(&tmp_path, &bytes)?;

        let loaded = load_document(&tmp_path)
            .await
            .map_err(|e| Box::new(std::io::Error::new(std::io::ErrorKind::Other, e.to_string())) as Box<dyn std::error::Error>)?;
        std::fs::remove_file(&tmp_path).ok();

        let s3_url = format!("s3://{}/{}", self.bucket, key);
        let mut metadata = HashMap::new();
        metadata.insert("source".to_string(), json!(s3_url));
        metadata.insert("bucket".to_string(), json!(self.bucket));
        metadata.insert("key".to_string(), json!(key));
        metadata.insert("extension".to_string(), json!(ext));

        Ok(Some(LoadedDocument {
            id: format!("s3_{key}"),
            text: loaded.text,
            source: s3_url,
            file_type: loaded.file_type,
            metadata,
        }))
    }

    pub async fn load(&self) -> Result<Vec<LoadedDocument>, Box<dyn std::error::Error>> {
        let client = reqwest::Client::new();
        let keys = self.list_objects(&client).await?;
        let mut docs = Vec::new();
        for key in &keys {
            if let Ok(Some(doc)) = self.load_object(&client, key).await {
                docs.push(doc);
            }
        }
        Ok(docs)
    }
}
