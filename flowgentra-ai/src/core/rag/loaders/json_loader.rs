//! JSON and JSONL Document Loaders
//!
//! - [`JsonLoader`] — loads a `.json` file containing an array of objects.
//!   Each object becomes one document.
//! - [`JsonlLoader`] — loads a `.jsonl` / `.ndjson` file with one JSON object
//!   per line. Each line becomes one document.
//!
//! ## Example
//!
//! ```rust,ignore
//! use flowgentra_ai::core::rag::loaders::{JsonLoader, JsonlLoader};
//!
//! // JSON array: [{"id": "1", "text": "..."}, ...]
//! let docs = JsonLoader::new("text").load("data.json").await?;
//!
//! // JSONL: {"id": "1", "content": "..."}\n{"id": "2", "content": "..."}
//! let docs = JsonlLoader::new("content").load("data.jsonl").await?;
//! ```

use serde_json::Value;
use std::collections::HashMap;
use std::path::Path;

use crate::core::rag::{document_loader::LoadedDocument, vector_db::VectorStoreError};

// ── JsonLoader ───────────────────────────────────────────────────────────────

/// Loads a JSON file containing an array of objects.
pub struct JsonLoader {
    /// JSON field to use as document text.
    text_field: String,
    /// JSON field to use as document id. Defaults to `"id"` or row index.
    id_field: Option<String>,
}

impl JsonLoader {
    /// Create a loader that reads the document text from `text_field`.
    pub fn new(text_field: impl Into<String>) -> Self {
        Self {
            text_field: text_field.into(),
            id_field: None,
        }
    }

    pub fn with_id_field(mut self, field: impl Into<String>) -> Self {
        self.id_field = Some(field.into());
        self
    }

    /// Load documents from a JSON array file.
    pub async fn load(
        &self,
        path: impl AsRef<Path>,
    ) -> Result<Vec<LoadedDocument>, VectorStoreError> {
        let path = path.as_ref();
        let source = path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("unknown")
            .to_string();

        let content = tokio::fs::read_to_string(path)
            .await
            .map_err(|e| VectorStoreError::Unknown(format!("Read JSON error: {e}")))?;

        self.parse_json(&content, &source)
    }

    /// Parse a JSON string (array of objects).
    pub fn parse_json(
        &self,
        content: &str,
        source: &str,
    ) -> Result<Vec<LoadedDocument>, VectorStoreError> {
        let value: Value = serde_json::from_str(content)
            .map_err(|e| VectorStoreError::SerializationError(format!("JSON parse error: {e}")))?;

        let array = value.as_array().ok_or_else(|| {
            VectorStoreError::Unknown("JSON root must be an array of objects".into())
        })?;

        let mut docs = Vec::new();
        for (i, item) in array.iter().enumerate() {
            let obj = item.as_object().ok_or_else(|| {
                VectorStoreError::Unknown(format!("JSON element {i} is not an object"))
            })?;

            let text = obj
                .get(&self.text_field)
                .and_then(|v| v.as_str())
                .ok_or_else(|| {
                    VectorStoreError::Unknown(format!(
                        "Field '{}' not found or not a string in element {i}",
                        self.text_field
                    ))
                })?
                .to_string();

            let id = self
                .id_field
                .as_ref()
                .and_then(|f| obj.get(f))
                .and_then(|v| v.as_str())
                .map(|s| s.to_string())
                .unwrap_or_else(|| format!("{source}_item_{i}"));

            let mut metadata: HashMap<String, Value> = HashMap::new();
            metadata.insert("source".to_string(), Value::String(source.to_string()));
            metadata.insert("index".to_string(), Value::Number(i.into()));
            for (k, v) in obj {
                if k != &self.text_field {
                    metadata.insert(k.clone(), v.clone());
                }
            }

            docs.push(LoadedDocument {
                id,
                text,
                source: source.to_string(),
                file_type: crate::core::rag::document_loader::FileType::Unknown,
                metadata,
            });
        }

        Ok(docs)
    }
}

// ── JsonlLoader ──────────────────────────────────────────────────────────────

/// Loads a JSONL (newline-delimited JSON) file.
pub struct JsonlLoader {
    text_field: String,
    id_field: Option<String>,
}

impl JsonlLoader {
    pub fn new(text_field: impl Into<String>) -> Self {
        Self {
            text_field: text_field.into(),
            id_field: None,
        }
    }

    pub fn with_id_field(mut self, field: impl Into<String>) -> Self {
        self.id_field = Some(field.into());
        self
    }

    pub async fn load(
        &self,
        path: impl AsRef<Path>,
    ) -> Result<Vec<LoadedDocument>, VectorStoreError> {
        let path = path.as_ref();
        let source = path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("unknown")
            .to_string();

        let content = tokio::fs::read_to_string(path)
            .await
            .map_err(|e| VectorStoreError::Unknown(format!("Read JSONL error: {e}")))?;

        self.parse_jsonl(&content, &source)
    }

    /// Parse JSONL content from a string.
    pub fn parse_jsonl(
        &self,
        content: &str,
        source: &str,
    ) -> Result<Vec<LoadedDocument>, VectorStoreError> {
        let mut docs = Vec::new();

        for (line_no, line) in content.lines().enumerate() {
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }

            let value: Value = serde_json::from_str(trimmed).map_err(|e| {
                VectorStoreError::SerializationError(format!(
                    "JSONL parse error at line {line_no}: {e}"
                ))
            })?;

            let obj = value.as_object().ok_or_else(|| {
                VectorStoreError::Unknown(format!("JSONL line {line_no} is not a JSON object"))
            })?;

            let text = obj
                .get(&self.text_field)
                .and_then(|v| v.as_str())
                .ok_or_else(|| {
                    VectorStoreError::Unknown(format!(
                        "Field '{}' not found at line {line_no}",
                        self.text_field
                    ))
                })?
                .to_string();

            let id = self
                .id_field
                .as_ref()
                .and_then(|f| obj.get(f))
                .and_then(|v| v.as_str())
                .map(|s| s.to_string())
                .unwrap_or_else(|| format!("{source}_line_{line_no}"));

            let mut metadata: HashMap<String, Value> = HashMap::new();
            metadata.insert("source".to_string(), Value::String(source.to_string()));
            metadata.insert("line".to_string(), Value::Number(line_no.into()));
            for (k, v) in obj {
                if k != &self.text_field {
                    metadata.insert(k.clone(), v.clone());
                }
            }

            docs.push(LoadedDocument {
                id,
                text,
                source: source.to_string(),
                file_type: crate::core::rag::document_loader::FileType::Unknown,
                metadata,
            });
        }

        Ok(docs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_json_loader_basic() {
        let json = r#"[
            {"id": "a", "text": "Hello world", "year": 2024},
            {"id": "b", "text": "Foo bar", "year": 2023}
        ]"#;

        let loader = JsonLoader::new("text").with_id_field("id");
        let docs = loader.parse_json(json, "test.json").unwrap();
        assert_eq!(docs.len(), 2);
        assert_eq!(docs[0].id, "a");
        assert_eq!(docs[0].text, "Hello world");
        assert_eq!(docs[0].metadata["year"], serde_json::json!(2024));
        assert_eq!(docs[1].id, "b");
    }

    #[test]
    fn test_json_loader_no_id_field() {
        let json = r#"[{"text": "one"}, {"text": "two"}]"#;
        let loader = JsonLoader::new("text");
        let docs = loader.parse_json(json, "f.json").unwrap();
        assert_eq!(docs[0].id, "f.json_item_0");
        assert_eq!(docs[1].id, "f.json_item_1");
    }

    #[test]
    fn test_jsonl_loader_basic() {
        let jsonl = r#"{"id": "1", "content": "First line", "tag": "a"}
{"id": "2", "content": "Second line", "tag": "b"}
"#;
        let loader = JsonlLoader::new("content").with_id_field("id");
        let docs = loader.parse_jsonl(jsonl, "data.jsonl").unwrap();
        assert_eq!(docs.len(), 2);
        assert_eq!(docs[0].id, "1");
        assert_eq!(docs[0].text, "First line");
        assert_eq!(docs[1].metadata["tag"], serde_json::json!("b"));
    }

    #[test]
    fn test_jsonl_skips_empty_lines() {
        let jsonl = "\n{\"content\": \"hello\"}\n\n{\"content\": \"world\"}\n";
        let loader = JsonlLoader::new("content");
        let docs = loader.parse_jsonl(jsonl, "f.jsonl").unwrap();
        assert_eq!(docs.len(), 2);
    }
}
