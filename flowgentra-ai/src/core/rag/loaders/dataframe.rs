//! DataFrame Loader
//!
//! Converts tabular data (represented as a `Vec<HashMap<String, String>>`) into
//! [`LoadedDocument`]s. One row = one document. Designed to interop with data
//! passed from Python pandas DataFrames via JSON serialization.

use crate::core::rag::document_loader::{FileType, LoadedDocument};
use serde_json::json;
use std::collections::HashMap;

/// Converts rows (HashMap-based) into documents.
///
/// The `page_content_column` field determines which column becomes the document
/// text. All other columns go into metadata.
pub struct DataFrameLoader {
    rows: Vec<HashMap<String, String>>,
    /// Name of the column whose value becomes `doc.text`.
    pub page_content_column: String,
}

impl DataFrameLoader {
    pub fn new(rows: Vec<HashMap<String, String>>, page_content_column: impl Into<String>) -> Self {
        Self {
            rows,
            page_content_column: page_content_column.into(),
        }
    }

    /// Construct from a JSON array of objects (e.g. from `df.to_json(orient="records")`).
    pub fn from_json(
        json_str: &str,
        page_content_column: impl Into<String>,
    ) -> Result<Self, serde_json::Error> {
        let rows: Vec<HashMap<String, serde_json::Value>> = serde_json::from_str(json_str)?;
        let string_rows = rows
            .into_iter()
            .map(|row| {
                row.into_iter()
                    .map(|(k, v)| {
                        (
                            k,
                            v.as_str()
                                .map(str::to_string)
                                .unwrap_or_else(|| v.to_string()),
                        )
                    })
                    .collect()
            })
            .collect();
        Ok(Self::new(string_rows, page_content_column))
    }

    pub fn load(&self) -> Vec<LoadedDocument> {
        self.rows
            .iter()
            .enumerate()
            .map(|(i, row)| {
                let text = row
                    .get(&self.page_content_column)
                    .cloned()
                    .unwrap_or_default();
                let mut metadata: HashMap<String, serde_json::Value> = row
                    .iter()
                    .filter(|(k, _)| *k != &self.page_content_column)
                    .map(|(k, v)| (k.clone(), json!(v)))
                    .collect();
                metadata.insert("row_index".to_string(), json!(i));

                LoadedDocument {
                    id: format!("dataframe_row_{i}"),
                    text,
                    source: format!("dataframe_row_{i}"),
                    file_type: FileType::PlainText,
                    metadata,
                }
            })
            .collect()
    }
}
