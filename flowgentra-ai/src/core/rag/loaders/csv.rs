//! CSV Loader
//!
//! Loads a CSV file and converts each row into a [`LoadedDocument`].
//! Column values are joined into a readable text representation. Individual
//! column values are also stored as document metadata.
//!
//! ## Example
//!
//! ```rust,ignore
//! use flowgentra_ai::core::rag::loaders::CsvLoader;
//!
//! // Each row becomes one document; text = "col1: val1 | col2: val2 | ..."
//! let docs = CsvLoader::new().load("data.csv").await?;
//!
//! // Custom text column — use only the "description" column as document text
//! let docs = CsvLoader::new()
//!     .with_text_column("description")
//!     .load("products.csv")
//!     .await?;
//! ```

use serde_json::json;
use std::collections::HashMap;
use std::path::Path;

use crate::core::rag::{document_loader::LoadedDocument, vector_db::VectorStoreError};

/// Loader for CSV files.
pub struct CsvLoader {
    /// If set, use only this column as the document text.
    /// If `None`, all columns are joined: "col1: val1 | col2: val2 | ..."
    text_column: Option<String>,
    /// Column to use as document id. Defaults to row index.
    id_column: Option<String>,
    /// CSV delimiter character (default: `,`).
    delimiter: u8,
    /// Whether the first row is a header row (default: `true`).
    has_headers: bool,
}

impl Default for CsvLoader {
    fn default() -> Self {
        Self {
            text_column: None,
            id_column: None,
            delimiter: b',',
            has_headers: true,
        }
    }
}

impl CsvLoader {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_text_column(mut self, col: impl Into<String>) -> Self {
        self.text_column = Some(col.into());
        self
    }

    pub fn with_id_column(mut self, col: impl Into<String>) -> Self {
        self.id_column = Some(col.into());
        self
    }

    pub fn with_delimiter(mut self, delim: u8) -> Self {
        self.delimiter = delim;
        self
    }

    pub fn without_headers(mut self) -> Self {
        self.has_headers = false;
        self
    }

    /// Load a CSV file and return one document per data row.
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
            .map_err(|e| VectorStoreError::Unknown(format!("Read CSV error: {e}")))?;

        self.parse_csv(&content, &source)
    }

    /// Parse CSV content from a string (useful for testing or in-memory data).
    pub fn parse_csv(
        &self,
        content: &str,
        source: &str,
    ) -> Result<Vec<LoadedDocument>, VectorStoreError> {
        let mut reader = csv_core_reader(content, self.delimiter, self.has_headers);

        let headers: Vec<String> = if self.has_headers {
            first_row_headers(&mut reader)?
        } else {
            vec![]
        };

        let mut docs = Vec::new();
        for (row_idx, result) in reader.records().enumerate() {
            let record = result.map_err(|e| {
                VectorStoreError::Unknown(format!("CSV parse error at row {row_idx}: {e}"))
            })?;

            let cols: Vec<String> = record.iter().map(|f| f.to_string()).collect();

            // Build metadata from headers + values
            let mut metadata: HashMap<String, serde_json::Value> = HashMap::new();
            metadata.insert("source".to_string(), json!(source));
            metadata.insert("row".to_string(), json!(row_idx));

            for (i, val) in cols.iter().enumerate() {
                let key = headers
                    .get(i)
                    .cloned()
                    .unwrap_or_else(|| format!("col{}", i));
                metadata.insert(key, json!(val));
            }

            // Determine document text
            let text = if let Some(tc) = &self.text_column {
                // Find the text column
                let col_idx = headers.iter().position(|h| h == tc);
                match col_idx {
                    Some(i) => cols.get(i).cloned().unwrap_or_default(),
                    None => {
                        return Err(VectorStoreError::Unknown(format!(
                            "Text column '{tc}' not found in headers"
                        )))
                    }
                }
            } else {
                // Join all columns
                cols.iter()
                    .enumerate()
                    .map(|(i, v)| {
                        let h = headers
                            .get(i)
                            .cloned()
                            .unwrap_or_else(|| format!("col{}", i));
                        format!("{h}: {v}")
                    })
                    .collect::<Vec<_>>()
                    .join(" | ")
            };

            // Determine document id
            let id = if let Some(ic) = &self.id_column {
                let col_idx = headers.iter().position(|h| h == ic);
                match col_idx {
                    Some(i) => cols
                        .get(i)
                        .cloned()
                        .unwrap_or_else(|| format!("{source}_row_{row_idx}")),
                    None => format!("{source}_row_{row_idx}"),
                }
            } else {
                format!("{source}_row_{row_idx}")
            };

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

// ── CSV parsing helpers (no extra crate — hand-rolled) ───────────────────────

fn parse_csv_line(line: &str, delimiter: char) -> Vec<String> {
    let mut fields = Vec::new();
    let mut current = String::new();
    let mut in_quotes = false;

    for ch in line.chars() {
        if ch == '"' {
            in_quotes = !in_quotes;
        } else if ch == delimiter && !in_quotes {
            fields.push(current.trim().to_string());
            current = String::new();
        } else {
            current.push(ch);
        }
    }
    fields.push(current.trim().to_string());
    fields
}

struct SimpleCsvReaderWrapper {
    headers: Vec<String>,
    rows: Vec<Vec<String>>,
}

impl SimpleCsvReaderWrapper {
    fn records(&self) -> impl Iterator<Item = Result<Record, VectorStoreError>> + '_ {
        self.rows.iter().map(|r| Ok(Record(r.clone())))
    }
}

struct Record(Vec<String>);

impl Record {
    fn iter(&self) -> impl Iterator<Item = &str> {
        self.0.iter().map(|s| s.as_str())
    }
}

fn csv_core_reader(content: &str, delimiter: u8, has_headers: bool) -> SimpleCsvReaderWrapper {
    let delim = delimiter as char;
    let mut lines = content.lines().filter(|l| !l.trim().is_empty());

    let headers = if has_headers {
        lines
            .next()
            .map(|l| parse_csv_line(l, delim))
            .unwrap_or_default()
    } else {
        vec![]
    };

    let rows: Vec<Vec<String>> = lines.map(|l| parse_csv_line(l, delim)).collect();

    SimpleCsvReaderWrapper { headers, rows }
}

fn first_row_headers(reader: &mut SimpleCsvReaderWrapper) -> Result<Vec<String>, VectorStoreError> {
    Ok(reader.headers.clone())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_csv_basic() {
        let csv = "name,description,year\nRust,Systems language,2015\nPython,Scripting,1991";
        let loader = CsvLoader::new();
        let docs = loader.parse_csv(csv, "test.csv").unwrap();
        assert_eq!(docs.len(), 2);
        assert!(docs[0].text.contains("name: Rust"));
        assert!(docs[0].text.contains("description: Systems language"));
    }

    #[test]
    fn test_csv_text_column() {
        let csv = "id,description\n1,Hello world\n2,Foo bar";
        let loader = CsvLoader::new().with_text_column("description");
        let docs = loader.parse_csv(csv, "test.csv").unwrap();
        assert_eq!(docs[0].text, "Hello world");
        assert_eq!(docs[1].text, "Foo bar");
    }

    #[test]
    fn test_csv_id_column() {
        let csv = "doc_id,text\nabc,first\ndef,second";
        let loader = CsvLoader::new()
            .with_id_column("doc_id")
            .with_text_column("text");
        let docs = loader.parse_csv(csv, "test.csv").unwrap();
        assert_eq!(docs[0].id, "abc");
        assert_eq!(docs[1].id, "def");
    }

    #[test]
    fn test_csv_metadata_stored() {
        let csv = "name,year\nRust,2015";
        let loader = CsvLoader::new();
        let docs = loader.parse_csv(csv, "test.csv").unwrap();
        assert_eq!(docs[0].metadata["name"], serde_json::json!("Rust"));
        assert_eq!(docs[0].metadata["year"], serde_json::json!("2015"));
    }
}
