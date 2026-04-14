//! Excel / XLSX Loader
//!
//! Loads rows from `.xlsx` spreadsheets. An XLSX file is a ZIP archive
//! containing XML files; this loader parses `xl/worksheets/sheet1.xml`
//! and `xl/sharedStrings.xml` using the `zip` crate and a minimal XML parser.
//!
//! Each row becomes one [`LoadedDocument`] where all cell values are joined
//! with " | " as the document text.

use std::collections::HashMap;
use std::io::Read;

use serde_json::json;

use crate::core::rag::document_loader::{FileType, LoadedDocument};

/// Loads rows from an `.xlsx` file as documents.
pub struct ExcelLoader {
    path: String,
    /// Zero-based sheet index to load (default 0 = first sheet).
    sheet_index: usize,
    /// If true, first row is treated as header and stored in metadata.
    has_header: bool,
}

impl ExcelLoader {
    pub fn new(path: impl Into<String>) -> Self {
        Self {
            path: path.into(),
            sheet_index: 0,
            has_header: true,
        }
    }

    pub fn with_sheet(mut self, index: usize) -> Self {
        self.sheet_index = index;
        self
    }

    pub fn without_header(mut self) -> Self {
        self.has_header = false;
        self
    }

    pub fn load(&self) -> Result<Vec<LoadedDocument>, Box<dyn std::error::Error>> {
        let file = std::fs::File::open(&self.path)?;
        let mut archive = zip::ZipArchive::new(file)?;

        // Read shared strings table
        let shared_strings = self.read_shared_strings(&mut archive)?;
        // Read the target worksheet
        let rows = self.read_sheet(&mut archive, &shared_strings)?;

        let source = self.path.clone();
        let mut docs = Vec::new();
        let start = if self.has_header && !rows.is_empty() { 1 } else { 0 };
        let headers: Vec<String> = if self.has_header && !rows.is_empty() {
            rows[0].clone()
        } else {
            vec![]
        };

        for (i, row) in rows[start..].iter().enumerate() {
            let text = row.join(" | ");
            let mut metadata = HashMap::new();
            metadata.insert("source".to_string(), json!(source));
            metadata.insert("sheet_index".to_string(), json!(self.sheet_index));
            metadata.insert("row_index".to_string(), json!(i));
            if !headers.is_empty() {
                for (h, v) in headers.iter().zip(row.iter()) {
                    metadata.insert(h.clone(), json!(v));
                }
            }
            docs.push(LoadedDocument {
                id: format!("excel_row_{i}"),
                text,
                source: format!("{}#row_{i}", self.path),
                file_type: FileType::PlainText,
                metadata,
            });
        }
        Ok(docs)
    }

    fn read_shared_strings(
        &self,
        archive: &mut zip::ZipArchive<std::fs::File>,
    ) -> Result<Vec<String>, Box<dyn std::error::Error>> {
        let mut strings = Vec::new();
        if let Ok(mut file) = archive.by_name("xl/sharedStrings.xml") {
            let mut xml = String::new();
            file.read_to_string(&mut xml)?;
            // Extract all <t> tag contents
            for part in xml.split("<t>").skip(1) {
                let text = part.split("</t>").next().unwrap_or("").to_string();
                let text = text
                    .replace("&amp;", "&")
                    .replace("&lt;", "<")
                    .replace("&gt;", ">")
                    .replace("&apos;", "'")
                    .replace("&quot;", "\"");
                strings.push(text);
            }
            // Also handle <t xml:space="preserve"> variants
            for part in xml.split("<t xml:space=\"preserve\">").skip(1) {
                let text = part.split("</t>").next().unwrap_or("").to_string();
                strings.push(text);
            }
        }
        Ok(strings)
    }

    fn read_sheet(
        &self,
        archive: &mut zip::ZipArchive<std::fs::File>,
        shared_strings: &[String],
    ) -> Result<Vec<Vec<String>>, Box<dyn std::error::Error>> {
        let sheet_name = format!("xl/worksheets/sheet{}.xml", self.sheet_index + 1);
        let mut file = archive.by_name(&sheet_name).map_err(|_| {
            std::io::Error::new(std::io::ErrorKind::NotFound, format!("Sheet {sheet_name} not found"))
        })?;
        let mut xml = String::new();
        file.read_to_string(&mut xml)?;

        let mut rows = Vec::new();
        for row_part in xml.split("<row").skip(1) {
            let row_xml = row_part.split("</row>").next().unwrap_or("");
            let mut cells = Vec::new();
            for cell_part in row_xml.split("<c ").chain(row_xml.split("<c>")).skip(1) {
                let type_attr = if cell_part.contains("t=\"s\"") {
                    "s" // shared string
                } else if cell_part.contains("t=\"str\"") {
                    "str"
                } else {
                    "n" // number
                };

                let value_str = cell_part
                    .split("<v>")
                    .nth(1)
                    .and_then(|s| s.split("</v>").next())
                    .unwrap_or("")
                    .trim();

                let cell_value = match type_attr {
                    "s" => {
                        let idx: usize = value_str.parse().unwrap_or(0);
                        shared_strings.get(idx).cloned().unwrap_or_default()
                    }
                    _ => value_str.to_string(),
                };
                cells.push(cell_value);
            }
            if !cells.is_empty() {
                rows.push(cells);
            }
        }
        Ok(rows)
    }
}
