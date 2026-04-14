//! DOCX (Word Document) Loader
//!
//! Extracts plain text from `.docx` Microsoft Word documents.
//!
//! DOCX is a ZIP archive containing `word/document.xml`. This loader:
//! 1. Reads the ZIP archive.
//! 2. Extracts `word/document.xml`.
//! 3. Parses the XML and concatenates all `<w:t>` text nodes.
//! 4. Returns a single [`LoadedDocument`] with the full document text.
//!
//! No external dependencies beyond the standard library and `zip` — which is
//! already a transitive dependency in most Rust projects.
//!
//! ## Example
//!
//! ```rust,ignore
//! use flowgentra_ai::core::rag::loaders::DocxLoader;
//!
//! let doc = DocxLoader::new().load("report.docx").await?;
//! println!("{}", doc.text);
//! ```

use std::io::{Cursor, Read};
use std::path::Path;
use std::collections::HashMap;
use serde_json::json;

use crate::core::rag::{document_loader::LoadedDocument, vector_db::VectorStoreError};

/// Loader for `.docx` Microsoft Word documents.
pub struct DocxLoader {
    /// If true, preserve paragraph breaks as double newlines.
    pub preserve_paragraphs: bool,
}

impl Default for DocxLoader {
    fn default() -> Self {
        Self {
            preserve_paragraphs: true,
        }
    }
}

impl DocxLoader {
    pub fn new() -> Self {
        Self::default()
    }

    /// Load a `.docx` file and extract its text content.
    pub async fn load(&self, path: impl AsRef<Path>) -> Result<LoadedDocument, VectorStoreError> {
        let path = path.as_ref();
        let source = path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("unknown")
            .to_string();

        let bytes = tokio::fs::read(path)
            .await
            .map_err(|e| VectorStoreError::Unknown(format!("Read DOCX error: {e}")))?;

        let text = self.extract_text_from_bytes(&bytes, &source)?;

        let mut metadata = HashMap::new();
        metadata.insert("source".to_string(), json!(source));
        metadata.insert("file_type".to_string(), json!("docx"));

        Ok(LoadedDocument {
            id: source.clone(),
            text,
            source,
            file_type: crate::core::rag::document_loader::FileType::Unknown,
            metadata,
        })
    }

    /// Load from bytes (useful for in-memory documents or testing).
    pub fn load_from_bytes(
        &self,
        bytes: &[u8],
        name: &str,
    ) -> Result<LoadedDocument, VectorStoreError> {
        let text = self.extract_text_from_bytes(bytes, name)?;
        let mut metadata = HashMap::new();
        metadata.insert("source".to_string(), json!(name));
        metadata.insert("file_type".to_string(), json!("docx"));
        Ok(LoadedDocument {
            id: name.to_string(),
            text,
            source: name.to_string(),
            file_type: crate::core::rag::document_loader::FileType::Unknown,
            metadata,
        })
    }

    fn extract_text_from_bytes(
        &self,
        bytes: &[u8],
        source: &str,
    ) -> Result<String, VectorStoreError> {
        // Open as ZIP archive
        let cursor = Cursor::new(bytes);
        let mut archive = zip::ZipArchive::new(cursor).map_err(|e| {
            VectorStoreError::Unknown(format!("Not a valid DOCX (ZIP) file '{source}': {e}"))
        })?;

        // Extract word/document.xml
        let mut doc_xml_entry = archive
            .by_name("word/document.xml")
            .map_err(|_| {
                VectorStoreError::Unknown(format!(
                    "word/document.xml not found in '{source}'"
                ))
            })?;

        let mut xml_content = String::new();
        doc_xml_entry
            .read_to_string(&mut xml_content)
            .map_err(|e| VectorStoreError::Unknown(format!("Read XML error: {e}")))?;

        let text = extract_text_from_xml(&xml_content, self.preserve_paragraphs);
        Ok(text)
    }
}

/// Extract plain text from `word/document.xml`.
///
/// Concatenates all `<w:t>` text nodes. Paragraph breaks (`<w:p>`) are
/// converted to double newlines when `preserve_paragraphs` is true.
fn extract_text_from_xml(xml: &str, preserve_paragraphs: bool) -> String {
    let mut result = String::new();
    let mut i = 0;
    let chars: Vec<char> = xml.chars().collect();
    let n = chars.len();

    while i < n {
        if chars[i] == '<' {
            // Read the tag name
            let tag_start = i + 1;
            let mut j = tag_start;
            while j < n && chars[j] != '>' && chars[j] != ' ' {
                j += 1;
            }
            let tag_name: String = chars[tag_start..j].iter().collect();

            // Find the end of the tag
            while i < n && chars[i] != '>' {
                i += 1;
            }
            i += 1; // skip '>'

            if tag_name == "w:t" || tag_name == "w:delText" {
                // Collect text content until </w:t> or </w:delText>
                let close_tag = format!("</{tag_name}>");
                let rest: String = chars[i..].iter().collect();
                if let Some(end) = rest.find(&close_tag) {
                    let text = &rest[..end];
                    result.push_str(text);
                    i += end + close_tag.len();
                }
            } else if tag_name == "w:p" && preserve_paragraphs {
                // Paragraph break
                if !result.is_empty() && !result.ends_with('\n') {
                    result.push('\n');
                }
            } else if tag_name == "w:br" {
                result.push('\n');
            }
        } else {
            i += 1;
        }
    }

    // Clean up consecutive blank lines
    let lines: Vec<&str> = result.lines().collect();
    let mut cleaned = Vec::new();
    let mut prev_empty = false;
    for line in lines {
        let is_empty = line.trim().is_empty();
        if is_empty && prev_empty {
            continue;
        }
        cleaned.push(line);
        prev_empty = is_empty;
    }
    cleaned.join("\n").trim().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_text_from_xml() {
        let xml = r#"<?xml version="1.0"?>
<w:document>
  <w:body>
    <w:p>
      <w:r><w:t>Hello</w:t></w:r>
      <w:r><w:t xml:space="preserve"> world</w:t></w:r>
    </w:p>
    <w:p>
      <w:r><w:t>Second paragraph</w:t></w:r>
    </w:p>
  </w:body>
</w:document>"#;

        let text = extract_text_from_xml(xml, true);
        assert!(text.contains("Hello world") || (text.contains("Hello") && text.contains("world")));
        assert!(text.contains("Second paragraph"));
    }
}
