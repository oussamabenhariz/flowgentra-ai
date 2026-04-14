//! Document Loaders — load and chunk different file types
//!
//! Supports: `.txt`, `.md`, `.html`, `.pdf`
//! Automatically detects the type from the file extension and applies
//! appropriate preprocessing (e.g. stripping HTML tags).

use std::collections::HashMap;
use std::path::Path;

use super::pdf;
use super::vector_db::VectorStoreError;

/// A loaded document with its text and metadata
#[derive(Debug, Clone)]
pub struct LoadedDocument {
    pub id: String,
    pub text: String,
    pub source: String,
    pub file_type: FileType,
    /// Optional key/value metadata (source URL, file path, page number, etc.)
    pub metadata: HashMap<String, serde_json::Value>,
}

impl LoadedDocument {
    /// Convert to a `vector_db::Document` for indexing. Embedding must be
    /// computed separately via an `Embeddings` provider.
    pub fn into_document(self) -> super::vector_db::Document {
        super::vector_db::Document {
            id: self.id,
            text: self.text,
            embedding: None,
            metadata: self.metadata,
        }
    }
}

/// Supported file types
#[derive(Debug, Clone, PartialEq)]
pub enum FileType {
    PlainText,
    Markdown,
    Html,
    Pdf,
    Unknown,
}

impl FileType {
    /// Detect file type from extension
    pub fn from_path(path: &Path) -> Self {
        match path.extension().and_then(|e| e.to_str()) {
            Some("txt") => Self::PlainText,
            Some("md" | "markdown") => Self::Markdown,
            Some("html" | "htm") => Self::Html,
            Some("pdf") => Self::Pdf,
            _ => Self::Unknown,
        }
    }
}

/// Strip HTML tags from text, including the contents of `<script>` and `<style>` blocks.
fn strip_html_tags(html: &str) -> String {
    let mut result = String::with_capacity(html.len());
    let mut in_tag = false;
    let mut in_skip_block = false; // true while inside <script> or <style>
    let mut tag_buf = String::new(); // accumulates tag content for name detection

    for ch in html.chars() {
        match ch {
            '<' => {
                in_tag = true;
                tag_buf.clear();
            }
            '>' => {
                in_tag = false;
                let tag_lower = tag_buf.trim().to_lowercase();
                // Opening <script> or <style> — start skipping content
                if tag_lower.starts_with("script") || tag_lower.starts_with("style") {
                    in_skip_block = true;
                }
                // Closing </script> or </style> — resume collecting content
                if tag_lower.starts_with("/script") || tag_lower.starts_with("/style") {
                    in_skip_block = false;
                }
            }
            _ if in_tag => {
                tag_buf.push(ch);
            }
            _ if !in_skip_block => {
                result.push(ch);
            }
            _ => {}
        }
    }

    // Decode common HTML entities
    result
        .replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", "\"")
        .replace("&#39;", "'")
        .replace("&nbsp;", " ")
}

/// Strip markdown formatting (headers, bold, italic, links, code blocks)
fn strip_markdown(md: &str) -> String {
    let mut lines: Vec<String> = Vec::new();
    let mut in_code_block = false;

    for line in md.lines() {
        if line.trim_start().starts_with("```") {
            in_code_block = !in_code_block;
            continue;
        }
        if in_code_block {
            lines.push(line.to_string());
            continue;
        }

        let mut cleaned = line.to_string();
        // Strip header markers
        cleaned = cleaned.trim_start_matches('#').to_string();
        // Strip bold/italic
        cleaned = cleaned.replace("**", "").replace("__", "");
        cleaned = cleaned.replace(['*', '_'], "");
        // Strip inline code
        cleaned = cleaned.replace('`', "");
        // Strip link syntax [text](url) → text
        while let Some(start) = cleaned.find('[') {
            if let Some(mid) = cleaned[start..].find("](") {
                if let Some(end) = cleaned[start + mid..].find(')') {
                    let text = &cleaned[start + 1..start + mid].to_string();
                    let before = &cleaned[..start];
                    let after = &cleaned[start + mid + end + 1..];
                    cleaned = format!("{}{}{}", before, text, after);
                    continue;
                }
            }
            break;
        }

        lines.push(cleaned);
    }

    lines.join("\n")
}

/// Load a document from a file path.
///
/// Automatically detects the file type and applies appropriate preprocessing.
pub async fn load_document(path: impl AsRef<Path>) -> Result<LoadedDocument, VectorStoreError> {
    let path = path.as_ref();
    let file_type = FileType::from_path(path);
    let source = path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("unknown")
        .to_string();
    let id = source.clone();

    let text = match file_type {
        FileType::Pdf => {
            let pdf_doc = pdf::extract_text(path).await?;
            pdf_doc.text
        }
        FileType::Html => {
            let raw = tokio::fs::read_to_string(path)
                .await
                .map_err(|e| VectorStoreError::Unknown(format!("Failed to read file: {}", e)))?;
            strip_html_tags(&raw)
        }
        FileType::Markdown => {
            let raw = tokio::fs::read_to_string(path)
                .await
                .map_err(|e| VectorStoreError::Unknown(format!("Failed to read file: {}", e)))?;
            strip_markdown(&raw)
        }
        FileType::PlainText | FileType::Unknown => tokio::fs::read_to_string(path)
            .await
            .map_err(|e| VectorStoreError::Unknown(format!("Failed to read file: {}", e)))?,
    };

    let mut metadata = HashMap::new();
    metadata.insert(
        "source".to_string(),
        serde_json::Value::String(source.clone()),
    );

    Ok(LoadedDocument {
        id,
        text,
        source,
        file_type,
        metadata,
    })
}

/// Load all supported documents from a directory (non-recursive)
pub async fn load_directory(
    dir: impl AsRef<Path>,
) -> Result<Vec<LoadedDocument>, VectorStoreError> {
    let dir = dir.as_ref();
    let mut docs = Vec::new();

    let mut entries = tokio::fs::read_dir(dir)
        .await
        .map_err(|e| VectorStoreError::Unknown(format!("Failed to read directory: {}", e)))?;

    while let Some(entry) = entries
        .next_entry()
        .await
        .map_err(|e| VectorStoreError::Unknown(format!("Failed to read entry: {}", e)))?
    {
        let path = entry.path();
        if !path.is_file() {
            continue;
        }

        let file_type = FileType::from_path(&path);
        if file_type == FileType::Unknown {
            continue;
        }

        match load_document(&path).await {
            Ok(doc) => docs.push(doc),
            Err(e) => {
                tracing::warn!("Skipping {:?}: {}", path, e);
            }
        }
    }

    Ok(docs)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_file_type_detection() {
        assert_eq!(
            FileType::from_path(Path::new("doc.txt")),
            FileType::PlainText
        );
        assert_eq!(
            FileType::from_path(Path::new("readme.md")),
            FileType::Markdown
        );
        assert_eq!(FileType::from_path(Path::new("page.html")), FileType::Html);
        assert_eq!(FileType::from_path(Path::new("report.pdf")), FileType::Pdf);
        assert_eq!(
            FileType::from_path(Path::new("data.csv")),
            FileType::Unknown
        );
    }

    #[test]
    fn test_strip_html() {
        let html = "<h1>Title</h1><p>Hello &amp; world</p>";
        let stripped = strip_html_tags(html);
        assert_eq!(stripped, "TitleHello & world");
    }

    #[test]
    fn test_strip_html_skips_script_content() {
        let html = "<p>Before</p><script>alert('xss')</script><p>After</p>";
        let stripped = strip_html_tags(html);
        assert!(stripped.contains("Before"));
        assert!(stripped.contains("After"));
        assert!(!stripped.contains("alert"));
        assert!(!stripped.contains("xss"));
    }

    #[test]
    fn test_strip_html_skips_style_content() {
        let html = "<p>Text</p><style>body { color: red; }</style><p>More</p>";
        let stripped = strip_html_tags(html);
        assert!(stripped.contains("Text"));
        assert!(stripped.contains("More"));
        assert!(!stripped.contains("color"));
    }

    #[test]
    fn test_strip_markdown() {
        let md = "# Header\n**bold** and *italic*\n`code`\n[link](http://example.com)";
        let stripped = strip_markdown(md);
        assert!(stripped.contains("Header"));
        assert!(stripped.contains("bold and italic"));
        assert!(stripped.contains("code"));
        assert!(stripped.contains("link"));
        assert!(!stripped.contains("http://"));
    }

    #[test]
    fn test_strip_markdown_code_block() {
        let md = "text\n```\ncode block\n```\nmore text";
        let stripped = strip_markdown(md);
        assert!(stripped.contains("code block"));
        assert!(stripped.contains("text"));
    }
}
