//! PDF Text Extraction
//!
//! Extracts text from PDF files and splits into chunks for indexing.
//! PDF extraction is offloaded to a blocking thread to avoid stalling
//! the async runtime.

use std::path::{Path, PathBuf};

use super::vector_db::VectorStoreError;

/// Result of extracting text from a PDF
#[derive(Debug, Clone)]
pub struct PdfDocument {
    /// Source file path
    pub source: String,
    /// Total number of pages
    pub page_count: usize,
    /// Full extracted text
    pub text: String,
}

/// Rough token count estimate (~4 chars per token for English)
pub fn estimate_tokens(text: &str) -> usize {
    // A common heuristic: 1 token ≈ 4 characters for English text.
    // CJK and code may differ, but this is a reasonable default.
    (text.len() + 3) / 4
}

/// Extract all text from a PDF file (async — offloads to blocking thread pool)
pub async fn extract_text(path: impl AsRef<Path>) -> Result<PdfDocument, VectorStoreError> {
    let path = path.as_ref().to_path_buf();

    tokio::task::spawn_blocking(move || extract_text_sync(&path))
        .await
        .map_err(|e| VectorStoreError::Unknown(format!("Task join error: {}", e)))?
}

/// Synchronous PDF text extraction (for use inside spawn_blocking)
fn extract_text_sync(path: &PathBuf) -> Result<PdfDocument, VectorStoreError> {
    let text = pdf_extract::extract_text(path).map_err(|e| {
        VectorStoreError::Unknown(format!(
            "Failed to extract text from '{}': {}",
            path.display(),
            e
        ))
    })?;

    // Estimate page count from form-feed characters, fallback to byte-based estimate
    let page_count = {
        let ff_count = text.chars().filter(|&c| c == '\x0C').count();
        if ff_count > 0 {
            ff_count + 1
        } else {
            (text.len() / 3000).max(1)
        }
    };

    Ok(PdfDocument {
        source: path.display().to_string(),
        page_count,
        text,
    })
}

/// Split text into overlapping chunks for better retrieval
///
/// # Arguments
/// * `text` - The full text to split
/// * `chunk_size` - Target size of each chunk in characters
/// * `overlap` - Number of overlapping characters between consecutive chunks
pub fn chunk_text(text: &str, chunk_size: usize, overlap: usize) -> Vec<String> {
    if text.is_empty() || chunk_size == 0 {
        return vec![];
    }

    let chars: Vec<char> = text.chars().collect();
    let total = chars.len();

    if total <= chunk_size {
        return vec![text.to_string()];
    }

    let step = chunk_size.saturating_sub(overlap).max(1);
    let mut chunks = Vec::new();
    let mut start = 0;

    while start < total {
        let end = (start + chunk_size).min(total);
        let chunk: String = chars[start..end].iter().collect();

        // Try to break at a sentence or paragraph boundary
        let trimmed = smart_break(&chunk);
        if !trimmed.trim().is_empty() {
            chunks.push(trimmed);
        }

        if end >= total {
            break;
        }
        start += step;
    }

    chunks
}

/// Split text into overlapping chunks with a target token budget
///
/// Uses a rough estimate of ~4 chars per token. More accurate than
/// pure character-based chunking for LLM context windows.
///
/// # Arguments
/// * `text` - The full text to split
/// * `max_tokens` - Target max tokens per chunk
/// * `overlap_tokens` - Overlap between chunks in tokens
pub fn chunk_text_by_tokens(text: &str, max_tokens: usize, overlap_tokens: usize) -> Vec<String> {
    let chars_per_token = 4;
    chunk_text(
        text,
        max_tokens * chars_per_token,
        overlap_tokens * chars_per_token,
    )
}

/// Try to break text at a natural boundary (paragraph or sentence end)
fn smart_break(text: &str) -> String {
    // Try paragraph break first
    if let Some(pos) = text.rfind("\n\n") {
        if pos > text.len() / 2 {
            return text[..pos].to_string();
        }
    }

    // Try sentence break
    for delim in &[". ", "! ", "? ", ".\n"] {
        if let Some(pos) = text.rfind(delim) {
            if pos > text.len() / 2 {
                return text[..pos + delim.len()].trim_end().to_string();
            }
        }
    }

    text.to_string()
}

/// Extract text from a PDF and split into chunks (async)
///
/// Returns (chunk_id, chunk_text) pairs.
///
/// # Arguments
/// * `path` - Path to the PDF file
/// * `chunk_size` - Target chunk size in characters
/// * `overlap` - Overlap between chunks in characters
pub async fn extract_and_chunk(
    path: impl AsRef<Path>,
    chunk_size: usize,
    overlap: usize,
) -> Result<Vec<(String, String)>, VectorStoreError> {
    let path_ref = path.as_ref();
    let file_stem = path_ref
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("doc")
        .to_string();

    let pdf = extract_text(path_ref).await?;
    let chunks = chunk_text(&pdf.text, chunk_size, overlap);

    Ok(chunks
        .into_iter()
        .enumerate()
        .map(|(i, text)| (format!("{}-chunk-{}", file_stem, i), text))
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chunk_text_small() {
        let text = "Hello world.";
        let chunks = chunk_text(text, 1000, 200);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0], "Hello world.");
    }

    #[test]
    fn test_chunk_text_splits() {
        let text = "A".repeat(2500);
        let chunks = chunk_text(&text, 1000, 200);
        assert!(chunks.len() >= 2);
    }

    #[test]
    fn test_chunk_text_empty() {
        let chunks = chunk_text("", 1000, 200);
        assert!(chunks.is_empty());
    }

    #[test]
    fn test_smart_break_sentence() {
        let text = "First sentence. Second sentence. Third sent";
        let result = smart_break(text);
        assert!(result.ends_with("Second sentence."));
    }

    #[test]
    fn test_estimate_tokens() {
        assert_eq!(estimate_tokens("hello world"), 3); // 11 chars / 4
        assert_eq!(estimate_tokens(""), 0);
        assert_eq!(estimate_tokens("abcd"), 1); // exactly 4 chars
    }

    #[test]
    fn test_chunk_by_tokens() {
        let text = "A".repeat(2000);
        // 250 tokens * 4 = 1000 chars, 50 tokens * 4 = 200 chars overlap
        let chunks = chunk_text_by_tokens(&text, 250, 50);
        assert!(chunks.len() >= 2);
    }
}
