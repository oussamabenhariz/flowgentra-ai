//! Recursive Directory Loader
//!
//! Loads documents from an entire directory tree, optionally filtered by file
//! extension glob patterns. Extends the flat `load_directory()` in
//! `document_loader.rs` with recursion and extension filtering.
//!
//! ## Example
//!
//! ```rust,ignore
//! use flowgentra_ai::core::rag::loaders::{DirectoryLoader, DirectoryLoaderConfig};
//!
//! // Load all .txt and .md files recursively
//! let loader = DirectoryLoader::new("./docs", DirectoryLoaderConfig {
//!     extensions: vec!["txt".into(), "md".into()],
//!     recursive: true,
//!     ..Default::default()
//! });
//! let docs = loader.load().await?;
//!
//! // Load all supported types in a flat directory
//! let docs = DirectoryLoader::default_for("./data").load().await?;
//! ```

use std::path::{Path, PathBuf};

use crate::core::rag::{
    document_loader::{load_document, FileType, LoadedDocument},
    loaders::{
        csv::CsvLoader,
        json_loader::{JsonLoader, JsonlLoader},
    },
    vector_db::VectorStoreError,
};

/// Configuration for the recursive directory loader.
#[derive(Debug, Clone)]
pub struct DirectoryLoaderConfig {
    /// File extensions to include (without leading dot). Empty = all supported.
    pub extensions: Vec<String>,
    /// Whether to recurse into subdirectories.
    pub recursive: bool,
    /// Maximum directory depth to recurse (0 = no limit).
    pub max_depth: usize,
    /// If true, skip files that fail to load instead of returning an error.
    pub skip_errors: bool,
    /// For JSON files: which field to use as document text.
    pub json_text_field: String,
    /// For JSONL files: which field to use as document text.
    pub jsonl_text_field: String,
}

impl Default for DirectoryLoaderConfig {
    fn default() -> Self {
        Self {
            extensions: vec![],
            recursive: true,
            max_depth: 0,
            skip_errors: true,
            json_text_field: "text".into(),
            jsonl_text_field: "text".into(),
        }
    }
}

/// Loads documents from a directory, with optional recursion and extension filtering.
pub struct DirectoryLoader {
    root: PathBuf,
    config: DirectoryLoaderConfig,
}

impl DirectoryLoader {
    pub fn new(root: impl AsRef<Path>, config: DirectoryLoaderConfig) -> Self {
        Self {
            root: root.as_ref().to_path_buf(),
            config,
        }
    }

    /// Create with default config (recursive, all supported types, skip errors).
    pub fn default_for(root: impl AsRef<Path>) -> Self {
        Self::new(root, DirectoryLoaderConfig::default())
    }

    /// Load all matching documents from the directory.
    pub async fn load(&self) -> Result<Vec<LoadedDocument>, VectorStoreError> {
        let mut docs = Vec::new();
        self.collect_docs(&self.root, 0, &mut docs).await?;
        Ok(docs)
    }

    async fn collect_docs(
        &self,
        dir: &Path,
        depth: usize,
        docs: &mut Vec<LoadedDocument>,
    ) -> Result<(), VectorStoreError> {
        // Stop if max_depth is set and exceeded
        if self.config.max_depth > 0 && depth >= self.config.max_depth {
            return Ok(());
        }

        let mut entries = tokio::fs::read_dir(dir)
            .await
            .map_err(|e| VectorStoreError::Unknown(format!("Read dir error: {e}")))?;

        while let Some(entry) = entries
            .next_entry()
            .await
            .map_err(|e| VectorStoreError::Unknown(format!("Read entry error: {e}")))?
        {
            let path = entry.path();

            if path.is_dir() {
                if self.config.recursive {
                    Box::pin(self.collect_docs(&path, depth + 1, docs)).await?;
                }
                continue;
            }

            if !path.is_file() {
                continue;
            }

            let ext = path
                .extension()
                .and_then(|e| e.to_str())
                .unwrap_or("")
                .to_string();

            // Extension filter
            if !self.config.extensions.is_empty()
                && !self.config.extensions.iter().any(|e| e == &ext)
            {
                continue;
            }

            // Load by extension
            let result = self.load_file(&path, &ext).await;
            match result {
                Ok(mut loaded) => docs.append(&mut loaded),
                Err(e) => {
                    if self.config.skip_errors {
                        tracing::warn!("Skipping {:?}: {}", path, e);
                    } else {
                        return Err(e);
                    }
                }
            }
        }

        Ok(())
    }

    async fn load_file(
        &self,
        path: &Path,
        ext: &str,
    ) -> Result<Vec<LoadedDocument>, VectorStoreError> {
        match ext {
            "json" => {
                let loader = JsonLoader::new(self.config.json_text_field.clone());
                loader
                    .load(path)
                    .await
                    .map_err(|e| VectorStoreError::Unknown(format!("{}: {}", path.display(), e)))
            }
            "jsonl" | "ndjson" => {
                let loader = JsonlLoader::new(self.config.jsonl_text_field.clone());
                loader
                    .load(path)
                    .await
                    .map_err(|e| VectorStoreError::Unknown(format!("{}: {}", path.display(), e)))
            }
            "csv" => {
                let loader = CsvLoader::new();
                loader
                    .load(path)
                    .await
                    .map_err(|e| VectorStoreError::Unknown(format!("{}: {}", path.display(), e)))
            }
            "txt" | "md" | "markdown" | "html" | "htm" | "pdf" => {
                let file_type = FileType::from_path(path);
                if file_type == FileType::Unknown {
                    return Ok(vec![]); // skip unknown types
                }
                let doc = load_document(path)
                    .await
                    .map_err(|e| VectorStoreError::Unknown(format!("{}: {}", path.display(), e)))?;
                Ok(vec![doc])
            }
            _ => Ok(vec![]), // silently skip unsupported extensions
        }
    }
}

// We need to add `metadata` field to `LoadedDocument` for the loaders.
// Since it doesn't exist yet, we'll work around it in the document_loader module.
// The loaders that return LoadedDocument work with the existing struct.

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_directory_loader_flat() {
        // Create a temp dir with some files
        let dir = tempfile::tempdir().unwrap();
        let txt_path = dir.path().join("hello.txt");
        tokio::fs::write(&txt_path, "Hello from text file")
            .await
            .unwrap();

        let loader = DirectoryLoader::default_for(dir.path());
        let docs = loader.load().await.unwrap();
        assert!(!docs.is_empty());
        assert!(docs.iter().any(|d| d.text.contains("Hello")));
    }

    #[tokio::test]
    async fn test_directory_loader_extension_filter() {
        let dir = tempfile::tempdir().unwrap();
        tokio::fs::write(dir.path().join("a.txt"), "text file")
            .await
            .unwrap();
        tokio::fs::write(dir.path().join("b.md"), "# Markdown")
            .await
            .unwrap();

        let loader = DirectoryLoader::new(
            dir.path(),
            DirectoryLoaderConfig {
                extensions: vec!["txt".into()],
                ..Default::default()
            },
        );
        let docs = loader.load().await.unwrap();
        assert_eq!(docs.len(), 1);
        assert!(docs[0].text.contains("text file"));
    }
}
