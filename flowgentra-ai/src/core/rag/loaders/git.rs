//! Git Repository Loader
//!
//! Loads source files from a local git repository. Filters by file extension
//! and optionally restricts to files changed since a given commit.
//!
//! Each file becomes one [`LoadedDocument`] with the file contents as text
//! and git metadata (branch, last commit hash, author) in the metadata field.

use crate::core::rag::document_loader::{FileType, LoadedDocument};
use serde_json::json;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone)]
pub struct GitLoaderConfig {
    /// Git repository root path.
    pub repo_path: String,
    /// File extensions to include (e.g. `["rs", "py", "md"]`). Empty = all.
    pub extensions: Vec<String>,
    /// Maximum file size in bytes (default 1 MB).
    pub max_file_size: usize,
    /// Branch name to inspect (default: current checked-out branch).
    pub branch: Option<String>,
    /// Load only files changed since this commit hash.
    pub since_commit: Option<String>,
}

impl Default for GitLoaderConfig {
    fn default() -> Self {
        Self {
            repo_path: ".".to_string(),
            extensions: vec![],
            max_file_size: 1_048_576,
            branch: None,
            since_commit: None,
        }
    }
}

pub struct GitLoader {
    config: GitLoaderConfig,
}

impl GitLoader {
    pub fn new(repo_path: impl Into<String>) -> Self {
        Self {
            config: GitLoaderConfig {
                repo_path: repo_path.into(),
                ..Default::default()
            },
        }
    }

    pub fn with_config(mut self, config: GitLoaderConfig) -> Self {
        self.config = config;
        self
    }

    pub fn with_extensions(mut self, exts: Vec<String>) -> Self {
        self.config.extensions = exts;
        self
    }

    pub fn load(&self) -> Result<Vec<LoadedDocument>, Box<dyn std::error::Error>> {
        let root = Path::new(&self.config.repo_path);

        // Get list of files to load
        let files = if let Some(since) = &self.config.since_commit {
            self.files_changed_since(root, since)?
        } else {
            self.all_tracked_files(root)?
        };

        let mut docs = Vec::new();
        for file_path in files {
            if let Some(doc) = self.load_file(root, &file_path)? {
                docs.push(doc);
            }
        }
        Ok(docs)
    }

    fn all_tracked_files(&self, root: &Path) -> Result<Vec<PathBuf>, Box<dyn std::error::Error>> {
        let output = std::process::Command::new("git")
            .args(["ls-files"])
            .current_dir(root)
            .output()?;
        let stdout = String::from_utf8_lossy(&output.stdout);
        let files = stdout
            .lines()
            .filter(|f| self.filter_extension(f))
            .map(PathBuf::from)
            .collect();
        Ok(files)
    }

    fn files_changed_since(
        &self,
        root: &Path,
        since: &str,
    ) -> Result<Vec<PathBuf>, Box<dyn std::error::Error>> {
        let output = std::process::Command::new("git")
            .args(["diff", "--name-only", since, "HEAD"])
            .current_dir(root)
            .output()?;
        let stdout = String::from_utf8_lossy(&output.stdout);
        let files = stdout
            .lines()
            .filter(|f| self.filter_extension(f))
            .map(PathBuf::from)
            .collect();
        Ok(files)
    }

    fn filter_extension(&self, path: &str) -> bool {
        if self.config.extensions.is_empty() {
            return true;
        }
        let ext = Path::new(path)
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("");
        self.config.extensions.iter().any(|e| e == ext)
    }

    fn load_file(
        &self,
        root: &Path,
        relative: &Path,
    ) -> Result<Option<LoadedDocument>, Box<dyn std::error::Error>> {
        let full = root.join(relative);
        if !full.exists() {
            return Ok(None);
        }
        let metadata_fs = std::fs::metadata(&full)?;
        if metadata_fs.len() as usize > self.config.max_file_size {
            return Ok(None);
        }
        let text = std::fs::read_to_string(&full).unwrap_or_else(|_| String::new());
        if text.trim().is_empty() {
            return Ok(None);
        }

        // Get git metadata for this file
        let commit = self.git_last_commit(root, relative).unwrap_or_default();
        let author = self.git_last_author(root, relative).unwrap_or_default();

        let path_str = relative.to_string_lossy().to_string();
        let mut metadata = HashMap::new();
        metadata.insert("source".to_string(), json!(path_str));
        metadata.insert("repo".to_string(), json!(self.config.repo_path));
        metadata.insert("last_commit".to_string(), json!(commit));
        metadata.insert("last_author".to_string(), json!(author));
        metadata.insert(
            "extension".to_string(),
            json!(relative.extension().and_then(|e| e.to_str()).unwrap_or("")),
        );

        let ext = relative.extension().and_then(|e| e.to_str()).unwrap_or("");
        let file_type = match ext {
            "html" | "htm" => FileType::Html,
            "pdf" => FileType::Pdf,
            _ => FileType::PlainText,
        };

        Ok(Some(LoadedDocument {
            id: format!("git_{path_str}"),
            text,
            source: format!("git://{}#{}", self.config.repo_path, path_str),
            file_type,
            metadata,
        }))
    }

    fn git_last_commit(
        &self,
        root: &Path,
        file: &Path,
    ) -> Result<String, Box<dyn std::error::Error>> {
        let out = std::process::Command::new("git")
            .args([
                "log",
                "-1",
                "--format=%H",
                "--",
                file.to_str().unwrap_or(""),
            ])
            .current_dir(root)
            .output()?;
        Ok(String::from_utf8_lossy(&out.stdout).trim().to_string())
    }

    fn git_last_author(
        &self,
        root: &Path,
        file: &Path,
    ) -> Result<String, Box<dyn std::error::Error>> {
        let out = std::process::Command::new("git")
            .args([
                "log",
                "-1",
                "--format=%an",
                "--",
                file.to_str().unwrap_or(""),
            ])
            .current_dir(root)
            .output()?;
        Ok(String::from_utf8_lossy(&out.stdout).trim().to_string())
    }
}
