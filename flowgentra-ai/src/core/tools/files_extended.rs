//! Extended sandboxed file tools: copy, delete, move, and content search.

use super::{JsonSchema, Tool, ToolDefinition};
use crate::prelude::*;
use async_trait::async_trait;
use regex::Regex;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

// =============================================================================
// Shared sandbox helpers
// =============================================================================

fn sandbox_validate(sandbox_root: &Path, user_path: &str) -> Result<PathBuf> {
    let p = Path::new(user_path);
    for component in p.components() {
        if component == std::path::Component::ParentDir {
            return Err(FlowgentraError::ToolError(
                "Path traversal via '..' is not allowed".to_string(),
            ));
        }
    }
    let joined = if p.is_absolute() {
        p.to_path_buf()
    } else {
        sandbox_root.join(p)
    };
    let canonical = std::fs::canonicalize(&joined).map_err(|e| {
        FlowgentraError::ToolError(format!("Path '{}' is invalid: {}", user_path, e))
    })?;
    if !canonical.starts_with(sandbox_root) {
        return Err(FlowgentraError::ToolError(format!(
            "Path '{}' resolves outside the allowed sandbox directory",
            user_path
        )));
    }
    Ok(canonical)
}

/// Like `sandbox_validate` but allows the final path component to not exist yet
/// (useful for write/copy destinations).
fn sandbox_validate_new(sandbox_root: &Path, user_path: &str) -> Result<PathBuf> {
    let p = Path::new(user_path);
    for component in p.components() {
        if component == std::path::Component::ParentDir {
            return Err(FlowgentraError::ToolError(
                "Path traversal via '..' is not allowed".to_string(),
            ));
        }
    }
    let joined = if p.is_absolute() {
        p.to_path_buf()
    } else {
        sandbox_root.join(p)
    };
    let parent = joined.parent().ok_or_else(|| {
        FlowgentraError::ToolError(format!("Path '{}' has no parent directory", user_path))
    })?;
    let canonical_parent = std::fs::canonicalize(parent).map_err(|e| {
        FlowgentraError::ToolError(format!(
            "Parent directory of '{}' is invalid: {}",
            user_path, e
        ))
    })?;
    if !canonical_parent.starts_with(sandbox_root) {
        return Err(FlowgentraError::ToolError(format!(
            "Path '{}' resolves outside the allowed sandbox directory",
            user_path
        )));
    }
    let file_name = joined.file_name().ok_or_else(|| {
        FlowgentraError::ToolError(format!("Path '{}' has no file name", user_path))
    })?;
    Ok(canonical_parent.join(file_name))
}

fn default_sandbox() -> PathBuf {
    std::fs::canonicalize(std::env::current_dir().expect("cwd")).expect("canonicalize cwd")
}

// =============================================================================
// CopyFileTool
// =============================================================================

pub struct CopyFileTool {
    sandbox_root: PathBuf,
}

impl CopyFileTool {
    pub fn new(sandbox_root: impl AsRef<Path>) -> Result<Self> {
        Ok(Self {
            sandbox_root: std::fs::canonicalize(sandbox_root)
                .map_err(|e| FlowgentraError::ToolError(format!("Invalid sandbox root: {}", e)))?,
        })
    }
}

impl Default for CopyFileTool {
    fn default() -> Self {
        Self {
            sandbox_root: default_sandbox(),
        }
    }
}

#[async_trait]
impl Tool for CopyFileTool {
    async fn call(&self, input: Value) -> Result<Value> {
        let src = input
            .get("src")
            .and_then(|v| v.as_str())
            .ok_or_else(|| FlowgentraError::ToolError("Missing 'src' field".to_string()))?;
        let dst = input
            .get("dst")
            .and_then(|v| v.as_str())
            .ok_or_else(|| FlowgentraError::ToolError("Missing 'dst' field".to_string()))?;

        let safe_src = sandbox_validate(&self.sandbox_root, src)?;
        let safe_dst = sandbox_validate_new(&self.sandbox_root, dst)?;

        tokio::fs::copy(&safe_src, &safe_dst)
            .await
            .map_err(|e| FlowgentraError::ToolError(format!("Copy failed: {}", e)))?;

        Ok(json!({"src": src, "dst": dst, "success": true}))
    }

    fn definition(&self) -> ToolDefinition {
        let mut props = HashMap::new();
        props.insert(
            "src".to_string(),
            JsonSchema::string().with_description("Source file path"),
        );
        props.insert(
            "dst".to_string(),
            JsonSchema::string().with_description("Destination file path"),
        );

        ToolDefinition::new(
            "copy_file",
            "Copy a file within the sandbox directory",
            JsonSchema::object()
                .with_properties(props)
                .with_required(vec!["src".to_string(), "dst".to_string()]),
            JsonSchema::object(),
        )
        .with_category("file")
    }
}

// =============================================================================
// DeleteFileTool
// =============================================================================

pub struct DeleteFileTool {
    sandbox_root: PathBuf,
}

impl DeleteFileTool {
    pub fn new(sandbox_root: impl AsRef<Path>) -> Result<Self> {
        Ok(Self {
            sandbox_root: std::fs::canonicalize(sandbox_root)
                .map_err(|e| FlowgentraError::ToolError(format!("Invalid sandbox root: {}", e)))?,
        })
    }
}

impl Default for DeleteFileTool {
    fn default() -> Self {
        Self {
            sandbox_root: default_sandbox(),
        }
    }
}

#[async_trait]
impl Tool for DeleteFileTool {
    async fn call(&self, input: Value) -> Result<Value> {
        let path = input
            .get("path")
            .and_then(|v| v.as_str())
            .ok_or_else(|| FlowgentraError::ToolError("Missing 'path' field".to_string()))?;
        let recursive = input
            .get("recursive")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        let safe = sandbox_validate(&self.sandbox_root, path)?;

        if safe.is_dir() {
            if recursive {
                tokio::fs::remove_dir_all(&safe)
                    .await
                    .map_err(|e| FlowgentraError::ToolError(format!("Delete failed: {}", e)))?;
            } else {
                tokio::fs::remove_dir(&safe).await.map_err(|e| {
                    FlowgentraError::ToolError(format!(
                        "Delete failed (use recursive=true for non-empty dirs): {}",
                        e
                    ))
                })?;
            }
        } else {
            tokio::fs::remove_file(&safe)
                .await
                .map_err(|e| FlowgentraError::ToolError(format!("Delete failed: {}", e)))?;
        }

        Ok(json!({"path": path, "success": true}))
    }

    fn definition(&self) -> ToolDefinition {
        let mut props = HashMap::new();
        props.insert(
            "path".to_string(),
            JsonSchema::string().with_description("Path to delete"),
        );
        props.insert(
            "recursive".to_string(),
            JsonSchema::boolean()
                .with_description("Delete directories recursively (default: false)"),
        );

        ToolDefinition::new(
            "delete_file",
            "Delete a file or directory within the sandbox",
            JsonSchema::object()
                .with_properties(props)
                .with_required(vec!["path".to_string()]),
            JsonSchema::object(),
        )
        .with_category("file")
    }
}

// =============================================================================
// MoveFileTool
// =============================================================================

pub struct MoveFileTool {
    sandbox_root: PathBuf,
}

impl MoveFileTool {
    pub fn new(sandbox_root: impl AsRef<Path>) -> Result<Self> {
        Ok(Self {
            sandbox_root: std::fs::canonicalize(sandbox_root)
                .map_err(|e| FlowgentraError::ToolError(format!("Invalid sandbox root: {}", e)))?,
        })
    }
}

impl Default for MoveFileTool {
    fn default() -> Self {
        Self {
            sandbox_root: default_sandbox(),
        }
    }
}

#[async_trait]
impl Tool for MoveFileTool {
    async fn call(&self, input: Value) -> Result<Value> {
        let src = input
            .get("src")
            .and_then(|v| v.as_str())
            .ok_or_else(|| FlowgentraError::ToolError("Missing 'src' field".to_string()))?;
        let dst = input
            .get("dst")
            .and_then(|v| v.as_str())
            .ok_or_else(|| FlowgentraError::ToolError("Missing 'dst' field".to_string()))?;

        let safe_src = sandbox_validate(&self.sandbox_root, src)?;
        let safe_dst = sandbox_validate_new(&self.sandbox_root, dst)?;

        tokio::fs::rename(&safe_src, &safe_dst)
            .await
            .map_err(|e| FlowgentraError::ToolError(format!("Move failed: {}", e)))?;

        Ok(json!({"src": src, "dst": dst, "success": true}))
    }

    fn definition(&self) -> ToolDefinition {
        let mut props = HashMap::new();
        props.insert(
            "src".to_string(),
            JsonSchema::string().with_description("Source path"),
        );
        props.insert(
            "dst".to_string(),
            JsonSchema::string().with_description("Destination path"),
        );

        ToolDefinition::new(
            "move_file",
            "Move or rename a file within the sandbox directory",
            JsonSchema::object()
                .with_properties(props)
                .with_required(vec!["src".to_string(), "dst".to_string()]),
            JsonSchema::object(),
        )
        .with_category("file")
    }
}

// =============================================================================
// FileSearchTool
// =============================================================================

pub struct FileSearchTool {
    sandbox_root: PathBuf,
}

impl FileSearchTool {
    pub fn new(sandbox_root: impl AsRef<Path>) -> Result<Self> {
        Ok(Self {
            sandbox_root: std::fs::canonicalize(sandbox_root)
                .map_err(|e| FlowgentraError::ToolError(format!("Invalid sandbox root: {}", e)))?,
        })
    }
}

impl Default for FileSearchTool {
    fn default() -> Self {
        Self {
            sandbox_root: default_sandbox(),
        }
    }
}

#[derive(serde::Serialize)]
struct FileMatch {
    file: String,
    line_number: usize,
    line_content: String,
}

fn walk_and_search(
    dir: &Path,
    pattern: &Regex,
    _sandbox_root: &Path,
    max_results: usize,
    results: &mut Vec<FileMatch>,
) {
    if results.len() >= max_results {
        return;
    }
    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(_) => return,
    };
    for entry in entries.flatten() {
        if results.len() >= max_results {
            break;
        }
        let path = entry.path();
        if path.is_dir() {
            walk_and_search(&path, pattern, _sandbox_root, max_results, results);
        } else if path.is_file() {
            let Ok(content) = std::fs::read_to_string(&path) else {
                continue;
            };
            for (i, line) in content.lines().enumerate() {
                if results.len() >= max_results {
                    break;
                }
                if pattern.is_match(line) {
                    results.push(FileMatch {
                        file: path.to_string_lossy().to_string(),
                        line_number: i + 1,
                        line_content: line.to_string(),
                    });
                }
            }
        }
    }
}

#[async_trait]
impl Tool for FileSearchTool {
    async fn call(&self, input: Value) -> Result<Value> {
        let directory = input
            .get("directory")
            .and_then(|v| v.as_str())
            .unwrap_or(".");

        let pattern_str = input
            .get("pattern")
            .and_then(|v| v.as_str())
            .ok_or_else(|| FlowgentraError::ToolError("Missing 'pattern' field".to_string()))?
            .to_string();

        let max_results = input
            .get("max_results")
            .and_then(|v| v.as_u64())
            .unwrap_or(50) as usize;

        let safe_dir = sandbox_validate(&self.sandbox_root, directory)?;
        let sandbox_root = self.sandbox_root.clone();

        let results = tokio::task::spawn_blocking(move || {
            let pattern = Regex::new(&pattern_str)
                .map_err(|e| FlowgentraError::ToolError(format!("Invalid regex: {}", e)))?;
            let mut matches: Vec<FileMatch> = Vec::new();
            walk_and_search(
                &safe_dir,
                &pattern,
                &sandbox_root,
                max_results,
                &mut matches,
            );
            Ok::<Vec<FileMatch>, FlowgentraError>(matches)
        })
        .await
        .map_err(|e| FlowgentraError::ToolError(format!("spawn_blocking error: {}", e)))??;

        let count = results.len();
        let results_val = serde_json::to_value(results).unwrap_or(Value::Array(vec![]));
        Ok(json!({"matches": results_val, "count": count}))
    }

    fn definition(&self) -> ToolDefinition {
        let mut props = HashMap::new();
        props.insert(
            "directory".to_string(),
            JsonSchema::string().with_description("Directory to search in (default: current dir)"),
        );
        props.insert(
            "pattern".to_string(),
            JsonSchema::string().with_description("Regex pattern to match against file lines"),
        );
        props.insert(
            "max_results".to_string(),
            JsonSchema::integer().with_description("Maximum matches to return (default: 50)"),
        );

        ToolDefinition::new(
            "file_search",
            "Search file contents recursively for lines matching a regex pattern",
            JsonSchema::object()
                .with_properties(props)
                .with_required(vec!["pattern".to_string()]),
            JsonSchema::object(),
        )
        .with_category("file")
    }
}
