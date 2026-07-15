//! Filesystem helpers for durable, safe persistence.

use std::io;
use std::path::{Component, Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

static TMP_COUNTER: AtomicU64 = AtomicU64::new(0);

fn tmp_sibling(path: &Path) -> PathBuf {
    let n = TMP_COUNTER.fetch_add(1, Ordering::Relaxed);
    let mut name = path
        .file_name()
        .map(|f| f.to_string_lossy().into_owned())
        .unwrap_or_else(|| "file".to_string());
    name.push_str(&format!(".{}-{}.tmp", std::process::id(), n));
    path.with_file_name(name)
}

/// Replace the destination with the temp file, tolerating Windows' refusal to
/// rename over an existing file.
fn replace_file(tmp: &Path, path: &Path) -> io::Result<()> {
    match std::fs::rename(tmp, path) {
        Ok(()) => Ok(()),
        Err(_) if path.exists() => {
            // Windows: rename fails if the destination exists. Remove and retry.
            // The destination briefly disappears, but it is never half-written.
            std::fs::remove_file(path)?;
            std::fs::rename(tmp, path)
        }
        Err(e) => Err(e),
    }
}

/// Write `contents` to `path` atomically: write to a unique temp file in the
/// same directory, flush, then rename into place. A crash mid-write leaves the
/// old file intact instead of a truncated one.
pub fn atomic_write(path: &Path, contents: &[u8]) -> io::Result<()> {
    let tmp = tmp_sibling(path);
    let result = (|| {
        std::fs::write(&tmp, contents)?;
        replace_file(&tmp, path)
    })();
    if result.is_err() {
        let _ = std::fs::remove_file(&tmp);
    }
    result
}

/// Async variant of [`atomic_write`].
pub async fn atomic_write_async(path: &Path, contents: &[u8]) -> io::Result<()> {
    let tmp = tmp_sibling(path);
    let result = async {
        tokio::fs::write(&tmp, contents).await?;
        replace_file(&tmp, path)
    }
    .await;
    if result.is_err() {
        let _ = tokio::fs::remove_file(&tmp).await;
    }
    result
}

/// `true` if `component` is safe to use as a single path component under a
/// storage root: non-empty, no path separators, no `..`, no drive/root parts.
pub fn is_safe_path_component(component: &str) -> bool {
    if component.is_empty() {
        return false;
    }
    let path = Path::new(component);
    let mut parts = path.components();
    matches!(
        (parts.next(), parts.next()),
        (Some(Component::Normal(_)), None)
    ) && !component.contains(['/', '\\'])
        && component != ".."
        && component != "."
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn atomic_write_creates_and_replaces() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("data.json");
        atomic_write(&path, b"first").unwrap();
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "first");
        atomic_write(&path, b"second").unwrap();
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "second");
        // No temp files left behind
        let leftovers: Vec<_> = std::fs::read_dir(dir.path())
            .unwrap()
            .filter_map(|e| e.ok())
            .filter(|e| e.file_name().to_string_lossy().ends_with(".tmp"))
            .collect();
        assert!(leftovers.is_empty());
    }

    #[test]
    fn safe_path_components() {
        assert!(is_safe_path_component("thread-1"));
        assert!(is_safe_path_component("user_42"));
        assert!(!is_safe_path_component(""));
        assert!(!is_safe_path_component(".."));
        assert!(!is_safe_path_component("../etc"));
        assert!(!is_safe_path_component("a/b"));
        assert!(!is_safe_path_component("a\\b"));
        assert!(!is_safe_path_component("C:\\x"));
    }
}
