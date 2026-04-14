//! Web / URL Loader
//!
//! Fetches a web page from an HTTP/HTTPS URL, strips HTML tags, and returns a
//! [`LoadedDocument`]. Uses the `reqwest` client already present in the dependency
//! graph — no extra crates needed.
//!
//! ## Example
//!
//! ```rust,ignore
//! use flowgentra_ai::core::rag::loaders::WebLoader;
//!
//! let loader = WebLoader::new();
//! let doc = loader.load("https://www.rust-lang.org").await?;
//!
//! // Load multiple URLs in one call
//! let docs = loader.load_many(vec![
//!     "https://www.rust-lang.org",
//!     "https://docs.rs",
//! ]).await?;
//! ```

use std::collections::HashMap;
use serde_json::json;

use crate::core::rag::{document_loader::LoadedDocument, vector_db::VectorStoreError};

/// Configuration for the web loader.
#[derive(Debug, Clone)]
pub struct WebLoaderConfig {
    /// Maximum redirects to follow.
    pub max_redirects: usize,
    /// Request timeout in seconds.
    pub timeout_secs: u64,
    /// User-Agent header to send.
    pub user_agent: String,
    /// If true, decode HTML entities in the stripped text.
    pub decode_entities: bool,
}

impl Default for WebLoaderConfig {
    fn default() -> Self {
        Self {
            max_redirects: 5,
            timeout_secs: 30,
            user_agent: "flowgentra-ai/1.0 (document-loader)".to_string(),
            decode_entities: true,
        }
    }
}

/// Loads documents from HTTP/HTTPS URLs.
pub struct WebLoader {
    client: reqwest::Client,
    config: WebLoaderConfig,
}

impl WebLoader {
    /// Create with default settings.
    pub fn new() -> Self {
        Self::with_config(WebLoaderConfig::default())
    }

    /// Create with custom configuration.
    pub fn with_config(config: WebLoaderConfig) -> Self {
        let client = reqwest::ClientBuilder::new()
            .redirect(reqwest::redirect::Policy::limited(config.max_redirects))
            .timeout(std::time::Duration::from_secs(config.timeout_secs))
            .user_agent(&config.user_agent)
            .build()
            .unwrap_or_default();
        Self { client, config }
    }

    /// Fetch a single URL and return a `LoadedDocument`.
    pub async fn load(&self, url: &str) -> Result<LoadedDocument, VectorStoreError> {
        let response = self
            .client
            .get(url)
            .send()
            .await
            .map_err(|e| VectorStoreError::Unknown(format!("HTTP request error: {e}")))?;

        let status = response.status();
        if !status.is_success() {
            return Err(VectorStoreError::Unknown(format!(
                "HTTP {status} fetching {url}"
            )));
        }

        let content_type = response
            .headers()
            .get(reqwest::header::CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            .unwrap_or("")
            .to_string();

        let body = response
            .text()
            .await
            .map_err(|e| VectorStoreError::Unknown(format!("Read body error: {e}")))?;

        let text = if content_type.contains("html") {
            strip_html(&body)
        } else {
            body
        };

        let text = if self.config.decode_entities {
            decode_html_entities(&text)
        } else {
            text
        };

        let mut metadata = HashMap::new();
        metadata.insert("source".to_string(), json!(url));
        metadata.insert("content_type".to_string(), json!(content_type));

        Ok(LoadedDocument {
            id: url_to_id(url),
            text: text.trim().to_string(),
            source: url.to_string(),
            file_type: crate::core::rag::document_loader::FileType::Html,
            metadata,
        })
    }

    /// Fetch multiple URLs concurrently. Failed URLs are returned as errors
    /// per-URL; the successful ones are collected into the output vector.
    pub async fn load_many(
        &self,
        urls: Vec<impl AsRef<str> + Send>,
    ) -> Vec<Result<LoadedDocument, VectorStoreError>> {
        let futs: Vec<_> = urls
            .iter()
            .map(|url| self.load(url.as_ref()))
            .collect();
        futures::future::join_all(futs).await
    }
}

impl Default for WebLoader {
    fn default() -> Self {
        Self::new()
    }
}

// ── Helpers ──────────────────────────────────────────────────────────────────

fn url_to_id(url: &str) -> String {
    url.replace("https://", "")
        .replace("http://", "")
        .replace(['/', '?', '=', '&', '#'], "_")
        .trim_end_matches('_')
        .to_string()
}

fn strip_html(html: &str) -> String {
    let mut result = String::with_capacity(html.len());
    let mut in_tag = false;
    let mut in_skip = false;
    let mut tag_buf = String::new();

    for ch in html.chars() {
        match ch {
            '<' => {
                in_tag = true;
                tag_buf.clear();
            }
            '>' => {
                in_tag = false;
                let tl = tag_buf.trim().to_lowercase();
                if tl.starts_with("script") || tl.starts_with("style") {
                    in_skip = true;
                }
                if tl.starts_with("/script") || tl.starts_with("/style") {
                    in_skip = false;
                }
            }
            _ if in_tag => tag_buf.push(ch),
            _ if !in_skip => result.push(ch),
            _ => {}
        }
    }

    // Collapse multiple whitespace characters
    let mut collapsed = String::with_capacity(result.len());
    let mut prev_whitespace = false;
    for ch in result.chars() {
        if ch.is_whitespace() {
            if !prev_whitespace {
                collapsed.push(' ');
            }
            prev_whitespace = true;
        } else {
            collapsed.push(ch);
            prev_whitespace = false;
        }
    }
    collapsed
}

fn decode_html_entities(text: &str) -> String {
    text.replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", "\"")
        .replace("&#39;", "'")
        .replace("&apos;", "'")
        .replace("&nbsp;", " ")
        .replace("&copy;", "©")
        .replace("&reg;", "®")
        .replace("&trade;", "™")
        .replace("&mdash;", "—")
        .replace("&ndash;", "–")
        .replace("&hellip;", "…")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_strip_html() {
        let html = "<h1>Hello</h1><script>evil()</script><p>World &amp; more</p>";
        let stripped = strip_html(html);
        assert!(stripped.contains("Hello"));
        assert!(stripped.contains("World"));
        assert!(!stripped.contains("evil"));
        assert!(!stripped.contains("<h1>"));
    }

    #[test]
    fn test_decode_entities() {
        assert_eq!(decode_html_entities("a &amp; b"), "a & b");
        assert_eq!(decode_html_entities("&lt;tag&gt;"), "<tag>");
    }

    #[test]
    fn test_url_to_id() {
        let id = url_to_id("https://example.com/page?q=1");
        assert!(!id.contains("https://"));
        assert!(!id.contains('?'));
    }
}
