//! Recursive URL Loader
//!
//! Crawls a website starting from a root URL, following internal links up to
//! a configurable depth. Each discovered page becomes a [`LoadedDocument`].

use std::collections::{HashSet, VecDeque};
use std::collections::HashMap;

use serde_json::json;

use crate::core::rag::document_loader::{FileType, LoadedDocument};

#[derive(Debug, Clone)]
pub struct RecursiveUrlConfig {
    /// Maximum crawl depth (0 = root page only).
    pub max_depth: usize,
    /// Maximum total pages to load.
    pub max_pages: usize,
    /// Only follow links that start with this prefix (defaults to the root URL).
    pub url_prefix: Option<String>,
    pub timeout_secs: u64,
    /// Ignore URLs matching these substrings.
    pub exclude_patterns: Vec<String>,
}

impl Default for RecursiveUrlConfig {
    fn default() -> Self {
        Self {
            max_depth: 2,
            max_pages: 100,
            url_prefix: None,
            timeout_secs: 30,
            exclude_patterns: vec![],
        }
    }
}

pub struct RecursiveUrlLoader {
    root_url: String,
    config: RecursiveUrlConfig,
}

impl RecursiveUrlLoader {
    pub fn new(root_url: impl Into<String>) -> Self {
        Self {
            root_url: root_url.into(),
            config: RecursiveUrlConfig::default(),
        }
    }

    pub fn with_config(mut self, config: RecursiveUrlConfig) -> Self {
        self.config = config;
        self
    }

    pub async fn load(&self) -> Result<Vec<LoadedDocument>, Box<dyn std::error::Error>> {
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(self.config.timeout_secs))
            .build()?;

        let prefix = self.config.url_prefix
            .clone()
            .unwrap_or_else(|| base_url(&self.root_url));

        let mut visited: HashSet<String> = HashSet::new();
        let mut queue: VecDeque<(String, usize)> = VecDeque::new();
        queue.push_back((self.root_url.clone(), 0));

        let mut docs = Vec::new();

        while let Some((url, depth)) = queue.pop_front() {
            if visited.contains(&url) || docs.len() >= self.config.max_pages {
                continue;
            }
            if self.config.exclude_patterns.iter().any(|p| url.contains(p.as_str())) {
                continue;
            }
            visited.insert(url.clone());

            let resp = match client
                .get(&url)
                .header("User-Agent", "flowgentra-ai/1.0")
                .send()
                .await
            {
                Ok(r) => r,
                Err(_) => continue,
            };

            let content_type = resp
                .headers()
                .get("content-type")
                .and_then(|v| v.to_str().ok())
                .unwrap_or("")
                .to_string();

            if !content_type.contains("html") {
                continue;
            }

            let html = match resp.text().await {
                Ok(h) => h,
                Err(_) => continue,
            };

            let text = strip_html_text(&html);
            if !text.trim().is_empty() {
                let mut metadata = HashMap::new();
                metadata.insert("source".to_string(), json!(url));
                metadata.insert("depth".to_string(), json!(depth));
                docs.push(LoadedDocument {
                    id: format!("url_{}", docs.len()),
                    text,
                    source: url.clone(),
                    file_type: FileType::Html,
                    metadata,
                });
            }

            // Discover links if not at max depth
            if depth < self.config.max_depth {
                let links = extract_links(&html, &url, &prefix);
                for link in links {
                    if !visited.contains(&link) {
                        queue.push_back((link, depth + 1));
                    }
                }
            }
        }

        Ok(docs)
    }
}

fn base_url(url: &str) -> String {
    let trimmed = url.trim_end_matches('/');
    if let Some(pos) = trimmed[8..].find('/') {
        trimmed[..8 + pos].to_string()
    } else {
        trimmed.to_string()
    }
}

fn extract_links(html: &str, base: &str, prefix: &str) -> Vec<String> {
    let mut links = Vec::new();
    for part in html.split("href=\"").skip(1) {
        let href = part.split('"').next().unwrap_or("").trim();
        let full = resolve_url(href, base);
        if full.starts_with(prefix) && !full.contains('#') {
            links.push(full);
        }
    }
    links
}

fn resolve_url(href: &str, base: &str) -> String {
    if href.starts_with("http") {
        href.to_string()
    } else if href.starts_with('/') {
        format!("{}{}", base_url(base), href)
    } else {
        // relative path
        let base_dir = base.rfind('/').map(|i| &base[..=i]).unwrap_or(base);
        format!("{base_dir}{href}")
    }
}

fn strip_html_text(html: &str) -> String {
    let mut out = String::new();
    let mut in_tag = false;
    for ch in html.chars() {
        match ch {
            '<' => in_tag = true,
            '>' => in_tag = false,
            c if !in_tag => out.push(c),
            _ => {}
        }
    }
    out.replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&nbsp;", " ")
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}
