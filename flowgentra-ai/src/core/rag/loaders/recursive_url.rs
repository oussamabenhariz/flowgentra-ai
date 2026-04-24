//! Recursive URL Loader
//!
//! Crawls a website starting from a root URL, following internal links up to
//! a configurable depth. Each discovered page becomes a [`LoadedDocument`].

use std::collections::HashMap;
use std::collections::{HashSet, VecDeque};
use std::net::IpAddr;

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
        validate_url(&self.root_url).map_err(|e| format!("Root URL rejected: {e}"))?;

        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(self.config.timeout_secs))
            .build()?;

        let prefix = self
            .config
            .url_prefix
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
            if self
                .config
                .exclude_patterns
                .iter()
                .any(|p| url.contains(p.as_str()))
            {
                continue;
            }
            // Re-validate each URL before fetching (handles redirected/resolved links)
            if validate_url(&url).is_err() {
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

/// Returns `Err` if `url` should not be fetched (wrong scheme, private IP, etc.).
fn validate_url(url: &str) -> Result<(), String> {
    let parsed = url::Url::parse(url).map_err(|e| format!("Invalid URL: {e}"))?;

    match parsed.scheme() {
        "http" | "https" => {}
        scheme => {
            return Err(format!(
                "Blocked scheme '{scheme}': only HTTP/HTTPS allowed"
            ))
        }
    }

    if let Some(host) = parsed.host_str() {
        if let Ok(ip) = host.parse::<IpAddr>() {
            if ip.is_loopback() || is_private_ip(ip) || ip.is_unspecified() {
                return Err(format!("Blocked private/loopback IP: {ip}"));
            }
        }
        // Block cloud metadata endpoints by hostname
        if host == "169.254.169.254"
            || host == "metadata.google.internal"
            || host.ends_with(".internal")
        {
            return Err(format!("Blocked metadata endpoint: {host}"));
        }
    }

    Ok(())
}

fn is_private_ip(ip: IpAddr) -> bool {
    match ip {
        IpAddr::V4(v4) => {
            let octets = v4.octets();
            octets[0] == 10
                || (octets[0] == 172 && (16..=31).contains(&octets[1]))
                || (octets[0] == 192 && octets[1] == 168)
                || v4.is_link_local()
        }
        IpAddr::V6(v6) => v6.is_loopback(),
    }
}

fn base_url(url: &str) -> String {
    if let Ok(parsed) = url::Url::parse(url) {
        let mut base = format!("{}://{}", parsed.scheme(), parsed.host_str().unwrap_or(""));
        if let Some(port) = parsed.port() {
            base.push_str(&format!(":{port}"));
        }
        base
    } else {
        // Fallback: trim trailing path component
        let trimmed = url.trim_end_matches('/');
        if let Some(pos) = trimmed
            .rfind("://")
            .and_then(|p| trimmed[p + 3..].find('/').map(|q| p + 3 + q))
        {
            trimmed[..pos].to_string()
        } else {
            trimmed.to_string()
        }
    }
}

fn extract_links(html: &str, base: &str, prefix: &str) -> Vec<String> {
    let mut links = Vec::new();
    for part in html.split("href=\"").skip(1) {
        let href = part.split('"').next().unwrap_or("").trim();
        let full = resolve_url(href, base);
        // Require exact host prefix match, not just string prefix, to prevent
        // attacks like https://example.com.evil.com/ passing a prefix check.
        if !full.contains('#') && url_matches_prefix(&full, prefix) && validate_url(&full).is_ok() {
            links.push(full);
        }
    }
    links
}

/// Returns true only when `url` is under the same origin + path prefix as `prefix`.
fn url_matches_prefix(url: &str, prefix: &str) -> bool {
    let Ok(u) = url::Url::parse(url) else {
        return false;
    };
    let Ok(p) = url::Url::parse(prefix) else {
        return url.starts_with(prefix);
    };

    u.scheme() == p.scheme()
        && u.host() == p.host()
        && u.port() == p.port()
        && u.path().starts_with(p.path())
}

fn resolve_url(href: &str, base: &str) -> String {
    if href.starts_with("http://") || href.starts_with("https://") {
        href.to_string()
    } else if let Ok(base_parsed) = url::Url::parse(base) {
        base_parsed
            .join(href)
            .map(|u| u.to_string())
            .unwrap_or_default()
    } else if href.starts_with('/') {
        format!("{}{}", base_url(base), href)
    } else {
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
