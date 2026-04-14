//! Sitemap Loader
//!
//! Fetches all URLs listed in an XML sitemap and loads each page's text
//! content. Supports standard sitemaps (`<urlset>`) and sitemap index files
//! (`<sitemapindex>`) with one level of nesting.

use std::collections::HashMap;

use serde_json::json;

use crate::core::rag::document_loader::{FileType, LoadedDocument};

/// Configuration for `SitemapLoader`.
#[derive(Debug, Clone)]
pub struct SitemapConfig {
    /// Max pages to load (None = unlimited).
    pub max_pages: Option<usize>,
    /// Request timeout in seconds.
    pub timeout_secs: u64,
    /// URL filter: only load URLs matching this substring.
    pub url_filter: Option<String>,
}

impl Default for SitemapConfig {
    fn default() -> Self {
        Self {
            max_pages: Some(50),
            timeout_secs: 30,
            url_filter: None,
        }
    }
}

/// Loads pages listed in an XML sitemap.
pub struct SitemapLoader {
    sitemap_url: String,
    config: SitemapConfig,
}

impl SitemapLoader {
    pub fn new(sitemap_url: impl Into<String>) -> Self {
        Self {
            sitemap_url: sitemap_url.into(),
            config: SitemapConfig::default(),
        }
    }

    pub fn with_config(mut self, config: SitemapConfig) -> Self {
        self.config = config;
        self
    }

    pub async fn load(&self) -> Result<Vec<LoadedDocument>, Box<dyn std::error::Error>> {
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(self.config.timeout_secs))
            .build()?;

        // Fetch sitemap XML
        let xml = client
            .get(&self.sitemap_url)
            .header("User-Agent", "flowgentra-ai/1.0")
            .send()
            .await?
            .text()
            .await?;

        // Collect all page URLs
        let mut urls = self.extract_urls(&xml, &client).await?;
        if let Some(filter) = &self.config.url_filter {
            urls.retain(|u| u.contains(filter.as_str()));
        }
        if let Some(max) = self.config.max_pages {
            urls.truncate(max);
        }

        // Fetch each page
        let mut docs = Vec::new();
        for url in urls {
            if let Ok(resp) = client
                .get(&url)
                .header("User-Agent", "flowgentra-ai/1.0")
                .send()
                .await
            {
                if let Ok(html) = resp.text().await {
                    let text = strip_html(&html);
                    if !text.trim().is_empty() {
                        let mut metadata = HashMap::new();
                        metadata.insert("source".to_string(), json!(url));
                        docs.push(LoadedDocument {
                            id: format!("sitemap_{}", docs.len()),
                            text,
                            source: url.clone(),
                            file_type: FileType::Html,
                            metadata,
                        });
                    }
                }
            }
        }
        Ok(docs)
    }

    async fn extract_urls(
        &self,
        xml: &str,
        client: &reqwest::Client,
    ) -> Result<Vec<String>, Box<dyn std::error::Error>> {
        let mut urls = Vec::new();

        if xml.contains("<sitemapindex") {
            // It's a sitemap index — recurse into each sitemap
            for part in xml.split("<loc>").skip(1) {
                let sitemap_url = part.split("</loc>").next().unwrap_or("").trim().to_string();
                if !sitemap_url.is_empty() {
                    if let Ok(resp) = client.get(&sitemap_url).send().await {
                        if let Ok(sub_xml) = resp.text().await {
                            let sub_urls = self.extract_page_urls(&sub_xml);
                            urls.extend(sub_urls);
                        }
                    }
                }
            }
        } else {
            urls = self.extract_page_urls(xml);
        }

        Ok(urls)
    }

    fn extract_page_urls(&self, xml: &str) -> Vec<String> {
        xml.split("<loc>")
            .skip(1)
            .filter_map(|part| {
                let url = part.split("</loc>").next()?.trim().to_string();
                if url.starts_with("http") {
                    Some(url)
                } else {
                    None
                }
            })
            .collect()
    }
}

fn strip_html(html: &str) -> String {
    let mut out = String::new();
    let mut in_tag = false;
    let mut in_script = false;
    let lower = html.to_ascii_lowercase();
    let chars: Vec<char> = html.chars().collect();
    let lower_chars: Vec<char> = lower.chars().collect();
    let mut i = 0;
    while i < chars.len() {
        if !in_script && lower_chars[i..].starts_with(&['<', 's', 'c', 'r', 'i', 'p', 't']) {
            in_script = true;
        }
        if in_script && lower_chars[i..].starts_with(&['<', '/', 's', 'c', 'r', 'i', 'p', 't', '>'])
        {
            in_script = false;
            i += 9;
            continue;
        }
        if in_script {
            i += 1;
            continue;
        }
        match chars[i] {
            '<' => in_tag = true,
            '>' => in_tag = false,
            c if !in_tag => out.push(c),
            _ => {}
        }
        i += 1;
    }
    let decoded = out
        .replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&nbsp;", " ")
        .replace("&quot;", "\"");
    decoded.split_whitespace().collect::<Vec<_>>().join(" ")
}
