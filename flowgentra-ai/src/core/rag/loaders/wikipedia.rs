//! Wikipedia Loader
//!
//! Fetches full Wikipedia article text (not just the summary) using the
//! Wikipedia REST API. Articles are split into sections when available.

use std::collections::HashMap;
use serde_json::json;
use crate::core::rag::document_loader::{FileType, LoadedDocument};

pub struct WikipediaLoader {
    pub lang: String,
    pub top_k: usize,
    /// Load full article text (true) or just summary (false).
    pub full_text: bool,
}

impl WikipediaLoader {
    pub fn new(top_k: usize) -> Self {
        Self { lang: "en".to_string(), top_k, full_text: true }
    }

    pub fn with_lang(mut self, lang: impl Into<String>) -> Self {
        self.lang = lang.into();
        self
    }

    pub fn summary_only(mut self) -> Self {
        self.full_text = false;
        self
    }

    pub async fn load(&self, query: &str) -> Result<Vec<LoadedDocument>, Box<dyn std::error::Error>> {
        let client = reqwest::Client::new();

        // Search for article titles
        let search_url = format!(
            "https://{}.wikipedia.org/w/api.php?action=opensearch&search={}&limit={}&format=json",
            self.lang,
            urlencoding::encode(query),
            self.top_k
        );
        let resp: serde_json::Value = client
            .get(&search_url)
            .header("User-Agent", "flowgentra-ai/1.0")
            .send().await?.json().await?;

        let titles = resp[1].as_array().cloned().unwrap_or_default();
        let mut docs = Vec::new();

        for title_val in &titles {
            let title = title_val.as_str().unwrap_or_default();
            let text = if self.full_text {
                self.fetch_full_text(&client, title).await.unwrap_or_default()
            } else {
                self.fetch_summary(&client, title).await.unwrap_or_default()
            };
            if text.trim().is_empty() { continue; }

            let url = format!("https://{}.wikipedia.org/wiki/{}", self.lang, urlencoding::encode(title));
            let mut metadata = HashMap::new();
            metadata.insert("source".to_string(), json!("wikipedia"));
            metadata.insert("title".to_string(), json!(title));
            metadata.insert("url".to_string(), json!(url));

            docs.push(LoadedDocument {
                id: format!("wiki_{}", docs.len()),
                text,
                source: url.clone(),
                file_type: FileType::PlainText,
                metadata,
            });
        }
        Ok(docs)
    }

    async fn fetch_summary(&self, client: &reqwest::Client, title: &str) -> Result<String, Box<dyn std::error::Error>> {
        let url = format!(
            "https://{}.wikipedia.org/api/rest_v1/page/summary/{}",
            self.lang, urlencoding::encode(title)
        );
        let resp: serde_json::Value = client.get(&url)
            .header("User-Agent", "flowgentra-ai/1.0")
            .send().await?.json().await?;
        Ok(resp["extract"].as_str().unwrap_or("").to_string())
    }

    async fn fetch_full_text(&self, client: &reqwest::Client, title: &str) -> Result<String, Box<dyn std::error::Error>> {
        let url = format!(
            "https://{}.wikipedia.org/w/api.php?action=query&titles={}&prop=extracts&explaintext=1&format=json",
            self.lang, urlencoding::encode(title)
        );
        let resp: serde_json::Value = client.get(&url)
            .header("User-Agent", "flowgentra-ai/1.0")
            .send().await?.json().await?;
        let pages = resp["query"]["pages"].as_object().cloned().unwrap_or_default();
        if let Some((_, page)) = pages.iter().next() {
            return Ok(page["extract"].as_str().unwrap_or("").to_string());
        }
        Ok(String::new())
    }
}
