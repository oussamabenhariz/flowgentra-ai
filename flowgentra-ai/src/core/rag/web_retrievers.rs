//! Web-based Retrievers
//!
//! Provides retrievers that fetch documents from public web APIs:
//!
//! | Type | Source | Notes |
//! |---|---|---|
//! | [`WikipediaRetriever`] | Wikipedia REST API | Free, no key required |
//! | [`ArxivRetriever`] | arXiv search API | Free, no key required |
//! | [`TavilySearchRetriever`] | Tavily AI Search API | Requires `TAVILY_API_KEY` |
//!
//! All retrievers implement [`AsyncRetriever`] so they can be used directly or
//! inside an [`EnsembleRetriever`].

use std::collections::HashMap;

use async_trait::async_trait;
use serde_json::json;

use super::ensemble_retriever::AsyncRetriever;
use super::vector_db::{SearchResult, VectorStoreError};

// ── WikipediaRetriever ────────────────────────────────────────────────────────

/// Retrieves Wikipedia article summaries for a search query.
///
/// Uses the Wikipedia REST API (`/page/summary/{title}`) and the OpenSearch
/// API to find matching articles.
pub struct WikipediaRetriever {
    top_k: usize,
    lang: String,
}

impl WikipediaRetriever {
    pub fn new(top_k: usize) -> Self {
        Self {
            top_k,
            lang: "en".to_string(),
        }
    }

    pub fn with_lang(mut self, lang: impl Into<String>) -> Self {
        self.lang = lang.into();
        self
    }
}

#[async_trait]
impl AsyncRetriever for WikipediaRetriever {
    async fn retrieve(&self, query: &str) -> Result<Vec<SearchResult>, VectorStoreError> {
        let client = reqwest::Client::new();
        // Step 1: OpenSearch to find article titles
        let search_url = format!(
            "https://{}.wikipedia.org/w/api.php?action=opensearch&search={}&limit={}&format=json",
            self.lang,
            urlencoding::encode(query),
            self.top_k
        );

        let resp = client
            .get(&search_url)
            .header("User-Agent", "flowgentra-ai/1.0")
            .send()
            .await
            .map_err(|e| VectorStoreError::Unknown(format!("Wikipedia search: {e}")))?;

        let arr: serde_json::Value = resp
            .json()
            .await
            .map_err(|e| VectorStoreError::Unknown(format!("Wikipedia JSON: {e}")))?;

        let titles = arr[1].as_array().cloned().unwrap_or_default();

        let mut results = Vec::new();
        for (i, title_val) in titles.iter().enumerate().take(self.top_k) {
            let title = title_val.as_str().unwrap_or_default();
            // Step 2: Fetch summary for each title
            let summary_url = format!(
                "https://{}.wikipedia.org/api/rest_v1/page/summary/{}",
                self.lang,
                urlencoding::encode(title)
            );
            if let Ok(resp) = client
                .get(&summary_url)
                .header("User-Agent", "flowgentra-ai/1.0")
                .send()
                .await
            {
                if let Ok(summary_json) = resp.json::<serde_json::Value>().await {
                    let extract = summary_json["extract"].as_str().unwrap_or("").to_string();
                    let page_url = summary_json["content_urls"]["desktop"]["page"]
                        .as_str()
                        .unwrap_or("")
                        .to_string();

                    let mut metadata = HashMap::new();
                    metadata.insert("source".to_string(), json!("wikipedia"));
                    metadata.insert("title".to_string(), json!(title));
                    metadata.insert("url".to_string(), json!(page_url));

                    results.push(SearchResult {
                        id: format!("wiki_{i}"),
                        text: extract,
                        score: 1.0 - (i as f32 * 0.1), // rank-based score
                        metadata,
                    });
                }
            }
        }

        Ok(results)
    }
}

// ── ArxivRetriever ────────────────────────────────────────────────────────────

/// Retrieves arXiv paper abstracts for a search query.
///
/// Uses the arXiv Atom feed API (`http://export.arxiv.org/api/query`).
pub struct ArxivRetriever {
    top_k: usize,
    /// Sort by relevance (default) or submittedDate.
    sort_by: String,
}

impl ArxivRetriever {
    pub fn new(top_k: usize) -> Self {
        Self {
            top_k,
            sort_by: "relevance".to_string(),
        }
    }

    pub fn sort_by_date(mut self) -> Self {
        self.sort_by = "submittedDate".to_string();
        self
    }
}

#[async_trait]
impl AsyncRetriever for ArxivRetriever {
    async fn retrieve(&self, query: &str) -> Result<Vec<SearchResult>, VectorStoreError> {
        let url = format!(
            "http://export.arxiv.org/api/query?search_query=all:{}&start=0&max_results={}&sortBy={}&sortOrder=descending",
            urlencoding::encode(query),
            self.top_k,
            self.sort_by,
        );

        let client = reqwest::Client::new();
        let resp = client
            .get(&url)
            .header("User-Agent", "flowgentra-ai/1.0")
            .send()
            .await
            .map_err(|e| VectorStoreError::Unknown(format!("ArXiv fetch: {e}")))?;

        let xml = resp
            .text()
            .await
            .map_err(|e| VectorStoreError::Unknown(format!("ArXiv text: {e}")))?;

        // Minimal XML parsing for <entry> blocks
        let results = parse_arxiv_feed(&xml);
        Ok(results.into_iter().take(self.top_k).collect())
    }
}

/// Parse arXiv Atom feed entries (title + summary) without an XML crate.
fn parse_arxiv_feed(xml: &str) -> Vec<SearchResult> {
    let mut results = Vec::new();
    let entries: Vec<&str> = xml.split("<entry>").skip(1).collect();

    for (i, entry) in entries.iter().enumerate() {
        let title = extract_xml_tag(entry, "title").unwrap_or_default();
        let summary = extract_xml_tag(entry, "summary").unwrap_or_default();
        let id_raw = extract_xml_tag(entry, "id").unwrap_or_default();
        let authors: Vec<&str> = entry
            .split("<name>")
            .skip(1)
            .map(|s| s.split("</name>").next().unwrap_or("").trim())
            .collect();

        let mut metadata = HashMap::new();
        metadata.insert("source".to_string(), json!("arxiv"));
        metadata.insert("title".to_string(), json!(title.trim()));
        metadata.insert("url".to_string(), json!(id_raw.trim()));
        metadata.insert("authors".to_string(), json!(authors));

        results.push(SearchResult {
            id: format!("arxiv_{i}"),
            text: format!("{}: {}", title.trim(), summary.trim()),
            score: 1.0 - (i as f32 * 0.05),
            metadata,
        });
    }
    results
}

fn extract_xml_tag(xml: &str, tag: &str) -> Option<String> {
    let open = format!("<{tag}>");
    let close = format!("</{tag}>");
    let start = xml.find(&open)? + open.len();
    let end = xml[start..].find(&close)?;
    Some(xml[start..start + end].to_string())
}

// ── TavilySearchRetriever ─────────────────────────────────────────────────────

/// Retrieves web search results using the Tavily AI Search API.
///
/// Requires a `TAVILY_API_KEY`. Results include web snippets with source URLs.
pub struct TavilySearchRetriever {
    api_key: String,
    top_k: usize,
    search_depth: TavilySearchDepth,
}

#[derive(Debug, Clone, Copy)]
pub enum TavilySearchDepth {
    Basic,
    Advanced,
}

impl TavilySearchRetriever {
    pub fn new(api_key: impl Into<String>, top_k: usize) -> Self {
        Self {
            api_key: api_key.into(),
            top_k,
            search_depth: TavilySearchDepth::Basic,
        }
    }

    pub fn with_advanced_search(mut self) -> Self {
        self.search_depth = TavilySearchDepth::Advanced;
        self
    }
}

#[async_trait]
impl AsyncRetriever for TavilySearchRetriever {
    async fn retrieve(&self, query: &str) -> Result<Vec<SearchResult>, VectorStoreError> {
        let client = reqwest::Client::new();
        let depth = match self.search_depth {
            TavilySearchDepth::Basic => "basic",
            TavilySearchDepth::Advanced => "advanced",
        };
        let body = json!({
            "api_key": self.api_key,
            "query": query,
            "max_results": self.top_k,
            "search_depth": depth,
        });

        let resp = client
            .post("https://api.tavily.com/search")
            .json(&body)
            .send()
            .await
            .map_err(|e| VectorStoreError::Unknown(format!("Tavily HTTP: {e}")))?;

        let json: serde_json::Value = resp
            .json()
            .await
            .map_err(|e| VectorStoreError::Unknown(format!("Tavily JSON: {e}")))?;

        let mut results = Vec::new();
        for (i, result) in json["results"]
            .as_array()
            .cloned()
            .unwrap_or_default()
            .iter()
            .enumerate()
        {
            let title = result["title"].as_str().unwrap_or("").to_string();
            let content = result["content"].as_str().unwrap_or("").to_string();
            let url = result["url"].as_str().unwrap_or("").to_string();
            let score = result["score"].as_f64().unwrap_or(0.0) as f32;

            let mut metadata = HashMap::new();
            metadata.insert("source".to_string(), json!("tavily"));
            metadata.insert("title".to_string(), json!(title));
            metadata.insert("url".to_string(), json!(url));

            results.push(SearchResult {
                id: format!("tavily_{i}"),
                text: content,
                score,
                metadata,
            });
        }
        Ok(results)
    }
}
