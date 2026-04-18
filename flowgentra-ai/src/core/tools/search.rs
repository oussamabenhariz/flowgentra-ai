//! Web search tools: DuckDuckGo (no key), Tavily, SerpApi, Google Serper, Brave.

use super::{JsonSchema, Tool, ToolDefinition};
use crate::prelude::*;
use async_trait::async_trait;
use serde_json::{json, Value};
use std::collections::HashMap;

// =============================================================================
// DuckDuckGoSearchTool
// =============================================================================

/// Search the web using DuckDuckGo — no API key required.
///
/// Uses the Instant Answer JSON API as primary source. Falls back to DDG Lite
/// HTML parsing when the JSON API returns no results.
pub struct DuckDuckGoSearchTool {
    client: reqwest::Client,
    max_results: usize,
}

impl DuckDuckGoSearchTool {
    pub fn new(max_results: usize) -> Self {
        Self {
            client: reqwest::Client::builder()
                .user_agent("Mozilla/5.0 (compatible; flowgentra-ai/1.0)")
                .build()
                .unwrap_or_default(),
            max_results,
        }
    }
}

impl Default for DuckDuckGoSearchTool {
    fn default() -> Self {
        Self::new(5)
    }
}

fn parse_ddg_lite_html(html: &str, max: usize) -> Vec<Value> {
    use scraper::{Html, Selector};

    let doc = Html::parse_document(html);
    // DDG Lite result links are <a class="result-link">
    let link_sel = Selector::parse("a.result-link").unwrap();
    // Snippets appear in the same table row as <td class="result-snippet">
    let snip_sel = Selector::parse("td.result-snippet").unwrap();

    let links: Vec<_> = doc.select(&link_sel).collect();
    let snippets: Vec<_> = doc.select(&snip_sel).collect();

    links
        .iter()
        .zip(snippets.iter())
        .take(max)
        .map(|(link, snip)| {
            let url = link.value().attr("href").unwrap_or("").to_string();
            let title = link.text().collect::<Vec<_>>().join("").trim().to_string();
            let snippet = snip.text().collect::<Vec<_>>().join("").trim().to_string();
            json!({"title": title, "url": url, "snippet": snippet})
        })
        .collect()
}

#[async_trait]
impl Tool for DuckDuckGoSearchTool {
    async fn call(&self, input: Value) -> Result<Value> {
        let query = input
            .get("query")
            .and_then(|v| v.as_str())
            .ok_or_else(|| FlowgentraError::ToolError("Missing 'query' field".to_string()))?
            .to_string();

        let max = input
            .get("max_results")
            .and_then(|v| v.as_u64())
            .unwrap_or(self.max_results as u64) as usize;

        // Try the Instant Answer API first (JSON, fast)
        let ia_url = format!(
            "https://api.duckduckgo.com/?q={}&format=json&no_html=1&skip_disambig=1",
            urlencoding::encode(&query)
        );

        let ia_resp: Value = self
            .client
            .get(&ia_url)
            .send()
            .await
            .map_err(|e| FlowgentraError::ToolError(format!("DDG request failed: {}", e)))?
            .json()
            .await
            .map_err(|e| FlowgentraError::ToolError(format!("DDG JSON parse failed: {}", e)))?;

        let mut results: Vec<Value> = Vec::new();

        // Collect from RelatedTopics
        if let Some(topics) = ia_resp.get("RelatedTopics").and_then(|v| v.as_array()) {
            for topic in topics.iter().take(max) {
                if let (Some(text), Some(url)) = (
                    topic.get("Text").and_then(|v| v.as_str()),
                    topic.get("FirstURL").and_then(|v| v.as_str()),
                ) {
                    if !text.is_empty() && !url.is_empty() {
                        results.push(json!({
                            "title": text.split(" - ").next().unwrap_or(text).trim(),
                            "url": url,
                            "snippet": text,
                        }));
                    }
                }
            }
        }

        // Include the abstract if available and results are sparse
        if results.len() < max {
            let abstract_text = ia_resp
                .get("AbstractText")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            let abstract_url = ia_resp
                .get("AbstractURL")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            let abstract_source = ia_resp
                .get("AbstractSource")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            if !abstract_text.is_empty() && !abstract_url.is_empty() {
                results.insert(
                    0,
                    json!({
                        "title": abstract_source,
                        "url": abstract_url,
                        "snippet": abstract_text,
                    }),
                );
            }
        }

        // Fall back to DDG Lite HTML scraping when IA API yields nothing
        if results.is_empty() {
            let lite_url = format!(
                "https://lite.duckduckgo.com/lite/?q={}",
                urlencoding::encode(&query)
            );
            if let Ok(resp) = self.client.get(&lite_url).send().await {
                if let Ok(html) = resp.text().await {
                    results = parse_ddg_lite_html(&html, max);
                }
            }
        }

        results.truncate(max);
        let count = results.len();
        Ok(json!({"query": query, "results": results, "count": count}))
    }

    fn definition(&self) -> ToolDefinition {
        let mut props = HashMap::new();
        props.insert(
            "query".to_string(),
            JsonSchema::string().with_description("Search query"),
        );
        props.insert(
            "max_results".to_string(),
            JsonSchema::integer().with_description("Maximum results to return (default: 5)"),
        );

        ToolDefinition::new(
            "duckduckgo_search",
            "Search the web using DuckDuckGo — no API key required",
            JsonSchema::object()
                .with_properties(props)
                .with_required(vec!["query".to_string()]),
            JsonSchema::object(),
        )
        .with_category("search")
    }
}

// =============================================================================
// TavilySearchTool
// =============================================================================

/// Web search powered by the Tavily AI search API.
pub struct TavilySearchTool {
    api_key: String,
    client: reqwest::Client,
}

impl TavilySearchTool {
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            client: reqwest::Client::new(),
        }
    }

    pub fn from_env() -> Result<Self> {
        let key = std::env::var("TAVILY_API_KEY").map_err(|_| {
            FlowgentraError::ToolError("TAVILY_API_KEY environment variable not set".to_string())
        })?;
        Ok(Self::new(key))
    }
}

#[async_trait]
impl Tool for TavilySearchTool {
    async fn call(&self, input: Value) -> Result<Value> {
        let query = input
            .get("query")
            .and_then(|v| v.as_str())
            .ok_or_else(|| FlowgentraError::ToolError("Missing 'query' field".to_string()))?;

        let max_results = input
            .get("max_results")
            .and_then(|v| v.as_u64())
            .unwrap_or(5);

        let body = json!({
            "api_key": self.api_key,
            "query": query,
            "max_results": max_results,
            "search_depth": "basic",
        });

        let resp: Value = self
            .client
            .post("https://api.tavily.com/search")
            .json(&body)
            .send()
            .await
            .map_err(|e| FlowgentraError::ToolError(format!("Tavily request failed: {}", e)))?
            .json()
            .await
            .map_err(|e| FlowgentraError::ToolError(format!("Tavily JSON parse failed: {}", e)))?;

        let results: Vec<Value> = resp
            .get("results")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .map(|r| {
                        json!({
                            "title": r.get("title").and_then(|v| v.as_str()).unwrap_or(""),
                            "url": r.get("url").and_then(|v| v.as_str()).unwrap_or(""),
                            "snippet": r.get("content").and_then(|v| v.as_str()).unwrap_or(""),
                        })
                    })
                    .collect()
            })
            .unwrap_or_default();

        let count = results.len();
        Ok(json!({"query": query, "results": results, "count": count}))
    }

    fn definition(&self) -> ToolDefinition {
        let mut props = HashMap::new();
        props.insert(
            "query".to_string(),
            JsonSchema::string().with_description("Search query"),
        );
        props.insert(
            "max_results".to_string(),
            JsonSchema::integer().with_description("Maximum results (default: 5)"),
        );

        ToolDefinition::new(
            "tavily_search",
            "AI-powered web search via Tavily API (requires TAVILY_API_KEY)",
            JsonSchema::object()
                .with_properties(props)
                .with_required(vec!["query".to_string()]),
            JsonSchema::object(),
        )
        .with_category("search")
    }
}

// =============================================================================
// SerpApiSearchTool
// =============================================================================

/// Google search results via SerpApi.
pub struct SerpApiSearchTool {
    api_key: String,
    client: reqwest::Client,
}

impl SerpApiSearchTool {
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            client: reqwest::Client::new(),
        }
    }

    pub fn from_env() -> Result<Self> {
        let key = std::env::var("SERPAPI_API_KEY").map_err(|_| {
            FlowgentraError::ToolError("SERPAPI_API_KEY environment variable not set".to_string())
        })?;
        Ok(Self::new(key))
    }
}

#[async_trait]
impl Tool for SerpApiSearchTool {
    async fn call(&self, input: Value) -> Result<Value> {
        let query = input
            .get("query")
            .and_then(|v| v.as_str())
            .ok_or_else(|| FlowgentraError::ToolError("Missing 'query' field".to_string()))?;

        let num = input
            .get("max_results")
            .and_then(|v| v.as_u64())
            .unwrap_or(5);

        let url = format!(
            "https://serpapi.com/search.json?q={}&api_key={}&num={}",
            urlencoding::encode(query),
            self.api_key,
            num
        );

        let resp: Value = self
            .client
            .get(&url)
            .send()
            .await
            .map_err(|e| FlowgentraError::ToolError(format!("SerpApi request failed: {}", e)))?
            .json()
            .await
            .map_err(|e| FlowgentraError::ToolError(format!("SerpApi JSON parse failed: {}", e)))?;

        let results: Vec<Value> = resp
            .get("organic_results")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .map(|r| {
                        json!({
                            "title": r.get("title").and_then(|v| v.as_str()).unwrap_or(""),
                            "url": r.get("link").and_then(|v| v.as_str()).unwrap_or(""),
                            "snippet": r.get("snippet").and_then(|v| v.as_str()).unwrap_or(""),
                        })
                    })
                    .collect()
            })
            .unwrap_or_default();

        let count = results.len();
        Ok(json!({"query": query, "results": results, "count": count}))
    }

    fn definition(&self) -> ToolDefinition {
        let mut props = HashMap::new();
        props.insert(
            "query".to_string(),
            JsonSchema::string().with_description("Search query"),
        );
        props.insert(
            "max_results".to_string(),
            JsonSchema::integer().with_description("Maximum results (default: 5)"),
        );

        ToolDefinition::new(
            "serpapi_search",
            "Google search results via SerpApi (requires SERPAPI_API_KEY)",
            JsonSchema::object()
                .with_properties(props)
                .with_required(vec!["query".to_string()]),
            JsonSchema::object(),
        )
        .with_category("search")
    }
}

// =============================================================================
// GoogleSerperTool
// =============================================================================

/// Google search via Serper.dev API.
pub struct GoogleSerperTool {
    api_key: String,
    client: reqwest::Client,
}

impl GoogleSerperTool {
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            client: reqwest::Client::new(),
        }
    }

    pub fn from_env() -> Result<Self> {
        let key = std::env::var("SERPER_API_KEY").map_err(|_| {
            FlowgentraError::ToolError("SERPER_API_KEY environment variable not set".to_string())
        })?;
        Ok(Self::new(key))
    }
}

#[async_trait]
impl Tool for GoogleSerperTool {
    async fn call(&self, input: Value) -> Result<Value> {
        let query = input
            .get("query")
            .and_then(|v| v.as_str())
            .ok_or_else(|| FlowgentraError::ToolError("Missing 'query' field".to_string()))?;

        let num = input
            .get("max_results")
            .and_then(|v| v.as_u64())
            .unwrap_or(5) as u32;

        let body = json!({"q": query, "num": num});

        let resp: Value = self
            .client
            .post("https://google.serper.dev/search")
            .header("X-API-KEY", &self.api_key)
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .await
            .map_err(|e| FlowgentraError::ToolError(format!("Serper request failed: {}", e)))?
            .json()
            .await
            .map_err(|e| FlowgentraError::ToolError(format!("Serper JSON parse failed: {}", e)))?;

        let results: Vec<Value> = resp
            .get("organic")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .map(|r| {
                        json!({
                            "title": r.get("title").and_then(|v| v.as_str()).unwrap_or(""),
                            "url": r.get("link").and_then(|v| v.as_str()).unwrap_or(""),
                            "snippet": r.get("snippet").and_then(|v| v.as_str()).unwrap_or(""),
                        })
                    })
                    .collect()
            })
            .unwrap_or_default();

        let count = results.len();
        Ok(json!({"query": query, "results": results, "count": count}))
    }

    fn definition(&self) -> ToolDefinition {
        let mut props = HashMap::new();
        props.insert(
            "query".to_string(),
            JsonSchema::string().with_description("Search query"),
        );
        props.insert(
            "max_results".to_string(),
            JsonSchema::integer().with_description("Maximum results (default: 5)"),
        );

        ToolDefinition::new(
            "google_serper",
            "Google search results via Serper.dev API (requires SERPER_API_KEY)",
            JsonSchema::object()
                .with_properties(props)
                .with_required(vec!["query".to_string()]),
            JsonSchema::object(),
        )
        .with_category("search")
    }
}

// =============================================================================
// BraveSearchTool
// =============================================================================

/// Web search via Brave Search API.
pub struct BraveSearchTool {
    api_key: String,
    client: reqwest::Client,
}

impl BraveSearchTool {
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            client: reqwest::Client::new(),
        }
    }

    pub fn from_env() -> Result<Self> {
        let key = std::env::var("BRAVE_API_KEY").map_err(|_| {
            FlowgentraError::ToolError("BRAVE_API_KEY environment variable not set".to_string())
        })?;
        Ok(Self::new(key))
    }
}

#[async_trait]
impl Tool for BraveSearchTool {
    async fn call(&self, input: Value) -> Result<Value> {
        let query = input
            .get("query")
            .and_then(|v| v.as_str())
            .ok_or_else(|| FlowgentraError::ToolError("Missing 'query' field".to_string()))?;

        let count = input
            .get("max_results")
            .and_then(|v| v.as_u64())
            .unwrap_or(5);

        let url = format!(
            "https://api.search.brave.com/res/v1/web/search?q={}&count={}",
            urlencoding::encode(query),
            count
        );

        let resp: Value = self
            .client
            .get(&url)
            .header("X-Subscription-Token", &self.api_key)
            .header("Accept", "application/json")
            .send()
            .await
            .map_err(|e| FlowgentraError::ToolError(format!("Brave Search request failed: {}", e)))?
            .json()
            .await
            .map_err(|e| {
                FlowgentraError::ToolError(format!("Brave Search JSON parse failed: {}", e))
            })?;

        let results: Vec<Value> = resp
            .pointer("/web/results")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .map(|r| {
                        json!({
                            "title": r.get("title").and_then(|v| v.as_str()).unwrap_or(""),
                            "url": r.get("url").and_then(|v| v.as_str()).unwrap_or(""),
                            "snippet": r.get("description").and_then(|v| v.as_str()).unwrap_or(""),
                        })
                    })
                    .collect()
            })
            .unwrap_or_default();

        let result_count = results.len();
        Ok(json!({"query": query, "results": results, "count": result_count}))
    }

    fn definition(&self) -> ToolDefinition {
        let mut props = HashMap::new();
        props.insert(
            "query".to_string(),
            JsonSchema::string().with_description("Search query"),
        );
        props.insert(
            "max_results".to_string(),
            JsonSchema::integer().with_description("Maximum results (default: 5)"),
        );

        ToolDefinition::new(
            "brave_search",
            "Web search via Brave Search API (requires BRAVE_API_KEY)",
            JsonSchema::object()
                .with_properties(props)
                .with_required(vec!["query".to_string()]),
            JsonSchema::object(),
        )
        .with_category("search")
    }
}
