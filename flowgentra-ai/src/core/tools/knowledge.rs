//! Knowledge-base tools: Wikipedia, ArXiv, PubMed, Wolfram Alpha.

use super::{JsonSchema, Tool, ToolDefinition};
use crate::prelude::*;
use async_trait::async_trait;
use serde_json::{json, Value};
use std::collections::HashMap;

// =============================================================================
// WikipediaTool
// =============================================================================

/// Fetch a Wikipedia article summary using the REST API (no API key required).
pub struct WikipediaTool {
    client: reqwest::Client,
}

impl WikipediaTool {
    pub fn new() -> Self {
        Self {
            client: reqwest::Client::new(),
        }
    }
}

impl Default for WikipediaTool {
    fn default() -> Self {
        Self::new()
    }
}

/// Strip HTML tags from text using the scraper crate.
fn strip_html(html: &str) -> String {
    let fragment = scraper::Html::parse_fragment(html);
    fragment.root_element().text().collect::<Vec<_>>().join("")
}

#[async_trait]
impl Tool for WikipediaTool {
    async fn call(&self, input: Value) -> Result<Value> {
        let title = input
            .get("title")
            .and_then(|v| v.as_str())
            .ok_or_else(|| FlowgentraError::ToolError("Missing 'title' field".to_string()))?;

        let encoded = urlencoding::encode(title);
        let url = format!(
            "https://en.wikipedia.org/api/rest_v1/page/summary/{}",
            encoded
        );

        let resp = self
            .client
            .get(&url)
            .header("User-Agent", "flowgentra-ai/1.0 (https://flowgentra.dev)")
            .send()
            .await
            .map_err(|e| FlowgentraError::ToolError(format!("Wikipedia request failed: {}", e)))?;

        if resp.status() == reqwest::StatusCode::NOT_FOUND {
            return Ok(json!({"found": false, "title": title}));
        }

        let body: Value = resp.json().await.map_err(|e| {
            FlowgentraError::ToolError(format!("Wikipedia JSON parse failed: {}", e))
        })?;

        let page_title = body
            .get("title")
            .and_then(|v| v.as_str())
            .unwrap_or(title)
            .to_string();
        let extract_raw = body
            .get("extract")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        let extract = strip_html(&extract_raw);
        let page_url = body
            .pointer("/content_urls/desktop/page")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();

        Ok(json!({
            "found": true,
            "title": page_title,
            "extract": extract,
            "url": page_url
        }))
    }

    fn definition(&self) -> ToolDefinition {
        let mut props = HashMap::new();
        props.insert(
            "title".to_string(),
            JsonSchema::string().with_description("Wikipedia article title to look up"),
        );

        ToolDefinition::new(
            "wikipedia",
            "Fetch the summary of a Wikipedia article (no API key required)",
            JsonSchema::object()
                .with_properties(props)
                .with_required(vec!["title".to_string()]),
            JsonSchema::object(),
        )
        .with_category("knowledge")
        .with_example(
            json!({"title": "Rust programming language"}),
            json!({"found": true, "title": "Rust (programming language)", "extract": "...", "url": "https://en.wikipedia.org/wiki/Rust_(programming_language)"}),
            "Look up the Rust programming language",
        )
    }
}

// =============================================================================
// ArxivTool
// =============================================================================

/// Search arXiv preprints via the public Atom/XML API (no API key required).
pub struct ArxivTool {
    client: reqwest::Client,
}

impl ArxivTool {
    pub fn new() -> Self {
        Self {
            client: reqwest::Client::new(),
        }
    }
}

impl Default for ArxivTool {
    fn default() -> Self {
        Self::new()
    }
}

fn parse_arxiv_xml(xml: &str) -> Vec<Value> {
    use quick_xml::events::Event;
    use quick_xml::Reader;

    let mut reader = Reader::from_str(xml);
    reader.trim_text(true);

    let mut entries: Vec<Value> = Vec::new();
    let mut current: HashMap<String, String> = HashMap::new();
    let mut in_entry = false;
    let mut current_tag = String::new();

    loop {
        match reader.read_event() {
            Ok(Event::Start(ref e)) => {
                let name = std::str::from_utf8(e.local_name().as_ref())
                    .unwrap_or("")
                    .to_string();
                match name.as_str() {
                    "entry" => {
                        in_entry = true;
                        current.clear();
                    }
                    "title" | "summary" | "id" | "published" if in_entry => {
                        current_tag = name;
                    }
                    _ => {}
                }
            }
            Ok(Event::Text(ref e)) => {
                if in_entry && !current_tag.is_empty() {
                    let text = e.unescape().unwrap_or_default().into_owned();
                    current
                        .entry(current_tag.clone())
                        .and_modify(|v| {
                            v.push(' ');
                            v.push_str(&text);
                        })
                        .or_insert(text);
                }
            }
            Ok(Event::End(ref e)) => {
                let name = std::str::from_utf8(e.local_name().as_ref())
                    .unwrap_or("")
                    .to_string();
                if name == "entry" && in_entry {
                    in_entry = false;
                    entries.push(json!({
                        "id": current.get("id").cloned().unwrap_or_default().trim().to_string(),
                        "title": current.get("title").cloned().unwrap_or_default().trim().to_string(),
                        "summary": current.get("summary").cloned().unwrap_or_default().trim().to_string(),
                        "published": current.get("published").cloned().unwrap_or_default().trim().to_string(),
                    }));
                    current_tag.clear();
                } else if name == current_tag {
                    current_tag.clear();
                }
            }
            Ok(Event::Eof) => break,
            Err(_) => break,
            _ => {}
        }
    }

    entries
}

#[async_trait]
impl Tool for ArxivTool {
    async fn call(&self, input: Value) -> Result<Value> {
        let query = input
            .get("query")
            .and_then(|v| v.as_str())
            .ok_or_else(|| FlowgentraError::ToolError("Missing 'query' field".to_string()))?;

        let max_results = input
            .get("max_results")
            .and_then(|v| v.as_u64())
            .unwrap_or(5);

        let encoded_query = urlencoding::encode(query);
        let url = format!(
            "http://export.arxiv.org/api/query?search_query=all:{}&max_results={}&sortBy=relevance",
            encoded_query, max_results
        );

        let body = self
            .client
            .get(&url)
            .header("User-Agent", "flowgentra-ai/1.0")
            .send()
            .await
            .map_err(|e| FlowgentraError::ToolError(format!("ArXiv request failed: {}", e)))?
            .text()
            .await
            .map_err(|e| FlowgentraError::ToolError(format!("ArXiv read failed: {}", e)))?;

        let results = parse_arxiv_xml(&body);
        let count = results.len();

        Ok(json!({"query": query, "results": results, "count": count}))
    }

    fn definition(&self) -> ToolDefinition {
        let mut props = HashMap::new();
        props.insert(
            "query".to_string(),
            JsonSchema::string().with_description("Search query for arXiv papers"),
        );
        props.insert(
            "max_results".to_string(),
            JsonSchema::integer()
                .with_description("Maximum number of papers to return (default: 5)"),
        );

        ToolDefinition::new(
            "arxiv",
            "Search arXiv for preprint papers by keyword (no API key required)",
            JsonSchema::object()
                .with_properties(props)
                .with_required(vec!["query".to_string()]),
            JsonSchema::object(),
        )
        .with_category("knowledge")
    }
}

// =============================================================================
// PubMedTool
// =============================================================================

/// Search PubMed via NCBI E-utilities (no API key required for basic use).
pub struct PubMedTool {
    client: reqwest::Client,
}

impl PubMedTool {
    pub fn new() -> Self {
        Self {
            client: reqwest::Client::new(),
        }
    }
}

impl Default for PubMedTool {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Tool for PubMedTool {
    async fn call(&self, input: Value) -> Result<Value> {
        let query = input
            .get("query")
            .and_then(|v| v.as_str())
            .ok_or_else(|| FlowgentraError::ToolError("Missing 'query' field".to_string()))?;

        let max_results = input
            .get("max_results")
            .and_then(|v| v.as_u64())
            .unwrap_or(5);

        // Step 1: esearch — get list of PubMed IDs
        let search_url = format!(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&term={}&retmax={}&retmode=json",
            urlencoding::encode(query),
            max_results
        );

        let search_resp: Value = self
            .client
            .get(&search_url)
            .header("User-Agent", "flowgentra-ai/1.0")
            .send()
            .await
            .map_err(|e| FlowgentraError::ToolError(format!("PubMed esearch failed: {}", e)))?
            .json()
            .await
            .map_err(|e| {
                FlowgentraError::ToolError(format!("PubMed esearch parse failed: {}", e))
            })?;

        let ids: Vec<String> = search_resp
            .pointer("/esearchresult/idlist")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(|s| s.to_string()))
                    .collect()
            })
            .unwrap_or_default();

        if ids.is_empty() {
            return Ok(json!({"query": query, "results": [], "count": 0}));
        }

        // Step 2: esummary — get titles and metadata
        let ids_str = ids.join(",");
        let summary_url = format!(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?db=pubmed&id={}&retmode=json",
            ids_str
        );

        let summary_resp: Value = self
            .client
            .get(&summary_url)
            .header("User-Agent", "flowgentra-ai/1.0")
            .send()
            .await
            .map_err(|e| FlowgentraError::ToolError(format!("PubMed esummary failed: {}", e)))?
            .json()
            .await
            .map_err(|e| {
                FlowgentraError::ToolError(format!("PubMed esummary parse failed: {}", e))
            })?;

        let result_map = summary_resp
            .pointer("/result")
            .and_then(|v| v.as_object())
            .cloned()
            .unwrap_or_default();

        let results: Vec<Value> = ids
            .iter()
            .filter_map(|id| {
                let article = result_map.get(id)?;
                Some(json!({
                    "uid": id,
                    "title": article.get("title").and_then(|v| v.as_str()).unwrap_or(""),
                    "source": article.get("source").and_then(|v| v.as_str()).unwrap_or(""),
                    "pubdate": article.get("pubdate").and_then(|v| v.as_str()).unwrap_or(""),
                    "url": format!("https://pubmed.ncbi.nlm.nih.gov/{}/", id),
                }))
            })
            .collect();

        let count = results.len();
        Ok(json!({"query": query, "results": results, "count": count}))
    }

    fn definition(&self) -> ToolDefinition {
        let mut props = HashMap::new();
        props.insert(
            "query".to_string(),
            JsonSchema::string().with_description("PubMed search query"),
        );
        props.insert(
            "max_results".to_string(),
            JsonSchema::integer().with_description("Maximum results (default: 5)"),
        );

        ToolDefinition::new(
            "pubmed",
            "Search PubMed for biomedical literature via NCBI E-utilities (no API key required)",
            JsonSchema::object()
                .with_properties(props)
                .with_required(vec!["query".to_string()]),
            JsonSchema::object(),
        )
        .with_category("knowledge")
    }
}

// =============================================================================
// WolframAlphaTool
// =============================================================================

/// Query Wolfram Alpha Short Answers API (requires an App ID).
pub struct WolframAlphaTool {
    app_id: String,
    client: reqwest::Client,
}

impl WolframAlphaTool {
    pub fn new(app_id: impl Into<String>) -> Self {
        Self {
            app_id: app_id.into(),
            client: reqwest::Client::new(),
        }
    }

    pub fn from_env() -> Result<Self> {
        let key = std::env::var("WOLFRAM_ALPHA_APPID").map_err(|_| {
            FlowgentraError::ToolError(
                "WOLFRAM_ALPHA_APPID environment variable not set".to_string(),
            )
        })?;
        Ok(Self::new(key))
    }
}

#[async_trait]
impl Tool for WolframAlphaTool {
    async fn call(&self, input: Value) -> Result<Value> {
        let query = input
            .get("query")
            .and_then(|v| v.as_str())
            .ok_or_else(|| FlowgentraError::ToolError("Missing 'query' field".to_string()))?;

        let url = format!(
            "https://api.wolframalpha.com/v2/query?input={}&appid={}&output=json",
            urlencoding::encode(query),
            self.app_id
        );

        let resp: Value = self
            .client
            .get(&url)
            .send()
            .await
            .map_err(|e| {
                FlowgentraError::ToolError(format!("Wolfram Alpha request failed: {}", e))
            })?
            .json()
            .await
            .map_err(|e| {
                FlowgentraError::ToolError(format!("Wolfram Alpha parse failed: {}", e))
            })?;

        // Extract the primary pod's plaintext (usually the "Result" pod)
        let result_text = resp
            .pointer("/queryresult/pods")
            .and_then(|pods| pods.as_array())
            .and_then(|pods| {
                // Prefer pod with title "Result"; fall back to first pod with text
                pods.iter()
                    .find(|p| p.get("title").and_then(|t| t.as_str()) == Some("Result"))
                    .or_else(|| pods.get(1)) // index 1 is typically the result
                    .and_then(|p| p.pointer("/subpods/0/plaintext"))
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string())
            })
            .unwrap_or_else(|| "No result found".to_string());

        let success = resp
            .pointer("/queryresult/success")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        Ok(json!({"query": query, "result": result_text, "success": success}))
    }

    fn definition(&self) -> ToolDefinition {
        let mut props = HashMap::new();
        props.insert(
            "query".to_string(),
            JsonSchema::string()
                .with_description("Mathematical or factual query for Wolfram Alpha"),
        );

        ToolDefinition::new(
            "wolfram_alpha",
            "Compute answers to mathematical, scientific, and factual queries via Wolfram Alpha (requires WOLFRAM_ALPHA_APPID)",
            JsonSchema::object()
                .with_properties(props)
                .with_required(vec!["query".to_string()]),
            JsonSchema::object(),
        )
        .with_category("knowledge")
        .with_example(
            json!({"query": "integrate x^2 dx"}),
            json!({"query": "integrate x^2 dx", "result": "x^3/3 + constant", "success": true}),
            "Compute a mathematical integral",
        )
    }
}
