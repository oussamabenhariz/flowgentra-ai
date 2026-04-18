//! News tool via NewsAPI.

use super::{JsonSchema, Tool, ToolDefinition};
use crate::prelude::*;
use async_trait::async_trait;
use serde_json::{json, Value};
use std::collections::HashMap;

pub struct NewsApiTool {
    api_key: String,
    client: reqwest::Client,
}

impl NewsApiTool {
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            client: reqwest::Client::new(),
        }
    }

    pub fn from_env() -> Result<Self> {
        let key = std::env::var("NEWS_API_KEY").map_err(|_| {
            FlowgentraError::ToolError("NEWS_API_KEY environment variable not set".to_string())
        })?;
        Ok(Self::new(key))
    }
}

#[async_trait]
impl Tool for NewsApiTool {
    async fn call(&self, input: Value) -> Result<Value> {
        let query = input
            .get("query")
            .and_then(|v| v.as_str())
            .ok_or_else(|| FlowgentraError::ToolError("Missing 'query' field".to_string()))?;

        let page_size = input
            .get("max_results")
            .and_then(|v| v.as_u64())
            .unwrap_or(5);

        let language = input
            .get("language")
            .and_then(|v| v.as_str())
            .unwrap_or("en");

        let url = format!(
            "https://newsapi.org/v2/everything?q={}&apiKey={}&pageSize={}&language={}",
            urlencoding::encode(query),
            self.api_key,
            page_size,
            language
        );

        let resp: Value = self
            .client
            .get(&url)
            .header("User-Agent", "flowgentra-ai/1.0")
            .send()
            .await
            .map_err(|e| FlowgentraError::ToolError(format!("NewsAPI request failed: {}", e)))?
            .json()
            .await
            .map_err(|e| FlowgentraError::ToolError(format!("NewsAPI JSON parse failed: {}", e)))?;

        if resp.get("status").and_then(|v| v.as_str()) == Some("error") {
            let msg = resp
                .get("message")
                .and_then(|v| v.as_str())
                .unwrap_or("Unknown error");
            return Err(FlowgentraError::ToolError(format!(
                "NewsAPI error: {}",
                msg
            )));
        }

        let total_results = resp
            .get("totalResults")
            .and_then(|v| v.as_i64())
            .unwrap_or(0);

        let articles: Vec<Value> = resp
            .get("articles")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .map(|a| {
                        json!({
                            "title": a.get("title").and_then(|v| v.as_str()).unwrap_or(""),
                            "source": a.pointer("/source/name").and_then(|v| v.as_str()).unwrap_or(""),
                            "description": a.get("description").and_then(|v| v.as_str()).unwrap_or(""),
                            "url": a.get("url").and_then(|v| v.as_str()).unwrap_or(""),
                            "published_at": a.get("publishedAt").and_then(|v| v.as_str()).unwrap_or(""),
                        })
                    })
                    .collect()
            })
            .unwrap_or_default();

        Ok(json!({
            "query": query,
            "total_results": total_results,
            "articles": articles,
            "count": articles.len(),
        }))
    }

    fn definition(&self) -> ToolDefinition {
        let mut props = HashMap::new();
        props.insert(
            "query".to_string(),
            JsonSchema::string().with_description("News search query"),
        );
        props.insert(
            "max_results".to_string(),
            JsonSchema::integer().with_description("Maximum articles to return (default: 5)"),
        );
        props.insert(
            "language".to_string(),
            JsonSchema::string()
                .with_description("Language code, e.g. \"en\", \"fr\" (default: en)"),
        );

        ToolDefinition::new(
            "news_api",
            "Search recent news articles via NewsAPI (requires NEWS_API_KEY)",
            JsonSchema::object()
                .with_properties(props)
                .with_required(vec!["query".to_string()]),
            JsonSchema::object(),
        )
        .with_category("news")
    }
}
