//! ArXiv Loader
//!
//! Fetches paper abstracts from the arXiv API. Each paper becomes a
//! [`LoadedDocument`] with the abstract as its text.

use crate::core::rag::document_loader::{FileType, LoadedDocument};
use serde_json::json;
use std::collections::HashMap;

pub struct ArxivLoader {
    pub max_results: usize,
    pub sort_by: String,
}

impl ArxivLoader {
    pub fn new(max_results: usize) -> Self {
        Self {
            max_results,
            sort_by: "relevance".to_string(),
        }
    }

    pub fn sort_by_date(mut self) -> Self {
        self.sort_by = "submittedDate".to_string();
        self
    }

    pub async fn load(
        &self,
        query: &str,
    ) -> Result<Vec<LoadedDocument>, Box<dyn std::error::Error>> {
        let url = format!(
            "http://export.arxiv.org/api/query?search_query=all:{}&start=0&max_results={}&sortBy={}&sortOrder=descending",
            urlencoding::encode(query), self.max_results, self.sort_by
        );
        let client = reqwest::Client::new();
        let xml = client
            .get(&url)
            .header("User-Agent", "flowgentra-ai/1.0")
            .send()
            .await?
            .text()
            .await?;

        let mut docs = Vec::new();
        for entry in xml.split("<entry>").skip(1) {
            let title = extract_tag(entry, "title").unwrap_or_default();
            let summary = extract_tag(entry, "summary").unwrap_or_default();
            let id = extract_tag(entry, "id").unwrap_or_default();
            let authors: Vec<String> = entry
                .split("<name>")
                .skip(1)
                .filter_map(|s| s.split("</name>").next().map(|a| a.trim().to_string()))
                .collect();
            let published = extract_tag(entry, "published").unwrap_or_default();

            let mut metadata = HashMap::new();
            metadata.insert("source".to_string(), json!("arxiv"));
            metadata.insert("title".to_string(), json!(title.trim()));
            metadata.insert("url".to_string(), json!(id.trim()));
            metadata.insert("authors".to_string(), json!(authors));
            metadata.insert("published".to_string(), json!(published.trim()));

            docs.push(LoadedDocument {
                id: format!("arxiv_{}", docs.len()),
                text: format!("{}: {}", title.trim(), summary.trim()),
                source: id.trim().to_string(),
                file_type: FileType::PlainText,
                metadata,
            });
        }
        Ok(docs)
    }
}

fn extract_tag(xml: &str, tag: &str) -> Option<String> {
    let open = format!("<{tag}>");
    let close = format!("</{tag}>");
    let start = xml.find(&open)? + open.len();
    let end = xml[start..].find(&close)?;
    Some(xml[start..start + end].to_string())
}
