//! RSS / Atom Feed Loader
//!
//! Parses RSS 2.0 and Atom feed XML. Each `<item>` / `<entry>` becomes one
//! [`LoadedDocument`] with title + description as its text.

use crate::core::rag::document_loader::{FileType, LoadedDocument};
use serde_json::json;
use std::collections::HashMap;

pub struct RssFeedLoader {
    feed_url: String,
    pub max_items: Option<usize>,
}

impl RssFeedLoader {
    pub fn new(feed_url: impl Into<String>) -> Self {
        Self {
            feed_url: feed_url.into(),
            max_items: None,
        }
    }

    pub fn with_max_items(mut self, n: usize) -> Self {
        self.max_items = Some(n);
        self
    }

    pub async fn load(&self) -> Result<Vec<LoadedDocument>, Box<dyn std::error::Error>> {
        let client = reqwest::Client::new();
        let xml = client
            .get(&self.feed_url)
            .header("User-Agent", "flowgentra-ai/1.0")
            .send()
            .await?
            .text()
            .await?;

        // Support both RSS (<item>) and Atom (<entry>)
        let separator = if xml.contains("<item>") || xml.contains("<item ") {
            "<item"
        } else {
            "<entry"
        };

        let mut docs = Vec::new();
        for part in xml.split(separator).skip(1) {
            let title = extract_cdata_or_tag(part, "title").unwrap_or_default();
            let description = extract_cdata_or_tag(part, "description")
                .or_else(|| extract_cdata_or_tag(part, "content"))
                .or_else(|| extract_cdata_or_tag(part, "summary"))
                .unwrap_or_default();
            let link = extract_cdata_or_tag(part, "link").unwrap_or_default();
            let pub_date = extract_cdata_or_tag(part, "pubDate")
                .or_else(|| extract_cdata_or_tag(part, "published"))
                .unwrap_or_default();

            let text = strip_html(&format!("{}: {}", title.trim(), description.trim()));
            if text.trim().is_empty() {
                continue;
            }

            let mut metadata = HashMap::new();
            metadata.insert("source".to_string(), json!("rss"));
            metadata.insert("title".to_string(), json!(title.trim()));
            metadata.insert("url".to_string(), json!(link.trim()));
            metadata.insert("pub_date".to_string(), json!(pub_date.trim()));
            metadata.insert("feed_url".to_string(), json!(self.feed_url));

            docs.push(LoadedDocument {
                id: format!("rss_{}", docs.len()),
                text,
                source: link.trim().to_string(),
                file_type: FileType::PlainText,
                metadata,
            });

            if let Some(max) = self.max_items {
                if docs.len() >= max {
                    break;
                }
            }
        }
        Ok(docs)
    }
}

fn extract_cdata_or_tag(xml: &str, tag: &str) -> Option<String> {
    let open = format!("<{tag}>");
    let close = format!("</{tag}>");
    let start = xml.find(&open)? + open.len();
    let end = xml[start..].find(&close)?;
    let raw = &xml[start..start + end];
    // Strip CDATA wrapper if present
    let content = if raw.trim_start().starts_with("<![CDATA[") {
        raw.trim_start()
            .trim_start_matches("<![CDATA[")
            .trim_end_matches("]]>")
            .to_string()
    } else {
        raw.to_string()
    };
    Some(content)
}

fn strip_html(html: &str) -> String {
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
