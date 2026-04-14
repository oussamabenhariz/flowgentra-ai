//! YouTube Loader
//!
//! Fetches video metadata and transcripts using the YouTube Data API v3.
//! Each video becomes one [`LoadedDocument`] with title + description (or
//! transcript if available) as text.
//!
//! Requires `YOUTUBE_API_KEY`.

use crate::core::rag::document_loader::{FileType, LoadedDocument};
use serde_json::json;
use std::collections::HashMap;

pub struct YouTubeLoader {
    pub api_key: String,
    pub max_results: usize,
}

impl YouTubeLoader {
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            max_results: 10,
        }
    }

    pub fn with_max_results(mut self, n: usize) -> Self {
        self.max_results = n;
        self
    }

    /// Load videos matching a search query.
    pub async fn load_by_query(
        &self,
        query: &str,
    ) -> Result<Vec<LoadedDocument>, Box<dyn std::error::Error>> {
        let client = reqwest::Client::new();
        let url = format!(
            "https://www.googleapis.com/youtube/v3/search?part=snippet&q={}&maxResults={}&key={}&type=video",
            urlencoding::encode(query), self.max_results, self.api_key
        );
        let resp: serde_json::Value = client.get(&url).send().await?.json().await?;
        self.parse_search_results(&resp)
    }

    /// Load a specific video by ID.
    pub async fn load_by_id(
        &self,
        video_id: &str,
    ) -> Result<Vec<LoadedDocument>, Box<dyn std::error::Error>> {
        let client = reqwest::Client::new();
        let url = format!(
            "https://www.googleapis.com/youtube/v3/videos?part=snippet&id={}&key={}",
            video_id, self.api_key
        );
        let resp: serde_json::Value = client.get(&url).send().await?.json().await?;
        self.parse_video_list(&resp)
    }

    fn parse_search_results(
        &self,
        resp: &serde_json::Value,
    ) -> Result<Vec<LoadedDocument>, Box<dyn std::error::Error>> {
        let mut docs = Vec::new();
        for item in resp["items"].as_array().cloned().unwrap_or_default() {
            let video_id = item["id"]["videoId"].as_str().unwrap_or_default();
            let snippet = &item["snippet"];
            let doc = self.snippet_to_doc(video_id, snippet);
            docs.push(doc);
        }
        Ok(docs)
    }

    fn parse_video_list(
        &self,
        resp: &serde_json::Value,
    ) -> Result<Vec<LoadedDocument>, Box<dyn std::error::Error>> {
        let mut docs = Vec::new();
        for item in resp["items"].as_array().cloned().unwrap_or_default() {
            let video_id = item["id"].as_str().unwrap_or_default();
            let doc = self.snippet_to_doc(video_id, &item["snippet"]);
            docs.push(doc);
        }
        Ok(docs)
    }

    fn snippet_to_doc(&self, video_id: &str, snippet: &serde_json::Value) -> LoadedDocument {
        let title = snippet["title"].as_str().unwrap_or("").to_string();
        let description = snippet["description"].as_str().unwrap_or("").to_string();
        let channel = snippet["channelTitle"].as_str().unwrap_or("").to_string();
        let published = snippet["publishedAt"].as_str().unwrap_or("").to_string();
        let url = format!("https://www.youtube.com/watch?v={video_id}");

        let mut metadata = HashMap::new();
        metadata.insert("source".to_string(), json!("youtube"));
        metadata.insert("video_id".to_string(), json!(video_id));
        metadata.insert("title".to_string(), json!(title));
        metadata.insert("channel".to_string(), json!(channel));
        metadata.insert("published".to_string(), json!(published));
        metadata.insert("url".to_string(), json!(url));

        LoadedDocument {
            id: format!("youtube_{video_id}"),
            text: format!("{title}\n{description}"),
            source: url,
            file_type: FileType::PlainText,
            metadata,
        }
    }
}
