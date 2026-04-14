//! Additional Embedding Providers
//!
//! HTTP-based implementations mirroring LangChain's embedding classes.
//!
//! | Type | Provider | API endpoint |
//! |---|---|---|
//! | [`CohereEmbeddings`] | Cohere | `https://api.cohere.ai/v1/embed` |
//! | [`AzureOpenAIEmbeddings`] | Azure OpenAI | `{endpoint}/openai/deployments/{model}/embeddings` |
//! | [`GoogleVertexEmbeddings`] | Google Vertex AI | `{endpoint}/predict` |
//! | [`BedrockEmbeddings`] | AWS Bedrock | `https://bedrock-runtime.{region}.amazonaws.com` |
//! | [`VoyageEmbeddings`] | Voyage AI | `https://api.voyageai.com/v1/embeddings` |
//! | [`JinaEmbeddings`] | Jina AI | `https://api.jina.ai/v1/embeddings` |
//! | [`TogetherEmbeddings`] | Together AI | `https://api.together.xyz/v1/embeddings` |
//! | [`NomicEmbeddings`] | Nomic Atlas | `https://api-atlas.nomic.ai/v1/embedding/text` |
//!
//! All types implement [`EmbeddingsProvider`].

use async_trait::async_trait;
use serde_json::{json, Value};

use super::embeddings::{EmbeddingError, EmbeddingsProvider};

// ── helper ────────────────────────────────────────────────────────────────────

async fn post_json(
    url: &str,
    bearer: &str,
    body: Value,
) -> Result<Value, EmbeddingError> {
    let client = reqwest::Client::new();
    let resp = client
        .post(url)
        .bearer_auth(bearer)
        .header("Content-Type", "application/json")
        .json(&body)
        .send()
        .await
        .map_err(|e| EmbeddingError::Network(e.to_string()))?;
    resp.json::<Value>()
        .await
        .map_err(|e| EmbeddingError::ApiError { status: 0, message: e.to_string() })
}

fn extract_float_array(val: &Value) -> Vec<f32> {
    val.as_array()
        .map(|arr| arr.iter().filter_map(|v| v.as_f64().map(|f| f as f32)).collect())
        .unwrap_or_default()
}

// ── Cohere ────────────────────────────────────────────────────────────────────

/// Embeddings via Cohere's `/v1/embed` endpoint.
pub struct CohereEmbeddings {
    pub api_key: String,
    /// Model id, e.g. `embed-english-v3.0`
    pub model: String,
    /// Input type: `search_document` or `search_query`
    pub input_type: String,
}

impl CohereEmbeddings {
    pub fn new(api_key: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            model: model.into(),
            input_type: "search_document".to_string(),
        }
    }

    pub fn for_query(mut self) -> Self {
        self.input_type = "search_query".to_string();
        self
    }
}

#[async_trait]
impl EmbeddingsProvider for CohereEmbeddings {
    async fn embed(&self, text: &str) -> Result<Vec<f32>, EmbeddingError> {
        let resp = post_json(
            "https://api.cohere.ai/v1/embed",
            &self.api_key,
            json!({
                "model": self.model,
                "texts": [text],
                "input_type": self.input_type,
            }),
        )
        .await?;
        Ok(extract_float_array(&resp["embeddings"][0]))
    }

    async fn embed_batch(&self, texts: Vec<&str>) -> Result<Vec<Vec<f32>>, EmbeddingError> {
        let resp = post_json(
            "https://api.cohere.ai/v1/embed",
            &self.api_key,
            json!({
                "model": self.model,
                "texts": texts,
                "input_type": self.input_type,
            }),
        )
        .await?;
        Ok(resp["embeddings"]
            .as_array()
            .map(|arr| arr.iter().map(|e| extract_float_array(e)).collect())
            .unwrap_or_default())
    }

    fn get_dimension(&self) -> usize {
        1024
    }
}

// ── Azure OpenAI ──────────────────────────────────────────────────────────────

/// Embeddings via Azure OpenAI's deployment endpoint.
pub struct AzureOpenAIEmbeddings {
    pub endpoint: String,
    pub deployment: String,
    pub api_key: String,
    pub api_version: String,
}

impl AzureOpenAIEmbeddings {
    pub fn new(
        endpoint: impl Into<String>,
        deployment: impl Into<String>,
        api_key: impl Into<String>,
    ) -> Self {
        Self {
            endpoint: endpoint.into(),
            deployment: deployment.into(),
            api_key: api_key.into(),
            api_version: "2024-02-01".to_string(),
        }
    }

    fn url(&self) -> String {
        format!(
            "{}/openai/deployments/{}/embeddings?api-version={}",
            self.endpoint.trim_end_matches('/'),
            self.deployment,
            self.api_version
        )
    }
}

#[async_trait]
impl EmbeddingsProvider for AzureOpenAIEmbeddings {
    async fn embed(&self, text: &str) -> Result<Vec<f32>, EmbeddingError> {
        let resp = post_json(
            &self.url(),
            &self.api_key,
            json!({ "input": text }),
        )
        .await?;
        Ok(extract_float_array(&resp["data"][0]["embedding"]))
    }

    async fn embed_batch(&self, texts: Vec<&str>) -> Result<Vec<Vec<f32>>, EmbeddingError> {
        let resp = post_json(
            &self.url(),
            &self.api_key,
            json!({ "input": texts }),
        )
        .await?;
        let mut items = resp["data"].as_array().cloned().unwrap_or_default();
        items.sort_by_key(|v| v["index"].as_u64().unwrap_or(0));
        Ok(items.iter().map(|v| extract_float_array(&v["embedding"])).collect())
    }

    fn get_dimension(&self) -> usize {
        1536 // text-embedding-ada-002 / 3-small defaults
    }
}

// ── Google Vertex AI ──────────────────────────────────────────────────────────

/// Embeddings via Google Vertex AI text embedding models.
pub struct GoogleVertexEmbeddings {
    /// Full predict endpoint, e.g.
    /// `https://us-central1-aiplatform.googleapis.com/v1/projects/PROJECT/locations/us-central1/publishers/google/models/textembedding-gecko:predict`
    pub endpoint: String,
    pub access_token: String,
}

impl GoogleVertexEmbeddings {
    pub fn new(endpoint: impl Into<String>, access_token: impl Into<String>) -> Self {
        Self {
            endpoint: endpoint.into(),
            access_token: access_token.into(),
        }
    }
}

#[async_trait]
impl EmbeddingsProvider for GoogleVertexEmbeddings {
    async fn embed(&self, text: &str) -> Result<Vec<f32>, EmbeddingError> {
        let resp = post_json(
            &self.endpoint,
            &self.access_token,
            json!({ "instances": [{ "content": text }] }),
        )
        .await?;
        Ok(extract_float_array(
            &resp["predictions"][0]["embeddings"]["values"],
        ))
    }

    async fn embed_batch(&self, texts: Vec<&str>) -> Result<Vec<Vec<f32>>, EmbeddingError> {
        let instances: Vec<Value> = texts
            .iter()
            .map(|t| json!({ "content": t }))
            .collect();
        let resp = post_json(
            &self.endpoint,
            &self.access_token,
            json!({ "instances": instances }),
        )
        .await?;
        Ok(resp["predictions"]
            .as_array()
            .map(|arr| {
                arr.iter()
                    .map(|p| extract_float_array(&p["embeddings"]["values"]))
                    .collect()
            })
            .unwrap_or_default())
    }

    fn get_dimension(&self) -> usize {
        768 // textembedding-gecko default
    }
}

// ── AWS Bedrock ───────────────────────────────────────────────────────────────

/// Embeddings via AWS Bedrock (Amazon Titan Embed or Cohere models).
///
/// Uses the Bedrock Runtime HTTP endpoint with AWS Signature V4.
/// For production, use the official `aws-sdk-bedrock-runtime` crate.
/// This implementation uses simple HTTP for zero-dependency embedding.
pub struct BedrockEmbeddings {
    pub region: String,
    pub model_id: String,
    pub access_key: String,
    pub secret_key: String,
}

impl BedrockEmbeddings {
    pub fn new(
        region: impl Into<String>,
        model_id: impl Into<String>,
        access_key: impl Into<String>,
        secret_key: impl Into<String>,
    ) -> Self {
        Self {
            region: region.into(),
            model_id: model_id.into(),
            access_key: access_key.into(),
            secret_key: secret_key.into(),
        }
    }

    async fn invoke(&self, body: Value) -> Result<Value, EmbeddingError> {
        // Simplified: actual AWS SigV4 signing omitted for brevity.
        // In production, integrate with aws-sigv4 crate.
        let url = format!(
            "https://bedrock-runtime.{}.amazonaws.com/model/{}/invoke",
            self.region, self.model_id
        );
        let client = reqwest::Client::new();
        let resp = client
            .post(&url)
            .header("Content-Type", "application/json")
            .header("X-Amz-Access-Key", &self.access_key)
            .json(&body)
            .send()
            .await
            .map_err(|e| EmbeddingError::Network(e.to_string()))?;
        resp.json::<Value>()
            .await
            .map_err(|e| EmbeddingError::ApiError { status: 0, message: e.to_string() })
    }
}

#[async_trait]
impl EmbeddingsProvider for BedrockEmbeddings {
    async fn embed(&self, text: &str) -> Result<Vec<f32>, EmbeddingError> {
        // Amazon Titan Embed format
        let body = if self.model_id.contains("titan") {
            json!({ "inputText": text })
        } else {
            // Cohere on Bedrock
            json!({ "texts": [text], "input_type": "search_document" })
        };
        let resp = self.invoke(body).await?;
        if self.model_id.contains("titan") {
            Ok(extract_float_array(&resp["embedding"]))
        } else {
            Ok(extract_float_array(&resp["embeddings"][0]))
        }
    }

    async fn embed_batch(&self, texts: Vec<&str>) -> Result<Vec<Vec<f32>>, EmbeddingError> {
        let mut result = Vec::new();
        for text in texts {
            result.push(self.embed(text).await?);
        }
        Ok(result)
    }

    fn get_dimension(&self) -> usize {
        1536 // Titan Embed v2 default
    }
}

// ── Voyage AI ─────────────────────────────────────────────────────────────────

/// Embeddings via Voyage AI (`voyage-2`, `voyage-code-2`, etc.).
pub struct VoyageEmbeddings {
    pub api_key: String,
    pub model: String,
    pub input_type: Option<String>,
}

impl VoyageEmbeddings {
    pub fn new(api_key: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            model: model.into(),
            input_type: None,
        }
    }

    pub fn with_input_type(mut self, input_type: impl Into<String>) -> Self {
        self.input_type = Some(input_type.into());
        self
    }
}

#[async_trait]
impl EmbeddingsProvider for VoyageEmbeddings {
    async fn embed(&self, text: &str) -> Result<Vec<f32>, EmbeddingError> {
        let mut body = json!({
            "model": self.model,
            "input": [text],
        });
        if let Some(t) = &self.input_type {
            body["input_type"] = json!(t);
        }
        let resp = post_json("https://api.voyageai.com/v1/embeddings", &self.api_key, body).await?;
        Ok(extract_float_array(&resp["data"][0]["embedding"]))
    }

    async fn embed_batch(&self, texts: Vec<&str>) -> Result<Vec<Vec<f32>>, EmbeddingError> {
        let mut body = json!({
            "model": self.model,
            "input": texts,
        });
        if let Some(t) = &self.input_type {
            body["input_type"] = json!(t);
        }
        let resp = post_json("https://api.voyageai.com/v1/embeddings", &self.api_key, body).await?;
        let mut items = resp["data"].as_array().cloned().unwrap_or_default();
        items.sort_by_key(|v| v["index"].as_u64().unwrap_or(0));
        Ok(items.iter().map(|v| extract_float_array(&v["embedding"])).collect())
    }

    fn get_dimension(&self) -> usize {
        match self.model.as_str() {
            "voyage-code-2" => 1536,
            _ => 1024,
        }
    }
}

// ── Jina AI ───────────────────────────────────────────────────────────────────

/// Embeddings via Jina AI (`jina-embeddings-v2-base-en`, etc.).
pub struct JinaEmbeddings {
    pub api_key: String,
    pub model: String,
}

impl JinaEmbeddings {
    pub fn new(api_key: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            model: model.into(),
        }
    }
}

#[async_trait]
impl EmbeddingsProvider for JinaEmbeddings {
    async fn embed(&self, text: &str) -> Result<Vec<f32>, EmbeddingError> {
        let resp = post_json(
            "https://api.jina.ai/v1/embeddings",
            &self.api_key,
            json!({ "model": self.model, "input": [text] }),
        )
        .await?;
        Ok(extract_float_array(&resp["data"][0]["embedding"]))
    }

    async fn embed_batch(&self, texts: Vec<&str>) -> Result<Vec<Vec<f32>>, EmbeddingError> {
        let resp = post_json(
            "https://api.jina.ai/v1/embeddings",
            &self.api_key,
            json!({ "model": self.model, "input": texts }),
        )
        .await?;
        let mut items = resp["data"].as_array().cloned().unwrap_or_default();
        items.sort_by_key(|v| v["index"].as_u64().unwrap_or(0));
        Ok(items.iter().map(|v| extract_float_array(&v["embedding"])).collect())
    }

    fn get_dimension(&self) -> usize {
        768 // jina-v2 base
    }
}

// ── Together AI ───────────────────────────────────────────────────────────────

/// Embeddings via Together AI (OpenAI-compatible endpoint).
pub struct TogetherEmbeddings {
    pub api_key: String,
    pub model: String,
}

impl TogetherEmbeddings {
    pub fn new(api_key: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            model: model.into(),
        }
    }
}

#[async_trait]
impl EmbeddingsProvider for TogetherEmbeddings {
    async fn embed(&self, text: &str) -> Result<Vec<f32>, EmbeddingError> {
        let resp = post_json(
            "https://api.together.xyz/v1/embeddings",
            &self.api_key,
            json!({ "model": self.model, "input": text }),
        )
        .await?;
        Ok(extract_float_array(&resp["data"][0]["embedding"]))
    }

    async fn embed_batch(&self, texts: Vec<&str>) -> Result<Vec<Vec<f32>>, EmbeddingError> {
        let mut result = Vec::new();
        for text in texts {
            result.push(self.embed(text).await?);
        }
        Ok(result)
    }

    fn get_dimension(&self) -> usize {
        4096 // Together m2-bert-80M default
    }
}

// ── Nomic ─────────────────────────────────────────────────────────────────────

/// Embeddings via Nomic Atlas (`nomic-embed-text-v1`, etc.).
pub struct NomicEmbeddings {
    pub api_key: String,
    pub model: String,
    pub task_type: String,
}

impl NomicEmbeddings {
    pub fn new(api_key: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            model: model.into(),
            task_type: "search_document".to_string(),
        }
    }

    pub fn for_query(mut self) -> Self {
        self.task_type = "search_query".to_string();
        self
    }
}

#[async_trait]
impl EmbeddingsProvider for NomicEmbeddings {
    async fn embed(&self, text: &str) -> Result<Vec<f32>, EmbeddingError> {
        let resp = post_json(
            "https://api-atlas.nomic.ai/v1/embedding/text",
            &self.api_key,
            json!({
                "model": self.model,
                "texts": [text],
                "task_type": self.task_type,
            }),
        )
        .await?;
        Ok(extract_float_array(&resp["embeddings"][0]))
    }

    async fn embed_batch(&self, texts: Vec<&str>) -> Result<Vec<Vec<f32>>, EmbeddingError> {
        let resp = post_json(
            "https://api-atlas.nomic.ai/v1/embedding/text",
            &self.api_key,
            json!({
                "model": self.model,
                "texts": texts,
                "task_type": self.task_type,
            }),
        )
        .await?;
        Ok(resp["embeddings"]
            .as_array()
            .map(|arr| arr.iter().map(|e| extract_float_array(e)).collect())
            .unwrap_or_default())
    }

    fn get_dimension(&self) -> usize {
        768
    }
}
