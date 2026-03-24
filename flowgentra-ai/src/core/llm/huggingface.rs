//! HuggingFace LLM Provider
//!
//! Supports both cloud-based HuggingFace Inference API and local models.
//!
//! ## Cloud-Based (HuggingFace Inference API)
//!
//! To use:
//! 1. Create an account at https://huggingface.co
//! 2. Get your API token from https://huggingface.co/settings/tokens
//! 3. Configure with a model from Model Hub
//!
//! ```yaml
//! llm:
//!   provider: huggingface
//!   model: mistralai/Mistral-7B-Instruct-v0.1
//!   api_key: ${HF_API_TOKEN}
//!   extra_params:
//!     mode: "inference-api"  # default
//! ```
//!
//! ## Local Models (Inference Server or TGI)
//!
//! For local deployment via HuggingFace Text Generation Inference (TGI):
//!
//! ```bash
//! # Run TGI server
//! docker run --gpus all -p 80:80 -v /tmp/data:/data \
//!     ghcr.io/huggingface/text-generation-inference:latest \
//!     --model-id mistralai/Mistral-7B-Instruct-v0.1
//! ```
//!
//! ```yaml
//! llm:
//!   provider: huggingface
//!   model: mistralai/Mistral-7B-Instruct-v0.1
//!   api_key: ""  # Not needed for local
//!   extra_params:
//!     mode: "local"
//!     endpoint: "http://localhost:80"
//! ```

use super::{LLMClient, LLMConfig, Message};
use crate::core::error::FlowgentraError;
use serde_json::json;

/// HuggingFace LLM client for cloud and local models
#[derive(Clone)]
pub struct HuggingFaceClient {
    config: LLMConfig,
    client: reqwest::Client,
    mode: HuggingFaceMode,
}

/// HuggingFace operation mode
#[derive(Debug, Clone)]
enum HuggingFaceMode {
    /// Cloud-based HuggingFace Inference API
    InferenceApi,
    /// Local TGI (Text Generation Inference) server
    LocalTgi { endpoint: String },
    /// Generic local inference server
    LocalServer { endpoint: String },
}

impl HuggingFaceClient {
    /// Create a new HuggingFace client
    pub fn new(config: LLMConfig) -> Self {
        let mode = Self::parse_mode(&config);
        Self {
            config,
            client: reqwest::Client::new(),
            mode,
        }
    }

    /// Parse the mode from configuration
    fn parse_mode(config: &LLMConfig) -> HuggingFaceMode {
        let mode_str = config
            .extra_params
            .get("mode")
            .and_then(|v| v.as_str())
            .unwrap_or("inference-api");

        match mode_str {
            "local" => {
                let endpoint = config
                    .extra_params
                    .get("endpoint")
                    .and_then(|v| v.as_str())
                    .unwrap_or("http://localhost:80")
                    .to_string();

                // Detect if it's TGI or generic server by checking endpoint
                if endpoint.contains("tgi") {
                    HuggingFaceMode::LocalTgi { endpoint }
                } else {
                    HuggingFaceMode::LocalServer { endpoint }
                }
            }
            "tgi" => {
                let endpoint = config
                    .extra_params
                    .get("endpoint")
                    .and_then(|v| v.as_str())
                    .unwrap_or("http://localhost:80")
                    .to_string();
                HuggingFaceMode::LocalTgi { endpoint }
            }
            _ => HuggingFaceMode::InferenceApi,
        }
    }

    /// Get the API endpoint URL
    fn get_endpoint(&self) -> String {
        match &self.mode {
            HuggingFaceMode::InferenceApi => {
                format!(
                    "https://api-inference.huggingface.co/models/{}",
                    self.config.model
                )
            }
            HuggingFaceMode::LocalTgi { endpoint } => {
                format!("{}/generate", endpoint)
            }
            HuggingFaceMode::LocalServer { endpoint } => {
                format!("{}/api/v1/generate", endpoint)
            }
        }
    }

    /// Build the request payload for HuggingFace API
    fn build_payload(&self, messages: Vec<Message>) -> serde_json::Value {
        // Convert messages to prompt format
        let mut prompt = String::new();
        for msg in messages {
            match msg.role {
                super::MessageRole::System => {
                    prompt.push_str(&format!("[SYSTEM] {}\n", msg.content));
                }
                super::MessageRole::User => {
                    prompt.push_str(&format!("[USER] {}\n", msg.content));
                }
                super::MessageRole::Assistant => {
                    prompt.push_str(&format!("[ASSISTANT] {}\n", msg.content));
                }
                super::MessageRole::Tool => {
                    prompt.push_str(&format!("[TOOL] {}\n", msg.content));
                }
            }
        }

        // Common parameters
        let mut payload = json!({
            "inputs": prompt,
            "parameters": {
                "temperature": self.config.temperature.unwrap_or(0.7),
                "max_new_tokens": self.config.max_tokens.unwrap_or(512),
                "top_p": self.config.top_p.unwrap_or(0.9),
                "do_sample": true,
                "return_full_text": false,
            }
        });

        // Mode-specific parameters
        match &self.mode {
            HuggingFaceMode::LocalTgi { .. } => {
                // TGI has slightly different parameter names
                if let Some(obj) = payload
                    .get_mut("parameters")
                    .and_then(|p| p.as_object_mut())
                {
                    obj.insert("best_of".to_string(), json!(1));
                    obj.insert("repetition_penalty".to_string(), json!(1.0));
                }
            }
            HuggingFaceMode::InferenceApi => {
                // Inference API specific parameters
                if let Some(obj) = payload
                    .get_mut("parameters")
                    .and_then(|p| p.as_object_mut())
                {
                    obj.insert(
                        "details".to_string(),
                        json!({
                            "best_of": 1,
                            "decoder_input_details": false,
                            "details": false,
                        }),
                    );
                }
            }
            _ => {}
        }

        payload
    }

    /// Extract response from different HuggingFace response formats
    fn extract_response(&self, response: serde_json::Value) -> crate::core::error::Result<String> {
        // Try different response formats
        if let Some(text) = response.get(0).and_then(|r| r.get("generated_text")) {
            return Ok(text.as_str().unwrap_or("").to_string());
        }

        // TGI response format
        if let Some(text) = response.get("generated_text") {
            return Ok(text.as_str().unwrap_or("").to_string());
        }

        // Generic server format
        if let Some(text) = response.get("output") {
            return Ok(text.as_str().unwrap_or("").to_string());
        }

        Err(FlowgentraError::LLMError(
            "Invalid HuggingFace response format".to_string(),
        ))
    }
}

#[async_trait::async_trait]
impl LLMClient for HuggingFaceClient {
    async fn chat(&self, messages: Vec<Message>) -> crate::core::error::Result<Message> {
        let payload = self.build_payload(messages);

        let mut request = self
            .client
            .post(self.get_endpoint())
            .header("Content-Type", "application/json");

        // Add authentication for cloud API
        if !self.config.api_key.is_empty() {
            request = request.header("Authorization", format!("Bearer {}", self.config.api_key));
        }

        let response =
            request.json(&payload).send().await.map_err(|e| {
                FlowgentraError::LLMError(format!("HuggingFace request failed: {}", e))
            })?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();
            return Err(FlowgentraError::LLMError(format!(
                "HuggingFace API error ({}): {}",
                status, error_text
            )));
        }

        let data: serde_json::Value = response.json().await.map_err(|e| {
            FlowgentraError::LLMError(format!("Failed to parse HuggingFace response: {}", e))
        })?;

        let content = self.extract_response(data)?;

        // Clean up response (remove trailing text if needed)
        let cleaned = content
            .trim_end_matches("[SYSTEM]")
            .trim_end_matches("[USER]")
            .trim_end_matches("[TOOL]")
            .trim()
            .to_string();

        Ok(Message::assistant(cleaned))
    }

    async fn chat_with_usage(
        &self,
        messages: Vec<Message>,
    ) -> crate::core::error::Result<(Message, Option<super::TokenUsage>)> {
        // HuggingFace API doesn't typically return token usage in standard responses
        // TGI does provide token counts in details
        let payload = self.build_payload(messages);

        let mut request = self
            .client
            .post(self.get_endpoint())
            .header("Content-Type", "application/json");

        // Add authentication for cloud API
        if !self.config.api_key.is_empty() {
            request = request.header("Authorization", format!("Bearer {}", self.config.api_key));
        }

        let response =
            request.json(&payload).send().await.map_err(|e| {
                FlowgentraError::LLMError(format!("HuggingFace request failed: {}", e))
            })?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();
            return Err(FlowgentraError::LLMError(format!(
                "HuggingFace API error ({}): {}",
                status, error_text
            )));
        }

        let data: serde_json::Value = response.json().await.map_err(|e| {
            FlowgentraError::LLMError(format!("Failed to parse HuggingFace response: {}", e))
        })?;

        let content = self.extract_response(data.clone())?;
        let cleaned = content
            .trim_end_matches("[SYSTEM]")
            .trim_end_matches("[USER]")
            .trim_end_matches("[TOOL]")
            .trim()
            .to_string();

        // Try to extract token usage from TGI response details
        let usage = match &self.mode {
            HuggingFaceMode::LocalTgi { .. } => {
                // TGI v1.0+ returns details object with token counts
                let prompt_tokens = data
                    .get("details")
                    .and_then(|d| d.get("prompt_tokens"))
                    .or_else(|| data.get("prompt_tokens"))
                    .and_then(|v| v.as_u64())
                    .unwrap_or(0);

                let completion_tokens = data
                    .get("details")
                    .and_then(|d| d.get("generated_tokens"))
                    .or_else(|| data.get("generated_tokens"))
                    .and_then(|v| v.as_u64())
                    .unwrap_or(0);

                if prompt_tokens > 0 || completion_tokens > 0 {
                    Some(super::TokenUsage::new(prompt_tokens, completion_tokens))
                } else {
                    None
                }
            }
            _ => None, // Inference API doesn't return usage in standard format
        };

        Ok((Message::assistant(cleaned), usage))
    }

    async fn chat_stream(
        &self,
        messages: Vec<Message>,
    ) -> crate::core::error::Result<tokio::sync::mpsc::Receiver<String>> {
        use futures::StreamExt;

        let (tx, rx) = tokio::sync::mpsc::channel(100);
        let client = self.clone();

        tokio::spawn(async move {
            // Build payload with stream: true
            let mut payload = client.build_payload(messages);
            if let Some(obj) = payload.as_object_mut() {
                obj.insert("stream".to_string(), json!(true));
            }

            let mut request = client
                .client
                .post(client.get_endpoint())
                .header("Content-Type", "application/json");

            if !client.config.api_key.is_empty() {
                request =
                    request.header("Authorization", format!("Bearer {}", client.config.api_key));
            }

            let response = match request.json(&payload).send().await {
                Ok(r) => r,
                Err(e) => {
                    let _ = tx.send(format!("ERROR: {}", e)).await;
                    return;
                }
            };

            if !response.status().is_success() {
                let status = response.status();
                let error_text = response.text().await.unwrap_or_default();
                let _ = tx.send(format!("ERROR: {} {}", status, error_text)).await;
                return;
            }

            // Read SSE stream from the response body
            let mut stream = response.bytes_stream();
            let mut buffer = String::new();

            while let Some(chunk_result) = stream.next().await {
                match chunk_result {
                    Ok(bytes) => {
                        buffer.push_str(&String::from_utf8_lossy(&bytes));

                        // Process complete SSE lines
                        while let Some(line_end) = buffer.find('\n') {
                            let line = buffer[..line_end].trim().to_string();
                            buffer = buffer[line_end + 1..].to_string();

                            if let Some(data) = line.strip_prefix("data:") {
                                let data = data.trim();
                                if data == "[DONE]" {
                                    return;
                                }
                                if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(data)
                                {
                                    // TGI SSE format: {"token": {"text": "..."}}
                                    // Inference API format: {"generated_text": "..."} or choices-based
                                    let token_text = parsed
                                        .get("token")
                                        .and_then(|t| t.get("text"))
                                        .and_then(|t| t.as_str())
                                        .or_else(|| {
                                            parsed
                                                .get("choices")
                                                .and_then(|c| c.get(0))
                                                .and_then(|c| c.get("delta"))
                                                .and_then(|d| d.get("content"))
                                                .and_then(|c| c.as_str())
                                        });

                                    if let Some(text) = token_text {
                                        if tx.send(text.to_string()).await.is_err() {
                                            return;
                                        }
                                    }
                                }
                            }
                        }
                    }
                    Err(e) => {
                        let _ = tx.send(format!("ERROR: {}", e)).await;
                        return;
                    }
                }
            }
        });

        Ok(rx)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_mode_inference_api() {
        let config = LLMConfig::new(
            super::super::LLMProvider::Custom("huggingface".to_string()),
            "mistralai/Mistral-7B".to_string(),
            "test-token".to_string(),
        );
        let client = HuggingFaceClient::new(config);
        matches!(client.mode, HuggingFaceMode::InferenceApi);
    }

    #[test]
    fn test_parse_mode_local_tgi() {
        let mut config = LLMConfig::new(
            super::super::LLMProvider::Custom("huggingface".to_string()),
            "mistralai/Mistral-7B".to_string(),
            "".to_string(),
        );
        config.extra_params.insert(
            "mode".to_string(),
            serde_json::Value::String("tgi".to_string()),
        );
        config.extra_params.insert(
            "endpoint".to_string(),
            serde_json::Value::String("http://localhost:80".to_string()),
        );

        let client = HuggingFaceClient::new(config);
        matches!(client.mode, HuggingFaceMode::LocalTgi { .. });
    }

    #[test]
    fn test_endpoint_url_inference_api() {
        let config = LLMConfig::new(
            super::super::LLMProvider::Custom("huggingface".to_string()),
            "mistralai/Mistral-7B".to_string(),
            "test-token".to_string(),
        );
        let client = HuggingFaceClient::new(config);
        assert!(client
            .get_endpoint()
            .contains("https://api-inference.huggingface.co"));
    }
}
