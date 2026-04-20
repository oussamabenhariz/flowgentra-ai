//! # LLM Provider Adapter System
//!
//! Provides a unified HTTP-based LLM that delegates provider-specific
//! behavior (payload format, auth headers, response parsing) to a `ProviderAdapter`.
//!
//! This eliminates code duplication across provider files — each provider only
//! needs to implement the adapter trait with its specific differences.

use super::{
    LLMConfig, Message, MessageRole, ResponseFormat, TokenUsage, ToolCall, ToolDefinition, LLM,
};
use crate::core::error::FlowgentraError;
use serde_json::{json, Value};

// =============================================================================
// Provider Adapter Trait
// =============================================================================

/// Trait encapsulating provider-specific HTTP LLM behavior.
///
/// Implementations define how to build payloads, set auth headers,
/// construct endpoints, and parse responses. The generic HTTP flow
/// (send request, check status, parse JSON) is handled by `HttpLLM`.
pub trait ProviderAdapter: Send + Sync {
    /// Human-readable provider name (for error messages)
    fn provider_name(&self) -> &str;

    /// The full HTTP endpoint URL for chat completions
    fn endpoint(&self, config: &LLMConfig) -> String;

    /// Build the JSON request payload from messages and config
    fn build_payload(&self, config: &LLMConfig, messages: &[Message]) -> Value;

    /// Return auth/custom headers as (header_name, header_value) pairs
    fn auth_headers(&self, config: &LLMConfig) -> Vec<(&'static str, String)>;

    /// Extract the assistant message content from the response JSON
    fn parse_response(&self, body: &Value) -> crate::core::error::Result<String>;

    /// Extract token usage from the response JSON (optional)
    fn parse_usage(&self, body: &Value) -> Option<TokenUsage> {
        let _body = body;
        None
    }

    /// Build the JSON request payload with tool definitions.
    /// Default falls back to `build_payload` (ignoring tools).
    fn build_payload_with_tools(
        &self,
        config: &LLMConfig,
        messages: &[Message],
        _tools: &[ToolDefinition],
    ) -> Value {
        self.build_payload(config, messages)
    }

    /// Parse tool calls from the response JSON.
    /// Returns None if the response is a regular text response (no tool calls).
    fn parse_tool_calls(&self, _body: &Value) -> Option<Vec<ToolCall>> {
        None
    }

    /// Extract text content from a streaming chunk
    fn parse_stream_chunk(&self, chunk: &Value) -> Option<String> {
        let _chunk = chunk;
        None
    }
}

// =============================================================================
// HTTP LLM
// =============================================================================

/// A generic HTTP-based LLM that delegates to a `ProviderAdapter`.
///
/// This struct handles the common HTTP request/response/error flow,
/// eliminating code duplication across providers.
#[derive(Clone)]
pub struct HttpLLM {
    config: LLMConfig,
    client: reqwest::Client,
    adapter: std::sync::Arc<dyn ProviderAdapter>,
}

impl HttpLLM {
    /// Create a new HTTP LLM with sensible default timeouts:
    /// - Connect timeout: 10 s
    /// - Total request timeout: 120 s
    pub fn new(config: LLMConfig, adapter: impl ProviderAdapter + 'static) -> Self {
        let client = reqwest::Client::builder()
            .connect_timeout(std::time::Duration::from_secs(10))
            .timeout(std::time::Duration::from_secs(120))
            .build()
            .unwrap_or_else(|_| reqwest::Client::new());
        Self {
            config,
            client,
            adapter: std::sync::Arc::new(adapter),
        }
    }

    /// Create with explicit timeouts.
    pub fn with_timeouts(
        config: LLMConfig,
        adapter: impl ProviderAdapter + 'static,
        connect_timeout: std::time::Duration,
        request_timeout: std::time::Duration,
    ) -> Self {
        let client = reqwest::Client::builder()
            .connect_timeout(connect_timeout)
            .timeout(request_timeout)
            .build()
            .unwrap_or_else(|_| reqwest::Client::new());
        Self {
            config,
            client,
            adapter: std::sync::Arc::new(adapter),
        }
    }
}

#[async_trait::async_trait]
impl LLM for HttpLLM {
    async fn chat(&self, messages: Vec<Message>) -> crate::core::error::Result<Message> {
        let (msg, _) = self.chat_with_usage(messages).await?;
        Ok(msg)
    }

    async fn chat_with_usage(
        &self,
        messages: Vec<Message>,
    ) -> crate::core::error::Result<(Message, Option<TokenUsage>)> {
        let name = self.adapter.provider_name();
        let endpoint = self.adapter.endpoint(&self.config);
        // Use the tool-aware payload builder so that any tool_calls / tool_call_id in
        // conversation history are preserved, even for non-tool-calling requests.
        let payload = self
            .adapter
            .build_payload_with_tools(&self.config, &messages, &[]);

        let mut request = self
            .client
            .post(&endpoint)
            .header("Content-Type", "application/json");

        // Apply provider-specific auth headers
        for (header_name, header_value) in self.adapter.auth_headers(&self.config) {
            request = request.header(header_name, header_value);
        }

        let response = request.json(&payload).send().await.map_err(|e| {
            FlowgentraError::LLMError(format!("{} API request failed: {}", name, e))
        })?;

        if !response.status().is_success() {
            let error_text = response.text().await.unwrap_or_default();
            return Err(FlowgentraError::LLMError(format!(
                "{} API error: {}",
                name, error_text
            )));
        }

        let data: Value = response.json().await.map_err(|e| {
            FlowgentraError::LLMError(format!("Failed to parse {} response: {}", name, e))
        })?;

        let content = self.adapter.parse_response(&data)?;
        let usage = self.adapter.parse_usage(&data);

        Ok((Message::assistant(content), usage))
    }

    async fn chat_with_tools(
        &self,
        messages: Vec<Message>,
        tools: &[ToolDefinition],
    ) -> crate::core::error::Result<Message> {
        let name = self.adapter.provider_name();
        let endpoint = self.adapter.endpoint(&self.config);
        let payload = self
            .adapter
            .build_payload_with_tools(&self.config, &messages, tools);

        let mut request = self
            .client
            .post(&endpoint)
            .header("Content-Type", "application/json");

        for (header_name, header_value) in self.adapter.auth_headers(&self.config) {
            request = request.header(header_name, header_value);
        }

        let response = request.json(&payload).send().await.map_err(|e| {
            FlowgentraError::LLMError(format!("{} API request failed: {}", name, e))
        })?;

        if !response.status().is_success() {
            let error_text = response.text().await.unwrap_or_default();
            return Err(FlowgentraError::LLMError(format!(
                "{} API error: {}",
                name, error_text
            )));
        }

        let data: Value = response.json().await.map_err(|e| {
            FlowgentraError::LLMError(format!("Failed to parse {} response: {}", name, e))
        })?;

        // Check for tool calls first
        if let Some(tool_calls) = self.adapter.parse_tool_calls(&data) {
            let content = data
                .get("choices")
                .and_then(|c| c.get(0))
                .and_then(|c| c.get("message"))
                .and_then(|m| m.get("content"))
                .and_then(|c| c.as_str())
                .unwrap_or("")
                .to_string();

            return Ok(Message {
                role: MessageRole::Assistant,
                content,
                tool_calls: Some(tool_calls),
                tool_call_id: None,
            });
        }

        // Regular text response
        let content = self.adapter.parse_response(&data)?;
        Ok(Message::assistant(content))
    }

    async fn chat_stream(
        &self,
        messages: Vec<Message>,
    ) -> crate::core::error::Result<tokio::sync::mpsc::Receiver<String>> {
        let name = self.adapter.provider_name();
        let endpoint = self.adapter.endpoint(&self.config);
        let mut payload = self.adapter.build_payload(&self.config, &messages);

        if let Some(obj) = payload.as_object_mut() {
            obj.insert("stream".to_string(), json!(true));
        }

        let mut request = self
            .client
            .post(&endpoint)
            .header("Content-Type", "application/json")
            .header("Accept", "text/event-stream");

        for (header_name, header_value) in self.adapter.auth_headers(&self.config) {
            request = request.header(header_name, header_value);
        }

        let response = request.json(&payload).send().await.map_err(|e| {
            FlowgentraError::LLMError(format!("{} API streaming request failed: {}", name, e))
        })?;

        if !response.status().is_success() {
            let error_text = response.text().await.unwrap_or_default();
            return Err(FlowgentraError::LLMError(format!(
                "{} API stream error: {}",
                name, error_text
            )));
        }

        let (tx, rx) = tokio::sync::mpsc::channel(100);
        let adapter = self.adapter.clone();
        let provider_name = name.to_string();

        tokio::spawn(async move {
            use futures::StreamExt;
            let mut stream = response.bytes_stream();
            let mut buffer = String::new();

            while let Some(chunk_result) = stream.next().await {
                match chunk_result {
                    Ok(bytes) => {
                        if let Ok(text) = std::str::from_utf8(&bytes) {
                            buffer.push_str(text);

                            while let Some(pos) = buffer.find('\n') {
                                let line = buffer[..pos].trim().to_string();
                                buffer = buffer[pos + 1..].to_string();

                                if let Some(data) = line.strip_prefix("data: ") {
                                    if data == "[DONE]" {
                                        return;
                                    }

                                    if let Ok(json) = serde_json::from_str::<Value>(data) {
                                        if let Some(content) = adapter.parse_stream_chunk(&json) {
                                            if !content.is_empty()
                                                && tx.send(content).await.is_err()
                                            {
                                                return; // Receiver dropped, stop streaming
                                            }
                                        }
                                    }
                                } else if line.starts_with("{") {
                                    // Handle NDJSON streams (e.g. Ollama)
                                    if let Ok(json) = serde_json::from_str::<Value>(&line) {
                                        if let Some(content) = adapter.parse_stream_chunk(&json) {
                                            if !content.is_empty()
                                                && tx.send(content).await.is_err()
                                            {
                                                return;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                    Err(e) => {
                        tracing::error!("Stream error from {}: {}", provider_name, e);
                        // Signal the error to the consumer so it can distinguish
                        // a mid-stream failure from a clean end-of-stream.
                        let _ = tx.send(format!("[STREAM_ERROR: {}]", e)).await;
                        break;
                    }
                }
            }
        });

        Ok(rx)
    }
}

// =============================================================================
// Helper: standard OpenAI-style message formatting
// =============================================================================

/// Apply `response_format` to an OpenAI-compatible payload if set.
fn apply_openai_response_format(payload: &mut Value, config: &LLMConfig) {
    match &config.response_format {
        ResponseFormat::Text => {}
        ResponseFormat::Json => {
            payload["response_format"] = json!({"type": "json_object"});
        }
        ResponseFormat::JsonSchema { name, schema } => {
            payload["response_format"] = json!({
                "type": "json_schema",
                "json_schema": {
                    "name": name,
                    "schema": schema,
                    "strict": true,
                }
            });
        }
    }
}

/// Convert messages to the OpenAI-compatible JSON array format.
/// Used by OpenAI, Mistral, Groq, Azure.
pub fn openai_messages_payload(messages: &[Message]) -> Vec<Value> {
    messages
        .iter()
        .map(|m| {
            json!({
                "role": match m.role {
                    MessageRole::System => "system",
                    MessageRole::User => "user",
                    MessageRole::Assistant => "assistant",
                    MessageRole::Tool => "tool",
                },
                "content": m.content,
            })
        })
        .collect()
}

/// Convert messages to the OpenAI-compatible format with tool call/result support.
///
/// Unlike `openai_messages_payload`, this includes `tool_calls` on assistant
/// messages and `tool_call_id` on tool result messages.
pub fn openai_messages_with_tools_payload(messages: &[Message]) -> Vec<Value> {
    messages
        .iter()
        .map(|m| {
            let role = match m.role {
                MessageRole::System => "system",
                MessageRole::User => "user",
                MessageRole::Assistant => "assistant",
                MessageRole::Tool => "tool",
            };

            let mut msg = json!({"role": role});

            // Assistant messages may have tool_calls
            if let Some(ref tool_calls) = m.tool_calls {
                if !tool_calls.is_empty() {
                    let tc: Vec<Value> = tool_calls
                        .iter()
                        .map(|tc| {
                            let args_str = if tc.arguments.is_string() {
                                tc.arguments.as_str().unwrap_or("{}").to_string()
                            } else {
                                serde_json::to_string(&tc.arguments)
                                    .unwrap_or_else(|_| "{}".to_string())
                            };
                            json!({
                                "id": tc.id,
                                "type": "function",
                                "function": {
                                    "name": tc.name,
                                    "arguments": args_str,
                                }
                            })
                        })
                        .collect();
                    msg["tool_calls"] = json!(tc);
                    // When tool_calls present, content can be null
                    msg["content"] = if m.content.is_empty() {
                        Value::Null
                    } else {
                        json!(m.content)
                    };
                    return msg;
                }
            }

            // Tool result messages need tool_call_id
            if m.role == MessageRole::Tool {
                if let Some(ref id) = m.tool_call_id {
                    msg["tool_call_id"] = json!(id);
                }
            }

            msg["content"] = json!(m.content);
            msg
        })
        .collect()
}

/// Parse tool calls from OpenAI-compatible `choices[0].message.tool_calls`.
pub fn openai_parse_tool_calls(body: &Value) -> Option<Vec<ToolCall>> {
    let tool_calls = body
        .get("choices")
        .and_then(|c| c.get(0))
        .and_then(|c| c.get("message"))
        .and_then(|m| m.get("tool_calls"))
        .and_then(|tc| tc.as_array())?;

    if tool_calls.is_empty() {
        return None;
    }

    let calls: Vec<ToolCall> = tool_calls
        .iter()
        .filter_map(|tc| {
            let id = tc.get("id")?.as_str()?.to_string();
            let func = tc.get("function")?;
            let name = func.get("name")?.as_str()?.to_string();
            let args_str = func.get("arguments")?.as_str().unwrap_or("{}");
            let arguments = serde_json::from_str(args_str).unwrap_or(json!({}));
            Some(ToolCall {
                id,
                name,
                arguments,
            })
        })
        .collect();

    if calls.is_empty() {
        None
    } else {
        Some(calls)
    }
}

/// Parse assistant content from OpenAI-compatible `choices[0].message.content`.
pub fn openai_parse_response(body: &Value) -> crate::core::error::Result<String> {
    body.get("choices")
        .and_then(|c| c.get(0))
        .and_then(|c| c.get("message"))
        .and_then(|m| m.get("content"))
        .and_then(|c| c.as_str())
        .map(|s| s.to_string())
        .ok_or_else(|| {
            FlowgentraError::LLMError(
                "Invalid response format: missing choices[0].message.content".to_string(),
            )
        })
}

/// Parse token usage from OpenAI-compatible `usage` object.
pub fn openai_parse_usage(body: &Value) -> Option<TokenUsage> {
    body.get("usage").map(|u| TokenUsage {
        prompt_tokens: u.get("prompt_tokens").and_then(|v| v.as_u64()).unwrap_or(0),
        completion_tokens: u
            .get("completion_tokens")
            .and_then(|v| v.as_u64())
            .unwrap_or(0),
        total_tokens: u.get("total_tokens").and_then(|v| v.as_u64()).unwrap_or(0),
    })
}

// =============================================================================
// Concrete Adapters
// =============================================================================

/// Adapter for OpenAI-compatible APIs (OpenAI, Mistral, Groq).
///
/// These providers share the exact same payload format, auth header,
/// endpoint pattern, and response structure.
pub struct OpenAICompatibleAdapter {
    name: &'static str,
}

impl OpenAICompatibleAdapter {
    pub fn new(name: &'static str) -> Self {
        Self { name }
    }
}

impl ProviderAdapter for OpenAICompatibleAdapter {
    fn provider_name(&self) -> &str {
        self.name
    }

    fn endpoint(&self, config: &LLMConfig) -> String {
        format!("{}/chat/completions", config.provider.base_url())
    }

    fn build_payload(&self, config: &LLMConfig, messages: &[Message]) -> Value {
        let mut payload = json!({
            "model": config.model,
            "messages": openai_messages_payload(messages),
            "temperature": config.temperature.unwrap_or(0.7),
            "max_tokens": config.max_tokens.unwrap_or(2048),
            "top_p": config.top_p.unwrap_or(1.0),
        });
        apply_openai_response_format(&mut payload, config);
        payload
    }

    fn build_payload_with_tools(
        &self,
        config: &LLMConfig,
        messages: &[Message],
        tools: &[ToolDefinition],
    ) -> Value {
        let mut payload = json!({
            "model": config.model,
            "messages": openai_messages_with_tools_payload(messages),
            "temperature": config.temperature.unwrap_or(0.7),
            "max_tokens": config.max_tokens.unwrap_or(2048),
            "top_p": config.top_p.unwrap_or(1.0),
        });

        if !tools.is_empty() {
            let tool_defs: Vec<Value> = tools
                .iter()
                .map(|t| {
                    json!({
                        "type": "function",
                        "function": {
                            "name": t.name,
                            "description": t.description,
                            "parameters": t.parameters,
                        }
                    })
                })
                .collect();
            payload["tools"] = json!(tool_defs);
            payload["tool_choice"] = json!("auto");
        }

        payload
    }

    fn parse_tool_calls(&self, body: &Value) -> Option<Vec<ToolCall>> {
        openai_parse_tool_calls(body)
    }

    fn auth_headers(&self, config: &LLMConfig) -> Vec<(&'static str, String)> {
        vec![("Authorization", format!("Bearer {}", config.api_key))]
    }

    fn parse_response(&self, body: &Value) -> crate::core::error::Result<String> {
        openai_parse_response(body)
    }

    fn parse_usage(&self, body: &Value) -> Option<TokenUsage> {
        openai_parse_usage(body)
    }

    fn parse_stream_chunk(&self, chunk: &Value) -> Option<String> {
        chunk
            .get("choices")
            .and_then(|c| c.get(0))
            .and_then(|c| c.get("delta"))
            .and_then(|d| d.get("content"))
            .and_then(|c| c.as_str())
            .map(|s| s.to_string())
    }
}

// -----------------------------------------------------------------------------

/// Adapter for Anthropic Claude API.
///
/// Differences from OpenAI: system message is a separate top-level param,
/// auth uses `x-api-key` header, response is `content[0].text`.
pub struct AnthropicAdapter;

impl ProviderAdapter for AnthropicAdapter {
    fn provider_name(&self) -> &str {
        "Anthropic"
    }

    fn endpoint(&self, config: &LLMConfig) -> String {
        format!("{}/messages", config.provider.base_url())
    }

    fn build_payload(&self, config: &LLMConfig, messages: &[Message]) -> Value {
        anthropic_build_payload(config, messages)
    }

    fn build_payload_with_tools(
        &self,
        config: &LLMConfig,
        messages: &[Message],
        tools: &[ToolDefinition],
    ) -> Value {
        let mut payload = anthropic_build_payload(config, messages);

        if !tools.is_empty() {
            // Anthropic uses `input_schema` instead of OpenAI's `parameters`
            let anthropic_tools: Vec<Value> = tools
                .iter()
                .map(|t| {
                    json!({
                        "name": t.name,
                        "description": t.description,
                        "input_schema": t.parameters,
                    })
                })
                .collect();
            payload["tools"] = json!(anthropic_tools);
        }

        payload
    }

    fn auth_headers(&self, config: &LLMConfig) -> Vec<(&'static str, String)> {
        vec![
            ("x-api-key", config.api_key.clone()),
            ("anthropic-version", "2023-06-01".to_string()),
        ]
    }

    fn parse_response(&self, body: &Value) -> crate::core::error::Result<String> {
        // Anthropic returns content as an array of blocks.
        // Look for the first `text` block.
        let content = body
            .get("content")
            .and_then(|c| c.as_array())
            .ok_or_else(|| {
                FlowgentraError::LLMError("Invalid Anthropic response format".to_string())
            })?;

        for block in content {
            if block.get("type").and_then(|t| t.as_str()) == Some("text") {
                if let Some(text) = block.get("text").and_then(|t| t.as_str()) {
                    return Ok(text.to_string());
                }
            }
        }

        // Fallback: if only tool_use blocks, return empty string
        Ok(String::new())
    }

    fn parse_usage(&self, body: &Value) -> Option<TokenUsage> {
        let usage = body.get("usage")?;
        Some(TokenUsage {
            prompt_tokens: usage
                .get("input_tokens")
                .and_then(|v| v.as_u64())
                .unwrap_or(0),
            completion_tokens: usage
                .get("output_tokens")
                .and_then(|v| v.as_u64())
                .unwrap_or(0),
            total_tokens: usage
                .get("input_tokens")
                .and_then(|v| v.as_u64())
                .unwrap_or(0)
                + usage
                    .get("output_tokens")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(0),
        })
    }

    fn parse_tool_calls(&self, body: &Value) -> Option<Vec<ToolCall>> {
        let content = body.get("content")?.as_array()?;

        let calls: Vec<ToolCall> = content
            .iter()
            .filter_map(|block| {
                if block.get("type")?.as_str()? != "tool_use" {
                    return None;
                }
                let id = block.get("id")?.as_str()?.to_string();
                let name = block.get("name")?.as_str()?.to_string();
                let arguments = block.get("input").cloned().unwrap_or(json!({}));
                Some(ToolCall {
                    id,
                    name,
                    arguments,
                })
            })
            .collect();

        if calls.is_empty() {
            None
        } else {
            Some(calls)
        }
    }

    fn parse_stream_chunk(&self, chunk: &Value) -> Option<String> {
        let event_type = chunk.get("type").and_then(|t| t.as_str()).unwrap_or("");
        if event_type == "content_block_delta" {
            chunk
                .get("delta")
                .and_then(|d| d.get("text"))
                .and_then(|t| t.as_str())
                .map(|s| s.to_string())
        } else {
            None
        }
    }
}

/// Build the Anthropic messages payload (shared between build_payload and build_payload_with_tools).
fn anthropic_build_payload(config: &LLMConfig, messages: &[Message]) -> Value {
    // Collect and concatenate all system messages (multiple system messages are valid).
    let mut system_parts: Vec<String> = Vec::new();
    let chat_messages: Vec<Value> = messages
        .iter()
        .filter_map(|m| {
            if m.role == MessageRole::System {
                system_parts.push(m.content.clone());
                return None;
            }
            // Tool results require Anthropic's structured content format.
            if m.role == MessageRole::Tool {
                let tool_use_id = m.tool_call_id.as_deref().unwrap_or("");
                return Some(json!({
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": tool_use_id,
                        "content": m.content,
                    }]
                }));
            }
            Some(json!({
                "role": match m.role {
                    MessageRole::User => "user",
                    MessageRole::Assistant => "assistant",
                    _ => "user",
                },
                "content": m.content,
            }))
        })
        .collect();

    let mut system_prompt = system_parts.join("\n\n");

    // For JSON mode, append instruction to the system prompt
    match &config.response_format {
        ResponseFormat::Json => {
            if !system_prompt.is_empty() {
                system_prompt.push_str("\n\n");
            }
            system_prompt.push_str(
                "You must respond with valid JSON only. No markdown formatting, no explanations, just raw JSON."
            );
        }
        ResponseFormat::JsonSchema { schema, .. } => {
            if !system_prompt.is_empty() {
                system_prompt.push_str("\n\n");
            }
            system_prompt.push_str(&format!(
                "You must respond with valid JSON matching this schema:\n{}\nNo markdown formatting, no explanations, just raw JSON.",
                serde_json::to_string_pretty(schema).unwrap_or_default()
            ));
        }
        ResponseFormat::Text => {}
    }

    let mut payload = json!({
        "model": config.model,
        "messages": chat_messages,
        "max_tokens": config.max_tokens.unwrap_or(2048),
    });

    if !system_prompt.is_empty() {
        payload["system"] = json!(system_prompt);
    }
    if let Some(temp) = config.temperature {
        payload["temperature"] = json!(temp);
    }
    if let Some(top_p) = config.top_p {
        payload["top_p"] = json!(top_p);
    }

    payload
}

// -----------------------------------------------------------------------------

/// Adapter for Ollama local LLM API.
///
/// Differences: endpoint is `/chat`, payload uses `options` block for params,
/// requires `stream: false`, response is `message.content`.
pub struct OllamaAdapter;

impl ProviderAdapter for OllamaAdapter {
    fn provider_name(&self) -> &str {
        "Ollama"
    }

    fn endpoint(&self, config: &LLMConfig) -> String {
        format!("{}/chat", config.provider.base_url())
    }

    fn build_payload(&self, config: &LLMConfig, messages: &[Message]) -> Value {
        json!({
            "model": config.model,
            "messages": openai_messages_payload(messages),
            "stream": false,
            "options": {
                "temperature": config.temperature.unwrap_or(0.7),
                "num_predict": config.max_tokens.unwrap_or(2048),
                "top_p": config.top_p.unwrap_or(1.0),
            }
        })
    }

    fn auth_headers(&self, _config: &LLMConfig) -> Vec<(&'static str, String)> {
        vec![] // Ollama runs locally, no auth needed
    }

    fn parse_response(&self, body: &Value) -> crate::core::error::Result<String> {
        body.get("message")
            .and_then(|m| m.get("content"))
            .and_then(|c| c.as_str())
            .map(|s| s.to_string())
            .ok_or_else(|| FlowgentraError::LLMError("Invalid Ollama response format".to_string()))
    }

    fn parse_stream_chunk(&self, chunk: &Value) -> Option<String> {
        chunk
            .get("message")
            .and_then(|m| m.get("content"))
            .and_then(|c| c.as_str())
            .map(|s| s.to_string())
    }

    fn parse_usage(&self, body: &Value) -> Option<TokenUsage> {
        // Ollama returns eval_count and prompt_eval_count in final response
        let completion_tokens = body.get("eval_count").and_then(|v| v.as_u64()).unwrap_or(0);
        let prompt_tokens = body
            .get("prompt_eval_count")
            .and_then(|v| v.as_u64())
            .unwrap_or(0);
        let total_tokens = prompt_tokens + completion_tokens;

        if total_tokens > 0 {
            Some(TokenUsage {
                prompt_tokens,
                completion_tokens,
                total_tokens,
            })
        } else {
            None
        }
    }
}

// -----------------------------------------------------------------------------

/// Adapter for Azure OpenAI deployments.
///
/// Similar to OpenAI but: auth uses `api-key` header, endpoint includes
/// deployment ID and api-version query param, no `model` param in payload.
pub struct AzureAdapter {
    deployment_id: String,
    api_version: String,
    base_url: String,
}

impl AzureAdapter {
    /// Create a new Azure adapter from the config's extra_params.
    pub fn from_config(config: &LLMConfig) -> crate::core::error::Result<Self> {
        let resource_name = config
            .extra_params
            .get("resource_name")
            .and_then(|v| v.as_str())
            .ok_or_else(|| {
                FlowgentraError::ConfigError(
                    "Azure OpenAI requires 'resource_name' in extra_params".to_string(),
                )
            })?
            .to_string();

        let api_version = config
            .extra_params
            .get("api_version")
            .and_then(|v| v.as_str())
            .unwrap_or("2024-02-15-preview")
            .to_string();

        Ok(Self {
            deployment_id: config.model.clone(),
            api_version,
            base_url: format!(
                "https://{}.openai.azure.com/openai/deployments",
                resource_name
            ),
        })
    }
}

impl ProviderAdapter for AzureAdapter {
    fn provider_name(&self) -> &str {
        "Azure OpenAI"
    }

    fn endpoint(&self, _config: &LLMConfig) -> String {
        format!(
            "{}/{}/chat/completions?api-version={}",
            self.base_url, self.deployment_id, self.api_version
        )
    }

    fn build_payload(&self, config: &LLMConfig, messages: &[Message]) -> Value {
        // Azure doesn't need the model field (it's in the URL via deployment ID)
        let mut payload = json!({
            "messages": openai_messages_payload(messages),
            "temperature": config.temperature.unwrap_or(0.7),
            "max_tokens": config.max_tokens.unwrap_or(2048),
            "top_p": config.top_p.unwrap_or(1.0),
        });
        apply_openai_response_format(&mut payload, config);
        payload
    }

    fn auth_headers(&self, config: &LLMConfig) -> Vec<(&'static str, String)> {
        vec![("api-key", config.api_key.clone())]
    }

    fn parse_response(&self, body: &Value) -> crate::core::error::Result<String> {
        openai_parse_response(body)
    }

    fn parse_usage(&self, body: &Value) -> Option<TokenUsage> {
        openai_parse_usage(body)
    }

    fn parse_stream_chunk(&self, chunk: &Value) -> Option<String> {
        // Azure follows OpenAI format for streaming
        chunk
            .get("choices")
            .and_then(|c| c.get(0))
            .and_then(|c| c.get("delta"))
            .and_then(|d| d.get("content"))
            .and_then(|c| c.as_str())
            .map(|s| s.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::llm::LLMProvider;

    #[test]
    fn test_openai_adapter_endpoint() {
        let config = LLMConfig::new(LLMProvider::OpenAI, "gpt-4".into(), "key".into());
        let adapter = OpenAICompatibleAdapter::new("OpenAI");
        assert!(adapter.endpoint(&config).contains("/chat/completions"));
    }

    #[test]
    fn test_openai_adapter_payload() {
        let config = LLMConfig::new(LLMProvider::OpenAI, "gpt-4".into(), "key".into());
        let adapter = OpenAICompatibleAdapter::new("OpenAI");
        let msgs = vec![Message::user("hello")];
        let payload = adapter.build_payload(&config, &msgs);
        assert_eq!(payload["model"], "gpt-4");
        assert!(payload["messages"].is_array());
    }

    #[test]
    fn test_anthropic_adapter_system_extraction() {
        let config = LLMConfig::new(LLMProvider::Anthropic, "claude-3".into(), "key".into());
        let adapter = AnthropicAdapter;
        let msgs = vec![Message::system("Be helpful"), Message::user("hello")];
        let payload = adapter.build_payload(&config, &msgs);
        assert_eq!(payload["system"], "Be helpful");
        // System message should not be in the messages array
        let messages = payload["messages"].as_array().unwrap();
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0]["role"], "user");
    }

    #[test]
    fn test_ollama_adapter_payload() {
        let config = LLMConfig::new(LLMProvider::Ollama, "mistral".into(), "".into());
        let adapter = OllamaAdapter;
        let msgs = vec![Message::user("hello")];
        let payload = adapter.build_payload(&config, &msgs);
        assert_eq!(payload["stream"], false);
        assert!(payload["options"].is_object());
    }

    #[test]
    fn test_openai_parse_response() {
        let body = json!({
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "Hello!"
                }
            }]
        });
        assert_eq!(openai_parse_response(&body).unwrap(), "Hello!");
    }

    #[test]
    fn test_openai_parse_usage() {
        let body = json!({
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "total_tokens": 30
            }
        });
        let usage = openai_parse_usage(&body).unwrap();
        assert_eq!(usage.prompt_tokens, 10);
        assert_eq!(usage.completion_tokens, 20);
        assert_eq!(usage.total_tokens, 30);
    }
}
