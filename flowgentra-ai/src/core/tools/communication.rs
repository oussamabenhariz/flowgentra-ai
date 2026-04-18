//! Communication tools: Gmail and Slack.

use super::{JsonSchema, Tool, ToolDefinition};
use crate::prelude::*;
use async_trait::async_trait;
use serde_json::{json, Value};
use std::collections::HashMap;

// =============================================================================
// GmailTool
// =============================================================================

/// Interact with Gmail via the Google Gmail REST API.
///
/// Operations: `send`, `list`, `read`.
///
/// Requires a valid OAuth 2.0 access token with Gmail scopes.
pub struct GmailTool {
    access_token: String,
    client: reqwest::Client,
}

impl GmailTool {
    pub fn new(access_token: impl Into<String>) -> Self {
        Self {
            access_token: access_token.into(),
            client: reqwest::Client::new(),
        }
    }

    pub fn from_env() -> Result<Self> {
        let token = std::env::var("GMAIL_ACCESS_TOKEN").map_err(|_| {
            FlowgentraError::ToolError(
                "GMAIL_ACCESS_TOKEN environment variable not set".to_string(),
            )
        })?;
        Ok(Self::new(token))
    }

    fn auth_header(&self) -> String {
        format!("Bearer {}", self.access_token)
    }

    async fn send_message(&self, to: &str, subject: &str, body: &str) -> Result<Value> {
        // Compose an RFC 2822 message and base64url-encode it
        let raw_message = format!(
            "To: {}\r\nSubject: {}\r\nContent-Type: text/plain; charset=UTF-8\r\n\r\n{}",
            to, subject, body
        );
        // base64url encode (URL-safe, no padding)
        let encoded = base64_url_encode(raw_message.as_bytes());

        let payload = json!({"raw": encoded});

        let resp: Value = self
            .client
            .post("https://gmail.googleapis.com/gmail/v1/users/me/messages/send")
            .header("Authorization", self.auth_header())
            .json(&payload)
            .send()
            .await
            .map_err(|e| FlowgentraError::ToolError(format!("Gmail send failed: {}", e)))?
            .json()
            .await
            .map_err(|e| FlowgentraError::ToolError(format!("Gmail send parse failed: {}", e)))?;

        Ok(resp)
    }

    async fn list_messages(&self, max: u64, query: &str) -> Result<Value> {
        let url = format!(
            "https://gmail.googleapis.com/gmail/v1/users/me/messages?maxResults={}&q={}",
            max,
            urlencoding::encode(query)
        );

        let resp: Value = self
            .client
            .get(&url)
            .header("Authorization", self.auth_header())
            .send()
            .await
            .map_err(|e| FlowgentraError::ToolError(format!("Gmail list failed: {}", e)))?
            .json()
            .await
            .map_err(|e| FlowgentraError::ToolError(format!("Gmail list parse failed: {}", e)))?;

        Ok(resp)
    }

    async fn read_message(&self, message_id: &str) -> Result<Value> {
        let url = format!(
            "https://gmail.googleapis.com/gmail/v1/users/me/messages/{}?format=full",
            message_id
        );

        let resp: Value = self
            .client
            .get(&url)
            .header("Authorization", self.auth_header())
            .send()
            .await
            .map_err(|e| FlowgentraError::ToolError(format!("Gmail read failed: {}", e)))?
            .json()
            .await
            .map_err(|e| FlowgentraError::ToolError(format!("Gmail read parse failed: {}", e)))?;

        Ok(resp)
    }
}

/// Simple base64url encoding (RFC 4648 §5) without padding.
fn base64_url_encode(data: &[u8]) -> String {
    const CHARS: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_";
    let mut out = String::with_capacity((data.len() * 4).div_ceil(3));
    for chunk in data.chunks(3) {
        let b0 = chunk[0] as usize;
        let b1 = chunk.get(1).copied().unwrap_or(0) as usize;
        let b2 = chunk.get(2).copied().unwrap_or(0) as usize;
        let n = (b0 << 16) | (b1 << 8) | b2;
        out.push(CHARS[(n >> 18) & 63] as char);
        out.push(CHARS[(n >> 12) & 63] as char);
        if chunk.len() > 1 {
            out.push(CHARS[(n >> 6) & 63] as char);
        }
        if chunk.len() > 2 {
            out.push(CHARS[n & 63] as char);
        }
    }
    out
}

#[async_trait]
impl Tool for GmailTool {
    async fn call(&self, input: Value) -> Result<Value> {
        let operation = input
            .get("operation")
            .and_then(|v| v.as_str())
            .ok_or_else(|| FlowgentraError::ToolError("Missing 'operation' field".to_string()))?;

        match operation {
            "send" => {
                let to = input.get("to").and_then(|v| v.as_str()).ok_or_else(|| {
                    FlowgentraError::ToolError("Missing 'to' field for send".to_string())
                })?;
                let subject = input
                    .get("subject")
                    .and_then(|v| v.as_str())
                    .unwrap_or("(no subject)");
                let body = input.get("body").and_then(|v| v.as_str()).unwrap_or("");
                let resp = self.send_message(to, subject, body).await?;
                Ok(json!({"operation": "send", "result": resp}))
            }
            "list" => {
                let max = input
                    .get("max_results")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(10);
                let query = input.get("query").and_then(|v| v.as_str()).unwrap_or("");
                let resp = self.list_messages(max, query).await?;
                Ok(json!({"operation": "list", "result": resp}))
            }
            "read" => {
                let message_id = input
                    .get("message_id")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| {
                        FlowgentraError::ToolError(
                            "Missing 'message_id' field for read".to_string(),
                        )
                    })?;
                let resp = self.read_message(message_id).await?;
                Ok(json!({"operation": "read", "result": resp}))
            }
            other => Err(FlowgentraError::ToolError(format!(
                "Unknown Gmail operation: '{}'. Use: send, list, read",
                other
            ))),
        }
    }

    fn definition(&self) -> ToolDefinition {
        let mut props = HashMap::new();
        props.insert(
            "operation".to_string(),
            JsonSchema::string().with_description("Gmail operation: send, list, or read"),
        );
        props.insert(
            "to".to_string(),
            JsonSchema::string().with_description("Recipient email (for send)"),
        );
        props.insert(
            "subject".to_string(),
            JsonSchema::string().with_description("Email subject (for send)"),
        );
        props.insert(
            "body".to_string(),
            JsonSchema::string().with_description("Email body text (for send)"),
        );
        props.insert(
            "max_results".to_string(),
            JsonSchema::integer()
                .with_description("Max messages to return (for list, default: 10)"),
        );
        props.insert(
            "query".to_string(),
            JsonSchema::string().with_description("Gmail search query (for list)"),
        );
        props.insert(
            "message_id".to_string(),
            JsonSchema::string().with_description("Message ID to fetch (for read)"),
        );

        ToolDefinition::new(
            "gmail",
            "Send, list, and read Gmail messages via Gmail API (requires GMAIL_ACCESS_TOKEN)",
            JsonSchema::object()
                .with_properties(props)
                .with_required(vec!["operation".to_string()]),
            JsonSchema::object(),
        )
        .with_category("communication")
    }
}

// =============================================================================
// SlackTool
// =============================================================================

/// Post messages and list channels via the Slack Web API.
pub struct SlackTool {
    bot_token: String,
    client: reqwest::Client,
}

impl SlackTool {
    pub fn new(bot_token: impl Into<String>) -> Self {
        Self {
            bot_token: bot_token.into(),
            client: reqwest::Client::new(),
        }
    }

    pub fn from_env() -> Result<Self> {
        let token = std::env::var("SLACK_BOT_TOKEN").map_err(|_| {
            FlowgentraError::ToolError("SLACK_BOT_TOKEN environment variable not set".to_string())
        })?;
        Ok(Self::new(token))
    }

    fn auth_header(&self) -> String {
        format!("Bearer {}", self.bot_token)
    }
}

#[async_trait]
impl Tool for SlackTool {
    async fn call(&self, input: Value) -> Result<Value> {
        let operation = input
            .get("operation")
            .and_then(|v| v.as_str())
            .ok_or_else(|| FlowgentraError::ToolError("Missing 'operation' field".to_string()))?;

        match operation {
            "post_message" => {
                let channel = input
                    .get("channel")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| {
                        FlowgentraError::ToolError("Missing 'channel' field".to_string())
                    })?;
                let text = input.get("text").and_then(|v| v.as_str()).ok_or_else(|| {
                    FlowgentraError::ToolError("Missing 'text' field".to_string())
                })?;

                let payload = json!({"channel": channel, "text": text});

                let resp: Value = self
                    .client
                    .post("https://slack.com/api/chat.postMessage")
                    .header("Authorization", self.auth_header())
                    .json(&payload)
                    .send()
                    .await
                    .map_err(|e| FlowgentraError::ToolError(format!("Slack post failed: {}", e)))?
                    .json()
                    .await
                    .map_err(|e| {
                        FlowgentraError::ToolError(format!("Slack post parse failed: {}", e))
                    })?;

                if resp.get("ok").and_then(|v| v.as_bool()) == Some(false) {
                    let err = resp
                        .get("error")
                        .and_then(|v| v.as_str())
                        .unwrap_or("unknown_error");
                    return Err(FlowgentraError::ToolError(format!(
                        "Slack API error: {}",
                        err
                    )));
                }

                Ok(json!({"operation": "post_message", "channel": channel, "ok": true}))
            }
            "list_channels" => {
                let resp: Value = self
                    .client
                    .get("https://slack.com/api/conversations.list")
                    .header("Authorization", self.auth_header())
                    .send()
                    .await
                    .map_err(|e| {
                        FlowgentraError::ToolError(format!("Slack list_channels failed: {}", e))
                    })?
                    .json()
                    .await
                    .map_err(|e| {
                        FlowgentraError::ToolError(format!(
                            "Slack list_channels parse failed: {}",
                            e
                        ))
                    })?;

                let channels: Vec<Value> = resp
                    .get("channels")
                    .and_then(|v| v.as_array())
                    .map(|arr| {
                        arr.iter()
                            .map(|c| {
                                json!({
                                    "id": c.get("id").and_then(|v| v.as_str()).unwrap_or(""),
                                    "name": c.get("name").and_then(|v| v.as_str()).unwrap_or(""),
                                    "is_private": c.get("is_private").and_then(|v| v.as_bool()).unwrap_or(false),
                                })
                            })
                            .collect()
                    })
                    .unwrap_or_default();

                Ok(
                    json!({"operation": "list_channels", "channels": channels, "count": channels.len()}),
                )
            }
            other => Err(FlowgentraError::ToolError(format!(
                "Unknown Slack operation: '{}'. Use: post_message, list_channels",
                other
            ))),
        }
    }

    fn definition(&self) -> ToolDefinition {
        let mut props = HashMap::new();
        props.insert(
            "operation".to_string(),
            JsonSchema::string().with_description("Slack operation: post_message or list_channels"),
        );
        props.insert(
            "channel".to_string(),
            JsonSchema::string()
                .with_description("Channel ID or name (for post_message, e.g. #general)"),
        );
        props.insert(
            "text".to_string(),
            JsonSchema::string().with_description("Message text (for post_message)"),
        );

        ToolDefinition::new(
            "slack",
            "Post messages and list channels via Slack Web API (requires SLACK_BOT_TOKEN)",
            JsonSchema::object()
                .with_properties(props)
                .with_required(vec!["operation".to_string()]),
            JsonSchema::object(),
        )
        .with_category("communication")
    }
}
