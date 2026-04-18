//! Financial data tool via Alpha Vantage API.

use super::{JsonSchema, Tool, ToolDefinition};
use crate::prelude::*;
use async_trait::async_trait;
use serde_json::{json, Value};
use std::collections::HashMap;

pub struct AlphaVantageTool {
    api_key: String,
    client: reqwest::Client,
}

impl AlphaVantageTool {
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            client: reqwest::Client::new(),
        }
    }

    pub fn from_env() -> Result<Self> {
        let key = std::env::var("ALPHA_VANTAGE_API_KEY").map_err(|_| {
            FlowgentraError::ToolError(
                "ALPHA_VANTAGE_API_KEY environment variable not set".to_string(),
            )
        })?;
        Ok(Self::new(key))
    }
}

#[async_trait]
impl Tool for AlphaVantageTool {
    async fn call(&self, input: Value) -> Result<Value> {
        let function = input
            .get("function")
            .and_then(|v| v.as_str())
            .ok_or_else(|| FlowgentraError::ToolError("Missing 'function' field".to_string()))?;

        let symbol = input.get("symbol").and_then(|v| v.as_str()).unwrap_or("");

        let mut url = format!(
            "https://www.alphavantage.co/query?function={}&apikey={}",
            urlencoding::encode(function),
            self.api_key
        );

        if !symbol.is_empty() {
            url.push_str(&format!("&symbol={}", urlencoding::encode(symbol)));
        }

        // Allow passing additional parameters (e.g. interval, from_currency, to_currency)
        if let Some(extra) = input.get("params").and_then(|v| v.as_object()) {
            for (k, v) in extra {
                if let Some(v_str) = v.as_str() {
                    url.push_str(&format!(
                        "&{}={}",
                        urlencoding::encode(k),
                        urlencoding::encode(v_str)
                    ));
                }
            }
        }

        let resp: Value = self
            .client
            .get(&url)
            .send()
            .await
            .map_err(|e| {
                FlowgentraError::ToolError(format!("Alpha Vantage request failed: {}", e))
            })?
            .json()
            .await
            .map_err(|e| {
                FlowgentraError::ToolError(format!("Alpha Vantage JSON parse failed: {}", e))
            })?;

        // Alpha Vantage returns error in "Note" or "Information" fields
        if let Some(note) = resp.get("Note").and_then(|v| v.as_str()) {
            return Err(FlowgentraError::ToolError(format!(
                "Alpha Vantage: {}",
                note
            )));
        }
        if let Some(info) = resp.get("Information").and_then(|v| v.as_str()) {
            return Err(FlowgentraError::ToolError(format!(
                "Alpha Vantage: {}",
                info
            )));
        }

        Ok(json!({
            "function": function,
            "symbol": symbol,
            "data": resp,
        }))
    }

    fn definition(&self) -> ToolDefinition {
        let mut props = HashMap::new();
        props.insert(
            "function".to_string(),
            JsonSchema::string().with_description(
                "Alpha Vantage function, e.g. GLOBAL_QUOTE, TIME_SERIES_DAILY, CURRENCY_EXCHANGE_RATE",
            ),
        );
        props.insert(
            "symbol".to_string(),
            JsonSchema::string().with_description("Stock ticker symbol, e.g. AAPL, MSFT"),
        );
        props.insert(
            "params".to_string(),
            JsonSchema::object().with_description(
                "Additional query parameters as key-value pairs (e.g. interval, from_currency)",
            ),
        );

        ToolDefinition::new(
            "alpha_vantage",
            "Fetch financial market data via Alpha Vantage API (requires ALPHA_VANTAGE_API_KEY)",
            JsonSchema::object()
                .with_properties(props)
                .with_required(vec!["function".to_string()]),
            JsonSchema::object(),
        )
        .with_category("finance")
        .with_example(
            json!({"function": "GLOBAL_QUOTE", "symbol": "AAPL"}),
            json!({"function": "GLOBAL_QUOTE", "symbol": "AAPL", "data": {"Global Quote": {"01. symbol": "AAPL", "05. price": "185.00"}}}),
            "Get Apple stock quote",
        )
    }
}
