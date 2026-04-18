//! Weather tool via OpenWeatherMap API.

use super::{JsonSchema, Tool, ToolDefinition};
use crate::prelude::*;
use async_trait::async_trait;
use serde_json::{json, Value};
use std::collections::HashMap;

pub struct OpenWeatherMapTool {
    api_key: String,
    client: reqwest::Client,
}

impl OpenWeatherMapTool {
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            client: reqwest::Client::new(),
        }
    }

    pub fn from_env() -> Result<Self> {
        let key = std::env::var("OPENWEATHERMAP_API_KEY").map_err(|_| {
            FlowgentraError::ToolError(
                "OPENWEATHERMAP_API_KEY environment variable not set".to_string(),
            )
        })?;
        Ok(Self::new(key))
    }
}

#[async_trait]
impl Tool for OpenWeatherMapTool {
    async fn call(&self, input: Value) -> Result<Value> {
        let city = input
            .get("city")
            .and_then(|v| v.as_str())
            .ok_or_else(|| FlowgentraError::ToolError("Missing 'city' field".to_string()))?;

        let units = input
            .get("units")
            .and_then(|v| v.as_str())
            .unwrap_or("metric");

        let url = format!(
            "https://api.openweathermap.org/data/2.5/weather?q={}&appid={}&units={}",
            urlencoding::encode(city),
            self.api_key,
            units
        );

        let resp: Value = self
            .client
            .get(&url)
            .send()
            .await
            .map_err(|e| {
                FlowgentraError::ToolError(format!("OpenWeatherMap request failed: {}", e))
            })?
            .json()
            .await
            .map_err(|e| {
                FlowgentraError::ToolError(format!("OpenWeatherMap JSON parse failed: {}", e))
            })?;

        if let Some(cod) = resp.get("cod").and_then(|v| v.as_i64()) {
            if cod != 200 {
                let msg = resp
                    .get("message")
                    .and_then(|v| v.as_str())
                    .unwrap_or("Unknown error");
                return Err(FlowgentraError::ToolError(format!(
                    "OpenWeatherMap error {}: {}",
                    cod, msg
                )));
            }
        }

        let temp = resp
            .pointer("/main/temp")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0);
        let feels_like = resp
            .pointer("/main/feels_like")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0);
        let humidity = resp
            .pointer("/main/humidity")
            .and_then(|v| v.as_i64())
            .unwrap_or(0);
        let description = resp
            .pointer("/weather/0/description")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        let wind_speed = resp
            .pointer("/wind/speed")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0);
        let city_name = resp.get("name").and_then(|v| v.as_str()).unwrap_or(city);
        let country = resp
            .pointer("/sys/country")
            .and_then(|v| v.as_str())
            .unwrap_or("");

        Ok(json!({
            "city": city_name,
            "country": country,
            "temperature": temp,
            "feels_like": feels_like,
            "humidity": humidity,
            "description": description,
            "wind_speed": wind_speed,
            "units": units,
        }))
    }

    fn definition(&self) -> ToolDefinition {
        let mut props = HashMap::new();
        props.insert(
            "city".to_string(),
            JsonSchema::string().with_description("City name, e.g. \"London\" or \"Paris,FR\""),
        );
        props.insert(
            "units".to_string(),
            JsonSchema::string().with_description(
                "Unit system: metric (°C), imperial (°F), or standard (K). Default: metric",
            ),
        );

        ToolDefinition::new(
            "openweathermap",
            "Get current weather for a city via OpenWeatherMap API (requires OPENWEATHERMAP_API_KEY)",
            JsonSchema::object()
                .with_properties(props)
                .with_required(vec!["city".to_string()]),
            JsonSchema::object(),
        )
        .with_category("weather")
        .with_example(
            json!({"city": "London", "units": "metric"}),
            json!({"city": "London", "country": "GB", "temperature": 15.2, "description": "light rain"}),
            "Get London weather in Celsius",
        )
    }
}
