//! Human-in-the-loop tool: blocks until the user types a response on stdin.

use super::{JsonSchema, Tool, ToolDefinition};
use crate::prelude::*;
use async_trait::async_trait;
use serde_json::{json, Value};
use std::collections::HashMap;

pub struct HumanInputTool;

#[async_trait]
impl Tool for HumanInputTool {
    async fn call(&self, input: Value) -> Result<Value> {
        let prompt = input
            .get("prompt")
            .and_then(|v| v.as_str())
            .unwrap_or("Input required: ")
            .to_string();

        // stdin is blocking; offload to a thread so we don't block the async executor.
        let user_input = tokio::task::spawn_blocking(move || {
            use std::io::Write;
            print!("{}", prompt);
            std::io::stdout()
                .flush()
                .map_err(|e| FlowgentraError::ToolError(format!("stdout flush error: {}", e)))?;
            let mut line = String::new();
            std::io::stdin()
                .read_line(&mut line)
                .map_err(|e| FlowgentraError::ToolError(format!("stdin read error: {}", e)))?;
            Ok::<String, FlowgentraError>(
                line.trim_end_matches('\n')
                    .trim_end_matches('\r')
                    .to_string(),
            )
        })
        .await
        .map_err(|e| FlowgentraError::ToolError(format!("spawn_blocking error: {}", e)))??;

        Ok(json!({"input": user_input}))
    }

    fn definition(&self) -> ToolDefinition {
        let mut props = HashMap::new();
        props.insert(
            "prompt".to_string(),
            JsonSchema::string()
                .with_description("Message to display to the user before reading input"),
        );

        ToolDefinition::new(
            "human_input",
            "Pause execution and ask the human operator for input via stdin",
            JsonSchema::object().with_properties(props),
            JsonSchema::object().with_properties({
                let mut out = HashMap::new();
                out.insert("input".to_string(), JsonSchema::string());
                out
            }),
        )
        .with_category("human")
        .with_example(
            json!({"prompt": "What is the capital of France? "}),
            json!({"input": "Paris"}),
            "Ask the user a question",
        )
    }
}
