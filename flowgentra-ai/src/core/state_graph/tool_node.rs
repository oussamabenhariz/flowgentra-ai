//! ToolNode and tools_condition — prebuilt helpers for ReAct agent patterns
//!
//! Works with any state that includes the required tool fields.
//! For convenience, provides `ToolState` — a pre-built typed state.

use std::sync::Arc;

use serde_json::Value;

use crate::core::llm::Message;
use crate::core::reducer::{Append, Overwrite, Reducer};
use crate::core::state::{Context, State};
use crate::core::state_graph::error::Result;
use crate::core::state_graph::node::{FunctionNode, Node};

/// Pre-built state for tool-calling workflows (ReAct pattern).
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ToolState {
    /// Chat messages — accumulated via Append.
    #[serde(default)]
    pub messages: Vec<Message>,

    /// Pending tool calls from the LLM.
    #[serde(default)]
    pub tool_calls: Vec<ToolCallInfo>,

    /// Results from executed tool calls.
    #[serde(default)]
    pub tool_results: Vec<ToolResult>,

    /// Last LLM response content.
    #[serde(default)]
    pub last_response: String,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ToolCallInfo {
    pub id: String,
    pub name: String,
    pub arguments: Value,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ToolResult {
    pub tool_call_id: String,
    pub name: String,
    pub result: Option<String>,
    pub error: Option<String>,
}

/// Partial update for ToolState.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct ToolStateUpdate {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub messages: Option<Vec<Message>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCallInfo>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_results: Option<Vec<ToolResult>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_response: Option<String>,
}

impl ToolStateUpdate {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn messages(mut self, v: Vec<Message>) -> Self {
        self.messages = Some(v);
        self
    }

    pub fn tool_calls(mut self, v: Vec<ToolCallInfo>) -> Self {
        self.tool_calls = Some(v);
        self
    }

    pub fn tool_results(mut self, v: Vec<ToolResult>) -> Self {
        self.tool_results = Some(v);
        self
    }

    pub fn last_response(mut self, v: String) -> Self {
        self.last_response = Some(v);
        self
    }
}

impl State for ToolState {
    type Update = ToolStateUpdate;

    fn apply_update(&mut self, update: Self::Update) {
        if let Some(messages) = update.messages {
            <Append as Reducer<Vec<Message>>>::merge(&mut self.messages, messages);
        }
        if let Some(tool_calls) = update.tool_calls {
            <Overwrite as Reducer<Vec<ToolCallInfo>>>::merge(&mut self.tool_calls, tool_calls);
        }
        if let Some(tool_results) = update.tool_results {
            <Overwrite as Reducer<Vec<ToolResult>>>::merge(&mut self.tool_results, tool_results);
        }
        if let Some(last_response) = update.last_response {
            <Overwrite as Reducer<String>>::merge(&mut self.last_response, last_response);
        }
    }
}

impl ToolState {
    pub fn new() -> Self {
        Self {
            messages: Vec::new(),
            tool_calls: Vec::new(),
            tool_results: Vec::new(),
            last_response: String::new(),
        }
    }

    /// Check if there are pending tool calls.
    pub fn has_tool_calls(&self) -> bool {
        !self.tool_calls.is_empty()
    }
}

impl Default for ToolState {
    fn default() -> Self {
        Self::new()
    }
}

/// A tool executor function: takes (tool_name, arguments) and returns the result.
pub type ToolExecutorFn = Arc<
    dyn Fn(
            String,
            Value,
        ) -> std::pin::Pin<
            Box<dyn std::future::Future<Output = std::result::Result<String, String>> + Send>,
        > + Send
        + Sync,
>;

/// Creates a graph node that executes tool calls found in `state.tool_calls`.
pub fn create_tool_node(executor: ToolExecutorFn) -> Arc<dyn Node<ToolState>> {
    Arc::new(FunctionNode::new(
        "tool_executor",
        move |state: &ToolState, _ctx: &Context| {
            let executor = executor.clone();
            let tool_calls = state.tool_calls.clone();

            Box::pin(async move {
                let mut results = Vec::new();
                let mut messages = Vec::new();

                for tc in &tool_calls {
                    match executor(tc.name.clone(), tc.arguments.clone()).await {
                        Ok(result) => {
                            results.push(ToolResult {
                                tool_call_id: tc.id.clone(),
                                name: tc.name.clone(),
                                result: Some(result.clone()),
                                error: None,
                            });
                            messages.push(Message::tool_result(&tc.id, &result));
                        }
                        Err(err) => {
                            results.push(ToolResult {
                                tool_call_id: tc.id.clone(),
                                name: tc.name.clone(),
                                result: None,
                                error: Some(err.clone()),
                            });
                            messages.push(Message::tool_result(&tc.id, format!("Error: {}", err)));
                        }
                    }
                }

                Ok(ToolStateUpdate::new()
                    .tool_results(results)
                    .tool_calls(Vec::new()) // clear pending calls
                    .messages(messages))
            })
        },
    ))
}

/// A router function for use with `add_conditional_edge`.
///
/// Routes to `tool_node_name` if the state contains tool calls,
/// or to `"__end__"` if there are none.
#[allow(clippy::type_complexity)]
pub fn tools_condition(
    tool_node_name: &str,
) -> Box<dyn Fn(&ToolState) -> Result<String> + Send + Sync> {
    let tool_node = tool_node_name.to_string();
    Box::new(move |state: &ToolState| {
        if state.has_tool_calls() {
            Ok(tool_node.clone())
        } else {
            Ok("__end__".to_string())
        }
    })
}

/// Helper to extract tool calls from an LLM response and produce a state update.
pub fn store_tool_calls(message: &Message) -> ToolStateUpdate {
    let tool_calls = message
        .tool_calls
        .as_ref()
        .map(|tcs| {
            tcs.iter()
                .map(|tc| ToolCallInfo {
                    id: tc.id.clone(),
                    name: tc.name.clone(),
                    arguments: tc.arguments.clone(),
                })
                .collect()
        })
        .unwrap_or_default();

    ToolStateUpdate::new()
        .tool_calls(tool_calls)
        .last_response(message.content.clone())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::llm::ToolCall;

    #[test]
    fn test_tools_condition_no_calls() {
        let state = ToolState::new();
        let router = tools_condition("tools");
        let result = router(&state).unwrap();
        assert_eq!(result, "__end__");
    }

    #[test]
    fn test_tools_condition_with_calls() {
        let mut state = ToolState::new();
        state.tool_calls = vec![ToolCallInfo {
            id: "1".into(),
            name: "calc".into(),
            arguments: serde_json::json!({}),
        }];

        let router = tools_condition("tools");
        let result = router(&state).unwrap();
        assert_eq!(result, "tools");
    }

    #[test]
    fn test_store_tool_calls() {
        let msg = Message {
            role: crate::core::llm::MessageRole::Assistant,
            content: "Let me calculate that.".to_string(),
            tool_calls: Some(vec![ToolCall {
                id: "call_1".to_string(),
                name: "calculator".to_string(),
                arguments: serde_json::json!({"expression": "2+2"}),
            }]),
            tool_call_id: None,
        };

        let update = store_tool_calls(&msg);
        assert!(update.tool_calls.is_some());
        let calls = update.tool_calls.unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "calculator");
    }

    #[tokio::test]
    async fn test_tool_node_execution() {
        let executor: ToolExecutorFn = Arc::new(|name, args| {
            Box::pin(async move {
                match name.as_str() {
                    "add" => {
                        let a = args.get("a").and_then(|v| v.as_i64()).unwrap_or(0);
                        let b = args.get("b").and_then(|v| v.as_i64()).unwrap_or(0);
                        Ok(format!("{}", a + b))
                    }
                    _ => Err(format!("Unknown tool: {}", name)),
                }
            })
        });

        let node = create_tool_node(executor);

        let mut state = ToolState::new();
        state.tool_calls = vec![ToolCallInfo {
            id: "call_1".into(),
            name: "add".into(),
            arguments: serde_json::json!({"a": 2, "b": 3}),
        }];

        let ctx = Context::new();
        let update = node.execute(&state, &ctx).await.unwrap();
        state.apply_update(update);

        assert!(state.tool_calls.is_empty());
        assert_eq!(state.tool_results.len(), 1);
        assert_eq!(state.tool_results[0].result.as_deref(), Some("5"));
    }
}
