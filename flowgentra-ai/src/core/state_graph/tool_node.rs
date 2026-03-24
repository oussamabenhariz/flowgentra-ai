//! ToolNode and tools_condition — prebuilt helpers for ReAct agent patterns
//!
//! - **ToolNode** — a graph node that automatically executes tool calls from LLM responses
//! - **tools_condition** — a router function that routes to tools or end based on tool calls

use std::sync::Arc;

use serde_json::Value;

use crate::core::llm::Message;
use crate::core::state::PlainState;
use crate::core::state_graph::error::Result;
use crate::core::state_graph::node::{FunctionNode, Node};

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

/// Creates a graph node that executes tool calls found in the state.
///
/// Looks for tool calls in `state["tool_calls"]` (a JSON array of `{id, name, arguments}`),
/// executes each one using the provided executor, and writes results to
/// `state["tool_results"]`.
///
/// # Example
/// ```ignore
/// use flowgentra_ai::core::state_graph::tool_node::create_tool_node;
/// use std::sync::Arc;
///
/// let tool_node = create_tool_node(Arc::new(|name, args| {
///     Box::pin(async move {
///         match name.as_str() {
///             "calculator" => Ok(format!("Result: {}", args)),
///             _ => Err(format!("Unknown tool: {}", name)),
///         }
///     })
/// }));
///
/// builder.add_node("tools", tool_node);
/// ```
pub fn create_tool_node(executor: ToolExecutorFn) -> Arc<dyn Node<PlainState>> {
    Arc::new(FunctionNode::new(
        "tool_executor",
        move |state: &PlainState| {
            let executor = executor.clone();
            let state = state.clone();

            Box::pin(async move {
                let mut new_state = state.clone();

                // Read tool calls from state
                let tool_calls = state
                    .get("tool_calls")
                    .and_then(|v| v.as_array())
                    .cloned()
                    .unwrap_or_default();

                let mut results = Vec::new();
                let mut messages = Vec::new();

                for tc_value in &tool_calls {
                    let id = tc_value
                        .get("id")
                        .and_then(|v| v.as_str())
                        .unwrap_or("unknown")
                        .to_string();
                    let name = tc_value
                        .get("name")
                        .and_then(|v| v.as_str())
                        .unwrap_or("unknown")
                        .to_string();
                    let arguments = tc_value
                        .get("arguments")
                        .cloned()
                        .unwrap_or(Value::Object(serde_json::Map::new()));

                    match executor(name.clone(), arguments).await {
                        Ok(result) => {
                            results.push(serde_json::json!({
                                "tool_call_id": id,
                                "name": name,
                                "result": result,
                                "error": null
                            }));
                            messages.push(Message::tool_result(&id, &result));
                        }
                        Err(err) => {
                            results.push(serde_json::json!({
                                "tool_call_id": id,
                                "name": name,
                                "result": null,
                                "error": err
                            }));
                            messages.push(Message::tool_result(&id, format!("Error: {}", err)));
                        }
                    }
                }

                // Write results to state
                new_state.set("tool_results", serde_json::json!(results));

                // Append tool result messages to the messages array
                if let Some(existing) = new_state
                    .get("messages")
                    .and_then(|v| v.as_array().cloned())
                {
                    let mut all_messages = existing;
                    for msg in &messages {
                        all_messages.push(serde_json::json!({
                            "role": "tool",
                            "content": msg.content,
                            "tool_call_id": msg.tool_call_id
                        }));
                    }
                    new_state.set("messages", serde_json::json!(all_messages));
                }

                // Clear tool_calls since they've been executed
                new_state.remove("tool_calls");

                Ok(new_state)
            })
        },
    ))
}

/// A router function for use with `add_conditional_edge`.
///
/// Routes to `tool_node_name` if the state contains tool calls,
/// or to `"__end__"` if there are none.
///
/// # Example
/// ```ignore
/// builder.add_conditional_edge("agent", tools_condition("tools"));
/// ```
#[allow(clippy::type_complexity)]
pub fn tools_condition(
    tool_node_name: &str,
) -> Box<dyn Fn(&PlainState) -> Result<String> + Send + Sync> {
    let tool_node = tool_node_name.to_string();
    Box::new(move |state: &PlainState| {
        let has_tool_calls = state
            .get("tool_calls")
            .and_then(|v| v.as_array())
            .map(|arr| !arr.is_empty())
            .unwrap_or(false);

        if has_tool_calls {
            Ok(tool_node.clone())
        } else {
            Ok("__end__".to_string())
        }
    })
}

/// Helper to extract tool calls from an LLM response Message and store them in state.
///
/// Call this in your agent node after getting an LLM response:
/// ```ignore
/// let response = llm.chat_with_tools(messages, &tools).await?;
/// state = store_tool_calls(state, &response);
/// ```
pub fn store_tool_calls(mut state: PlainState, message: &Message) -> PlainState {
    if let Some(tool_calls) = &message.tool_calls {
        let tc_values: Vec<Value> = tool_calls
            .iter()
            .map(|tc| {
                serde_json::json!({
                    "id": tc.id,
                    "name": tc.name,
                    "arguments": tc.arguments,
                })
            })
            .collect();
        state.set("tool_calls", serde_json::json!(tc_values));
    }

    // Store the assistant message content
    state.set("last_response", serde_json::json!(message.content));

    state
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::llm::ToolCall;

    #[test]
    fn test_tools_condition_no_calls() {
        let state = PlainState::new();
        let router = tools_condition("tools");
        let result = router(&state).unwrap();
        assert_eq!(result, "__end__");
    }

    #[test]
    fn test_tools_condition_with_calls() {
        let mut state = PlainState::new();
        state.set(
            "tool_calls",
            serde_json::json!([{"id": "1", "name": "calc", "arguments": {}}]),
        );

        let router = tools_condition("tools");
        let result = router(&state).unwrap();
        assert_eq!(result, "tools");
    }

    #[test]
    fn test_tools_condition_empty_array() {
        let mut state = PlainState::new();
        state.set("tool_calls", serde_json::json!([]));

        let router = tools_condition("tools");
        let result = router(&state).unwrap();
        assert_eq!(result, "__end__");
    }

    #[test]
    fn test_store_tool_calls() {
        let state = PlainState::new();
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

        let state = store_tool_calls(state, &msg);

        let calls = state.get("tool_calls").unwrap().as_array().unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0]["name"], "calculator");
    }

    #[tokio::test]
    async fn test_tool_node_execution() {
        use crate::core::state_graph::node::Node;

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

        let mut state = PlainState::new();
        state.set(
            "tool_calls",
            serde_json::json!([{
                "id": "call_1",
                "name": "add",
                "arguments": {"a": 2, "b": 3}
            }]),
        );

        let result = node.execute(&state).await.unwrap();

        // tool_calls should be cleared
        assert!(result.get("tool_calls").is_none());

        // Results should be stored
        let results = result.get("tool_results").unwrap().as_array().unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0]["result"], "5");
    }
}
