//! # Planner Node
//!
//! LLM-driven dynamic routing. The planner uses the LLM to decide the next node
//! given the current state and reachable nodes (constrained by the graph).
//!
//! ## Flow
//!
//! 1. Runtime injects `_current_node` and `_reachable_nodes` into state before calling the planner.
//! 2. Planner reads state, calls LLM with a prompt and reachable node list.
//! 3. LLM returns the next node name (or END).
//! 4. Planner sets `_next_node` in state and returns.
//! 5. Runtime uses `_next_node` instead of following edges.
//!
//! ## Re-planning
//!
//! After each non-planner node, execution returns to the planner (via graph edges).
//! The planner runs again with updated state, enabling dynamic multi-step planning.

use crate::core::error::ErenFlowError;
use crate::core::llm::{LLMClient, Message};
use crate::core::state::State;
use serde_json::json;
use std::sync::Arc;

/// Default system prompt for the planner
const DEFAULT_PLANNER_PROMPT: &str = r#"You are a planning agent. Given the current state and a list of available next nodes, you must choose exactly one node to execute next.

Rules:
- Reply with ONLY the node name, nothing else.
- Use "END" to finish execution.
- You MUST choose one of the available nodes listed below.
- Do not add quotes, punctuation, or explanation."#;

/// Create a NodeFunction that implements the planner node.
///
/// The planner expects `_current_node` and `_reachable_nodes` (JSON array of strings) in state.
/// It sets `_next_node` with the chosen node name after calling the LLM.
pub fn create_planner_handler(
    llm_client: Arc<dyn LLMClient>,
    prompt_template: Option<String>,
) -> super::NodeFunction {
    Box::new(move |mut state: State| {
        let client = Arc::clone(&llm_client);
        let template = prompt_template.clone();
        Box::pin(async move {
            let current = state
                .get_str("_current_node")
                .unwrap_or("unknown")
                .to_string();
            let reachable: Vec<String> = state
                .get("_reachable_nodes")
                .and_then(|v| serde_json::from_value(v.clone()).ok())
                .unwrap_or_default();

            if reachable.is_empty() {
                state.set("_next_node", json!("END"));
                return Ok(state);
            }

            let state_summary = truncate_for_prompt(
                &state.to_json_string().unwrap_or_else(|_| "{}".to_string()),
                1000,
            );
            let nodes_list = reachable.join(", ");

            let user_prompt = format!(
                "Current node: {}\n\nState summary:\n{}\n\nAvailable next nodes: {}\n\nChoose the next node (reply with only the node name):",
                current, state_summary, nodes_list
            );

            let system_prompt = template.as_deref().unwrap_or(DEFAULT_PLANNER_PROMPT);

            let messages = vec![Message::system(system_prompt), Message::user(&user_prompt)];

            let response = client
                .chat(messages)
                .await
                .map_err(|e| ErenFlowError::LLMError(format!("Planner LLM call failed: {}", e)))?;

            let chosen = parse_next_node(&response.content, &reachable);
            state.set("_next_node", json!(chosen));

            Ok(state)
        })
    })
}

/// Truncate a string for use in the prompt
fn truncate_for_prompt(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else {
        format!("{}... [truncated]", &s[..max_len.saturating_sub(20)])
    }
}

/// Parse the LLM response to extract a valid next node name
fn parse_next_node(response: &str, valid: &[String]) -> String {
    let trimmed = response
        .trim()
        .trim_matches(|c: char| c == '"' || c == '\'' || c.is_whitespace())
        .to_string();

    let upper = trimmed.to_uppercase();
    if upper == "END" {
        return "END".to_string();
    }

    for node in valid {
        if node.eq_ignore_ascii_case(&trimmed) {
            return node.clone();
        }
    }

    if valid.contains(&trimmed) {
        return trimmed;
    }

    if let Some(first_line) = trimmed.lines().next() {
        let first = first_line.trim().trim_matches('"').trim();
        for node in valid {
            if node.eq_ignore_ascii_case(first) {
                return node.clone();
            }
        }
        if first.eq_ignore_ascii_case("END") {
            return "END".to_string();
        }
    }

    valid.first().cloned().unwrap_or_else(|| "END".to_string())
}
