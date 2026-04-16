//! # Agent Graph Nodes
//!
//! Implements the concrete graph nodes used by agent types:
//!
//! - [`AgentReasoningNode`] — Calls the LLM with system prompt + tools + user input
//! - [`ToolExecutorNode`] — Parses `<action>` tags from LLM response and executes tools
//! - [`ConversationalNode`] — Simple LLM call for conversational agents
//! - [`reasoning_router`] — Routes to tool executor or END based on LLM response

use super::builders::PrebuiltAgentConfig;
use crate::core::error::FlowgentraError;
use crate::core::llm::{LLMClient, LLMConfig, LLMProvider, Message};
use crate::core::state::DynState;
use std::sync::Arc;
use tracing::{debug, info};

/// Type alias for tool executor callbacks.
///
/// The function receives `(tool_name, arguments)` and returns the tool result as a string.
pub type ToolExecutorFn = Arc<dyn Fn(&str, &str) -> String + Send + Sync>;

// =============================================================================
// Helper: Resolve LLM provider from model name
// =============================================================================

/// Resolve a model name string to an LLM provider.
///
/// Uses naming conventions:
/// - `mistral-*` → Mistral
/// - `gpt-*` → OpenAI
/// - `claude-*` → Anthropic
/// - `llama-*`, `mixtral-*` → Groq
/// - `gemma-*` → Ollama (local)
/// - Otherwise → OpenAI (default)
fn resolve_provider(model: &str) -> LLMProvider {
    let lower = model.to_lowercase();
    if lower.starts_with("mistral")
        || lower.starts_with("pixtral")
        || lower.starts_with("codestral")
    {
        LLMProvider::Mistral
    } else if lower.starts_with("gpt") || lower.starts_with("o1") || lower.starts_with("o3") {
        LLMProvider::OpenAI
    } else if lower.starts_with("claude") {
        LLMProvider::Anthropic
    } else if lower.starts_with("llama")
        || lower.starts_with("mixtral")
        || lower.starts_with("gemma")
    {
        LLMProvider::Groq
    } else {
        LLMProvider::OpenAI
    }
}

/// Returns true for providers that require an API key.
fn provider_requires_api_key(provider: &LLMProvider) -> bool {
    !matches!(provider, LLMProvider::Ollama)
}

/// Create an LLM client from a PrebuiltAgentConfig.
fn create_client_from_config(
    config: &PrebuiltAgentConfig,
) -> Result<Arc<dyn LLMClient>, FlowgentraError> {
    // Use the full LLMConfig when provided via Agent.create(llm=...) / AgentBuilder::with_llm()
    if let Some(llm_config) = &config.llm {
        return llm_config
            .create_client()
            .map_err(|e| FlowgentraError::ConfigError(format!("Failed to create LLM client: {}", e)));
    }

    // Fallback: derive provider from model name string
    let provider = resolve_provider(&config.llm_model);

    let api_key = match &config.api_key {
        Some(key) => key.clone(),
        None if provider_requires_api_key(&provider) => {
            return Err(FlowgentraError::ConfigError(format!(
                "api_key is required for provider '{}'. Pass it via PrebuiltAgentConfig::api_key.",
                config.llm_model
            )));
        }
        None => String::new(),
    };

    let mut llm_config = LLMConfig::new(provider, config.llm_model.clone(), api_key);
    llm_config = llm_config.with_temperature(config.temperature);
    llm_config = llm_config.with_max_tokens(config.max_tokens);

    llm_config
        .create_client()
        .map_err(|e| FlowgentraError::ConfigError(format!("Failed to create LLM client: {}", e)))
}

// =============================================================================
// Action tag parsing
// =============================================================================

/// Parsed action from LLM response.
#[derive(Debug, Clone)]
pub struct ParsedAction {
    pub tool_name: String,
    pub arguments: String,
}

/// Parse `<action>tool_name(arguments)</action>` from LLM response text.
pub fn parse_action_tags(text: &str) -> Option<ParsedAction> {
    let start_tag = "<action>";
    let end_tag = "</action>";

    let start = text.find(start_tag)?;
    let end = text.find(end_tag)?;
    let start_idx = start + start_tag.len();

    if start_idx >= end {
        return None;
    }

    let content = text[start_idx..end].trim();

    // Parse "tool_name(arguments)" format
    if let Some(paren_start) = content.find('(') {
        let tool_name = content[..paren_start].trim().to_string();
        let args = if content.ends_with(')') {
            content[paren_start + 1..content.len() - 1].to_string()
        } else {
            content[paren_start + 1..].to_string()
        };
        Some(ParsedAction {
            tool_name,
            arguments: args,
        })
    } else {
        // No parentheses — treat entire content as tool name with empty args
        Some(ParsedAction {
            tool_name: content.to_string(),
            arguments: String::new(),
        })
    }
}

/// Parse `<answer>...</answer>` from LLM response text.
pub fn parse_answer_tags(text: &str) -> Option<String> {
    let start_tag = "<answer>";
    let end_tag = "</answer>";

    let start = text.find(start_tag)?;
    let end = text.find(end_tag)?;
    let start_idx = start + start_tag.len();

    if start_idx >= end {
        return None;
    }

    Some(text[start_idx..end].trim().to_string())
}

// =============================================================================
// Format tools for the prompt
// =============================================================================

fn format_tools_for_prompt(config: &PrebuiltAgentConfig) -> String {
    if config.tools.is_empty() {
        return "No tools available.".to_string();
    }

    let mut formatted = String::new();
    for (idx, (name, spec)) in config.tools.iter().enumerate() {
        formatted.push_str(&format!("{}. {}: {}\n", idx + 1, name, spec.description));
        if !spec.parameters.is_empty() {
            for (param_name, param_type) in &spec.parameters {
                let req = if spec.required.contains(param_name) {
                    " (required)"
                } else {
                    " (optional)"
                };
                formatted.push_str(&format!("   - {}: {}{}\n", param_name, param_type, req));
            }
        }
    }
    formatted
}

// =============================================================================
// AgentReasoningNode
// =============================================================================

/// Node that calls the LLM with the system prompt, available tools, and user input.
///
/// Sets the following state keys:
/// - `llm_response` — raw LLM response text
/// - `needs_tool` — bool, whether the response contains an `<action>` tag
/// - `pending_tool_name` — tool name (if action detected)
/// - `pending_tool_args` — tool arguments (if action detected)
pub struct AgentReasoningNode {
    config: PrebuiltAgentConfig,
}

impl AgentReasoningNode {
    pub fn new(config: PrebuiltAgentConfig) -> Self {
        Self { config }
    }

    pub async fn execute(&self, state: &DynState) -> Result<DynState, FlowgentraError> {
        let client = create_client_from_config(&self.config)?;

        // Build the user prompt with tools and input
        let input = state
            .get("input")
            .and_then(|v| v.as_str().map(|s| s.to_string()))
            .unwrap_or_default();

        let tools_str = format_tools_for_prompt(&self.config);

        // Build message list
        let mut messages = Vec::new();

        // System prompt
        messages.push(Message::system(&self.config.system_prompt));

        // If there's a previous tool result, include it in context.
        // NOTE: We use Message::user() not Message::tool() because we use text-based
        // <action> tags, not native OpenAI/Mistral function calling. The "tool" role
        // requires a tool_call_id that we don't generate.
        if let Some(tool_result) = state.get("tool_result") {
            if let Some(result_str) = tool_result.as_str() {
                // Re-send the previous LLM response for context
                if let Some(prev_response) = state.get("llm_response") {
                    if let Some(prev_str) = prev_response.as_str() {
                        messages.push(Message::assistant(prev_str));
                    }
                }
                messages.push(Message::user(format!(
                    "Tool result: {}\n\nUse this result to provide your final answer using <answer>...</answer> tags.",
                    result_str
                )));
            }
        }

        // User message: combine input with tool listing
        let user_message = format!("Available Tools:\n{}\n\nUser Query: {}", tools_str, input);
        messages.push(Message::user(&user_message));

        info!(
            model = %self.config.llm_model,
            input = %input,
            tools_count = %self.config.tools.len(),
            "Calling LLM"
        );

        // Call the LLM
        let response = client
            .chat(messages)
            .await
            .map_err(|e| FlowgentraError::LLMError(format!("LLM call failed: {}", e)))?;

        let response_text = response.content.clone();
        debug!(response = %response_text, "LLM response received");

        // Store raw response
        state.set("llm_response", serde_json::json!(response_text));

        // Check for action tags (tool calls)
        if let Some(action) = parse_action_tags(&response_text) {
            info!(tool = %action.tool_name, args = %action.arguments, "Tool call detected");
            state.set("needs_tool", serde_json::json!(true));
            state.set("pending_tool_name", serde_json::json!(action.tool_name));
            state.set("pending_tool_args", serde_json::json!(action.arguments));
        } else {
            state.set("needs_tool", serde_json::json!(false));
            // Clean up any stale tool state
            state.remove("pending_tool_name");
            state.remove("pending_tool_args");
        }

        Ok(state.clone())
    }
}

// =============================================================================
// ToolExecutorNode
// =============================================================================

/// Node that executes a pending tool call and stores the result.
///
/// Reads `pending_tool_name` and `pending_tool_args` from state,
/// calls the tool executor function, and stores the result in `tool_result`.
pub struct ToolExecutorNode {
    config: PrebuiltAgentConfig,
    tool_executor: Option<ToolExecutorFn>,
}

impl ToolExecutorNode {
    pub fn new(config: PrebuiltAgentConfig) -> Self {
        Self {
            config,
            tool_executor: None,
        }
    }

    pub fn with_executor(mut self, executor: ToolExecutorFn) -> Self {
        self.tool_executor = Some(executor);
        self
    }

    pub async fn execute(&self, state: &DynState) -> Result<DynState, FlowgentraError> {
        let tool_name = state
            .get("pending_tool_name")
            .and_then(|v| v.as_str().map(|s| s.to_string()))
            .ok_or_else(|| {
                FlowgentraError::ExecutionError("No pending tool name in state".to_string())
            })?;

        let tool_args = state
            .get("pending_tool_args")
            .and_then(|v| v.as_str().map(|s| s.to_string()))
            .unwrap_or_default();

        info!(tool = %tool_name, args = %tool_args, "Executing tool");

        // Execute the tool
        let result = if let Some(executor) = &self.tool_executor {
            executor(&tool_name, &tool_args)
        } else {
            // Check if the tool exists in config
            if self.config.tools.contains_key(&tool_name) {
                format!(
                    "Tool '{}' is registered but no executor function was provided. \
                     Use .with_tool_executor() on the AgentBuilder.",
                    tool_name
                )
            } else {
                format!("Unknown tool: '{}'", tool_name)
            }
        };

        info!(tool = %tool_name, result = %result, "Tool execution complete");

        // Store result and clear pending state
        state.set("tool_result", serde_json::json!(result));
        state.set("needs_tool", serde_json::json!(false));
        state.remove("pending_tool_name");
        state.remove("pending_tool_args");

        Ok(state.clone())
    }
}

// =============================================================================
// ConversationalNode
// =============================================================================

/// Simple conversational node: system prompt + user input → LLM response.
pub struct ConversationalNode {
    config: PrebuiltAgentConfig,
}

impl ConversationalNode {
    pub fn new(config: PrebuiltAgentConfig) -> Self {
        Self { config }
    }

    pub async fn execute(&self, state: &DynState) -> Result<DynState, FlowgentraError> {
        let client = create_client_from_config(&self.config)?;

        let input = state
            .get("input")
            .and_then(|v| v.as_str().map(|s| s.to_string()))
            .unwrap_or_default();

        let mut messages = vec![Message::system(&self.config.system_prompt)];

        // Include conversation history for multi-turn context
        if let Some(history) = state.get("conversation_history") {
            if let Some(history_arr) = history.as_array() {
                for entry in history_arr {
                    let role = entry.get("role").and_then(|r| r.as_str()).unwrap_or("user");
                    let content = entry.get("content").and_then(|c| c.as_str()).unwrap_or("");
                    match role {
                        "assistant" => messages.push(Message::assistant(content)),
                        _ => messages.push(Message::user(content)),
                    }
                }
            }
        }

        // Current user input
        messages.push(Message::user(&input));

        info!(
            model = %self.config.llm_model,
            history_messages = %(messages.len() - 2), // exclude system + current user
            "Calling LLM (conversational)"
        );

        let response = client
            .chat(messages)
            .await
            .map_err(|e| FlowgentraError::LLMError(format!("LLM call failed: {}", e)))?;

        state.set("response", serde_json::json!(response.content));

        Ok(state.clone())
    }
}

// =============================================================================
// Reasoning Router
// =============================================================================

/// Router function for ReAct agents.
///
/// Returns `"tool_executor"` if the LLM response contains a tool call,
/// or `"END"` if the agent has reached a final answer.
pub fn reasoning_router(state: &DynState) -> Result<String, FlowgentraError> {
    let needs_tool = state
        .get("needs_tool")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);

    if needs_tool {
        debug!("Router: directing to tool_executor");
        Ok("tool_executor".to_string())
    } else {
        debug!("Router: directing to END");
        Ok("END".to_string())
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_action_tags_with_args() {
        let text = "I need to calculate this.\n<action>calculator(2 + 2)</action>\nLet me check.";
        let action = parse_action_tags(text).expect("Should parse action");
        assert_eq!(action.tool_name, "calculator");
        assert_eq!(action.arguments, "2 + 2");
    }

    #[test]
    fn test_parse_action_tags_complex_args() {
        let text = "<action>create_file(hello.txt:Hello World!)</action>";
        let action = parse_action_tags(text).expect("Should parse action");
        assert_eq!(action.tool_name, "create_file");
        assert_eq!(action.arguments, "hello.txt:Hello World!");
    }

    #[test]
    fn test_parse_action_tags_no_args() {
        let text = "<action>list_tools</action>";
        let action = parse_action_tags(text).expect("Should parse action");
        assert_eq!(action.tool_name, "list_tools");
        assert_eq!(action.arguments, "");
    }

    #[test]
    fn test_parse_action_tags_none() {
        let text = "I don't need any tools for this. The answer is 42.";
        assert!(parse_action_tags(text).is_none());
    }

    #[test]
    fn test_parse_answer_tags() {
        let text = "After analysis:\n<answer>The result is 2816</answer>";
        let answer = parse_answer_tags(text).expect("Should parse answer");
        assert_eq!(answer, "The result is 2816");
    }

    #[test]
    fn test_parse_answer_tags_none() {
        let text = "Still thinking about this...";
        assert!(parse_answer_tags(text).is_none());
    }

    #[test]
    fn test_reasoning_router_needs_tool() {
        let state = DynState::default();
        state.set("needs_tool", serde_json::json!(true));
        let result = reasoning_router(&state).unwrap();
        assert_eq!(result, "tool_executor");
    }

    #[test]
    fn test_reasoning_router_end() {
        let state = DynState::default();
        state.set("needs_tool", serde_json::json!(false));
        let result = reasoning_router(&state).unwrap();
        assert_eq!(result, "END");
    }

    #[test]
    fn test_reasoning_router_default() {
        let state = DynState::default();
        // No "needs_tool" key — should default to END
        let result = reasoning_router(&state).unwrap();
        assert_eq!(result, "END");
    }

    #[test]
    fn test_resolve_provider() {
        assert!(matches!(
            resolve_provider("mistral-medium"),
            LLMProvider::Mistral
        ));
        assert!(matches!(resolve_provider("gpt-4"), LLMProvider::OpenAI));
        assert!(matches!(
            resolve_provider("claude-3-opus"),
            LLMProvider::Anthropic
        ));
        assert!(matches!(resolve_provider("llama-3-70b"), LLMProvider::Groq));
    }
}
