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
use crate::core::llm::{LLMClient, LLMConfig, LLMProvider, Message, ToolCall, ToolDefinition};
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
        return llm_config.create_client().map_err(|e| {
            FlowgentraError::ConfigError(format!("Failed to create LLM client: {}", e))
        });
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
// Shared helpers
// =============================================================================

/// Convert a `ToolSpec` into an LLM `ToolDefinition` with a JSON Schema body.
fn tool_spec_to_definition(spec: &super::ToolSpec) -> ToolDefinition {
    let mut properties = serde_json::Map::new();
    for (name, param_type) in &spec.parameters {
        let json_type = match param_type.to_lowercase().as_str() {
            "integer" | "int" => "integer",
            "boolean" | "bool" => "boolean",
            "number" | "float" | "f32" | "f64" => "number",
            "array" | "list" | "vec" => "array",
            "object" | "map" | "dict" => "object",
            _ => "string",
        };
        properties.insert(name.clone(), serde_json::json!({ "type": json_type }));
    }
    let parameters = serde_json::json!({
        "type": "object",
        "properties": properties,
        "required": spec.required,
    });
    ToolDefinition::new(&spec.name, &spec.description, parameters)
}

// =============================================================================
// ToolCallingNode
// =============================================================================

/// Node that uses the provider's native tool/function-calling API.
///
/// Unlike `AgentReasoningNode` (which parses `<action>` text tags), this node
/// passes structured `ToolDefinition`s to `chat_with_tools()` and reads
/// `response.tool_calls` for the next action.
///
/// State keys written:
/// - `llm_response`       — raw text content of the LLM response
/// - `needs_tool`         — bool, true when a tool call was requested
/// - `pending_tool_name`  — name of the tool to call
/// - `pending_tool_args`  — JSON-string arguments for the tool
/// - `tc_call_id`         — tool call ID (for the tool-result message)
/// - `tc_assistant_content` / `tc_tool_calls_json` — stored for replay on next turn
pub struct ToolCallingNode {
    config: PrebuiltAgentConfig,
}

impl ToolCallingNode {
    pub fn new(config: PrebuiltAgentConfig) -> Self {
        Self { config }
    }

    pub async fn execute(&self, state: &DynState) -> Result<DynState, FlowgentraError> {
        let client = create_client_from_config(&self.config)?;

        let tool_defs: Vec<ToolDefinition> = self
            .config
            .tools
            .values()
            .map(tool_spec_to_definition)
            .collect();

        let input = state
            .get("input")
            .and_then(|v| v.as_str().map(|s| s.to_string()))
            .unwrap_or_default();

        let mut messages = vec![Message::system(&self.config.system_prompt)];
        messages.push(Message::user(&input));

        // Re-attach prior assistant message (with tool_calls) + tool result
        if let Some(result_str) = state
            .get("tool_result")
            .and_then(|v| v.as_str().map(String::from))
        {
            let call_id = state
                .get("tc_call_id")
                .and_then(|v| v.as_str().map(String::from))
                .unwrap_or_default();
            let prev_content = state
                .get("tc_assistant_content")
                .and_then(|v| v.as_str().map(String::from))
                .unwrap_or_default();
            let prev_tool_calls: Option<Vec<ToolCall>> = state
                .get("tc_tool_calls_json")
                .and_then(|v| serde_json::from_value(v.clone()).ok());

            let mut assistant_msg = Message::assistant(&prev_content);
            assistant_msg.tool_calls = prev_tool_calls;
            messages.push(assistant_msg);
            messages.push(Message::tool_result(&call_id, &result_str));
        }

        info!(
            model = %self.config.llm_model,
            input = %input,
            tools_count = %tool_defs.len(),
            "Calling LLM (tool-calling)"
        );

        let response = client
            .chat_with_tools(messages, &tool_defs)
            .await
            .map_err(|e| FlowgentraError::LLMError(format!("LLM tool-calling failed: {}", e)))?;

        let response_text = response.content.clone();
        debug!(response = %response_text, "Tool-calling LLM response");

        state.set("llm_response", serde_json::json!(response_text));

        if response.has_tool_calls() {
            let tc = &response.tool_calls.as_ref().unwrap()[0];
            info!(tool = %tc.name, id = %tc.id, "Native tool call detected");

            state.set("needs_tool", serde_json::json!(true));
            state.set("pending_tool_name", serde_json::json!(tc.name));
            state.set(
                "pending_tool_args",
                serde_json::json!(tc.arguments.to_string()),
            );
            // Persist for replay on the next iteration
            state.set("tc_call_id", serde_json::json!(tc.id));
            state.set("tc_assistant_content", serde_json::json!(response_text));
            state.set("tc_tool_calls_json", serde_json::json!(response.tool_calls));
        } else {
            state.set("needs_tool", serde_json::json!(false));
            state.remove("pending_tool_name");
            state.remove("pending_tool_args");
            state.remove("tc_call_id");
            state.remove("tc_assistant_content");
            state.remove("tc_tool_calls_json");
        }

        Ok(state.clone())
    }
}

/// Router for the ToolCalling agent — identical logic to `reasoning_router`.
pub fn tool_calling_router(state: &DynState) -> Result<String, FlowgentraError> {
    reasoning_router(state)
}

// =============================================================================
// StructuredChatNode
// =============================================================================

/// Parsed structured-chat action from JSON blob.
#[derive(Debug, Clone)]
pub struct StructuredAction {
    pub action: String,
    pub action_input: serde_json::Value,
}

/// Parse the JSON blob from a structured-chat response.
///
/// Looks for the last ```...``` fenced block and deserialises the JSON inside.
/// Falls back to scanning the raw text for `{"action": ...}`.
pub fn parse_structured_action(text: &str) -> Option<StructuredAction> {
    // Try fenced code block first
    let json_str = if let Some(start) = text.find("```") {
        let after = &text[start + 3..];
        // Skip optional language tag ("json\n" or just "\n")
        let body = after
            .trim_start_matches("json")
            .trim_start_matches('\n')
            .trim_start_matches('\r');
        if let Some(end) = body.find("```") {
            body[..end].trim().to_string()
        } else {
            body.trim().to_string()
        }
    } else {
        // Fallback: find first `{` in the text
        let start = text.find('{')?;
        let end = text.rfind('}')? + 1;
        text[start..end].trim().to_string()
    };

    let v: serde_json::Value = serde_json::from_str(&json_str).ok()?;
    let action = v.get("action")?.as_str()?.to_string();
    let action_input = v.get("action_input")?.clone();
    Some(StructuredAction {
        action,
        action_input,
    })
}

/// Node that drives the StructuredChat-ZeroShot-ReAct agent.
///
/// Calls the LLM with the structured-chat prompt, parses the JSON blob from
/// the response, and sets the standard `needs_tool` / `pending_tool_*` keys so
/// the existing `ToolExecutorNode` and `reasoning_router` can be reused.
///
/// The tool `action_input` may be a JSON object or a plain string.  We
/// serialise it to a string and pass it as `pending_tool_args`.
pub struct StructuredChatNode {
    config: PrebuiltAgentConfig,
}

impl StructuredChatNode {
    pub fn new(config: PrebuiltAgentConfig) -> Self {
        Self { config }
    }

    pub async fn execute(&self, state: &DynState) -> Result<DynState, FlowgentraError> {
        let client = create_client_from_config(&self.config)?;

        let input = state
            .get("input")
            .and_then(|v| v.as_str().map(|s| s.to_string()))
            .unwrap_or_default();

        let tools_str = format_tools_for_prompt(&self.config);

        // Inject actual tool list into system prompt placeholder
        let system_prompt = self
            .config
            .system_prompt
            .replace("{tools}", &tools_str)
            .replace(
                "{tool_names}",
                &self
                    .config
                    .tools
                    .keys()
                    .cloned()
                    .collect::<Vec<_>>()
                    .join(", "),
            );

        let mut messages = vec![Message::system(&system_prompt)];

        // Include prior tool result as an Observation in the scratchpad
        if let Some(tool_result) = state.get("tool_result").and_then(|v| v.as_str().map(String::from)) {
            if let Some(prev) = state
                .get("llm_response")
                .and_then(|v| v.as_str().map(String::from))
            {
                messages.push(Message::assistant(&prev));
            }
            messages.push(Message::user(format!("Observation: {}", tool_result)));
        }

        messages.push(Message::user(format!("Question: {}", input)));

        info!(
            model = %self.config.llm_model,
            input = %input,
            "Calling LLM (structured-chat)"
        );

        let response = client
            .chat(messages)
            .await
            .map_err(|e| FlowgentraError::LLMError(format!("LLM call failed: {}", e)))?;

        let response_text = response.content.clone();
        debug!(response = %response_text, "Structured-chat LLM response");

        state.set("llm_response", serde_json::json!(response_text));

        if let Some(parsed) = parse_structured_action(&response_text) {
            if parsed.action == "Final Answer" {
                // Extract the final answer string
                let answer = match &parsed.action_input {
                    serde_json::Value::String(s) => s.clone(),
                    other => other.to_string(),
                };
                state.set("structured_final_answer", serde_json::json!(answer));
                state.set("needs_tool", serde_json::json!(false));
                state.remove("pending_tool_name");
                state.remove("pending_tool_args");
            } else {
                let args_str = match &parsed.action_input {
                    serde_json::Value::String(s) => s.clone(),
                    other => other.to_string(),
                };
                info!(tool = %parsed.action, args = %args_str, "Structured-chat tool call");
                state.set("needs_tool", serde_json::json!(true));
                state.set("pending_tool_name", serde_json::json!(parsed.action));
                state.set("pending_tool_args", serde_json::json!(args_str));
            }
        } else {
            // No parseable JSON → treat full response as final answer
            state.set("needs_tool", serde_json::json!(false));
            state.remove("pending_tool_name");
            state.remove("pending_tool_args");
        }

        Ok(state.clone())
    }
}

// =============================================================================
// SelfAskNode
// =============================================================================

/// Parsed output from the SelfAsk agent.
#[derive(Debug, Clone)]
pub enum SelfAskOutput {
    FollowUp(String),
    FinalAnswer(String),
}

/// Parse the LLM response from a self-ask-with-search agent.
///
/// Looks for:
/// - `Follow up: <question>` → another search is needed
/// - `So the final answer is: <answer>` → done
pub fn parse_self_ask_response(text: &str) -> Option<SelfAskOutput> {
    for line in text.lines() {
        let trimmed = line.trim();
        if let Some(rest) = trimmed
            .strip_prefix("Follow up:")
            .or_else(|| trimmed.strip_prefix("Follow Up:"))
        {
            return Some(SelfAskOutput::FollowUp(rest.trim().to_string()));
        }
        if let Some(rest) = trimmed
            .strip_prefix("So the final answer is:")
            .or_else(|| trimmed.strip_prefix("So the final answer is "))
        {
            return Some(SelfAskOutput::FinalAnswer(rest.trim().to_string()));
        }
    }
    None
}

/// Node that drives the SelfAskWithSearch agent.
///
/// On each iteration:
/// 1. Builds the prompt from the few-shot examples + current question + scratchpad
/// 2. Calls the LLM
/// 3. Parses the response:
///    - "Follow up: <q>"  → route to search tool, append to scratchpad
///    - "So the final answer is: <a>" → set final answer, route to END
///
/// State keys:
/// - `scratchpad`         — growing string of Follow-up/Intermediate-answer pairs
/// - `sa_follow_up`       — current sub-question sent to the search tool
/// - `needs_tool`         — bool
/// - `pending_tool_name`  — always "search"
/// - `pending_tool_args`  — the follow-up question
pub struct SelfAskNode {
    config: PrebuiltAgentConfig,
}

impl SelfAskNode {
    pub fn new(config: PrebuiltAgentConfig) -> Self {
        Self { config }
    }

    pub async fn execute(&self, state: &DynState) -> Result<DynState, FlowgentraError> {
        let client = create_client_from_config(&self.config)?;

        let input = state
            .get("input")
            .and_then(|v| v.as_str().map(|s| s.to_string()))
            .unwrap_or_default();

        // Append the latest intermediate answer to scratchpad (if any)
        let mut scratchpad = state
            .get("scratchpad")
            .and_then(|v| v.as_str().map(String::from))
            .unwrap_or_default();

        if let Some(tool_result) = state
            .get("tool_result")
            .and_then(|v| v.as_str().map(String::from))
        {
            let follow_up = state
                .get("sa_follow_up")
                .and_then(|v| v.as_str().map(String::from))
                .unwrap_or_default();
            scratchpad.push_str(&format!(
                "Follow up: {}\nIntermediate answer: {}\n",
                follow_up, tool_result
            ));
            state.set("scratchpad", serde_json::json!(scratchpad));
        }

        // Build prompt: few-shot examples + question + scratchpad so far
        let prompt = format!(
            "{}\n\nQuestion: {}\nAre follow up questions needed here: {}",
            self.config.system_prompt, input, scratchpad
        );

        let messages = vec![Message::user(&prompt)];

        info!(
            model = %self.config.llm_model,
            input = %input,
            "Calling LLM (self-ask-with-search)"
        );

        let response = client
            .chat(messages)
            .await
            .map_err(|e| FlowgentraError::LLMError(format!("LLM call failed: {}", e)))?;

        let response_text = response.content.clone();
        debug!(response = %response_text, "Self-ask LLM response");
        state.set("llm_response", serde_json::json!(response_text));

        match parse_self_ask_response(&response_text) {
            Some(SelfAskOutput::FollowUp(question)) => {
                info!(question = %question, "Self-ask follow-up");
                state.set("sa_follow_up", serde_json::json!(question));
                state.set("needs_tool", serde_json::json!(true));
                state.set("pending_tool_name", serde_json::json!("search"));
                state.set("pending_tool_args", serde_json::json!(question));
            }
            Some(SelfAskOutput::FinalAnswer(answer)) => {
                info!(answer = %answer, "Self-ask final answer");
                state.set("sa_final_answer", serde_json::json!(answer));
                state.set("needs_tool", serde_json::json!(false));
                state.remove("pending_tool_name");
                state.remove("pending_tool_args");
            }
            None => {
                // Treat the full response as the final answer
                state.set("sa_final_answer", serde_json::json!(response_text));
                state.set("needs_tool", serde_json::json!(false));
            }
        }

        Ok(state.clone())
    }
}

/// Router for SelfAskWithSearch — same logic as reasoning_router.
pub fn self_ask_router(state: &DynState) -> Result<String, FlowgentraError> {
    reasoning_router(state)
}

// =============================================================================
// DocstoreNode
// =============================================================================

/// Possible actions from the ReAct Docstore agent.
#[derive(Debug, Clone)]
pub enum DocstoreAction {
    Search(String),
    Lookup(String),
    Finish(String),
}

/// Parse `Action: Search[query]`, `Action: Lookup[term]`, or `Action: Finish[answer]`.
pub fn parse_docstore_action(text: &str) -> Option<DocstoreAction> {
    for line in text.lines() {
        let trimmed = line.trim();
        if let Some(rest) = trimmed.strip_prefix("Action:") {
            let rest = rest.trim();
            if let Some(inner) = rest
                .strip_prefix("Search[")
                .and_then(|s| s.strip_suffix(']'))
            {
                return Some(DocstoreAction::Search(inner.to_string()));
            }
            if let Some(inner) = rest
                .strip_prefix("Lookup[")
                .and_then(|s| s.strip_suffix(']'))
            {
                return Some(DocstoreAction::Lookup(inner.to_string()));
            }
            if let Some(inner) = rest
                .strip_prefix("Finish[")
                .and_then(|s| s.strip_suffix(']'))
            {
                return Some(DocstoreAction::Finish(inner.to_string()));
            }
        }
    }
    None
}

/// Node that drives the ReactDocstore agent.
///
/// Builds a growing Thought/Action/Observation scratchpad and routes to:
/// - "search" tool for `Search[...]`
/// - "lookup" tool for `Lookup[...]`
/// - END for `Finish[...]`
///
/// State keys:
/// - `scratchpad`           — accumulated Thought/Action/Observation text
/// - `ds_action_type`       — "search" | "lookup" | "finish"
/// - `needs_tool`           — bool
/// - `pending_tool_name`    — "search" or "lookup"
/// - `pending_tool_args`    — query / term
/// - `ds_final_answer`      — set when Finish[...] is parsed
pub struct DocstoreNode {
    config: PrebuiltAgentConfig,
}

impl DocstoreNode {
    pub fn new(config: PrebuiltAgentConfig) -> Self {
        Self { config }
    }

    pub async fn execute(&self, state: &DynState) -> Result<DynState, FlowgentraError> {
        let client = create_client_from_config(&self.config)?;

        let input = state
            .get("input")
            .and_then(|v| v.as_str().map(|s| s.to_string()))
            .unwrap_or_default();

        // Append previous observation to scratchpad
        let mut scratchpad = state
            .get("scratchpad")
            .and_then(|v| v.as_str().map(String::from))
            .unwrap_or_default();

        if let Some(observation) = state
            .get("tool_result")
            .and_then(|v| v.as_str().map(String::from))
        {
            scratchpad.push_str(&format!("Observation: {}\n", observation));
            state.set("scratchpad", serde_json::json!(scratchpad));
        }

        let prompt = format!(
            "{}\n\nQuestion: {}\n{}",
            self.config.system_prompt, input, scratchpad
        );

        let messages = vec![Message::user(&prompt)];

        info!(
            model = %self.config.llm_model,
            input = %input,
            "Calling LLM (react-docstore)"
        );

        let response = client
            .chat(messages)
            .await
            .map_err(|e| FlowgentraError::LLMError(format!("LLM call failed: {}", e)))?;

        let response_text = response.content.clone();
        debug!(response = %response_text, "Docstore LLM response");
        state.set("llm_response", serde_json::json!(response_text));

        // Append Thought + Action lines to scratchpad
        scratchpad.push_str(&response_text);
        scratchpad.push('\n');
        state.set("scratchpad", serde_json::json!(scratchpad));

        match parse_docstore_action(&response_text) {
            Some(DocstoreAction::Search(query)) => {
                info!(query = %query, "Docstore Search");
                state.set("ds_action_type", serde_json::json!("search"));
                state.set("needs_tool", serde_json::json!(true));
                state.set("pending_tool_name", serde_json::json!("search"));
                state.set("pending_tool_args", serde_json::json!(query));
            }
            Some(DocstoreAction::Lookup(term)) => {
                info!(term = %term, "Docstore Lookup");
                state.set("ds_action_type", serde_json::json!("lookup"));
                state.set("needs_tool", serde_json::json!(true));
                state.set("pending_tool_name", serde_json::json!("lookup"));
                state.set("pending_tool_args", serde_json::json!(term));
            }
            Some(DocstoreAction::Finish(answer)) => {
                info!(answer = %answer, "Docstore finish");
                state.set("ds_final_answer", serde_json::json!(answer));
                state.set("ds_action_type", serde_json::json!("finish"));
                state.set("needs_tool", serde_json::json!(false));
                state.remove("pending_tool_name");
                state.remove("pending_tool_args");
            }
            None => {
                state.set("needs_tool", serde_json::json!(false));
            }
        }

        Ok(state.clone())
    }
}

/// Router for the ReactDocstore agent.
///
/// Routes to the appropriate tool node ("search", "lookup") or END.
pub fn docstore_router(state: &DynState) -> Result<String, FlowgentraError> {
    let needs_tool = state
        .get("needs_tool")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);

    if needs_tool {
        let tool_name = state
            .get("pending_tool_name")
            .and_then(|v| v.as_str().map(String::from))
            .unwrap_or_else(|| "search".to_string());

        // Both "search" and "lookup" route to the shared tool_executor node.
        // The executor dispatches by tool name to the user-provided function.
        debug!(
            "Docstore router: directing to tool_executor ({})",
            tool_name
        );
        Ok("tool_executor".to_string())
    } else {
        debug!("Docstore router: directing to END");
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

    // ----- StructuredChat parser -----------------------------------------

    #[test]
    fn test_parse_structured_action_fenced() {
        let text = "Thought: I need to search.\nAction:\n```json\n{\n  \"action\": \"web_search\",\n  \"action_input\": \"Rust programming language\"\n}\n```";
        let parsed = parse_structured_action(text).expect("should parse");
        assert_eq!(parsed.action, "web_search");
        assert_eq!(
            parsed.action_input,
            serde_json::json!("Rust programming language")
        );
    }

    #[test]
    fn test_parse_structured_action_final_answer() {
        let text = "Thought: I know the answer.\nAction:\n```\n{\"action\": \"Final Answer\", \"action_input\": \"42\"}\n```";
        let parsed = parse_structured_action(text).expect("should parse");
        assert_eq!(parsed.action, "Final Answer");
    }

    #[test]
    fn test_parse_structured_action_object_input() {
        let text = "{\"action\": \"calculator\", \"action_input\": {\"expression\": \"2 + 2\"}}";
        let parsed = parse_structured_action(text).expect("should parse");
        assert_eq!(parsed.action, "calculator");
        assert_eq!(
            parsed.action_input,
            serde_json::json!({"expression": "2 + 2"})
        );
    }

    // ----- SelfAsk parser -------------------------------------------------

    #[test]
    fn test_parse_self_ask_follow_up() {
        let text = "Yes.\nFollow up: Who directed Jaws?";
        match parse_self_ask_response(text).expect("should parse") {
            SelfAskOutput::FollowUp(q) => assert_eq!(q, "Who directed Jaws?"),
            _ => panic!("Expected FollowUp"),
        }
    }

    #[test]
    fn test_parse_self_ask_final_answer() {
        let text = "So the final answer is: Steven Spielberg";
        match parse_self_ask_response(text).expect("should parse") {
            SelfAskOutput::FinalAnswer(a) => assert_eq!(a, "Steven Spielberg"),
            _ => panic!("Expected FinalAnswer"),
        }
    }

    #[test]
    fn test_parse_self_ask_none() {
        let text = "I am still thinking...";
        assert!(parse_self_ask_response(text).is_none());
    }

    // ----- Docstore parser ------------------------------------------------

    #[test]
    fn test_parse_docstore_search() {
        let text = "Thought: I need to search.\nAction: Search[Colorado orogeny]";
        match parse_docstore_action(text).expect("should parse") {
            DocstoreAction::Search(q) => assert_eq!(q, "Colorado orogeny"),
            _ => panic!("Expected Search"),
        }
    }

    #[test]
    fn test_parse_docstore_lookup() {
        let text = "Thought: Need detail.\nAction: Lookup[eastern sector]";
        match parse_docstore_action(text).expect("should parse") {
            DocstoreAction::Lookup(t) => assert_eq!(t, "eastern sector"),
            _ => panic!("Expected Lookup"),
        }
    }

    #[test]
    fn test_parse_docstore_finish() {
        let text = "Thought: Done.\nAction: Finish[1,800 to 7,000 ft]";
        match parse_docstore_action(text).expect("should parse") {
            DocstoreAction::Finish(a) => assert_eq!(a, "1,800 to 7,000 ft"),
            _ => panic!("Expected Finish"),
        }
    }

    #[test]
    fn test_parse_docstore_none() {
        let text = "Thought: I need to think more.";
        assert!(parse_docstore_action(text).is_none());
    }
}
