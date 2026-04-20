/// Self Ask With Search Agent
///
/// Decomposes complex questions into a chain of simpler sub-questions, answers
/// each with a **search** tool, and accumulates an `Intermediate answer` for
/// each.  Terminates when the LLM emits `"So the final answer is: ..."`.
///
/// The agent **requires** a tool named `"search"` (case-insensitive) to be
/// registered — all other tools are ignored.
///
/// Equivalent to LangChain's `self-ask-with-search` agent type.
///
/// ## Output format
/// ```text
/// Question: <user question>
/// Are follow up questions needed here: Yes.
/// Follow up: <sub-question>
/// Intermediate answer: <search result>
/// Follow up: <sub-question-2>
/// Intermediate answer: <search result 2>
/// ...
/// So the final answer is: <final answer>
/// ```
use super::{Agent, AgentType, PrebuiltAgentConfig, ToolSpec};
use crate::core::error::FlowgentraError;
use std::collections::HashMap;

/// Self Ask With Search agent implementation
pub struct SelfAskWithSearchAgent {
    config: PrebuiltAgentConfig,
    tools: HashMap<String, ToolSpec>,
}

impl SelfAskWithSearchAgent {
    /// Create new Self Ask With Search agent
    pub fn new(config: PrebuiltAgentConfig) -> Self {
        let tools = config.tools.clone();
        Self { config, tools }
    }

    /// Return the registered search tool, if any.
    pub fn search_tool(&self) -> Option<&ToolSpec> {
        self.tools
            .values()
            .find(|t| t.name.to_lowercase() == "search")
    }
}

impl Default for SelfAskWithSearchAgent {
    fn default() -> Self {
        let mut config = crate::core::agents::builders::new_prebuilt_agent_config(
            "self_ask_with_search",
            AgentType::SelfAskWithSearch,
        );
        // Pre-register the required search tool stub so users see it in `.tools()`
        config.tools.insert(
            "search".to_string(),
            ToolSpec::new(
                "search",
                "Search for information to answer follow-up questions",
            )
            .with_parameter("query", "string")
            .required("query"),
        );
        Self::new(config)
    }
}

impl Agent for SelfAskWithSearchAgent {
    fn name(&self) -> &str {
        &self.config.name
    }

    fn agent_type(&self) -> AgentType {
        AgentType::SelfAskWithSearch
    }

    fn initialize(
        &mut self,
        state: &mut crate::core::state::DynState,
    ) -> Result<(), FlowgentraError> {
        state.set(
            "__agent_name",
            serde_json::Value::String(self.config.name.clone()),
        );
        state.set(
            "__agent_type",
            serde_json::Value::String("self-ask-with-search".to_string()),
        );
        state.set("scratchpad", serde_json::Value::String(String::new()));
        Ok(())
    }

    fn process(
        &self,
        input: &str,
        _state: &crate::core::state::DynState,
    ) -> Result<String, FlowgentraError> {
        Ok(format!(
            "Agent: {}\nType: {}\nMode: Self Ask With Search\n\nQuestion: {}\nAre follow up questions needed here:",
            self.config.name,
            self.agent_type(),
            input
        ))
    }

    fn config(&self) -> &PrebuiltAgentConfig {
        &self.config
    }

    fn add_tool(&mut self, tool_name: &str, tool_spec: ToolSpec) -> Result<(), FlowgentraError> {
        if self.tools.contains_key(tool_name) {
            return Err(FlowgentraError::ConfigError(format!(
                "Tool '{}' already exists",
                tool_name
            )));
        }
        self.tools.insert(tool_name.to_string(), tool_spec);
        self.config
            .tools
            .insert(tool_name.to_string(), self.tools[tool_name].clone());
        Ok(())
    }

    fn tools(&self) -> Vec<&ToolSpec> {
        self.tools.values().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_self_ask_creation() {
        let agent = SelfAskWithSearchAgent::default();
        assert_eq!(agent.name(), "self_ask_with_search");
        assert_eq!(agent.agent_type(), AgentType::SelfAskWithSearch);
    }

    #[test]
    fn test_default_has_search_tool() {
        let agent = SelfAskWithSearchAgent::default();
        assert!(agent.search_tool().is_some());
        assert_eq!(agent.tools().len(), 1);
    }

    #[test]
    fn test_initialize_sets_scratchpad() {
        use crate::core::state::DynState;
        let mut agent = SelfAskWithSearchAgent::default();
        let mut state = DynState::new();
        agent.initialize(&mut state).unwrap();
        assert_eq!(
            state.get("__agent_type").unwrap().as_str().unwrap(),
            "self-ask-with-search"
        );
        assert!(state.get("scratchpad").is_some());
    }

    #[test]
    fn test_process_output_format() {
        use crate::core::state::DynState;
        let agent = SelfAskWithSearchAgent::default();
        let state = DynState::new();
        let out = agent
            .process("Who is the director of Jaws?", &state)
            .unwrap();
        assert!(out.contains("Self Ask With Search"));
        assert!(out.contains("follow up questions"));
    }
}
