/// ReAct Docstore Agent
///
/// ReAct loop specialised for document-store retrieval.  The agent uses exactly
/// two operations:
///
/// - `Search[query]`  — find a document or passage
/// - `Lookup[term]`   — look up a specific term in the last found document
/// - `Finish[answer]` — signal the final answer
///
/// The user must register tools named **"search"** and **"lookup"** (or a single
/// tool executor that dispatches on the tool name).
///
/// Equivalent to LangChain's `react-docstore` agent type.
///
/// ## Output format
/// ```text
/// Thought: I need to search for ...
/// Action: Search[query]
/// Observation: ...
/// Thought: I need to look up ...
/// Action: Lookup[term]
/// Observation: ...
/// Thought: I now know the final answer.
/// Action: Finish[answer]
/// ```
use super::{Agent, AgentType, PrebuiltAgentConfig, ToolSpec};
use crate::core::error::FlowgentraError;
use std::collections::HashMap;

/// ReAct Docstore agent implementation
pub struct ReactDocstoreAgent {
    config: PrebuiltAgentConfig,
    tools: HashMap<String, ToolSpec>,
}

impl ReactDocstoreAgent {
    /// Create new ReactDocstore agent
    pub fn new(config: PrebuiltAgentConfig) -> Self {
        let tools = config.tools.clone();
        Self { config, tools }
    }
}

impl Default for ReactDocstoreAgent {
    fn default() -> Self {
        let mut config = crate::core::agents::builders::new_prebuilt_agent_config("react_docstore", AgentType::ReactDocstore);
        // Pre-register the required Search and Lookup tool stubs
        config.tools.insert(
            "search".to_string(),
            ToolSpec::new("search", "Search the document store for a query")
                .with_parameter("query", "string")
                .required("query"),
        );
        config.tools.insert(
            "lookup".to_string(),
            ToolSpec::new(
                "lookup",
                "Look up a term in the most recently found document",
            )
            .with_parameter("term", "string")
            .required("term"),
        );
        Self::new(config)
    }
}

impl Agent for ReactDocstoreAgent {
    fn name(&self) -> &str {
        &self.config.name
    }

    fn agent_type(&self) -> AgentType {
        AgentType::ReactDocstore
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
            serde_json::Value::String("react-docstore".to_string()),
        );
        // Reset the Thought/Action/Observation scratchpad
        state.set("scratchpad", serde_json::Value::String(String::new()));
        Ok(())
    }

    fn process(
        &self,
        input: &str,
        _state: &crate::core::state::DynState,
    ) -> Result<String, FlowgentraError> {
        Ok(format!(
            "Agent: {}\nType: {}\nMode: ReAct Docstore\n\nQuestion: {}\nThought:",
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
    fn test_react_docstore_creation() {
        let agent = ReactDocstoreAgent::default();
        assert_eq!(agent.name(), "react_docstore");
        assert_eq!(agent.agent_type(), AgentType::ReactDocstore);
    }

    #[test]
    fn test_default_has_search_and_lookup() {
        let agent = ReactDocstoreAgent::default();
        let names: Vec<&str> = agent.tools().iter().map(|t| t.name.as_str()).collect();
        assert!(names.contains(&"search"));
        assert!(names.contains(&"lookup"));
        assert_eq!(agent.tools().len(), 2);
    }

    #[test]
    fn test_initialize_sets_scratchpad() {
        use crate::core::state::DynState;
        let mut agent = ReactDocstoreAgent::default();
        let mut state = DynState::new();
        agent.initialize(&mut state).unwrap();
        assert_eq!(
            state.get("__agent_type").unwrap().as_str().unwrap(),
            "react-docstore"
        );
        assert!(state.get("scratchpad").is_some());
    }

    #[test]
    fn test_process_output_format() {
        use crate::core::state::DynState;
        let agent = ReactDocstoreAgent::default();
        let state = DynState::new();
        let out = agent
            .process("What is the elevation of the High Plains?", &state)
            .unwrap();
        assert!(out.contains("ReAct Docstore"));
        assert!(out.contains("Thought"));
    }
}
