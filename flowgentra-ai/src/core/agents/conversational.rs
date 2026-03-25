/// Conversational Agent
///
/// Multi-turn dialogue agent with memory support.
/// Maintains conversation history and context across multiple interactions.
use super::{Agent, AgentType, PrebuiltAgentConfig, ToolSpec};
use crate::core::error::FlowgentraError;
use std::collections::HashMap;
use std::collections::VecDeque;

/// Conversation message
#[derive(Debug, Clone)]
pub struct Message {
    pub role: String, // "user" or "assistant"
    pub content: String,
    pub timestamp: u64,
}

impl Message {
    /// Create new message
    pub fn new(role: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: role.into(),
            content: content.into(),
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
        }
    }

    /// Format as text
    pub fn format(&self) -> String {
        format!("{}: {}", self.role, self.content)
    }
}

/// Conversational agent with memory
pub struct ConversationalAgent {
    config: PrebuiltAgentConfig,
    tools: HashMap<String, ToolSpec>,
    history: VecDeque<Message>,
}

impl ConversationalAgent {
    /// Create new conversational agent
    pub fn new(mut config: PrebuiltAgentConfig) -> Self {
        let tools = config.tools.clone();

        // Enable memory by default for conversational agents
        if !config.memory_enabled {
            config.memory_enabled = true;
            if config.memory_steps == 0 {
                config.memory_steps = 10;
            }
        }

        Self {
            config,
            tools,
            history: VecDeque::new(),
        }
    }

    /// Add message to history
    pub fn add_message(&mut self, message: Message) {
        self.history.push_back(message);

        // Trim history to configured max steps
        while self.history.len() > self.config.memory_steps {
            self.history.pop_front();
        }
    }

    /// Add user message
    pub fn add_user_message(&mut self, content: impl Into<String>) {
        self.add_message(Message::new("user", content));
    }

    /// Add assistant message
    pub fn add_assistant_message(&mut self, content: impl Into<String>) {
        self.add_message(Message::new("assistant", content));
    }

    /// Get formatted history for context
    fn get_history_context(&self) -> String {
        if self.history.is_empty() {
            return "(No previous messages)".to_string();
        }

        let mut context = String::new();
        for msg in &self.history {
            context.push_str(&format!("{}\n", msg.format()));
        }
        context
    }

    /// Clear history
    pub fn clear_history(&mut self) {
        self.history.clear();
    }

    /// Get conversation length
    pub fn history_len(&self) -> usize {
        self.history.len()
    }

    /// Get messages
    pub fn messages(&self) -> Vec<&Message> {
        self.history.iter().collect()
    }

    /// Format tools for prompt
    fn format_tools(&self) -> String {
        if self.tools.is_empty() {
            return "No tools available".to_string();
        }

        let mut formatted = String::new();
        for (idx, (name, spec)) in self.tools.iter().enumerate() {
            formatted.push_str(&format!("{}. {}: {}\n", idx + 1, name, spec.description));
        }
        formatted
    }

    /// Format MCP configurations for agent prompt
    fn format_mcps(&self) -> String {
        if self.config.mcps.is_empty() {
            return String::new();
        }

        let mut formatted = String::from("\nAvailable MCP Services:\n");
        for (idx, mcp) in self.config.mcps.iter().enumerate() {
            formatted.push_str(&format!(
                "{}. {} ({})\n   Connection: {}\n",
                idx + 1,
                mcp.name,
                mcp.connection_type.as_str(),
                mcp.uri
            ));
        }
        formatted
    }

    /// Generate contextual response prompt
    fn generate_response_prompt(&self, input: &str) -> String {
        let history = self.get_history_context();
        let tools = self.format_tools();
        let mcps = self.format_mcps();

        format!(
            "{}\n\nPrevious Conversation:\n{}\nUser: {}\n\nRespond naturally and helpfully, considering the conversation context.\n\nAvailable tools:\n{}{}",
            self.config.system_prompt,
            history,
            input,
            tools,
            mcps
        )
    }
}

impl Default for ConversationalAgent {
    fn default() -> Self {
        Self::new(PrebuiltAgentConfig::new(
            "conversational",
            AgentType::Conversational,
        ))
    }
}

impl Agent for ConversationalAgent {
    fn name(&self) -> &str {
        &self.config.name
    }

    fn agent_type(&self) -> AgentType {
        AgentType::Conversational
    }

    fn initialize(
        &mut self,
        state: &mut crate::core::state::SharedState,
    ) -> Result<(), FlowgentraError> {
        state.set(
            "__agent_name",
            serde_json::Value::String(self.config.name.clone()),
        );
        state.set(
            "__agent_type",
            serde_json::Value::String("conversational".to_string()),
        );
        state.set(
            "__agent_memory_steps",
            serde_json::Value::String(self.config.memory_steps.to_string()),
        );
        Ok(())
    }

    fn process(
        &self,
        input: &str,
        _state: &crate::core::state::SharedState,
    ) -> Result<String, FlowgentraError> {
        let prompt = self.generate_response_prompt(input);

        Ok(format!(
            "Agent: {}\nType: Conversational\nHistory: {} messages\n\nPrompt:\n{}",
            self.config.name,
            self.history.len(),
            prompt
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
    fn test_conversational_creation() {
        let agent = ConversationalAgent::default();
        assert_eq!(agent.name(), "conversational");
        assert_eq!(agent.agent_type(), AgentType::Conversational);
        assert!(agent.config().memory_enabled);
    }

    #[test]
    fn test_add_message() {
        let mut agent = ConversationalAgent::default();
        agent.add_user_message("Hello");
        agent.add_assistant_message("Hi there!");

        assert_eq!(agent.history_len(), 2);
        assert_eq!(agent.messages().len(), 2);
    }

    #[test]
    fn test_memory_trimming() {
        let mut config = PrebuiltAgentConfig::new("test", AgentType::Conversational);
        config.memory_steps = 3;
        let mut agent = ConversationalAgent::new(config);

        for i in 0..5 {
            agent.add_user_message(format!("Message {}", i));
        }

        assert_eq!(agent.history_len(), 3);
    }

    #[test]
    fn test_clear_history() {
        let mut agent = ConversationalAgent::default();
        agent.add_user_message("Test");
        assert!(agent.history_len() > 0);

        agent.clear_history();
        assert_eq!(agent.history_len(), 0);
    }

    #[test]
    fn test_message_formatting() {
        let msg = Message::new("user", "Hello world");
        let formatted = msg.format();
        assert_eq!(formatted, "user: Hello world");
    }

    #[test]
    fn test_process() {
        use crate::core::state::SharedState;
        let mut agent = ConversationalAgent::default();
        agent.add_user_message("What is AI?");

        let state = SharedState::new(Default::default());
        let result = agent.process("Tell me more", &state);

        assert!(result.is_ok());
        let output = result.unwrap();
        assert!(output.contains("Conversational"));
    }
}
