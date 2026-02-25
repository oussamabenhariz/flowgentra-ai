/// Few-Shot ReAct Agent
///
/// Reasoning + action agent with example demonstrations.
/// Provides better guidance through concrete examples of problem-solving.
use super::{Agent, AgentConfig, AgentType, ToolSpec};
use crate::core::error::ErenFlowError;
use crate::core::state::State;
use std::collections::HashMap;

/// Example demonstration for agent
#[derive(Debug, Clone)]
pub struct ExampleDemonstration {
    pub input: String,
    pub thought: String,
    pub action: String,
    pub observation: String,
    pub output: String,
}

impl ExampleDemonstration {
    /// Create new example
    pub fn new(
        input: impl Into<String>,
        thought: impl Into<String>,
        action: impl Into<String>,
        observation: impl Into<String>,
        output: impl Into<String>,
    ) -> Self {
        Self {
            input: input.into(),
            thought: thought.into(),
            action: action.into(),
            observation: observation.into(),
            output: output.into(),
        }
    }

    /// Format as text for prompt
    pub fn format(&self) -> String {
        format!(
            "Input: {}\nThought: {}\nAction: {}\nObservation: {}\nOutput: {}",
            self.input, self.thought, self.action, self.observation, self.output
        )
    }
}

/// Few-shot ReAct agent with example demonstrations
pub struct FewShotReActAgent {
    config: AgentConfig,
    tools: HashMap<String, ToolSpec>,
    examples: Vec<ExampleDemonstration>,
}

impl FewShotReActAgent {
    /// Create new few-shot ReAct agent
    pub fn new(config: AgentConfig) -> Self {
        let tools = config.tools.clone();
        Self {
            config,
            tools,
            examples: Vec::new(),
        }
    }

    /// Add example demonstration
    pub fn add_example(&mut self, example: ExampleDemonstration) {
        self.examples.push(example);
    }

    /// Add multiple examples
    pub fn add_examples(&mut self, examples: Vec<ExampleDemonstration>) {
        self.examples.extend(examples);
    }

    /// Get examples
    pub fn examples(&self) -> &[ExampleDemonstration] {
        &self.examples
    }

    /// Format tools for agent prompt
    fn format_tools(&self) -> String {
        if self.tools.is_empty() {
            return "No tools available".to_string();
        }

        let mut formatted = String::new();
        for (idx, (name, spec)) in self.tools.iter().enumerate() {
            formatted.push_str(&format!("{}. {}: {}\n", idx + 1, name, spec.description));
            if !spec.parameters.is_empty() {
                formatted.push_str("   Parameters:\n");
                for (param_name, param_type) in &spec.parameters {
                    let required = if spec.required.contains(param_name) {
                        "(required)"
                    } else {
                        "(optional)"
                    };
                    formatted.push_str(&format!(
                        "     - {}: {} {}\n",
                        param_name, param_type, required
                    ));
                }
            }
        }
        formatted
    }

    /// Format examples for prompt
    fn format_examples(&self) -> String {
        if self.examples.is_empty() {
            return String::new();
        }

        let mut formatted = String::from("EXAMPLES:\n");
        for (idx, example) in self.examples.iter().enumerate() {
            formatted.push_str(&format!("\nExample {}:\n{}\n", idx + 1, example.format()));
            formatted.push_str("---\n");
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

    /// Generate reasoning chain with examples
    fn generate_reasoning_chain(&self, input: &str) -> String {
        let examples = self.format_examples();
        let tools = self.format_tools();
        let mcps = self.format_mcps();

        format!(
            r#"{}

Available Tools:
{}{}

{}

Input: {}

Follow the example format above to solve this problem:
Thought: (think about what you need to do)
Action: (the action to take)
Observation: (result of the action)
... (repeat Thought/Action/Observation as needed)
Thought: I now know the final answer
Output: (your answer here)"#,
            self.config.system_prompt, tools, mcps, examples, input
        )
    }
}

impl Default for FewShotReActAgent {
    fn default() -> Self {
        Self::new(AgentConfig::new("few_shot_react", AgentType::FewShotReAct))
    }
}

impl Agent for FewShotReActAgent {
    fn name(&self) -> &str {
        &self.config.name
    }

    fn agent_type(&self) -> AgentType {
        AgentType::FewShotReAct
    }

    fn initialize(&mut self, state: &mut State) -> Result<(), ErenFlowError> {
        // Store agent metadata
        state.set(
            "__agent_name",
            serde_json::Value::String(self.config.name.clone()),
        );
        state.set(
            "__agent_type",
            serde_json::Value::String("few-shot-react".to_string()),
        );
        state.set(
            "__agent_tools_count",
            serde_json::Value::String(self.tools.len().to_string()),
        );
        state.set(
            "__agent_examples_count",
            serde_json::Value::String(self.examples.len().to_string()),
        );

        Ok(())
    }

    fn process(&self, input: &str, _state: &State) -> Result<String, ErenFlowError> {
        let reasoning = self.generate_reasoning_chain(input);

        Ok(format!(
            "Agent: {}\nType: {}\nMode: Few-Shot ReAct (Examples: {})\n\nReasoning:\n{}",
            self.config.name,
            self.agent_type(),
            self.examples.len(),
            reasoning
        ))
    }

    fn config(&self) -> &AgentConfig {
        &self.config
    }

    fn add_tool(&mut self, tool_name: &str, tool_spec: ToolSpec) -> Result<(), ErenFlowError> {
        if self.tools.contains_key(tool_name) {
            return Err(ErenFlowError::ConfigError(format!(
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
    fn test_few_shot_react_creation() {
        let agent = FewShotReActAgent::default();
        assert_eq!(agent.name(), "few_shot_react");
        assert_eq!(agent.agent_type(), AgentType::FewShotReAct);
    }

    #[test]
    fn test_add_example() {
        let mut agent = FewShotReActAgent::default();
        let example = ExampleDemonstration::new(
            "What is 2+2?",
            "I need to add two numbers",
            "calculate(2 + 2)",
            "Result is 4",
            "The answer is 4",
        );

        agent.add_example(example);
        assert_eq!(agent.examples().len(), 1);
    }

    #[test]
    fn test_add_multiple_examples() {
        let mut agent = FewShotReActAgent::default();
        let examples = vec![
            ExampleDemonstration::new("2+2?", "Add", "calc", "4", "4"),
            ExampleDemonstration::new("3*3?", "Multiply", "calc", "9", "9"),
        ];

        agent.add_examples(examples);
        assert_eq!(agent.examples().len(), 2);
    }

    #[test]
    fn test_example_formatting() {
        let example = ExampleDemonstration::new(
            "Test input",
            "Test thought",
            "Test action",
            "Test observation",
            "Test output",
        );

        let formatted = example.format();
        assert!(formatted.contains("Input: Test input"));
        assert!(formatted.contains("Thought: Test thought"));
    }

    #[test]
    fn test_process_with_examples() {
        let mut agent = FewShotReActAgent::default();
        agent.add_example(ExampleDemonstration::new(
            "What is 1+1?",
            "Simple addition",
            "add(1,1)",
            "2",
            "The answer is 2",
        ));

        let state = State::new();
        let result = agent.process("What is 5+3?", &state);

        assert!(result.is_ok());
        let output = result.unwrap();
        assert!(output.contains("few-shot-react"));
        assert!(output.contains("Examples: 1"));
    }

    #[test]
    fn test_add_tool() {
        let mut agent = FewShotReActAgent::default();
        let tool = ToolSpec::new("search", "Search online");

        assert!(agent.add_tool("search", tool).is_ok());
        assert_eq!(agent.tools().len(), 1);
    }
}
