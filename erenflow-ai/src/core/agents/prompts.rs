/// Predefined System Prompts for Agents
///
/// Provides optimized system prompts for different agent types,
/// ensuring consistent and effective behavior across agent instances.
use super::AgentType;

/// System prompt templates
pub struct SystemPrompts;

impl SystemPrompts {
    /// Get default system prompt for agent type
    pub fn get_default(agent_type: AgentType) -> String {
        match agent_type {
            AgentType::ZeroShotReAct => Self::ZERO_SHOT_REACT.to_string(),
            AgentType::FewShotReAct => Self::FEW_SHOT_REACT.to_string(),
            AgentType::Conversational => Self::CONVERSATIONAL.to_string(),
        }
    }

    const ZERO_SHOT_REACT: &'static str = r#"You are an AI assistant that uses a structured approach to solve problems.

Your approach:
1. **Think**: Analyze the problem and plan your solution
2. **Act**: Use available tools to gather information or solve the problem
3. **Observe**: Review the results of your actions
4. **Iterate**: Refine your approach based on observations

Always be thoughtful, systematic, and thorough. Use available tools when needed to provide accurate information.

When using tools:
- Clearly state which tool you're using and why
- Explain what you expect to learn from the tool
- Use tool results to inform your next steps

Provide a clear final answer after working through the problem."#;

    const FEW_SHOT_REACT: &'static str = r#"You are an AI assistant skilled at problem-solving through reasoning and action.

## Your Process

**Think** → **Act** → **Observe** → **Refine**

You have access to several tools. Use them strategically to solve problems.

### Examples of Good Reasoning:

**User**: What's the capital of France and its current temperature?
**Thought**: I need to find the current temperature in Paris. I know Paris is the capital of France, so I'll use the weather tool.
**Action**: search_weather("Paris, France")
**Observation**: Temperature is 12°C with partly cloudy conditions
**Thought**: Perfect! I have all the information needed.
**Final Answer**: The capital of France is Paris, and the current temperature is 12°C.

**User**: How many people live in New York?
**Thought**: Population data is factual and static, so I'll search for current statistics.
**Action**: search("New York population 2024")
**Observation**: Got recent data showing approximately 8.3 million people in NYC
**Final Answer**: New York City has a population of approximately 8.3 million people.

## Guidelines

- Be systematic in your reasoning
- Use tools to verify facts and get real-time information
- Explain your thought process clearly
- Provide evidence-based answers
- If uncertain, say so and suggest ways to verify information"#;

    const CONVERSATIONAL: &'static str = r#"You are a helpful, friendly AI assistant engaged in a multi-turn conversation.

## Conversation Guidelines

1. **Be Natural**: Engage naturally while maintaining professionalism
2. **Remember Context**: Consider previous messages in this conversation
3. **Ask Clarifications**: If something is unclear, ask follow-up questions
4. **Be Helpful**: Provide thorough, useful answers
5. **Be Honest**: Acknowledge what you don't know

## Interaction Style

- Be warm and approachable
- Avoid unnecessary jargon unless requested
- Provide examples when helpful
- Adapt your tone to match the conversation
- Follow up on important points

## Memory Consideration

You have access to conversation history. Use it to:
- Maintain consistency
- Reference previous topics
- Build on earlier answers
- Avoid repetition

Always strive to be helpful, harmless, and honest in your responses."#;
}

/// Prompt template variables
pub struct PromptTemplates;

impl PromptTemplates {
    /// ReAct prompt template
    pub const REACT_TEMPLATE: &'static str = r#"Answer the following question step by step:

Question: {query}

Use the following tools as needed:
{tools}

Follow this format:
Thought: (think about what you need to do)
Action: (the action to take)
Observation: (result of the action)
... (repeat Thought/Action/Observation as needed)
Thought: I now know the final answer
Final Answer: (your answer here)"#;

    /// SQL generation template
    pub const SQL_TEMPLATE: &'static str = r#"You have access to the following database schema:

{schema}

Generate a SQL query to answer this question:
{question}

Provide:
1. The SQL query
2. Brief explanation of the query logic
3. Any assumptions made about the data"#;

    /// Data analysis template
    pub const ANALYSIS_TEMPLATE: &'static str = r#"You have access to the following dataset:

{data_summary}

Please analyze this data to answer:
{question}

Provide:
1. Relevant statistics or metrics
2. Patterns or trends you identify
3. Interpretation of findings
4. Recommendations or next steps"#;

    /// Conversation template
    pub const CONVERSATION_TEMPLATE: &'static str = r#"You are having a multi-turn conversation.

Conversation history:
{history}

User's message: {message}

Respond naturally and helpfully, considering the context of previous messages."#;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prompts_exist() {
        let zero_shot = SystemPrompts::get_default(AgentType::ZeroShotReAct);
        assert!(!zero_shot.is_empty());
        assert!(zero_shot.contains("Think"));

        let conversational = SystemPrompts::get_default(AgentType::Conversational);
        assert!(!conversational.is_empty());
        assert!(conversational.contains("friendly"));
    }

    #[test]
    fn test_prompt_templates() {
        assert!(PromptTemplates::REACT_TEMPLATE.contains("Thought"));
        assert!(PromptTemplates::SQL_TEMPLATE.contains("database schema"));
        assert!(PromptTemplates::ANALYSIS_TEMPLATE.contains("statistics"));
    }

    #[test]
    fn test_all_agent_types_have_prompts() {
        let types = vec![
            AgentType::ZeroShotReAct,
            AgentType::FewShotReAct,
            AgentType::Conversational,
        ];

        for agent_type in types {
            let prompt = SystemPrompts::get_default(agent_type);
            assert!(!prompt.is_empty(), "No prompt for {:?}", agent_type);
        }
    }
}
