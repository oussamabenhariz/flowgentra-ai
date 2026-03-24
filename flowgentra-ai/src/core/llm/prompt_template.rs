//! Prompt Templates — variable interpolation and composition for LLM prompts
//!
//! Provides structured prompt building similar to LangChain's PromptTemplate,
//! ChatPromptTemplate, and FewShotPromptTemplate.

use std::collections::HashMap;

use super::{Message, MessageRole};

/// A simple prompt template with `{variable}` interpolation.
///
/// # Example
/// ```
/// use flowgentra_ai::core::llm::prompt_template::PromptTemplate;
///
/// let template = PromptTemplate::new("Hello, {name}! You are a {role}.");
/// let result = template.format(&[("name", "Alice"), ("role", "developer")]).unwrap();
/// assert_eq!(result, "Hello, Alice! You are a developer.");
/// ```
#[derive(Debug, Clone)]
pub struct PromptTemplate {
    template: String,
    input_variables: Vec<String>,
}

impl PromptTemplate {
    /// Create a new prompt template. Variables are extracted automatically from `{var}` patterns.
    pub fn new(template: impl Into<String>) -> Self {
        let template = template.into();
        let input_variables = Self::extract_variables(&template);
        Self {
            template,
            input_variables,
        }
    }

    /// Create with explicit variable list (skips auto-detection).
    pub fn with_variables(template: impl Into<String>, variables: Vec<String>) -> Self {
        Self {
            template: template.into(),
            input_variables: variables,
        }
    }

    /// Get the list of input variables this template expects.
    pub fn input_variables(&self) -> &[String] {
        &self.input_variables
    }

    /// Format the template by replacing variables with values.
    pub fn format(&self, variables: &[(&str, &str)]) -> Result<String, PromptError> {
        let map: HashMap<&str, &str> = variables.iter().copied().collect();
        self.format_map(&map)
    }

    /// Format using a HashMap of variable values.
    pub fn format_map(&self, variables: &HashMap<&str, &str>) -> Result<String, PromptError> {
        let mut result = self.template.clone();

        for var in &self.input_variables {
            let placeholder = format!("{{{}}}", var);
            if let Some(value) = variables.get(var.as_str()) {
                result = result.replace(&placeholder, value);
            } else {
                return Err(PromptError::MissingVariable(var.clone()));
            }
        }

        Ok(result)
    }

    /// Partial format — fill some variables, leave others as placeholders.
    pub fn partial(&self, variables: &[(&str, &str)]) -> Self {
        let mut result = self.template.clone();
        let map: HashMap<&str, &str> = variables.iter().copied().collect();
        let mut remaining = Vec::new();

        for var in &self.input_variables {
            let placeholder = format!("{{{}}}", var);
            if let Some(value) = map.get(var.as_str()) {
                result = result.replace(&placeholder, value);
            } else {
                remaining.push(var.clone());
            }
        }

        Self {
            template: result,
            input_variables: remaining,
        }
    }

    fn extract_variables(template: &str) -> Vec<String> {
        let mut vars = Vec::new();
        let mut chars = template.chars().peekable();

        while let Some(ch) = chars.next() {
            if ch == '{' {
                let mut var_name = String::new();
                let mut found_close = false;

                for next_ch in chars.by_ref() {
                    if next_ch == '}' {
                        found_close = true;
                        break;
                    }
                    if next_ch.is_alphanumeric() || next_ch == '_' {
                        var_name.push(next_ch);
                    } else {
                        break;
                    }
                }

                if found_close && !var_name.is_empty() && !vars.contains(&var_name) {
                    vars.push(var_name);
                }
            }
        }

        vars
    }
}

/// Error type for prompt template operations
#[derive(Debug, thiserror::Error)]
pub enum PromptError {
    #[error("Missing variable: {0}")]
    MissingVariable(String),

    #[error("Invalid template: {0}")]
    InvalidTemplate(String),
}

// =============================================================================
// ChatPromptTemplate
// =============================================================================

/// A single message template within a ChatPromptTemplate.
#[derive(Debug, Clone)]
pub struct MessageTemplate {
    pub role: MessageRole,
    pub template: PromptTemplate,
}

/// A template for building multi-message chat prompts.
///
/// # Example
/// ```
/// use flowgentra_ai::core::llm::prompt_template::ChatPromptTemplate;
///
/// let template = ChatPromptTemplate::new()
///     .system("You are a {role} assistant.")
///     .user("{question}");
///
/// let messages = template.format(&[
///     ("role", "helpful"),
///     ("question", "What is Rust?"),
/// ]).unwrap();
///
/// assert_eq!(messages.len(), 2);
/// assert_eq!(messages[0].content, "You are a helpful assistant.");
/// assert_eq!(messages[1].content, "What is Rust?");
/// ```
#[derive(Debug, Clone)]
pub struct ChatPromptTemplate {
    messages: Vec<MessageTemplate>,
}

impl ChatPromptTemplate {
    pub fn new() -> Self {
        Self {
            messages: Vec::new(),
        }
    }

    /// Add a system message template.
    pub fn system(mut self, template: impl Into<String>) -> Self {
        self.messages.push(MessageTemplate {
            role: MessageRole::System,
            template: PromptTemplate::new(template),
        });
        self
    }

    /// Add a user message template.
    pub fn user(mut self, template: impl Into<String>) -> Self {
        self.messages.push(MessageTemplate {
            role: MessageRole::User,
            template: PromptTemplate::new(template),
        });
        self
    }

    /// Add an assistant message template.
    pub fn assistant(mut self, template: impl Into<String>) -> Self {
        self.messages.push(MessageTemplate {
            role: MessageRole::Assistant,
            template: PromptTemplate::new(template),
        });
        self
    }

    /// Add a message with a specific role.
    pub fn message(mut self, role: MessageRole, template: impl Into<String>) -> Self {
        self.messages.push(MessageTemplate {
            role,
            template: PromptTemplate::new(template),
        });
        self
    }

    /// Get all input variables across all message templates.
    pub fn input_variables(&self) -> Vec<String> {
        let mut vars = Vec::new();
        for msg in &self.messages {
            for var in msg.template.input_variables() {
                if !vars.contains(var) {
                    vars.push(var.clone());
                }
            }
        }
        vars
    }

    /// Format all message templates and return a Vec<Message>.
    pub fn format(&self, variables: &[(&str, &str)]) -> Result<Vec<Message>, PromptError> {
        let map: HashMap<&str, &str> = variables.iter().copied().collect();

        self.messages
            .iter()
            .map(|msg_template| {
                let content = msg_template.template.format_map(&map)?;
                Ok(Message {
                    role: msg_template.role.clone(),
                    content,
                    ..Default::default()
                })
            })
            .collect()
    }
}

impl Default for ChatPromptTemplate {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// FewShotPromptTemplate
// =============================================================================

/// A few-shot prompt template that includes examples before the actual query.
///
/// # Example
/// ```
/// use flowgentra_ai::core::llm::prompt_template::FewShotPromptTemplate;
///
/// let template = FewShotPromptTemplate::new(
///     "Translate English to French:",
///     "Input: {input}\nOutput: {output}",
///     "Input: {query}\nOutput:",
/// )
/// .add_example(&[("input", "Hello"), ("output", "Bonjour")])
/// .add_example(&[("input", "Goodbye"), ("output", "Au revoir")]);
///
/// let result = template.format(&[("query", "Thank you")]).unwrap();
/// assert!(result.contains("Bonjour"));
/// assert!(result.contains("Thank you"));
/// ```
#[derive(Debug, Clone)]
pub struct FewShotPromptTemplate {
    prefix: String,
    example_template: PromptTemplate,
    suffix_template: PromptTemplate,
    examples: Vec<Vec<(String, String)>>,
    example_separator: String,
}

impl FewShotPromptTemplate {
    pub fn new(
        prefix: impl Into<String>,
        example_template: impl Into<String>,
        suffix_template: impl Into<String>,
    ) -> Self {
        Self {
            prefix: prefix.into(),
            example_template: PromptTemplate::new(example_template),
            suffix_template: PromptTemplate::new(suffix_template),
            examples: Vec::new(),
            example_separator: "\n\n".to_string(),
        }
    }

    /// Add an example to the few-shot prompt.
    pub fn add_example(mut self, variables: &[(&str, &str)]) -> Self {
        self.examples.push(
            variables
                .iter()
                .map(|(k, v)| (k.to_string(), v.to_string()))
                .collect(),
        );
        self
    }

    /// Set the separator between examples.
    pub fn with_separator(mut self, sep: impl Into<String>) -> Self {
        self.example_separator = sep.into();
        self
    }

    /// Format the complete few-shot prompt.
    pub fn format(&self, variables: &[(&str, &str)]) -> Result<String, PromptError> {
        let mut parts = vec![self.prefix.clone()];

        // Format examples
        for example in &self.examples {
            let example_vars: Vec<(&str, &str)> = example
                .iter()
                .map(|(k, v)| (k.as_str(), v.as_str()))
                .collect();
            parts.push(self.example_template.format(&example_vars)?);
        }

        // Format suffix with query variables
        parts.push(self.suffix_template.format(variables)?);

        Ok(parts.join(&self.example_separator))
    }

    /// Format as a single Message (user role).
    pub fn format_message(&self, variables: &[(&str, &str)]) -> Result<Message, PromptError> {
        let content = self.format(variables)?;
        Ok(Message::user(content))
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prompt_template_basic() {
        let template = PromptTemplate::new("Hello, {name}!");
        assert_eq!(template.input_variables(), &["name"]);

        let result = template.format(&[("name", "World")]).unwrap();
        assert_eq!(result, "Hello, World!");
    }

    #[test]
    fn test_prompt_template_multiple_vars() {
        let template = PromptTemplate::new("{greeting}, {name}! You are {age} years old.");
        assert_eq!(template.input_variables().len(), 3);

        let result = template
            .format(&[("greeting", "Hi"), ("name", "Alice"), ("age", "30")])
            .unwrap();
        assert_eq!(result, "Hi, Alice! You are 30 years old.");
    }

    #[test]
    fn test_prompt_template_missing_variable() {
        let template = PromptTemplate::new("Hello, {name}!");
        let result = template.format(&[]);
        assert!(result.is_err());
    }

    #[test]
    fn test_prompt_template_partial() {
        let template = PromptTemplate::new("Hello, {name}! You are a {role}.");
        let partial = template.partial(&[("name", "Alice")]);
        assert_eq!(partial.input_variables(), &["role"]);

        let result = partial.format(&[("role", "developer")]).unwrap();
        assert_eq!(result, "Hello, Alice! You are a developer.");
    }

    #[test]
    fn test_chat_prompt_template() {
        let template = ChatPromptTemplate::new()
            .system("You are a {role} assistant.")
            .user("{question}");

        let vars = template.input_variables();
        assert!(vars.contains(&"role".to_string()));
        assert!(vars.contains(&"question".to_string()));

        let messages = template
            .format(&[("role", "helpful"), ("question", "What is Rust?")])
            .unwrap();

        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0].role, MessageRole::System);
        assert_eq!(messages[0].content, "You are a helpful assistant.");
        assert_eq!(messages[1].role, MessageRole::User);
        assert_eq!(messages[1].content, "What is Rust?");
    }

    #[test]
    fn test_few_shot_prompt_template() {
        let template = FewShotPromptTemplate::new(
            "Classify the sentiment:",
            "Text: {text}\nSentiment: {sentiment}",
            "Text: {query}\nSentiment:",
        )
        .add_example(&[("text", "I love this!"), ("sentiment", "positive")])
        .add_example(&[("text", "This is terrible."), ("sentiment", "negative")]);

        let result = template.format(&[("query", "It's okay I guess")]).unwrap();
        assert!(result.contains("Classify the sentiment:"));
        assert!(result.contains("I love this!"));
        assert!(result.contains("positive"));
        assert!(result.contains("It's okay I guess"));
    }

    #[test]
    fn test_few_shot_as_message() {
        let template = FewShotPromptTemplate::new("prefix", "Q: {q}\nA: {a}", "Q: {query}\nA:")
            .add_example(&[("q", "1+1?"), ("a", "2")]);

        let msg = template.format_message(&[("query", "2+2?")]).unwrap();
        assert!(msg.is_user());
        assert!(msg.content.contains("2+2?"));
    }

    #[test]
    fn test_no_variables() {
        let template = PromptTemplate::new("No variables here.");
        assert!(template.input_variables().is_empty());

        let result = template.format(&[]).unwrap();
        assert_eq!(result, "No variables here.");
    }
}
