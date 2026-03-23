// # Memory-Aware Agent
//
// Provides a high-level wrapper around Agent that handles conversation memory automatically.
// Inspired by LangChain's approach: memory is completely transparent to the user.
//
// # Example
//
// ```ignore
// use flowgentra_ai::agent::MemoryAwareAgent;
//
// #[tokio::main]
// async fn main() -> Result<()> {
//     // Create agent with automatic memory
//     let mut agent = MemoryAwareAgent::from_config("config.yaml")?;
//
//     // Turn 1 - Memory automatically created
//     let answer1 = agent.run_turn("What is Rust?").await?;
//     println!("Agent: {}", answer1);
//
//     // Turn 2 - Memory automatically included, no manual state management!
//     let answer2 = agent.run_turn("What are its features?").await?;
//     println!("Agent: {}", answer2);
//
//     // Turn 3 - Full conversation memory maintained
//     let answer3 = agent.run_turn("How does it compare to Python?").await?;
//     println!("Agent: {}", answer3);
//
//     Ok(())
// }
// ```

use crate::core::agent::Agent;
use crate::core::error::{FlowgentraError, Result};
use crate::core::llm::{Message, MessageRole};
use crate::core::memory::ConversationMemory;
use serde_json::json;
use std::sync::Arc;

/// Memory-aware agent wrapper that handles conversation memory automatically.
///
/// This is the simplest way to use FlowgentraAI for multi-turn conversations.
/// Memory is completely automatic - you just call `run_turn()` and it handles everything:
/// - Injecting previous messages into prompts
/// - Tracking new exchanges
/// - Persisting state between turns
/// - Trimming to buffer window
///
/// Inspired by LangChain's automatic memory management.
pub struct MemoryAwareAgent {
    agent: Agent,
    thread_id: String,
    conversation_memory: Option<Arc<dyn ConversationMemory>>,
}

impl MemoryAwareAgent {
    /// Create a new memory-aware agent from config file
    ///
    /// Automatically:
    /// - Loads config from YAML
    /// - Creates ConversationMemory if enabled
    /// - Initializes state
    ///
    /// # Arguments
    ///
    /// * `config_path` - Path to config.yaml file
    ///
    /// # Example
    ///
    /// ```ignore
    /// let mut agent = MemoryAwareAgent::from_config("config.yaml")?;
    /// ```
    pub fn from_config(config_path: &str) -> Result<Self> {
        let agent = crate::core::agent::from_config_path(config_path)?;
        let conversation_memory = agent.conversation_memory();

        Ok(Self {
            agent,
            thread_id: "default".to_string(),
            conversation_memory,
        })
    }

    /// Set the thread ID for conversation memory (for multi-tenant support)
    ///
    /// # Arguments
    ///
    /// * `thread_id` - Unique conversation thread identifier
    ///
    /// # Example
    ///
    /// ```ignore
    /// let mut agent = MemoryAwareAgent::from_config("config.yaml")?;
    /// agent.set_thread_id("user_123");  // Different conversation per user
    /// ```
    pub fn set_thread_id(&mut self, thread_id: impl Into<String>) -> &mut Self {
        self.thread_id = thread_id.into();
        self
    }

    /// Run an agent turn with automatic memory management
    ///
    /// This method:
    /// 1. Injects previous messages from memory
    /// 2. Sets the input
    /// 3. Runs the agent through all handlers
    /// 4. Extracts output
    /// 5. Tracks user input + assistant response in memory
    /// 6. Persists state for next turn
    ///
    /// # Arguments
    ///
    /// * `input` - User input text
    ///
    /// # Returns
    ///
    /// The final output message from the agent
    ///
    /// # Example
    ///
    /// ```ignore
    /// let answer = agent.run_turn("What is Rust?").await?;
    /// println!("Agent: {}", answer);
    /// ```
    pub async fn run_turn(&mut self, input: impl Into<String>) -> Result<String> {
        let input_str = input.into();

        // Step 1: Inject previous messages into state if memory exists
        self.inject_previous_messages()?;

        // Step 2: Set the user input
        self.agent.state.set("input", json!(input_str.clone()));

        // Step 3: Run the agent
        let result_state = self.agent.run().await?;

        // Step 4: Extract the final output
        let output_value = result_state
            .get("final_output")
            .and_then(|v| v.as_str().map(|s| s.to_string()))
            .ok_or_else(|| {
                FlowgentraError::StateError(
                    "No final_output in state. Check your handlers return this field.".to_string(),
                )
            })?;

        // Step 5: Track this exchange in memory if memory is enabled
        self.track_in_memory(&input_str, &output_value)?;

        // Step 6: Persist state for next turn
        self.agent.state = result_state;

        Ok(output_value)
    }

    /// Inject previous messages from memory into state before running
    fn inject_previous_messages(&self) -> Result<()> {
        if let Some(memory) = &self.conversation_memory {
            // Get previous messages from memory
            let messages = memory.messages(&self.thread_id, None)?;

            if !messages.is_empty() {
                // Format as conversation history string
                let mut history = String::new();
                for msg in &messages {
                    let role = match msg.role {
                        MessageRole::User => "User",
                        MessageRole::Assistant => "Assistant",
                        MessageRole::System => "System",
                        MessageRole::Tool => "Tool",
                    };
                    history.push_str(&format!("{}: {}\n", role, msg.content));
                }

                // Inject into state for handlers to use
                self.agent.state.set("conversation_history", json!(history));
            }
        }

        Ok(())
    }

    /// Track the current exchange in memory for future turns
    fn track_in_memory(&self, input: &str, output: &str) -> Result<()> {
        if let Some(memory) = &self.conversation_memory {
            // Extract just the response content (remove any "FINAL OUTPUT:" prefix)
            let response_content = output.strip_prefix("FINAL OUTPUT:\n").unwrap_or(output);

            // Add to memory
            memory.add_message(&self.thread_id, Message::user(input))?;
            memory.add_message(&self.thread_id, Message::assistant(response_content))?;
        }

        Ok(())
    }

    /// Get the underlying Agent (for advanced use cases)
    pub fn agent(&self) -> &Agent {
        &self.agent
    }

    /// Get mutable access to Agent (for advanced use cases)
    pub fn agent_mut(&mut self) -> &mut Agent {
        &mut self.agent
    }

    /// Get conversation memory if enabled
    pub fn memory(&self) -> Option<&Arc<dyn ConversationMemory>> {
        self.conversation_memory.as_ref()
    }

    /// Get current thread ID
    pub fn thread_id(&self) -> &str {
        &self.thread_id
    }

    /// Clear memory for this thread (start fresh conversation)
    pub fn clear_memory(&self) -> Result<()> {
        if let Some(memory) = &self.conversation_memory {
            memory.clear(&self.thread_id)?;
        }
        Ok(())
    }

    /// Get memory statistics (number of messages, tokens, etc.)
    pub fn memory_stats(&self) -> Result<MemoryStats> {
        let mut stats = MemoryStats::default();

        if let Some(memory) = &self.conversation_memory {
            let messages = memory.messages(&self.thread_id, None)?;
            stats.message_count = messages.len();
            stats.user_messages = messages
                .iter()
                .filter(|m| matches!(m.role, MessageRole::User))
                .count();
            stats.assistant_messages = messages
                .iter()
                .filter(|m| matches!(m.role, MessageRole::Assistant))
                .count();

            // Rough token estimate (4 chars ≈ 1 token)
            let total_chars: usize = messages.iter().map(|m| m.content.len()).sum();
            stats.approximate_tokens = total_chars / 4;
        }

        Ok(stats)
    }
}

/// Statistics about memory usage
#[derive(Debug, Clone, Default)]
pub struct MemoryStats {
    /// Total messages stored
    pub message_count: usize,
    /// User messages
    pub user_messages: usize,
    /// Assistant messages
    pub assistant_messages: usize,
    /// Approximate token count (rough estimate)
    pub approximate_tokens: usize,
}

impl std::fmt::Display for MemoryStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Messages: {}, User: {}, Assistant: {}, ~Tokens: {}",
            self.message_count,
            self.user_messages,
            self.assistant_messages,
            self.approximate_tokens
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_stats_display() {
        let stats = MemoryStats {
            message_count: 4,
            user_messages: 2,
            assistant_messages: 2,
            approximate_tokens: 500,
        };

        assert_eq!(
            stats.to_string(),
            "Messages: 4, User: 2, Assistant: 2, ~Tokens: 500"
        );
    }
}
