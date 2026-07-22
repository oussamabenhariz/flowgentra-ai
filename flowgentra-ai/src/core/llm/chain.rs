//! `Chain` — sequential prompt-template → LLM composition sugar.
//!
//! Not a new execution engine: [`StateGraph`](crate::core::state_graph::StateGraph)
//! remains the tool for anything with branching, loops, retries, or
//! persistence. `Chain` exists because building a whole graph for the
//! extremely common "fill in a prompt, call the LLM" pipeline is pure
//! boilerplate — this collapses it to one call.
//!
//! ```
//! use flowgentra_ai::core::llm::{Chain, MockLLM, PromptTemplate};
//! use std::sync::Arc;
//!
//! # async fn demo() {
//! let prompt = PromptTemplate::new("Translate '{text}' to French.");
//! let llm = Arc::new(MockLLM::always("Bonjour"));
//! let chain = Chain::new(prompt, llm);
//!
//! let reply = chain.invoke(&[("text", "Hello")]).await.unwrap();
//! assert_eq!(reply.content, "Bonjour");
//! # }
//! ```

use std::sync::Arc;

use super::prompt_template::PromptTemplate;
use super::{Message, LLM};
use crate::core::error::{FlowgentraError, Result};

/// A prompt template piped into an LLM. See the [module docs](self) for why
/// this exists alongside `StateGraph`.
#[derive(Clone)]
pub struct Chain {
    prompt: PromptTemplate,
    llm: Arc<dyn LLM>,
}

impl Chain {
    /// Compose a prompt template with an LLM.
    pub fn new(prompt: PromptTemplate, llm: Arc<dyn LLM>) -> Self {
        Self { prompt, llm }
    }

    /// Format the prompt with `variables`, send it to the LLM, return the reply.
    pub async fn invoke(&self, variables: &[(&str, &str)]) -> Result<Message> {
        let text = self
            .prompt
            .format(variables)
            .map_err(|e| FlowgentraError::LLMError(e.to_string()))?;
        self.llm.chat(vec![Message::user(text)]).await
    }

    /// Like [`invoke`](Self::invoke), but parses the reply as JSON — pipes
    /// through [`LLM::chat_structured`], so the same "respond with valid
    /// JSON" instruction and parsing applies.
    pub async fn invoke_structured(&self, variables: &[(&str, &str)]) -> Result<serde_json::Value> {
        let text = self
            .prompt
            .format(variables)
            .map_err(|e| FlowgentraError::LLMError(e.to_string()))?;
        self.llm.chat_structured(vec![Message::user(text)]).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::llm::MockLLM;

    #[tokio::test]
    async fn invoke_formats_prompt_and_calls_llm() {
        let prompt = PromptTemplate::new("Translate '{text}' to French.");
        let llm = Arc::new(MockLLM::always("Bonjour"));
        let chain = Chain::new(prompt, llm);

        let reply = chain.invoke(&[("text", "Hello")]).await.unwrap();
        assert_eq!(reply.content, "Bonjour");
    }

    #[tokio::test]
    async fn missing_variable_surfaces_as_llm_error() {
        let prompt = PromptTemplate::new("Translate '{text}' to French.");
        let llm = Arc::new(MockLLM::always("unused"));
        let chain = Chain::new(prompt, llm);

        let err = chain.invoke(&[]).await.unwrap_err();
        assert!(matches!(err, FlowgentraError::LLMError(_)), "{err:?}");
    }

    #[tokio::test]
    async fn invoke_structured_parses_json_reply() {
        let prompt = PromptTemplate::new("List colors matching {query}");
        let llm = Arc::new(MockLLM::always(r#"["red", "blue"]"#));
        let chain = Chain::new(prompt, llm);

        let value = chain
            .invoke_structured(&[("query", "primary")])
            .await
            .unwrap();
        assert_eq!(value, serde_json::json!(["red", "blue"]));
    }
}
