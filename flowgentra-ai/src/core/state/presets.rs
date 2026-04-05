//! # Preset State Types
//!
//! Common state types for typical agent workflows.
//! These replace the old DynState with compile-time type safety.

use crate::core::llm::Message;
use crate::core::rag::Document;
use serde::{Deserialize, Serialize};

/// Minimal state for simple workflows - just input and output.
#[derive(::flowgentra_ai_macros::State, Clone, Debug, Serialize, Deserialize)]
pub struct SimpleState {
    /// The input to process
    pub input: String,
    /// The resulting output
    pub output: Option<String>,
}

impl SimpleState {
    /// Create a new SimpleState with input
    pub fn new(input: impl Into<String>) -> Self {
        Self {
            input: input.into(),
            output: None,
        }
    }
}

/// Message-focused state for chat workflows.
#[derive(::flowgentra_ai_macros::State, Clone, Debug, Serialize, Deserialize)]
pub struct MessageState {
    /// Accumulated messages in the conversation
    #[reducer(Append)]
    pub messages: Vec<Message>,

    /// Optional summary of the conversation
    pub summary: Option<String>,
}

impl MessageState {
    /// Create a new MessageState with initial messages
    pub fn new(messages: Vec<Message>) -> Self {
        Self {
            messages,
            summary: None,
        }
    }

    /// Create an empty MessageState
    pub fn empty() -> Self {
        Self {
            messages: Vec::new(),
            summary: None,
        }
    }
}

/// Agent state for task-oriented workflows.
#[derive(::flowgentra_ai_macros::State, Clone, Debug, Serialize, Deserialize)]
pub struct AgentState {
    /// The user's query or goal
    pub query: String,

    /// Conversation history
    #[reducer(Append)]
    pub messages: Vec<Message>,

    /// List of tools the agent has used
    pub tools_used: Vec<String>,

    /// The final result or answer
    pub result: Option<String>,
}

impl AgentState {
    /// Create a new AgentState with a query
    pub fn new(query: impl Into<String>) -> Self {
        Self {
            query: query.into(),
            messages: Vec::new(),
            tools_used: Vec::new(),
            result: None,
        }
    }
}

/// RAG-focused state for retrieval-augmented generation workflows.
#[derive(::flowgentra_ai_macros::State, Clone, Debug, Serialize, Deserialize)]
pub struct RAGState {
    /// The user's query
    pub query: String,

    /// Retrieved documents
    #[reducer(Append)]
    pub documents: Vec<Document>,

    /// The generated answer using the retrieved documents
    pub answer: Option<String>,

    /// Relevance scores for the retrieved documents
    pub relevance_scores: Vec<f64>,
}

impl RAGState {
    /// Create a new RAGState with a query
    pub fn new(query: impl Into<String>) -> Self {
        Self {
            query: query.into(),
            documents: Vec::new(),
            answer: None,
            relevance_scores: Vec::new(),
        }
    }
}

/// Evaluation state for workflows that include evaluation/grading.
#[derive(::flowgentra_ai_macros::State, Clone, Debug, Serialize, Deserialize)]
pub struct EvaluationState {
    /// Input to evaluate
    pub input: String,

    /// Generated output/answer
    pub output: String,

    /// Evaluation score (0.0 to 1.0)
    pub score: f64,

    /// Evaluation feedback or explanation
    pub feedback: String,

    /// Whether the output passed evaluation criteria
    pub passed: bool,
}

impl EvaluationState {
    /// Create a new EvaluationState
    pub fn new(input: impl Into<String>, output: impl Into<String>) -> Self {
        Self {
            input: input.into(),
            output: output.into(),
            score: 0.0,
            feedback: String::new(),
            passed: false,
        }
    }
}

/// Multi-agent orchestration state - tracks multiple agent executions.
#[derive(::flowgentra_ai_macros::State, Clone, Debug, Serialize, Deserialize)]
pub struct SupervisorState {
    /// The original task or query
    pub task: String,

    /// Results from each child agent
    #[reducer(Append)]
    pub agent_results: Vec<AgentResult>,

    /// Final aggregated result
    pub final_result: Option<String>,

    /// Overall status
    pub status: String,
}

/// Result from a single agent execution
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AgentResult {
    /// Name of the agent that ran
    pub agent_name: String,
    /// The result from this agent
    pub result: String,
    /// Confidence score
    pub confidence: f64,
}

impl SupervisorState {
    /// Create a new SupervisorState with a task
    pub fn new(task: impl Into<String>) -> Self {
        Self {
            task: task.into(),
            agent_results: Vec::new(),
            final_result: None,
            status: "pending".to_string(),
        }
    }
}
