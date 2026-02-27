/// Handler implementations for the QA Agent
/// Each handler is registered using #[register_handler] for auto-discovery

use erenflow_ai::prelude::*;
use serde_json::json;

/// Handler 1: Validate and prepare the user input
#[register_handler]
pub async fn validate_input(mut state: State) -> Result<State> {
    println!("  ✓ Handler: validate_input");

    // Get the user question
    let question = state
        .get("user_question")
        .and_then(|v| v.as_str())
        .unwrap_or("No question provided");

    // Validate
    if question.is_empty() {
        return Err(ErenFlowError::InvalidInput(
            "Question cannot be empty".to_string(),
        ));
    }

    let question_length = question.len();
    println!("    Input validation passed. Question length: {} chars", question_length);

    // Store validated question
    state.set("validated_question", json!(question.to_string()));
    state.set(
        "validation_timestamp",
        json!(chrono::Utc::now().to_rfc3339()),
    );

    Ok(state)
}

/// Handler 2: Retrieve context for the question
#[register_handler]
pub async fn get_context(mut state: State) -> Result<State> {
    println!("  ✓ Handler: get_context");

    let question = state
        .get("validated_question")
        .and_then(|v| v.as_str())
        .unwrap_or("Unknown");

    // In a real application, you would query a database or knowledge base
    // For this demo, we'll provide mock context
    let context = if question.to_lowercase().contains("rust") {
        "Rust is a systems programming language that runs blazingly fast, \
         prevents segfaults, and guarantees thread safety."
    } else if question.to_lowercase().contains("ai") {
        "Artificial Intelligence (AI) is the simulation of human intelligence \
         processes by computer systems."
    } else {
        "No specific context found for this question."
    };

    println!("    Context retrieved for question");
    state.set("context", json!(context.to_string()));

    Ok(state)
}

/// Handler 3: Generate answer using LLM
/// This handler uses the configured Mistral LLM
#[register_handler]
pub async fn generate_answer(mut state: State) -> Result<State> {
    println!("  ✓ Handler: generate_answer");

    let question = state
        .get("validated_question")
        .and_then(|v| v.as_str())
        .unwrap_or("Unknown question");

    let context = state
        .get("context")
        .and_then(|v| v.as_str())
        .unwrap_or("");

    // Construct prompt for LLM
    let prompt = format!(
        "Context: {}\n\nQuestion: {}\n\nProvide a concise and helpful answer.",
        context, question
    );

    println!("    Sending to Mistral LLM...");

    // The LLM would be called here in a real implementation
    // For now, we'll simulate the response
    let answer = format!(
        "Based on the provided context and question: '{}', \
         this is a comprehensive answer that addresses the key aspects of the topic.",
        question
    );

    state.set("llm_prompt", json!(prompt));
    state.set("llm_answer", json!(answer));

    Ok(state)
}

/// Handler 4: Format the response for the user
#[register_handler]
pub async fn format_response(mut state: State) -> Result<State> {
    println!("  ✓ Handler: format_response");

    let question = state
        .get("validated_question")
        .and_then(|v| v.as_str())
        .unwrap_or("Unknown");

    let answer = state
        .get("llm_answer")
        .and_then(|v| v.as_str())
        .unwrap_or("Unable to generate answer");

    // Format the response nicely
    let formatted_response = json!({
        "question": question,
        "answer": answer,
        "timestamp": chrono::Utc::now().to_rfc3339(),
        "status": "success"
    });

    println!("    Response formatted successfully");
    state.set("final_response", formatted_response);

    Ok(state)
}
