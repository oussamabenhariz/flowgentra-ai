/// Simple QA Agent Example demonstrating:
/// - Loading state and managing it across handlers
/// - Multi-step handler execution
/// - Using serde_json for state manipulation
///
/// Run with:
/// ```bash
/// cargo run --example simple_agent
/// ```

use erenflow_ai::prelude::*;
use serde_json::json;

// ============================================================================
// HANDLERS
// ============================================================================

/// Handler 1: Validate and prepare user input
async fn validate_input(mut state: State) -> Result<State> {
    println!("    [1/4] validate_input");

    let question = state
        .get("user_question")
        .and_then(|v| v.as_str())
        .unwrap_or("No question");

    if question.is_empty() {
        return Err(ErenFlowError::ValidationError(
            "Question cannot be empty".to_string(),
        ));
    }

    state.set("validated_question", json!(question.to_string()));
    state.set(
        "validation_timestamp",
        json!(chrono::Utc::now().to_rfc3339()),
    );

    Ok(state)
}

/// Handler 2: Enrich with context
async fn enrich_context(mut state: State) -> Result<State> {
    println!("    [2/4] enrich_context");

    let question = state
        .get("validated_question")
        .and_then(|v| v.as_str())
        .unwrap_or("Unknown");

    let context = if question.to_lowercase().contains("rust") {
        "Rust is a systems programming language focusing on safety, speed, and concurrency."
    } else if question.to_lowercase().contains("ai") {
        "AI involves using computers to simulate human intelligence and decision-making."
    } else {
        "General knowledge is available for this question."
    };

    state.set("context", json!(context.to_string()));
    Ok(state)
}

/// Handler 3: Generate response
async fn generate_response(mut state: State) -> Result<State> {
    println!("    [3/4] generate_response");

    let question = state
        .get("validated_question")
        .and_then(|v| v.as_str())
        .unwrap_or("Unknown");

    let context = state
        .get("context")
        .and_then(|v| v.as_str())
        .unwrap_or("");

    let response = format!(
        "Based on the question '{}' and context '{}': This demonstrates how handlers can process state sequentially in a workflow.",
        question, context
    );

    state.set("generated_response", json!(response));
    Ok(state)
}

/// Handler 4: Format final output
async fn format_output(mut state: State) -> Result<State> {
    println!("    [4/4] format_output");

    let question = state
        .get("validated_question")
        .and_then(|v| v.as_str())
        .unwrap_or("Unknown");

    let response = state
        .get("generated_response")
        .and_then(|v| v.as_str())
        .unwrap_or("No response");

    let final_output = json!({
        "status": "success",
        "question": question,
        "response": response,
        "timestamp": chrono::Utc::now().to_rfc3339(),
    });

    state.set("final_output", final_output);
    Ok(state)
}

// ============================================================================
// MAIN
// ============================================================================

#[tokio::main]
async fn main() -> Result<()> {
    println!("\n{}", "=".repeat(70));
    println!("🤖 Simple Agent Example - Config, Handlers, and Main");
    println!("{}\n", "=".repeat(70));

    // Initialize state
    println!("📋 Initializing agent state...");
    let mut state = State::new();
    let user_question = "What is Rust and why should I learn it?";

    state.set("user_question", json!(user_question));
    state.set("agent_name", json!("SimpleAgent"));
    state.set("request_id", json!(uuid::Uuid::new_v4().to_string()));

    println!("✓ State initialized\n");

    println!("❓ User Question:");
    println!("   \"{}\"\n", user_question);

    println!("🔄 Executing handler pipeline:\n");

    // Execute handlers in sequence
    state = validate_input(state).await?;
    state = enrich_context(state).await?;
    state = generate_response(state).await?;
    state = format_output(state).await?;

    println!("\n{}", "=".repeat(70));
    println!("✅ Agent execution completed successfully\n");

    // Display results
    if let Some(output) = state.get("final_output") {
        println!("📤 Final Output:\n");
        println!("{}\n", serde_json::to_string_pretty(&output)?);
    }

    // Display all state
    println!("📊 Full State:");
    println!("{}\n", serde_json::to_string_pretty(&state.to_value())?);

    println!("{}", "=".repeat(70));
    println!("✨ Example complete\n");
    println!("This demonstrates:");
    println!("  • Handler functions processing state");
    println!("  • State flowing between handlers");
    println!("  • JSON manipulation with serde_json");
    println!("  • Async/await with tokio");
    println!("  • Error handling with Result type\n");
    println!("{}\n", "=".repeat(70));

    Ok(())
}
