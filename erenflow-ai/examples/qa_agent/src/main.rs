/// QA Agent Example
/// 
/// This example demonstrates:
/// 1. Loading agent configuration from config.yaml
/// 2. Automatic handler discovery via #[register_handler] macro
/// 3. Running a multi-step workflow
/// 4. State management across handlers
///
/// To run:
/// 1. Set your Mistral API key: export MISTRAL_API_KEY=your_key_here
/// 2. Run: cargo run --example qa_agent
///    Or:  cd examples/qa_agent && cargo run

mod handlers;

use erenflow_ai::prelude::*;
use serde_json::json;
use std::env;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

    println!("\n{}", "=".repeat(70));
    println!("🤖 QA Agent Example - Loading from config.yaml");
    println!("{}\n", "=".repeat(70));

    // Check for API key
    if env::var("MISTRAL_API_KEY").is_err() {
        eprintln!("❌ Error: MISTRAL_API_KEY environment variable not set");
        eprintln!("\nSet it with:");
        eprintln!("  export MISTRAL_API_KEY=your_mistral_api_key");
        eprintln!("\nGet a free key at: https://console.mistral.ai/");
        std::process::exit(1);
    }

    // Load agent from config.yaml
    println!("📋 Loading configuration from config.yaml...\n");

    let config_path = "examples/qa_agent/config.yaml";
    
    let agent = match from_config_path(config_path) {
        Ok(agent) => {
            println!("✅ Configuration loaded successfully!");
            println!("   Agent: {}\n", "qa_agent");
            agent
        }
        Err(e) => {
            eprintln!("❌ Failed to load config: {}", e);
            eprintln!("\nMake sure you're running from the project root:");
            eprintln!("  cd erenflow-ai");
            std::process::exit(1);
        }
    };

    // Create initial state with user question
    let mut state = State::new();
    let user_question = "What is Rust and why is it important?";
    
    state.set("user_question", json!(user_question));
    state.set("request_id", json!(uuid::Uuid::new_v4().to_string()));

    println!("📝 User Question:");
    println!("   \"{}\"\n", user_question);

    println!("🔄 Executing workflow:\n");
    println!("   START");

    // Run the agent (this will execute all handlers in the workflow)
    match agent.run(state).await {
        Ok(final_state) => {
            println!("   ↓");
            println!("   END\n");

            println!("{}", "=".repeat(70));
            println!("✅ Workflow completed successfully!\n");

            // Display the final response
            if let Some(response) = final_state.get("final_response") {
                println!("📤 Final Response:\n");
                println!("{}\n", serde_json::to_string_pretty(&response).unwrap());
            }

            // Display execution summary
            println!("{}", "-".repeat(70));
            println!("📊 Execution Summary:");
            println!("   ✓ Input validation");
            println!("   ✓ Context retrieval");
            println!("   ✓ Answer generation");
            println!("   ✓ Response formatting");
            
            if let Some(timestamp) = final_state.get("validation_timestamp") {
                println!("\n⏱️  Workflow started at: {}", timestamp);
            }

            println!("\n{}", "=".repeat(70));
            println!("✨ Example completed successfully!");
            println!("{}\n", "=".repeat(70));
        }
        Err(e) => {
            println!("   ❌ ERROR\n");
            println!("{}", "=".repeat(70));
            eprintln!("❌ Workflow failed: {}\n", e);

            eprintln!("Troubleshooting:");
            eprintln!("  1. Verify MISTRAL_API_KEY is set correctly");
            eprintln!("  2. Check your internet connection");
            eprintln!("  3. Ensure config.yaml exists in the correct location");
            eprintln!("  4. Check that all handlers are properly registered");

            println!("{}\n", "=".repeat(70));
            std::process::exit(1);
        }
    }

    Ok(())
}
