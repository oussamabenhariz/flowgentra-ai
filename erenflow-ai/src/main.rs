//! ErenFlowAI - Simple Example
//!
//! This is a basic example showing how to use ErenFlowAI to create and run an agent.
//! For more complex examples, see the `examples/` directory.
//!
//! Run with: `cargo run --example simple_example`

use erenflow_ai::prelude::*;
use serde_json::json;

#[tokio::main]
async fn main() -> Result<()> {
    println!("=== ErenFlowAI Example ===\n");

    // Create initial state
    let mut state = State::new();
    state.set("message", json!("Hello ErenFlowAI!"));
    state.set("step", json!(0));

    println!("Initial state: {}\n", state.to_json_string()?);

    // Example of state transformations
    state.set("step", json!(1));
    state.set("result", json!("Processing complete"));

    println!("Final state: {}\n", state.to_json_string()?);

    println!("✓ Example completed successfully!");
    Ok(())
}
