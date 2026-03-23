//! FlowgentraAI - Simple Example
//!
//! This is a basic example showing how to use FlowgentraAI to create and run an agent.
//! For more complex examples, see the `examples/` directory.
//!
//! Run with: `cargo run --example simple_example`

use flowgentra_ai::prelude::*;

#[tokio::main]
async fn main() -> Result<()> {
    println!("=== FlowgentraAI Example ===\n");

    // Create initial state
    let state = PlainState::new();

    println!("Initial state: {}\n", state.to_json_string()?);

    // Example state completed
    println!("✓ Example completed successfully!");
    Ok(())
}
