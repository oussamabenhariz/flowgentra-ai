//! FlowgentraAI - Example
//!
//! This example demonstrates how to define and use custom typed states.
//! With the new architecture, users define their own states using the #[derive(State)] macro.
//!\
//! For detailed examples, see: examples/typed_states.rs

fn main() {
    println!("=== FlowgentraAI Example ===\n");

    println!("Welcome to FlowgentraAI!");
    println!("\nWith the refactored state system, you can now:");
    println!("  ✓ Define custom state types with #[derive(State)]");
    println!("  ✓ Control field merge behavior with #[reducer(...)]");
    println!("  ✓ Build type-safe graphs with StateGraph<S: State>");
    println!("  ✓ Get full IDE support with compile-time type checking\n");

    println!("For examples, run:");
    println!("  cargo run --example typed_states\n");

    println!("Quick start:");
    println!("  use flowgentra_ai::prelude::*;");
    println!("  use serde::{{Serialize, Deserialize}};\n");

    println!("  #[derive(State, Clone, Debug, Serialize, Deserialize)]");
    println!("  struct MyState {{");
    println!("      input: String,");
    println!("      #[reducer(Append)]");
    println!("      messages: Vec<Message>,");
    println!("  }}\n");

    println!("✓ Ready to build workflows!");
}
