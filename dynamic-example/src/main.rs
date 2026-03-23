use flowgentra_ai::prelude::*;
use serde_json::json;
use dotenv::dotenv;

mod handlers;

#[tokio::main]
async fn main() -> Result<()> {
    dotenv().ok();
    tracing_subscriber::fmt().with_max_level(tracing::Level::WARN).init();

    println!("\n=== Dynamic (LLM-driven) Strategy Example ===");
    println!("The supervisor asks an LLM which agents to call at runtime.");
    println!("The LLM sees the current state and decides the next step.\n");
    println!("dynamic_coordinator (strategy: dynamic)");
    println!("  ├─ researcher_agent  → gather raw data");
    println!("  ├─ analyst_agent     → analyze the data");
    println!("  └─ summarizer_agent  → produce summary\n");

    let mut agent = from_config_path("dynamic-example/config.yaml")
        .or_else(|_| from_config_path("config.yaml"))?;

    agent.state.set("topic", json!("Rust Programming Language"));
    println!("Topic: Rust Programming Language\n");
    println!("Running dynamic pipeline (LLM decides order)...\n");

    let final_state = agent.run().await?;

    println!("\n{}", "=".repeat(60));
    println!("RESULTS");
    println!("{}\n", "=".repeat(60));

    if let Some(v) = final_state.get("summary") {
        println!("{}", v.as_str().unwrap_or_default());
    }

    if let Some(meta) = final_state.get("__supervisor_meta__dynamic_coordinator") {
        println!("\ndynamic_coordinator stats:");
        let duration = meta.get("duration_ms").and_then(|v| v.as_u64()).unwrap_or(0);
        println!("  duration: {duration}ms");
        if let Some(iterations) = meta.get("iterations").and_then(|v| v.as_array()) {
            println!("  iterations: {}", iterations.len());
            for iter in iterations {
                let n = iter.get("iteration").and_then(|v| v.as_u64()).unwrap_or(0);
                let called = iter.get("agents_called").and_then(|v| v.as_array())
                    .map(|a| a.iter().filter_map(|v| v.as_str()).collect::<Vec<_>>().join(", "))
                    .unwrap_or_default();
                println!("    [{n}] LLM chose: [{called}]");
            }
        }
    }

    println!();
    Ok(())
}
