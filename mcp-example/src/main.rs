use flowgentra_ai::prelude::*;
use serde_json::json;
use dotenv::dotenv;

mod handlers;

#[tokio::main]
async fn main() -> Result<()> {
    dotenv().ok();
    tracing_subscriber::fmt().with_max_level(tracing::Level::WARN).init();

    // Accept optional config path: cargo run -- config-docker.yaml
    let config_arg = std::env::args().nth(1).unwrap_or_default();
    let config_file = if config_arg.is_empty() { "config.yaml" } else { &config_arg };

    println!("\n=== MCP (Model Context Protocol) Example ===");
    println!("LLM autonomously discovers and calls MCP tools to answer questions.");
    println!("Config: {config_file}\n");
    println!("tool_coordinator (strategy: sequential)");
    println!("  |- discover_tools   -> list available MCP tools");
    println!("  |- solve_math       -> LLM uses tools to solve math");
    println!("  |- analyze_text     -> LLM uses tools to analyze text");
    println!("  \\- answer_question  -> LLM uses tools for a multi-step question\n");

    let mut agent = from_config_path(&format!("mcp-example/{}", config_file))
        .or_else(|_| from_config_path(config_file))?;

    agent.state.set("expression", json!("sqrt(625) + 3**3"));
    agent.state.set(
        "text",
        json!("Rust is a systems programming language focused on safety, speed, and concurrency. \
               It achieves memory safety without a garbage collector through its ownership system. \
               Rust has been voted the most loved programming language for several years running."),
    );
    agent.state.set(
        "question",
        json!("What is 100 degrees Celsius in Fahrenheit? Also what day of the week is it right now in UTC?"),
    );

    println!("Running MCP pipeline...\n");

    let final_state = agent.run().await?;

    // ── Results ──────────────────────────────────────────────────────────────
    println!("\n{}", "=".repeat(60));
    println!("RESULTS");
    println!("{}\n", "=".repeat(60));

    if let Some(tools) = final_state.get("tool_list").and_then(|v| v.as_array().cloned()) {
        println!("Discovered {} MCP tools", tools.len());
    }

    if let Some(calc) = final_state.get("calc_result") {
        println!("\nMath: {}", calc.as_str().unwrap_or("(no answer)"));
    }

    if let Some(analysis) = final_state.get("analysis_result") {
        println!("\nText analysis: {}", analysis.as_str().unwrap_or("(no answer)"));
    }

    if let Some(answer) = final_state.get("llm_answer") {
        println!("\nQuestion: {}", answer.as_str().unwrap_or("(no answer)"));
    }

    if let Some(meta) = final_state.get("__supervisor_meta__tool_coordinator") {
        let duration = meta.get("duration_ms").and_then(|v| v.as_u64()).unwrap_or(0);
        let success = meta.get("success").and_then(|v| v.as_bool()).unwrap_or(false);
        println!(
            "\nPipeline: {} in {duration}ms",
            if success { "success" } else { "failed" }
        );
        if let Some(errors) = meta.get("errors").and_then(|v| v.as_array()) {
            for err in errors {
                println!("  Error: {}", err);
            }
        }
    }

    println!();
    Ok(())
}
