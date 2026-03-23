use flowgentra_ai::prelude::*;
use flowgentra_ai::register_handler;
use serde_json::json;

// ─────────────────────────────────────────────────────────────────────────────
// Handler 1: Discover available tools from the node's assigned MCP
// ─────────────────────────────────────────────────────────────────────────────

#[register_handler]
pub async fn discover_tools(state: SharedState) -> Result<SharedState> {
    println!("  [discover]    Connecting to MCP server...");

    let client = state.get_node_mcp_client()?;
    let tools = client.list_tools().await?;

    println!("  [discover]    Found {} tools:", tools.len());
    for tool in &tools {
        println!("                  - {} : {}", tool.name, tool.description.as_deref().unwrap_or("(no description)"));
    }

    let tool_list: Vec<_> = tools
        .iter()
        .map(|t| json!({"name": t.name, "description": t.description.as_deref().unwrap_or("")}))
        .collect();

    state.set("tool_list", json!(tool_list));
    Ok(state)
}

// ─────────────────────────────────────────────────────────────────────────────
// Handler 2: LLM solves a math problem using the node's MCP tools
// ─────────────────────────────────────────────────────────────────────────────

#[register_handler]
pub async fn solve_math(state: SharedState) -> Result<SharedState> {
    let expression = state
        .get("expression")
        .and_then(|v| v.as_str().map(String::from))
        .unwrap_or_else(|| "sqrt(144) + 2**10".to_string());

    println!("  [math]        Question: {expression}");
    println!("  [math]        LLM deciding which tools to use...");

    let response = state
        .chat_with_node_mcp_tools(
            vec![
                Message::system(
                    "You are a precise calculator assistant. Use the available tools to compute the answer. \
                     Give a short final answer with just the number and a brief explanation.",
                ),
                Message::user(format!("Calculate: {expression}")),
            ],
            5,
        )
        .await?;

    println!("  [math]        Answer: {}", response.content);
    state.set("calc_result", json!(response.content));
    Ok(state)
}

// ─────────────────────────────────────────────────────────────────────────────
// Handler 3: LLM analyzes text using the node's MCP tools
// ─────────────────────────────────────────────────────────────────────────────

#[register_handler]
pub async fn analyze_text(state: SharedState) -> Result<SharedState> {
    let text = state
        .get("text")
        .and_then(|v| v.as_str().map(String::from))
        .unwrap_or_else(|| {
            "Rust is a systems programming language focused on safety, speed, and concurrency. \
             It achieves memory safety without a garbage collector through its ownership system. \
             Rust has been voted the most loved programming language for several years running."
                .to_string()
        });

    println!("  [analyze]     Analyzing {} chars of text...", text.len());
    println!("  [analyze]     LLM deciding which tools to use...");

    let response = state
        .chat_with_node_mcp_tools(
            vec![
                Message::system(
                    "You are a text analysis assistant. Use the available tools to analyze the given text. \
                     Get word/sentence statistics, find top words, and compute a SHA-256 hash of the text. \
                     Summarize your findings clearly.",
                ),
                Message::user(format!("Analyze this text:\n\n{text}")),
            ],
            5,
        )
        .await?;

    println!("  [analyze]     Result: {}", response.content);
    state.set("analysis_result", json!(response.content));
    Ok(state)
}

// ─────────────────────────────────────────────────────────────────────────────
// Handler 4: LLM answers a multi-step question using the node's MCP tools
// ─────────────────────────────────────────────────────────────────────────────

#[register_handler]
pub async fn answer_question(state: SharedState) -> Result<SharedState> {
    let question = state
        .get("question")
        .and_then(|v| v.as_str().map(String::from))
        .unwrap_or_else(|| {
            "What is 100 degrees Celsius in Fahrenheit? \
             Also what day of the week is it right now in UTC?"
                .to_string()
        });

    println!("  [question]    Question: {question}");
    println!("  [question]    LLM deciding which tools to use...");

    let response = state
        .chat_with_node_mcp_tools(
            vec![
                Message::system(
                    "You are a helpful assistant with access to external tools. \
                     Use the available tools to answer the user's question accurately. \
                     Always use tools for calculations, conversions, and time queries rather than guessing.",
                ),
                Message::user(&question),
            ],
            5,
        )
        .await?;

    println!("  [question]    Answer: {}", response.content);
    state.set("llm_answer", json!(response.content));
    Ok(state)
}
