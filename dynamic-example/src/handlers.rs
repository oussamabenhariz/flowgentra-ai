use flowgentra_ai::prelude::*;
use flowgentra_ai::register_handler;
use serde_json::json;

#[register_handler]
pub async fn gather_data(state: SharedState) -> Result<SharedState> {
    let topic = state
        .get("topic")
        .and_then(|v| v.as_str().map(String::from))
        .unwrap_or_else(|| "Unknown".to_string());

    println!("  [researcher]  Researching: {topic}");

    let llm = state.get_llm_client()?;

    let messages = vec![
        Message::system(
            "You are a research assistant. Given a topic, provide a detailed factual overview. \
             Include: description, key features, history, strengths, and weaknesses. \
             Respond in plain text, be thorough and accurate.",
        ),
        Message::user(format!("Research this topic thoroughly: {topic}")),
    ];

    println!("  [researcher]  Calling LLM...");
    let response = llm.chat(messages).await?;
    let description = response.content.trim().to_string();
    println!("  [researcher]  Got {} chars from LLM", description.len());

    state.set(
        "raw_data",
        json!({
            "topic": topic,
            "description": description,
        }),
    );
    Ok(state)
}

#[register_handler]
pub async fn analyze_data(state: SharedState) -> Result<SharedState> {
    let raw = match state.get("raw_data") {
        Some(v) => v,
        None => return Err(flowgentra_ai::core::error::FlowgentraError::ToolError(
            "analyze_data: no raw_data in state".to_string(),
        )),
    };

    let description = raw
        .get("description")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    let topic = raw
        .get("topic")
        .and_then(|v| v.as_str())
        .unwrap_or("unknown");

    println!("  [analyst]     Analyzing {} chars about '{topic}'...", description.len());

    let llm = state.get_llm_client()?;

    let messages = vec![
        Message::system(
            "You are a data analyst. Analyze the provided text and return a JSON object with:\n\
             - \"strengths\": array of key strengths mentioned\n\
             - \"weaknesses\": array of weaknesses or challenges mentioned\n\
             - \"sentiment\": \"positive\", \"negative\", or \"mixed\"\n\
             - \"key_points\": array of 3-5 most important takeaways\n\
             - \"word_count\": approximate word count of the input\n\n\
             Respond with ONLY valid JSON, nothing else.",
        ),
        Message::user(format!("Analyze this text about \"{topic}\":\n\n{description}")),
    ];

    println!("  [analyst]     Calling LLM...");
    let response = llm.chat(messages).await?;
    let content = response.content.trim();
    println!("  [analyst]     LLM response: {content}");

    // Strip markdown code fences if present (e.g. ```json ... ```)
    let content = content
        .strip_prefix("```json")
        .or_else(|| content.strip_prefix("```"))
        .unwrap_or(content)
        .strip_suffix("```")
        .unwrap_or(content)
        .trim();

    // Try to parse as JSON, if it fails store as raw text
    let analysis = match serde_json::from_str::<serde_json::Value>(content) {
        Ok(v) => v,
        Err(e) => {
            println!("  [analyst]     WARNING: LLM did not return valid JSON: {e}");
            json!({ "raw_analysis": content, "parse_error": e.to_string() })
        }
    };

    state.set("analysis", analysis);
    println!("  [analyst]     Done.");
    Ok(state)
}

#[register_handler]
pub async fn summarize(state: SharedState) -> Result<SharedState> {
    let raw = match state.get("raw_data") {
        Some(v) => v,
        None => return Err(flowgentra_ai::core::error::FlowgentraError::ToolError(
            "summarize: no raw_data in state".to_string(),
        )),
    };
    let analysis = match state.get("analysis") {
        Some(v) => v,
        None => return Err(flowgentra_ai::core::error::FlowgentraError::ToolError(
            "summarize: no analysis in state".to_string(),
        )),
    };

    let topic = raw
        .get("topic")
        .and_then(|v| v.as_str())
        .unwrap_or("unknown");

    println!("  [summarizer]  Generating summary for '{topic}'...");

    let llm = state.get_llm_client()?;

    let messages = vec![
        Message::system(
            "You are a technical writer. Given research data and its analysis, \
             write a clear, well-structured summary report. Use markdown formatting.",
        ),
        Message::user(format!(
            "Write a summary report for \"{topic}\".\n\n\
             Research data:\n{}\n\n\
             Analysis:\n{}",
            serde_json::to_string_pretty(&raw).unwrap_or_default(),
            serde_json::to_string_pretty(&analysis).unwrap_or_default(),
        )),
    ];

    println!("  [summarizer]  Calling LLM...");
    let response = llm.chat(messages).await?;
    let summary = response.content.trim().to_string();
    println!("  [summarizer]  Got {} chars summary", summary.len());

    state.set("summary", json!(summary));
    Ok(state)
}
