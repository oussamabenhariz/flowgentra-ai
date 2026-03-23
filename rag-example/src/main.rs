use flowgentra_ai::prelude::*;
use serde_json::json;
use dotenv;

mod handlers;

#[tokio::main]
async fn main() -> Result<()> {
    // Load .env from rag-example/ dir (works whether run from project root or rag-example/)
    dotenv::from_filename("rag-example/.env").ok();
    dotenv::dotenv().ok();
    tracing_subscriber::fmt().with_max_level(tracing::Level::WARN).init();

    println!("\n=== RAG Pipeline with ChromaDB + OpenAI Embeddings ===");
    println!("Index PDF documents (or built-in knowledge) into ChromaDB,");
    println!("retrieve relevant context, and answer using an LLM.\n");

    // Build agent (RAG is auto-initialized from config.yaml graph.rag section)
    let mut agent = from_config_path("rag-example/config.yaml")
        .or_else(|_| from_config_path("config.yaml"))?;

    // Print RAG config info
    if let Some(ref rag_config) = agent.config().graph.rag {
        println!("Vector store:  ChromaDB at {}", rag_config.vector_store.endpoint.as_deref().unwrap_or("localhost:8000"));
        println!("Embeddings:    {} ({})", rag_config.embeddings.provider, rag_config.embeddings.model.as_deref().unwrap_or("default"));
        println!("Collection:    {}", rag_config.vector_store.collection);
        println!("Retrieval:     top_k={}, threshold={:.2}", rag_config.retrieval.top_k, rag_config.retrieval.similarity_threshold);
        println!("PDF chunking:  {}ch / {}ch overlap\n", rag_config.pdf.chunk_size, rag_config.pdf.chunk_overlap);
    }

    println!("rag_coordinator (strategy: sequential)");
    println!("  |- index_documents     -> index PDFs / knowledge base into ChromaDB");
    println!("  |- retrieve_context    -> semantic search for relevant chunks");
    println!("  \\- answer_with_context -> LLM answers with retrieved context\n");

    agent.state.set(
        "query",
        json!("what is The state of the art of AI and future prospects ?"),
    );
    agent.state.set("pdf_dir", json!("rag-example/data"));

    let query_display = agent
        .state
        .get("query")
        .and_then(|v| v.as_str().map(String::from))
        .unwrap_or_default();
    println!("Query: \"{query_display}\"\n");
    println!("Running RAG pipeline...\n");

    let final_state = agent.run().await?;

    // ── Results ──────────────────────────────────────────────────────────────
    println!("\n{}", "=".repeat(60));
    println!("RESULTS");
    println!("{}\n", "=".repeat(60));

    if let Some(count) = final_state.get("indexed_count").and_then(|v| v.as_u64()) {
        println!("Indexed: {} chunks/documents", count);
    }

    if let Some(docs) = final_state.get("retrieved_documents").and_then(|v| v.as_array().cloned()) {
        println!("Retrieved: {} relevant documents", docs.len());
        for doc in &docs {
            let id = doc.get("id").and_then(|v| v.as_str()).unwrap_or("?");
            let score = doc.get("score").and_then(|v| v.as_f64()).unwrap_or(0.0);
            let source = doc
                .get("metadata")
                .and_then(|m| m.get("source"))
                .and_then(|s| s.as_str())
                .unwrap_or("unknown");
            println!("  - {} (score: {:.4}, source: {})", id, score, source);
        }
    }

    if let Some(answer) = final_state.get("answer") {
        println!("\nAnswer:\n{}", answer.as_str().unwrap_or("(no answer)"));
    }

    if let Some(used) = final_state.get("context_used").and_then(|v| v.as_bool()) {
        println!("\nContext used: {}", if used { "yes (RAG)" } else { "no (direct LLM)" });
    }

    if let Some(meta) = final_state.get("__supervisor_meta__rag_coordinator") {
        let duration = meta.get("duration_ms").and_then(|v| v.as_u64()).unwrap_or(0);
        let success = meta.get("success").and_then(|v| v.as_bool()).unwrap_or(false);
        println!(
            "\nPipeline: {} in {duration}ms",
            if success { "success" } else { "failed" }
        );
    }

    println!();
    Ok(())
}
