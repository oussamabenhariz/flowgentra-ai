use flowgentra_ai::core::rag::pdf;
use flowgentra_ai::core::rag::vector_db::{Document, VectorStoreBackend};
use flowgentra_ai::prelude::*;
use flowgentra_ai::register_handler;
use serde_json::json;

// ─────────────────────────────────────────────────────────────────────────────
// Handler 1: Index PDF documents (or plain-text fallback) into ChromaDB
// ─────────────────────────────────────────────────────────────────────────────

#[register_handler]
pub async fn index_documents(state: SharedState) -> Result<SharedState> {
    let rag_config = state.get_rag_config()?;
    let embeddings = state.get_rag_embeddings()?;
    let backend = state.get_rag_store().await?;

    let pdf_dir = state
        .get("pdf_dir")
        .and_then(|v| v.as_str().map(String::from))
        .unwrap_or_else(|| {
            // Support running from project root or from rag-example/
            if std::path::Path::new("rag-example/data").exists() {
                "rag-example/data".to_string()
            } else {
                "data".to_string()
            }
        });

    let chunk_size = rag_config.pdf.chunk_size;
    let chunk_overlap = rag_config.pdf.chunk_overlap;

    // Discover PDF files
    let pdf_files: Vec<_> = std::fs::read_dir(&pdf_dir)
        .map(|entries| {
            entries
                .filter_map(|e| e.ok())
                .filter(|e| {
                    e.path()
                        .extension()
                        .map(|ext| ext == "pdf")
                        .unwrap_or(false)
                })
                .map(|e| e.path())
                .collect()
        })
        .unwrap_or_default();

    if pdf_files.is_empty() {
        println!("  [index]       No PDF files found in '{pdf_dir}'.");
        println!("  [index]       Place PDF files in the data/ directory and try again.");
        state.set("indexed_count", json!(0));
        return Ok(state);
    }

    println!(
        "  [index]       Found {} PDF file(s) in '{pdf_dir}':",
        pdf_files.len()
    );
    for p in &pdf_files {
        let name = p.file_name().and_then(|n| n.to_str()).unwrap_or("?");
        let size = std::fs::metadata(p).map(|m| m.len()).unwrap_or(0);
        let size_str = if size >= 1_048_576 {
            format!("{:.1} MB", size as f64 / 1_048_576.0)
        } else {
            format!("{:.1} KB", size as f64 / 1024.0)
        };
        println!("  [index]         - {name} ({size_str})");
    }

    let mut total_chunks = 0usize;

    for pdf_path in &pdf_files {
        let file_name = pdf_path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("unknown.pdf");

        println!("  [index]       Processing: {file_name}");

        match pdf::extract_and_chunk(pdf_path, chunk_size, chunk_overlap).await {
            Ok(chunks) => {
                println!("  [index]         {} chunks extracted", chunks.len());

                // Batch embed to reduce API calls (16 texts per batch)
                let batch_size = 16;
                for batch in chunks.chunks(batch_size) {
                    let texts: Vec<&str> = batch.iter().map(|(_, text)| text.as_str()).collect();

                    let batch_embeddings = embeddings.embed_batch(texts).await.map_err(|e| {
                        FlowgentraError::ToolError(format!("Batch embedding failed: {}", e))
                    })?;

                    for ((chunk_id, chunk_text), embedding) in batch.iter().zip(batch_embeddings) {
                        let mut doc = Document::new(chunk_id.clone(), chunk_text.clone());
                        doc.embedding = Some(embedding);
                        doc.metadata
                            .insert("source".to_string(), json!(file_name));
                        doc.metadata
                            .insert("chunk_index".to_string(), json!(total_chunks));

                        backend.index(doc).await.map_err(|e| {
                            FlowgentraError::ToolError(format!("ChromaDB index failed: {}", e))
                        })?;

                        total_chunks += 1;
                    }

                    println!("  [index]         embedded batch ({} chunks)", batch.len());

                    // Small delay between batches to avoid rate limits
                    tokio::time::sleep(std::time::Duration::from_millis(500)).await;
                }
            }
            Err(e) => {
                println!("  [index]         Failed to extract: {e}");
            }
        }
    }

    println!("  [index]       Done. {total_chunks} chunks indexed into ChromaDB.");
    state.set("indexed_count", json!(total_chunks));
    Ok(state)
}

// ─────────────────────────────────────────────────────────────────────────────
// Handler 2: Retrieve relevant documents from ChromaDB
// ─────────────────────────────────────────────────────────────────────────────

#[register_handler]
pub async fn retrieve_context(state: SharedState) -> Result<SharedState> {
    let rag_config = state.get_rag_config()?;
    let embeddings = state.get_rag_embeddings()?;
    let backend = state.get_rag_store().await?;

    let query = state
        .get("query")
        .and_then(|v| v.as_str().map(String::from))
        .unwrap_or_else(|| "What is Rust?".to_string());

    let top_k = rag_config.retrieval.top_k;
    let threshold = rag_config.retrieval.similarity_threshold;

    println!("  [retrieve]    Query: \"{query}\"");

    let query_embedding = embeddings.embed(&query).await.map_err(|e| {
        FlowgentraError::ToolError(format!("Query embedding failed: {}", e))
    })?;

    let results = backend.search(query_embedding, top_k, None).await.map_err(|e| {
        FlowgentraError::ToolError(format!("ChromaDB search failed: {}", e))
    })?;

    // Filter by similarity threshold
    let filtered: Vec<_> = results
        .into_iter()
        .filter(|r| r.score >= threshold)
        .collect();

    println!("  [retrieve]    Found {} results (threshold >= {:.2}):", filtered.len(), threshold);
    for r in &filtered {
        let preview: String = r.text.chars().take(70).collect();
        println!(
            "                  - {} (score: {:.4}): {}...",
            r.id, r.score, preview
        );
    }

    let retrieved: Vec<_> = filtered
        .iter()
        .map(|r| {
            json!({
                "id": r.id,
                "text": r.text,
                "score": r.score,
                "metadata": r.metadata,
            })
        })
        .collect();

    let context = filtered
        .iter()
        .map(|r| r.text.as_str())
        .collect::<Vec<_>>()
        .join("\n\n");

    state.set("retrieved_documents", json!(retrieved));
    state.set("retrieval_count", json!(filtered.len()));
    state.set("context", json!(context));

    Ok(state)
}

// ─────────────────────────────────────────────────────────────────────────────
// Handler 3: LLM answers the question using retrieved context (RAG)
// ─────────────────────────────────────────────────────────────────────────────

#[register_handler]
pub async fn answer_with_context(state: SharedState) -> Result<SharedState> {
    let query = state
        .get("query")
        .and_then(|v| v.as_str().map(String::from))
        .unwrap_or_else(|| "What is Rust?".to_string());

    let context = state
        .get("context")
        .and_then(|v| v.as_str().map(String::from))
        .unwrap_or_default();

    let retrieval_count = state
        .get("retrieval_count")
        .and_then(|v| v.as_u64())
        .unwrap_or(0);

    println!("  [answer]      Augmenting prompt with {retrieval_count} retrieved documents...");

    let llm = state.get_llm_client()?;

    let messages = if context.is_empty() {
        println!("  [answer]      No context found, answering without RAG...");
        vec![
            Message::system(
                "You are a helpful assistant. Answer the user's question to the best of your knowledge.",
            ),
            Message::user(&query),
        ]
    } else {
        println!("  [answer]      Sending augmented prompt to LLM...");
        vec![
            Message::system(
                "You are a helpful assistant. Answer the user's question based ONLY on the \
                 provided context. If the context doesn't contain enough information, say so. \
                 Be concise and accurate.",
            ),
            Message::user(format!(
                "Context:\n{context}\n\nQuestion: {query}\n\nAnswer based on the context above:"
            )),
        ]
    };

    let response = match llm.chat(messages).await {
        Ok(r) => r,
        Err(e) => {
            println!("  [answer]      LLM error: {e}");
            state.set("answer", json!(format!("LLM error: {e}")));
            state.set("context_used", json!(!context.is_empty()));
            return Ok(state);
        }
    };
    let answer = response.content.trim().to_string();

    println!("  [answer]      Got {} chars from LLM", answer.len());

    state.set("answer", json!(answer));
    state.set("context_used", json!(!context.is_empty()));

    Ok(state)
}
