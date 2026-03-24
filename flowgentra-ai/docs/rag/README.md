# RAG (Retrieval-Augmented Generation) Guide

Give your agent access to your own documents and data through vector search.

## How RAG Works

1. **Index** -- Upload documents as vector embeddings into a store
2. **Search** -- For each query, find the most relevant documents
3. **Augment** -- Pass retrieved documents as context to the LLM

## Vector Stores

| Store | Type | Best For |
|-------|------|----------|
| **PineconeStore** | Cloud (REST API) | Production, managed, scalable |
| **QdrantStore** | Self-hosted (REST API) | Privacy, full control |
| **ChromaStore** | Local / self-hosted | Small datasets, development |
| **InMemoryStore** | In-process | Testing, prototyping |

### PineconeStore

Full REST API integration with Pinecone:

```rust
use flowgentra_ai::core::rag::PineconeStore;

let store = PineconeStore::new(
    "your-api-key",
    "https://your-index-abc123.svc.pinecone.io"
);

// Upsert vectors
store.upsert(vectors).await?;

// Query similar vectors
let results = store.query(query_vector, 5).await?;

// Fetch by ID
let vectors = store.fetch(&["id1", "id2"]).await?;

// Delete
store.delete(&["id1"]).await?;
```

### QdrantStore

Full REST API integration with Qdrant:

```rust
use flowgentra_ai::core::rag::QdrantStore;

let store = QdrantStore::new(
    "http://localhost:6333",
    "my_collection"
);

// Upsert points
store.upsert(points).await?;

// Search
let results = store.query(query_vector, 5).await?;

// Delete
store.delete(&["id1"]).await?;
```

### Config-Based Setup

```yaml
rag:
  enabled: true
  vector_store:
    type: pinecone              # or qdrant, chroma, memory
    index_name: "my-documents"
    api_key: ${PINECONE_API_KEY}
  embedding_model: openai
  embedding_dimension: 1536
  top_k: 5
  similarity_threshold: 0.7
```

For Qdrant:

```yaml
rag:
  enabled: true
  vector_store:
    type: qdrant
    url: "http://localhost:6333"
    collection: "documents"
  embedding_model: openai
```

## Using RAG in Handlers

```rust
pub async fn answer_with_context(mut state: State) -> Result<State> {
    let query = state.get_str("user_question")?;

    // Retrieve relevant documents
    let docs = rag_client.retrieve(query, 5).await?;

    // Build context string
    let context = docs.iter()
        .map(|d| d.content.clone())
        .collect::<Vec<_>>()
        .join("\n---\n");

    // LLM generates answer using retrieved context
    let answer = llm.chat(vec![
        Message::system(format!("Use this context:\n{}", context)),
        Message::user(query),
    ]).await?;

    state.set("answer", json!(answer.content));
    Ok(state)
}
```

## Retrieval Strategies

| Strategy | How It Works | When to Use |
|----------|-------------|-------------|
| `semantic` | Similarity-based vector search | Default, works for most cases |
| `bm25` | Keyword matching | When exact terms matter |
| `hybrid` | Combines semantic + keyword | Best accuracy, higher latency |

## Chunking Configuration

```yaml
rag:
  chunk_size: 1024       # Characters per chunk
  chunk_overlap: 200     # Overlap between chunks
  separator: "\n\n"      # Chunk boundary
```

## Text Splitters

Split documents into chunks before indexing. FlowgentraAI provides multiple splitter types through a unified `TextSplitter` trait:

| Splitter | Best For |
|----------|----------|
| **RecursiveCharacterTextSplitter** | General-purpose, respects paragraph boundaries |
| **MarkdownTextSplitter** | Markdown docs, preserves heading structure |
| **CodeTextSplitter** | Source code, respects language syntax (9 languages) |
| **HTMLTextSplitter** | Web pages, splits on HTML tags |
| **TokenTextSplitter** | Token-budget-aware chunking |

```rust
use flowgentra_ai::prelude::*;

// Recursive (general purpose)
let splitter = RecursiveCharacterTextSplitter::new(500, 50);
let chunks = splitter.split("Your long document...");

// Code-aware (Rust, Python, JS, TS, Go, Java, C, C++, Ruby)
let splitter = CodeTextSplitter::new(Language::Python, 500, 50);
let chunks = splitter.split("def hello():\n    print('hi')\n...");

// Markdown-aware
let splitter = MarkdownTextSplitter::new(500, 50);
let chunks = splitter.split("# Title\nContent...\n## Section\nMore content...");
```

Each chunk includes metadata: `start_index`, `end_index`, `chunk_index`, and optional `source`.

---

## Embeddings Providers

| Provider | Auth | Dimension |
|----------|------|-----------|
| **OpenAIEmbeddings** | Bearer token | 1536 (ada-002) |
| **MistralEmbeddings** | Bearer token | 1024 |
| **HuggingFaceEmbeddings** | Bearer token | Auto-detected |
| **OllamaEmbeddings** | None (local) | Model-dependent |
| **MockEmbeddings** | None | Configurable |

All implement the `EmbeddingsProvider` trait with `embed()` and `embed_batch()`.

### HuggingFace Embeddings

```rust
use flowgentra_ai::prelude::*;

let embeddings = HuggingFaceEmbeddings::new(
    "sentence-transformers/all-MiniLM-L6-v2",
    "hf_your_key",
);

// Auto-detects dimension from model name
let vector = embeddings.embed("Hello world").await?;
assert_eq!(vector.len(), 384); // MiniLM-L6 = 384

// Self-hosted TEI server
let embeddings = HuggingFaceEmbeddings::new("model", "key")
    .with_endpoint("http://localhost:8080/embed")
    .with_dimension(768);
```

### Cached Embeddings

Wrap any provider to avoid re-embedding identical texts:

```rust
use flowgentra_ai::prelude::*;

let cached = CachedEmbeddings::new(embeddings_provider);
let v1 = cached.embed("hello").await?; // computes
let v2 = cached.embed("hello").await?; // cache hit
```

---

## Retriever

The `Retriever` orchestrates a full retrieval pipeline: embed query → vector search → optional hybrid merge → rerank → dedup → filter.

```rust
use flowgentra_ai::prelude::*;

let retriever = Retriever::new(store, embeddings)
    .with_top_k(10)
    .with_reranker(reranker);

let results = retriever.retrieve("What is ownership in Rust?").await?;

// Multi-query retrieval (query expansion)
let results = retriever.retrieve_multi_query(vec![
    "What is ownership?",
    "How does borrowing work?",
]).await?;
```

---

## Rerankers

Improve retrieval quality by reranking initial results:

| Reranker | How It Works |
|----------|-------------|
| **CrossEncoderReranker** | HuggingFace cross-encoder model API |
| **LLMReranker** | Uses an LLM to score relevance |
| **RRFReranker** | Reciprocal Rank Fusion for merging multiple result sets |
| **NoopReranker** | Pass-through (no reranking) |

```rust
use flowgentra_ai::prelude::*;

let reranker = CrossEncoderReranker::new(
    "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "hf_your_key",
).with_top_k(5);

let reranked = reranker.rerank("query", results).await?;
```

---

## Document Loaders

Load documents from various file formats:

```rust
use flowgentra_ai::prelude::*;

// Single file (auto-detects format)
let doc = load_document("path/to/file.pdf")?;

// Entire directory (recursively)
let docs = load_directory("path/to/docs/")?;
```

Supported formats: PDF, plain text, Markdown, JSON, CSV, HTML.

---

## Ingestion Pipeline

End-to-end pipeline: load → split → embed → index.

```rust
use flowgentra_ai::prelude::*;

let pipeline = IngestionPipeline::new(splitter, embeddings, store);
let stats = pipeline.ingest(documents).await?;
println!("Indexed {} chunks from {} docs", stats.chunks, stats.documents);
```

---

## Hybrid Search

Combine semantic vector search with BM25 keyword matching:

```rust
use flowgentra_ai::prelude::*;

let merged = hybrid_merge(semantic_results, keyword_results, 0.7);
// 70% weight to semantic, 30% to keyword
```

---

## RAG Evaluation

Measure retrieval quality with standard metrics:

```rust
use flowgentra_ai::prelude::*;

let results = evaluate(queries, retriever).await?;
println!("Hit rate: {:.2}", hit_rate(&results));
println!("MRR: {:.2}", mrr(&results));
println!("NDCG: {:.2}", mean_ndcg(&results));
```

---

## Best Practices

1. **Choose the right store** -- Pinecone for managed production, Qdrant for self-hosted, InMemory for tests
2. **Tune top_k** -- Start with 5, increase if answers lack context
3. **Set similarity_threshold** -- Filter out low-relevance results (0.7 is a good default)
4. **Chunk wisely** -- Too small = lost context, too large = noise in results
5. **Use text splitters** -- Pick the right splitter for your content type (Markdown, code, HTML)
6. **Rerank for quality** -- A cross-encoder reranker significantly improves precision
7. **Cache embeddings** -- Use `CachedEmbeddings` to avoid recomputing identical texts
8. **Test with InMemoryStore first** -- Switch to a real store when ready for production

---

See [FEATURES.md](../FEATURES.md) for the complete feature list.
