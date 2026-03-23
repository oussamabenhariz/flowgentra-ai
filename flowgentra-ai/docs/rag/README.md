# RAG (Retrieval-Augmented Generation) Guide

Let your agent search through your documents to find relevant context and give better answers.

## What RAG Does

With RAG enabled, your agent can:
1. Upload documents to a vector store
2. Search for relevant documents based on queries
3. Include retrieved documents in prompts for better answers

## Setup

### Step 1: Choose a Vector Store

#### Pinecone (Cloud)

```yaml
rag:
  enabled: true
  vector_store:
    type: pinecone
    index_name: "my-documents"
    environment: "us-west-2-aws"
    api_key: ${PINECONE_API_KEY}
  embedding_model: openai
  embedding_dimension: 1536
  top_k: 5
```

#### Weaviate (Self-hosted)

```yaml
rag:
  enabled: true
  vector_store:
    type: weaviate
    url: "http://localhost:8080"
    api_key: ${WEAVIATE_API_KEY}
  embedding_model: openai
  embedding_dimension: 1536
```

#### Chroma (Local)

```yaml
rag:
  enabled: true
  vector_store:
    type: chroma
    collection_name: "documents"
    host: localhost
    port: 8000
  embedding_model: openai
```

### Step 2: Configure Handler to Use RAG

```yaml
graph:
  nodes:
    - name: retrieve_docs
      handler: handlers::retrieve_docs
      uses_rag: true  # Enable RAG for this node
      timeout: 15
```

### Step 3: Use RAG in Code

```rust
pub async fn retrieve_docs(mut state: State) -> Result<State> {
    let query = state.get_str("user_query")?;
    
    // Retrieve from vector store
    let docs = rag_client.retrieve(query, 5).await?;
    
    // Process documents
    let context = docs.iter()
        .map(|d| d.content.clone())
        .collect::<Vec<_>>()
        .join("\n---\n");
    
    state.set("context", json!(context));
    Ok(state)
}
```

## Configuration Options

### Retrieval Strategy
- `semantic` - Similarity-based search
- `bm25` - Keyword-based search
- `hybrid` - Combination of both

### top_k
Number of documents to retrieve (default: 5)

### similarity_threshold
Minimum similarity score to include (0.0 - 1.0)

### Chunking
```yaml
chunk_size: 1024          # Characters per chunk
chunk_overlap: 200        # Overlap between chunks
separator: "\n\n"         # Chunk boundary
```

## Embedding Models

- `openai` - OpenAI embeddings (1536 dimensions)
- `huggingface` - HuggingFace models

## Use Cases

- **Company Knowledge Base** - Search internal documentation
- **FAQ System** - Find relevant answers
- **Research** - Retrieve relevant papers
- **Customer Support** - Find solution documents

## Example: Knowledge Base Search

```yaml
rag:
  enabled: true
  vector_store:
    type: pinecone
    index_name: "kb"
    api_key: ${PINECONE_API_KEY}
  embedding_model: openai
  top_k: 3
  similarity_threshold: 0.7

graph:
  nodes:
    - name: search_kb
      handler: handlers::search_knowledge_base
      uses_rag: true
    - name: answer
      handler: handlers::answer
```

---

See [configuration/CONFIG_GUIDE.md](../configuration/CONFIG_GUIDE.md) for complete reference.
