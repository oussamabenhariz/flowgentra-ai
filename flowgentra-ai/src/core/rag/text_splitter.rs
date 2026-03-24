//! Text Splitters — split documents into chunks using various strategies
//!
//! Provides multiple splitting strategies beyond basic character chunking:
//! - **RecursiveCharacterTextSplitter** — tries multiple separators in order
//! - **MarkdownTextSplitter** — respects markdown structure (headers, code blocks)
//! - **CodeTextSplitter** — language-aware splitting for source code
//! - **HTMLTextSplitter** — splits on HTML tags
//! - **TokenTextSplitter** — splits by estimated token count

use std::collections::HashMap;

/// Metadata attached to each chunk
#[derive(Debug, Clone, Default)]
pub struct ChunkMetadata {
    pub source: Option<String>,
    pub chunk_index: usize,
    pub start_char: usize,
    pub end_char: usize,
    pub extra: HashMap<String, String>,
}

/// A text chunk with metadata
#[derive(Debug, Clone)]
pub struct TextChunk {
    pub text: String,
    pub metadata: ChunkMetadata,
}

/// Trait for all text splitters
pub trait TextSplitter: Send + Sync {
    /// Split text into chunks
    fn split_text(&self, text: &str) -> Vec<TextChunk>;

    /// Split text and attach source metadata
    fn split_with_source(&self, text: &str, source: &str) -> Vec<TextChunk> {
        let mut chunks = self.split_text(text);
        for chunk in &mut chunks {
            chunk.metadata.source = Some(source.to_string());
        }
        chunks
    }
}

// =============================================================================
// RecursiveCharacterTextSplitter
// =============================================================================

/// Splits text by trying a hierarchy of separators, falling back to the next
/// separator when chunks are still too large. Similar to LangChain's
/// RecursiveCharacterTextSplitter.
///
/// Default separators: `["\n\n", "\n", ". ", " ", ""]`
pub struct RecursiveCharacterTextSplitter {
    pub chunk_size: usize,
    pub chunk_overlap: usize,
    pub separators: Vec<String>,
}

impl RecursiveCharacterTextSplitter {
    pub fn new(chunk_size: usize, chunk_overlap: usize) -> Self {
        Self {
            chunk_size,
            chunk_overlap,
            separators: vec![
                "\n\n".to_string(),
                "\n".to_string(),
                ". ".to_string(),
                " ".to_string(),
                "".to_string(),
            ],
        }
    }

    pub fn with_separators(mut self, separators: Vec<String>) -> Self {
        self.separators = separators;
        self
    }

    fn split_recursive(&self, text: &str, separators: &[String]) -> Vec<String> {
        if text.len() <= self.chunk_size {
            return vec![text.to_string()];
        }

        if separators.is_empty() {
            // Last resort: hard split by character
            return self.hard_split(text);
        }

        let sep = &separators[0];
        let remaining_seps = &separators[1..];

        if sep.is_empty() {
            return self.hard_split(text);
        }

        let parts: Vec<&str> = text.split(sep.as_str()).collect();

        let mut chunks = Vec::new();
        let mut current = String::new();

        for part in parts {
            let candidate = if current.is_empty() {
                part.to_string()
            } else {
                format!("{}{}{}", current, sep, part)
            };

            if candidate.len() > self.chunk_size && !current.is_empty() {
                // Current chunk is full, save it and try to split if needed
                if current.len() > self.chunk_size {
                    chunks.extend(self.split_recursive(&current, remaining_seps));
                } else {
                    chunks.push(current.clone());
                }

                // Start new chunk with overlap
                let overlap_start = current.len().saturating_sub(self.chunk_overlap);
                current = format!("{}{}{}", &current[overlap_start..], sep, part);
                if current.len() > self.chunk_size {
                    current = part.to_string();
                }
            } else {
                current = candidate;
            }
        }

        if !current.is_empty() {
            if current.len() > self.chunk_size {
                chunks.extend(self.split_recursive(&current, remaining_seps));
            } else {
                chunks.push(current);
            }
        }

        chunks
    }

    fn hard_split(&self, text: &str) -> Vec<String> {
        let chars: Vec<char> = text.chars().collect();
        let step = self.chunk_size.saturating_sub(self.chunk_overlap).max(1);
        let mut chunks = Vec::new();
        let mut start = 0;

        while start < chars.len() {
            let end = (start + self.chunk_size).min(chars.len());
            let chunk: String = chars[start..end].iter().collect();
            if !chunk.trim().is_empty() {
                chunks.push(chunk);
            }
            if end >= chars.len() {
                break;
            }
            start += step;
        }

        chunks
    }
}

impl TextSplitter for RecursiveCharacterTextSplitter {
    fn split_text(&self, text: &str) -> Vec<TextChunk> {
        let raw_chunks = self.split_recursive(text, &self.separators);
        let mut offset = 0;

        raw_chunks
            .into_iter()
            .enumerate()
            .filter(|(_, c)| !c.trim().is_empty())
            .map(|(i, chunk)| {
                let start = text[offset..]
                    .find(&chunk)
                    .map(|p| p + offset)
                    .unwrap_or(offset);
                let end = start + chunk.len();
                offset = start + 1; // advance past to avoid re-matching
                TextChunk {
                    text: chunk,
                    metadata: ChunkMetadata {
                        chunk_index: i,
                        start_char: start,
                        end_char: end,
                        ..Default::default()
                    },
                }
            })
            .collect()
    }
}

// =============================================================================
// MarkdownTextSplitter
// =============================================================================

/// Splits markdown documents respecting structure: headers, code blocks, lists.
/// Uses headers as natural split points, keeping code blocks intact.
pub struct MarkdownTextSplitter {
    pub chunk_size: usize,
    pub chunk_overlap: usize,
}

impl MarkdownTextSplitter {
    pub fn new(chunk_size: usize, chunk_overlap: usize) -> Self {
        Self {
            chunk_size,
            chunk_overlap,
        }
    }
}

impl TextSplitter for MarkdownTextSplitter {
    fn split_text(&self, text: &str) -> Vec<TextChunk> {
        // Split on markdown headers (## and above), keeping code blocks intact
        let separators = vec![
            "\n# ".to_string(),
            "\n## ".to_string(),
            "\n### ".to_string(),
            "\n```".to_string(),
            "\n\n".to_string(),
            "\n".to_string(),
            " ".to_string(),
        ];

        let recursive = RecursiveCharacterTextSplitter {
            chunk_size: self.chunk_size,
            chunk_overlap: self.chunk_overlap,
            separators,
        };

        recursive.split_text(text)
    }
}

// =============================================================================
// CodeTextSplitter
// =============================================================================

/// Language hint for code splitting
#[derive(Debug, Clone, PartialEq)]
pub enum Language {
    Rust,
    Python,
    JavaScript,
    TypeScript,
    Go,
    Java,
    CSharp,
    Cpp,
    Ruby,
    Generic,
}

impl Language {
    /// Get natural code separators for this language
    pub fn separators(&self) -> Vec<String> {
        match self {
            Language::Rust => vec![
                "\nfn ".to_string(),
                "\npub fn ".to_string(),
                "\nimpl ".to_string(),
                "\npub struct ".to_string(),
                "\npub enum ".to_string(),
                "\nmod ".to_string(),
                "\n\n".to_string(),
                "\n".to_string(),
                " ".to_string(),
            ],
            Language::Python => vec![
                "\nclass ".to_string(),
                "\ndef ".to_string(),
                "\n    def ".to_string(),
                "\n\n".to_string(),
                "\n".to_string(),
                " ".to_string(),
            ],
            Language::JavaScript | Language::TypeScript => vec![
                "\nfunction ".to_string(),
                "\nconst ".to_string(),
                "\nclass ".to_string(),
                "\nexport ".to_string(),
                "\n\n".to_string(),
                "\n".to_string(),
                " ".to_string(),
            ],
            Language::Go => vec![
                "\nfunc ".to_string(),
                "\ntype ".to_string(),
                "\n\n".to_string(),
                "\n".to_string(),
                " ".to_string(),
            ],
            Language::Java | Language::CSharp => vec![
                "\npublic class ".to_string(),
                "\npublic ".to_string(),
                "\nprivate ".to_string(),
                "\n\n".to_string(),
                "\n".to_string(),
                " ".to_string(),
            ],
            Language::Cpp => vec![
                "\nclass ".to_string(),
                "\nvoid ".to_string(),
                "\nint ".to_string(),
                "\nnamespace ".to_string(),
                "\n\n".to_string(),
                "\n".to_string(),
                " ".to_string(),
            ],
            Language::Ruby => vec![
                "\nclass ".to_string(),
                "\ndef ".to_string(),
                "\nmodule ".to_string(),
                "\n\n".to_string(),
                "\n".to_string(),
                " ".to_string(),
            ],
            Language::Generic => vec!["\n\n".to_string(), "\n".to_string(), " ".to_string()],
        }
    }

    /// Detect language from file extension
    pub fn from_extension(ext: &str) -> Self {
        match ext.to_lowercase().as_str() {
            "rs" => Language::Rust,
            "py" => Language::Python,
            "js" | "jsx" | "mjs" => Language::JavaScript,
            "ts" | "tsx" => Language::TypeScript,
            "go" => Language::Go,
            "java" => Language::Java,
            "cs" => Language::CSharp,
            "cpp" | "cc" | "cxx" | "c" | "h" | "hpp" => Language::Cpp,
            "rb" => Language::Ruby,
            _ => Language::Generic,
        }
    }
}

/// Splits source code respecting language-specific boundaries (functions, classes, etc.)
pub struct CodeTextSplitter {
    pub chunk_size: usize,
    pub chunk_overlap: usize,
    pub language: Language,
}

impl CodeTextSplitter {
    pub fn new(chunk_size: usize, chunk_overlap: usize, language: Language) -> Self {
        Self {
            chunk_size,
            chunk_overlap,
            language,
        }
    }
}

impl TextSplitter for CodeTextSplitter {
    fn split_text(&self, text: &str) -> Vec<TextChunk> {
        let recursive = RecursiveCharacterTextSplitter {
            chunk_size: self.chunk_size,
            chunk_overlap: self.chunk_overlap,
            separators: self.language.separators(),
        };
        recursive.split_text(text)
    }
}

// =============================================================================
// HTMLTextSplitter
// =============================================================================

/// Splits HTML content on structural tags while preserving content integrity.
pub struct HTMLTextSplitter {
    pub chunk_size: usize,
    pub chunk_overlap: usize,
}

impl HTMLTextSplitter {
    pub fn new(chunk_size: usize, chunk_overlap: usize) -> Self {
        Self {
            chunk_size,
            chunk_overlap,
        }
    }
}

impl TextSplitter for HTMLTextSplitter {
    fn split_text(&self, text: &str) -> Vec<TextChunk> {
        let separators = vec![
            "</div>".to_string(),
            "</section>".to_string(),
            "</article>".to_string(),
            "</p>".to_string(),
            "</h1>".to_string(),
            "</h2>".to_string(),
            "</h3>".to_string(),
            "</li>".to_string(),
            "<br>".to_string(),
            "\n\n".to_string(),
            "\n".to_string(),
            " ".to_string(),
        ];

        let recursive = RecursiveCharacterTextSplitter {
            chunk_size: self.chunk_size,
            chunk_overlap: self.chunk_overlap,
            separators,
        };
        recursive.split_text(text)
    }
}

// =============================================================================
// TokenTextSplitter
// =============================================================================

/// Splits text by estimated token count (~4 chars per token).
/// Better for LLM context window management than character-based splitting.
pub struct TokenTextSplitter {
    pub max_tokens: usize,
    pub overlap_tokens: usize,
}

impl TokenTextSplitter {
    pub fn new(max_tokens: usize, overlap_tokens: usize) -> Self {
        Self {
            max_tokens,
            overlap_tokens,
        }
    }
}

impl TextSplitter for TokenTextSplitter {
    fn split_text(&self, text: &str) -> Vec<TextChunk> {
        let chars_per_token: usize = 4;
        let recursive = RecursiveCharacterTextSplitter::new(
            self.max_tokens * chars_per_token,
            self.overlap_tokens * chars_per_token,
        );
        recursive.split_text(text)
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_recursive_splitter_small_text() {
        let splitter = RecursiveCharacterTextSplitter::new(1000, 100);
        let chunks = splitter.split_text("Hello world.");
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].text, "Hello world.");
    }

    #[test]
    fn test_recursive_splitter_paragraph_boundary() {
        let text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph.";
        let splitter = RecursiveCharacterTextSplitter::new(30, 0);
        let chunks = splitter.split_text(text);
        assert!(chunks.len() >= 2);
        assert!(chunks[0].text.contains("First"));
    }

    #[test]
    fn test_recursive_splitter_large_text() {
        let text = "A ".repeat(500); // 1000 chars
        let splitter = RecursiveCharacterTextSplitter::new(200, 50);
        let chunks = splitter.split_text(&text);
        assert!(chunks.len() >= 4);
    }

    #[test]
    fn test_markdown_splitter() {
        let md =
            "# Title\n\nIntro text.\n\n## Section 1\n\nContent 1.\n\n## Section 2\n\nContent 2.";
        let splitter = MarkdownTextSplitter::new(40, 0);
        let chunks = splitter.split_text(md);
        assert!(chunks.len() >= 2);
    }

    #[test]
    fn test_code_splitter_rust() {
        let code = "fn foo() {\n    println!(\"foo\");\n}\n\nfn bar() {\n    println!(\"bar\");\n}\n\nfn baz() {\n    println!(\"baz\");\n}";
        let splitter = CodeTextSplitter::new(40, 0, Language::Rust);
        let chunks = splitter.split_text(code);
        assert!(chunks.len() >= 2);
    }

    #[test]
    fn test_html_splitter() {
        let html =
            "<div><p>Paragraph 1.</p><p>Paragraph 2.</p></div><div><p>Paragraph 3.</p></div>";
        let splitter = HTMLTextSplitter::new(40, 0);
        let chunks = splitter.split_text(html);
        assert!(chunks.len() >= 1);
    }

    #[test]
    fn test_token_splitter() {
        let text = "word ".repeat(200); // ~200 tokens
        let splitter = TokenTextSplitter::new(50, 10);
        let chunks = splitter.split_text(&text);
        assert!(chunks.len() >= 3);
    }

    #[test]
    fn test_language_detection() {
        assert_eq!(Language::from_extension("rs"), Language::Rust);
        assert_eq!(Language::from_extension("py"), Language::Python);
        assert_eq!(Language::from_extension("ts"), Language::TypeScript);
        assert_eq!(Language::from_extension("go"), Language::Go);
        assert_eq!(Language::from_extension("unknown"), Language::Generic);
    }

    #[test]
    fn test_split_with_source() {
        let splitter = RecursiveCharacterTextSplitter::new(1000, 0);
        let chunks = splitter.split_with_source("Hello world.", "test.txt");
        assert_eq!(chunks[0].metadata.source, Some("test.txt".to_string()));
    }

    #[test]
    fn test_chunk_metadata() {
        let text = "First part.\n\nSecond part.";
        let splitter = RecursiveCharacterTextSplitter::new(15, 0);
        let chunks = splitter.split_text(text);
        assert!(chunks.len() >= 2);
        assert_eq!(chunks[0].metadata.chunk_index, 0);
    }
}
