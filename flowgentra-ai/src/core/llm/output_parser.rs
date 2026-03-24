//! Output Parsers — parse and validate LLM text output into structured data
//!
//! Provides parsers for common output formats:
//! - **JsonOutputParser** — parse JSON from LLM output
//! - **ListOutputParser** — parse comma/newline-separated lists
//! - **StructuredOutputParser** — parse key-value fields with instructions

use serde_json::Value;

/// Error type for output parsing
#[derive(Debug, thiserror::Error)]
pub enum ParseError {
    #[error("Failed to parse output: {0}")]
    ParseFailed(String),

    #[error("JSON parse error: {0}")]
    JsonError(String),

    #[error("Missing required field: {0}")]
    MissingField(String),

    #[error("Invalid format: expected {expected}, got: {actual}")]
    InvalidFormat { expected: String, actual: String },
}

/// Trait for output parsers
pub trait OutputParser: Send + Sync {
    type Output;

    /// Parse the LLM output string into structured data.
    fn parse(&self, text: &str) -> Result<Self::Output, ParseError>;

    /// Get format instructions to include in the prompt.
    fn format_instructions(&self) -> String;
}

// =============================================================================
// JsonOutputParser
// =============================================================================

/// Parses JSON from LLM output, handling common issues like markdown fences.
///
/// # Example
/// ```
/// use flowgentra_ai::core::llm::output_parser::JsonOutputParser;
/// use flowgentra_ai::core::llm::output_parser::OutputParser;
///
/// let parser = JsonOutputParser::new();
/// let result = parser.parse(r#"```json
/// {"name": "Alice", "age": 30}
/// ```"#).unwrap();
/// assert_eq!(result["name"], "Alice");
/// ```
pub struct JsonOutputParser {
    schema_hint: Option<String>,
}

impl JsonOutputParser {
    pub fn new() -> Self {
        Self { schema_hint: None }
    }

    /// Provide a schema hint to include in format instructions.
    pub fn with_schema(mut self, schema: impl Into<String>) -> Self {
        self.schema_hint = Some(schema.into());
        self
    }

    /// Extract JSON from text that may contain markdown fences or other text.
    fn extract_json(text: &str) -> &str {
        let trimmed = text.trim();

        // Try to extract from ```json ... ``` blocks
        if let Some(start) = trimmed.find("```json") {
            let content_start = start + 7;
            if let Some(end) = trimmed[content_start..].find("```") {
                return trimmed[content_start..content_start + end].trim();
            }
        }

        // Try to extract from ``` ... ``` blocks
        if let Some(start) = trimmed.find("```") {
            let content_start = start + 3;
            // Skip to next line if there's a language tag
            let content_start = trimmed[content_start..]
                .find('\n')
                .map(|p| content_start + p + 1)
                .unwrap_or(content_start);
            if let Some(end) = trimmed[content_start..].find("```") {
                return trimmed[content_start..content_start + end].trim();
            }
        }

        // Try to find JSON object or array directly
        if let Some(start) = trimmed.find('{') {
            if let Some(end) = trimmed.rfind('}') {
                return &trimmed[start..=end];
            }
        }

        if let Some(start) = trimmed.find('[') {
            if let Some(end) = trimmed.rfind(']') {
                return &trimmed[start..=end];
            }
        }

        trimmed
    }
}

impl Default for JsonOutputParser {
    fn default() -> Self {
        Self::new()
    }
}

impl OutputParser for JsonOutputParser {
    type Output = Value;

    fn parse(&self, text: &str) -> Result<Value, ParseError> {
        let json_str = Self::extract_json(text);
        serde_json::from_str(json_str).map_err(|e| ParseError::JsonError(e.to_string()))
    }

    fn format_instructions(&self) -> String {
        let mut instructions =
            "Respond with valid JSON only. Do not include markdown fences or any other text."
                .to_string();
        if let Some(schema) = &self.schema_hint {
            instructions.push_str(&format!("\n\nExpected schema:\n{}", schema));
        }
        instructions
    }
}

// =============================================================================
// ListOutputParser
// =============================================================================

/// Parses comma-separated or newline-separated lists from LLM output.
///
/// # Example
/// ```
/// use flowgentra_ai::core::llm::output_parser::{ListOutputParser, OutputParser};
///
/// let parser = ListOutputParser::comma_separated();
/// let result = parser.parse("apple, banana, cherry").unwrap();
/// assert_eq!(result, vec!["apple", "banana", "cherry"]);
/// ```
pub struct ListOutputParser {
    separator: ListSeparator,
}

/// How to split the list
#[derive(Debug, Clone)]
pub enum ListSeparator {
    Comma,
    Newline,
    Numbered,
    Custom(String),
}

impl ListOutputParser {
    pub fn comma_separated() -> Self {
        Self {
            separator: ListSeparator::Comma,
        }
    }

    pub fn newline_separated() -> Self {
        Self {
            separator: ListSeparator::Newline,
        }
    }

    /// Parse numbered lists like "1. item\n2. item"
    pub fn numbered() -> Self {
        Self {
            separator: ListSeparator::Numbered,
        }
    }

    pub fn custom(separator: impl Into<String>) -> Self {
        Self {
            separator: ListSeparator::Custom(separator.into()),
        }
    }
}

impl OutputParser for ListOutputParser {
    type Output = Vec<String>;

    fn parse(&self, text: &str) -> Result<Vec<String>, ParseError> {
        let items: Vec<String> = match &self.separator {
            ListSeparator::Comma => text
                .split(',')
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect(),
            ListSeparator::Newline => text
                .lines()
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect(),
            ListSeparator::Numbered => text
                .lines()
                .map(|line| {
                    let trimmed = line.trim();
                    // Strip numbering like "1. ", "2) ", "- "
                    let stripped = trimmed
                        .trim_start_matches(|c: char| {
                            c.is_ascii_digit() || c == '.' || c == ')' || c == '-'
                        })
                        .trim();
                    stripped.to_string()
                })
                .filter(|s| !s.is_empty())
                .collect(),
            ListSeparator::Custom(sep) => text
                .split(sep.as_str())
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect(),
        };

        if items.is_empty() {
            return Err(ParseError::ParseFailed("Empty list".to_string()));
        }

        Ok(items)
    }

    fn format_instructions(&self) -> String {
        match &self.separator {
            ListSeparator::Comma => {
                "Respond with a comma-separated list. Example: item1, item2, item3".to_string()
            }
            ListSeparator::Newline => "Respond with one item per line.".to_string(),
            ListSeparator::Numbered => {
                "Respond with a numbered list. Example:\n1. First item\n2. Second item".to_string()
            }
            ListSeparator::Custom(sep) => {
                format!("Respond with items separated by '{}'.", sep)
            }
        }
    }
}

// =============================================================================
// StructuredOutputParser
// =============================================================================

/// A field definition for structured output.
#[derive(Debug, Clone)]
pub struct FieldSpec {
    pub name: String,
    pub description: String,
    pub required: bool,
}

impl FieldSpec {
    pub fn required(name: impl Into<String>, description: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            required: true,
        }
    }

    pub fn optional(name: impl Into<String>, description: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            required: false,
        }
    }
}

/// Parses structured key-value output into a JSON object.
///
/// Instructs the LLM to respond in JSON with specific fields.
///
/// # Example
/// ```
/// use flowgentra_ai::core::llm::output_parser::{StructuredOutputParser, FieldSpec, OutputParser};
///
/// let parser = StructuredOutputParser::new(vec![
///     FieldSpec::required("answer", "The answer to the question"),
///     FieldSpec::required("confidence", "Confidence score 0-1"),
/// ]);
///
/// let result = parser.parse(r#"{"answer": "42", "confidence": "0.95"}"#).unwrap();
/// assert_eq!(result["answer"], "42");
/// ```
pub struct StructuredOutputParser {
    fields: Vec<FieldSpec>,
}

impl StructuredOutputParser {
    pub fn new(fields: Vec<FieldSpec>) -> Self {
        Self { fields }
    }
}

impl OutputParser for StructuredOutputParser {
    type Output = Value;

    fn parse(&self, text: &str) -> Result<Value, ParseError> {
        let json_str = JsonOutputParser::extract_json(text);
        let value: Value =
            serde_json::from_str(json_str).map_err(|e| ParseError::JsonError(e.to_string()))?;

        // Validate required fields
        if let Some(obj) = value.as_object() {
            for field in &self.fields {
                if field.required && !obj.contains_key(&field.name) {
                    return Err(ParseError::MissingField(field.name.clone()));
                }
            }
        } else {
            return Err(ParseError::InvalidFormat {
                expected: "JSON object".to_string(),
                actual: format!("{}", value),
            });
        }

        Ok(value)
    }

    fn format_instructions(&self) -> String {
        let mut schema_lines = Vec::new();
        for field in &self.fields {
            let req = if field.required {
                "required"
            } else {
                "optional"
            };
            schema_lines.push(format!(
                "  \"{}\": \"{}\" ({})",
                field.name, field.description, req
            ));
        }

        format!(
            "Respond with a JSON object containing these fields:\n{{\n{}\n}}",
            schema_lines.join(",\n")
        )
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_json_parser_basic() {
        let parser = JsonOutputParser::new();
        let result = parser.parse(r#"{"key": "value"}"#).unwrap();
        assert_eq!(result["key"], "value");
    }

    #[test]
    fn test_json_parser_with_markdown() {
        let parser = JsonOutputParser::new();
        let input = "Here is the result:\n```json\n{\"answer\": 42}\n```";
        let result = parser.parse(input).unwrap();
        assert_eq!(result["answer"], 42);
    }

    #[test]
    fn test_json_parser_with_surrounding_text() {
        let parser = JsonOutputParser::new();
        let input = "The answer is: {\"result\": true} as expected.";
        let result = parser.parse(input).unwrap();
        assert_eq!(result["result"], true);
    }

    #[test]
    fn test_json_parser_array() {
        let parser = JsonOutputParser::new();
        let result = parser.parse("[1, 2, 3]").unwrap();
        assert_eq!(result.as_array().unwrap().len(), 3);
    }

    #[test]
    fn test_json_parser_format_instructions() {
        let parser = JsonOutputParser::new().with_schema("{\"name\": \"string\"}");
        let instructions = parser.format_instructions();
        assert!(instructions.contains("valid JSON"));
        assert!(instructions.contains("name"));
    }

    #[test]
    fn test_list_parser_comma() {
        let parser = ListOutputParser::comma_separated();
        let result = parser.parse("apple, banana, cherry").unwrap();
        assert_eq!(result, vec!["apple", "banana", "cherry"]);
    }

    #[test]
    fn test_list_parser_newline() {
        let parser = ListOutputParser::newline_separated();
        let result = parser.parse("apple\nbanana\ncherry").unwrap();
        assert_eq!(result, vec!["apple", "banana", "cherry"]);
    }

    #[test]
    fn test_list_parser_numbered() {
        let parser = ListOutputParser::numbered();
        let result = parser.parse("1. Apple\n2. Banana\n3. Cherry").unwrap();
        assert_eq!(result, vec!["Apple", "Banana", "Cherry"]);
    }

    #[test]
    fn test_list_parser_empty() {
        let parser = ListOutputParser::comma_separated();
        assert!(parser.parse("").is_err());
    }

    #[test]
    fn test_structured_parser() {
        let parser = StructuredOutputParser::new(vec![
            FieldSpec::required("answer", "The answer"),
            FieldSpec::optional("notes", "Additional notes"),
        ]);

        let result = parser
            .parse(r#"{"answer": "42", "notes": "good question"}"#)
            .unwrap();
        assert_eq!(result["answer"], "42");
    }

    #[test]
    fn test_structured_parser_missing_required() {
        let parser = StructuredOutputParser::new(vec![FieldSpec::required("answer", "The answer")]);

        let result = parser.parse(r#"{"other": "value"}"#);
        assert!(result.is_err());
    }

    #[test]
    fn test_structured_format_instructions() {
        let parser = StructuredOutputParser::new(vec![
            FieldSpec::required("name", "Person name"),
            FieldSpec::optional("age", "Person age"),
        ]);
        let instructions = parser.format_instructions();
        assert!(instructions.contains("name"));
        assert!(instructions.contains("required"));
        assert!(instructions.contains("optional"));
    }
}
