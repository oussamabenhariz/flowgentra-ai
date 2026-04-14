//! Type-safe metadata filter expressions for vector store queries.
//!
//! Replace the old `HashMap<String, Value>` approach with a structured enum that
//! each backend translates into its own native query format.
//!
//! # Example
//! ```rust
//! use flowgentra_ai::core::rag::{FilterExpr, MetadataFilter};
//! use serde_json::json;
//!
//! // Simple equality
//! let f: MetadataFilter = FilterExpr::eq("source", json!("pdf"));
//!
//! // Compound: source == "pdf" AND page > 5
//! let f = FilterExpr::and(vec![
//!     FilterExpr::eq("source", json!("pdf")),
//!     FilterExpr::gt("page", json!(5)),
//! ]);
//!
//! // IN list
//! let f = FilterExpr::in_values("category", vec![json!("news"), json!("blog")]);
//! ```

use serde_json::Value;

/// A typed filter expression for metadata-scoped vector searches.
///
/// Each backend converts this into its native filter format:
/// - Pinecone → JSON metadata filter
/// - Qdrant   → `must`/`should` filter objects
/// - Chroma   → `$eq`/`$and` where clause
/// - Milvus   → scalar filter string
/// - Weaviate → GraphQL `where` clause
/// - InMemory → evaluated directly in Rust
#[derive(Debug, Clone)]
pub enum FilterExpr {
    /// `field == value`
    Eq(String, Value),
    /// `field != value`
    Ne(String, Value),
    /// `field > value`
    Gt(String, Value),
    /// `field < value`
    Lt(String, Value),
    /// `field >= value`
    Gte(String, Value),
    /// `field <= value`
    Lte(String, Value),
    /// `field IN [values]`
    In(String, Vec<Value>),
    /// All sub-expressions must match (logical AND)
    And(Vec<FilterExpr>),
    /// Any sub-expression must match (logical OR)
    Or(Vec<FilterExpr>),
}

impl FilterExpr {
    /// `field == value`
    pub fn eq(key: impl Into<String>, value: Value) -> Self {
        Self::Eq(key.into(), value)
    }

    /// `field != value`
    pub fn ne(key: impl Into<String>, value: Value) -> Self {
        Self::Ne(key.into(), value)
    }

    /// `field > value`
    pub fn gt(key: impl Into<String>, value: Value) -> Self {
        Self::Gt(key.into(), value)
    }

    /// `field < value`
    pub fn lt(key: impl Into<String>, value: Value) -> Self {
        Self::Lt(key.into(), value)
    }

    /// `field >= value`
    pub fn gte(key: impl Into<String>, value: Value) -> Self {
        Self::Gte(key.into(), value)
    }

    /// `field <= value`
    pub fn lte(key: impl Into<String>, value: Value) -> Self {
        Self::Lte(key.into(), value)
    }

    /// `field IN [values]`
    pub fn in_values(key: impl Into<String>, values: Vec<Value>) -> Self {
        Self::In(key.into(), values)
    }

    /// All sub-expressions must match (AND)
    pub fn and(exprs: Vec<FilterExpr>) -> Self {
        Self::And(exprs)
    }

    /// Any sub-expression must match (OR)
    pub fn or(exprs: Vec<FilterExpr>) -> Self {
        Self::Or(exprs)
    }
}

/// The metadata filter type used across all vector store backends.
///
/// Constructed via [`FilterExpr`] helpers.
pub type MetadataFilter = FilterExpr;
