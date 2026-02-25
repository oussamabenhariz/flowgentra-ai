//! # Conditional Routing DSL
//!
//! Provides a type-safe, composable way to define routing conditions
//! instead of string-based conditions.
//!
//! ## Features
//!
//! - **Type-Safe** - Conditions are validated at compile time
//! - **Composable** - Combine conditions with AND, OR, NOT
//! - **Readable** - DSL syntax is intuitive and clear
//! - **Custom Logic** - Support for custom predicates
//! - **Efficient** - Optimized condition evaluation
//!
//! ## Example
//!
//! ```ignore
//! use erenflow_ai::core::routing::{Condition, ComparisonOp};
//!
//! let condition = Condition::and(vec![
//!     Condition::compare("confidence", ComparisonOp::GreaterThan, 0.8),
//!     Condition::compare("attempts", ComparisonOp::LessThan, 3),
//! ]);
//!
//! // Use in edge
//! edge.with_condition(condition);
//! ```

use crate::core::state::State;
use serde_json::Value;

/// Comparison operators for field comparisons
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComparisonOp {
    /// ==
    Equal,
    /// !=
    NotEqual,
    /// <
    LessThan,
    /// <=
    LessOrEqual,
    /// >
    GreaterThan,
    /// >=
    GreaterOrEqual,
}

impl ComparisonOp {
    /// Apply operator to two values
    pub fn apply(&self, left: &Value, right: &Value) -> bool {
        match self {
            ComparisonOp::Equal => left == right,
            ComparisonOp::NotEqual => left != right,
            ComparisonOp::LessThan => Self::compare_numeric(left, right, |l, r| l < r),
            ComparisonOp::LessOrEqual => Self::compare_numeric(left, right, |l, r| l <= r),
            ComparisonOp::GreaterThan => Self::compare_numeric(left, right, |l, r| l > r),
            ComparisonOp::GreaterOrEqual => Self::compare_numeric(left, right, |l, r| l >= r),
        }
    }

    fn compare_numeric<F>(left: &Value, right: &Value, f: F) -> bool
    where
        F: Fn(f64, f64) -> bool,
    {
        match (left.as_f64(), right.as_f64()) {
            (Some(l), Some(r)) => f(l, r),
            _ => false,
        }
    }

    /// Get string representation
    pub fn as_str(&self) -> &'static str {
        match self {
            ComparisonOp::Equal => "==",
            ComparisonOp::NotEqual => "!=",
            ComparisonOp::LessThan => "<",
            ComparisonOp::LessOrEqual => "<=",
            ComparisonOp::GreaterThan => ">",
            ComparisonOp::GreaterOrEqual => ">=",
        }
    }
}

impl std::fmt::Display for ComparisonOp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// Type-safe routing condition
#[derive(Clone)]
pub enum Condition {
    /// Compare a field: field op value
    Compare {
        field: String,
        op: ComparisonOp,
        value: Value,
    },

    /// Field presence check
    FieldExists(String),

    /// Field type check
    FieldType {
        field: String,
        expected_type: FieldTypeCheck,
    },

    /// Logical AND of conditions
    And(Vec<Condition>),

    /// Logical OR of conditions
    Or(Vec<Condition>),

    /// Logical NOT of condition
    Not(Box<Condition>),

    /// Custom predicate function
    Custom {
        name: String,
        predicate: std::sync::Arc<dyn Fn(&State) -> bool + Send + Sync>,
    },
}

/// Field type check
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FieldTypeCheck {
    String,
    Number,
    Boolean,
    Array,
    Object,
    Null,
}

impl FieldTypeCheck {
    /// Check if a value matches this type
    pub fn matches(&self, value: &Value) -> bool {
        match self {
            FieldTypeCheck::String => value.is_string(),
            FieldTypeCheck::Number => value.is_number(),
            FieldTypeCheck::Boolean => value.is_boolean(),
            FieldTypeCheck::Array => value.is_array(),
            FieldTypeCheck::Object => value.is_object(),
            FieldTypeCheck::Null => value.is_null(),
        }
    }
}

impl Condition {
    /// Create a comparison condition
    pub fn compare(field: impl Into<String>, op: ComparisonOp, value: impl Into<Value>) -> Self {
        Condition::Compare {
            field: field.into(),
            op,
            value: value.into(),
        }
    }

    /// Create a field existence check
    pub fn field_exists(field: impl Into<String>) -> Self {
        Condition::FieldExists(field.into())
    }

    /// Create a field type check
    pub fn field_type(field: impl Into<String>, expected_type: FieldTypeCheck) -> Self {
        Condition::FieldType {
            field: field.into(),
            expected_type,
        }
    }

    /// Create AND condition
    pub fn and(conditions: Vec<Condition>) -> Self {
        Condition::And(conditions)
    }

    /// Create OR condition
    pub fn or(conditions: Vec<Condition>) -> Self {
        Condition::Or(conditions)
    }

    /// Create NOT condition
    pub fn not_condition(condition: Condition) -> Self {
        Condition::Not(Box::new(condition))
    }

    /// Create custom predicate condition
    pub fn custom<F>(name: impl Into<String>, predicate: F) -> Self
    where
        F: Fn(&State) -> bool + Send + Sync + 'static,
    {
        Condition::Custom {
            name: name.into(),
            predicate: std::sync::Arc::new(predicate),
        }
    }

    /// Evaluate the condition against a state
    pub fn evaluate(&self, state: &State) -> bool {
        match self {
            Condition::Compare { field, op, value } => match state.get(field) {
                Some(field_value) => op.apply(field_value, value),
                None => false,
            },

            Condition::FieldExists(field) => state.contains_key(field),

            Condition::FieldType {
                field,
                expected_type,
            } => match state.get(field) {
                Some(value) => expected_type.matches(value),
                None => false,
            },

            Condition::And(conditions) => conditions.iter().all(|c| c.evaluate(state)),

            Condition::Or(conditions) => conditions.iter().any(|c| c.evaluate(state)),

            Condition::Not(condition) => !condition.evaluate(state),

            Condition::Custom { predicate, .. } => predicate(state),
        }
    }

    /// Convert to string for display/logging
    pub fn to_string_representation(&self) -> String {
        match self {
            Condition::Compare { field, op, value } => {
                format!("{} {} {}", field, op, value)
            }

            Condition::FieldExists(field) => {
                format!("exists({})", field)
            }

            Condition::FieldType {
                field,
                expected_type,
            } => {
                format!("typeof({}) == {:?}", field, expected_type)
            }

            Condition::And(conditions) => {
                let parts: Vec<_> = conditions
                    .iter()
                    .map(|c| c.to_string_representation())
                    .collect();
                format!("({})", parts.join(" AND "))
            }

            Condition::Or(conditions) => {
                let parts: Vec<_> = conditions
                    .iter()
                    .map(|c| c.to_string_representation())
                    .collect();
                format!("({})", parts.join(" OR "))
            }

            Condition::Not(condition) => {
                format!("NOT({})", condition.to_string_representation())
            }

            Condition::Custom { name, .. } => {
                format!("custom({})", name)
            }
        }
    }

    /// Simplify condition by removing double negations and redundant ORs/ANDs
    pub fn simplify(self) -> Self {
        match self {
            // Simplify double negations
            Condition::Not(inner) => {
                if let Condition::Not(inner_inner) = *inner {
                    inner_inner.simplify()
                } else {
                    Condition::Not(inner)
                }
            }

            Condition::And(conditions) => {
                // Remove redundant nested ANDs
                let mut flat = Vec::new();
                for cond in conditions {
                    if let Condition::And(inner) = cond {
                        flat.extend(inner);
                    } else {
                        flat.push(cond);
                    }
                }
                Condition::And(flat.into_iter().map(|c| c.simplify()).collect())
            }

            Condition::Or(conditions) => {
                // Remove redundant nested ORs
                let mut flat = Vec::new();
                for cond in conditions {
                    if let Condition::Or(inner) = cond {
                        flat.extend(inner);
                    } else {
                        flat.push(cond);
                    }
                }
                Condition::Or(flat.into_iter().map(|c| c.simplify()).collect())
            }

            _ => self,
        }
    }
}

impl std::fmt::Debug for Condition {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.to_string_representation())
    }
}

impl std::fmt::Display for Condition {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.to_string_representation())
    }
}

/// Builder for constructing complex conditions
pub struct ConditionBuilder {
    conditions: Vec<Condition>,
    mode: ConditionMode,
}

#[derive(Debug, Clone, Copy)]
enum ConditionMode {
    And,
    Or,
}

impl ConditionBuilder {
    /// Create a new builder in AND mode
    pub fn and() -> Self {
        Self {
            conditions: Vec::new(),
            mode: ConditionMode::And,
        }
    }

    /// Create a new builder in OR mode
    pub fn or() -> Self {
        Self {
            conditions: Vec::new(),
            mode: ConditionMode::Or,
        }
    }

    /// Add a condition
    pub fn add_condition(self, condition: Condition) -> Self {
        let mut conditions = self.conditions;
        conditions.push(condition);
        Self {
            conditions,
            mode: self.mode,
        }
    }

    /// Add a comparison condition
    pub fn compare(
        self,
        field: impl Into<String>,
        op: ComparisonOp,
        value: impl Into<Value>,
    ) -> Self {
        self.add_condition(Condition::compare(field, op, value))
    }

    /// Add a field existence check
    pub fn field_exists(self, field: impl Into<String>) -> Self {
        self.add_condition(Condition::field_exists(field))
    }

    /// Build the final condition
    pub fn build(self) -> Condition {
        if self.conditions.is_empty() {
            Condition::and(vec![]) // No conditions = always true for AND
        } else if self.conditions.len() == 1 {
            self.conditions.into_iter().next().unwrap()
        } else {
            match self.mode {
                ConditionMode::And => Condition::and(self.conditions),
                ConditionMode::Or => Condition::or(self.conditions),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn comparison_operators() {
        assert!(ComparisonOp::Equal.apply(&json!(5), &json!(5)));
        assert!(!ComparisonOp::Equal.apply(&json!(5), &json!(6)));
        assert!(ComparisonOp::GreaterThan.apply(&json!(6), &json!(5)));
        assert!(ComparisonOp::LessThan.apply(&json!(5), &json!(6)));
    }

    #[test]
    fn condition_compare() {
        let mut state = State::new();
        state.set("confidence", json!(0.9));

        let condition = Condition::compare("confidence", ComparisonOp::GreaterThan, 0.8);
        assert!(condition.evaluate(&state));
    }

    #[test]
    fn condition_field_exists() {
        let mut state = State::new();
        state.set("field", json!("value"));

        let condition = Condition::field_exists("field");
        assert!(condition.evaluate(&state));

        let missing = Condition::field_exists("missing");
        assert!(!missing.evaluate(&state));
    }

    #[test]
    fn condition_and() {
        let mut state = State::new();
        state.set("a", json!(true));
        state.set("b", json!(true));

        let condition = Condition::and(vec![
            Condition::field_exists("a"),
            Condition::field_exists("b"),
        ]);
        assert!(condition.evaluate(&state));

        let condition = Condition::and(vec![
            Condition::field_exists("a"),
            Condition::field_exists("missing"),
        ]);
        assert!(!condition.evaluate(&state));
    }

    #[test]
    fn condition_or() {
        let mut state = State::new();
        state.set("a", json!(true));

        let condition = Condition::or(vec![
            Condition::field_exists("a"),
            Condition::field_exists("missing"),
        ]);
        assert!(condition.evaluate(&state));

        let condition = Condition::or(vec![
            Condition::field_exists("missing1"),
            Condition::field_exists("missing2"),
        ]);
        assert!(!condition.evaluate(&state));
    }

    #[test]
    fn condition_not() {
        let mut state = State::new();
        state.set("a", json!(true));

        let condition = Condition::Not(Box::new(Condition::field_exists("missing")));
        assert!(condition.evaluate(&state));
    }

    #[test]
    fn condition_builder() {
        let condition = ConditionBuilder::and()
            .compare("confidence", ComparisonOp::GreaterThan, 0.8)
            .compare("attempts", ComparisonOp::LessThan, 3)
            .build();

        let mut state = State::new();
        state.set("confidence", json!(0.9));
        state.set("attempts", json!(2));

        assert!(condition.evaluate(&state));
    }

    #[test]
    fn condition_simplify() {
        let condition = Condition::Not(Box::new(Condition::Not(Box::new(
            Condition::field_exists("a"),
        ))));
        let simplified = condition.simplify();

        // Should simplify to just field_exists("a")
        let mut state = State::new();
        state.set("a", json!(true));
        assert!(simplified.evaluate(&state));
    }

    #[test]
    fn field_type_check() {
        let mut state = State::new();
        state.set("name", json!("Alice"));
        state.set("age", json!(30));

        let string_check = Condition::field_type("name", FieldTypeCheck::String);
        assert!(string_check.evaluate(&state));

        let number_check = Condition::field_type("age", FieldTypeCheck::Number);
        assert!(number_check.evaluate(&state));

        let wrong_check = Condition::field_type("name", FieldTypeCheck::Number);
        assert!(!wrong_check.evaluate(&state));
    }
}
