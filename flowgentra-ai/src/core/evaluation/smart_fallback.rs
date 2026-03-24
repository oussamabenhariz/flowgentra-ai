//! # Smart Fallback System
//!
//! Progressive fallback content based on retry attempts.
//! Provides increasingly degraded but still useful responses.

use serde::{Deserialize, Serialize};

/// Fallback level based on retry attempts
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum FallbackLevel {
    /// Level 0: First attempt, generate normal content
    Initial,
    /// Level 1: After 1st retry, use shorter form
    Degraded,
    /// Level 2: After 2nd retry, use minimal form
    Minimal,
    /// Level 3+: Basic template only
    Template,
}

impl FallbackLevel {
    /// Get level from retry count
    pub fn from_retries(retries: u32) -> Self {
        match retries {
            0 => FallbackLevel::Initial,
            1 => FallbackLevel::Degraded,
            2 => FallbackLevel::Minimal,
            _ => FallbackLevel::Template,
        }
    }
}

/// Smart content fallback generator
pub struct SmartFallback;

impl SmartFallback {
    /// Generate fallback content for content generation
    ///
    /// Provides progressively simpler responses based on retry level
    ///
    /// # Example
    /// ```ignore
    /// let retries = state.get_int("__retry_count__generate_content", 0) as u32;
    /// let level = FallbackLevel::from_retries(retries);
    /// let fallback = SmartFallback::generate_content_fallback(
    ///     "Rust programming",
    ///     level,
    ///     None
    /// );
    /// ```
    pub fn generate_content_fallback(
        topic: &str,
        level: FallbackLevel,
        previous: Option<&str>,
    ) -> String {
        match level {
            FallbackLevel::Initial => {
                // Should not reach here on first try, but provide safe default
                format!(
                    "Unable to generate detailed content about {} at this time.",
                    topic
                )
            }
            FallbackLevel::Degraded => {
                // After 1st retry: shorter, simpler response
                format!(
                    "## {}\n\n{} is an important topic. It encompasses various aspects and concepts that are worth exploring. Please try again for more detailed information.",
                    topic, topic
                )
            }
            FallbackLevel::Minimal => {
                // After 2nd retry: just key facts
                if let Some(prev) = previous {
                    format!(
                        "## {}\n\n### Background\nPrevious attempt:\n{}\n\n### Note\nUnable to generate additional details. Please refer to the previous attempt above.",
                        topic, prev
                    )
                } else {
                    format!(
                        "## {}\n\n{} is a notable subject with multiple dimensions. Detailed information is currently unavailable.",
                        topic, topic
                    )
                }
            }
            FallbackLevel::Template => {
                // Final fallback: basic structure only
                format!(
                    "# {}\n\nContent generation attempted but unable to complete.\n\nPlease review previous attempts or provide additional context for better results.",
                    topic
                )
            }
        }
    }

    /// Generate fallback content for refinement
    pub fn refine_content_fallback(content: &str, level: FallbackLevel) -> String {
        match level {
            FallbackLevel::Initial => {
                format!("{}\n\n[Unable to refine at this time]", content)
            }
            FallbackLevel::Degraded => {
                // Return lightly formatted version
                format!(
                    "{}\n\n---\n*Note: Refinement service temporarily unavailable. Using original content.*",
                    content
                )
            }
            FallbackLevel::Minimal => {
                // Return with minimum formatting
                content.to_string()
            }
            FallbackLevel::Template => {
                // Return first 500 chars or original
                if content.len() > 500 {
                    format!("{}...\n\n[Content truncated]", &content[..500])
                } else {
                    content.to_string()
                }
            }
        }
    }

    /// Generate retry message for LLM
    pub fn retry_message(level: FallbackLevel, previous_feedback: Option<&str>) -> String {
        match level {
            FallbackLevel::Degraded => {
                "Your previous attempt was not satisfactory. Please try again, focusing on clarity and accuracy. Consider the following feedback if provided.".to_string()
            }
            FallbackLevel::Minimal => {
                if let Some(feedback) = previous_feedback {
                    format!(
                        "Second retry needed. Critical issues from evaluation:\n{}\n\nPlease address these specific issues.",
                        feedback
                    )
                } else {
                    "Second retry needed. Please significantly improve the quality of your response.".to_string()
                }
            }
            FallbackLevel::Template => {
                "Unable to generate satisfactory content after multiple attempts. Returning template response.".to_string()
            }
            FallbackLevel::Initial => String::new(),
        }
    }

    /// Calculate if should give up on retries and use fallback
    pub fn should_fallback(retry_count: u32, max_retries: u32) -> bool {
        retry_count > max_retries
    }

    /// Log fallback usage
    pub fn log_fallback(node_name: &str, level: FallbackLevel, reason: &str) {
        use tracing::warn;
        warn!(
            "[FALLBACK] Node: {}, Level: {:?}, Reason: {}",
            node_name, level, reason
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fallback_level_from_retries() {
        assert!(matches!(
            FallbackLevel::from_retries(0),
            FallbackLevel::Initial
        ));
        assert!(matches!(
            FallbackLevel::from_retries(1),
            FallbackLevel::Degraded
        ));
        assert!(matches!(
            FallbackLevel::from_retries(2),
            FallbackLevel::Minimal
        ));
        assert!(matches!(
            FallbackLevel::from_retries(3),
            FallbackLevel::Template
        ));
    }

    #[test]
    fn test_generate_content_fallback() {
        let minimal =
            SmartFallback::generate_content_fallback("Rust", FallbackLevel::Minimal, None);
        assert!(minimal.contains("Rust"));
        assert!(minimal.contains("##"));
    }

    #[test]
    fn test_should_fallback() {
        assert!(!SmartFallback::should_fallback(0, 3));
        assert!(!SmartFallback::should_fallback(2, 3));
        assert!(SmartFallback::should_fallback(4, 3));
    }
}
