//! # Automatic Retry with Self-Correction
//!
//! Implements retry logic for low-confidence outputs.
//!
//! Features:
//! - Exponential backoff
//! - Self-correction with feedback
//! - Adaptive strategies
//! - Circuit breaker pattern

use serde::{Deserialize, Serialize};

/// Configuration for retry behavior
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RetryConfig {
    /// Maximum number of retries
    #[serde(default = "default_max_retries")]
    pub max_retries: u32,

    /// Initial delay in milliseconds before first retry
    #[serde(default = "default_initial_delay")]
    pub initial_delay_ms: u64,

    /// Multiplier for exponential backoff
    #[serde(default = "default_backoff_multiplier")]
    pub backoff_multiplier: f64,

    /// Maximum delay in milliseconds
    #[serde(default = "default_max_delay")]
    pub max_delay_ms: u64,

    /// Confidence threshold for retry (below this triggers retry)
    #[serde(default = "default_retry_threshold")]
    pub confidence_threshold: f64,

    /// Include feedback from evaluation in retry prompt
    #[serde(default = "default_true")]
    pub include_feedback: bool,

    /// Use temperature increase for more diverse retry attempts
    #[serde(default = "default_true")]
    pub increase_temperature: bool,

    /// Enable circuit breaker (stop retrying after N consecutive failures)
    #[serde(default = "default_true")]
    pub enable_circuit_breaker: bool,

    /// Circuit breaker failure threshold before stopping
    #[serde(default = "default_circuit_breaker_threshold")]
    pub circuit_breaker_threshold: u32,
}

fn default_max_retries() -> u32 {
    3
}

fn default_initial_delay() -> u64 {
    100
}

fn default_backoff_multiplier() -> f64 {
    2.0
}

fn default_max_delay() -> u64 {
    5000
}

fn default_retry_threshold() -> f64 {
    0.65
}

fn default_true() -> bool {
    true
}

fn default_circuit_breaker_threshold() -> u32 {
    3
}

impl Default for RetryConfig {
    fn default() -> Self {
        RetryConfig {
            max_retries: 3,
            initial_delay_ms: 100,
            backoff_multiplier: 2.0,
            max_delay_ms: 5000,
            confidence_threshold: 0.65,
            include_feedback: true,
            increase_temperature: true,
            enable_circuit_breaker: true,
            circuit_breaker_threshold: 3,
        }
    }
}

/// Result information for a retry attempt
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RetryResult {
    /// Whether output was retried
    pub was_retried: bool,

    /// Number of retries performed
    pub retry_count: u32,

    /// Confidences of each attempt (in order)
    pub confidence_history: Vec<f64>,

    /// Whether retry succeeded (final confidence >= threshold)
    pub success: bool,

    /// Improvement from first to final attempt
    pub improvement: f64,

    /// Reason for stopping retries
    pub stop_reason: String,
}

impl RetryResult {
    /// Create a new retry result for non-retried output
    pub fn new_no_retry() -> Self {
        RetryResult {
            was_retried: false,
            retry_count: 0,
            confidence_history: Vec::new(),
            success: false,
            improvement: 0.0,
            stop_reason: "No retry needed".into(),
        }
    }

    /// Create a new retry result for a retried output
    pub fn new_retried(
        retry_count: u32,
        confidence_history: Vec<f64>,
        success: bool,
        stop_reason: String,
    ) -> Self {
        let improvement = if confidence_history.len() > 1 {
            confidence_history.last().unwrap_or(&0.0) - confidence_history.first().unwrap_or(&0.0)
        } else {
            0.0
        };

        RetryResult {
            was_retried: true,
            retry_count,
            confidence_history,
            success,
            improvement,
            stop_reason,
        }
    }
}

/// Manages retry logic and self-correction
pub struct RetryPolicy;

impl RetryPolicy {
    /// Determine if a retry should happen
    pub fn should_retry(confidence: f64, retry_count: u32, config: &RetryConfig) -> bool {
        if retry_count >= config.max_retries {
            return false;
        }

        confidence < config.confidence_threshold
    }

    /// Get the delay for the next retry
    pub fn get_retry_delay(retry_count: u32, config: &RetryConfig) -> u64 {
        let delay = (config.initial_delay_ms as f64
            * config.backoff_multiplier.powi(retry_count as i32)) as u64;
        delay.min(config.max_delay_ms)
    }

    /// Calculate temperature adjustment for retry.
    ///
    /// Higher temperature = more creative/diverse output.
    /// Capped at 1.0 — values above 1.0 are rejected by most providers.
    pub fn get_temperature_adjustment(retry_count: u32) -> f64 {
        (0.7 + (retry_count as f64 * 0.15)).min(1.0)
    }

    /// Build retry feedback prompt
    pub fn build_retry_feedback(
        feedback: &str,
        issues: &[String],
        suggestions: &[String],
    ) -> String {
        let mut prompt =
            String::from("Your previous attempt was not satisfactory. Please improve:\n\n");

        prompt.push_str(&format!("Feedback: {}\n\n", feedback));

        if !issues.is_empty() {
            prompt.push_str("Issues to address:\n");
            for issue in issues {
                prompt.push_str(&format!("- {}\n", issue));
            }
            prompt.push('\n');
        }

        if !suggestions.is_empty() {
            prompt.push_str("Suggestions for improvement:\n");
            for sugg in suggestions {
                prompt.push_str(&format!("- {}\n", sugg));
            }
            prompt.push('\n');
        }

        prompt.push_str("Please provide a revised response addressing these points.");

        prompt
    }

    /// Check circuit breaker status
    pub fn check_circuit_breaker(consecutive_failures: u32, config: &RetryConfig) -> bool {
        if !config.enable_circuit_breaker {
            return true; // Circuit breaker disabled
        }

        consecutive_failures < config.circuit_breaker_threshold
    }

    /// Generate detailed retry report
    pub fn generate_report(result: &RetryResult) -> String {
        if !result.was_retried {
            return "No retries performed".to_string();
        }

        let mut report = "Retry Report:\n".to_string();
        report.push_str(&format!("Attempts: {}\n", result.retry_count + 1));
        report.push_str(&format!("Success: {}\n", result.success));
        report.push_str(&format!(
            "Improvement: {:.1}%\n",
            result.improvement * 100.0
        ));

        report.push_str("\nConfidence trajectory:\n");
        for (i, confidence) in result.confidence_history.iter().enumerate() {
            let bar = Self::confidence_bar(*confidence);
            report.push_str(&format!("  Attempt {}: {:.2} {}\n", i + 1, confidence, bar));
        }

        report.push_str(&format!("\nStop reason: {}", result.stop_reason));

        report
    }

    /// Generate ASCII confidence visualization.
    /// Input is clamped to [0.0, 1.0] to prevent panic on out-of-range values.
    fn confidence_bar(confidence: f64) -> String {
        let confidence = confidence.clamp(0.0, 1.0);
        let bar_len = (confidence * 20.0) as usize;
        let filled = "█".repeat(bar_len);
        let empty = "░".repeat(20 - bar_len);
        format!("[{}{}]", filled, empty)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_should_retry() {
        let config = RetryConfig::default();

        // Low confidence should trigger retry
        assert!(RetryPolicy::should_retry(0.5, 0, &config));

        // Above threshold should not retry
        assert!(!RetryPolicy::should_retry(0.8, 0, &config));

        // Max retries exceeded should not retry
        assert!(!RetryPolicy::should_retry(0.5, config.max_retries, &config));
    }

    #[test]
    fn test_retry_delay_backoff() {
        let config = RetryConfig::default();

        let delay_0 = RetryPolicy::get_retry_delay(0, &config);
        let delay_1 = RetryPolicy::get_retry_delay(1, &config);
        let delay_2 = RetryPolicy::get_retry_delay(2, &config);

        assert!(delay_1 > delay_0);
        assert!(delay_2 > delay_1);

        // Should not exceed max delay
        assert!(delay_2 <= config.max_delay_ms);
    }

    #[test]
    fn test_temperature_adjustment() {
        let temp_0 = RetryPolicy::get_temperature_adjustment(0);
        let temp_1 = RetryPolicy::get_temperature_adjustment(1);

        assert!(temp_1 > temp_0);
    }

    #[test]
    fn test_retry_result() {
        let result = RetryResult::new_retried(
            2,
            vec![0.5, 0.6, 0.8],
            true,
            "Confidence threshold reached".into(),
        );

        assert!(result.was_retried);
        assert_eq!(result.retry_count, 2);
        assert!(result.success);
        assert!(result.improvement > 0.0);
    }

    #[test]
    fn test_circuit_breaker() {
        let mut config = RetryConfig::default();
        config.circuit_breaker_threshold = 3;

        assert!(RetryPolicy::check_circuit_breaker(0, &config));
        assert!(RetryPolicy::check_circuit_breaker(2, &config));
        assert!(!RetryPolicy::check_circuit_breaker(3, &config));
    }

    #[test]
    fn test_retry_feedback_prompt() {
        let prompt = RetryPolicy::build_retry_feedback(
            "Not detailed enough",
            &["Missing examples".into()],
            &["Add 2-3 concrete examples".into()],
        );

        assert!(prompt.contains("Not detailed enough"));
        assert!(prompt.contains("Missing examples"));
        assert!(prompt.contains("Add 2-3 concrete examples"));
    }

    #[test]
    fn test_confidence_bar() {
        let bar_low = RetryPolicy::confidence_bar(0.2);
        let bar_high = RetryPolicy::confidence_bar(0.9);

        assert!(bar_low.contains("█"));
        assert!(bar_high.contains("█"));
        assert!(bar_low.contains("░"));
        assert_eq!(bar_low.len(), bar_high.len());
    }

    #[test]
    fn test_temperature_never_exceeds_one() {
        for retry in 0..=10 {
            let temp = RetryPolicy::get_temperature_adjustment(retry);
            assert!(temp <= 1.0, "Temperature {} at retry {} exceeds 1.0", temp, retry);
        }
    }

    #[test]
    fn test_confidence_bar_out_of_range_no_panic() {
        // Must not panic for values outside [0, 1]
        let _ = RetryPolicy::confidence_bar(1.5);
        let _ = RetryPolicy::confidence_bar(-0.1);
    }
}
