//! # Evaluation Reporting
//!
//! Generate readable evaluation reports and summaries from execution data.

use crate::core::error::Result;
use crate::core::state::SharedState;
use chrono::Utc;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::fs;

/// Result for a single node
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NodeResult {
    pub node_name: String,
    pub score: f64,
    pub confidence: f64,
    pub retries: u32,
    pub issues: Vec<String>,
    pub suggestions: Vec<String>,
}

/// Complete evaluation report
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EvaluationReport {
    pub nodes: Vec<NodeResult>,
    pub overall_score: f64,
    pub total_retries: u32,
    pub passed: bool,
    pub timestamp: String,
}

impl EvaluationReport {
    /// Extract report from state
    pub fn from_state(state: &SharedState) -> Self {
        let mut nodes = vec![];
        let mut total_score = 0.0;
        let mut total_retries = 0u32;
        let mut node_count = 0usize;

        // Look for evaluation metadata keys
        for key in state.keys() {
            if key.starts_with("__eval_meta__") {
                let node_name = key.replace("__eval_meta__", "");

                if let Some(meta) = state.get(&key) {
                    let score = meta.get("score").and_then(|v| v.as_f64()).unwrap_or(0.5);

                    let confidence = meta
                        .get("confidence")
                        .and_then(|v| v.as_f64())
                        .unwrap_or(score);

                    let retries = meta.get("retries").and_then(|v| v.as_u64()).unwrap_or(0) as u32;

                    let issues: Vec<String> = meta
                        .get("issues")
                        .and_then(|v| v.as_array())
                        .map(|arr| {
                            arr.iter()
                                .filter_map(|v| v.as_str().map(|s| s.to_string()))
                                .collect()
                        })
                        .unwrap_or_default();

                    let suggestions: Vec<String> = meta
                        .get("suggestions")
                        .and_then(|v| v.as_array())
                        .map(|arr| {
                            arr.iter()
                                .filter_map(|v| v.as_str().map(|s| s.to_string()))
                                .collect()
                        })
                        .unwrap_or_default();

                    nodes.push(NodeResult {
                        node_name,
                        score,
                        confidence,
                        retries,
                        issues,
                        suggestions,
                    });

                    total_score += score;
                    total_retries += retries;
                    node_count += 1;
                }
            }
        }

        let overall_score = if node_count > 0 {
            total_score / node_count as f64
        } else {
            0.0
        };

        let passed = overall_score >= 0.80;

        EvaluationReport {
            nodes,
            overall_score,
            total_retries,
            passed,
            timestamp: Utc::now().to_rfc3339(),
        }
    }

    /// Print a formatted report to console
    pub fn print(&self) {
        println!("\n╔════════════════════════════════════════╗");
        println!("║         EVALUATION REPORT              ║");
        println!("╠════════════════════════════════════════╣");
        println!(
            "║ Overall Score:     {:.2}/1.0              ║",
            self.overall_score
        );
        println!(
            "║ Status:            {}                   ║",
            if self.passed {
                "✅ PASS     "
            } else {
                "❌ NEEDS WORK"
            }
        );
        println!(
            "║ Total Retries:     {}                    ║",
            self.total_retries
        );
        println!("╠════════════════════════════════════════╣");

        if self.nodes.is_empty() {
            println!("║ No evaluation data found               ║");
        } else {
            for (i, node) in self.nodes.iter().enumerate() {
                if i > 0 {
                    println!("├────────────────────────────────────────┤");
                }

                println!(
                    "║ Node: {}                          ║",
                    node.node_name.chars().take(30).collect::<String>()
                );
                println!(
                    "║   Score:       {:.2} | Confidence: {:.2}   ║",
                    node.score, node.confidence
                );
                println!("║   Retries:     {}                     ║", node.retries);

                if !node.issues.is_empty() {
                    println!("║   Issues:                              ║");
                    for issue in &node.issues {
                        let truncated = if issue.len() > 32 {
                            format!("{}...", &issue[..29])
                        } else {
                            issue.clone()
                        };
                        println!("║     • {}                      ║", truncated);
                    }
                }

                if !node.suggestions.is_empty() {
                    println!("║   Suggestions:                         ║");
                    for sugg in node.suggestions.iter().take(2) {
                        let truncated = if sugg.len() > 30 {
                            format!("{}...", &sugg[..27])
                        } else {
                            sugg.clone()
                        };
                        println!("║     • {}                    ║", truncated);
                    }
                }
            }
        }

        println!("╚════════════════════════════════════════╝\n");
    }

    /// Print a simple oneline summary
    pub fn print_summary(&self) {
        let status = if self.passed { "✅" } else { "❌" };
        println!(
            "{} Evaluation: {:.2}/1.0 | {} retries",
            status, self.overall_score, self.total_retries
        );
    }

    /// Save report to JSON file
    pub fn save_json(&self, path: &str) -> Result<()> {
        let json = serde_json::to_string_pretty(self)?;
        fs::write(path, json)?;
        tracing::info!("✅ Evaluation report saved to {}", path);
        Ok(())
    }

    /// Get report as JSON value
    pub fn to_json(&self) -> Value {
        json!(self)
    }

    /// Compare two reports and show differences
    pub fn compare_with(&self, other: &EvaluationReport) -> String {
        let mut output = String::new();
        output.push_str("📊 Comparison Report\n");
        output.push_str("════════════════════════════════════════\n");

        let score_diff = self.overall_score - other.overall_score;
        output.push_str(&format!(
            "Overall Score: {:.2} → {:.2} ({}{:.2})\n",
            other.overall_score,
            self.overall_score,
            if score_diff > 0.0 { "+" } else { "" },
            score_diff
        ));

        let retry_diff = self.total_retries as i32 - other.total_retries as i32;
        output.push_str(&format!(
            "Total Retries: {} → {} ({}{}\n",
            other.total_retries,
            self.total_retries,
            if retry_diff > 0 { "+" } else { "" },
            retry_diff
        ));

        output
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_report_from_empty_state() {
        let state = SharedState::default();
        let report = EvaluationReport::from_state(&state);
        assert_eq!(report.overall_score, 0.0);
        assert!(report.nodes.is_empty());
    }

    #[test]
    fn test_report_passed_threshold() {
        let state = SharedState::default();
        // A report with no data has 0.0 score, which fails
        let report = EvaluationReport::from_state(&state);
        assert!(!report.passed);
    }
}
