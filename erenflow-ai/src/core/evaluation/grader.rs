//! # LLM-Based Output Grading
//!
//! Uses LLMs to evaluate output quality intelligently.
//!
//! The grader prompts an LLM to assess:
//! - Quality of the response
//! - Relevance to the task
//! - Presence of errors or hallucinations
//! - Suggestions for improvement

use crate::core::error::Result;
use crate::core::llm::{LLMClient, Message, MessageRole};
use crate::core::state::State;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::sync::Arc;

/// Grading result from LLM evaluation
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GradeResult {
    /// Overall quality score (0.0-1.0)
    pub score: f64,

    /// Quality rating (Poor, Fair, Good, Excellent)
    pub rating: String,

    /// Whether output meets requirements
    pub passes: bool,

    /// Detailed feedback on the output
    pub feedback: String,

    /// Specific errors or issues found (if any)
    pub issues: Vec<String>,

    /// Suggestions for improvement
    pub suggestions: Vec<String>,
}

impl GradeResult {
    /// Create a new grade result
    pub fn new(score: f64, rating: String, passes: bool, feedback: String) -> Self {
        GradeResult {
            score,
            rating,
            passes,
            feedback,
            issues: Vec::new(),
            suggestions: Vec::new(),
        }
    }

    /// Add an issue
    pub fn with_issue(mut self, issue: String) -> Self {
        self.issues.push(issue);
        self
    }

    /// Add a suggestion
    pub fn with_suggestion(mut self, suggestion: String) -> Self {
        self.suggestions.push(suggestion);
        self
    }

    /// Parse score from rating
    #[allow(dead_code)]
    fn score_from_rating(rating: &str) -> f64 {
        match rating.to_lowercase().as_str() {
            "excellent" => 1.0,
            "good" => 0.8,
            "fair" => 0.6,
            "poor" => 0.2,
            _ => 0.5,
        }
    }
}

/// Grades node outputs using LLM
pub struct LLMGrader;

impl LLMGrader {
    /// Grade an output using an LLM
    pub async fn grade(
        output: &Value,
        task_description: &str,
        context: &str,
        llm_client: Arc<dyn LLMClient>,
    ) -> Result<GradeResult> {
        let prompt = Self::build_grading_prompt(output, task_description, context);

        let messages = vec![Message {
            role: MessageRole::User,
            content: prompt,
            tool_calls: None,
        }];

        let response_msg = llm_client.chat(messages).await?;

        Self::parse_grade_response(&response_msg.content)
    }

    /// Grade a self-correcting attempt
    pub async fn grade_self_correction(
        original_output: &Value,
        corrected_output: &Value,
        feedback: &str,
        llm_client: Arc<dyn LLMClient>,
    ) -> Result<GradeResult> {
        let prompt =
            Self::build_correction_grading_prompt(original_output, corrected_output, feedback);

        let messages = vec![Message {
            role: MessageRole::User,
            content: prompt,
            tool_calls: None,
        }];

        let response_msg = llm_client.chat(messages).await?;

        Self::parse_grade_response(&response_msg.content)
    }

    /// Build a grading prompt for the LLM
    fn build_grading_prompt(output: &Value, task: &str, context: &str) -> String {
        let output_str =
            serde_json::to_string_pretty(output).unwrap_or_else(|_| output.to_string());

        format!(
            r#"You are an expert evaluator. Grade the following output.

TASK: {}

CONTEXT: {}

OUTPUT TO GRADE:
{}

Please provide a structured evaluation:
1. SCORE: A number from 0.0 to 1.0
2. RATING: One of [Excellent, Good, Fair, Poor]
3. PASSES: true or false - does it meet the task requirements?
4. FEEDBACK: Brief overall assessment
5. ISSUES: List of specific problems (if any)
6. SUGGESTIONS: List of improvements

Format your response as JSON:
{{
  "score": 0.85,
  "rating": "good",
  "passes": true,
  "feedback": "Clear and accurate response addressing the main question.",
  "issues": [],
  "suggestions": ["Could provide more detail in X"]
}}
"#,
            task, context, output_str
        )
    }

    /// Build a prompt for grading self-correction
    fn build_correction_grading_prompt(
        original: &Value,
        corrected: &Value,
        feedback: &str,
    ) -> String {
        let original_str =
            serde_json::to_string_pretty(original).unwrap_or_else(|_| original.to_string());
        let corrected_str =
            serde_json::to_string_pretty(corrected).unwrap_or_else(|_| corrected.to_string());

        format!(
            r#"Evaluate how well the agent self-corrected based on feedback.

ORIGINAL OUTPUT:
{}

FEEDBACK RECEIVED:
{}

CORRECTED OUTPUT:
{}

Did the correction address the feedback? Provide:
1. SCORE (0.0-1.0): How well was the feedback addressed?
2. RATING: [Excellent, Good, Fair, Poor]
3. PASSES: Is the corrected output now acceptable?
4. FEEDBACK: Assessment of the correction
5. ISSUES: Remaining problems (if any)
6. SUGGESTIONS: Further improvements

Format as JSON same as before.
"#,
            original_str, feedback, corrected_str
        )
    }

    /// Parse LLM response into GradeResult
    fn parse_grade_response(response: &str) -> Result<GradeResult> {
        // Try to extract JSON from response
        if let Ok(json) = serde_json::from_str::<serde_json::Value>(response) {
            Self::parse_json_grade(&json)
        } else {
            // Try to find JSON in the response
            if let Some(start) = response.find('{') {
                if let Some(end) = response.rfind('}') {
                    let json_str = &response[start..=end];
                    if let Ok(json) = serde_json::from_str::<serde_json::Value>(json_str) {
                        return Self::parse_json_grade(&json);
                    }
                }
            }

            // Fallback: parse the response text
            Self::parse_text_grade(response)
        }
    }

    /// Parse JSON grade response
    fn parse_json_grade(json: &Value) -> Result<GradeResult> {
        let obj = json.as_object().ok_or_else(|| {
            crate::core::error::ErenFlowError::LLMError("Invalid grading format".into())
        })?;

        let score = obj.get("score").and_then(|v| v.as_f64()).unwrap_or(0.5);

        let rating = obj
            .get("rating")
            .and_then(|v| v.as_str())
            .unwrap_or("Fair")
            .to_string();

        let passes = obj
            .get("passes")
            .and_then(|v| v.as_bool())
            .unwrap_or(score > 0.6);

        let feedback = obj
            .get("feedback")
            .and_then(|v| v.as_str())
            .unwrap_or("Output evaluated.")
            .to_string();

        let issues = obj
            .get("issues")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str())
                    .map(|s| s.to_string())
                    .collect()
            })
            .unwrap_or_default();

        let suggestions = obj
            .get("suggestions")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str())
                    .map(|s| s.to_string())
                    .collect()
            })
            .unwrap_or_default();

        Ok(GradeResult {
            score,
            rating,
            passes,
            feedback,
            issues,
            suggestions,
        })
    }

    /// Parse text response into GradeResult (fallback)
    fn parse_text_grade(response: &str) -> Result<GradeResult> {
        // Simple parsing of text response
        let lines: Vec<&str> = response.lines().collect();

        let score = if let Some(line) = lines.iter().find(|l| l.to_lowercase().contains("score")) {
            line.split(':')
                .last()
                .and_then(|s| s.trim().parse::<f64>().ok())
                .unwrap_or(0.7)
        } else {
            0.7
        };

        let rating = if let Some(line) = lines.iter().find(|l| l.to_lowercase().contains("rating"))
        {
            line.split(':').last().unwrap_or(&"Good").trim().to_string()
        } else {
            "Good".to_string()
        };

        let passes = score > 0.6;

        Ok(GradeResult::new(
            score,
            rating,
            passes,
            response.to_string(),
        ))
    }

    /// Generate feedback for self-correction
    pub fn generate_correction_prompt(
        _output: &Value,
        grade: &GradeResult,
        _state: &State,
    ) -> String {
        let mut prompt = format!(
            "Your previous output received the following evaluation:\n\nScore: {}/1.0\nFeedback: {}\n",
            grade.score, grade.feedback
        );

        if !grade.issues.is_empty() {
            prompt.push_str("\nIssues identified:\n");
            for issue in &grade.issues {
                prompt.push_str(&format!("- {}\n", issue));
            }
        }

        if !grade.suggestions.is_empty() {
            prompt.push_str("\nSuggestions for improvement:\n");
            for sugg in &grade.suggestions {
                prompt.push_str(&format!("- {}\n", sugg));
            }
        }

        prompt.push_str("\nPlease revise your answer addressing these points.\n");

        prompt
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_parse_json_grade() {
        let json_response = json!({
            "score": 0.85,
            "rating": "Good",
            "passes": true,
            "feedback": "Well done",
            "issues": [],
            "suggestions": ["More detail"]
        });

        let grade = LLMGrader::parse_json_grade(&json_response);
        assert!(grade.is_ok());

        let grade = grade.unwrap();
        assert_eq!(grade.score, 0.85);
        assert_eq!(grade.rating, "Good");
        assert!(grade.passes);
    }

    #[test]
    fn test_grade_result_builder() {
        let grade = GradeResult::new(0.8, "Good".to_string(), true, "Nice response".to_string())
            .with_issue("Could be more detailed".to_string())
            .with_suggestion("Add examples".to_string());

        assert_eq!(grade.issues.len(), 1);
        assert_eq!(grade.suggestions.len(), 1);
    }
}
