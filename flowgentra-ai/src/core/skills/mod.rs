//! # Skills System
//!
//! Instruction-first capability bundles following the agentskills.io open standard.
//!
//! A skill is a directory containing:
//! - `SKILL.md` (required) — YAML frontmatter + markdown instructions body
//! - `scripts/`  (optional) — skill-specific tool implementations
//! - `references/` (optional) — extra context injected into the system prompt
//! - `assets/` (optional) — static templates and data files
//!
//! ## Two-phase interaction
//!
//! **Phase 1 — Discovery**: LLM sees only skill names + descriptions via [`SkillRegistry::build_menu`].
//! **Phase 2 — Execution**: LLM sees full instructions + scoped tools via
//! [`SkillRegistry::build_system_prompt`].

use crate::core::error::{FlowgentraError, Result};
use indexmap::IndexMap;
use serde::{Deserialize, Serialize};

// ── Skill struct ───────────────────────────────────────────────────────────────

/// A fully parsed skill ready to be used by an agent.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Skill {
    /// Lowercase, hyphen-separated identifier. Matches the directory name.
    pub name: String,
    /// One-sentence description shown to the LLM during skill discovery.
    pub description: String,
    /// Optional semver string.
    pub version: Option<String>,
    /// Optional SPDX license identifier.
    pub license: Option<String>,
    /// The SKILL.md markdown body — injected into the agent's system prompt.
    pub instructions: String,
    /// Tool names visible to the LLM when this skill is active.
    /// May reference built-in ToolRegistry tools or skill-specific tools from scripts/.
    pub allowed_tools: Vec<String>,
    /// Text content loaded from the references/ directory.
    pub references: Vec<String>,
}

// ── SKILL.md frontmatter ───────────────────────────────────────────────────────

#[derive(Debug, Deserialize, Default)]
pub struct SkillFrontmatter {
    pub name: Option<String>,
    pub description: Option<String>,
    pub version: Option<String>,
    pub license: Option<String>,
    #[serde(rename = "allowed-tools")]
    pub allowed_tools: Option<Vec<String>>,
}

// ── Parser ─────────────────────────────────────────────────────────────────────

/// Parse a SKILL.md file contents into a [`Skill`].
///
/// The `dir_name` is used as a fallback `name` when the frontmatter omits it.
pub fn parse_skill_md(content: &str, _dir_name: &str) -> Result<(SkillFrontmatter, String)> {
    let content = content.trim_start();

    if !content.starts_with("---") {
        return Ok((SkillFrontmatter::default(), content.to_string()));
    }

    // Skip the opening `---` and find the closing `\n---`
    let after_open = &content[3..];
    let close_pos = after_open.find("\n---").ok_or_else(|| {
        FlowgentraError::ConfigError(
            "SKILL.md has an unclosed frontmatter block (missing closing ---)".into(),
        )
    })?;

    let frontmatter_str = &after_open[..close_pos];
    let body_start = close_pos + 4; // len("\n---") == 4
    let body = after_open
        .get(body_start..)
        .unwrap_or("")
        .trim()
        .to_string();

    let frontmatter: SkillFrontmatter = serde_yml::from_str(frontmatter_str)
        .map_err(|e| FlowgentraError::ConfigError(format!("Invalid SKILL.md frontmatter: {e}")))?;

    Ok((frontmatter, body))
}

// ── Errors ─────────────────────────────────────────────────────────────────────

#[derive(Debug, thiserror::Error)]
pub enum SkillError {
    #[error("Skill '{name}' lists tool '{tool}' in allowed-tools but it could not be resolved")]
    ToolNotFound { name: String, tool: String },

    #[error("Skill '{0}' is already registered. Use allow_override=true to replace it")]
    AlreadyRegistered(String),

    #[error("Skill '{0}' not found")]
    NotFound(String),
}

impl From<SkillError> for FlowgentraError {
    fn from(e: SkillError) -> Self {
        FlowgentraError::ConfigError(e.to_string())
    }
}

// ── SkillRegistry ──────────────────────────────────────────────────────────────

/// Central registry for managing skills.
///
/// Stores [`Skill`] structs keyed by name and generates the system prompts
/// for both the discovery phase and the execution phase.
pub struct SkillRegistry {
    /// Skills in insertion order.
    skills: IndexMap<String, Skill>,
}

impl SkillRegistry {
    pub fn new() -> Self {
        Self {
            skills: IndexMap::new(),
        }
    }

    /// Register a [`Skill`].
    ///
    /// # Errors
    /// Returns [`SkillError::AlreadyRegistered`] if a skill with the same name
    /// exists and `allow_override` is `false`.
    pub fn register(&mut self, skill: Skill, allow_override: bool) -> Result<()> {
        if !allow_override && self.skills.contains_key(&skill.name) {
            return Err(SkillError::AlreadyRegistered(skill.name).into());
        }
        self.skills.insert(skill.name.clone(), skill);
        Ok(())
    }

    /// Build a [`Skill`] from raw SKILL.md content plus pre-loaded reference texts.
    ///
    /// `dir_name` is used as a fallback name when the frontmatter omits `name`.
    pub fn build_skill(content: &str, dir_name: &str, references: Vec<String>) -> Result<Skill> {
        let (fm, instructions) = parse_skill_md(content, dir_name)?;
        Ok(Skill {
            name: fm.name.unwrap_or_else(|| dir_name.to_string()),
            description: fm.description.unwrap_or_default(),
            version: fm.version,
            license: fm.license,
            instructions,
            allowed_tools: fm.allowed_tools.unwrap_or_default(),
            references,
        })
    }

    /// Validate that every name in `skill.allowed_tools` appears in `known_tools`.
    ///
    /// `known_tools` should be the union of the global ToolRegistry names and the
    /// names of skill-specific tools discovered from scripts/.
    pub fn validate_tools(skill: &Skill, known_tools: &[String]) -> Result<()> {
        for tool_name in &skill.allowed_tools {
            if !known_tools.contains(tool_name) {
                return Err(SkillError::ToolNotFound {
                    name: skill.name.clone(),
                    tool: tool_name.clone(),
                }
                .into());
            }
        }
        Ok(())
    }

    // ── Read ──────────────────────────────────────────────────────────────────

    pub fn get(&self, name: &str) -> Option<&Skill> {
        self.skills.get(name)
    }

    pub fn list(&self) -> Vec<&str> {
        self.skills.keys().map(|s| s.as_str()).collect()
    }

    pub fn len(&self) -> usize {
        self.skills.len()
    }

    pub fn is_empty(&self) -> bool {
        self.skills.is_empty()
    }

    pub fn contains(&self, name: &str) -> bool {
        self.skills.contains_key(name)
    }

    // ── Phase 1 — Discovery ───────────────────────────────────────────────────

    /// Build the discovery-phase system prompt.
    ///
    /// The LLM receives only skill names and descriptions — no instructions,
    /// no tools. It selects a skill via the `activate_skill` tool.
    pub fn build_menu(&self) -> String {
        let mut lines = vec![
            "You have access to the following skills.".to_string(),
            "Read the user's request and select the most relevant skill.\n".to_string(),
            "Available skills:".to_string(),
        ];
        for skill in self.skills.values() {
            lines.push(format!("- {}: {}", skill.name, skill.description));
        }
        lines.join("\n")
    }

    // ── Phase 2 — Execution ───────────────────────────────────────────────────

    /// Build the execution-phase system prompt for one skill (or all if `None`).
    ///
    /// Injects the skill's instruction body and any reference content.
    ///
    /// # Errors
    /// Returns [`SkillError::NotFound`] if `skill_name` does not exist.
    pub fn build_system_prompt(&self, skill_name: Option<&str>) -> Result<String> {
        let skills: Vec<&Skill> = match skill_name {
            Some(name) => {
                let skill = self
                    .skills
                    .get(name)
                    .ok_or_else(|| SkillError::NotFound(name.to_string()))?;
                vec![skill]
            }
            None => self.skills.values().collect(),
        };

        let sections: Vec<String> = skills
            .iter()
            .map(|skill| {
                let mut parts = vec![format!(
                    "## Skill: {}\n\n{}",
                    skill.name, skill.instructions
                )];
                if !skill.references.is_empty() {
                    parts.push(skill.references.join("\n\n"));
                }
                parts.join("\n\n")
            })
            .collect();

        Ok(sections.join("\n\n---\n\n"))
    }

    /// Return the `allowed_tools` list for a skill.
    ///
    /// The Python binding uses this to resolve ToolSpec objects.
    pub fn allowed_tools(&self, skill_name: &str) -> Result<&[String]> {
        let skill = self
            .skills
            .get(skill_name)
            .ok_or_else(|| SkillError::NotFound(skill_name.to_string()))?;
        Ok(&skill.allowed_tools)
    }

    /// Return all unique allowed_tools across all skills (in insertion order).
    pub fn all_allowed_tools(&self) -> Vec<&str> {
        let mut seen = std::collections::HashSet::new();
        let mut result = Vec::new();
        for skill in self.skills.values() {
            for tool in &skill.allowed_tools {
                if seen.insert(tool.as_str()) {
                    result.push(tool.as_str());
                }
            }
        }
        result
    }
}

impl Default for SkillRegistry {
    fn default() -> Self {
        Self::new()
    }
}
