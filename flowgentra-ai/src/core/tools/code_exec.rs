//! Code execution tools: Python REPL, Node.js REPL, and configurable Shell.

use super::{JsonSchema, Tool, ToolDefinition};
use crate::prelude::*;
use async_trait::async_trait;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::process::Stdio;
use std::time::Duration;
use tokio::io::AsyncWriteExt;
use tokio::process::Command;

// =============================================================================
// Shared subprocess runner
// =============================================================================

struct ExecOutput {
    stdout: String,
    stderr: String,
    exit_code: i32,
}

async fn run_subprocess(
    program: &str,
    args: &[&str],
    stdin_data: Option<&str>,
    timeout: Duration,
) -> Result<ExecOutput> {
    let mut child = Command::new(program)
        .args(args)
        .stdin(if stdin_data.is_some() {
            Stdio::piped()
        } else {
            Stdio::null()
        })
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| FlowgentraError::ToolError(format!("Failed to spawn '{}': {}", program, e)))?;

    // Write to stdin if provided, then drop it to signal EOF.
    if let Some(data) = stdin_data {
        if let Some(mut stdin) = child.stdin.take() {
            stdin
                .write_all(data.as_bytes())
                .await
                .map_err(|e| FlowgentraError::ToolError(format!("Failed to write stdin: {}", e)))?;
            // stdin dropped here, EOF sent
        }
    }

    match tokio::time::timeout(timeout, child.wait_with_output()).await {
        Ok(Ok(output)) => Ok(ExecOutput {
            stdout: String::from_utf8_lossy(&output.stdout).into_owned(),
            stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
            exit_code: output.status.code().unwrap_or(-1),
        }),
        Ok(Err(e)) => Err(FlowgentraError::ToolError(format!(
            "Process wait error: {}",
            e
        ))),
        Err(_) => {
            // Timeout — process is still running; it will be cleaned up by the OS.
            Ok(ExecOutput {
                stdout: String::new(),
                stderr: "Execution timed out".to_string(),
                exit_code: -1,
            })
        }
    }
}

// =============================================================================
// PythonReplTool
// =============================================================================

/// Execute Python code in a subprocess and return stdout/stderr/exit_code.
pub struct PythonReplTool {
    python_path: String,
    timeout_secs: u64,
}

impl PythonReplTool {
    pub fn new(python_path: impl Into<String>, timeout_secs: u64) -> Self {
        Self {
            python_path: python_path.into(),
            timeout_secs,
        }
    }
}

impl Default for PythonReplTool {
    fn default() -> Self {
        #[cfg(target_os = "windows")]
        let python = "python";
        #[cfg(not(target_os = "windows"))]
        let python = "python3";
        Self::new(python, 30)
    }
}

#[async_trait]
impl Tool for PythonReplTool {
    async fn call(&self, input: Value) -> Result<Value> {
        let code = input
            .get("code")
            .and_then(|v| v.as_str())
            .ok_or_else(|| FlowgentraError::ToolError("Missing 'code' field".to_string()))?;

        let timeout = input
            .get("timeout_secs")
            .and_then(|v| v.as_u64())
            .unwrap_or(self.timeout_secs);

        let out = run_subprocess(
            &self.python_path,
            &["-c", code],
            None,
            Duration::from_secs(timeout),
        )
        .await?;

        Ok(json!({
            "output": out.stdout,
            "error": out.stderr,
            "exit_code": out.exit_code,
        }))
    }

    fn definition(&self) -> ToolDefinition {
        let mut props = HashMap::new();
        props.insert(
            "code".to_string(),
            JsonSchema::string().with_description("Python code to execute"),
        );
        props.insert(
            "timeout_secs".to_string(),
            JsonSchema::integer().with_description("Execution timeout in seconds (default: 30)"),
        );

        ToolDefinition::new(
            "python_repl",
            "Execute Python code in a subprocess and return stdout, stderr, and exit code",
            JsonSchema::object()
                .with_properties(props)
                .with_required(vec!["code".to_string()]),
            JsonSchema::object().with_properties({
                let mut out = HashMap::new();
                out.insert("output".to_string(), JsonSchema::string());
                out.insert("error".to_string(), JsonSchema::string());
                out.insert("exit_code".to_string(), JsonSchema::integer());
                out
            }),
        )
        .with_category("code")
        .with_example(
            json!({"code": "print(2 + 2)"}),
            json!({"output": "4\n", "error": "", "exit_code": 0}),
            "Evaluate a simple Python expression",
        )
    }
}

// =============================================================================
// NodeJsReplTool
// =============================================================================

/// Execute JavaScript code via Node.js in a subprocess.
pub struct NodeJsReplTool {
    timeout_secs: u64,
}

impl NodeJsReplTool {
    pub fn new(timeout_secs: u64) -> Self {
        Self { timeout_secs }
    }
}

impl Default for NodeJsReplTool {
    fn default() -> Self {
        Self::new(30)
    }
}

#[async_trait]
impl Tool for NodeJsReplTool {
    async fn call(&self, input: Value) -> Result<Value> {
        let code = input
            .get("code")
            .and_then(|v| v.as_str())
            .ok_or_else(|| FlowgentraError::ToolError("Missing 'code' field".to_string()))?;

        let timeout = input
            .get("timeout_secs")
            .and_then(|v| v.as_u64())
            .unwrap_or(self.timeout_secs);

        let out = run_subprocess("node", &["-e", code], None, Duration::from_secs(timeout)).await?;

        Ok(json!({
            "output": out.stdout,
            "error": out.stderr,
            "exit_code": out.exit_code,
        }))
    }

    fn definition(&self) -> ToolDefinition {
        let mut props = HashMap::new();
        props.insert(
            "code".to_string(),
            JsonSchema::string().with_description("JavaScript code to execute with Node.js"),
        );
        props.insert(
            "timeout_secs".to_string(),
            JsonSchema::integer().with_description("Execution timeout in seconds (default: 30)"),
        );

        ToolDefinition::new(
            "nodejs_repl",
            "Execute JavaScript code with Node.js and return stdout, stderr, and exit code",
            JsonSchema::object()
                .with_properties(props)
                .with_required(vec!["code".to_string()]),
            JsonSchema::object().with_properties({
                let mut out = HashMap::new();
                out.insert("output".to_string(), JsonSchema::string());
                out.insert("error".to_string(), JsonSchema::string());
                out.insert("exit_code".to_string(), JsonSchema::integer());
                out
            }),
        )
        .with_category("code")
        .with_example(
            json!({"code": "console.log(2 + 2)"}),
            json!({"output": "4\n", "error": "", "exit_code": 0}),
            "Evaluate a simple JavaScript expression",
        )
    }
}

// =============================================================================
// ShellTool
// =============================================================================

/// Run shell commands via `sh -c`.
///
/// By default requires an explicit allowlist. Use `ShellTool::unrestricted()` to
/// permit any command (use only in trusted environments).
pub struct ShellTool {
    /// `Some(list)` — only commands whose first token is in the list are allowed.
    /// `None` — all commands are permitted (unrestricted mode).
    allowed_commands: Option<Vec<String>>,
    timeout_secs: u64,
}

impl ShellTool {
    /// Create a tool that only allows commands whose first token is in `allowed`.
    pub fn new(allowed_commands: Vec<String>, timeout_secs: u64) -> Self {
        Self {
            allowed_commands: Some(allowed_commands),
            timeout_secs,
        }
    }

    /// Create a tool with no command restrictions.
    pub fn unrestricted(timeout_secs: u64) -> Self {
        Self {
            allowed_commands: None,
            timeout_secs,
        }
    }

    fn check_allowed(&self, command: &str) -> Result<()> {
        if let Some(ref allowed) = self.allowed_commands {
            let first_token = command.split_whitespace().next().unwrap_or("");
            if !allowed.iter().any(|a| a == first_token) {
                return Err(FlowgentraError::ToolError(format!(
                    "Command '{}' is not in the allowed list: {:?}",
                    first_token, allowed
                )));
            }
        }
        Ok(())
    }
}

impl Default for ShellTool {
    /// Default is a restricted shell with no allowed commands.
    fn default() -> Self {
        Self::new(vec![], 30)
    }
}

#[async_trait]
impl Tool for ShellTool {
    async fn call(&self, input: Value) -> Result<Value> {
        let command = input
            .get("command")
            .and_then(|v| v.as_str())
            .ok_or_else(|| FlowgentraError::ToolError("Missing 'command' field".to_string()))?;

        self.check_allowed(command)?;

        let timeout = input
            .get("timeout_secs")
            .and_then(|v| v.as_u64())
            .unwrap_or(self.timeout_secs);

        let out =
            run_subprocess("sh", &["-c", command], None, Duration::from_secs(timeout)).await?;

        Ok(json!({
            "command": command,
            "stdout": out.stdout,
            "stderr": out.stderr,
            "exit_code": out.exit_code,
        }))
    }

    fn definition(&self) -> ToolDefinition {
        let mut props = HashMap::new();
        props.insert(
            "command".to_string(),
            JsonSchema::string().with_description("Shell command to execute"),
        );
        props.insert(
            "timeout_secs".to_string(),
            JsonSchema::integer().with_description("Execution timeout in seconds (default: 30)"),
        );

        ToolDefinition::new(
            "shell",
            "Execute a shell command via sh -c (restricted by allowlist unless created with unrestricted())",
            JsonSchema::object()
                .with_properties(props)
                .with_required(vec!["command".to_string()]),
            JsonSchema::object().with_properties({
                let mut out = HashMap::new();
                out.insert("command".to_string(), JsonSchema::string());
                out.insert("stdout".to_string(), JsonSchema::string());
                out.insert("stderr".to_string(), JsonSchema::string());
                out.insert("exit_code".to_string(), JsonSchema::integer());
                out
            }),
        )
        .with_category("code")
        .with_example(
            json!({"command": "echo hello"}),
            json!({"command": "echo hello", "stdout": "hello\n", "stderr": "", "exit_code": 0}),
            "Run a simple echo command",
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_python_repl_basic() {
        let tool = PythonReplTool::default();
        let result = tool.call(json!({"code": "print(1 + 1)"})).await.unwrap();
        assert_eq!(result["output"].as_str().unwrap().trim(), "2");
        assert_eq!(result["exit_code"], 0);
    }

    #[tokio::test]
    async fn test_shell_allowlist_blocks() {
        let tool = ShellTool::new(vec!["echo".to_string()], 10);
        let err = tool.call(json!({"command": "rm -rf /"})).await.unwrap_err();
        assert!(err.to_string().contains("not in the allowed list"));
    }

    #[tokio::test]
    async fn test_shell_allowlist_permits() {
        let tool = ShellTool::new(vec!["echo".to_string()], 10);
        let result = tool.call(json!({"command": "echo hello"})).await.unwrap();
        assert_eq!(result["stdout"].as_str().unwrap().trim(), "hello");
    }
}
