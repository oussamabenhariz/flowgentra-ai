//! Secret string wrapper that keeps credentials out of logs and serialized output.

use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::fmt;
use zeroize::Zeroize;

/// Placeholder written wherever a [`Secret`] would otherwise be printed or serialized.
pub const REDACTED: &str = "***REDACTED***";

/// A sensitive string (API key, token, password) that cannot leak by accident.
///
/// - `Debug` and `Display` print `Secret(***REDACTED***)`, so the value never
///   appears in `format!`, logs, traces, or error messages.
/// - `Serialize` writes `"***REDACTED***"`, so checkpoints and config dumps
///   never contain the raw key. `Deserialize` accepts a plain string, so
///   loading configs works normally — but serializing and re-loading a config
///   will *not* round-trip the key. Re-resolve it from the environment instead.
/// - The underlying buffer is zeroized on drop.
///
/// Read the value only at the point of use with [`Secret::expose`]:
///
/// ```
/// use flowgentra_ai::core::llm::Secret;
///
/// let key = Secret::new("sk-super-secret");
/// assert_eq!(format!("{:?}", key), "Secret(***REDACTED***)");
/// assert_eq!(key.expose(), "sk-super-secret");
/// ```
#[derive(Clone, Default, PartialEq, Eq)]
pub struct Secret(String);

impl Secret {
    /// Wrap a sensitive value.
    pub fn new(value: impl Into<String>) -> Self {
        Secret(value.into())
    }

    /// Read the raw value. Use only at the point where the credential is sent
    /// to the provider (e.g. building an `Authorization` header).
    pub fn expose(&self) -> &str {
        &self.0
    }

    /// `true` if no value is set.
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

impl From<String> for Secret {
    fn from(value: String) -> Self {
        Secret(value)
    }
}

impl From<&str> for Secret {
    fn from(value: &str) -> Self {
        Secret(value.to_string())
    }
}

impl fmt::Debug for Secret {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Secret({})", REDACTED)
    }
}

impl fmt::Display for Secret {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Secret({})", REDACTED)
    }
}

impl Serialize for Secret {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(REDACTED)
    }
}

impl<'de> Deserialize<'de> for Secret {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        Ok(Secret(String::deserialize(deserializer)?))
    }
}

impl Drop for Secret {
    fn drop(&mut self) {
        self.0.zeroize();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn debug_and_display_redact() {
        let s = Secret::new("sk-abc123");
        assert!(!format!("{:?}", s).contains("sk-abc123"));
        assert!(!format!("{}", s).contains("sk-abc123"));
    }

    #[test]
    fn serialize_redacts_deserialize_accepts() {
        let s = Secret::new("sk-abc123");
        let json = serde_json::to_string(&s).unwrap();
        assert!(!json.contains("sk-abc123"));
        assert_eq!(json, format!("\"{}\"", REDACTED));

        let loaded: Secret = serde_json::from_str("\"sk-live\"").unwrap();
        assert_eq!(loaded.expose(), "sk-live");
    }

    #[test]
    fn expose_returns_value() {
        let s = Secret::new("value");
        assert_eq!(s.expose(), "value");
        assert!(!s.is_empty());
        assert!(Secret::default().is_empty());
    }
}
