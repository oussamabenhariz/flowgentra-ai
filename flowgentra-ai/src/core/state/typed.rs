use crate::core::error::{FlowgentraError, Result};
use crate::core::state::{PlainState, SharedState};
use serde::{de::DeserializeOwned, Serialize};

/// A strongly-typed wrapper around `State` that provides compile-time guarantees
/// for handlers and nodes.
///
/// This eliminates the need for dynamic string-based JSON parsing inside node handlers,
/// resolving type mismatch errors at compile-time instead of runtime.
pub struct TypedState<T> {
    /// The strongly-typed user configuration / state
    pub data: T,
    /// The underlying raw state (used by the engine for routing metadata)
    raw: SharedState,
}

impl<T: DeserializeOwned + Serialize> TypedState<T> {
    /// Deserializes a `State` into a `TypedState`
    pub fn from_state(state: SharedState) -> Result<Self> {
        let value = state.to_value();
        let data: T = serde_json::from_value(value).map_err(|e| {
            FlowgentraError::StateError(format!(
                "Failed to deserialize state into typed struct: {}",
                e
            ))
        })?;
        Ok(Self { data, raw: state })
    }

    /// Serializes the typed data back into the underlying `State`
    ///
    /// This merges the strongly typed fields back into the raw State so that
    /// the framework's routing and metadata still operate correctly.
    pub fn into_state(self) -> Result<SharedState> {
        let value = serde_json::to_value(&self.data).map_err(|e| {
            FlowgentraError::StateError(format!("Failed to serialize typed struct to state: {}", e))
        })?;
        let plain = PlainState::from_json(value)?;
        self.raw.merge(SharedState::new(plain))?;
        Ok(self.raw)
    }

    /// Extracs the strongly typed data and discards the raw state tracking.
    pub fn into_inner(self) -> T {
        self.data
    }
}

/// Helper extension trait to easily transform State into TypedState.
pub trait StateExt {
    /// Convert the generic state into a strongly-typed `TypedState<T>`.
    fn into_typed<T: DeserializeOwned + Serialize>(self) -> Result<TypedState<T>>;
}

impl StateExt for crate::core::state::SharedState {
    fn into_typed<U: DeserializeOwned + Serialize>(self) -> Result<TypedState<U>> {
        TypedState::from_state(self)
    }
}
