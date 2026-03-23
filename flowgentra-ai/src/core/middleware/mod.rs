//! # Middleware System
//!
//! Middleware provides hooks into the agent execution lifecycle for cross-cutting concerns
//! like logging, metrics collection, and validation.

mod interceptors;

pub use interceptors::*;
