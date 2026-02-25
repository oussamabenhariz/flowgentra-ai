//! # ErenFlowAI Handler Registration Macro
//!
//! Provides the `#[register_handler]` attribute for automatic handler registration.
//!
//! ## Quick Start
//!
//! ```ignore
//! use erenflow_ai::prelude::*;
//!
//! #[register_handler]
//! pub async fn my_handler(mut state: State) -> Result<State> {
//!     state.set("output", json!("processed"));
//!     Ok(state)
//! }
//!
//! // Create agent - handler auto-discovered!
//! let agent = from_config_path("config.yaml")?;
//! ```

use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, ItemFn};

/// Register a handler function for automatic discovery.
///
/// This attribute macro registers the decorated async function to a global inventory,
/// enabling automatic discovery when creating agents with `from_config_path()`.
///
/// The handler name in the inventory is automatically derived from the function name.
/// Make sure your `config.yaml` references the handler by its function name.
///
/// # Requirements
///
/// Decorated function must be:
/// - `pub async fn`
/// - Takes single `State` parameter (mut or immutable)
/// - Returns `Result<State>`
///
/// # Example
///
/// ```ignore
/// #[register_handler]
/// pub async fn validate_input(mut state: State) -> Result<State> {
///     let input = state.get("input").and_then(|v| v.as_str()).unwrap_or("");
///     state.set("valid", json!(!input.is_empty()));
///     Ok(state)
/// }
/// ```
///
/// Then in your config.yaml:
/// ```yaml
/// nodes:
///   - name: "step1"
///     handler: "validate_input"  # Matches function name
/// ```
#[proc_macro_attribute]
pub fn register_handler(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let input = parse_macro_input!(item as ItemFn);

    let func_name = &input.sig.ident;
    let handler_name = func_name.to_string();

    let expanded = quote! {
        #input

        inventory::submit! {
            erenflow_ai::core::agent::HandlerEntry::new(
                #handler_name,
                std::sync::Arc::new(|state| Box::pin(#func_name(state)))
            )
        }
    };

    TokenStream::from(expanded)
}
