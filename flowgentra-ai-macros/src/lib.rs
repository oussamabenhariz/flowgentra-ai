use proc_macro::TokenStream;
use quote::{format_ident, quote};
use syn::{parse_macro_input, Data, DeriveInput, Fields, ItemFn};

/// Derive the `State` trait for a struct.
///
/// Generates:
/// - A `{Name}Update` struct with all fields wrapped in `Option<T>`
/// - Builder methods on the update struct for ergonomic partial updates
/// - `State` trait impl with `type Update` and `apply_update` using per-field reducers
///
/// # Attributes
///
/// Two equivalent syntaxes are supported for configuring per-field reducers:
///
/// **Ident syntax** — `#[reducer(Kind)]`:
/// - `Overwrite` (default) — replaces the value
/// - `Append` — extends `Vec<T>` fields
/// - `Sum` — adds numeric fields
/// - `MergeMap` — merges `HashMap` fields
///
/// **String syntax** — `#[state(reducer = "kind")]` (LangGraph-style):
/// - `"overwrite"` / `"replace"` / `"last_value"` → `Overwrite`
/// - `"append"` / `"topic"` → `Append`
/// - `"sum"` → `Sum`
/// - `"merge_map"` / `"merge"` → `MergeMap`
///
/// # Example
///
/// ```ignore
/// use flowgentra_ai::prelude::*;
/// use serde::{Serialize, Deserialize};
///
/// #[derive(State, Clone, Debug, Serialize, Deserialize)]
/// struct AgentState {
///     query: String,
///
///     // Ident syntax
///     #[reducer(Append)]
///     messages: Vec<Message>,
///
///     result: Option<String>,
///
///     // String syntax
///     #[state(reducer = "sum")]
///     retry_count: i32,
/// }
///
/// // Nodes return partial updates:
/// let update = AgentStateUpdate::new()
///     .result(Some("done".into()));
/// ```
#[proc_macro_derive(State, attributes(reducer, state))]
pub fn derive_state(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let struct_name = &input.ident;
    let vis = &input.vis;

    // Use relative crate path - macros always work relative to the target crate
    let flowgentra_ai_crate = quote! { crate };

    let fields = match &input.data {
        Data::Struct(data_struct) => match &data_struct.fields {
            Fields::Named(fields_named) => &fields_named.named,
            _ => panic!("#[derive(State)] requires a struct with named fields"),
        },
        _ => panic!("#[derive(State)] requires a struct"),
    };

    let update_name = format_ident!("{}Update", struct_name);

    let mut update_field_defs = Vec::new();
    let mut update_defaults = Vec::new();
    let mut setters = Vec::new();
    let mut apply_arms = Vec::new();

    for field in fields {
        let name = field.ident.as_ref().unwrap();
        let ty = &field.ty;

        // Parse reducer from either #[reducer(Kind)] or #[state(reducer = "kind")].
        // Default to Overwrite.
        let mut reducer_path = quote! { #flowgentra_ai_crate::core::reducer::Overwrite };
        for attr in &field.attrs {
            // ── #[reducer(Kind)] ─────────────────────────────────────────────
            if attr.path().is_ident("reducer") {
                if let Ok(ident) = attr.parse_args::<syn::Ident>() {
                    reducer_path = quote! { #flowgentra_ai_crate::core::reducer::#ident };
                }
                break;
            }
            // ── #[state(reducer = "string")] ─────────────────────────────────
            if attr.path().is_ident("state") {
                // parse as a key = "value" meta
                let _ = attr.parse_nested_meta(|meta| {
                    if meta.path.is_ident("reducer") {
                        let value: syn::LitStr = meta.value()?.parse()?;
                        let kind = match value.value().to_lowercase().as_str() {
                            "append" | "topic" => "Append",
                            "sum" => "Sum",
                            "merge_map" | "merge" => "MergeMap",
                            // "overwrite" | "replace" | "last_value" | _ → Overwrite (default)
                            _ => "Overwrite",
                        };
                        let kind_ident = format_ident!("{}", kind);
                        reducer_path = quote! { #flowgentra_ai_crate::core::reducer::#kind_ident };
                    }
                    Ok(())
                });
                break;
            }
        }

        // Update struct: all fields are Option<T>
        update_field_defs.push(quote! {
            #[serde(default, skip_serializing_if = "Option::is_none")]
            pub #name: Option<#ty>
        });
        update_defaults.push(quote! { #name: None });

        // Builder setter
        setters.push(quote! {
            pub fn #name(mut self, value: #ty) -> Self {
                self.#name = Some(value);
                self
            }
        });

        // apply_update arm: apply reducer if field is Some
        apply_arms.push(quote! {
            if let Some(value) = update.#name {
                <#reducer_path as #flowgentra_ai_crate::core::reducer::Reducer<#ty>>::merge(
                    &mut self.#name,
                    value,
                );
            }
        });
    }

    let expanded = quote! {
        /// Partial update struct — all fields are `Option<T>`.
        ///
        /// Nodes return this to indicate which fields changed.
        /// Only `Some` fields are applied via their configured reducer.
        #[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
        #vis struct #update_name {
            #(#update_field_defs,)*
        }

        impl #update_name {
            /// Create an empty update (all fields `None`).
            pub fn new() -> Self {
                Self {
                    #(#update_defaults,)*
                }
            }

            #(#setters)*
        }

        impl #flowgentra_ai_crate::core::state::State for #struct_name {
            type Update = #update_name;

            fn apply_update(&mut self, update: Self::Update) {
                #(#apply_arms)*
            }
        }
    };

    TokenStream::from(expanded)
}

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
/// - Takes `(&S, &Context)` parameters
/// - Returns `Result<S::Update>`
///
/// # Example
///
/// ```ignore
/// #[register_handler]
/// pub async fn validate_input(state: &MyState, ctx: &Context) -> Result<MyStateUpdate> {
///     Ok(MyStateUpdate::new().valid(!state.input.is_empty()))
/// }
/// ```
#[proc_macro_attribute]
pub fn register_handler(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let input = parse_macro_input!(item as ItemFn);

    let func_name = &input.sig.ident;
    let handler_name = func_name.to_string();

    let expanded = quote! {
        #input

        inventory::submit! {
            flowgentra_ai::core::agent::HandlerEntry::new(
                #handler_name,
                std::sync::Arc::new(|state, ctx| Box::pin(#func_name(state, ctx)))
            )
        }
    };

    TokenStream::from(expanded)
}

/// Attribute macro to turn an `async fn` into a `FunctionNode`.
///
/// Generates a factory function `{name}_node()` that returns
/// `Arc<FunctionNode<S, _>>` for use with `StateGraphBuilder::add_node`.
///
/// If the function has **no return type**, the macro infers it from the first
/// parameter (`state: &MyState` → `Result<MyStateUpdate>`) and wraps the last
/// expression in `Ok(...)` automatically.
///
/// # Example — minimal form (no return type, no `Ok`)
///
/// ```ignore
/// use flowgentra_ai_macros::node;
///
/// #[node]
/// async fn summarize(state: &MyState, _ctx: &Context) {
///     update! { summary: "done".into() }
/// }
///
/// // Use: graph.add_node("summarize", summarize_node())
/// ```
///
/// # Example — explicit form (full control)
///
/// ```ignore
/// #[node]
/// async fn summarize(state: &MyState, _ctx: &Context) -> Result<MyStateUpdate> {
///     Ok(update! { summary: "done".into() })
/// }
/// ```
#[proc_macro_attribute]
pub fn node(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let mut input = parse_macro_input!(item as ItemFn);
    let func_name = &input.sig.ident;
    let node_factory_name = format_ident!("{}_node", func_name);
    let node_name_str = func_name.to_string();

    // If no return type is written, infer it from `state: &MyState` → `Result<MyStateUpdate>`
    // and wrap the last expression in `Ok(...)`.
    if matches!(input.sig.output, syn::ReturnType::Default) {
        if let Some(update_ident) = extract_update_ident(&input.sig.inputs) {
            input.sig.output = syn::parse_quote! {
                -> flowgentra_ai::core::state_graph::error::Result<#update_ident>
            };

            // Wrap the tail expression (last stmt with no semicolon) in Ok(...)
            if let Some(syn::Stmt::Expr(expr, None)) = input.block.stmts.last_mut() {
                let inner = expr.clone();
                *expr = syn::parse_quote! { Ok(#inner) };
            }
        }
    }

    let expanded = quote! {
        #input

        /// Auto-generated node factory for use with `StateGraphBuilder::add_node`.
        pub fn #node_factory_name() -> std::sync::Arc<
            flowgentra_ai::core::state_graph::node::FunctionNode<
                _,
                impl Fn(&_, &flowgentra_ai::core::state::Context) -> std::pin::Pin<Box<dyn std::future::Future<Output = flowgentra_ai::core::state_graph::error::Result<_>> + Send>> + Send + Sync,
            >
        > {
            std::sync::Arc::new(flowgentra_ai::core::state_graph::node::FunctionNode::new(
                #node_name_str,
                |state, ctx| Box::pin(#func_name(state, ctx)),
            ))
        }
    };

    TokenStream::from(expanded)
}

/// Extract `MyStateUpdate` ident from `state: &MyState` (first function parameter).
fn extract_update_ident(
    inputs: &syn::punctuated::Punctuated<syn::FnArg, syn::token::Comma>,
) -> Option<syn::Ident> {
    let first = inputs.first()?;
    let syn::FnArg::Typed(pat_type) = first else {
        return None;
    };
    let syn::Type::Reference(type_ref) = pat_type.ty.as_ref() else {
        return None;
    };
    let syn::Type::Path(type_path) = type_ref.elem.as_ref() else {
        return None;
    };
    let state_ident = &type_path.path.segments.last()?.ident;
    Some(format_ident!("{}Update", state_ident))
}
