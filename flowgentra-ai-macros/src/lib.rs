use quote::{format_ident, quote};
use syn::{parse_macro_input, DeriveInput, Data, Fields};
// use syn::parse::Parse;
use convert_case::Casing;

/// Procedural macro to derive the State trait and generate StateUpdate, reducers, and setters
#[proc_macro_derive(State, attributes(state))]
pub fn derive_state(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let struct_name = &input.ident;
    let vis = &input.vis;

    // Collect fields and their types
    let fields = match &input.data {
        Data::Struct(data_struct) => match &data_struct.fields {
            Fields::Named(fields_named) => &fields_named.named,
            _ => panic!("State must be a struct with named fields"),
        },
        _ => panic!("State must be a struct"),
    };

    let mut update_fields = Vec::new();
    let mut update_setters = Vec::new();
    let mut reducer_fields = Vec::new();
    let mut reducer_inits = Vec::new();

    for field in fields {
        let field_name = field.ident.as_ref().unwrap();
        let field_ty = &field.ty;
        // Find reducer attribute
        let mut reducer = quote! { ::flowgentra_ai::core::reducer::Overwrite };
        for attr in &field.attrs {
            if attr.path().is_ident("state") {
                let _ = attr.parse_nested_meta(|meta| {
                    if meta.path.is_ident("reducer") {
                        if let Ok(buf) = meta.value() {
                            if let Ok(litstr) = buf.parse::<syn::LitStr>() {
                                let reducer_ident = format_ident!("{}", litstr.value().to_case(convert_case::Case::UpperCamel));
                                reducer = quote! { ::flowgentra_ai::core::reducer::#reducer_ident };
                            }
                        }
                    }
                    Ok(())
                });
            }
        }
        update_fields.push(quote! { pub #field_name: Option<#field_ty> });
        let setter_name = format_ident!("set_{}", field_name);
        update_setters.push(quote! {
            pub fn #setter_name(mut self, value: #field_ty) -> Self {
                self.#field_name = Some(value);
                self
            }
        });
        reducer_fields.push(quote! { pub #field_name: fn(&mut #field_ty, #field_ty) });
        reducer_inits.push(quote! { #field_name: <#reducer as ::flowgentra_ai::core::reducer::Reducer<#field_ty>>::merge });
    }

    let update_struct = format_ident!("StateUpdate__{}", struct_name);
    let reducers_struct = format_ident!("Reducers__{}", struct_name);

    let expanded = quote! {
        impl ::flowgentra_ai::core::state::State for #struct_name {}

        #[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
        #vis struct #update_struct {
            #(#update_fields,)*
        }

        impl #update_struct {
            pub fn new() -> Self {
                Self { #( #update_fields: None, )* }
            }
            #(#update_setters)*
        }

        #vis struct #reducers_struct {
            #(#reducer_fields,)*
        }

        impl #reducers_struct {
            pub fn new() -> Self {
                Self {
                    #(#reducer_inits,)*
                }
            }
        }
    };
    TokenStream::from(expanded)
}
// # FlowgentraAI Handler Registration Macro
//
// Provides the `#[register_handler]` attribute for automatic handler registration.
//
// ## Quick Start
//
// ```ignore
// use flowgentra_ai::prelude::*;
//
// #[register_handler]
// pub async fn my_handler(mut state: State) -> Result<State> {
//     state.set("output", json!("processed"));
//     Ok(state)
// }
//
// // Create agent - handler auto-discovered!
// let agent = from_config_path("config.yaml")?;
// ```

use proc_macro::TokenStream;
use syn::ItemFn;

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
            flowgentra_ai::core::agent::HandlerEntry::new(
                #handler_name,
                std::sync::Arc::new(|state| Box::pin(#func_name(state)))
            )
        }
    };

    TokenStream::from(expanded)
}

/// Attribute macro to turn an `async fn(&S) -> Result<S>` into a `FunctionNode`.
///
/// Generates a factory function `{name}_node()` that returns
/// `Arc<FunctionNode<S, _>>` for use with `StateGraphBuilder::add_node`.
///
/// # Example
///
/// ```ignore
/// use flowgentra_ai_macros::node;
///
/// #[node]
/// async fn summarize(state: &PlainState) -> flowgentra_ai::core::state_graph::error::Result<PlainState> {
///     let mut s = state.clone();
///     s.set("summary", serde_json::json!("done"));
///     Ok(s)
/// }
///
/// // Use: graph.add_node("summarize", summarize_node())
/// ```
#[proc_macro_attribute]
pub fn node(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let input = parse_macro_input!(item as ItemFn);
    let func_name = &input.sig.ident;
    let node_factory_name = format_ident!("{}_node", func_name);
    let node_name_str = func_name.to_string();

    let expanded = quote! {
        #input

        /// Auto-generated node factory for use with `StateGraphBuilder::add_node`.
        pub fn #node_factory_name() -> std::sync::Arc<
            flowgentra_ai::core::state_graph::node::FunctionNode<
                _,
                impl Fn(&_) -> std::pin::Pin<Box<dyn std::future::Future<Output = flowgentra_ai::core::state_graph::error::Result<_>> + Send>> + Send + Sync,
            >
        > {
            std::sync::Arc::new(flowgentra_ai::core::state_graph::node::FunctionNode::new(
                #node_name_str,
                |state| Box::pin(#func_name(state)),
            ))
        }
    };

    TokenStream::from(expanded)
}
