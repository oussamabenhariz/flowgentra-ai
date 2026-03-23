//! ## Extensibility
//!
//! - Implement `Reducer<T>` for your own types for custom merge logic.
//! - Register custom reducers with the macro using #[state(reducer = "...")].
//! - See example in docs and tests.
//! # Reducer system for flowgentra
//!
//! ## Error Handling
//!
//! - Reducers should never panic; always handle errors gracefully.
//! - If a reducer cannot merge, return a Result or log an error (future extension).
//! - Use clear error messages for debugging.
//! Reducer system for flowgentra

/// Trait for merging state field updates
#[allow(dead_code)]
pub trait Reducer<T> {
    fn merge(current: &mut T, update: T);
}

/// Overwrite reducer: replaces the field value
#[allow(dead_code)]
pub struct Overwrite;
impl<T> Reducer<T> for Overwrite {
    fn merge(current: &mut T, update: T) {
        *current = update;
    }
}

/// Append reducer: appends to a Vec
#[allow(dead_code)]
pub struct Append;
impl<T> Reducer<Vec<T>> for Append {
    fn merge(current: &mut Vec<T>, update: Vec<T>) {
        current.extend(update);
    }
}

/// Sum reducer: sums numeric types
#[allow(dead_code)]
pub struct Sum;
impl Reducer<i32> for Sum {
    fn merge(current: &mut i32, update: i32) {
        *current += update;
    }
}
impl Reducer<f64> for Sum {
    fn merge(current: &mut f64, update: f64) {
        *current += update;
    }
}

/// MergeMap reducer: merges HashMaps
use std::collections::HashMap;
#[allow(dead_code)]
pub struct MergeMap;
impl<K, V> Reducer<HashMap<K, V>> for MergeMap
where
    K: std::cmp::Eq + std::hash::Hash,
    V: Clone,
{
    fn merge(current: &mut HashMap<K, V>, update: HashMap<K, V>) {
        current.extend(update);
    }
}

// Users can implement Reducer<T> for custom strategies

// Example: Custom reducer for a user type
//
// ```rust
// use flowgentra_ai::core::reducer::Reducer;
// struct MyType(u32);
// struct MyCustomReducer;
    // impl Reducer<MyType> for MyCustomReducer {
    //     fn merge(current: &mut MyType, update: MyType) {
//         current.0 = current.0.max(update.0);
//     }
// }
// ```
