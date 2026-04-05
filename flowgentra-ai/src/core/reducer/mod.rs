//! # Typed Reducer System
//!
//! Reducers define how state fields are merged when applying partial updates.
//! Each field in a `#[derive(State)]` struct can have a reducer annotation:
//!
//! ```ignore
//! #[derive(State, Clone, Serialize, Deserialize)]
//! struct MyState {
//!     name: String,                    // default: Overwrite
//!
//!     #[reducer(Append)]
//!     messages: Vec<Message>,          // new items are appended
//!
//!     #[reducer(Sum)]
//!     retry_count: i32,                // values are summed
//!
//!     #[reducer(MergeMap)]
//!     metadata: HashMap<String, String>, // maps are merged
//! }
//! ```

use std::collections::HashMap;

/// Trait for merging a state field with an update value.
///
/// Implement this for custom merge strategies.
pub trait Reducer<T> {
    fn merge(current: &mut T, update: T);
}

/// Overwrite reducer (default): replaces the field value entirely.
pub struct Overwrite;
impl<T> Reducer<T> for Overwrite {
    fn merge(current: &mut T, update: T) {
        *current = update;
    }
}

/// Append reducer: extends `Vec<T>` with new items.
pub struct Append;
impl<T> Reducer<Vec<T>> for Append {
    fn merge(current: &mut Vec<T>, update: Vec<T>) {
        current.extend(update);
    }
}

/// Sum reducer: adds numeric values.
pub struct Sum;
impl Reducer<i32> for Sum {
    fn merge(current: &mut i32, update: i32) {
        *current += update;
    }
}
impl Reducer<i64> for Sum {
    fn merge(current: &mut i64, update: i64) {
        *current += update;
    }
}
impl Reducer<f64> for Sum {
    fn merge(current: &mut f64, update: f64) {
        *current += update;
    }
}
impl Reducer<usize> for Sum {
    fn merge(current: &mut usize, update: usize) {
        *current += update;
    }
}

/// MergeMap reducer: merges HashMap entries (update keys overwrite existing).
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

/// Max reducer: keeps the maximum numeric value.
pub struct Max;
impl Reducer<i32> for Max {
    fn merge(current: &mut i32, update: i32) {
        if update > *current {
            *current = update;
        }
    }
}
impl Reducer<i64> for Max {
    fn merge(current: &mut i64, update: i64) {
        if update > *current {
            *current = update;
        }
    }
}
impl Reducer<f64> for Max {
    fn merge(current: &mut f64, update: f64) {
        if update > *current {
            *current = update;
        }
    }
}

/// Min reducer: keeps the minimum numeric value.
pub struct Min;
impl Reducer<i32> for Min {
    fn merge(current: &mut i32, update: i32) {
        if update < *current {
            *current = update;
        }
    }
}
impl Reducer<i64> for Min {
    fn merge(current: &mut i64, update: i64) {
        if update < *current {
            *current = update;
        }
    }
}
impl Reducer<f64> for Min {
    fn merge(current: &mut f64, update: f64) {
        if update < *current {
            *current = update;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_overwrite() {
        let mut val = 10;
        Overwrite::merge(&mut val, 20);
        assert_eq!(val, 20);
    }

    #[test]
    fn test_append() {
        let mut val = vec![1, 2];
        Append::merge(&mut val, vec![3, 4]);
        assert_eq!(val, vec![1, 2, 3, 4]);
    }

    #[test]
    fn test_sum() {
        let mut val = 10i32;
        Sum::merge(&mut val, 5);
        assert_eq!(val, 15);
    }

    #[test]
    fn test_merge_map() {
        let mut val: HashMap<String, String> = HashMap::new();
        val.insert("a".into(), "1".into());

        let mut update = HashMap::new();
        update.insert("b".into(), "2".into());
        update.insert("a".into(), "overwritten".into());

        MergeMap::merge(&mut val, update);
        assert_eq!(val.get("a").unwrap(), "overwritten");
        assert_eq!(val.get("b").unwrap(), "2");
    }

    #[test]
    fn test_max() {
        let mut val = 10i32;
        Max::merge(&mut val, 5);
        assert_eq!(val, 10);
        Max::merge(&mut val, 20);
        assert_eq!(val, 20);
    }

    #[test]
    fn test_min() {
        let mut val = 10i32;
        Min::merge(&mut val, 15);
        assert_eq!(val, 10);
        Min::merge(&mut val, 5);
        assert_eq!(val, 5);
    }
}
