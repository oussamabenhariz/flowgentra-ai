//! Unit tests for state validation and error handling
use flowgentra_ai::prelude::*;
use flowgentra_ai::core::state::SharedState;
use serde_json::json;

#[test]
fn test_shared_state_basic_ops() {
    let state = SharedState::new(Default::default());
    state.set("input", json!("hello"));
    assert_eq!(state.get("input"), Some(json!("hello")));
    assert!(state.contains_key("input"));
    assert!(!state.is_empty());
}

#[test]
fn test_shared_state_remove() {
    let state = SharedState::new(Default::default());
    state.set("temp", json!(42));
    let removed = state.remove("temp");
    assert_eq!(removed, Some(json!(42)));
    assert!(state.is_empty());
}

#[test]
fn test_shared_state_merge() {
    let s1 = SharedState::new(Default::default());
    s1.set("a", json!(1));

    let s2 = SharedState::new(Default::default());
    s2.set("b", json!(2));

    s1.merge(s2).expect("merge failed");
    assert_eq!(s1.get("a"), Some(json!(1)));
    assert_eq!(s1.get("b"), Some(json!(2)));
}

#[test]
fn test_shared_state_deep_clone() {
    let s1 = SharedState::new(Default::default());
    s1.set("key", json!("val"));

    let s2 = s1.deep_clone();
    s2.set("key", json!("changed"));

    // Deep clone is independent
    assert_eq!(s1.get("key"), Some(json!("val")));
    assert_eq!(s2.get("key"), Some(json!("changed")));
}

#[test]
fn test_shared_state_serialization() {
    let state = SharedState::new(Default::default());
    state.set("x", json!(99));

    let json_str = state.to_json_string().expect("serialization failed");
    assert!(json_str.contains("99"));

    let restored = SharedState::from_json(serde_json::from_str(&json_str).unwrap()).unwrap();
    assert_eq!(restored.get("x"), Some(json!(99)));
}
