//! Example usage of the new typed state engine

use crate::core::macros::State;
use crate::core::runtime::{merge_state, Reducers__AgentState};

#[derive(State, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct AgentState {
    pub input: String,
    #[state(reducer = "append")]
    pub messages: Vec<String>,
    pub result: Option<String>,
}

// Example node handler: returns a partial state update
pub async fn summarize(state: &AgentState) -> StateUpdate__AgentState {
    StateUpdate__AgentState::new().set_result(Some("done".to_string()))
}

// Example: merging updates from multiple nodes
pub fn merge_example() {
    let state = AgentState {
        input: "hello".to_string(),
        messages: vec![],
        result: None,
    };
    let update1 = StateUpdate__AgentState::new().set_messages(vec!["msg1".to_string()]);
    let update2 = StateUpdate__AgentState::new().set_result(Some("finished".to_string()));
    let reducers = Reducers__AgentState::new();
    let new_state = merge_state(&state, vec![update1, update2], &reducers);
    println!("Merged state: {:?}", new_state);
}
