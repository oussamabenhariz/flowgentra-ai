//! Baseline benchmarks for the hot paths identified in AUDIT.md (F-20):
//! graph compile time, per-node dispatch overhead, and DynState get/set/clone.
//!
//! Run with: `cargo bench -p flowgentra-ai`
//! Results are tracked in BENCHMARKS.md — update it when these move.

use criterion::{criterion_group, criterion_main, Criterion};
use std::sync::Arc;

use flowgentra_ai::core::llm::Message;
use flowgentra_ai::core::state::{Context, DynState};
use flowgentra_ai::core::state_graph::message_graph::{MessageState, MessageStateUpdate};
use flowgentra_ai::core::state_graph::node::{FunctionNode, Node};
use flowgentra_ai::core::state_graph::StateGraph;

fn noop_node(name: &str) -> Arc<dyn Node<MessageState>> {
    Arc::new(FunctionNode::new(
        name.to_string(),
        |_state: &MessageState, _ctx: &Context| {
            Box::pin(async move { Ok(MessageStateUpdate::default()) })
        },
    ))
}

fn bench_graph_compile(c: &mut Criterion) {
    c.bench_function("graph_compile_10_nodes", |b| {
        b.iter(|| {
            let mut builder = StateGraph::<MessageState>::builder();
            for i in 0..10 {
                builder = builder.add_node(format!("n{i}"), noop_node(&format!("n{i}")));
            }
            builder = builder.set_entry_point("n0");
            for i in 0..9 {
                builder = builder.add_edge(format!("n{i}"), format!("n{}", i + 1));
            }
            builder = builder.add_edge("n9", "__end__");
            std::hint::black_box(builder.compile().unwrap())
        })
    });
}

fn bench_node_dispatch(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();

    // 10 sequential no-op nodes: measures per-node dispatch overhead
    // (middleware, checkpointing to memory, event emission, state clone).
    let mut builder = StateGraph::<MessageState>::builder();
    for i in 0..10 {
        builder = builder.add_node(format!("n{i}"), noop_node(&format!("n{i}")));
    }
    builder = builder.set_entry_point("n0");
    for i in 0..9 {
        builder = builder.add_edge(format!("n{i}"), format!("n{}", i + 1));
    }
    builder = builder.add_edge("n9", "__end__");
    let graph = builder.compile().unwrap();

    c.bench_function("invoke_10_noop_nodes", |b| {
        b.iter(|| {
            let state = MessageState::new(vec![Message::user("bench")]);
            rt.block_on(std::hint::black_box(&graph).invoke(state))
                .unwrap()
        })
    });
}

fn bench_dyn_state(c: &mut Criterion) {
    c.bench_function("dyn_state_set_get", |b| {
        let state = DynState::new();
        b.iter(|| {
            state.set_raw("key", serde_json::json!({"a": 1, "b": [1, 2, 3]}));
            std::hint::black_box(state.get("key"))
        })
    });

    c.bench_function("dyn_state_clone_20_keys", |b| {
        let state = DynState::new();
        for i in 0..20 {
            state.set_raw(format!("k{i}"), serde_json::json!(i));
        }
        b.iter(|| std::hint::black_box(state.deep_clone()))
    });
}

criterion_group!(
    benches,
    bench_graph_compile,
    bench_node_dispatch,
    bench_dyn_state
);
criterion_main!(benches);
