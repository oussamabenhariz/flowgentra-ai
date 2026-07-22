//! Minimal local dev server — live graph viewer + execution event feed.
//!
//! Not LangGraph Studio: no state editing, no time-travel, no hosted
//! anything. It's the smallest thing that's actually useful: point a
//! browser at a compiled graph and watch `invoke()` calls happen in
//! real time, from the same process that's running them.
//!
//! ```ignore
//! let graph = StateGraph::<MyState>::builder() /* ... */ .compile()?;
//! let handle = graph.serve_dev(7878);
//! println!("dev viewer: {}", handle.url());
//! // ... call graph.invoke(...) from anywhere; the browser updates live.
//! ```
//!
//! `GET /` serves a single self-contained HTML page (embedded via
//! `include_str!`, no CDN/external assets — works offline). `GET /graph`
//! returns `{"nodes": [...], "entry_point": "..."}`. `GET /events` is a
//! `text/event-stream` of `ExecutionEvent` JSON, one event per SSE message.

use axum::{
    extract::State,
    response::{sse::Event, Html, IntoResponse, Json, Sse},
    routing::get,
    Router,
};
use futures::stream::{self, Stream};
use serde::Serialize;
use std::convert::Infallible;
use std::sync::Arc;
use tokio::sync::broadcast;

use super::events::ExecutionEvent;

#[derive(Clone, Serialize)]
struct GraphSummary {
    nodes: Vec<String>,
    entry_point: String,
}

#[derive(Clone)]
struct DevServerState {
    summary: Arc<GraphSummary>,
    events_tx: broadcast::Sender<ExecutionEvent>,
}

/// Handle to a running dev server. Dropping it does not stop the server —
/// it keeps running for the process lifetime; call [`DevServerHandle::shutdown`]
/// to stop it explicitly.
pub struct DevServerHandle {
    port: u16,
    forward_task: tokio::task::JoinHandle<()>,
    server_task: tokio::task::JoinHandle<()>,
}

impl DevServerHandle {
    /// The URL to open in a browser.
    pub fn url(&self) -> String {
        format!("http://127.0.0.1:{}/", self.port)
    }

    /// Stop the server and the event-forwarding task.
    pub fn shutdown(self) {
        self.forward_task.abort();
        self.server_task.abort();
    }
}

/// Start the dev server in the background. Returns immediately; binding
/// failures (e.g. port already in use) surface as a task panic visible in
/// logs rather than a `Result` here, since the caller has already moved on
/// by the time `axum::serve` would fail — check `handle.url()` reachability
/// if you need to confirm startup.
///
/// `node_names`/`entry_point` describe the graph structure (shown once,
/// static — a graph's shape doesn't change after `compile()`).
/// `event_source` is a fresh subscription (e.g. from `EventBroadcaster::subscribe()`)
/// that will be forwarded to every connected browser tab.
pub fn start(
    port: u16,
    node_names: Vec<String>,
    entry_point: String,
    mut event_source: broadcast::Receiver<ExecutionEvent>,
) -> DevServerHandle {
    let (events_tx, _) = broadcast::channel::<ExecutionEvent>(256);
    let fanout_tx = events_tx.clone();

    // Forward the graph's single broadcaster subscription to our own
    // channel, which supports multiple SSE subscribers (one per browser tab)
    // without each of them needing a separate subscription on the graph's
    // (finite-capacity) broadcaster.
    let forward_task = tokio::spawn(async move {
        while let Ok(event) = event_source.recv().await {
            let _ = fanout_tx.send(event);
        }
    });

    let state = DevServerState {
        summary: Arc::new(GraphSummary {
            nodes: node_names,
            entry_point,
        }),
        events_tx,
    };

    let app = Router::new()
        .route("/", get(index_handler))
        .route("/graph", get(graph_handler))
        .route("/events", get(events_handler))
        .with_state(state);

    let server_task = tokio::spawn(async move {
        let listener = match tokio::net::TcpListener::bind(("127.0.0.1", port)).await {
            Ok(l) => l,
            Err(e) => {
                tracing::error!(port, error = %e, "flowgentra dev server: bind failed");
                return;
            }
        };
        tracing::info!(port, "flowgentra dev server listening");
        if let Err(e) = axum::serve(listener, app).await {
            tracing::error!(error = %e, "flowgentra dev server: serve failed");
        }
    });

    DevServerHandle {
        port,
        forward_task,
        server_task,
    }
}

async fn index_handler() -> Html<&'static str> {
    Html(include_str!("dev_server/index.html"))
}

async fn graph_handler(State(state): State<DevServerState>) -> impl IntoResponse {
    Json((*state.summary).clone())
}

async fn events_handler(
    State(state): State<DevServerState>,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    let rx = state.events_tx.subscribe();
    let stream = stream::unfold(rx, |mut rx| async move {
        loop {
            match rx.recv().await {
                Ok(event) => match serde_json::to_string(&event) {
                    Ok(json) => return Some((Ok(Event::default().data(json)), rx)),
                    Err(_) => continue, // shouldn't happen; ExecutionEvent always serializes
                },
                // A slow client fell behind the channel's ring buffer — skip
                // the events it missed and keep streaming rather than closing.
                Err(broadcast::error::RecvError::Lagged(_)) => continue,
                Err(broadcast::error::RecvError::Closed) => return None,
            }
        }
    });
    Sse::new(stream).keep_alive(axum::response::sse::KeepAlive::default())
}
