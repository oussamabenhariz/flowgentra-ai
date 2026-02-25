//! # Tracing UI Server
//!
//! Built-in web UI for inspecting execution traces.
//!
//! ```ignore
//! use erenflow_ai::core::observability::{ObservabilityMiddleware, TracingUIServer};
//!
//! // After execution, get trace from middleware and serve UI
//! let trace = observability_mw.trace().await;
//! TracingUIServer::serve_on_port(7777, vec![trace]).await?;
//! ```

use super::trace::ExecutionTrace;
use axum::{
    extract::State,
    response::{Html, IntoResponse, Json},
    routing::get,
    Router,
};
use std::sync::Arc;
use tokio::sync::RwLock;

type TraceState = Arc<RwLock<Vec<ExecutionTrace>>>;

/// Serves the tracing UI for inspecting execution traces
pub struct TracingUIServer;

impl TracingUIServer {
    /// Serve the tracing UI on the given port with the provided traces
    pub async fn serve_on_port(
        port: u16,
        traces: Vec<ExecutionTrace>,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let state: TraceState = Arc::new(RwLock::new(traces));

        let app = Router::new()
            .route("/", get(Self::index_handler))
            .route("/api/traces", get(Self::traces_handler))
            .with_state(state);

        let listener = tokio::net::TcpListener::bind(format!("127.0.0.1:{}", port)).await?;
        tracing::info!("Tracing UI available at http://127.0.0.1:{}/", port);
        axum::serve(listener, app).await?;
        Ok(())
    }

    async fn index_handler() -> Html<&'static str> {
        Html(include_str!("ui/index.html"))
    }

    async fn traces_handler(State(state): State<TraceState>) -> impl IntoResponse {
        let traces = state.read().await.clone();
        Json(traces)
    }
}
