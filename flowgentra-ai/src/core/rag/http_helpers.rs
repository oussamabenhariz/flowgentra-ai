//! Shared HTTP utilities for vector store backends.
//!
//! Reduces boilerplate across all REST-based vector store implementations.

use reqwest::{Client, RequestBuilder};

use super::vector_db::VectorStoreError;

// ─── Header list ─────────────────────────────────────────────────────────────

/// A small, allocation-free list of `(name, value)` header pairs.
pub(crate) type Headers<'a> = &'a [(&'a str, &'a str)];

// ─── Core helpers ─────────────────────────────────────────────────────────────

/// Attach headers to a request builder.
#[allow(dead_code)]
fn with_headers(mut req: RequestBuilder, headers: Headers<'_>) -> RequestBuilder {
    for (k, v) in headers {
        req = req.header(*k, *v);
    }
    req
}

/// POST JSON body, check for HTTP success, and return parsed response JSON.
#[allow(dead_code)]
pub(crate) async fn post_json(
    client: &Client,
    url: &str,
    body: &serde_json::Value,
    headers: Headers<'_>,
    context: &str,
) -> Result<serde_json::Value, VectorStoreError> {
    let resp = with_headers(client.post(url).json(body), headers)
        .send()
        .await
        .map_err(|e| VectorStoreError::ConnectionError(format!("{context}: {e}")))?;

    let status = resp.status();
    if !status.is_success() {
        let text = resp.text().await.unwrap_or_default();
        return Err(VectorStoreError::ApiError(format!(
            "{context} failed ({status}): {text}"
        )));
    }

    resp.json()
        .await
        .map_err(|e| VectorStoreError::SerializationError(format!("{context} parse: {e}")))
}

/// POST JSON body, check for HTTP success, return nothing (fire-and-forget style).
#[allow(dead_code)]
pub(crate) async fn post_json_ok(
    client: &Client,
    url: &str,
    body: &serde_json::Value,
    headers: Headers<'_>,
    context: &str,
) -> Result<(), VectorStoreError> {
    let resp = with_headers(client.post(url).json(body), headers)
        .send()
        .await
        .map_err(|e| VectorStoreError::ConnectionError(format!("{context}: {e}")))?;

    let status = resp.status();
    if !status.is_success() {
        let text = resp.text().await.unwrap_or_default();
        return Err(VectorStoreError::ApiError(format!(
            "{context} failed ({status}): {text}"
        )));
    }
    Ok(())
}

/// PUT JSON body, check for HTTP success, return nothing.
#[allow(dead_code)]
pub(crate) async fn put_json_ok(
    client: &Client,
    url: &str,
    body: &serde_json::Value,
    headers: Headers<'_>,
    context: &str,
) -> Result<(), VectorStoreError> {
    let resp = with_headers(client.put(url).json(body), headers)
        .send()
        .await
        .map_err(|e| VectorStoreError::ConnectionError(format!("{context}: {e}")))?;

    let status = resp.status();
    if !status.is_success() {
        let text = resp.text().await.unwrap_or_default();
        return Err(VectorStoreError::ApiError(format!(
            "{context} failed ({status}): {text}"
        )));
    }
    Ok(())
}

/// GET, check for HTTP success, return parsed JSON. Returns `None` on 404.
#[allow(dead_code)]
pub(crate) async fn get_json(
    client: &Client,
    url: &str,
    headers: Headers<'_>,
    context: &str,
) -> Result<Option<serde_json::Value>, VectorStoreError> {
    let resp = with_headers(client.get(url), headers)
        .send()
        .await
        .map_err(|e| VectorStoreError::ConnectionError(format!("{context}: {e}")))?;

    let status = resp.status();
    if status == reqwest::StatusCode::NOT_FOUND {
        return Ok(None);
    }
    if !status.is_success() {
        let text = resp.text().await.unwrap_or_default();
        return Err(VectorStoreError::ApiError(format!(
            "{context} failed ({status}): {text}"
        )));
    }

    let val = resp
        .json()
        .await
        .map_err(|e| VectorStoreError::SerializationError(format!("{context} parse: {e}")))?;
    Ok(Some(val))
}

/// DELETE, ignore 404, fail on other non-success statuses.
#[allow(dead_code)]
pub(crate) async fn delete_ok(
    client: &Client,
    url: &str,
    headers: Headers<'_>,
    context: &str,
) -> Result<(), VectorStoreError> {
    let resp = with_headers(client.delete(url), headers)
        .send()
        .await
        .map_err(|e| VectorStoreError::ConnectionError(format!("{context}: {e}")))?;

    let status = resp.status();
    if !status.is_success() && status != reqwest::StatusCode::NOT_FOUND {
        let text = resp.text().await.unwrap_or_default();
        return Err(VectorStoreError::ApiError(format!(
            "{context} failed ({status}): {text}"
        )));
    }
    Ok(())
}

// ─── Env-var substitution ────────────────────────────────────────────────────

/// Replace `${VAR_NAME}` tokens in `s` with the corresponding environment variable.
///
/// Unknown variables are left as-is so the caller can detect misconfiguration.
pub fn resolve_env_vars(s: &str) -> String {
    let mut result = s.to_string();
    // Simple iterative replacement — handles multiple tokens in one string.
    loop {
        let Some(start) = result.find("${") else {
            break;
        };
        let Some(end) = result[start..].find('}') else {
            break;
        };
        let end = start + end;
        let var_name = &result[start + 2..end];
        if let Ok(val) = std::env::var(var_name) {
            result = format!("{}{}{}", &result[..start], val, &result[end + 1..]);
        } else {
            // No such variable — leave the token, advance past it to avoid a loop.
            break;
        }
    }
    result
}

/// Apply `resolve_env_vars` to an `Option<String>`.
pub fn resolve_opt(opt: Option<&str>) -> Option<String> {
    opt.map(|s| resolve_env_vars(s))
}
