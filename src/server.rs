//! OpenAI-compatible HTTP server: `POST /v1/chat/completions` (plain and SSE
//! streaming) and `GET /v1/models`.
//!
//! The engine is CPU-bound and already uses every core through rayon, so
//! requests are served one at a time behind a single [`Session`]. Sharing one
//! session also gives chat clients prefix reuse for free: they re-send the
//! whole conversation each turn, and only the new suffix is computed.

use std::net::SocketAddr;
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::Result;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::sse::{Event, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::{Deserialize, Serialize};
use serde_json::json;
use tokio::sync::mpsc;
use tokio_stream::StreamExt;
use tokio_stream::wrappers::UnboundedReceiverStream;

use kiln::chat::{Message, Role};
use kiln::generate::{Session, StopReason};
use kiln::sampler::{Sampler, SamplingParams};

use crate::Engine;

struct AppState {
    engine: Engine,
    session: Mutex<Session>,
    model_id: String,
    defaults: SamplingParams,
    default_max_tokens: usize,
}

#[derive(Deserialize)]
struct ChatRequest {
    messages: Vec<ChatMessage>,
    #[serde(default)]
    stream: bool,
    // `max_completion_tokens` is the newer name in the OpenAI API.
    #[serde(alias = "max_completion_tokens")]
    max_tokens: Option<usize>,
    temperature: Option<f32>,
    top_p: Option<f32>,
    seed: Option<u64>,
}

#[derive(Deserialize, Serialize)]
struct ChatMessage {
    role: String,
    content: String,
}

pub fn serve(
    engine: Engine,
    addr: SocketAddr,
    ctx: usize,
    defaults: SamplingParams,
    default_max_tokens: usize,
    model_id: String,
) -> Result<()> {
    let session = Mutex::new(Session::new(&engine.model, ctx));
    let state = Arc::new(AppState {
        engine,
        session,
        model_id,
        defaults,
        default_max_tokens,
    });
    let app = Router::new()
        .route("/v1/models", get(models))
        .route("/v1/chat/completions", post(chat_completions))
        .with_state(state);

    tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2) // HTTP only; generation runs on the blocking pool + rayon
        .enable_all()
        .build()?
        .block_on(async {
            let listener = tokio::net::TcpListener::bind(addr).await?;
            eprintln!("serving OpenAI-compatible API on http://{addr}/v1");
            axum::serve(listener, app)
                .with_graceful_shutdown(async {
                    let _ = tokio::signal::ctrl_c().await;
                })
                .await?;
            Ok(())
        })
}

async fn models(State(state): State<Arc<AppState>>) -> Json<serde_json::Value> {
    Json(json!({
        "object": "list",
        "data": [{ "id": state.model_id, "object": "model", "owned_by": "kiln" }],
    }))
}

fn error(status: StatusCode, message: impl Into<String>) -> Response {
    let body = json!({ "error": { "message": message.into(), "type": "invalid_request_error" } });
    (status, Json(body)).into_response()
}

fn parse_role(role: &str) -> Option<Role> {
    match role {
        "system" | "developer" => Some(Role::System),
        "user" => Some(Role::User),
        "assistant" => Some(Role::Assistant),
        _ => None,
    }
}

fn finish_reason(stop: StopReason) -> &'static str {
    match stop {
        StopReason::StopToken => "stop",
        StopReason::MaxTokens | StopReason::ContextFull => "length",
    }
}

fn now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_secs())
}

/// What the generation thread reports back to the HTTP handler.
enum Update {
    Text(String),
    Done {
        stop: StopReason,
        prompt_tokens: usize,
        completion_tokens: usize,
    },
    Failed(String),
}

async fn chat_completions(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ChatRequest>,
) -> Response {
    let mut messages = Vec::with_capacity(req.messages.len());
    for m in &req.messages {
        let Some(role) = parse_role(&m.role) else {
            return error(
                StatusCode::BAD_REQUEST,
                format!("unsupported role {:?}", m.role),
            );
        };
        messages.push(Message::new(role, m.content.clone()));
    }
    if messages.is_empty() {
        return error(StatusCode::BAD_REQUEST, "messages must not be empty");
    }

    let prompt = state
        .engine
        .tokenizer
        .encode(&state.engine.template.render(&messages, true));
    let capacity = state.session.lock().map_or(0, |s| s.capacity());
    if prompt.len() >= capacity {
        return error(
            StatusCode::BAD_REQUEST,
            format!(
                "prompt is {} tokens; the context holds {capacity}",
                prompt.len()
            ),
        );
    }

    let params = SamplingParams {
        temperature: req.temperature.unwrap_or(state.defaults.temperature),
        top_p: req.top_p.unwrap_or(state.defaults.top_p),
        seed: req.seed.unwrap_or(state.defaults.seed),
        ..state.defaults
    };
    let max_tokens = req.max_tokens.unwrap_or(state.default_max_tokens).max(1);

    let (tx, rx) = mpsc::unbounded_channel();
    let worker = Arc::clone(&state);
    tokio::task::spawn_blocking(move || {
        let state = worker;
        let mut session = match state.session.lock() {
            Ok(s) => s,
            // A panic mid-generation leaves the cache half-written; start clean.
            Err(poisoned) => {
                let mut s = poisoned.into_inner();
                *s = Session::new(&state.engine.model, s.capacity());
                s
            }
        };
        let mut sampler = Sampler::new(params);
        let mut decoder = state.engine.tokenizer.stream_decoder(true);
        let result = session.generate(
            &state.engine.model,
            &prompt,
            max_tokens,
            &state.engine.stop_tokens,
            &mut sampler,
            |id| {
                let piece = decoder.push(id);
                if !piece.is_empty() {
                    // The client may have gone away; generation finishes anyway
                    // so the session's cache stays consistent.
                    let _ = tx.send(Update::Text(piece));
                }
            },
        );
        let tail = decoder.finish();
        if !tail.is_empty() {
            let _ = tx.send(Update::Text(tail));
        }
        let _ = tx.send(match result {
            Ok(g) => {
                let s = &g.stats;
                eprintln!(
                    "[prompt {} tok, {} reused from cache | prefill {:.1} tok/s | decode {} tok at {:.1} tok/s]",
                    prompt.len(),
                    prompt.len() - s.prefill_tokens,
                    s.prefill_tok_per_s(),
                    s.decode_tokens,
                    s.decode_tok_per_s(),
                );
                Update::Done {
                    stop: g.stop,
                    prompt_tokens: prompt.len(),
                    completion_tokens: g.tokens.len(),
                }
            }
            Err(e) => Update::Failed(format!("{e:#}")),
        });
    });

    // Ids only need to be distinct in logs, and requests are serialized.
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_nanos());
    let id = format!("chatcmpl-{nanos:x}");
    if req.stream {
        stream_response(rx, id, state.model_id.clone())
    } else {
        collect_response(rx, id, state.model_id.clone()).await
    }
}

async fn collect_response(
    mut rx: mpsc::UnboundedReceiver<Update>,
    id: String,
    model: String,
) -> Response {
    let mut text = String::new();
    while let Some(update) = rx.recv().await {
        match update {
            Update::Text(piece) => text.push_str(&piece),
            Update::Failed(e) => return error(StatusCode::INTERNAL_SERVER_ERROR, e),
            Update::Done {
                stop,
                prompt_tokens,
                completion_tokens,
            } => {
                return Json(json!({
                    "id": id,
                    "object": "chat.completion",
                    "created": now(),
                    "model": model,
                    "choices": [{
                        "index": 0,
                        "message": { "role": "assistant", "content": text.trim_end() },
                        "finish_reason": finish_reason(stop),
                    }],
                    "usage": {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": prompt_tokens + completion_tokens,
                    },
                }))
                .into_response();
            }
        }
    }
    error(
        StatusCode::INTERNAL_SERVER_ERROR,
        "generation ended unexpectedly",
    )
}

fn stream_response(rx: mpsc::UnboundedReceiver<Update>, id: String, model: String) -> Response {
    let created = now();
    let chunk = move |delta: serde_json::Value, finish: Option<&str>| {
        json!({
            "id": id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{ "index": 0, "delta": delta, "finish_reason": finish }],
        })
    };
    let first = Event::default().data(chunk(json!({ "role": "assistant" }), None).to_string());
    let events = UnboundedReceiverStream::new(rx).map(move |update| match update {
        Update::Text(piece) => {
            Event::default().data(chunk(json!({ "content": piece }), None).to_string())
        }
        Update::Done { stop, .. } => {
            Event::default().data(chunk(json!({}), Some(finish_reason(stop))).to_string())
        }
        Update::Failed(e) => {
            Event::default().data(json!({ "error": { "message": e } }).to_string())
        }
    });
    // The worker drops its sender after Done/Failed, which ends `events`.
    let stream = tokio_stream::once(first)
        .chain(events)
        .chain(tokio_stream::once(Event::default().data("[DONE]")))
        .map(Ok::<_, std::convert::Infallible>);
    Sse::new(stream).into_response()
}
