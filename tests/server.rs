//! End-to-end test of `kiln serve`: starts the real binary and speaks HTTP to it.

mod common;

use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

/// Kills the server even when an assertion fails.
struct Server {
    child: Child,
    port: u16,
}

impl Drop for Server {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

fn start() -> Server {
    // Grab a free port, release it, and hand it to the server.
    let port = TcpListener::bind("127.0.0.1:0")
        .and_then(|l| l.local_addr())
        .expect("free port")
        .port();
    let child = Command::new(env!("CARGO_BIN_EXE_kiln"))
        .args([
            "serve",
            "--quant",
            "q8",
            "--addr",
            &format!("127.0.0.1:{port}"),
        ])
        .arg("--model")
        .arg(common::model_dir())
        .stderr(Stdio::null())
        .spawn()
        .expect("spawn kiln serve");
    let server = Server { child, port };
    let deadline = Instant::now() + Duration::from_secs(60);
    while TcpStream::connect(("127.0.0.1", port)).is_err() {
        assert!(Instant::now() < deadline, "server did not start");
        std::thread::sleep(Duration::from_millis(100));
    }
    server
}

/// Minimal HTTP/1.1 client: returns (status, body). The server closes the
/// connection because we ask it to, so the body is everything after the headers.
fn request(port: u16, method: &str, path: &str, body: &str) -> (u16, String) {
    let mut stream = TcpStream::connect(("127.0.0.1", port)).expect("connect");
    write!(
        stream,
        "{method} {path} HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\n\
         Content-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    )
    .expect("write request");
    let mut reader = BufReader::new(stream);
    let mut status_line = String::new();
    reader.read_line(&mut status_line).expect("status line");
    let status = status_line
        .split_whitespace()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .expect("status code");
    let mut line = String::new();
    while reader.read_line(&mut line).expect("header") > 2 {
        line.clear();
    }
    let mut rest = String::new();
    reader.read_to_string(&mut rest).expect("body");
    (status, rest)
}

fn json(body: &str) -> serde_json::Value {
    // Non-streaming responses may arrive chunked; the JSON object is the part between braces.
    let start = body.find('{').expect("json start");
    let end = body.rfind('}').expect("json end");
    serde_json::from_str(&body[start..=end]).expect("valid json")
}

#[test]
fn serves_openai_compatible_chat_completions() {
    let server = start();

    let (status, body) = request(server.port, "GET", "/v1/models", "");
    assert_eq!(status, 200);
    assert_eq!(json(&body)["data"][0]["object"], "model");

    let req = r#"{"messages":[{"role":"user","content":"Name a planet."}],"max_tokens":8,"temperature":0}"#;
    let (status, body) = request(server.port, "POST", "/v1/chat/completions", req);
    assert_eq!(status, 200);
    let resp = json(&body);
    assert_eq!(resp["object"], "chat.completion");
    assert_eq!(resp["choices"][0]["message"]["role"], "assistant");
    assert!(
        !resp["choices"][0]["message"]["content"]
            .as_str()
            .unwrap()
            .is_empty()
    );
    assert_eq!(resp["usage"]["completion_tokens"], 8);
    assert_eq!(resp["choices"][0]["finish_reason"], "length");

    let req = r#"{"messages":[{"role":"user","content":"Hi"}],"max_tokens":4,"stream":true}"#;
    let (status, body) = request(server.port, "POST", "/v1/chat/completions", req);
    assert_eq!(status, 200);
    let chunks: Vec<&str> = body
        .lines()
        .filter_map(|l| l.strip_prefix("data: "))
        .collect();
    assert_eq!(chunks.last(), Some(&"[DONE]"));
    assert!(chunks.iter().any(|c| c.contains(r#""content""#)));
    assert!(
        chunks
            .iter()
            .any(|c| c.contains(r#""finish_reason":"length""#))
    );

    let (status, body) = request(
        server.port,
        "POST",
        "/v1/chat/completions",
        r#"{"messages":[{"role":"robot","content":"x"}]}"#,
    );
    assert_eq!(status, 400);
    assert!(body.contains("unsupported role"));
}
