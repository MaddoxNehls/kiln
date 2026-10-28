mod server;

use std::io::{self, BufRead, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result};
use clap::{Args, Parser, Subcommand};

use kiln::chat::{ChatTemplate, Message, Role};
use kiln::generate::{Generation, Session, StopReason};
use kiln::model::Model;
use kiln::perplexity;
use kiln::sampler::{Sampler, SamplingParams};
use kiln::tensor::Quant;
use kiln::tokenizer::{TokenId, Tokenizer};

#[derive(Parser)]
#[command(version, about = "Kiln: a from-scratch LLM inference engine")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Generate a single completion and stream it to stdout.
    Generate {
        #[command(flatten)]
        model: ModelArgs,
        #[command(flatten)]
        args: GenerationArgs,
        #[arg(long)]
        prompt: String,
        /// Feed the prompt as-is instead of wrapping it in the chat template.
        #[arg(long)]
        raw: bool,
    },
    /// Interactive multi-turn chat.
    Chat {
        #[command(flatten)]
        model: ModelArgs,
        #[command(flatten)]
        args: GenerationArgs,
    },
    /// Serve an OpenAI-compatible API (`/v1/chat/completions`, `/v1/models`).
    Serve {
        #[command(flatten)]
        model: ModelArgs,
        #[command(flatten)]
        args: GenerationArgs,
        #[arg(long, default_value = "127.0.0.1:8080")]
        addr: std::net::SocketAddr,
    },
    /// Perplexity of a text file (e.g. WikiText-2 test, see scripts/download-wikitext.sh).
    Perplexity {
        #[command(flatten)]
        model: ModelArgs,
        #[arg(long, default_value = "data/wikitext-2-raw/wiki.test.raw")]
        file: PathBuf,
        /// Tokens per evaluation window (windows do not overlap).
        #[arg(long, default_value_t = 512)]
        ctx: usize,
        /// Number of windows to evaluate.
        #[arg(long, default_value_t = 40)]
        windows: usize,
    },
}

#[derive(Args)]
struct ModelArgs {
    /// Directory with config.json, tokenizer.json, tokenizer_config.json, model.safetensors.
    #[arg(long, default_value = "models/smollm2-135m-instruct")]
    model: PathBuf,
    /// Weight format, quantized at load time: f32, q8 (int8 per row), q4 (4-bit,
    /// groups of 32) or q4g<N> (groups of N).
    #[arg(long, default_value = "f32")]
    quant: Quant,
    /// Worker threads (default: all logical cores).
    #[arg(long)]
    threads: Option<usize>,
}

#[derive(Args)]
struct GenerationArgs {
    #[arg(long, default_value_t = 256)]
    max_tokens: usize,
    /// 0 = greedy.
    #[arg(long, default_value_t = 0.7)]
    temperature: f32,
    /// 0 = disabled.
    #[arg(long, default_value_t = 40)]
    top_k: usize,
    #[arg(long, default_value_t = 0.9)]
    top_p: f32,
    #[arg(long, default_value_t = 1.1)]
    repeat_penalty: f32,
    #[arg(long, default_value_t = 64)]
    repeat_last_n: usize,
    /// RNG seed; random if omitted.
    #[arg(long)]
    seed: Option<u64>,
    /// Override the chat template's default system prompt.
    #[arg(long)]
    system: Option<String>,
    /// KV cache capacity in tokens.
    #[arg(long, default_value_t = 2048)]
    ctx: usize,
}

impl GenerationArgs {
    fn sampling(&self) -> SamplingParams {
        let seed = self.seed.unwrap_or_else(|| {
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_or(0, |d| d.as_nanos() as u64)
        });
        SamplingParams {
            temperature: self.temperature,
            top_k: self.top_k,
            top_p: self.top_p,
            repetition_penalty: self.repeat_penalty,
            repeat_last_n: self.repeat_last_n,
            seed,
        }
    }
}

/// Everything loaded from a model directory.
struct Engine {
    model: Model,
    tokenizer: Tokenizer,
    template: ChatTemplate,
    stop_tokens: Vec<TokenId>,
}

impl Engine {
    fn load(args: &ModelArgs) -> Result<Self> {
        if let Some(n) = args.threads {
            rayon::ThreadPoolBuilder::new()
                .num_threads(n)
                .build_global()
                .context("configuring thread pool")?;
        }
        let dir: &Path = &args.model;
        let t0 = Instant::now();
        let tokenizer = Tokenizer::from_file(dir.join("tokenizer.json"))?;
        let template = ChatTemplate::from_tokenizer_config(dir.join("tokenizer_config.json"))?;
        let model = Model::load_quantized(dir, args.quant)
            .with_context(|| format!("loading model from {}", dir.display()))?;
        let mut stop_tokens = model.config().eos_token_ids();
        stop_tokens.extend(tokenizer.token_id(template.end_of_turn()));
        stop_tokens.extend(tokenizer.token_id("<|endoftext|>"));
        let c = model.config();
        eprintln!(
            "loaded {} ({} layers, d={}, {}q/{}kv heads) as {}: {:.1} MB of weights, {:.2}s, {} threads",
            dir.display(),
            c.num_hidden_layers,
            c.hidden_size,
            c.num_attention_heads,
            c.num_kv_heads(),
            model.quant(),
            model.weight_bytes() as f64 / 1e6,
            t0.elapsed().as_secs_f64(),
            rayon::current_num_threads()
        );
        Ok(Self {
            model,
            tokenizer,
            template,
            stop_tokens,
        })
    }

    fn conversation_start(&self, system: Option<&str>) -> Vec<Message> {
        system
            .map(|s| vec![Message::new(Role::System, s)])
            .unwrap_or_default()
    }

    /// Run one generation, streaming decoded text to stdout.
    fn stream(
        &self,
        session: &mut Session,
        prompt: &[TokenId],
        max_tokens: usize,
        sampler: &mut Sampler,
    ) -> Result<(Generation, String)> {
        let mut decoder = self.tokenizer.stream_decoder(true);
        let mut text = String::new();
        let mut stdout = io::stdout().lock();
        let generation = session.generate(
            &self.model,
            prompt,
            max_tokens,
            &self.stop_tokens,
            sampler,
            |id| {
                let piece = decoder.push(id);
                // A failed write to stdout (e.g. closed pipe) is not worth
                // aborting generation over; the text is still returned.
                let _ = stdout
                    .write_all(piece.as_bytes())
                    .and_then(|()| stdout.flush());
                text.push_str(&piece);
            },
        )?;
        let tail = decoder.finish();
        print!("{tail}");
        text.push_str(&tail);
        println!();
        Ok((generation, text))
    }
}

fn report(g: &Generation) {
    let s = &g.stats;
    let reason = match g.stop {
        StopReason::StopToken => "stop token",
        StopReason::MaxTokens => "max tokens",
        StopReason::ContextFull => "context full",
    };
    eprintln!(
        "[prefill {} tok in {:.0} ms = {:.1} tok/s | decode {} tok in {:.0} ms = {:.1} tok/s | {reason}]",
        s.prefill_tokens,
        s.prefill_time.as_secs_f64() * 1e3,
        s.prefill_tok_per_s(),
        s.decode_tokens,
        s.decode_time.as_secs_f64() * 1e3,
        s.decode_tok_per_s(),
    );
}

fn main() -> Result<()> {
    match Cli::parse().command {
        Command::Generate {
            model,
            args,
            prompt,
            raw,
        } => {
            let engine = Engine::load(&model)?;
            let mut sampler = Sampler::new(args.sampling());
            let mut session = Session::new(&engine.model, args.ctx);
            let text = if raw {
                prompt
            } else {
                let mut messages = engine.conversation_start(args.system.as_deref());
                messages.push(Message::new(Role::User, prompt));
                engine.template.render(&messages, true)
            };
            let ids = engine.tokenizer.encode(&text);
            let (generation, _) =
                engine.stream(&mut session, &ids, args.max_tokens, &mut sampler)?;
            report(&generation);
        }
        Command::Chat { model, args } => {
            let engine = Engine::load(&model)?;
            chat(&engine, &args)?;
        }
        Command::Serve { model, args, addr } => {
            let engine = Engine::load(&model)?;
            let model_id = model
                .model
                .file_name()
                .map_or_else(|| "kiln".into(), |n| n.to_string_lossy().into_owned());
            server::serve(
                engine,
                addr,
                args.ctx,
                args.sampling(),
                args.max_tokens,
                model_id,
            )?;
        }
        Command::Perplexity {
            model,
            file,
            ctx,
            windows,
        } => {
            let engine = Engine::load(&model)?;
            let text = std::fs::read_to_string(&file).with_context(|| {
                format!(
                    "reading {} (run scripts/download-wikitext.sh)",
                    file.display()
                )
            })?;
            let tokens = engine.tokenizer.encode(&text);
            eprintln!(
                "{} tokens; scoring {windows} windows of {ctx}",
                tokens.len()
            );
            let t0 = Instant::now();
            let result = perplexity::evaluate(&engine.model, &tokens, ctx, windows, |p| {
                eprint!("\r  window {:>3}: ppl {:.4}", p.windows, p.ppl());
            })?;
            eprintln!();
            println!(
                "{}: perplexity {:.4} over {} tokens ({} windows × {ctx}) in {:.1}s",
                engine.model.quant(),
                result.ppl(),
                result.scored_tokens,
                result.windows,
                t0.elapsed().as_secs_f64()
            );
        }
    }
    Ok(())
}

fn chat(engine: &Engine, args: &GenerationArgs) -> Result<()> {
    let mut sampler = Sampler::new(args.sampling());
    let mut session = Session::new(&engine.model, args.ctx);
    eprintln!("chat mode: empty line or Ctrl-Z/Ctrl-D to quit, /reset to clear history");
    let mut messages = engine.conversation_start(args.system.as_deref());
    let stdin = io::stdin();
    loop {
        eprint!("\n> ");
        io::stderr().flush()?;
        let mut line = String::new();
        if stdin.lock().read_line(&mut line)? == 0 || line.trim().is_empty() {
            return Ok(());
        }
        if line.trim() == "/reset" {
            messages = engine.conversation_start(args.system.as_deref());
            continue;
        }
        messages.push(Message::new(Role::User, line.trim_end()));
        // Re-render the whole conversation; the session reuses the cached
        // prefix, so only the new turn is actually computed.
        let ids = engine
            .tokenizer
            .encode(&engine.template.render(&messages, true));
        if ids.len() >= session.capacity() {
            eprintln!("[context full; use /reset]");
            messages.pop();
            continue;
        }
        let (generation, reply) =
            engine.stream(&mut session, &ids, args.max_tokens, &mut sampler)?;
        report(&generation);
        messages.push(Message::new(Role::Assistant, reply.trim_end()));
    }
}
