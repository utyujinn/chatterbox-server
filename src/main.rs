// chatterbox-server: the child process Mutelink starts for English speech in a reference voice (see engine.rs, protocol.rs).
// Commands (the request header's "cmd"):
//   load            {models_dir, data_dir, ort_dll, cuda, voice?}            -> {ok}   `voice`: a voice file to speak one short word in, so that the
//                   first real sentence does not pay for loading the GPU kernels
//   register_voice  {out}  body: a WAV file (any rate, mono or stereo)       -> {ok, t3_tokens}   writes the voice file `out`
//   synthesize      {text, voice, seed?, temperature?, stream?}              -> frames {ok, more: true, piece, pieces, text, rate, last}, body a
//                   16-bit mono WAV at 24 kHz, then {ok, more: false}. The text is cut into pieces (about a sentence each). With `stream` every
//                   piece comes as several frames (the first after about 0.3 s), each frame a WAV that continues the one before it (they are
//                   cross-faded already: play them back to back); `last` is true on the final frame of a piece. Without it, one frame per piece.
// Anything printed for a human goes to stderr: stdout carries the frames only.

use anyhow::{anyhow, bail, Context, Result};
use chatterbox_server::engine::{Engine, EngineConfig};
use chatterbox_server::protocol::{read_frame, write_frame};
use chatterbox_server::sampler::{Rng, SamplerSettings};
use chatterbox_server::text::split_pieces;
use chatterbox_server::voice::Voice;
use chatterbox_server::wavio::encode_wav;
use serde_json::{json, Value};
use std::io::{self, BufReader, BufWriter, Write};
use std::path::Path;

const RATE: u32 = 24000;

fn text_param<'a>(header: &'a Value, key: &str) -> Result<&'a str> {
    header[key].as_str().ok_or_else(|| anyhow!("`{key}` is missing"))
}

fn settings_from(header: &Value) -> Result<SamplerSettings> {
    let mut settings = SamplerSettings::default();
    if let Some(t) = header["temperature"].as_f64() {
        anyhow::ensure!(t.is_finite() && (0.1..=1.5).contains(&t), "temperature must be between 0.1 and 1.5 (it is {t})");
        settings.temperature = t;
    }
    Ok(settings)
}

fn seed_from(header: &Value) -> u64 {
    header["seed"].as_u64().unwrap_or_else(|| std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_nanos() as u64).unwrap_or(1))
}

fn handle(engine: &mut Option<Engine>, header: &Value, body: &[u8], out: &mut impl Write) -> Result<()> {
    if header["too_large"] == true {
        bail!("the request is larger than the server accepts");
    }
    match text_param(header, "cmd")? {
        "load" => {
            let config = EngineConfig {
                models_dir: text_param(header, "models_dir")?.into(),
                data_dir: text_param(header, "data_dir")?.into(),
                ort_dll: text_param(header, "ort_dll")?.into(),
                use_cuda: header["cuda"].as_bool().unwrap_or(true),
            };
            // the old engine goes first: two of them at once would need twice the VRAM
            *engine = None;
            let mut loaded = Engine::load(&config)?;
            // The warm-up is only a courtesy: a voice file that is damaged (or from an older build) must not keep the engine from loading,
            // or the user could not even register a new voice to replace it.
            if let Some(path) = header["voice"].as_str() {
                let warm_up = Voice::load(Path::new(path)).and_then(|voice| {
                    loaded.synthesize(&voice, "Hello.", &SamplerSettings::default(), &mut Rng::new(1)).context("the warm-up synthesis")
                });
                if let Err(error) = warm_up {
                    eprintln!("chatterbox-server: the warm-up was skipped: {error:#}");
                }
            }
            *engine = Some(loaded);
            write_frame(out, &json!({"ok": true}), &[])?;
        }
        "register_voice" => {
            let engine = engine.as_mut().ok_or_else(|| anyhow!("load first"))?;
            let voice = engine.register_voice(body)?;
            voice.save(Path::new(text_param(header, "out")?))?;
            write_frame(out, &json!({"ok": true, "t3_tokens": voice.t3_tokens.len()}), &[])?;
        }
        "synthesize" => {
            let engine = engine.as_mut().ok_or_else(|| anyhow!("load first"))?;
            let voice = Voice::load(Path::new(text_param(header, "voice")?))?;
            let settings = settings_from(header)?;
            let mut rng = Rng::new(seed_from(header));
            let stream = header["stream"].as_bool().unwrap_or(false);
            let pieces = split_pieces(text_param(header, "text")?);
            if pieces.is_empty() {
                bail!("there is nothing to say in that text");
            }
            let count = pieces.len();
            let mut spoken = 0;
            for (index, piece) in pieces.iter().enumerate() {
                let frame = |last: bool| json!({"ok": true, "more": true, "piece": index, "pieces": count, "text": piece, "rate": RATE, "last": last});
                // A piece that comes out empty (the model ended it at once: "..." or a lone tag) is skipped, and the rest of the text is spoken;
                // the request fails only if nothing at all was said.
                let skipped = |error: &anyhow::Error| error.to_string().starts_with("no speech came out");
                if stream {
                    // a chunk is sent once the next one is there (or the piece ends), so `last` is known when it goes out
                    let mut pending: Option<Vec<f32>> = None;
                    let result = engine.synthesize_streaming(&voice, piece, &settings, &mut rng, |chunk| {
                        if let Some(previous) = pending.replace(chunk.to_vec()) {
                            write_frame(out, &frame(false), &encode_wav(&previous, RATE))?;
                        }
                        Ok(())
                    });
                    match result {
                        Ok(()) => match pending.take() {
                            Some(last) => {
                                write_frame(out, &frame(true), &encode_wav(&last, RATE))?;
                                spoken += 1;
                            }
                            None => eprintln!("chatterbox-server: piece {index} came out empty: {piece:?}"),
                        },
                        Err(error) if skipped(&error) && pending.is_none() => eprintln!("chatterbox-server: piece {index} came out empty: {piece:?}"),
                        Err(error) => return Err(error.context(format!("piece {index}: {piece:?}"))),
                    }
                } else {
                    match engine.synthesize(&voice, piece, &settings, &mut rng) {
                        Ok(wave) => {
                            write_frame(out, &frame(true), &encode_wav(&wave, RATE))?;
                            spoken += 1;
                        }
                        Err(error) if skipped(&error) => eprintln!("chatterbox-server: piece {index} came out empty: {piece:?}"),
                        Err(error) => return Err(error.context(format!("piece {index}: {piece:?}"))),
                    }
                }
            }
            if spoken == 0 {
                bail!("no speech came out of that text");
            }
            write_frame(out, &json!({"ok": true, "more": false}), &[])?;
        }
        other => bail!("unknown command {other:?}"),
    }
    Ok(())
}

fn main() -> Result<()> {
    let mut input = BufReader::new(io::stdin().lock());
    let mut output = BufWriter::new(io::stdout().lock());
    let mut engine: Option<Engine> = None;
    while let Some(frame) = read_frame(&mut input)? {
        // A failed request is an answer, not the end of the process: the app keeps the loaded models. A panic is turned into an answer
        // too (the code is written not to panic, this is the last wall): the engine is still usable, and a process that dies costs the
        // user a reload of several seconds.
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| handle(&mut engine, &frame.header, &frame.body, &mut output)));
        let failure = match outcome {
            Ok(Ok(())) => None,
            Ok(Err(error)) => Some(format!("{error:#}")),
            Err(panic) => {
                let what = panic.downcast_ref::<&str>().map(|s| s.to_string()).or_else(|| panic.downcast_ref::<String>().cloned()).unwrap_or_default();
                Some(format!("internal error: {what}"))
            }
        };
        if let Some(message) = failure {
            eprintln!("chatterbox-server: {message}");
            write_frame(&mut output, &json!({"ok": false, "error": message}), &[])?;
        }
    }
    Ok(())
}
