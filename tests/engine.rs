// The engine against the golden data of the Python original, stage by stage (spec §10). The sampler test needs nothing but the repository; the
// others need the models and the GPU:
//   CB_ORT_DLL=<onnxruntime.dll>  [CB_MODELS=<models folder>]  cargo test --release --test engine -- --ignored --nocapture --test-threads=1
// with the folder of the CUDA/cuDNN DLLs on PATH.
use chatterbox_server::engine::{Engine, EngineConfig};
use chatterbox_server::npy;
use chatterbox_server::sampler::{draw, probabilities, Rng, SamplerSettings};
use chatterbox_server::synth::StreamDecoder;
use chatterbox_server::voice::Voice;
use serde_json::Value;
use std::path::PathBuf;

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}
fn golden(path: &str) -> PathBuf {
    root().join("tests/golden/us_f_50s").join(path)
}
fn load_engine() -> Engine {
    let dll = std::env::var("CB_ORT_DLL").expect("set CB_ORT_DLL");
    let models = std::env::var("CB_MODELS").map(PathBuf::from).unwrap_or_else(|_| root().join("models"));
    Engine::load(&EngineConfig { models_dir: models, data_dir: root().join("data"), ort_dll: dll.into(), use_cuda: true }).unwrap()
}
fn golden_voice() -> Voice {
    Voice {
        speaker_cond: npy::read(&golden("speaker_cond.npy")).unwrap().f32().unwrap(),
        t3_tokens: npy::read(&golden("t3_tokens.npy")).unwrap().i64().unwrap(),
        dec_tokens: npy::read(&golden("dec_tokens.npy")).unwrap().i64().unwrap(),
        dec_feat: npy::read(&golden("dec_feat.npy")).unwrap().f32().unwrap(),
        xvector: npy::read(&golden("dec_xvec.npy")).unwrap().f32().unwrap(),
    }
}
fn manifest() -> Value {
    serde_json::from_str(&std::fs::read_to_string(root().join("tests/golden/manifest.json")).unwrap()).unwrap()
}
fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let dot: f64 = a.iter().zip(b).map(|(x, y)| *x as f64 * *y as f64).sum();
    let na: f64 = a.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
    let nb: f64 = b.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
    dot / (na * nb)
}
fn max_difference(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max)
}

#[test]
fn the_sampler_matches_the_python() {
    let tokens = npy::read(&golden("M/tokens_seed0.npy")).unwrap().i64().unwrap();
    let logits = npy::read(&golden("M/tf_logits_first32.npy")).unwrap().f32().unwrap();
    let cases = manifest()["refs"]["us_f_50s"]["texts"]["M"]["sampler_cases"].as_array().unwrap().clone();
    assert!(!cases.is_empty());
    for case in cases {
        let step = case["step"].as_u64().unwrap() as usize;
        let history: Vec<i64> = if step == 0 { vec![6561] } else { tokens[..step.max(1)].to_vec() };
        assert_eq!(history.len() as u64, case["history_len"].as_u64().unwrap());
        let p = probabilities(&logits[step * 6563..(step + 1) * 6563], &history, &SamplerSettings::default());
        let want = npy::read(&golden(&format!("M/sampler_probs_step{step}.npy"))).unwrap().f64().unwrap();
        let worst = p.iter().zip(&want).map(|(a, b)| (a - b).abs()).fold(0f64, f64::max);
        assert!(worst < 1e-9, "step {step}: probabilities differ by {worst}");
        assert_eq!(p.iter().filter(|&&v| v > 0.0).count() as u64, case["n_nonzero"].as_u64().unwrap(), "step {step}");
        assert_eq!(draw(&p, case["u"].as_f64().unwrap()) as u64, case["token"].as_u64().unwrap(), "step {step}");
    }
    let _ = Rng::new(1);
}

#[test]
#[ignore]
fn the_text_rows_match_the_python() {
    let mut engine = load_engine();
    for (name, text) in [("M", None), ("E", None)] {
        let _: Option<&str> = text;
        let m = manifest();
        let text = m["refs"]["us_f_50s"]["texts"][name]["text"].as_str().unwrap().to_string();
        let want = npy::read(&golden(&format!("{name}/text_embeds_with_start.npy"))).unwrap().f32().unwrap();
        let got = engine.text_embeds(&text).unwrap();
        assert_eq!(got.len(), want.len(), "{name}");
        assert!(max_difference(&got, &want) < 1e-3, "{name}: difference {}", max_difference(&got, &want));
    }
}

#[test]
#[ignore]
fn decoding_the_golden_tokens_matches_the_python() {
    let mut engine = load_engine();
    let voice = golden_voice();
    let tokens = npy::read(&golden("M/tokens_seed0.npy")).unwrap().i64().unwrap();
    let want = npy::read(&golden("M/wav_det.npy")).unwrap().f32().unwrap();
    let got = engine.decode(&voice, &tokens, true).unwrap();
    assert_eq!(got.len(), want.len());
    // not bit for bit: the onnxruntime builds differ (the golden data was made with 1.31, this runs the 1.23 that Irodori shares), and fp16
    // convolutions on the GPU round differently; the waveforms must still be the same signal
    let worst = max_difference(&got, &want);
    let dot: f64 = got.iter().zip(&want).map(|(a, b)| *a as f64 * *b as f64).sum();
    let norm = |v: &[f32]| v.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
    let correlation = dot / (norm(&got) * norm(&want));
    println!("decoder: largest difference {worst}, correlation {correlation}");
    assert!(correlation > 0.995, "correlation {correlation}");
}

#[test]
#[ignore]
fn streaming_the_golden_tokens_matches_the_python() {
    let mut engine = load_engine();
    let voice = golden_voice();
    let tokens = npy::read(&golden("M/tokens_seed0.npy")).unwrap().i64().unwrap();
    let want = npy::read(&golden("M/wav_stream.npy")).unwrap().f32().unwrap();
    let want_chunks: Vec<i64> = manifest()["refs"]["us_f_50s"]["texts"]["M"]["stream_chunk_samples"].as_array().unwrap().iter().map(|v| v.as_i64().unwrap()).collect();
    let mut stream = StreamDecoder::new(&voice);
    let mut chunks: Vec<Vec<f32>> = Vec::new();
    for &t in &tokens {
        if let Some(chunk) = stream.push(&mut engine, &voice, t).unwrap() {
            chunks.push(chunk);
        }
    }
    chunks.push(stream.finish(&mut engine, &voice).unwrap());
    let sizes: Vec<i64> = chunks.iter().map(|c| c.len() as i64).collect();
    println!("chunk sizes {sizes:?}, wanted {want_chunks:?}");
    assert_eq!(sizes.len(), want_chunks.len());
    for (got, want) in sizes.iter().zip(&want_chunks) {
        assert!((got - want).abs() <= 120, "chunk of {got} samples, {want} wanted");
    }
    let all: Vec<f32> = chunks.concat();
    let n = all.len().min(want.len());
    let c = {
        let (a, b) = (&all[..n], &want[..n]);
        let dot: f64 = a.iter().zip(b).map(|(x, y)| *x as f64 * *y as f64).sum();
        dot / ((a.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt()) * (b.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt()))
    };
    println!("streamed audio correlation with the Python's {c}");
    assert!(c > 0.99, "correlation {c}");
}

#[test]
#[ignore]
fn registering_the_golden_recording_gives_the_python_voice() {
    let mut engine = load_engine();
    let recording = std::fs::read(golden("ref.wav")).unwrap();
    let voice = engine.register_voice(&recording).unwrap();
    let want = golden_voice();
    let agree = |a: &[i64], b: &[i64]| a.iter().zip(b).filter(|(x, y)| x == y).count() as f64 / a.len().max(b.len()) as f64;
    println!(
        "t3 tokens {} vs {} ({:.3} equal), decoder tokens {} vs {} ({:.3} equal), frames {} vs {}, x-vector cosine {:.6}, speaker cosine {:.6}",
        voice.t3_tokens.len(),
        want.t3_tokens.len(),
        agree(&voice.t3_tokens, &want.t3_tokens),
        voice.dec_tokens.len(),
        want.dec_tokens.len(),
        agree(&voice.dec_tokens, &want.dec_tokens),
        voice.frames(),
        want.frames(),
        cosine(&voice.xvector, &want.xvector),
        cosine(&voice.speaker_cond, &want.speaker_cond),
    );
    assert_eq!(voice.t3_tokens.len(), want.t3_tokens.len());
    assert_eq!(voice.dec_tokens.len(), want.dec_tokens.len());
    assert!(agree(&voice.t3_tokens, &want.t3_tokens) > 0.9);
    assert!(agree(&voice.dec_tokens, &want.dec_tokens) > 0.9);
    assert!(cosine(&voice.xvector, &want.xvector) > 0.99);
    assert!(cosine(&voice.speaker_cond, &want.speaker_cond) > 0.99);
}

#[test]
#[ignore]
fn speaking_in_the_registered_voice_writes_audio() {
    let mut engine = load_engine();
    let recording = std::fs::read(golden("ref.wav")).unwrap();
    let voice = engine.register_voice(&recording).unwrap();
    let out = root().join("target/test-audio");
    std::fs::create_dir_all(&out).unwrap();
    let mut rng = Rng::new(0);
    for (name, text) in [
        ("hello", "Hello, nice to meet you."),
        ("laugh", "Wait... you actually did that for me? [laugh] Oh my gosh, thank you so much!"),
        ("long", "If you have a moment, I would love to show you around this world. There are a lot of hidden places that most people never find, and I think you will really like them."),
    ] {
        let started = std::time::Instant::now();
        let wave = engine.synthesize(&voice, text, &SamplerSettings::default(), &mut rng).unwrap();
        println!("{name}: {:.1} s of audio in {:.2} s", wave.len() as f64 / 24000.0, started.elapsed().as_secs_f64());
        std::fs::write(out.join(format!("{name}.wav")), chatterbox_server::wavio::encode_wav(&wave, 24000)).unwrap();
        let mut first = None;
        let started = std::time::Instant::now();
        let mut samples = 0usize;
        engine
            .synthesize_streaming(&voice, text, &SamplerSettings::default(), &mut Rng::new(0), |chunk| {
                if first.is_none() {
                    first = Some(started.elapsed().as_secs_f64());
                }
                samples += chunk.len();
                Ok(())
            })
            .unwrap();
        println!("{name} streamed: first audio after {:.2} s, {:.1} s of audio in {:.2} s", first.unwrap(), samples as f64 / 24000.0, started.elapsed().as_secs_f64());
    }
}

#[test]
#[ignore]
fn a_very_short_recording_is_repeated_and_still_speaks() {
    let mut engine = load_engine();
    let full = hound::WavReader::open(golden("ref.wav")).unwrap();
    let spec = full.spec();
    let samples: Vec<i16> = full.into_samples::<i16>().map(|s| s.unwrap()).collect();
    let rate = spec.sample_rate as usize;
    let out = root().join("target/test-audio");
    std::fs::create_dir_all(&out).unwrap();
    for (name, seconds) in [("short_0_6", 0.6f64), ("short_1_2", 1.2)] {
        let start = 3 * rate;
        let clip = &samples[start..start + (seconds * rate as f64) as usize];
        let mut bytes = std::io::Cursor::new(Vec::new());
        {
            let mut writer = hound::WavWriter::new(&mut bytes, spec).unwrap();
            for &v in clip {
                writer.write_sample(v).unwrap();
            }
            writer.finalize().unwrap();
        }
        let voice = engine.register_voice(&bytes.into_inner()).unwrap();
        let wave = engine.synthesize(&voice, "Hello, nice to meet you.", &SamplerSettings::default(), &mut Rng::new(0)).unwrap();
        assert!(wave.len() > 24000, "{name}: {} samples", wave.len());
        std::fs::write(out.join(format!("{name}.wav")), chatterbox_server::wavio::encode_wav(&wave, 24000)).unwrap();
    }
    // nothing but silence is refused
    let mut silent = std::io::Cursor::new(Vec::new());
    {
        let mut writer = hound::WavWriter::new(&mut silent, hound::WavSpec { channels: 1, sample_rate: 24000, bits_per_sample: 16, sample_format: hound::SampleFormat::Int }).unwrap();
        for _ in 0..24000 {
            writer.write_sample(0i16).unwrap();
        }
        writer.finalize().unwrap();
    }
    assert!(engine.register_voice(&silent.into_inner()).is_err());
}
