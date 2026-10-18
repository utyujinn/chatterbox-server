// The language model against the golden logits of the Python original. Needs the models and the GPU:
//   CB_ORT_DLL=<onnxruntime.dll>  [CB_MODELS=<models folder>]  cargo test --release --test lm -- --ignored --nocapture
// with the folder of the CUDA/cuDNN DLLs on PATH.
use chatterbox_server::engine::{Engine, EngineConfig, LmState};
use chatterbox_server::npy;
use std::path::PathBuf;

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

pub fn load_engine() -> Engine {
    let dll = std::env::var("CB_ORT_DLL").expect("set CB_ORT_DLL");
    let models = std::env::var("CB_MODELS").map(PathBuf::from).unwrap_or_else(|_| root().join("models"));
    Engine::load(&EngineConfig { models_dir: models, data_dir: root().join("data"), ort_dll: dll.into(), use_cuda: true }).unwrap()
}

#[test]
#[ignore]
fn the_logits_follow_the_python_teacher_forced() {
    let mut engine = load_engine();
    let golden = root().join("tests/golden/us_f_50s/M");
    let prefill = npy::read(&golden.join("prefill_embeds.npy")).unwrap().f32().unwrap();
    let tokens = npy::read(&golden.join("tokens_seed0.npy")).unwrap().i64().unwrap();
    let want = npy::read(&golden.join("tf_logits_first32.npy")).unwrap().f32().unwrap();
    let mut state = LmState::new();
    let mut worst = 0f32;
    let mut agree = 0;
    for step in 0..32 {
        let logits = if step == 0 {
            engine.lm_step(&mut state, &prefill).unwrap()
        } else {
            let row = engine.embed(&[tokens[step - 1]], 1).unwrap();
            engine.lm_step(&mut state, &row).unwrap()
        };
        let expected = &want[step * 6563..(step + 1) * 6563];
        worst = worst.max(logits.iter().zip(expected).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max));
        let argmax = |v: &[f32]| v.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0;
        if argmax(&logits) == argmax(expected) {
            agree += 1;
        }
    }
    println!("worst logit difference {worst}, argmax agreement {agree}/32");
    assert!(worst < 0.1, "worst difference {worst}");
    assert!(agree >= 31);
}
