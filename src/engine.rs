// The engine: Chatterbox Turbo as ONNX graphs run with ONNX Runtime (through `ort`, loading the DLL at run time).
//
//   text -> tokens -> embeddings -> language model (GPT-2 medium, KV cache on the GPU) -> speech tokens -> conditional decoder -> 24 kHz audio
//
// The ONNX files are the official ones of ResembleAI/chatterbox-turbo-ONNX except the reference voice encoder, which is exported from
// the PyTorch weights, and the decoder, whose random nodes are replaced by fixed noise (spec: the golden tests compare with it).

use anyhow::{anyhow, ensure, Result};
use half::f16;
use ort::environment::Environment;
use ort::ep;
use ort::memory::{AllocationDevice, AllocatorType, MemoryInfo, MemoryType};
use ort::session::Session;
use ort::value::{DynValue, Tensor};
use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use tokenizers::Tokenizer;

pub const HIDDEN: usize = 1024;
pub const VOCAB: usize = 6563;
pub const LAYERS: usize = 24;
pub const HEADS: usize = 16;
pub const HEAD_DIM: usize = 64;
// speech token ids: START and STOP of the language model, and the silence token the decoder is closed with
pub const START: i64 = 6561;
pub const STOP: i64 = 6562;
pub const SILENCE: i64 = 4299;
// The most tokens sampled for one piece of text (the original's limit: 40 s of speech)
pub const MAX_NEW_TOKENS_LIMIT: usize = 1000;

pub struct EngineConfig {
    // Folder of the ONNX models (see `Engine::load`)
    pub models_dir: PathBuf,
    // Folder of tokenizer.json and ve_mel_basis_40x201.npy
    pub data_dir: PathBuf,
    // onnxruntime.dll (the build matching the provider DLLs next to it)
    pub ort_dll: PathBuf,
    pub use_cuda: bool,
}

// ONNX Runtime allows one environment per process, ever: once built (or failed to build) it is what every later `load` gets, so the
// DLL path of the first call is the one that counts, and a failed first call is reported again by the following ones.
fn environment(dll: &Path) -> Result<&'static Environment> {
    static ENVIRONMENT: OnceLock<std::result::Result<Environment, String>> = OnceLock::new();
    ENVIRONMENT
        .get_or_init(|| {
            ort::init_from(dll)
                .map_err(|e| format!("{}: {e}", dll.display()))
                .and_then(|builder| builder.with_name("chatterbox").build().map_err(|e| e.to_string()))
        })
        .as_ref()
        .map_err(|e| anyhow!("{e}"))
}

// `gpu`: the CUDA provider with the convolution algorithm picked by heuristics (the default search benchmarks every new input shape, and
// the decoder's input length differs for every sentence). The memory arena stays at its default (grow to the next power of two):
// with "same as requested" the language model's KV cache, which grows by one position per token, made the process take 11 GB.
pub(crate) fn session(environment: &Environment, path: &Path, gpu: bool) -> Result<Session> {
    let mut builder = Session::builder(environment).map_err(|e| anyhow!("{e}"))?;
    // The engine runs next to a game: busy-waiting threads would take CPU time from it.
    builder = builder.with_intra_threads(if gpu { 1 } else { 2 }).map_err(|e| anyhow!("{e}"))?;
    if gpu {
        let cuda = ep::CUDA::default().with_conv_algorithm_search(ep::cuda::ConvAlgorithmSearch::Heuristic).build().error_on_failure();
        builder = builder.with_execution_providers([cuda]).map_err(|e| anyhow!("{e}"))?;
    }
    builder.commit_from_file(path).map_err(|e| anyhow!("{}: {e}", path.display()))
}

pub struct Engine {
    pub(crate) embed: Session,
    pub(crate) lm: Session,
    pub(crate) decoder: Session,
    gpu_memory: MemoryInfo<'static>,
    cpu_memory: MemoryInfo<'static>,
    kv_names: Vec<String>,
    present_names: Vec<String>,
    pub(crate) tokenizer: Tokenizer,
    pub(crate) environment: &'static Environment,
    pub(crate) models_dir: PathBuf,
    pub(crate) data_dir: PathBuf,
}

// The language model's state between two calls: the KV cache (48 tensors, on the GPU) and how many positions it holds.
pub struct LmState {
    past: Vec<DynValue>,
    pub positions: usize,
}

impl LmState {
    pub fn new() -> LmState {
        LmState { past: Vec::new(), positions: 0 }
    }
}

impl Engine {
    // models_dir holds: embed_tokens_fp16.onnx, language_model_fp16.onnx, conditional_decoder_fp16_det.onnx (each with its .onnx_data).
    pub fn load(config: &EngineConfig) -> Result<Engine> {
        let environment = environment(&config.ort_dll)?;
        let dir = &config.models_dir;
        let mut kv_names = Vec::new();
        let mut present_names = Vec::new();
        for layer in 0..LAYERS {
            for kind in ["key", "value"] {
                kv_names.push(format!("past_key_values.{layer}.{kind}"));
                present_names.push(format!("present.{layer}.{kind}"));
            }
        }
        Ok(Engine {
            // the text and speech embedding tables run on the CPU: it saves 290 MiB of VRAM and costs no time
            embed: session(environment, &dir.join("embed_tokens_fp16.onnx"), false)?,
            lm: session(environment, &dir.join("language_model_fp16.onnx"), config.use_cuda)?,
            decoder: session(environment, &dir.join("conditional_decoder_fp16_det.onnx"), config.use_cuda)?,
            gpu_memory: MemoryInfo::new(AllocationDevice::CUDA, 0, AllocatorType::Device, MemoryType::Default).map_err(|e| anyhow!("{e}"))?,
            cpu_memory: MemoryInfo::new(AllocationDevice::CPU, 0, AllocatorType::Arena, MemoryType::CPUOutput).map_err(|e| anyhow!("{e}"))?,
            kv_names,
            present_names,
            tokenizer: Tokenizer::from_file(config.data_dir.join("tokenizer.json")).map_err(|e| anyhow!("tokenizer.json: {e}"))?,
            environment,
            models_dir: config.models_dir.clone(),
            data_dir: config.data_dir.clone(),
        })
    }

    // The embedding rows of token ids: positions 0..L-3 text rows, the last two positions speech rows (START for 50256), as the graph is
    // built. `ids` is [rows, length]; returns rows * length * 1024 floats.
    pub fn embed(&mut self, ids: &[i64], rows: usize) -> Result<Vec<f32>> {
        ensure!(rows > 0 && ids.len() % rows == 0, "the ids do not fill {rows} rows");
        let input = Tensor::from_array((vec![rows, ids.len() / rows], ids.to_vec())).map_err(|e| anyhow!("{e}"))?;
        let outputs = self.embed.run(ort::inputs!["input_ids" => input]).map_err(|e| anyhow!("{e}"))?;
        let (_, data) = outputs["inputs_embeds"].try_extract_tensor::<f32>().map_err(|e| anyhow!("{e}"))?;
        Ok(data.to_vec())
    }

    // Speech token embeddings (the prompt of the reference voice): the table is reached by feeding the ids as pairs, whose two positions
    // are both speech rows. An odd count is padded and cut again.
    pub fn speech_rows(&mut self, tokens: &[i64]) -> Result<Vec<f32>> {
        let mut padded = tokens.to_vec();
        if padded.len() % 2 == 1 {
            padded.push(0);
        }
        let rows = padded.len() / 2;
        let mut out = self.embed(&padded, rows)?;
        out.truncate(tokens.len() * HIDDEN);
        Ok(out)
    }

    // One call of the language model on `embeds` ([S, 1024] flattened): the logits of its last position. The KV cache moves from the
    // previous call's outputs to this call's inputs without leaving the GPU.
    pub fn lm_step(&mut self, state: &mut LmState, embeds: &[f32]) -> Result<Vec<f32>> {
        ensure!(embeds.len() % HIDDEN == 0 && !embeds.is_empty(), "the embeddings are not a whole number of rows");
        let new = embeds.len() / HIDDEN;
        let past_len = state.positions;
        let mut binding = self.lm.create_binding().map_err(|e| anyhow!("{e}"))?;
        let e = |err: ort::Error| anyhow!("{err}");
        binding.bind_input("inputs_embeds", &Tensor::from_array((vec![1usize, new, HIDDEN], embeds.to_vec())).map_err(e)?).map_err(e)?;
        binding.bind_input("attention_mask", &Tensor::from_array((vec![1usize, past_len + new], vec![1i64; past_len + new])).map_err(e)?).map_err(e)?;
        binding
            .bind_input("position_ids", &Tensor::from_array((vec![1usize, new], (past_len as i64..(past_len + new) as i64).collect::<Vec<i64>>())).map_err(e)?)
            .map_err(e)?;
        if state.past.is_empty() {
            for name in &self.kv_names {
                let empty = Tensor::<f16>::from_array((vec![1usize, HEADS, 0, HEAD_DIM], Vec::<f16>::new())).map_err(e)?;
                binding.bind_input(name.as_str(), &empty).map_err(e)?;
            }
        } else {
            for (name, value) in self.kv_names.iter().zip(&state.past) {
                binding.bind_input(name.as_str(), value).map_err(e)?;
            }
        }
        binding.bind_output_to_device("logits", &self.cpu_memory).map_err(e)?;
        for name in &self.present_names {
            binding.bind_output_to_device(name.as_str(), &self.gpu_memory).map_err(e)?;
        }
        let mut outputs = self.lm.run_binding(&binding).map_err(e)?;
        let logits = {
            let (shape, data) = outputs["logits"].try_extract_tensor::<f32>().map_err(e)?;
            ensure!(shape.len() == 3 && shape[2] as usize == VOCAB, "unexpected logits shape {shape:?}");
            let last = (shape[1] as usize - 1) * VOCAB;
            data[last..last + VOCAB].to_vec()
        };
        let mut past = Vec::with_capacity(self.present_names.len());
        for name in &self.present_names {
            past.push(outputs.remove(name.as_str()).ok_or_else(|| anyhow!("the language model gave no {name}"))?);
        }
        drop(outputs);
        state.past = past;
        state.positions = past_len + new;
        Ok(logits)
    }
}
