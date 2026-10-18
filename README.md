# chatterbox-server

English text to speech in a registered reference voice, with laughs, sighs and other sounds written as tags (`[laugh]`, `[sigh]`, ...),
for [Mutelink](https://github.com/utyujinn/Mutelink-releases): [Chatterbox Turbo](https://github.com/resemble-ai/chatterbox) (MIT) as ONNX
graphs on ONNX Runtime, in Rust. It is a child process of the app, like `irodori-server` and (before it) `openvoice-server`: it talks over
stdin / stdout (`src/protocol.rs`), so the app can end it to give back all the GPU memory.

It is its own Cargo workspace on purpose: it needs the upstream `ort` crate, and the app's graph already holds voicevox_core's fork of it
(both declare `links = "onnxruntime"`). Build and test it here, not from `tauri-src/src-tauri`.

```
cargo build --release          # target/release/chatterbox-server.exe
cargo test --release           # everything that needs no models or GPU
```

The tests that need the models and the GPU are `#[ignore]`d: put the models of release `models-v1` in `models/` and run

```
set CB_ORT_DLL=<path to onnxruntime.dll, 1.23 GPU build>      (the CUDA 12 and cuDNN 9 DLLs on PATH)
cargo test --release --test lm --test engine --test server -- --ignored --nocapture --test-threads=1
```

## What it does

- **Registering a voice** (`register_voice`): a recording of 2 s or more (shorter than 6 s it is repeated until it is long enough, because
  the language model babbles on a prompt under about 5 s). The recording is brought to -27 LUFS, the reference encoders run once on the CPU,
  and what speech needs from it (about 160 KB) is written to a voice file. The recording is not kept.
- **Speaking** (`synthesize`): the text is cut into pieces of about a sentence; each piece goes through the language model (GPT-2 medium,
  KV cache kept on the GPU) and the decoder. With `stream`, the audio of a piece comes in chunks, the first after about 0.3 s, joined with
  cross-fades so that they can be played back to back.
- About 1.7 GB of VRAM while loaded (fp16), 1.4 GB of files. Measured against the PyTorch original over 13 voices: the voice resembles the
  reference as much as the original does, with the same intelligibility (`docs/SPEC.md`, §0).

## Layout

- `src/protocol.rs`: the frames on stdin / stdout. `src/main.rs`: the commands (`load`, `register_voice`, `synthesize`), documented at its top.
- `src/engine.rs`: the graphs and the language model's step; `src/synth.rs`: text, the generation loop, decoding, streaming, registration.
- `src/audio.rs`, `src/resample.rs`: loudness, silence trim, the VoiceEncoder's mel input and resampling, as the original does them.
- `src/sampler.rs`: temperature, top-k, top-p and repetition penalty in the order of the original; `src/stream.rs`: joining streamed chunks.
- `src/voice.rs`: the voice file. `src/text.rs`: the text normalization of the original and the cutting into pieces.
- `models/` (not in git): the ONNX files, release `models-v1`. `data/`: `tokenizer.json` and the VoiceEncoder's mel filterbank.
- `docs/SPEC.md`: the specification the Rust code was written from (every graph's inputs and outputs, preprocessing constants, the sampler,
  streaming, precision and VRAM measurements, why the official ONNX recipe resembled the reference less). `tools/`: the Python that made the
  models and the golden data (`tests/golden/`), written for the folder layout of the experiment where it was made.

## Release

The executable is built by GitHub Actions (`.github/workflows/release.yml`). Raise `version` in `Cargo.toml`, commit, and push a tag
`server-vX.Y.Z` with the same number: the workflow runs the tests, builds with `--locked`, and publishes the exe as the release of that tag with
its SHA-256 in the notes (Mutelink pins that checksum). "Run workflow" on the Actions page only builds and tests. The models are the assets of
release `models-v1`, uploaded by hand.

## License

MIT ([LICENSE](LICENSE)). The models and the software this builds on are under their own licenses: see [NOTICE.md](NOTICE.md) and
[MODEL_LICENSES.md](MODEL_LICENSES.md).
