# Chatterbox Turbo: ONNX-only inference spec for the Rust port

Status legend: **[V]** verified numerically in this folder (script named), **[I]** inferred (source/graph reading, not
separately measured).  Python reference implementation: `cbx.py` (onnxruntime 1.31, CUDA EP), configuration
`Opt(spk="ve", t3p="ref", start="one", dec="ref")`.  All quality numbers: 13 references (multiref 10 + pack_bf greet/love/
playful) x 4 sentences (S/M/L/E) x seeds; WeSpeaker ResNet34 cosine to the reference clip A (simA) and a held-out clip B
(simB); WER = faster-whisper small.en vs. the input text.  You cannot trust these proxies blindly: see "Listen" at the end.

## 0. Results in one table [V] (`summary.py`)

| pipeline | n | simA | simB | simA − PyTorch (paired) | WER | peak VRAM | M sentence total |
|---|---|---|---|---|---|---|---|
| PyTorch ChatterboxTurboTTS | 104 | 0.744 | 0.683 | 0 | 0.013 | ~3.6 GB (earlier bench) | — |
| official ONNX recipe (old `gen_onnx_turbo.py` logic) | 104 | 0.655 | 0.617 | −0.089 ± 0.008 | 0.019 | — | — |
| **this spec, fp16** (`cfg_fp16`) | 104 | **0.746** | **0.695** | **+0.001 ± 0.006** | **0.011** | **1723 MiB** | 1.40 s (whole) |
| this spec, LM q4f16 + decoder q4f16 (`cfg_allq4`) | 52 | 0.746 | 0.692 | +0.004 ± 0.010 | 0.012 | 1211 MiB | 1.44 s |
| this spec fp16, streamed (`str_default`) vs same tokens whole (`whole_det`) | 52 | 0.746 / 0.749 | 0.699 / 0.700 | | 0.013 / 0.010 | | first audio 0.39–0.43 s |

VRAM = dedicated GPU memory of the process (Windows "GPU Process Memory" counter, `cbx.GpuMem`), CUDA context included,
default arena (`kNextPowerOfTwo`), VRChat not running.  Timings in this table: **VRChat not running**, RTX 4070.

## 1. Files

| file | runs on | size (MB) | role |
|---|---|---|---|
| `models/ref_encoder_w16.onnx` (self-contained) | CPU EP, once per voice | 282 | S3 tokenizer (T3 prompt + decoder prompt), CAMPPlus x-vector, 24 kHz prompt mel. fp32 compute, fp16-STORED weights (DSP matrices fp32). Exported from the PyTorch weights (MIT): `export_ref_encoder.py` then `shrink_refenc.py` |
| `models/ve_spk.onnx` | CPU EP, once per voice | 6.8 | VoiceEncoder LSTM + mean/L2 + T3 `cond_enc.spkr_enc` projection (`pt_ref.py export`) |
| `onnx_turbo/embed_tokens_fp16.onnx` + `.onnx_data` | **CPU EP** | 0.002 + 116 | text/speech embedding tables (official) |
| `onnx_turbo/language_model_fp16.onnx` + `.onnx_data` | CUDA EP | 0.2 + 635 | GPT-2-medium T3 + speech head, KV cache (official) |
| `models/conditional_decoder_fp16_det.onnx` + `conditional_decoder_fp16.onnx_data` | CUDA EP | 4.5 + 384 | official S3Gen flow (meanflow, 2 steps) + HiFT; its 4 random ops replaced by fixed noise buffers (`make_det_decoder.py`). `.onnx_data` = the official file, unchanged; it must sit next to the graph under exactly this name (here: a hard link) |
| `onnx_turbo/tokenizer.json` | host | 3.6 | GPT-2 byte-level BPE + 20 added tokens |
| `models/ve_mel_basis_40x201.npy` | host | 0.03 | VoiceEncoder mel filterbank |

Recommended set: **1.43 GB on disk**.  Low-VRAM set (LM `language_model_q4f16` 184 MB, decoder `conditional_decoder_q4f16_det`
163 + 4.8 MB, the rest the same): 0.76 GB.  NOT used: the official `speech_encoder_*.onnx` (§9).
`models/ref_encoder.onnx` (542 MB, fp32 weights) is equivalent to the w16 file [V] and only kept for reference.

## 2. Graph signatures [V] (`inspect_graphs.py`, `check_det.py`)

### language_model_fp16 (opset 21 + com.microsoft GroupQueryAttention; q4f16: MatMulNBits)
| name | dtype | shape |
|---|---|---|
| in `inputs_embeds` | float32 | [1, S, 1024] |
| in `attention_mask` | int64 | [1, P+S], all ones |
| in `position_ids` | int64 | [1, S] = P .. P+S−1 |
| in `past_key_values.{0..23}.{key,value}` | float16 | [1, 16, P, 64] (first call: P = 0, zero-size) |
| out `logits` | float32 | [1, S, 6563] (only the last row is used) |
| out `present.{0..23}.{key,value}` | float16 | [1, 16, P+S, 64] |

24 layers, 16 heads x 64, hidden 1024, learned absolute position embedding inside the graph (n_positions 8196) driven by
`position_ids`.  Input/output order: layer0 key, layer0 value, layer1 key, ...  The fp16 graph reproduces PyTorch T3:
teacher-forced on PyTorch's exact embeddings, top-1 agreement 98–100 %, KL(temperature-0.8 distributions) < 1e-4,
max |Δlogit| 0.02 (`check_lm.py`).

### embed_tokens_fp16 (opset 20)
`input_ids` int64 [B, L] → `inputs_embeds` float32 [B, L, 1024].  Semantics [V]: positions 0..L−3 = `text_emb[id]`;
**the last two positions** = `speech_emb[id]` with id 50256 mapped to 6561 (START).  `[[t]]` → `speech_emb[t]`;
a batch of pairs `[[a,b],[c,d],…]` → speech rows (used for the prompt).

### conditional_decoder_fp16_det (opset 17 + com.microsoft)
| name | dtype | shape |
|---|---|---|
| in `speech_tokens` | int64 | [1, Td + N] = decoder prompt tokens ++ speech tokens (++ 3 x 4299 on a final call) |
| in `speaker_embeddings` | float32 | [1, 192] x-vector |
| in `speaker_features` | float32 | [1, F, 80] prompt log-mel |
| out `waveform` | float32 | [1, (2·(Td+N) − F)·480], 24 kHz |

Normally F = 2·Td and the output is N·960 samples (40 ms per token).  F can be 2·Td+1 (e.g. a 9.716 s reference:
F = 485, Td = 242): the output is then 480 samples shorter and token i starts at sample `i·960 + (2·Td − F)·480`.
The first 480 output samples are zero and samples 480..960 a raised-cosine fade-in (PyTorch `trim_fade`, inside the graph).
Bit-identical across runs [V].  The noise buffers: CFM noise `cfm_noise_z`/`cfm_noise_gen` [1,80,1024] tiled along time,
HiFT `sine_phase_u` [1,9,1], `sine_noise` [1,9,48000] tiled — all indexed from the first frame/sample of the call.
Quality vs. the official random graph: simA +0.001 vs −0.002, WER 0.011 vs 0.020 [V].

### ref_encoder_w16
| tensor | dtype / shape | meaning |
|---|---|---|
| in `wav24_dec` | float32 [1, n24] | first ≤ 10 s (240 000 samples) of the −27 LUFS reference, 24 kHz |
| in `wav16_t3` | float32 [1, n16] | first ≤ 15 s (240 000 samples) of the same clip resampled to 16 kHz (§3) |
| out `dec_tokens` | int64 [1, Td] | S3 tokens of `wav24_dec` (resampled inside with torchaudio's 24k→16k sinc kernel), truncated to ⌊F/2⌋ |
| out `dec_feat` | float32 [1, F, 80] | S3Gen prompt mel, F = n24 // 480 |
| out `xvector` | float32 [1, 192] | CAMPPlus (Kaldi fbank 80, per-utterance mean removed) |
| out `t3_tokens` | int64 [1, Tt] | S3 tokens of `wav16_t3`, Tt = ⌈(n16 // 160) / 4⌉ |

Parity with PyTorch `prepare_conditionals` on 13 refs [V] (`check_refenc.py`): tokens 98.4–100 % identical (the rest
are FSQ rounding boundaries; PyTorch's own CPU/CUDA runs differ the same way), x-vector cos ≥ 0.999998, prompt mel
max |Δ| ≤ 7e-4.  ~0.4 s per voice on the CPU (+ ~4 s one-time session creation).

### ve_spk
`mel_partials` float32 [n_partials, 160, 40] → `ve_embedding` float32 [1, 256], `speaker_cond` float32 [1, 1, 1024].
Cosine to PyTorch `ve_embed` = 1.000000 on all 13 refs [V].

## 3. Reference voice registration [V] (`check_pre.py`, `check_refenc.py`)
1. Decode to mono float32, resample to 24 kHz (Python: `librosa.load(sr=24000)` = soxr_hq) [I: any good resampler].
2. Loudness: integrated loudness L (ITU-R BS.1770-4 with gating = pyloudnorm; Rust: `ebur128` crate, mode I);
   gain = 10^((−27 − L)/20), applied if finite and > 0 → `wav24`.
3. Length: PyTorch asserts > 5 s.  **If the clip is shorter than 6 s, tile it**: clip + 0.15 s silence + clip … until
   ≥ 6 s (§8 — untiled short clips make the LM babble).
4. `wav16` = resample(wav24, 24 k → 16 k) of the WHOLE clip (Python: `librosa.resample`, soxr_hq).
   `wav24_dec = wav24[:240000]`, `wav16_t3 = wav16[:240000]`.
5. VoiceEncoder input (`cbx.ve_partials`, whole `wav16`):
   * trim (= `librosa.effects.trim(top_db=20)`): zero-pad by 1024 both sides; frames of 2048, hop 512;
     p_i = mean(x²); db_i = 10·log10(max(1e-10, p_i)) − 10·log10(max(1e-10, max p)); keep samples
     [first·512, min(n, (last+1)·512)) over frames with db_i > −20.
   * STFT: reflect-pad 200 both sides, 400-sample frames, hop 160 (T = 1 + n//160 frames), periodic Hann(400),
     |rfft|² (201 bins); mel = power · `ve_mel_basis_40x201`ᵀ (no log, no normalisation) → [T, 40].
   * partials: n_w, rem = divmod(max(T − 160 + 77, 0), 77); if n_w == 0 or (rem + 83)/160 ≥ 0.8: n_w += 1;
     zero-pad / cut mel to 160 + 77·(n_w − 1) frames; partial i = frames [77i, 77i + 160).
   * `ve_spk.onnx` → `speaker_cond`.
6. `ref_encoder_w16` → dec_tokens, dec_feat, xvector, t3_tokens.
7. `cond` = concat(speaker_cond, speech_emb[t3_tokens[:375]]) → [1, 1+Tt, 1024] (speech rows via embed_tokens, pairs).

### Per-voice cache (store this; synthesis never runs the encoders)
| field | type | size |
|---|---|---|
| `speaker_cond` | f32 [1024] | 4 KB |
| `t3_tokens` | u16 [Tt ≤ 375] | ≤ 750 B |
| `dec_tokens` | u16 [Td ≤ 250] | ≤ 500 B |
| `dec_feat` | f32 [F ≤ 500, 80] | ≤ 160 KB |
| `xvector` | f32 [192] | 768 B |
| `model_id` | string | hash of ref_encoder / ve_spk / embed_tokens files (invalidate the cache when they change) |
At voice load build `cond` once (one embed_tokens call) and keep it in RAM (≤ 1.5 MB).  Longer than 15 s: only the
first 15 s (T3) / 10 s (decoder) are used, the VoiceEncoder averages the whole clip — prefer registering the best
10–15 s.

## 4. Text [V]
1. `punc_norm` (= `cbx.punc_norm`, copy of PyTorch): empty → "You need to add some text for me to talk."; uppercase the
   first char if lowercase; collapse whitespace; replace `…`→`, `, `:`→`,`, `—`/`–`→`-`, ` ,`→`,`, `“”`→`"`, `‘’`→`'`;
   strip trailing spaces; append `.` unless it ends with `.`, `!`, `?`, `-` or `,`.
2. `tokenizer.json` (Rust `tokenizers::Tokenizer::from_file`, encode WITH special tokens) → ids ending in
   `50256 50256` (template post-processor).  Paralinguistic tags are added tokens (ids in `golden/manifest.json`
   `tokenizer_special`, e.g. `[laugh]` 50275); the space before a tag becomes token 220 (identical to PyTorch).
3. `embed_tokens(ids)` → text rows, START, START.  **Drop the last row** → text rows + exactly one START row (PyTorch:
   no end-of-text token, one START).  The official recipe keeps both START rows (§9).

## 5. Generation loop [V] (`cbx.Engine.iter_tokens`)
* Prefill: `inputs_embeds` = concat(cond, text rows, START row) [1, L, 1024]; mask ones [1, L]; position_ids 0..L−1;
  zero-size KV.  Use `logits[0, −1]`.
* Each step: sample t; if t == 6562 (STOP) stop.  Otherwise feed `embed_tokens([[t]])` [1,1,1024], mask ones [1, P+1],
  position_ids [[P]], KV = previous `present.*`.  At most 1001 sampled tokens (first + 1000).
* Keep `present.*` on the GPU (IoBinding with device-allocated outputs, fed back as inputs); only logits return to the
  host.  This is also what makes the default arena behave (§7).
* Tokens for the decoder: all sampled tokens < 6561 (a sampled START is still fed back to the LM).
### Sampling (order of PyTorch `t3.inference_turbo`'s HF processors) — `cbx.Engine.probs/sample`
1. x = logits / 0.8 (float64)
2. top-k 1000: kth = 1000-th largest value; x[x < kth] = −∞ (ties kept)
3. top-p 0.95: sort ascending, p = softmax(sorted), remove entries where cumsum(p) ≤ 0.05, never the largest
4. repetition penalty 1.2 on the UNIQUE ids of the history (x ≥ 0 → x/1.2, x < 0 → x·1.2; −∞ stays): history =
   [6561] on the first step, then the generated tokens only (START is not part of it)
5. p = softmax(x); draw u ~ U[0,1); token = first index with cumsum(p) > u (clamp to 6562)
Golden cases (step, history, u → token, full processed p): `golden/<ref>/<text>/sampler_probs_step*.npy` + manifest.

## 6. Decoding
### Whole utterance [V]
`speech_tokens = dec_tokens ++ tokens ++ [4299, 4299, 4299]` → waveform.  No watermark (PyTorch adds Perth; optional).

### Streaming / chunked [V] (`cbx.Engine.stream`, `stream_eval.py`)
Facts that shape the scheme [V]: the flow uses full (non-causal) attention in both the conformer and the CFM estimator
(`static_chunk_size = 0`), and the ONNX graph is the `finalize=True` path, so decoding a token prefix does NOT give a
prefix of the full waveform.  The official graph is also random: two decodes of the same tokens differ as much as two
different renderings (waveform diff/rms 1.34, 40 ms log-spectral distance 8.2 dB), so cross-fading two decodes of the
official graph mixes uncorrelated waveforms.  A prefix decode's last ~7 tokens are rendered as an utterance ending
(LSD 10.9 dB at the last token vs ~6.3 dB from 8 tokens back, `sweep_prefix.py`).
Scheme (parameters = defaults of `Engine.stream`):
* chunk targets in tokens: first = 12 (or 20), then sizes 24, 48, 64, 64 … (`growth` 2, `max_chunk` 64)
* when `len(tokens) ≥ target + hold` (hold = 8): decode ALL tokens so far from the utterance start with the
  deterministic decoder (no silence tokens).  Because the noise buffers are indexed from the start of the call and the
  HiFT sine phase integrates from the start too, every re-decode uses the same noise for the same frame.
* junction J: the quietest 10 ms window (240 samples, step 120) in [target·960 + off − 960, target·960 + off],
  off = (2·Td − F)·480.  Emit up to J; keep the next 480 samples (20 ms) as the pending tail.
* next decode: find the shift s ∈ [−60, 60] samples maximising dot(tail, new[pos+s : pos+s+480]); output
  tail·cos-fade + new·sin-fade (equal power: fade_in(n) = sin(π/2·(n+0.5)/480), fade_out = reversed), then continue
  from pos+s+480.
* on STOP: one final decode with the 3 silence tokens, joined the same way, emitted to the end.
Measured (13 refs x 4 sentences, same tokens decoded whole vs. streamed): simA 0.746 vs 0.749, WER 0.013 vs 0.010.
Junction metrics on voiced joins (streamed / whole decode at the same place): median discontinuity 1.40 / 1.07, spectral-
flux percentile 0.88 / 0.85, no energy dips (fraction < 0.5: same as control).  For calibration, a hard cut at the
target scores 2.30 / 1.54 and 0.96 / 0.82, a 40 ms cross-fade 1.90 / 1.35 and 0.89 / 0.86 — the chosen join is the
closest to control, but **not proven inaudible** (listen list at the end).
Live latency (`stream_eval.py live`; LM fp16 CUDA, decoder det fp16 CUDA, embed CPU; 4 refs x 4 sentences;
**VRChat not running**): first=12 + decoder `cudnn_conv_algo_search=HEURISTIC`: time to first audio 0.39 s (max 0.40),
M sentence finished 1.98 s for ~5.5 s of audio, 0 underruns; first=20: 0.43 s, 0 underruns (with and without HEURISTIC).
Re-decoding from the start costs ~2x the decoder time of a whole decode (M: 1.9 s streamed vs 1.4 s whole) — still
~3x faster than real time.  With VRChat running only decoder micro-timings were taken (`probe_decoder.py`, official
decoder: 190–410 ms per call for 12–114 tokens); **re-measure `stream_eval.py live first=12` with VRChat running**.

## 7. Runtime settings, precision and VRAM [V] (`bench_cfg.py`)
| config (encoders on CPU in all) | simA − PT | WER (>0.15) | peak VRAM | ms/token | M total |
|---|---|---|---|---|---|
| embed fp16 **CPU**, LM fp16, dec fp16 (**recommended**) | +0.001 | 0.011 (0/104) | **1723 MiB** | 6.8–7.3 | 1.40–1.50 s |
| embed fp16 on GPU, LM fp16, dec fp16 | same | same | 2013 MiB | 7.5 | 1.58 s |
| embed q4f16 CPU, LM fp16, dec fp16 | +0.007 | **0.044 (2/52)** | 1723 | 7.3 | 1.45 s |
| embed fp16 CPU, LM fp16, dec q4f16 | −0.004 | 0.012 (0/52) | 1725 | 7.4 | 1.50 s |
| embed fp16 CPU, LM q4f16, dec fp16 | +0.001 | 0.014 (1/104) | 1723 | 6.7 | 1.39 s |
| embed q4f16 CPU, LM q4f16, dec q4f16 | +0.004 | 0.012 (0/52) | **1211 MiB** | 6.6 | 1.44 s |
| recommended + decoder arena kSameAsRequested, cuDNN heuristic, no max workspace ("lean") | | | 1971 MiB (worse) | | |
| all q4f16 + lean decoder | | | 1291 MiB (worse) | | |
| recommended with arena **kSameAsRequested** for all | | | **11.6 GB** (blow-up) | 21.6 | 3.5 s |
* Arena: keep the default `kNextPowerOfTwo`.  With `kSameAsRequested` the LM's KV tensors grow by one position per step
  and no freed block fits the next request → 11.6 GB and 3x slower (reproduced).
* Peak VRAM is set by arena growth, not by weight size: decoder q4 alone or LM q4 alone leave the peak at 1723 MiB;
  only both together drop it to 1211 MiB.
* embed_tokens: run it on the CPU EP (fp16 tables, 116 MB of RAM): saves 290 MiB of VRAM and is free in time.  Do NOT
  use the q4f16 table: its START row is off by 0.11 and text rows by 0.025 (max abs), and it produced the only gross
  errors in this study ("Hello, Master Meteor", a laugh tag rendered as "F-f-f-f").
* LM q4f16: teacher-forced against PyTorch it distorts every step (KL 0.05–0.09, top-1 agreement 71–79 %, max |Δlogit|
  2–3; `check_lm.py q4f16`); with the fixed conditioning that showed up as only slightly more WER (0.014 vs 0.011).
  Use it only if 1.7 GB is too much, and then together with the q4f16 decoder.
* Earlier "q4f16 adds words / truncates" (old pipeline: "Rest down." appended, "meet-" cut): explained by the q4 LM
  distortion plus the q4 embedding table (both used there), on top of the weak conditioning (§9).
* The speech_encoder is not on the GPU at all (its official fp16 version cost 522 MB of VRAM).

## 8. Reference length [V] (`short_ref.py`)
| reference | simA | simB | WER |
|---|---|---|---|
| us_f_50s full 12 s | 0.712 | 0.584 | — |
| us_f_50s first 3 s as is | 0.616 | 0.478 | **1.53** (babble) |
| us_f_50s 3 s tiled to 6.15 s | 0.701 | 0.555 | 0.000 |
| pack_bf one_clip 2.0 s as is | 0.501 | 0.503 | **3.25** (babble) |
| one_clip tiled to 6.3 s | 0.625 | 0.567 | 0.009 |
Rule: require ≥ 2 s of speech; below 6 s tile (clip + 0.15 s silence)… to ≥ 6 s; warn the user below ~5 s of
original audio (timbre similarity drops).  Above 15 s nothing breaks (see §3).

## 9. Why the official ONNX recipe sounded less like the reference (measured, `ablate.py`) [V]
| step (cumulative) | simA − PT | WER |
|---|---|---|
| official: constant speaker row, 2x START, 6 s prompt, official speech_encoder | −0.089 | 0.019 |
| + VoiceEncoder speaker row (`ve_spk.onnx`) | −0.078 | 0.009 |
| + one START row | −0.069 | 0.017 |
| + T3 prompt up to 15 s (speech_encoder on 6 s windows) | −0.063 | 0.017 |
| + PyTorch's exact T3 prompt tokens (oracle) | −0.025 | 0.011 |
| + PyTorch's exact decoder conditioning (oracle) | −0.002 | 0.013 |
| = this spec (ref_encoder reproduces both oracles) | +0.001 | 0.011 |
Root causes [V]:
1. `speech_encoder`'s first conditioning row is a constant equal to `spkr_enc.bias` (cos 1.0000), i.e. the export
   used a ZERO VoiceEncoder embedding: the LM never sees the speaker embedding (`pt_ref.py probe`, `trace_cond.py`).
2. The tokenizer appends 2x `<|endoftext|>`, which embed_tokens turns into two START rows; PyTorch has one.
3. `speech_encoder` returns only 150 prompt tokens (6 s) for the T3 prompt; PyTorch uses up to 375 (15 s).
4. Its S3 tokens disagree with PyTorch's (17–48 % identical, ~1 of 8 FSQ digits flipped per token, on CPU and CUDA alike;
   PyTorch's own two resampling paths agree 82–94 %), its x-vector has cos 0.88–0.92 to PyTorch's and its prompt mel
   differs by 0.03–0.04 mean abs.  Accurate T3 tokens were worth +0.04 simA, accurate decoder conditioning +0.02.
5. Not a cause: the LM graph (fp16 is exact), the sampler (same as HF), the decoder graph itself.

## 10. Test plan / golden data (`golden.py` → `golden/`, 26 MB) [V]
Config of the goldens: `golden/manifest.json` (`config`, per-ref sizes, LUFS/gain, VE trim bounds, tokenizer
special ids, sampler cases, stream chunk sizes).  Refs: `us_f_50s` (12.07 s), `greet` (8.91 s); texts `M` and `E` (tag).
| stage | files (`golden/<ref>/…`) | check | suggested tolerance [I] |
|---|---|---|---|
| decode + resample | `ref.wav` → `wav24_raw.npy` | own resampler | corr > 0.999 (resamplers differ) |
| loudness | manifest `lufs_raw`, `gain` → `wav24.npy` | ebur128 | ±0.05 LU |
| 16 k | `wav16.npy` | resample(`wav24`) | corr > 0.999 |
| VE | manifest `ve_trim_16k`, `ve_mel_partials.npy`, `ve_embedding.npy`, `speaker_cond.npy` | from golden `wav16.npy` | trim exact; partials rel 1e-4; cos > 0.9999 |
| ref encoder | `t3_tokens.npy`, `dec_tokens.npy`, `dec_feat.npy`, `dec_xvec.npy` | from golden `wav24/wav16` | tokens ≥ 98 % equal; feat 1e-3; cos > 0.9999 |
| cond | `cond.npy` | | 1e-3 on rows whose token matches |
| text | `<text>/input_ids.npy`, `text_embeds_with_start.npy` | tokenizer + embed (drop last row) | ids exact; 1e-3 |
| LM | `prefill_embeds.npy`, `tf_logits_first32.npy` (steps 0–31 teacher-forced with `tokens_seed0.npy`), `tf_argmax.npy` (all steps) | feed golden embeds | logits 0.05 abs; argmax ≥ 97 % |
| sampler | `sampler_probs_step*.npy` + manifest cases (step, u, token) | `probs()` on golden logits | p 1e-9; token exact |
| decoder | `wav_det.npy` (tokens_seed0 + 3 SIL) | det graph | 1e-3 abs on CUDA fp16 (bit-exact on the same EP/GPU) |
| streaming | `wav_stream.npy`, manifest `stream_chunk_samples` | stream on golden tokens | chunk sizes exact (±60 per join), samples 1e-3 |
End-to-end (cannot be bit-compared: the RNG differs): 13 refs x 4 sentences, simA − PyTorch within ±0.02 and WER ≤ 0.02
with `metrics.py` / `asr.py` on the Rust outputs.

## Listen (a human should check)
* parity: `out/pt/<ref>/<S|M|L|E>_s0.wav` vs `out/cfg_fp16/…` (same voice, different sample) and `out/base/…` (old recipe).
* low-VRAM option: `out/cfg_allq4/…` vs `out/cfg_fp16/…`.
* streaming joins with the largest excess discontinuity (`out/str_default/junctions.json`):
  `playful/S_s0.wav` @1.42 s, `sco_f_60s/E_s0.wav` @5.92 s, `us_m_teen/L_s0.wav` @3.36 s, `sco_f_60s/L_s0.wav` @8.48 s,
  `greet/L_s0.wav` @3.34 s, `love/E_s0.wav` @0.43 s; compare with `out/whole_det/` (same tokens) and `out/str_hard/` (naive cut).
* short references: `out/short_asis*/` (babble) vs `out/short_tile*/`.

## Open problems
* Streaming latency with VRChat running was not measured end to end (VRChat was closed during those runs).
* Junction inaudibility is supported by metrics only; the streamed decoder costs ~2x the GPU time of a whole decode
  (O(n²) re-decodes); for very long sentences cap chunk growth or split text at sentence boundaries first.
* `ref_encoder_w16` is 282 MB of CPU-only weights loaded only for registration; it can be dropped from RAM afterwards.
* Paralinguistic tags were checked only for `[laugh]` (and only via ASR).
