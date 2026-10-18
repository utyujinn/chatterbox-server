"""Chatterbox Turbo, ONNX-only reference pipeline (the Python reference the Rust port is checked against; SPEC.md).

Graphs: official ResembleAI/chatterbox-turbo-ONNX (speech_encoder / embed_tokens / language_model / conditional_decoder)
+ models/ve_spk.onnx (exported by pt_ref.py: the VoiceEncoder LSTM and T3's spkr_enc projection, which the official
export left out -- its speaker slot is the constant spkr_enc(0) = bias, i.e. no speaker identity reaches the LM).

Everything that differs from PyTorch ChatterboxTurboTTS is an Opt flag so the ablation (ablate.py) can measure it:
  spk      'zero' official constant row | 've'  VoiceEncoder embedding -> spkr_enc (PyTorch behaviour)
  t3p      'onnx150' official audio_features (first 6 s = 150 tokens) | 'win' encoder windows -> up to 15 s / 375 tokens
           (PyTorch behaviour) | 'pt' oracle tokens from ref_pt/<ref>.npz
  start    'two' official tokenizer (2x <|endoftext|> -> 2x START row) | 'one' text + one START (PyTorch)
  dec      'onnx' encoder on the first 10 s | 'pt' oracle S3Gen ref dict from ref_pt/<ref>.npz
Host-side preprocessing (loudness, resampling, VoiceEncoder mel/trim) is written out explicitly in numpy so it can be
ported line by line; each piece is checked against librosa / PyTorch in check_pre.py."""
import os, time
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np, onnxruntime as ort, pyloudnorm, librosa, soundfile as sf
from transformers import AutoTokenizer

ort.set_default_logger_severity(3)
HERE = Path(__file__).resolve().parent
OD =HERE.parent / "multiref" / "onnx_turbo"
SR, SR16 = 24000, 16000
START, STOP, SIL, EOT = 6561, 6562, 4299, 50256
T3_PROMPT_MAX = 375          # hp.speech_cond_prompt_len (15 s at 25 tokens/s)
ENC_T3_TOKENS = 150          # what the official speech_encoder returns in audio_features (6 s)
DEC_COND_S = 10              # ChatterboxTurboTTS.DEC_COND_LEN
WIN_STEP_S = 6               # 'win' mode: windows start every 6 s (= the 150 tokens each window contributes)


@dataclass
class Opt:
    spk: str = "ve"
    t3p: str = "win"
    start: str = "one"
    dec: str = "onnx"
    temperature: float = 0.8
    top_k: int = 1000
    top_p: float = 0.95
    rep: float = 1.2
    max_new: int = 1000


def punc_norm(text):  # = chatterbox.tts_turbo.punc_norm
    if len(text) == 0: return "You need to add some text for me to talk."
    if text[0].islower(): text = text[0].upper() + text[1:]
    text = " ".join(text.split())
    for a, b in [("…", ", "), (":", ","), ("—", "-"), ("–", "-"), (" ,", ","), ("“", '"'), ("”", '"'), ("‘", "'"), ("’", "'")]:
        text = text.replace(a, b)
    text = text.rstrip(" ")
    if not any(text.endswith(p) for p in {".", "!", "?", "-", ","}): text += "."
    return text


# ---------------- VoiceEncoder host preprocessing (explicit; = voice_encoder.embeds_from_wavs) ----------------
VE_NFFT, VE_HOP, VE_PART, VE_STEP = 400, 160, 160, 77   # step = round((16000/1.3)/160)
_VE_MEL = None


def ve_mel_basis():
    global _VE_MEL
    if _VE_MEL is None:   # Slaney-style mel (librosa default: htk=False, norm='slaney'), 40 x 201; shipped as a .npy
        p = HERE / "models" / "ve_mel_basis_40x201.npy"
        if not p.exists(): np.save(p, librosa.filters.mel(sr=SR16, n_fft=VE_NFFT, n_mels=40, fmin=0, fmax=8000).astype(np.float32))
        _VE_MEL = np.load(p).astype(np.float64)
    return _VE_MEL


def trim_silence(y, top_db=20.0, frame=2048, hop=512):
    """= librosa.effects.trim(y, top_db=20): frame RMS (centered, zero padded), dB re max power, keep frames > -top_db."""
    yp = np.pad(y.astype(np.float64), frame // 2)
    n = 1 + (len(yp) - frame) // hop
    idx = np.arange(frame)[None] + hop * np.arange(n)[:, None]
    p = np.mean(yp[idx] ** 2, axis=1)                                 # rms**2
    db = 10 * np.log10(np.maximum(1e-10, p)) - 10 * np.log10(np.maximum(1e-10, p.max()))
    nz = np.flatnonzero(db > -top_db)
    if len(nz) == 0: return y[:0]
    return y[nz[0] * hop: min(len(y), (nz[-1] + 1) * hop)]


def ve_partials(wav16):
    y = trim_silence(wav16).astype(np.float64)
    yp = np.pad(y, VE_NFFT // 2, mode="reflect")
    n = 1 + len(y) // VE_HOP
    win = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(VE_NFFT) / VE_NFFT)   # periodic Hann
    fr = yp[np.arange(VE_NFFT)[None] + VE_HOP * np.arange(n)[:, None]] * win
    mel = (np.abs(np.fft.rfft(fr, axis=1)) ** 2) @ ve_mel_basis().T     # (T, 40) power mel, no log, no normalisation
    n_w, rem = divmod(max(n - VE_PART + VE_STEP, 0), VE_STEP)
    if n_w == 0 or (rem + (VE_PART - VE_STEP)) / VE_PART >= 0.8: n_w += 1
    tgt = VE_PART + VE_STEP * (n_w - 1)
    mel = np.concatenate([mel, np.zeros((max(0, tgt - n), 40))])[:tgt].astype(np.float32)
    return np.stack([mel[i * VE_STEP: i * VE_STEP + VE_PART] for i in range(n_w)])


def load_ref(path):
    """24 kHz mono float, normalised to -27 LUFS (BS.1770 integrated), as ChatterboxTurboTTS.prepare_conditionals."""
    y, _ = librosa.load(str(path), sr=SR)
    lufs = pyloudnorm.Meter(SR).integrated_loudness(y)
    g = 10 ** ((-27.0 - lufs) / 20)
    return (y * g if np.isfinite(g) and g > 0 else y).astype(np.float32)


class GpuMem:
    """Dedicated GPU memory of THIS process (Windows 'GPU Process Memory' counter; works under WDDM where nvidia-smi
    shows N/A per process), sampled every second by a background Get-Counter.  .peak / .last in MiB."""
    def __init__(self):
        import subprocess, threading
        pid = os.getpid()
        # one Get-Counter per sample inside try/catch: -Continuous aborts when the instance does not exist yet
        cmd = (f"while ($true) {{ try {{ $s = (Get-Counter -Counter '\\GPU Process Memory(pid_{pid}_*)\\Dedicated Usage' -ErrorAction Stop).CounterSamples; "
               "[Console]::Out.WriteLine(($s | Measure-Object CookedValue -Sum).Sum) } catch { [Console]::Out.WriteLine('0') }; "
               "[Console]::Out.Flush(); Start-Sleep -Milliseconds 300 }")
        self.p = subprocess.Popen(["pwsh", "-NoProfile", "-Command", cmd], stdout=subprocess.PIPE, text=True)
        self.peak = self.last = 0.0
        def rd():
            for line in self.p.stdout:
                try: v = float(line) / 2 ** 20
                except ValueError: continue
                self.last = v; self.peak = max(self.peak, v)
        threading.Thread(target=rd, daemon=True).start()

    def settle(self, s=2.5):
        time.sleep(s); return self.last

    def stop(self):
        self.p.kill()


class Engine:
    """dtypes: {graph: 'fp16'|'q4f16'|None (None = do not load)}.  cpu: graphs to put on the CPU EP.
    det: load models/conditional_decoder_<dt>_det.onnx INSTEAD of the official decoder (synth then uses it too)."""
    def __init__(self, dtypes=None, providers=None, arena="kNextPowerOfTwo", enc_cpu=False, mem_limit_gb=None, cpu=(), det=False, dec_cuda=None):
        ort.preload_dlls()
        self.dt = {"speech_encoder": "fp16", "embed_tokens": "fp16", "language_model": "fp16", "conditional_decoder": "fp16", **(dtypes or {})}
        so = ort.SessionOptions(); so.log_severity_level = 3
        co = {"arena_extend_strategy": arena}
        if mem_limit_gb: co["gpu_mem_limit"] = str(int(mem_limit_gb * 2 ** 30))
        gpu = providers or [("CUDAExecutionProvider", co), "CPUExecutionProvider"]
        self.gpu = gpu
        self.S = {}
        for n, d in self.dt.items():
            if d is None or (det and n == "conditional_decoder"): continue
            p = ["CPUExecutionProvider"] if ((enc_cpu and n == "speech_encoder") or n in cpu) else gpu
            self.S[n] = ort.InferenceSession(str(OD / f"{n}_{d}.onnx"), so, providers=p)
        # dec_cuda: CUDA EP options for the decoder session only (each session has its own arena)
        if det: self.load_det_decoder(self.dt["conditional_decoder"], [("CUDAExecutionProvider", {**co, **dec_cuda}), "CPUExecutionProvider"] if dec_cuda else gpu)
        self.ve = ort.InferenceSession(str(HERE / "models" / "ve_spk.onnx"), so, providers=["CPUExecutionProvider"])
        self.tok = AutoTokenizer.from_pretrained(str(OD))
        lm = self.S["language_model"]
        self.kv_in = [i.name for i in lm.get_inputs() if i.name.startswith("past_key_values")]
        self.kv_out = [o.name for o in lm.get_outputs() if o.name.startswith("present")]
        self.semb = None

    # ---------------- reference voice -> cached conditioning ----------------
    def encode(self, wav24):
        return self.S["speech_encoder"].run(None, {"audio_values": wav24[None].astype(np.float32)})

    def prepare_voice(self, wav24, opt: Opt, ref_pt=None):
        """Returns the per-voice cache: everything synthesis needs, so the encoders never run again for this voice."""
        assert len(wav24) / SR > 5.0 or os.environ.get("ALLOW_SHORT"), "reference must be > 5 s (PyTorch asserts this)"
        t = {}
        t0 = time.perf_counter()
        if opt.t3p == "ref" or opt.dec == "ref":
            return self.prepare_voice_ref(wav24, opt)
        feat, dec_tok, xvec, sfeat = self.encode(wav24[: DEC_COND_S * SR])
        t["enc_s"] = time.perf_counter() - t0
        # T3 prompt rows (speech_emb of the prompt tokens)
        if opt.t3p == "onnx150":
            prompt = feat[:, 1:]
        elif opt.t3p == "win":
            rows = [feat[:, 1:]]; s = WIN_STEP_S
            while sum(r.shape[1] for r in rows) < T3_PROMPT_MAX and s * SR < min(len(wav24), 15 * SR) - SR // 25:
                f2 = self.encode(wav24[s * SR: min(len(wav24), 15 * SR, (s + DEC_COND_S) * SR)])[0]
                rows.append(f2[:, 1:]); s += WIN_STEP_S
            prompt = np.concatenate(rows, 1)[:, :T3_PROMPT_MAX]
            # PyTorch tokenises min(len,15 s) of the 16 kHz clip: ceil((n16 // 160) / 4) tokens (mel hop 160, 4x down)
            n16 = (min(len(wav24), 15 * SR) * 2) // 3
            prompt = prompt[:, : -(-(n16 // 160) // 4)]
        elif opt.t3p == "pt":
            if self.semb is None: self.semb = np.load(HERE / "models" / "speech_emb_fp16.npy").astype(np.float32)
            prompt = self.semb[ref_pt["t3_prompt_tokens"][0]][None]
        t["t3_prompt_s"] = time.perf_counter() - t0
        if opt.spk == "zero":
            spk = feat[:, :1]
        else:
            # PyTorch: librosa.resample(24k->16k, default soxr_hq) of the whole loudness-normalised clip, no length cut
            wav16 = librosa.resample(wav24, orig_sr=SR, target_sr=SR16)
            spk = self.ve.run(None, {"mel_partials": ve_partials(wav16)})[1]
        t["prep_s"] = time.perf_counter() - t0
        cond = np.concatenate([spk, prompt], 1).astype(np.float32)
        if opt.dec == "pt":
            dec_tok, sfeat, xvec = ref_pt["gen_prompt_token"], ref_pt["gen_prompt_feat"], ref_pt["gen_embedding"]
        return {"cond": cond, "dec_tokens": dec_tok.astype(np.int64), "dec_xvec": xvec.astype(np.float32), "dec_feat": sfeat.astype(np.float32), "t": t}

    def speech_rows(self, tokens):
        """speech_emb rows for token ids via embed_tokens itself (its last 2 positions are speech rows): feed the ids
        as a batch of pairs, so the prompt uses exactly the LM-side table of the configured dtype."""
        t = np.asarray(tokens, np.int64); n = len(t)
        if n % 2: t = np.append(t, 0)
        return self.S["embed_tokens"].run(None, {"input_ids": t.reshape(-1, 2)})[0].reshape(1, -1, 1024)[:, :n]

    def prepare_voice_ref(self, wav24, opt):
        """Fixed path (opt.t3p == opt.dec == 'ref'): models/ref_encoder.onnx (fp32, CPU) = PyTorch prepare_conditionals."""
        if not hasattr(self, "renc"):
            so = ort.SessionOptions(); so.log_severity_level = 3
            self.renc = ort.InferenceSession(str(HERE / "models" / os.environ.get("REFENC", "ref_encoder.onnx")), so, providers=["CPUExecutionProvider"])
        t = {}; t0 = time.perf_counter()
        wav16 = librosa.resample(wav24, orig_sr=SR, target_sr=SR16)       # PyTorch: librosa soxr_hq, whole clip
        dtok, dfeat, xvec, t3tok = self.renc.run(None, {"wav24_dec": wav24[None, : DEC_COND_S * SR], "wav16_t3": wav16[None, : 15 * SR16]})
        t["ref_encoder_s"] = time.perf_counter() - t0
        spk = self.ve.run(None, {"mel_partials": ve_partials(wav16)})[1]
        t["prep_s"] = time.perf_counter() - t0
        cond = np.concatenate([spk, self.speech_rows(t3tok[0, :T3_PROMPT_MAX])], 1).astype(np.float32)
        return {"cond": cond, "dec_tokens": dtok.astype(np.int64), "dec_xvec": xvec.astype(np.float32), "dec_feat": dfeat.astype(np.float32),
                "t3_tokens": t3tok, "t": t}

    # ---------------- text -> speech tokens ----------------
    def text_embeds(self, text, opt):
        ids = self.tok(punc_norm(text), return_tensors="np")["input_ids"].astype(np.int64)  # ends with 2x EOT
        e = self.S["embed_tokens"].run(None, {"input_ids": ids})[0]   # last 2 rows = speech_emb(START)
        return e[:, :-1] if opt.start == "one" else e

    @staticmethod
    def probs(logits, hist, opt):
        """HF LogitsProcessorList order of t3.inference_turbo: temperature, top-k, top-p, repetition penalty; softmax."""
        x = logits.astype(np.float64) / opt.temperature
        if opt.top_k < len(x):
            kth = np.partition(x, -opt.top_k)[-opt.top_k]; x = np.where(x < kth, -np.inf, x)   # ties at the k-th kept
        order = np.argsort(x, kind="stable"); p = np.exp(x[order] - x[order][-1]); p /= p.sum()
        rm = np.cumsum(p) <= (1 - opt.top_p); rm[-1] = False                                   # ascending cumsum
        x[order[rm]] = -np.inf
        h = np.unique(hist); s = x[h]; x[h] = np.where(s < 0, s * opt.rep, s / opt.rep)        # after filtering
        p = np.exp(x - x.max()); return p / p.sum()

    @staticmethod
    def sample(logits, hist, rng, opt):
        """Inverse CDF with one uniform u in [0,1): first index whose cumulative probability exceeds u."""
        p = Engine.probs(logits, hist, opt)
        return int(min(np.searchsorted(np.cumsum(p), rng.random(), side="right"), len(p) - 1))

    def gen_tokens(self, text, voice, opt, rng):
        return list(self.iter_tokens(text, voice, opt, rng))

    # ---------------- speech tokens -> waveform ----------------
    def decode(self, tokens, voice, final=True):
        tk = np.asarray(tokens, np.int64)[None]
        if final: tk = np.concatenate([tk, np.full((1, 3), SIL, np.int64)], 1)
        toks = np.concatenate([voice["dec_tokens"], tk], 1)
        return self.S["conditional_decoder"].run(None, {"speech_tokens": toks, "speaker_embeddings": voice["dec_xvec"], "speaker_features": voice["dec_feat"]})[0][0]

    # ---------------- streaming (chunked) decoding ----------------
    def load_det_decoder(self, dt="fp16", providers=None):
        """models/conditional_decoder_<dt>_det.onnx (make_det_decoder.py): random ops -> fixed noise buffers."""
        so = ort.SessionOptions(); so.log_severity_level = 3
        self.det = ort.InferenceSession(str(HERE / "models" / f"conditional_decoder_{dt}_det.onnx"), so, providers=providers or self.gpu)

    def decode_det(self, tokens, voice, final):
        tk = np.asarray(tokens, np.int64)
        if final: tk = np.concatenate([tk, np.full(3, SIL, np.int64)])
        return self.det.run(None, {"speech_tokens": np.concatenate([voice["dec_tokens"], tk[None]], 1),
                                   "speaker_embeddings": voice["dec_xvec"], "speaker_features": voice["dec_feat"]})[0][0]

    def stream(self, token_iter, voice, sc=None):
        """Chunked decoding. token_iter yields speech tokens one by one (live LM or a replayed list).
        Every chunk re-decodes ALL tokens so far from the utterance start with the deterministic decoder (so noise and
        HiFT sine phase are indexed identically in every decode) but emits only audio up to `hold` tokens before the
        newest token (the last tokens of a prefix decode are rendered as an utterance ending).  The junction is placed
        at the quietest 10 ms frame in [target - search, target], the new decode is shifted by up to +-align samples to
        best correlate with the previous decode's pending tail, and the two are joined with an equal-power cross-fade.
        Yields (audio float32, info dict)."""
        sc = {"first": 12, "growth": 2.0, "max_chunk": 64, "hold": 8, "xfade": 480, "search": 960, "align": 60, **(sc or {})}
        off = (2 * voice["dec_tokens"].shape[1] - voice["dec_feat"].shape[1]) * 480   # decoder output starts at mel frame F
        toks, tail, pos, size = [], None, 0, sc["first"]
        target = sc["first"]
        X = sc["xfade"]
        fade = np.sin(0.5 * np.pi * (np.arange(X) + 0.5) / X)                       # equal power: fade_in^2 + fade_out^2 = 1

        def join(w, stop):
            nonlocal tail, pos
            if tail is None:
                head, start = [], 0
            else:  # align the new decode to the pending tail, then cross-fade over X samples
                A = sc["align"]
                best = max(range(-A, A + 1), key=lambda s: float(np.dot(tail, w[pos + s: pos + s + X])) if pos + s >= 0 else -1e9)
                start = pos + best
                head = [tail * fade[::-1] + w[start: start + X] * fade]
                start += X
            if stop is None:                       # final chunk: everything to the end
                out = np.concatenate(head + [w[start:]]); tail = None; return out
            J = stop
            lo = max(start + X, stop - sc["search"])
            if lo < stop:                          # quietest 10 ms frame (240 samples) in the search window
                e = [np.sum(w[j: j + 240] ** 2) for j in range(lo, stop - 240 + 1, 120)]
                if e: J = lo + 120 * int(np.argmin(e)) + 120
            out = np.concatenate(head + [w[start: J]]); tail = w[J: J + X].copy(); pos = J
            return out

        for t in token_iter:
            toks.append(t)
            if len(toks) >= target + sc["hold"]:
                w = self.decode_det(toks, voice, final=False)
                yield join(w, target * 960 + off).astype(np.float32), {"n_tokens": len(toks), "target": target}
                size = min(int(size * sc["growth"]), sc["max_chunk"]); target += size
        w = self.decode_det(toks, voice, final=True)
        yield join(w, None).astype(np.float32), {"n_tokens": len(toks), "final": True}

    def iter_tokens(self, text, voice, opt, rng):
        """LM generation loop as a generator of speech tokens (< START). KV cache stays on the GPU (IOBinding); logits
        come back to the host for sampling. Repetition-penalty history = [START] on step 0, then generated tokens only
        (PyTorch t3.inference_turbo)."""
        S = self.S; lm = S["language_model"]
        emb = np.concatenate([voice["cond"], self.text_embeds(text, opt)], 1).astype(np.float32)
        L = emb.shape[1]; mask = np.ones((1, L), np.int64); pos = np.arange(L, dtype=np.int64)[None]
        past = None; gen = []
        for step in range(opt.max_new + 1):
            b = lm.io_binding()
            b.bind_cpu_input("inputs_embeds", emb); b.bind_cpu_input("attention_mask", mask); b.bind_cpu_input("position_ids", pos)
            for j, n in enumerate(self.kv_in):
                if past is None: b.bind_cpu_input(n, np.zeros((1, 16, 0, 64), np.float16))
                else: b.bind_ortvalue_input(n, past[j])
            b.bind_output("logits", "cpu")
            for n in self.kv_out: b.bind_output(n, "cuda")
            lm.run_with_iobinding(b)
            outs = b.get_outputs()
            nxt = self.sample(outs[0].numpy()[0, -1], np.array(gen if gen else [START]), rng, opt)
            gen.append(nxt)
            if nxt == STOP: return
            if nxt < START: yield nxt
            past = outs[1:]
            emb = S["embed_tokens"].run(None, {"input_ids": np.array([[nxt]], np.int64)})[0]
            mask = np.concatenate([mask, np.ones((1, 1), np.int64)], 1); pos = pos[:, -1:] + 1

    def synth(self, text, voice, opt, rng):
        t0 = time.perf_counter()
        toks = self.gen_tokens(text, voice, opt, rng)
        t1 = time.perf_counter()
        wav = self.decode(toks, voice) if "conditional_decoder" in self.S else self.decode_det(toks, voice, final=True)
        t2 = time.perf_counter()
        return wav, toks, {"lm_s": t1 - t0, "dec_s": t2 - t1, "s": t2 - t0, "n_tokens": len(toks)}
