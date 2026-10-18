"""Golden tensors for the Rust port (multiref/.venv, GPU).  -> golden/<ref>/..., golden/manifest.json
Recommended configuration: ref_encoder_w16 (CPU) + ve_spk (CPU), embed_tokens fp16 (CPU), language_model fp16 (CUDA),
conditional_decoder_fp16_det (CUDA).  All arrays are .npy (little-endian, C order).  See SPEC.md 'Test plan'."""
import json, shutil
from pathlib import Path
import numpy as np, librosa, pyloudnorm, soundfile as sf
import cbx

HERE = Path(__file__).resolve().parent
G = HERE / "golden"; G.mkdir(exist_ok=True)
import os
os.environ.setdefault("REFENC", "ref_encoder_w16.onnx")
OPT = cbx.Opt(spk="ve", t3p="ref", start="one", dec="ref")
e = cbx.Engine({"speech_encoder": None}, cpu=("embed_tokens",), det=True)
REFS = {"us_f_50s": HERE.parent / "multiref" / "refs" / "us_f_50s_A.wav", "greet": HERE.parent / "pack_bf" / "refs" / "greet_A.wav"}
TEXTS = {"M": "I just got back from the convention, and honestly, I'm too tired to even talk about it right now.",
         "E": "Wait... you actually did that for me? [laugh] Oh my gosh, thank you so much!"}
man = {"config": {"ref_encoder": os.environ["REFENC"], "embed_tokens": "fp16 (CPU EP)", "language_model": "fp16 (CUDA EP)",
                  "conditional_decoder": "fp16_det (CUDA EP)", "opt": OPT.__dict__}, "refs": {}}
sv = lambda d, n, a: np.save(d / f"{n}.npy", np.ascontiguousarray(a))
for ref, path in REFS.items():
    d = G / ref; d.mkdir(exist_ok=True)
    shutil.copy(path, d / "ref.wav")
    raw, _ = librosa.load(str(path), sr=cbx.SR); sv(d, "wav24_raw", raw)
    lufs = pyloudnorm.Meter(cbx.SR).integrated_loudness(raw)
    w24 = cbx.load_ref(path); sv(d, "wav24", w24)
    w16 = librosa.resample(w24, orig_sr=cbx.SR, target_sr=cbx.SR16); sv(d, "wav16", w16)
    # trim bounds (in samples of wav16) reproduced from cbx.trim_silence
    yp = np.pad(w16.astype(np.float64), 1024); n = 1 + (len(yp) - 2048) // 512
    p = np.mean(yp[np.arange(2048)[None] + 512 * np.arange(n)[:, None]] ** 2, axis=1)
    db = 10 * np.log10(np.maximum(1e-10, p)) - 10 * np.log10(np.maximum(1e-10, p.max())); k = np.flatnonzero(db > -20)
    trim = [int(k[0] * 512), int(min(len(w16), (k[-1] + 1) * 512))]
    parts = cbx.ve_partials(w16); sv(d, "ve_mel_partials", parts)
    ve_emb, spk = e.ve.run(None, {"mel_partials": parts}); sv(d, "ve_embedding", ve_emb); sv(d, "speaker_cond", spk)
    v = e.prepare_voice(w24, OPT)
    for k2 in ["dec_tokens", "dec_feat", "dec_xvec", "cond", "t3_tokens"]: sv(d, k2, v[k2])
    man["refs"][ref] = {"source": str(path.name), "lufs_raw": lufs, "gain": float(10 ** ((-27 - lufs) / 20)), "n24": len(w24), "n16": len(w16),
                        "ve_trim_16k": trim, "n_partials": int(parts.shape[0]), "t3_tokens": int(v["t3_tokens"].shape[1]),
                        "dec_tokens": int(v["dec_tokens"].shape[1]), "dec_feat_frames": int(v["dec_feat"].shape[1]), "texts": {}}
    for tid, text in TEXTS.items():
        td = d / tid; td.mkdir(exist_ok=True)
        norm = cbx.punc_norm(text)
        ids = e.tok(norm, return_tensors="np")["input_ids"].astype(np.int64); sv(td, "input_ids", ids)
        te = e.text_embeds(text, OPT); sv(td, "text_embeds_with_start", te)
        emb = np.concatenate([v["cond"], te], 1).astype(np.float32); sv(td, "prefill_embeds", emb)
        # seeded run -> tokens; then teacher-forced logits of every step (one forward over the whole sequence)
        toks = e.gen_tokens(text, v, OPT, np.random.default_rng(0)); sv(td, "tokens_seed0", np.array(toks, np.int64))
        seq = np.array([cbx.START] + toks[:-1], np.int64) if toks else np.array([cbx.START])
        se = e.speech_rows(seq[1:]) if len(seq) > 1 else np.zeros((1, 0, 1024), np.float32)
        full = np.concatenate([emb, se], 1).astype(np.float32); L = full.shape[1]
        feed = {"inputs_embeds": full, "attention_mask": np.ones((1, L), np.int64), "position_ids": np.arange(L, dtype=np.int64)[None]}
        feed.update({k3: np.zeros((1, 16, 0, 64), np.float16) for k3 in e.kv_in})
        lg = e.S["language_model"].run(["logits"], feed)[0][0, emb.shape[1] - 1:]          # step 0 = last prefill row
        sv(td, "tf_logits_first32", lg[:32].astype(np.float32)); sv(td, "tf_argmax", lg.argmax(-1).astype(np.int64))
        # sampler unit cases: (step, history, u) -> token, with the processed distribution
        cases = []
        rng = np.random.default_rng(123)
        for st in [0, 1, 5, 17, 31]:
            if st >= len(lg): continue
            hist = [cbx.START] if st == 0 else toks[:st]
            u = float(rng.random()); pr = cbx.Engine.probs(lg[st], np.array(hist), OPT)
            tok = int(min(np.searchsorted(np.cumsum(pr), u, side="right"), len(pr) - 1))
            sv(td, f"sampler_probs_step{st}", pr.astype(np.float64))
            cases.append({"step": st, "history_len": len(hist), "u": u, "token": tok, "n_nonzero": int((pr > 0).sum())})
        wav = e.decode_det(toks, v, final=True); sv(td, "wav_det", wav); sf.write(td / "wav_det.wav", wav, cbx.SR)
        chunks = list(e.stream(iter(toks), v)); y = np.concatenate([c for c, _ in chunks])
        sv(td, "wav_stream", y); sf.write(td / "wav_stream.wav", y, cbx.SR)
        man["refs"][ref]["texts"][tid] = {"text": text, "normalized": norm, "n_input_ids": int(ids.shape[1]), "n_tokens": len(toks),
                                          "sampler_cases": cases, "stream_chunk_samples": [len(c) for c, _ in chunks],
                                          "stream_chunk_info": [i for _, i in chunks], "wav_det_samples": len(wav)}
        print(ref, tid, len(toks), "tokens", [len(c) for c, _ in chunks], flush=True)
import onnxruntime as ort
man["tokenizer_special"] = {t["content"]: t["id"] for t in json.loads((cbx.OD / "tokenizer.json").read_text(encoding="utf-8"))["added_tokens"]}
man["onnxruntime"] = ort.__version__
(G / "manifest.json").write_text(json.dumps(man, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)), encoding="utf-8")
print("done")
