"""PyTorch ChatterboxTurboTTS reference side (lang_spike/chatterbox/.venv). Steps (argv selects: probe export dump gen tf):
  probe : is speech_encoder's constant speaker slot (/Constant_93) == spkr_enc(builtin conds.pt speaker_emb)?
  export: VoiceEncoder LSTM + mean/L2 + T3 cond_enc.spkr_enc -> models/ve_spk.onnx (fp32; the official ONNX build has
          no VoiceEncoder at all, see SPEC.md).  Input = mel partials computed on the host (spec'd), output = 256-d
          embedding and the 1024-d first conditioning row.
  dump  : per reference, the exact PyTorch conditioning tensors -> ref_pt/<ref>.npz (golden for the ONNX path).
  gen   : PyTorch baseline audio, 13 refs x 4 sentences x seeds 0,1 -> out/pt/<ref>/<sent>_s<seed>.wav (+tokens).
  tf    : teacher-forced speech logits for 'M' (seed-0 tokens) on 4 refs -> ref_pt/tf_<ref>.npz."""
import json, os, sys, time
from pathlib import Path
HERE = Path(__file__).resolve().parent
os.environ["HF_HOME"] = str(HERE.parent.parent / "lang_spike" / "chatterbox" / "hf_home")
os.environ["HF_HUB_OFFLINE"] = "1"
import numpy as np, torch, soundfile as sf, librosa
from chatterbox.tts_turbo import ChatterboxTurboTTS, punc_norm
from chatterbox.models.voice_encoder.voice_encoder import get_frame_step, get_num_wins
from chatterbox.models.voice_encoder.melspec import melspectrogram

MR = HERE.parent / "multiref"
REFS = {k: MR / "refs" / f"{k}_A.wav" for k in json.loads((MR / "refs" / "refs.json").read_text(encoding="utf-8"))}
REFS.update({k: HERE.parent / "pack_bf" / "refs" / f"{k}_A.wav" for k in ["greet", "love", "playful"]})
SENT = json.loads((MR / "sentences.json").read_text(encoding="utf-8"))
steps = sys.argv[1:] or ["probe", "export", "dump", "gen", "tf"]
(HERE / "models").mkdir(exist_ok=True); (HERE / "ref_pt").mkdir(exist_ok=True)

m = ChatterboxTurboTTS.from_pretrained(device="cuda")
t3, ve = m.t3, m.ve

if "probe" in steps:
    import onnx
    from onnx import numpy_helper
    g = onnx.load(str(MR / "onnx_turbo" / "speech_encoder_fp16.onnx"), load_external_data=True).graph
    c = next(numpy_helper.to_array(t) for t in g.initializer if t.name == "/Constant_93_output_0").astype(np.float32).ravel()
    snap = next((Path(os.environ["HF_HOME"]) / "hub" / "models--ResembleAI--chatterbox-turbo" / "snapshots").iterdir())
    from chatterbox.tts_turbo import Conditionals
    builtin = Conditionals.load(snap / "conds.pt").to("cuda")
    with torch.inference_mode():
        p = t3.cond_enc.spkr_enc(builtin.t3.speaker_emb.view(1, 256)).float().cpu().numpy().ravel()
        z = t3.cond_enc.spkr_enc.bias.float().cpu().numpy()
    cos = lambda a, b: float(a @ b / np.linalg.norm(a) / np.linalg.norm(b))
    print(f"Constant_93 vs spkr_enc(builtin conds speaker_emb): cos {cos(c, p):.5f} maxabs {np.abs(c - p).max():.4f}; vs spkr_enc bias only: cos {cos(c, z):.4f}")
    print("builtin cond prompt tokens", tuple(builtin.t3.cond_prompt_speech_tokens.shape), "gen prompt", tuple(builtin.gen["prompt_token"].shape))


class VESpk(torch.nn.Module):
    """partials (N,160,40) unscaled power-mel -> per-partial LSTM embeds, mean, L2 -> (1,256); spkr_enc -> (1,1,1024)."""
    def __init__(self, ve, lin):
        super().__init__(); self.ve, self.lin = ve, lin
    def forward(self, mel_partials):
        e = self.ve(mel_partials)                      # (N,256), each L2-normalised (ReLU before norm)
        e = e.mean(0, keepdim=True); e = e / torch.linalg.norm(e, dim=1, keepdim=True)
        return e, self.lin(e)[:, None]


def ve_partials(wav16):
    """Exact host-side preprocessing of VoiceEncoder.embeds_from_wavs(rate=1.3, trim_top_db=20, min_coverage=0.8)."""
    w = librosa.effects.trim(wav16, top_db=20)[0]
    mel = melspectrogram(w, ve.hp).T                   # (T,40)
    step = get_frame_step(0.5, 1.3, ve.hp)
    n, tgt = get_num_wins(len(mel), step, 0.8, ve.hp)
    if tgt > len(mel): mel = np.concatenate([mel, np.zeros((tgt - len(mel), 40))])
    mel = mel[:tgt].astype(np.float32)
    return np.stack([mel[i * step: i * step + 160] for i in range(n)])


if "export" in steps:
    mod = VESpk(ve, t3.cond_enc.spkr_enc).float().cpu().eval()
    x = torch.from_numpy(ve_partials(librosa.load(str(REFS["us_f_50s"]), sr=16000)[0]))
    torch.onnx.export(mod, (x,), str(HERE / "models" / "ve_spk.onnx"), input_names=["mel_partials"],
                      output_names=["ve_embedding", "speaker_cond"], dynamic_axes={"mel_partials": {0: "n_partials"}},
                      opset_version=17, dynamo=False)
    with torch.inference_mode():
        print("export ok; parity input", tuple(x.shape), [tuple(t.shape) for t in mod(x)])
    ve.to("cuda"); t3.to("cuda")  # VESpk(...).cpu() moved the shared modules

if "dump" in steps:
    for ref, path in REFS.items():
        m.prepare_conditionals(str(path))
        c = m.conds
        with torch.inference_mode():
            cond_emb = t3.prepare_conditioning(c.t3).float().cpu().numpy()
        wav24, _ = librosa.load(str(path), sr=24000)
        wav24 = m.norm_loudness(wav24, 24000)
        wav16 = librosa.resample(wav24, orig_sr=24000, target_sr=16000)
        parts = ve_partials(wav16)
        with torch.inference_mode():
            ve_from_parts = VESpk(ve, t3.cond_enc.spkr_enc)(torch.from_numpy(parts).cuda())[0].cpu().numpy()
        np.savez(HERE / "ref_pt" / f"{ref}.npz", wav24=wav24, wav16=wav16, ve_partials=parts,
                 ve_embed=c.t3.speaker_emb.float().cpu().numpy(), ve_from_parts=ve_from_parts,
                 t3_prompt_tokens=c.t3.cond_prompt_speech_tokens.cpu().numpy(), cond_emb=cond_emb,
                 gen_prompt_token=c.gen["prompt_token"].cpu().numpy(), gen_prompt_feat=c.gen["prompt_feat"].float().cpu().numpy(),
                 gen_embedding=c.gen["embedding"].detach().float().cpu().numpy())
        print(ref, "dur", len(wav24) / 24000, "cond", cond_emb.shape, "t3 prompt", tuple(c.t3.cond_prompt_speech_tokens.shape),
              "gen prompt", tuple(c.gen["prompt_token"].shape), "feat", tuple(c.gen["prompt_feat"].shape),
              "ve parts", parts.shape, "ve(parts) vs ve_embed cos", float(ve_from_parts.ravel() @ c.t3.speaker_emb.float().cpu().numpy().ravel()))

if "gen" in steps:
    out = HERE / "out" / "pt"; timing = []
    m.prepare_conditionals(str(REFS["us_f_50s"])); m.generate("Warm up.")
    for ref, path in REFS.items():
        m.prepare_conditionals(str(path)); d = out / ref; d.mkdir(parents=True, exist_ok=True)
        for s in SENT:
            for seed in (0, 1):
                torch.manual_seed(seed); torch.cuda.synchronize(); t = time.perf_counter()
                wav = m.generate(s["text"]); torch.cuda.synchronize(); dt = time.perf_counter() - t
                sf.write(d / f"{s['id']}_s{seed}.wav", wav.squeeze(0).numpy(), 24000)
                timing.append({"ref": ref, "sent": s["id"], "seed": seed, "s": dt, "audio_s": wav.shape[-1] / 24000})
        print(ref, flush=True)
    (out / "timing.json").write_text(json.dumps(timing, indent=1))

if "tf" in steps:
    M = next(s["text"] for s in SENT if s["id"] == "M")
    for ref in ["us_f_50s", "sco_f_60s", "noisy_f", "aus_m_40s"]:
        m.prepare_conditionals(str(REFS[ref]))
        ids = m.tokenizer(punc_norm(M), return_tensors="pt").input_ids.cuda()
        torch.manual_seed(0)
        with torch.inference_mode():
            tok = t3.inference_turbo(t3_cond=m.conds.t3, text_tokens=ids, temperature=0.8, top_k=1000, top_p=0.95, repetition_penalty=1.2)
            seq = torch.cat([torch.full((1, 1), 6561, device="cuda"), tok[:, :-1]], 1)
            emb, _ = t3.prepare_input_embeds(t3_cond=m.conds.t3, text_tokens=ids, speech_tokens=seq)
            h = t3.tfmr(inputs_embeds=emb)[0]
            n_sp = seq.shape[1]
            logits = t3.speech_head(h[:, -n_sp:]).float().cpu().numpy()[0]
        np.savez(HERE / "ref_pt" / f"tf_{ref}.npz", text_ids=ids.cpu().numpy(), tokens=tok.cpu().numpy(), logits=logits.astype(np.float32),
                 embeds=emb.float().cpu().numpy())
        print(ref, "tf tokens", tok.shape, "logits", logits.shape)

