"""Export an fp32 reference-voice encoder from the PyTorch Chatterbox Turbo weights (MIT) -> models/ref_encoder.onnx.
Replaces the official speech_encoder, whose S3 tokens / x-vector / prompt mels deviate from PyTorch (fp16 weights;
measured token agreement 17-45 % vs PyTorch's own 82-94 % self-consistency, x-vector cos 0.88-0.92) and which caps the
T3 prompt at 6 s.  Runs once per registered voice, so it is fp32 and meant for the CPU EP (no VRAM).

Inputs  wav24_dec (1, N24) float32 : first <=10 s of the -27 LUFS reference at 24 kHz  (S3Gen conditioning)
        wav16_t3  (1, N16) float32 : first <=15 s of the same clip resampled to 16 kHz  (T3 prompt tokens)
Outputs dec_tokens (1, Td) int64, dec_feat (1, 2*Td', 80) float32, xvector (1, 192) float32, t3_tokens (1, Tt) int64
Every signal-processing step is written as conv/matmul so the graph is self-contained (no STFT/complex ops):
  * 24k->16k: torchaudio.transforms.Resample(24000, 16000) kernel (sinc_interp_hann, width 6, rolloff 0.99) as a conv
  * S3 log-mel: |STFT(400, hop 160, periodic hann, reflect-centred)|^2 (last frame dropped) -> 128 Slaney mels ->
    log10(max(1e-10)) -> max(x, max(x)-8) -> (x+4)/4
  * Kaldi fbank (torchaudio.compliance.kaldi.fbank defaults, 80 bins) -> minus per-utterance mean -> CAMPPlus
  * S3Gen mel: reflect pad 720, |STFT(1920, hop 480, hann)| (+1e-9 inside sqrt) -> 80 Slaney mels (fmax 8 k) -> ln(max(1e-5))
Run with lang_spike/chatterbox/.venv.  `python export_ref_encoder.py` exports and checks parity with PyTorch."""
import math, os
from pathlib import Path
HERE = Path(__file__).resolve().parent
os.environ["HF_HOME"] = str(HERE.parent.parent / "lang_spike" / "chatterbox" / "hf_home"); os.environ["HF_HUB_OFFLINE"] = "1"
import numpy as np, torch, torch.nn.functional as F, librosa, torchaudio
import s3tokenizer.model_v2 as mv2
from chatterbox.tts_turbo import ChatterboxTurboTTS

torch.set_grad_enabled(False)


def rot_real(xq, xk, freqs_cis):  # freqs_cis here is REAL (T, 64, 2) = view_as_real(original complex buffer)
    cos, sin = freqs_cis[:, :, 0].unsqueeze(0).unsqueeze(2), freqs_cis[:, :, 1].unsqueeze(0).unsqueeze(2)
    D = xq.shape[-1]
    xq_r = torch.cat((-xq[..., D // 2:], xq[..., :D // 2]), -1); xk_r = torch.cat((-xk[..., D // 2:], xk[..., :D // 2]), -1)
    return xq * cos + xq_r * sin, xk * cos + xk_r * sin


mv2.apply_rotary_emb = rot_real   # FSMNMultiHeadAttention.qkv_attention looks it up at call time


def stft_conv(n_fft, win):
    k = np.arange(n_fft)[None]; f = np.arange(n_fft // 2 + 1)[:, None]
    ang = 2 * np.pi * f * k / n_fft
    w = np.concatenate([np.cos(ang) * win, -np.sin(ang) * win])[:, None, :]
    return torch.tensor(w, dtype=torch.float32)


class S3Tok(torch.nn.Module):
    def __init__(self, tk):
        super().__init__()
        self.enc, self.q = tk.encoder, tk.quantizer._codebook
        self.register_buffer("w", stft_conv(400, torch.hann_window(400).numpy()))
        self.register_buffer("mel", tk._mel_filters.float())
        self.register_buffer("fc", torch.view_as_real(tk.encoder.freqs_cis).float().contiguous())

    def forward(self, wav16):
        x = F.pad(wav16[:, None], (200, 200), mode="reflect")
        s = F.conv1d(x, self.w, stride=160)                      # (1, 402, 1 + N//160)
        p = s[:, :201] ** 2 + s[:, 201:] ** 2
        p = p[..., :-1]
        lm = torch.clamp(self.mel @ p, min=1e-10).log10()
        lm = torch.maximum(lm, lm.max() - 8.0)
        x = (lm + 4.0) / 4.0                                      # (1, 128, T)
        # AudioEncoderV2.forward for batch 1 / no padding (masks are all ones)
        x = F.gelu(self.enc.conv1(x)); x = F.gelu(self.enc.conv2(x)).permute(0, 2, 1)
        fc = self.fc[: x.shape[1]]
        ones = torch.ones_like(x[:, :, :1])                    # mask_pad (B,T,1); batch 1 without padding
        for b in self.enc.blocks: x = b(x, None, ones, fc)     # attention bias mask: none needed
        h = torch.tanh(self.q.project_down(x)) * 0.9990000128746033
        h = torch.round(h) + 1
        pw = torch.tensor([3.0 ** i for i in range(8)])
        return (h * pw).sum(-1).long()                            # (1, T)


class Fbank(torch.nn.Module):
    """torchaudio.compliance.kaldi.fbank(w, num_mel_bins=80) with defaults (dither 0, povey, preemph .97, snip_edges)."""
    def __init__(self):
        super().__init__()
        self.register_buffer("eye", torch.eye(400)[:, None, :])
        win = torch.hann_window(400, periodic=False).pow(0.85)
        self.register_buffer("win", win)
        k = np.arange(512)[None]; f = np.arange(257)[:, None]; ang = 2 * np.pi * f * k / 512
        self.register_buffer("cosm", torch.tensor(np.cos(ang).T, dtype=torch.float32)); self.register_buffer("sinm", torch.tensor(-np.sin(ang).T, dtype=torch.float32))
        mb, _ = torchaudio.compliance.kaldi.get_mel_banks(80, 512, 16000.0, 20.0, 0.0, 100.0, -500.0, 1.0)
        self.register_buffer("mb", F.pad(mb, (0, 1)).float())     # (80, 257)

    def forward(self, wav16):
        fr = F.conv1d(wav16[:, None], self.eye, stride=160)[0].transpose(0, 1)    # (F, 400)
        fr = fr - fr.mean(1, keepdim=True)
        fr = fr - 0.97 * torch.cat([fr[:, :1], fr[:, :-1]], 1)
        fr = F.pad(fr * self.win, (0, 112))
        re, im = fr @ self.cosm, fr @ self.sinm
        e = (re ** 2 + im ** 2) @ self.mb.T
        return torch.clamp(e, min=float(torch.finfo(torch.float32).eps)).log()     # (F, 80)


class RefEncoder(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        s3 = m.s3gen
        self.tok = S3Tok(s3.tokenizer); self.fb = Fbank(); self.camp = s3.speaker_encoder
        rs = torchaudio.transforms.Resample(24000, 16000)
        self.register_buffer("rk", rs.kernel.float()); self.rw = rs.width       # kernel (2,1,K)
        mel = librosa.filters.mel(sr=24000, n_fft=1920, n_mels=80, fmin=0, fmax=8000)
        self.register_buffer("m24", torch.tensor(mel, dtype=torch.float32))
        self.register_buffer("w24", stft_conv(1920, torch.hann_window(1920).numpy()))

    def resample(self, w24):  # = torchaudio.functional._apply_sinc_resample_kernel(orig 3, new 2)
        n = torch._shape_as_tensor(w24)[1]
        x = F.pad(w24[:, None], (self.rw, self.rw + 3))
        y = F.conv1d(x, self.rk, stride=3)                       # (1, 2, L)
        y = y.transpose(1, 2).reshape(1, -1)
        return y[:, : torch.div(2 * n + 2, 3, rounding_mode="floor")]   # ceil(2n/3) samples

    def mel24(self, w24):
        x = F.pad(w24[:, None], (720, 720), mode="reflect")
        s = F.conv1d(x, self.w24, stride=480)
        mag = torch.sqrt(s[:, :961] ** 2 + s[:, 961:] ** 2 + 1e-9)
        return torch.log(torch.clamp(self.m24 @ mag, min=1e-5)).transpose(1, 2)    # (1, T, 80)

    def forward(self, wav24_dec, wav16_t3):
        w16 = self.resample(wav24_dec)
        feat = self.mel24(wav24_dec)
        dtok = self.tok(w16)
        dtok = dtok[:, : torch.div(torch._shape_as_tensor(feat)[1], 2, rounding_mode="floor")]
        fb = self.fb(w16); fb = fb - fb.mean(0, keepdim=True)
        xv = self.camp(fb[None])
        return dtok, feat, xv, self.tok(wav16_t3)


if __name__ == "__main__":
    m = ChatterboxTurboTTS.from_pretrained(device="cpu")
    enc = RefEncoder(m).eval().float()
    # parity before export, against the PyTorch pieces on a real clip
    z = np.load(HERE / "ref_pt" / "us_f_50s.npz")
    w24 = torch.from_numpy(z["wav24"][: 10 * 24000])[None]; w16 = torch.from_numpy(z["wav16"][: 15 * 16000])[None]
    rs = torchaudio.transforms.Resample(24000, 16000)
    print("resample max|d|", (enc.resample(w24) - rs(w24)).abs().max().item(), enc.resample(w24).shape, rs(w24).shape)
    fb_ref = torchaudio.compliance.kaldi.fbank(rs(w24), num_mel_bins=80)
    print("fbank max|d|", (enc.fb(rs(w24)) - fb_ref).abs().max().item())
    out = enc(w24, w16)
    for name, a, b in [("dec_tokens", out[0], z["gen_prompt_token"]), ("t3_tokens", out[3], z["t3_prompt_tokens"])]:
        a = a.numpy()[0]; b = b[0]; n = min(len(a), len(b)); print(name, len(a), len(b), "agree", np.mean(a[:n] == b[:n]))
    print("feat max|d|", np.abs(out[1].numpy() - z["gen_prompt_feat"]).max(), "xvec cos", float(torch.nn.functional.cosine_similarity(out[2], torch.from_numpy(z["gen_embedding"])).item()))
    torch.onnx.export(enc, (w24, w16), str(HERE / "models" / "ref_encoder.onnx"), input_names=["wav24_dec", "wav16_t3"],
                      output_names=["dec_tokens", "dec_feat", "xvector", "t3_tokens"],
                      dynamic_axes={"wav24_dec": {1: "n24"}, "wav16_t3": {1: "n16"}, "dec_tokens": {1: "td"}, "dec_feat": {1: "tf"}, "t3_tokens": {1: "tt"}},
                      opset_version=17, dynamo=False)
    print("exported")
