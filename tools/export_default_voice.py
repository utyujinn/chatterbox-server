"""The built-in voice of Chatterbox Turbo (conds.pt of ResembleAI/chatterbox-turbo, which the original uses when it is given no reference
recording) as a voice file of this server (src/voice.rs, "CBV1"), so that speaking needs no reference at all.
  python export_default_voice.py <out.cbvoice>       (the PyTorch chatterbox package, e.g. the experiment's chatterbox venv)
The speaker row is the VoiceEncoder embedding through the language model's `cond_enc.spkr_enc` projection, as in `pt_ref.py export`."""
import os, struct, sys
from pathlib import Path

import numpy as np
import torch
from chatterbox.tts_turbo import ChatterboxTurboTTS, Conditionals

out = Path(sys.argv[1])
model = ChatterboxTurboTTS.from_pretrained(device="cpu")
snapshot = next((Path(os.environ["HF_HOME"]) / "hub" / "models--ResembleAI--chatterbox-turbo" / "snapshots").iterdir())
conds = Conditionals.load(snapshot / "conds.pt").to("cpu")
torch.set_grad_enabled(False)
with torch.inference_mode():
    speaker_cond = model.t3.cond_enc.spkr_enc(conds.t3.speaker_emb.view(1, 256)).float().detach().cpu().numpy().ravel()
t3_tokens = conds.t3.cond_prompt_speech_tokens.detach().cpu().numpy().ravel().astype(np.int64)[:375]
dec_tokens = conds.gen["prompt_token"].detach().cpu().numpy().ravel().astype(np.int64)
dec_feat = conds.gen["prompt_feat"].float().detach().cpu().numpy().reshape(-1, 80)
xvector = conds.gen["embedding"].float().detach().cpu().numpy().ravel()
assert speaker_cond.shape == (1024,) and xvector.shape == (192,), (speaker_cond.shape, xvector.shape)
assert dec_feat.shape[0] in (2 * len(dec_tokens), 2 * len(dec_tokens) + 1), (dec_feat.shape, len(dec_tokens))
data = b"CBV1" + struct.pack("<III", len(t3_tokens), len(dec_tokens), dec_feat.shape[0])
data += speaker_cond.astype("<f4").tobytes() + t3_tokens.astype("<u2").tobytes() + dec_tokens.astype("<u2").tobytes()
data += dec_feat.astype("<f4").tobytes() + xvector.astype("<f4").tobytes()
out.parent.mkdir(parents=True, exist_ok=True)
out.write_bytes(data)
print(f"{out}: {len(data)} bytes; t3 tokens {len(t3_tokens)}, decoder tokens {len(dec_tokens)}, mel frames {dec_feat.shape[0]}")
