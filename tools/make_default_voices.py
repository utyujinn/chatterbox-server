"""The built-in voices Mutelink speaks in without a license (default_1..default_5.cbvoice, release `models-v1`): five speakers of the GLOBE corpus
(CC0-1.0, derived from Mozilla Common Voice), registered by the server itself, so nothing here depends on the PyTorch original.
  python make_default_voices.py <server.exe> <models_dir> <data_dir> <ort_dll> <out_dir>   (CUDA 12 / cuDNN 9 DLLs on PATH)
The recordings are the experiment's refs (multiref/refs/<key>_A.wav, silence-trimmed utterances of one speaker joined with short gaps)."""
import json, struct, subprocess, sys
from pathlib import Path

REFS = Path(r"C:\Users\Utyujin\Dev\tts-test\emotion_en\multiref\refs")
# a mix of voices: US female (fifties), US male (teens), England male, Scottish female, Australian male
KEYS = ["us_f_50s", "us_m_teen", "uk_m_50s", "sco_f_60s", "aus_m_40s"]
server, models, data, ort, out = sys.argv[1:6]
out = Path(out)
out.mkdir(parents=True, exist_ok=True)
p = subprocess.Popen([server], stdin=subprocess.PIPE, stdout=subprocess.PIPE)


def call(header, body=b""):
    h = json.dumps(header).encode()
    p.stdin.write(struct.pack("<I", len(h)) + h + struct.pack("<I", len(body)) + body)
    p.stdin.flush()
    n = struct.unpack("<I", p.stdout.read(4))[0]
    answer = json.loads(p.stdout.read(n))
    m = struct.unpack("<I", p.stdout.read(4))[0]
    p.stdout.read(m)
    assert answer["ok"], answer
    return answer


call({"cmd": "load", "models_dir": models, "data_dir": data, "ort_dll": ort, "cuda": True})
for i, key in enumerate(KEYS, 1):
    target = out / f"default_{i}.cbvoice"
    call({"cmd": "register_voice", "out": str(target)}, (REFS / f"{key}_A.wav").read_bytes())
    print(key, "->", target, target.stat().st_size)
p.stdin.close()
p.wait()
