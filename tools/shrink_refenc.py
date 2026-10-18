"""models/ref_encoder.onnx (fp32) -> models/ref_encoder_w16.onnx: large float initialisers STORED as fp16 and Cast back
to fp32 inside the graph (compute stays fp32).  Halves the file; parity checked by check_refenc.py W16=1.  chatterbox venv."""
from pathlib import Path
import numpy as np, onnx
from onnx import helper, numpy_helper, TensorProto

HERE = Path(__file__).resolve().parent
m = onnx.load(str(HERE / "models" / "ref_encoder.onnx"))
g = m.graph
keep = []; casts = []; n = 0
for t in list(g.initializer):
    a = numpy_helper.to_array(t)
    # DFT / mel / fbank / resampler matrices stay fp32 (fp16 rounding there moved low-energy log-mel bins by up to 0.4)
    dsp = t.name.split(".")[-1] in ("w", "mel", "w24", "m24", "cosm", "sinm", "mb", "eye", "rk", "fc")
    if a.dtype == np.float32 and a.size >= 4096 and not dsp:
        g.initializer.remove(t)
        g.initializer.append(numpy_helper.from_array(a.astype(np.float16), t.name + "_f16"))
        casts.append(helper.make_node("Cast", [t.name + "_f16"], [t.name], to=TensorProto.FLOAT)); n += 1
for c in reversed(casts): g.node.insert(0, c)
onnx.save(m, str(HERE / "models" / "ref_encoder_w16.onnx"), save_as_external_data=False)
print("converted", n, "initialisers;", (HERE / "models" / "ref_encoder_w16.onnx").stat().st_size / 1e6, "MB")
