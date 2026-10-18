"""Graph surgery (chatterbox venv): conditional_decoder_<dt>.onnx -> models/conditional_decoder_<dt>_det.onnx whose 4
random ops (CFM noise x2, HiFT sine initial phase, HiFT excitation noise) are replaced by slices of FIXED noise buffers,
tiled along time.  Same distribution as before, but deterministic, and -- because each buffer is indexed from the start of
the decoded sequence -- the same frame/sample always gets the same noise.  That is what lets chunked decoding re-decode
from the utterance start and cross-fade overlapping audio that is (near) identical.  Weights stay in the original
.onnx_data file (hard-linked next to the new graph; the graph proto only gains ~4 MB of noise initialisers).
Usage: python make_det_decoder.py fp16|q4f16"""
import os, sys
from pathlib import Path
import numpy as np, onnx
from onnx import helper, numpy_helper, TensorProto

HERE = Path(__file__).resolve().parent
OD = HERE.parent / "multiref" / "onnx_turbo"
dt = sys.argv[1]
src = OD / f"conditional_decoder_{dt}.onnx"
m = onnx.load(str(src), load_external_data=False)
g = m.graph
rng = np.random.default_rng(20251018)
# buffer time lengths (tiled when the input is longer): CFM 1024 mel frames (20 s), HiFT 48000 samples (2 s)
BUF = {"/RandomNormalLike_1": ("cfm_noise_z", (1, 80, 1024), "normal"),
       "/RandomNormalLike": ("cfm_noise_gen", (1, 80, 1024), "normal"),
       "/m_source/l_sin_gen/RandomUniformLike": ("sine_phase_u", (1, 9, 1), "uniform"),
       "/m_source/l_sin_gen/RandomNormalLike": ("sine_noise", (1, 9, 48000), "normal")}
# like-input layouts (verified with DEBUG_SHAPES): CFM (1, 80, 2*(prompt+new tokens)); phase (1, 9, 1); sine noise (1, 9, samples)
new_nodes = []
for n in list(g.node):
    if not n.op_type.startswith("Random"): continue
    name, shape, kind = BUF[n.name]
    arr = (rng.standard_normal(shape) if kind == "normal" else rng.random(shape)).astype(np.float32)
    g.initializer.append(numpy_helper.from_array(arr, name))
    like, out = n.input[0], n.output[0]
    p = n.name + "_det"
    rank = len(shape)
    tdim = 2                                     # all four are (B, C, T)
    if os.environ.get("DEBUG_SHAPES"):           # only expose the like-input shape as a graph output (no replacement)
        g.node.append(helper.make_node("Shape", [like], [p + "_dbg"]))
        g.output.append(helper.make_tensor_value_info(p + "_dbg", TensorProto.INT64, [None]))
        continue
    g.initializer.extend([numpy_helper.from_array(np.array(shape, np.int64), p + "_bufshape"),
                          numpy_helper.from_array(np.zeros(rank, np.int64), p + "_zeros"),
                          numpy_helper.from_array(np.array([shape[tdim] - 1], np.int64), p + "_lm1"),
                          numpy_helper.from_array(np.array([shape[tdim]], np.int64), p + "_len"),
                          numpy_helper.from_array(np.array([tdim], np.int64), p + "_tdim"),
                          numpy_helper.from_array(np.ones(rank, np.int64), p + "_ones")])
    # reps[tdim] = ceil(T / buflen); other reps 1 (batch is 1; non-time dims equal the buffer's)
    nn = [helper.make_node("Shape", [like], [p + "_shp"]),
          helper.make_node("Gather", [p + "_shp", p + "_tdim"], [p + "_T"], axis=0),
          helper.make_node("Add", [p + "_T", p + "_lm1"], [p + "_Tp"]),
          helper.make_node("Div", [p + "_Tp", p + "_len"], [p + "_rep"]),
          helper.make_node("ScatterElements", [p + "_ones", p + "_tdim", p + "_rep"], [p + "_reps"], axis=0),
          helper.make_node("Tile", [name, p + "_reps"], [p + "_tiled"]),
          helper.make_node("Slice", [p + "_tiled", p + "_zeros", p + "_shp"], [out])]
    idx = list(g.node).index(n)
    g.node.remove(n)
    for k, x in enumerate(nn): g.node.insert(idx + k, x)
    print("replaced", n.name, "->", name, shape, "time dim", tdim)
dst = HERE / "models" / f"conditional_decoder_{dt}_det.onnx"
data = HERE / "models" / f"conditional_decoder_{dt}.onnx_data"
if not data.exists(): os.link(OD / f"conditional_decoder_{dt}.onnx_data", data)   # hard link: no extra disk
onnx.save(m, str(dst))
print("saved", dst, os.path.getsize(dst) / 1e6, "MB graph")
