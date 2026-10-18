# Licenses of the models

The ONNX files published with this project (release `models-v1`) are the official ONNX export of Chatterbox Turbo by Resemble AI, plus a
reference encoder exported from the PyTorch weights of the same model (`tools/export_ref_encoder.py`, `tools/shrink_refenc.py`) and a
decoder whose random nodes were replaced by fixed noise (`tools/make_det_decoder.py`). They keep the license of the original.

## Chatterbox Turbo (resemble-ai/chatterbox)

Language model (T3), speech embeddings, speech tokenizer (S3), speaker encoders (VoiceEncoder, CAM++), conditional decoder (S3Gen flow and
HiFT vocoder). https://github.com/resemble-ai/chatterbox, https://huggingface.co/ResembleAI/chatterbox-turbo

MIT License

Copyright (c) 2025 Resemble AI

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Parts of Chatterbox that come from other projects

Chatterbox's own README names them in its acknowledgements: CosyVoice (https://github.com/FunAudioLLM/CosyVoice, Apache-2.0), HiFT-GAN
(https://github.com/yl4579/HiFTNet, MIT), Real-Time-Voice-Cloning (https://github.com/CorentinJ/Real-Time-Voice-Cloning, MIT), S3Tokenizer
(https://github.com/xingchensong/S3Tokenizer, Apache-2.0). Their license texts are in those repositories; the weights above are Resemble AI's
and are distributed by it under the MIT License quoted above.

## Test data

`tests/golden/us_f_50s/ref.wav`: a speaker of the GLOBE corpus (CC0 1.0), cut to 12 s.
