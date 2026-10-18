# Third-party models, data and software

This repository holds the Rust runtime, the specification and the scripts that made the ONNX files. It contains **no model weights**: they are
published as assets of a GitHub release (`models-v1`) together with [MODEL_LICENSES.md](MODEL_LICENSES.md), which holds the license text of the
model.

| Component | License | Used for |
|---|---|---|
| [resemble-ai/chatterbox](https://github.com/resemble-ai/chatterbox) (Chatterbox Turbo) | MIT | the language model, speech tokenizer, voice encoders and decoder, as ONNX |
| [ResembleAI/chatterbox-turbo-ONNX](https://huggingface.co/ResembleAI/chatterbox-turbo-ONNX) | MIT | the official ONNX export of the text/speech embeddings, the language model and the conditional decoder (the decoder's random nodes are replaced by fixed noise) |
| CosyVoice, S3Tokenizer, HiFT-GAN, Real-Time-Voice-Cloning | see MODEL_LICENSES.md | the parts of Chatterbox that come from them (named in its acknowledgements) |
| [ONNX Runtime](https://github.com/microsoft/onnxruntime) | MIT | inference |
| NVIDIA CUDA / cuDNN runtime | NVIDIA proprietary licenses | **Not redistributed by this repository.** Check the CUDA EULA / cuDNN SLA before shipping them. |

The reference voice of the tests (`tests/golden/us_f_50s/ref.wav`) is a 12 s cut of a speaker of the GLOBE corpus (CC0, from Mozilla Common
Voice), used here as test data only.

## Voice cloning

The model makes speech in the voice of whatever reference recording it is given. Applications built on this repository should tell their users
to register only their own voice, or a voice whose owner has clearly agreed, and should not use it to impersonate anyone or to mislead others.
**This runtime does not add a watermark** to the audio (the original Python package adds an inaudible one, Perth; the ONNX graphs do not).
