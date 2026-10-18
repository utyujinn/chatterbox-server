# Golden data

Made by the Python original (PyTorch Chatterbox Turbo and the ONNX graphs, `tools/golden.py`) for one reference voice, `us_f_50s` (a speaker of
the GLOBE corpus, CC0), and checked by the tests:

| file | checked by |
|---|---|
| `manifest.json` | loudness, trim bounds, sampler cases, the chunk sizes of the streamed decode |
| `us_f_50s/wav24_raw.npy`, `wav24.npy`, `wav16.npy` | `tests/frontend.rs` (loudness, resampling) |
| `us_f_50s/ve_mel_partials.npy` | `tests/frontend.rs` (the VoiceEncoder's input) |
| `us_f_50s/{t3_tokens,dec_tokens,dec_feat,dec_xvec,speaker_cond}.npy`, `ref.wav` | `tests/engine.rs` (registration; the decoder and the stream use them as the voice) |
| `us_f_50s/M/{text_embeds_with_start,prefill_embeds,tf_logits_first32,tokens_seed0}.npy` | `tests/engine.rs`, `tests/lm.rs` |
| `us_f_50s/M/sampler_probs_step*.npy` | `tests/engine.rs` (the sampler, to 1e-9) |
| `us_f_50s/M/wav_det.npy`, `wav_stream.npy` | `tests/engine.rs` (decoder, streaming) |

The golden audio of the decoder was made with ONNX Runtime 1.31; the app runs 1.23, which shares its GPU runtime with Irodori-TTS, so the
waveforms are compared by correlation, not sample by sample.
