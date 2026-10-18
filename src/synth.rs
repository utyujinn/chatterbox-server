// The rest of the engine: text, generation, decoding, streaming and the registration of a reference voice (see engine.rs for the parts
// that load the graphs and run the language model).

use crate::engine::{session, Engine, LmState, HIDDEN, MAX_NEW_TOKENS_LIMIT, SILENCE, START, STOP};
use crate::sampler::{draw, probabilities, Rng, SamplerSettings};
use crate::text::punc_norm;
use crate::voice::{Voice, MEL_BINS, T3_PROMPT_MAX};
use crate::stream::{Joiner, FIRST_CHUNK, HOLD, MAX_CHUNK, SAMPLES_PER_TOKEN};
use anyhow::{anyhow, ensure, Context, Result};
use ort::value::Tensor;

const REFERENCE_RATE: u32 = 24000;
const T3_RATE: u32 = 16000;
// The decoder's prompt is the first 10 s of the reference, the language model's the first 15 s
const DECODER_REFERENCE_SAMPLES: usize = 10 * REFERENCE_RATE as usize;
const T3_REFERENCE_SAMPLES: usize = 15 * T3_RATE as usize;
// Below TILE_TO_SECONDS a recording is repeated (see `register_voice`); below MIN_SECONDS of sound it is refused (nothing to repeat)
const TILE_TO_SECONDS: f64 = 6.0;
const MIN_SECONDS: f64 = 0.5;
const TILE_GAP_SECONDS: f64 = 0.15;

fn ort_error(err: ort::Error) -> anyhow::Error {
    anyhow!("{err}")
}

impl Engine {
    // The text as embeddings for the language model: normalized, tokenized (ids end in two end-of-text tokens), embedded (the last two
    // rows are START rows) and with the last row dropped, which leaves the text rows and one START row, as the original has them.
    pub fn text_embeds(&mut self, text: &str) -> Result<Vec<f32>> {
        let encoding = self.tokenizer.encode(punc_norm(text), true).map_err(|e| anyhow!("{e}"))?;
        let ids: Vec<i64> = encoding.get_ids().iter().map(|&i| i as i64).collect();
        ensure!(ids.len() >= 3 && ids.len() < 1500, "the text is {} tokens long", ids.len());
        let mut rows = self.embed(&ids, 1)?;
        rows.truncate((ids.len() - 1) * HIDDEN);
        Ok(rows)
    }

    // The language model's prompt for a voice: its speaker row, then the speech token rows of the reference.
    fn conditioning(&mut self, voice: &Voice) -> Result<Vec<f32>> {
        let mut cond = voice.speaker_cond.clone();
        let take = voice.t3_tokens.len().min(T3_PROMPT_MAX);
        cond.extend(self.speech_rows(&voice.t3_tokens[..take])?);
        Ok(cond)
    }

    // The speech tokens of one piece of text: the language model's loop, each token below START handed to `on_token` as it is sampled
    // (`on_token` returns false to stop). The repetition penalty looks at START on the first step and at the generated tokens after.
    pub fn generate(&mut self, voice: &Voice, text: &str, settings: &SamplerSettings, rng: &mut Rng, mut on_token: impl FnMut(&mut Engine, i64) -> Result<bool>) -> Result<()> {
        let mut embeds = self.conditioning(voice)?;
        embeds.extend(self.text_embeds(text)?);
        let mut state = LmState::new();
        let mut history: Vec<i64> = Vec::new();
        let mut logits = self.lm_step(&mut state, &embeds)?;
        for _ in 0..=MAX_NEW_TOKENS_LIMIT {
            let token = {
                let seen: &[i64] = if history.is_empty() { &[START] } else { &history };
                draw(&probabilities(&logits, seen, settings), rng.unit()) as i64
            };
            history.push(token);
            if token == STOP {
                return Ok(());
            }
            if token < START && !on_token(self, token)? {
                return Ok(());
            }
            let row = self.embed(&[token], 1)?;
            logits = self.lm_step(&mut state, &row)?;
        }
        Ok(())
    }

    // The audio of speech tokens: the voice's own prompt tokens in front, and for the end of an utterance three silence tokens behind.
    pub fn decode(&mut self, voice: &Voice, tokens: &[i64], finish: bool) -> Result<Vec<f32>> {
        let mut all = voice.dec_tokens.clone();
        all.extend_from_slice(tokens);
        if finish {
            all.extend([SILENCE; 3]);
        }
        let outputs = self
            .decoder
            .run(ort::inputs![
                "speech_tokens" => Tensor::from_array((vec![1usize, all.len()], all)).map_err(ort_error)?,
                "speaker_embeddings" => Tensor::from_array((vec![1usize, voice.xvector.len()], voice.xvector.clone())).map_err(ort_error)?,
                "speaker_features" => Tensor::from_array((vec![1usize, voice.frames(), MEL_BINS], voice.dec_feat.clone())).map_err(ort_error)?,
            ])
            .map_err(ort_error)?;
        let (_, wave) = outputs["waveform"].try_extract_tensor::<f32>().map_err(ort_error)?;
        Ok(wave.to_vec())
    }

    // One piece of text as a whole: all the tokens, then one decode.
    pub fn synthesize(&mut self, voice: &Voice, text: &str, settings: &SamplerSettings, rng: &mut Rng) -> Result<Vec<f32>> {
        let mut tokens = Vec::new();
        self.generate(voice, text, settings, rng, |_, t| {
            tokens.push(t);
            Ok(true)
        })?;
        ensure!(!tokens.is_empty(), "no speech came out of that text");
        self.decode(voice, &tokens, true)
    }

    // One piece of text as it is made: chunks of audio handed to `on_audio` while the language model is still at work, the first one after
    // a dozen tokens (about half a second of speech). Every chunk is a decode of everything so far; see stream.rs.
    pub fn synthesize_streaming(&mut self, voice: &Voice, text: &str, settings: &SamplerSettings, rng: &mut Rng, mut on_audio: impl FnMut(&[f32]) -> Result<()>) -> Result<()> {
        let mut stream = StreamDecoder::new(voice);
        self.generate(voice, text, settings, rng, |engine, token| {
            if let Some(chunk) = stream.push(engine, voice, token)? {
                on_audio(&chunk)?;
            }
            Ok(true)
        })?;
        on_audio(&stream.finish(self, voice)?)
    }

    // Registers a reference voice from a recording (WAV bytes): the signal processing of spec §3 and the two encoder graphs, which run on
    // the CPU and are created for this call alone (they are 290 MB of weights nothing else needs).
    pub fn register_voice(&mut self, recording: &[u8]) -> Result<Voice> {
        let (audio, rate) = crate::wavio::decode_wav(recording)?;
        let mut wav24 = crate::resample::resample(&audio, rate, REFERENCE_RATE);
        let (loudness, gain) = crate::audio::loudness_and_gain(&wav24, REFERENCE_RATE)?;
        ensure!(loudness.is_finite(), "there is no sound in the recording");
        // the speech, without the silence at its ends: it is that which has to be long enough
        let (from, to) = crate::audio::trim_bounds(&wav24);
        let seconds = to.saturating_sub(from) as f64 / REFERENCE_RATE as f64;
        ensure!(seconds >= MIN_SECONDS, "the recording holds {seconds:.1} s of speech; at least {MIN_SECONDS} s is needed");
        for v in wav24.iter_mut() {
            *v *= gain;
        }
        // The language model babbles on a prompt shorter than about 5 s (the original refuses it): a short recording is repeated, with a
        // little silence between the repeats, until it is long enough.
        if (wav24.len() as f64 / REFERENCE_RATE as f64) < TILE_TO_SECONDS {
            let clip = wav24.clone();
            let gap = vec![0f32; (TILE_GAP_SECONDS * REFERENCE_RATE as f64) as usize];
            while (wav24.len() as f64) < TILE_TO_SECONDS * REFERENCE_RATE as f64 {
                wav24.extend_from_slice(&gap);
                wav24.extend_from_slice(&clip);
            }
        }
        let wav16 = crate::resample::resample(&wav24, REFERENCE_RATE, T3_RATE);
        let basis = crate::npy::read(&self.data_dir.join("ve_mel_basis_40x201.npy"))?.f32()?;
        let (partials, n_partials) = crate::audio::ve_mel_partials(&wav16, &basis)?;

        let mut ve = session(self.environment, &self.models_dir.join("ve_spk.onnx"), false)?;
        let outputs = ve
            .run(ort::inputs!["mel_partials" => Tensor::from_array((vec![n_partials, crate::audio::VE_PARTIAL_FRAMES, 40usize], partials)).map_err(ort_error)?])
            .map_err(ort_error)?;
        let speaker_cond = outputs["speaker_cond"].try_extract_tensor::<f32>().map_err(ort_error)?.1.to_vec();
        drop(outputs);
        drop(ve);

        let dec_len = wav24.len().min(DECODER_REFERENCE_SAMPLES);
        let t3_len = wav16.len().min(T3_REFERENCE_SAMPLES);
        let mut encoder = session(self.environment, &self.models_dir.join("ref_encoder_w16.onnx"), false)?;
        let outputs = encoder
            .run(ort::inputs![
                "wav24_dec" => Tensor::from_array((vec![1usize, dec_len], wav24[..dec_len].to_vec())).map_err(ort_error)?,
                "wav16_t3" => Tensor::from_array((vec![1usize, t3_len], wav16[..t3_len].to_vec())).map_err(ort_error)?,
            ])
            .map_err(ort_error)?;
        let dec_tokens = outputs["dec_tokens"].try_extract_tensor::<i64>().map_err(ort_error)?.1.to_vec();
        let dec_feat = outputs["dec_feat"].try_extract_tensor::<f32>().map_err(ort_error)?.1.to_vec();
        let xvector = outputs["xvector"].try_extract_tensor::<f32>().map_err(ort_error)?.1.to_vec();
        let t3_tokens = outputs["t3_tokens"].try_extract_tensor::<i64>().map_err(ort_error)?.1.to_vec();
        let voice = Voice { speaker_cond, t3_tokens, dec_tokens, dec_feat, xvector };
        voice.validate().context("the encoders gave an unusable voice")?;
        Ok(voice)
    }
}

// The streamed decode of one utterance, fed one token at a time: `push` returns a chunk of audio when enough tokens have come in.
pub struct StreamDecoder {
    joiner: Joiner,
    tokens: Vec<i64>,
    target: usize,
    size: usize,
    // where the decoder's output starts: its prompt is 2 mel frames per token, and the prompt mel may be one frame longer
    offset: i64,
}

impl StreamDecoder {
    pub fn new(voice: &Voice) -> StreamDecoder {
        StreamDecoder {
            joiner: Joiner::new(),
            tokens: Vec::new(),
            target: FIRST_CHUNK,
            size: FIRST_CHUNK,
            offset: (2 * voice.dec_tokens.len() as i64 - voice.frames() as i64) * 480,
        }
    }

    pub fn push(&mut self, engine: &mut Engine, voice: &Voice, token: i64) -> Result<Option<Vec<f32>>> {
        self.tokens.push(token);
        if self.tokens.len() < self.target + HOLD {
            return Ok(None);
        }
        let wave = engine.decode(voice, &self.tokens, false)?;
        let stop = (self.target as i64 * SAMPLES_PER_TOKEN + self.offset).max(0) as usize;
        let chunk = self.joiner.join(&wave, Some(stop));
        self.size = (self.size * 2).min(MAX_CHUNK);
        self.target += self.size;
        Ok(Some(chunk))
    }

    // The last chunk: everything that is left, decoded with the closing silence.
    pub fn finish(&mut self, engine: &mut Engine, voice: &Voice) -> Result<Vec<f32>> {
        ensure!(!self.tokens.is_empty(), "no speech came out of that text");
        let wave = engine.decode(voice, &self.tokens, true)?;
        Ok(self.joiner.join(&wave, None))
    }
}
