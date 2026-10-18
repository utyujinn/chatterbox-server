// A registered reference voice: everything the language model and the decoder need from the recording, so that speaking never runs the
// reference encoders again (spec §3, "per-voice cache"). The recording itself is not kept.
//
// File: "CBV1", then Tt, Td, F as u32 (little endian), then speaker_cond (1024 f32), t3_tokens (Tt u16), dec_tokens (Td u16),
// dec_feat (F x 80 f32) and the x-vector (192 f32).

use anyhow::{bail, ensure, Context, Result};
use std::path::Path;

const MAGIC: &[u8; 4] = b"CBV1";
pub const SPEAKER_COND: usize = 1024;
pub const XVECTOR: usize = 192;
pub const MEL_BINS: usize = 80;
// The language model takes at most this many speech tokens of the reference as its prompt (15 s at 25 tokens per second)
pub const T3_PROMPT_MAX: usize = 375;

#[derive(Clone, Debug, PartialEq)]
pub struct Voice {
    pub speaker_cond: Vec<f32>,
    pub t3_tokens: Vec<i64>,
    pub dec_tokens: Vec<i64>,
    // [frames, 80] flattened
    pub dec_feat: Vec<f32>,
    pub xvector: Vec<f32>,
}

impl Voice {
    pub fn frames(&self) -> usize {
        self.dec_feat.len() / MEL_BINS
    }

    pub fn validate(&self) -> Result<()> {
        ensure!(self.speaker_cond.len() == SPEAKER_COND, "the speaker condition has {} numbers, not {SPEAKER_COND}", self.speaker_cond.len());
        ensure!(self.xvector.len() == XVECTOR, "the x-vector has {} numbers, not {XVECTOR}", self.xvector.len());
        ensure!(self.dec_feat.len() % MEL_BINS == 0 && !self.dec_feat.is_empty(), "the prompt mel is not a whole number of frames");
        ensure!(!self.t3_tokens.is_empty() && !self.dec_tokens.is_empty(), "the voice has no speech tokens");
        // the decoder's prompt is two mel frames per token, and the mel may be one frame longer
        let (td, frames) = (self.dec_tokens.len(), self.frames());
        ensure!(frames == 2 * td || frames == 2 * td + 1, "the prompt mel has {frames} frames for {td} prompt tokens");
        ensure!(self.t3_tokens.iter().chain(&self.dec_tokens).all(|&t| (0..6561).contains(&t)), "a speech token is out of range");
        ensure!(self.speaker_cond.iter().chain(&self.xvector).chain(&self.dec_feat).all(|v| v.is_finite()), "the voice holds numbers that are not numbers");
        Ok(())
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        self.validate()?;
        let mut bytes = Vec::new();
        bytes.extend_from_slice(MAGIC);
        for n in [self.t3_tokens.len(), self.dec_tokens.len(), self.frames()] {
            bytes.extend_from_slice(&(n as u32).to_le_bytes());
        }
        for v in &self.speaker_cond {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        for t in self.t3_tokens.iter().chain(&self.dec_tokens) {
            bytes.extend_from_slice(&(*t as u16).to_le_bytes());
        }
        for v in self.dec_feat.iter().chain(&self.xvector) {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir)?;
        }
        std::fs::write(path, bytes).with_context(|| format!("writing {}", path.display()))
    }

    pub fn load(path: &Path) -> Result<Voice> {
        let bytes = std::fs::read(path).with_context(|| format!("reading the voice {}", path.display()))?;
        Voice::parse(&bytes).with_context(|| format!("{} is not a usable voice file", path.display()))
    }

    pub fn parse(bytes: &[u8]) -> Result<Voice> {
        ensure!(bytes.len() >= 16 && &bytes[..4] == MAGIC, "wrong header");
        let count = |at: usize| u32::from_le_bytes([bytes[at], bytes[at + 1], bytes[at + 2], bytes[at + 3]]) as usize;
        let (tt, td, frames) = (count(4), count(8), count(12));
        ensure!(tt <= T3_PROMPT_MAX && td <= 1000 && frames <= 2000, "unreasonable sizes");
        let expected = 16 + 4 * SPEAKER_COND + 2 * (tt + td) + 4 * (frames * MEL_BINS + XVECTOR);
        if bytes.len() != expected {
            bail!("{} bytes, {} expected", bytes.len(), expected);
        }
        let mut at = 16;
        let floats = |at: &mut usize, n: usize| -> Vec<f32> {
            let v = bytes[*at..*at + 4 * n].chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect();
            *at += 4 * n;
            v
        };
        let tokens = |at: &mut usize, n: usize| -> Vec<i64> {
            let v = bytes[*at..*at + 2 * n].chunks_exact(2).map(|b| u16::from_le_bytes([b[0], b[1]]) as i64).collect();
            *at += 2 * n;
            v
        };
        let speaker_cond = floats(&mut at, SPEAKER_COND);
        let t3_tokens = tokens(&mut at, tt);
        let dec_tokens = tokens(&mut at, td);
        let dec_feat = floats(&mut at, frames * MEL_BINS);
        let xvector = floats(&mut at, XVECTOR);
        let voice = Voice { speaker_cond, t3_tokens, dec_tokens, dec_feat, xvector };
        voice.validate()?;
        Ok(voice)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> Voice {
        Voice {
            speaker_cond: (0..SPEAKER_COND).map(|i| i as f32 * 0.01).collect(),
            t3_tokens: (0..300).map(|i| (i * 7 % 6561) as i64).collect(),
            dec_tokens: (0..250).map(|i| (i * 11 % 6561) as i64).collect(),
            dec_feat: (0..500 * MEL_BINS).map(|i| (i % 97) as f32 - 40.0).collect(),
            xvector: (0..XVECTOR).map(|i| i as f32 * -0.02).collect(),
        }
    }

    #[test]
    fn a_voice_file_round_trips_and_a_damaged_one_is_refused() {
        let dir = std::env::temp_dir().join(format!("chatterbox-voice-test-{}", std::process::id()));
        let path = dir.join("a.cbvoice");
        let voice = sample();
        voice.save(&path).unwrap();
        assert_eq!(Voice::load(&path).unwrap(), voice);
        let mut damaged = std::fs::read(&path).unwrap();
        damaged.pop();
        assert!(Voice::parse(&damaged).is_err());
        assert!(Voice::parse(b"nope").is_err());
        let mut wrong = sample();
        wrong.xvector.pop();
        assert!(wrong.save(&path).is_err());
        let mut mismatched = sample();
        mismatched.dec_tokens.pop();
        assert!(mismatched.save(&path).is_err());
        let mut out_of_range = sample();
        out_of_range.t3_tokens[0] = 7000;
        assert!(out_of_range.save(&path).is_err());
        std::fs::remove_dir_all(dir).unwrap();
    }
}
