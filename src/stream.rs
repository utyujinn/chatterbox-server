// Joining the chunks of a streamed decode (spec §6). The decoder does not give a prefix of the full waveform for a prefix of the tokens,
// so every chunk is a decode of ALL tokens so far, from the start of the utterance (with the decoder's fixed noise, so the same frame
// is rendered the same way every time); this takes the part of it that is new and joins it to what was already sent:
//   * the junction is the quietest 10 ms window in the last 960 samples before the chunk's target,
//   * the new decode is shifted by up to 60 samples to best match the pending tail of the previous one,
//   * the two are cross-faded over 480 samples with equal power.

pub const CROSSFADE: usize = 480;
const SEARCH: usize = 960;
const ALIGN: i64 = 60;
const QUIET_WINDOW: usize = 240;
const QUIET_STEP: usize = 120;

// Tokens per chunk: the first target and the growth of the next ones, and how many of the newest tokens are held back (the end of a
// prefix decode is rendered as the end of an utterance).
pub const FIRST_CHUNK: usize = 12;
pub const MAX_CHUNK: usize = 64;
pub const HOLD: usize = 8;
// 40 ms of audio per speech token
pub const SAMPLES_PER_TOKEN: i64 = 960;

pub struct Joiner {
    tail: Option<Vec<f32>>,
    position: usize,
    fade: Vec<f32>,
}

impl Joiner {
    pub fn new() -> Joiner {
        let fade = (0..CROSSFADE).map(|n| (0.5 * std::f64::consts::PI * (n as f64 + 0.5) / CROSSFADE as f64).sin() as f32).collect();
        Joiner { tail: None, position: 0, fade }
    }

    fn slice(w: &[f32], from: i64, len: usize) -> Vec<f32> {
        let from = from.max(0) as usize;
        let end = (from + len).min(w.len());
        if from >= end {
            return Vec::new();
        }
        w[from..end].to_vec()
    }

    // `w`: a decode of everything so far. `stop`: where this chunk ends in `w` (a sample index), or None for the last chunk, which goes
    // to the end of `w`. Returns the samples to send.
    pub fn join(&mut self, w: &[f32], stop: Option<usize>) -> Vec<f32> {
        let mut out: Vec<f32> = Vec::new();
        let start;
        match self.tail.take() {
            None => start = 0usize,
            Some(tail) => {
                // align the new decode to the pending tail
                let mut best = 0i64;
                let mut best_score = f64::NEG_INFINITY;
                for shift in -ALIGN..=ALIGN {
                    let from = self.position as i64 + shift;
                    if from < 0 {
                        continue;
                    }
                    let candidate = Self::slice(w, from, CROSSFADE);
                    let score: f64 = tail.iter().zip(&candidate).map(|(a, b)| *a as f64 * *b as f64).sum();
                    if score > best_score {
                        best_score = score;
                        best = shift;
                    }
                }
                let from = (self.position as i64 + best).max(0) as usize;
                let incoming = Self::slice(w, from as i64, CROSSFADE);
                for i in 0..CROSSFADE.min(tail.len()).min(incoming.len()) {
                    out.push(tail[i] * self.fade[CROSSFADE - 1 - i] + incoming[i] * self.fade[i]);
                }
                start = from + CROSSFADE;
            }
        }
        let Some(stop) = stop else {
            if start < w.len() {
                out.extend_from_slice(&w[start..]);
            }
            return out;
        };
        let stop = stop.min(w.len());
        let mut junction = stop;
        let low = (start + CROSSFADE).max(stop.saturating_sub(SEARCH));
        if low < stop && stop >= QUIET_WINDOW {
            let mut best_energy = f64::INFINITY;
            let mut best_at = None;
            let mut j = low;
            while j + QUIET_WINDOW <= stop {
                let energy: f64 = w[j..j + QUIET_WINDOW].iter().map(|&v| v as f64 * v as f64).sum();
                if energy < best_energy {
                    best_energy = energy;
                    best_at = Some(j);
                }
                j += QUIET_STEP;
            }
            if let Some(at) = best_at {
                junction = at + QUIET_STEP;
            }
        }
        let junction = junction.min(w.len()).max(start.min(w.len()));
        if start < junction {
            out.extend_from_slice(&w[start..junction]);
        }
        self.tail = Some(Self::slice(w, junction as i64, CROSSFADE));
        self.position = junction;
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tone(n: usize, offset: usize) -> Vec<f32> {
        (0..n).map(|i| (2.0 * std::f32::consts::PI * 220.0 * (i + offset) as f32 / 24000.0).sin() * 0.5).collect()
    }

    #[test]
    fn two_decodes_of_one_signal_join_without_a_jump() {
        let full = tone(20000, 0);
        let mut joiner = Joiner::new();
        let first = joiner.join(&full[..9000], Some(8000));
        let second = joiner.join(&full, None);
        let mut joined = first.clone();
        joined.extend_from_slice(&second);
        assert!(joined.len() > 19000 && joined.len() <= 20000);
        let worst = joined.windows(2).map(|w| (w[1] - w[0]).abs()).fold(0f32, f32::max);
        assert!(worst < 0.1, "largest step {worst}");
    }

    #[test]
    fn the_junction_goes_to_the_quietest_place() {
        let mut w = tone(12000, 0);
        for v in &mut w[7300..7600] {
            *v = 0.0;
        }
        let mut joiner = Joiner::new();
        let first = joiner.join(&w, Some(8000));
        assert!(first.len() <= 8000 && first.len() >= 7000, "ended at {}", first.len());
        assert!(first.len() >= 7300 && first.len() <= 7700, "the junction sits in the gap: {}", first.len());
    }
}
