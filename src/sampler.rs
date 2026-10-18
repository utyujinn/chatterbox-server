// Sampling of the next speech token, in the order of the HuggingFace logits processors of the original's `t3.inference_turbo`:
// temperature, top-k, top-p, repetition penalty, then a draw from the softmax. All in f64, as the golden data was made.

pub struct SamplerSettings {
    pub temperature: f64,
    pub top_k: usize,
    pub top_p: f64,
    pub repetition_penalty: f64,
}

impl Default for SamplerSettings {
    fn default() -> Self {
        SamplerSettings { temperature: 0.8, top_k: 1000, top_p: 0.95, repetition_penalty: 1.2 }
    }
}

// The probability of every token after the processors. `history`: the repetition penalty looks at its unique ids.
pub fn probabilities(logits: &[f32], history: &[i64], settings: &SamplerSettings) -> Vec<f64> {
    let mut x: Vec<f64> = logits.iter().map(|&v| v as f64 / settings.temperature).collect();
    // top-k: everything below the k-th largest value goes (ties at the k-th are kept)
    if settings.top_k < x.len() {
        let mut sorted = x.clone();
        sorted.sort_by(|a, b| b.total_cmp(a));
        let kth = sorted[settings.top_k - 1];
        for v in x.iter_mut() {
            if *v < kth {
                *v = f64::NEG_INFINITY;
            }
        }
    }
    // top-p: sort ascending, drop the entries whose cumulative probability stays within 1 - top_p, never the largest
    let mut order: Vec<usize> = (0..x.len()).collect();
    order.sort_by(|&a, &b| x[a].total_cmp(&x[b]));
    let top = x[*order.last().unwrap()];
    let mut p: Vec<f64> = order.iter().map(|&i| (x[i] - top).exp()).collect();
    let total: f64 = p.iter().sum();
    for v in p.iter_mut() {
        *v /= total;
    }
    let mut cumulative = 0.0;
    let last = order.len() - 1;
    for (rank, &i) in order.iter().enumerate() {
        cumulative += p[rank];
        if rank != last && cumulative <= 1.0 - settings.top_p {
            x[i] = f64::NEG_INFINITY;
        }
    }
    // repetition penalty on the unique ids of the history, after the filtering (a masked token stays masked)
    let mut seen: Vec<usize> = history.iter().filter(|&&t| t >= 0 && (t as usize) < x.len()).map(|&t| t as usize).collect();
    seen.sort_unstable();
    seen.dedup();
    for i in seen {
        let s = x[i];
        x[i] = if s < 0.0 { s * settings.repetition_penalty } else { s / settings.repetition_penalty };
    }
    let max = x.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let mut p: Vec<f64> = x.iter().map(|&v| (v - max).exp()).collect();
    let total: f64 = p.iter().sum();
    for v in p.iter_mut() {
        *v /= total;
    }
    p
}

// The first index whose cumulative probability exceeds `u` (in [0, 1)).
pub fn draw(p: &[f64], u: f64) -> usize {
    let mut cumulative = 0.0;
    for (i, &v) in p.iter().enumerate() {
        cumulative += v;
        if cumulative > u {
            return i;
        }
    }
    p.len() - 1
}

// A small random generator (xorshift64*): the draws only have to be uniform and, for a given seed, repeatable.
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Rng {
        Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1)
    }
    pub fn unit(&mut self) -> f64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        let bits = self.0.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 11;
        bits as f64 / (1u64 << 53) as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn probabilities_sum_to_one_and_the_filters_apply() {
        let mut logits = vec![0.0f32; 6563];
        for (i, v) in logits.iter_mut().enumerate() {
            *v = (i as f32 * 0.37).sin() * 6.0 + (i as f32 * 0.011).cos();
        }
        let p = probabilities(&logits, &[3, 3, 5], &SamplerSettings::default());
        assert!((p.iter().sum::<f64>() - 1.0).abs() < 1e-9);
        assert!(p.iter().filter(|&&v| v > 0.0).count() < 1000, "top-k and top-p cut the tail");
    }

    #[test]
    fn the_draw_follows_the_cumulative_distribution() {
        let p = [0.2, 0.3, 0.5];
        assert_eq!(draw(&p, 0.0), 0);
        assert_eq!(draw(&p, 0.19), 0);
        assert_eq!(draw(&p, 0.21), 1);
        assert_eq!(draw(&p, 0.51), 2);
        assert_eq!(draw(&p, 0.999999), 2);
    }

    #[test]
    fn the_generator_is_uniform_enough_and_repeatable() {
        let mut a = Rng::new(7);
        let mut b = Rng::new(7);
        let draws: Vec<f64> = (0..10000).map(|_| a.unit()).collect();
        assert!(draws.iter().all(|&u| (0.0..1.0).contains(&u)));
        assert_eq!(draws[0], b.unit());
        let mean = draws.iter().sum::<f64>() / draws.len() as f64;
        assert!((mean - 0.5).abs() < 0.02, "mean {mean}");
    }
}
