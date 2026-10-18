// The signal processing of a reference voice that is not a neural network: loudness, the silence trim and the mel spectrogram the
// VoiceEncoder takes. Constants and steps are those of the PyTorch original (spec §3), which the golden tests compare with.

use anyhow::{ensure, Result};
use ebur128::{EbuR128, Mode};
use realfft::RealFftPlanner;

// The loudness every reference is brought to before anything else (the PyTorch pipeline's own).
pub const TARGET_LUFS: f64 = -27.0;

// (integrated loudness, gain) of `audio` (mono): the gain that brings it to TARGET_LUFS, by integrated loudness with gating
// (ITU-R BS.1770-4); 1.0 for a recording with nothing loud enough to measure.
pub fn loudness_and_gain(audio: &[f32], rate: u32) -> Result<(f64, f32)> {
    let mut meter = EbuR128::new(1, rate, Mode::I)?;
    meter.add_frames_f32(audio)?;
    let loudness = meter.loudness_global()?;
    if !loudness.is_finite() {
        return Ok((loudness, 1.0));
    }
    let gain = 10f64.powf((TARGET_LUFS - loudness) / 20.0);
    Ok((loudness, if gain.is_finite() && gain > 0.0 { gain as f32 } else { 1.0 }))
}

// librosa.effects.trim(top_db=20): the sample range between the first and the last frame (2048 long, hop 512, the signal zero padded by
// 1024 on both sides) whose power is within 20 dB of the loudest frame.
pub fn trim_bounds(audio: &[f32]) -> (usize, usize) {
    const FRAME: usize = 2048;
    const HOP: usize = 512;
    const PAD: usize = FRAME / 2;
    let n = audio.len();
    let mut padded = vec![0f32; n + 2 * PAD];
    padded[PAD..PAD + n].copy_from_slice(audio);
    let frames = 1 + n / HOP;
    let power: Vec<f64> = (0..frames)
        .map(|f| {
            let start = f * HOP;
            let sum: f64 = padded[start..start + FRAME].iter().map(|&x| (x as f64) * (x as f64)).sum();
            sum / FRAME as f64
        })
        .collect();
    let reference = power.iter().cloned().fold(0.0, f64::max).max(1e-10);
    let db = |p: f64| 10.0 * p.max(1e-10).log10() - 10.0 * reference.log10();
    let loud: Vec<usize> = (0..frames).filter(|&f| db(power[f]) > -20.0).collect();
    match (loud.first(), loud.last()) {
        (Some(&first), Some(&last)) => (first * HOP, n.min((last + 1) * HOP)),
        _ => (0, 0),
    }
}

const VE_FFT: usize = 400;
const VE_HOP: usize = 160;
const VE_BINS: usize = VE_FFT / 2 + 1;
const VE_MELS: usize = 40;
pub const VE_PARTIAL_FRAMES: usize = 160;
const VE_PARTIAL_STEP: usize = 77;

// The VoiceEncoder's input: the trimmed 16 kHz recording as mel spectrogram power (no log) cut into overlapping 160-frame partials.
// `mel_basis` is the [40, 201] filterbank. Returns the partials flattened as [n_partials, 160, 40] and n_partials.
pub fn ve_mel_partials(wav16: &[f32], mel_basis: &[f32]) -> Result<(Vec<f32>, usize)> {
    ensure!(mel_basis.len() == VE_MELS * VE_BINS, "the mel filterbank has {} values, not {}", mel_basis.len(), VE_MELS * VE_BINS);
    let (start, end) = trim_bounds(wav16);
    let trimmed = if end > start { &wav16[start..end] } else { wav16 };
    let n = trimmed.len();
    let pad = VE_FFT / 2;
    ensure!(n > pad, "the recording is too short");
    // reflect padding, 200 on both sides (librosa center=True)
    let mut padded = Vec::with_capacity(n + 2 * pad);
    for i in (1..=pad).rev() {
        padded.push(trimmed[i]);
    }
    padded.extend_from_slice(trimmed);
    for i in 1..=pad {
        padded.push(trimmed[n - 1 - i]);
    }
    let frames = 1 + n / VE_HOP;
    let window: Vec<f32> = (0..VE_FFT).map(|i| (0.5 - 0.5 * (2.0 * std::f64::consts::PI * i as f64 / VE_FFT as f64).cos()) as f32).collect();
    let fft = RealFftPlanner::<f32>::new().plan_fft_forward(VE_FFT);
    let mut input = fft.make_input_vec();
    let mut output = fft.make_output_vec();
    let mut mel = vec![0f32; frames * VE_MELS];
    for t in 0..frames {
        let from = t * VE_HOP;
        for i in 0..VE_FFT {
            input[i] = padded[from + i] * window[i];
        }
        fft.process(&mut input, &mut output)?;
        let power: Vec<f32> = output.iter().map(|c| c.re * c.re + c.im * c.im).collect();
        for m in 0..VE_MELS {
            let row = &mel_basis[m * VE_BINS..(m + 1) * VE_BINS];
            mel[t * VE_MELS + m] = row.iter().zip(&power).map(|(a, b)| a * b).sum();
        }
    }
    // how many 160-frame partials, stepping by 77 frames (resemblyzer's compute_partial_slices)
    let usable = (frames + VE_PARTIAL_STEP).saturating_sub(VE_PARTIAL_FRAMES);
    let (mut n_partials, rem) = (usable / VE_PARTIAL_STEP, usable % VE_PARTIAL_STEP);
    if n_partials == 0 || (rem + 83) as f32 / VE_PARTIAL_FRAMES as f32 >= 0.8 {
        n_partials += 1;
    }
    let wanted = VE_PARTIAL_FRAMES + VE_PARTIAL_STEP * (n_partials - 1);
    mel.resize(wanted * VE_MELS, 0.0);
    let mut out = Vec::with_capacity(n_partials * VE_PARTIAL_FRAMES * VE_MELS);
    for i in 0..n_partials {
        let from = i * VE_PARTIAL_STEP * VE_MELS;
        out.extend_from_slice(&mel[from..from + VE_PARTIAL_FRAMES * VE_MELS]);
    }
    Ok((out, n_partials))
}
