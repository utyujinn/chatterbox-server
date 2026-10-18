// A port of resampy's `kaiser_best` resampler (the one librosa uses by default), kept from openvoice-server: converters and encoders that
// were trained on audio resampled with it are sensitive to the filter's transition band, so a plain linear or short-sinc resampler
// is not good enough. Phase tables are cached per ratio.
// Zeroth order modified Bessel function of the first kind, by its power series (enough terms for the beta used here).
fn bessel_i0(x: f64) -> f64 {
    let mut sum = 1.0;
    let mut term = 1.0;
    let half = x / 2.0;
    for k in 1..200 {
        term *= (half / k as f64) * (half / k as f64);
        sum += term;
        if term < sum * 1e-17 {
            break;
        }
    }
    sum
}

fn sinc(x: f64) -> f64 {
    if x == 0.0 {
        1.0
    } else {
        (std::f64::consts::PI * x).sin() / (std::f64::consts::PI * x)
    }
}

// resampy's "kaiser_best" filter, the one librosa.load uses when it changes the sample rate: the right half of a sinc
// (50 zero crossings, 8192 table entries each, cut-off at 0.9173 of Nyquist) under a Kaiser window (beta 12.98). The numbers
// are what resampy's data file holds (checked against it to 4e-16).
// The converter and the reference voice encoder react to the faintest content of the spectrum (they see sqrt(power + 1e-6) of
// every bin), so a resampler with a slightly different transition band gives a quite different voice; this one is a
// step by step port of resampy so that it does not.
const NUM_ZEROS: usize = 50;
const NUM_TABLE: usize = 8192;
const ROLLOFF: f64 = 0.9173473712608761;
const BETA: f64 = 12.984585247040012;

struct Filter {
    table: Vec<f64>,
    delta: Vec<f64>,
}

fn kaiser_best() -> &'static Filter {
    static FILTER: std::sync::OnceLock<Filter> = std::sync::OnceLock::new();
    FILTER.get_or_init(|| {
        let n = NUM_TABLE * NUM_ZEROS;
        let denominator = bessel_i0(BETA);
        let table: Vec<f64> = (0..=n)
            .map(|i| {
                let sinc_win = ROLLOFF * sinc(ROLLOFF * (i as f64 * NUM_ZEROS as f64 / n as f64));
                // the second half of the symmetric Kaiser window of length 2n + 1: positions n..=2n
                let r = i as f64 / n as f64;
                let taper = bessel_i0(BETA * (1.0 - r * r).max(0.0).sqrt()) / denominator;
                taper * sinc_win
            })
            .collect();
        let mut delta = vec![0.0; table.len()];
        for i in 0..table.len() - 1 {
            delta[i] = table[i + 1] - table[i];
        }
        Filter { table, delta }
    })
}

fn gcd(a: u64, b: u64) -> u64 {
    if b == 0 {
        a
    } else {
        gcd(b, a % b)
    }
}

// The filter weights for one phase: where the output sample falls between two input samples, as a fraction p / q (q = the
// rate ratio's denominator). The weights depend only on that, not on which output sample it is, so they are worked out once
// per phase: `left[i]` multiplies the input i samples before the sample just before the output position, `right[k]` the
// k-th one after it. The values are what resampy computes in its loop (including the linear interpolation in the table).
struct Phase {
    left: Vec<f32>,
    right: Vec<f32>,
}

fn phase_weights(frac_num: u64, denominator: u64, scale: f64, gain: f64) -> Phase {
    let filter = kaiser_best();
    let nwin = filter.table.len();
    let index_step = (scale * NUM_TABLE as f64) as usize;
    let frac = scale * (frac_num as f64 / denominator as f64);
    let weights = |frac: f64| -> Vec<f32> {
        let index_frac = frac * NUM_TABLE as f64;
        let offset = index_frac as usize;
        let eta = index_frac - offset as f64;
        (0..(nwin - offset) / index_step)
            .map(|i| {
                let at = offset + i * index_step;
                ((filter.table[at] + eta * filter.delta[at]) * gain) as f32
            })
            .collect()
    };
    Phase { left: weights(frac), right: weights(scale - frac) }
}

// resampy.resample + librosa's fix_length: the same samples as librosa.load(..., sr=to) gives from audio at `from`.
// An empty result for a rate of 0 and for a conversion that would produce an absurd amount of audio (the callers refuse such
// rates before this; this keeps a missed one from dividing by zero or asking for gigabytes).
const MAX_OUTPUT_SAMPLES: u64 = 20_000_000;

pub fn resample(audio: &[f32], from: u32, to: u32) -> Vec<f32> {
    if from == 0 || to == 0 {
        return Vec::new();
    }
    if from == to {
        return audio.to_vec();
    }
    if audio.len() as u64 * to as u64 / from as u64 > MAX_OUTPUT_SAMPLES {
        return Vec::new();
    }
    let sample_ratio = to as f64 / from as f64;
    let scale = sample_ratio.min(1.0);
    if (scale * NUM_TABLE as f64) < 1.0 {
        return Vec::new(); // a ratio below 1/8192: the filter table has no entry per input sample
    }
    // resampy multiplies the whole table by the ratio when going down in rate; the sum is linear in it
    let gain = if sample_ratio < 1.0 { sample_ratio } else { 1.0 };
    let n_orig = audio.len();
    let n_out = (n_orig as u64 * to as u64 / from as u64) as usize; // resampy truncates ...
    // Output sample t sits at input position t * from / to: its whole part and its fraction, in exact integers
    let divisor = gcd(from as u64, to as u64);
    let denominator = to as u64 / divisor;
    let mut phases: std::collections::HashMap<u64, Phase> = std::collections::HashMap::new();
    let mut out = vec![0.0f32; n_out];
    for (t, slot) in out.iter_mut().enumerate() {
        let position = t as u64 * from as u64;
        let n = (position / to as u64) as usize;
        let phase_index = (position % to as u64) / divisor;
        let phase = phases.entry(phase_index).or_insert_with(|| phase_weights(phase_index, denominator, scale, gain));
        let mut sum = 0.0f32;
        let left_count = (n + 1).min(phase.left.len());
        // left wing: the input going backwards from n
        for (weight, sample) in phase.left[..left_count].iter().zip(audio[n + 1 - left_count..=n].iter().rev()) {
            sum += weight * sample;
        }
        // right wing: the input going forwards from n + 1
        let right_count = (n_orig - n - 1).min(phase.right.len());
        for (weight, sample) in phase.right[..right_count].iter().zip(&audio[n + 1..n + 1 + right_count]) {
            sum += weight * sample;
        }
        *slot = sum;
    }
    // ... and librosa then pads (or cuts) to the rounded-up length
    out.resize((n_orig as f64 * to as f64 / from as f64).ceil() as usize, 0.0);
    out
}
