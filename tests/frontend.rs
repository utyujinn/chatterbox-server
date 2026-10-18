// The reference voice's signal processing against the golden data made by the Python original (tests/golden/README.md).
use chatterbox_server::{audio, npy, resample};
use std::path::PathBuf;

fn golden(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/golden/us_f_50s").join(name)
}
fn data(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("data").join(name)
}
fn correlation(a: &[f32], b: &[f32]) -> f64 {
    let n = a.len().min(b.len());
    let (ma, mb) = (a[..n].iter().map(|&x| x as f64).sum::<f64>() / n as f64, b[..n].iter().map(|&x| x as f64).sum::<f64>() / n as f64);
    let (mut ab, mut aa, mut bb) = (0.0, 0.0, 0.0);
    for i in 0..n {
        let (x, y) = (a[i] as f64 - ma, b[i] as f64 - mb);
        ab += x * y;
        aa += x * x;
        bb += y * y;
    }
    ab / (aa.sqrt() * bb.sqrt())
}

#[test]
fn loudness_gain_matches_the_manifest() {
    let raw = npy::read(&golden("wav24_raw.npy")).unwrap().f32().unwrap();
    let (loudness, gain) = audio::loudness_and_gain(&raw, 24000).unwrap();
    // pyloudnorm and ebur128 derive the K-weighting filter at 24 kHz differently; 0.12 LU (1.4 % in amplitude) is far below what the
    // model notices, so the bound is loose
    assert!((loudness - (-24.065034592107587)).abs() < 0.25, "loudness {loudness}");
    assert!((gain as f64 - 0.7132663400707989).abs() < 0.02, "gain {gain}");
}

#[test]
fn resampling_24k_to_16k_follows_the_python() {
    let wav24 = npy::read(&golden("wav24.npy")).unwrap().f32().unwrap();
    let want = npy::read(&golden("wav16.npy")).unwrap().f32().unwrap();
    let got = resample::resample(&wav24, 24000, 16000);
    assert_eq!(got.len(), want.len());
    let c = correlation(&got, &want);
    assert!(c > 0.999, "correlation {c}");
}

#[test]
fn the_voice_encoder_input_matches_the_python() {
    let wav16 = npy::read(&golden("wav16.npy")).unwrap().f32().unwrap();
    assert_eq!(audio::trim_bounds(&wav16), (0, wav16.len()), "the trim of this recording keeps everything");
    let basis = npy::read(&data("ve_mel_basis_40x201.npy")).unwrap().f32().unwrap();
    let (got, n) = audio::ve_mel_partials(&wav16, &basis).unwrap();
    let want = npy::read(&golden("ve_mel_partials.npy")).unwrap();
    assert_eq!(want.shape, vec![15, 160, 40]);
    assert_eq!(n, 15);
    let want = want.f32().unwrap();
    assert_eq!(got.len(), want.len());
    let scale = want.iter().cloned().fold(0f32, f32::max);
    let worst = got.iter().zip(&want).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
    assert!(worst / scale < 1e-3, "relative difference {}", worst / scale);
}
