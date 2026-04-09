use std::thread;
use std::time::Duration;

use crossbeam::channel::{self, Receiver};
use ringbuf::traits::Consumer;
use ringbuf::HeapCons;
use rustfft::num_complex::Complex;
use rustfft::FftPlanner;

pub const FFT_SIZE: usize = 4096;
pub const NUM_BINS: usize = FFT_SIZE / 2;
pub const MAX_TARTINI: usize = 8;

#[derive(Clone, Copy)]
pub struct DspFrame {
    pub magnitudes: [f32; NUM_BINS],
    pub num_tartini: usize,
    pub tartini_bins: [(usize, usize, f32); MAX_TARTINI],
    /// Multi-band flux: [Low (20-250Hz), Mid (250-4k), High (4k-16k)]
    pub flux: [f32; 3],
}

impl DspFrame {
    fn new() -> Self {
        Self {
            magnitudes: [0.0; NUM_BINS],
            num_tartini: 0,
            tartini_bins: [(0, 0, 0.0); MAX_TARTINI],
            flux: [0.0; 3],
        }
    }
}

pub fn spawn_dsp_thread(mut consumer: HeapCons<f32>, sample_rate: f32) -> Receiver<DspFrame> {
    let (sender, receiver) = channel::bounded::<DspFrame>(2);

    thread::Builder::new()
        .name("dsp".into())
        .spawn(move || {
            // --- Pre-allocate everything before the loop ---
            let mut slide_buf = vec![0.0f32; FFT_SIZE];
            let mut read_buf = vec![0.0f32; FFT_SIZE];
            let mut fft_buf: Vec<Complex<f32>> = vec![Complex::new(0.0, 0.0); FFT_SIZE];

            let window: Vec<f32> = (0..FFT_SIZE)
                .map(|i| {
                    let phase = 2.0 * std::f32::consts::PI * i as f32 / FFT_SIZE as f32;
                    0.5 * (1.0 - phase.cos())
                })
                .collect();

            let mut planner = FftPlanner::<f32>::new();
            let fft = planner.plan_fft_forward(FFT_SIZE);
            let mut scratch = vec![Complex::new(0.0f32, 0.0); fft.get_inplace_scratch_len()];

            let mag_scale = 2.0 / FFT_SIZE as f32;
            const PEAK_THRESHOLD: f32 = 0.001;
            const TOP_N: usize = 3;

            // Frequency band indices
            let bin_freq = sample_rate / FFT_SIZE as f32;
            let k_low_end = (250.0 / bin_freq).round() as usize;
            let k_mid_end = (4000.0 / bin_freq).round() as usize;
            let k_high_end = (16000.0 / bin_freq).round() as usize;

            let k_low_end = k_low_end.min(NUM_BINS - 1);
            let k_mid_end = k_mid_end.min(NUM_BINS - 1).max(k_low_end);
            let k_high_end = k_high_end.min(NUM_BINS - 1).max(k_mid_end);

            let mut prev_magnitudes = vec![0.0f32; NUM_BINS];

            loop {
                let n = consumer.pop_slice(&mut read_buf);
                if n == 0 {
                    thread::sleep(Duration::from_millis(1));
                    continue;
                }

                if n >= FFT_SIZE {
                    slide_buf.copy_from_slice(&read_buf[n - FFT_SIZE..n]);
                } else {
                    slide_buf.copy_within(n.., 0);
                    slide_buf[FFT_SIZE - n..].copy_from_slice(&read_buf[..n]);
                }

                for i in 0..FFT_SIZE {
                    fft_buf[i] = Complex::new(slide_buf[i] * window[i], 0.0);
                }

                fft.process_with_scratch(&mut fft_buf, &mut scratch);

                // --- Magnitudes with Psychoacoustic Weighting ---
                let mut frame = DspFrame::new();
                for k in 0..NUM_BINS {
                    let c = fft_buf[k];
                    let mut mag = (c.re * c.re + c.im * c.im).sqrt() * mag_scale;
                    let freq = k as f32 * bin_freq;

                    // A-weighting approximation / Perceptual curve
                    // Boosts presence (2-5kHz), slight roll-off for sub/extreme-high
                    let f_sq = freq * freq;
                    let weight = if freq < 10.0 { 0.1 } else {
                        let w = (12200.0 * 12200.0 * f_sq * f_sq) /
                            ((f_sq + 20.6 * 20.6) * 
                             ((f_sq + 107.7 * 107.7) * (f_sq + 737.9 * 737.9)).sqrt() * 
                             (f_sq + 12200.0 * 12200.0));
                        w.max(0.01)
                    };
                    
                    mag *= weight * 4.0; // Scale back up after weighting
                    mag = mag.clamp(0.0, 5.0);
                    frame.magnitudes[k] = mag;
                }

                // --- Multi-Band Spectral Flux (Onset Detection) ---
                // Low (20 - 250Hz)
                for k in 2..k_low_end {
                    frame.flux[0] += (frame.magnitudes[k] - prev_magnitudes[k]).max(0.0);
                }
                // Mid (250Hz - 4kHz)
                for k in k_low_end..k_mid_end {
                    frame.flux[1] += (frame.magnitudes[k] - prev_magnitudes[k]).max(0.0);
                }
                // High (4kHz - 16kHz)
                for k in k_mid_end..k_high_end {
                    frame.flux[2] += (frame.magnitudes[k] - prev_magnitudes[k]).max(0.0);
                }

                prev_magnitudes.copy_from_slice(&frame.magnitudes);

                // --- Peak detection: top 3 local maxima ---
                let mut peaks = [(0usize, 0.0f32); TOP_N];
                let mut peak_count = 0usize;

                for i in 1..NUM_BINS - 1 {
                    let m = frame.magnitudes[i];
                    if m > frame.magnitudes[i - 1]
                        && m > frame.magnitudes[i + 1]
                        && m > PEAK_THRESHOLD
                    {
                        if peak_count < TOP_N {
                            peaks[peak_count] = (i, m);
                            peak_count += 1;
                        } else {
                            let mut min_j = 0;
                            for j in 1..TOP_N {
                                if peaks[j].1 < peaks[min_j].1 {
                                    min_j = j;
                                }
                            }
                            if m > peaks[min_j].1 {
                                peaks[min_j] = (i, m);
                            }
                        }
                    }
                }

                // --- Tartini tones: one entry per peak pair ---
                // The GPU shader derives both |bin_a - bin_b| (difference tone)
                // and bin_a + bin_b (summation tone) from each stored pair.
                let mut tartini_count = 0usize;
                for i in 0..peak_count {
                    for j in (i + 1)..peak_count {
                        if tartini_count >= MAX_TARTINI {
                            break;
                        }
                        let (bin_a, mag_a) = peaks[i];
                        let (bin_b, mag_b) = peaks[j];
                        frame.tartini_bins[tartini_count] =
                            (bin_a, bin_b, (mag_a * mag_b).sqrt());
                        tartini_count += 1;
                    }
                }
                frame.num_tartini = tartini_count;

                // By-value send (~9KB copy). Silently drop if channel is full.
                let _ = sender.try_send(frame);
            }
        })
        .expect("failed to spawn DSP thread");

    receiver
}
