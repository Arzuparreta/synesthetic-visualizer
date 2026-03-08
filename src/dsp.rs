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
    pub bass_flux: f32,
}

impl DspFrame {
    fn new() -> Self {
        Self {
            magnitudes: [0.0; NUM_BINS],
            num_tartini: 0,
            tartini_bins: [(0, 0, 0.0); MAX_TARTINI],
            bass_flux: 0.0,
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

            let k_low = (20.0 * FFT_SIZE as f32 / sample_rate).ceil() as usize;
            let k_high = (150.0 * FFT_SIZE as f32 / sample_rate).floor() as usize;
            let k_low = k_low.min(NUM_BINS - 1);
            let k_high = k_high.min(NUM_BINS - 1).max(k_low);

            let mut prev_bass_energy = 0.0f32;

            loop {
                let n = consumer.pop_slice(&mut read_buf);
                if n == 0 {
                    thread::sleep(Duration::from_millis(1));
                    continue;
                }

                // Sliding window: shift old samples left, append new ones right.
                // This gives overlap-add style processing without waiting for
                // a full FFT_SIZE worth of new samples.
                if n >= FFT_SIZE {
                    slide_buf.copy_from_slice(&read_buf[n - FFT_SIZE..n]);
                } else {
                    slide_buf.copy_within(n.., 0);
                    slide_buf[FFT_SIZE - n..].copy_from_slice(&read_buf[..n]);
                }

                // Apply Hanning window into the complex FFT buffer
                for i in 0..FFT_SIZE {
                    fft_buf[i] = Complex::new(slide_buf[i] * window[i], 0.0);
                }

                fft.process_with_scratch(&mut fft_buf, &mut scratch);

                // --- Magnitudes (perceptual weighting: boost highs, compress sub-bass) ---
                let mut frame = DspFrame::new();
                for k in 0..NUM_BINS {
                    let c = fft_buf[k];
                    let mut mag = (c.re * c.re + c.im * c.im).sqrt() * mag_scale;
                    let freq = k as f32 * (sample_rate / FFT_SIZE as f32);
                    let weight = (freq / 1000.0).max(0.1).sqrt();
                    mag *= weight;
                    mag = mag.clamp(0.0, 5.0);
                    frame.magnitudes[k] = mag;
                }

                let current_bass_energy: f32 =
                    frame.magnitudes[k_low..=k_high].iter().sum();
                frame.bass_flux = (current_bass_energy - prev_bass_energy).max(0.0);
                prev_bass_energy = current_bass_energy;

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
