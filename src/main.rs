mod audio;
mod dsp;
mod gpu;

use nannou::prelude::*;

const EMA_ALPHA: f32 = 0.3;
const GAIN_ATTACK: f32 = 0.05;
const GAIN_RELEASE: f32 = 0.0005;
const GAIN_FLOOR: f32 = 0.001;
const BASE_ROTATION_SPEED: f32 = 0.04;
const MAX_TEMPO_SPEED: f32 = 0.25;
const MAX_PUNCH_SPEED: f32 = 0.85;
const TEMPO_DECAY: f32 = 0.997; // Very slow for stable tempo estimation
const PUNCH_DECAY: f32 = 0.92;  // Increased for a more "weighted" heavy feel
const BEAT_MULTIPLIER: f32 = 1.6;
const HIGH_MULTIPLIER: f32 = 2.2; // Sharper threshold for high-end transients
const LOW_DEBOUNCE_WINDOW: f32 = 0.080; 

struct Model {
    _stream: cpal::Stream,
    dsp_receiver: crossbeam::channel::Receiver<dsp::DspFrame>,
    #[allow(dead_code)]
    sample_rate: u32,
    gpu: gpu::GpuState,
    smoothed_magnitudes: [f32; dsp::NUM_BINS],
    peak_envelope: f32,
    tartini_gpu: [gpu::TartiniGpu; dsp::MAX_TARTINI],
    num_tartini: u32,
    camera_yaw: f32,
    camera_pitch: f32,
    camera_radius: f32,
    tempo_drive: f32,
    beat_punch: f32,
    flux_ema: [f32; 3],
    last_low_trigger: f32,
    mouse_pressed: bool,
    last_mouse: Vec2,
}

fn main() {
    nannou::app(model).update(update).run();
}

fn use_mic() -> bool {
    std::env::args().any(|a| a == "--mic")
}

fn model(app: &App) -> Model {
    let w_id = app
        .new_window()
        .title("Synesthetic Visualizer")
        .size(1280, 720)
        .event(event)
        .view(view)
        .build()
        .unwrap();

    let window = app.window(w_id).unwrap();
    let device = window.device();
    let msaa = window.msaa_samples();

    let capture = audio::start_capture(use_mic());
    let sample_rate = capture.sample_rate;
    let dsp_receiver = dsp::spawn_dsp_thread(capture.consumer, sample_rate as f32);

    let gpu = gpu::GpuState::new(device, msaa, sample_rate as f32);

    Model {
        _stream: capture.stream,
        dsp_receiver,
        sample_rate,
        gpu,
        smoothed_magnitudes: [0.0; dsp::NUM_BINS],
        peak_envelope: GAIN_FLOOR,
        tartini_gpu: [bytemuck::Zeroable::zeroed(); dsp::MAX_TARTINI],
        num_tartini: 0,
        camera_yaw: 0.0,
        camera_pitch: 0.3,
        camera_radius: 12.0,
        tempo_drive: 0.0,
        beat_punch: 0.0,
        flux_ema: [0.0; 3],
        last_low_trigger: 0.0,
        mouse_pressed: false,
        last_mouse: Vec2::ZERO,
    }
}

fn event(_app: &App, model: &mut Model, event: WindowEvent) {
    match event {
        MousePressed(button) => {
            if button == MouseButton::Left {
                model.mouse_pressed = true;
            }
        }
        MouseReleased(button) => {
            if button == MouseButton::Left {
                model.mouse_pressed = false;
            }
        }
        MouseWheel(delta, _) => {
            let scroll = match delta {
                MouseScrollDelta::LineDelta(_, y) => y,
                MouseScrollDelta::PixelDelta(p) => p.y as f32 * 0.01,
            };
            model.camera_radius = (model.camera_radius - scroll * 0.5).clamp(3.0, 30.0);
        }
        _ => {}
    }
}

fn update(app: &App, model: &mut Model, _update: Update) {
    // --- Camera orbit ---
    let current_mouse = vec2(app.mouse.x, app.mouse.y);
    if app.mouse.buttons.left().is_down() {
        let delta = current_mouse - model.last_mouse;
        model.camera_yaw -= delta.x * 0.005;
        model.camera_pitch = (model.camera_pitch + delta.y * 0.005).clamp(-1.5, 1.5);
    } else {
        let delta_time = app.duration.since_prev_update.as_secs_f32();
        // Speed = Base + Tempo (stable) + Punch (instant)
        let current_speed = BASE_ROTATION_SPEED
            + (model.tempo_drive * MAX_TEMPO_SPEED)
            + (model.beat_punch * MAX_PUNCH_SPEED);
        model.camera_yaw += current_speed * delta_time;
    }
    model.last_mouse = current_mouse;

    let cy = model.camera_pitch.cos();
    let sy = model.camera_pitch.sin();
    let cam_pos = Vec3::new(
        model.camera_radius * cy * model.camera_yaw.sin(),
        model.camera_radius * sy,
        model.camera_radius * cy * model.camera_yaw.cos(),
    );

    let view = Mat4::look_at_rh(cam_pos, Vec3::ZERO, Vec3::Y);
    let proj = Mat4::perspective_rh(
        std::f32::consts::FRAC_PI_3,
        1280.0 / 720.0,
        0.1,
        100.0,
    );
    let view_proj = proj * view;

    // --- DSP data ---
    let mut latest: Option<dsp::DspFrame> = None;
    let current_time = app.time;

    while let Ok(frame) = model.dsp_receiver.try_recv() {
        // Multi-band adaptive thresholding
        // We track EMA for Low, Mid, and High independently
        for i in 0..3 {
            model.flux_ema[i] = (model.flux_ema[i] * 0.9) + (frame.flux[i] * 0.1);
        }

        // --- Scientific Responsiveness (Additive Stacking) ---
        // Every frame, all bands that hit their threshold contribute to the punch.
        let mut frame_punch = 0.0;
        let mut frame_tempo_boost = 0.0;
        
        // Low: Foundation (80ms debounce to avoid multiple clicks on one thump)
        if frame.flux[0] > (model.flux_ema[0] * BEAT_MULTIPLIER).max(0.1) {
            if (current_time - model.last_low_trigger) > LOW_DEBOUNCE_WINDOW {
                model.last_low_trigger = current_time;
                frame_punch += 0.71;
                frame_tempo_boost += 0.14;
            }
        }
        
        // Mid: Body
        if frame.flux[1] > (model.flux_ema[1] * BEAT_MULTIPLIER).max(0.1) {
            frame_punch += 0.3;
            frame_tempo_boost += 0.04;
        }

        // High: Clarity (Sharper threshold to avoid jitter on noise)
        if frame.flux[2] > (model.flux_ema[2] * HIGH_MULTIPLIER).max(0.1) {
            frame_punch += 0.4;
            frame_tempo_boost += 0.08;
        }

        // Apply cumulative effects
        if frame_punch > 0.0 {
            model.beat_punch = (model.beat_punch + frame_punch).min(1.0);
            model.tempo_drive = (model.tempo_drive + frame_tempo_boost).min(1.0);
        }

        latest = Some(frame);
    }
    
    model.tempo_drive *= TEMPO_DECAY;
    model.beat_punch *= PUNCH_DECAY;

    if let Some(frame) = latest {
        for i in 0..dsp::NUM_BINS {
            model.smoothed_magnitudes[i] = EMA_ALPHA * frame.magnitudes[i]
                + (1.0 - EMA_ALPHA) * model.smoothed_magnitudes[i];
        }

        let num_t = frame.num_tartini.min(dsp::MAX_TARTINI);
        model.num_tartini = num_t as u32;
        for i in 0..num_t {
            let (p1, p2, mag) = frame.tartini_bins[i];
            model.tartini_gpu[i] = gpu::TartiniGpu {
                diff_bin: p1.abs_diff(p2) as u32,
                sum_bin: (p1 + p2) as u32,
                magnitude: mag,
                _pad: 0,
            };
        }
        for i in num_t..dsp::MAX_TARTINI {
            model.tartini_gpu[i] = bytemuck::Zeroable::zeroed();
        }
    }

    // --- Auto-gain: track peak with fast attack, slow release ---
    let current_peak = model
        .smoothed_magnitudes
        .iter()
        .copied()
        .fold(0.0f32, f32::max);

    if current_peak > model.peak_envelope {
        model.peak_envelope += (current_peak - model.peak_envelope) * GAIN_ATTACK;
    } else {
        model.peak_envelope += (current_peak - model.peak_envelope) * GAIN_RELEASE;
    }
    model.peak_envelope = model.peak_envelope.max(GAIN_FLOOR);

    let gain = 1.0 / model.peak_envelope;

    // --- Upload gained magnitudes ---
    let mut upload_mags = [0.0f32; dsp::NUM_BINS];
    for i in 0..dsp::NUM_BINS {
        upload_mags[i] = model.smoothed_magnitudes[i] * gain;
    }

    let window = app.main_window();
    let queue = window.queue();
    let time = app.elapsed_frames() as f32 / 60.0;
    model.gpu.upload(
        queue,
        &upload_mags,
        &model.tartini_gpu,
        view_proj.to_cols_array_2d(),
        cam_pos.to_array(),
        time,
        model.beat_punch,
        model.num_tartini,
    );
}

fn view(_app: &App, model: &Model, frame: Frame) {
    {
        let mut encoder = frame.command_encoder();
        let mut pass = wgpu::RenderPassBuilder::new()
            .color_attachment(frame.texture_view(), |c| c)
            .begin(&mut encoder);

        pass.set_bind_group(0, &model.gpu.bind_group, &[]);

        pass.set_pipeline(&model.gpu.pipeline);
        pass.draw(0..6, 0..dsp::NUM_BINS as u32);

        pass.set_pipeline(&model.gpu.tartini_pipeline);
        pass.draw(0..6, 0..(dsp::MAX_TARTINI * 2) as u32);
    }
}
