mod audio;
mod dsp;
mod gpu;

use nannou::prelude::*;

const EMA_ALPHA: f32 = 0.3;

struct Model {
    _stream: cpal::Stream,
    dsp_receiver: crossbeam::channel::Receiver<dsp::DspFrame>,
    #[allow(dead_code)]
    sample_rate: u32,
    gpu: gpu::GpuState,
    smoothed_magnitudes: [f32; dsp::NUM_BINS],
    tartini_gpu: [gpu::TartiniGpu; dsp::MAX_TARTINI],
    num_tartini: u32,
    camera_yaw: f32,
    camera_pitch: f32,
    camera_radius: f32,
    mouse_pressed: bool,
    last_mouse: Vec2,
}

fn main() {
    nannou::app(model).update(update).run();
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

    let capture = audio::start_capture();
    let sample_rate = capture.sample_rate;
    let dsp_receiver = dsp::spawn_dsp_thread(capture.consumer);

    let gpu = gpu::GpuState::new(device, msaa, sample_rate as f32);

    Model {
        _stream: capture.stream,
        dsp_receiver,
        sample_rate,
        gpu,
        smoothed_magnitudes: [0.0; dsp::NUM_BINS],
        tartini_gpu: [bytemuck::Zeroable::zeroed(); dsp::MAX_TARTINI],
        num_tartini: 0,
        camera_yaw: 0.0,
        camera_pitch: 0.3,
        camera_radius: 12.0,
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
    if model.mouse_pressed {
        let delta = current_mouse - model.last_mouse;
        model.camera_yaw -= delta.x * 0.005;
        model.camera_pitch = (model.camera_pitch + delta.y * 0.005).clamp(-1.5, 1.5);
    } else {
        model.camera_yaw += 0.003;
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
    while let Ok(frame) = model.dsp_receiver.try_recv() {
        latest = Some(frame);
    }

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

    // --- Upload ---
    let window = app.main_window();
    let queue = window.queue();
    let time = app.elapsed_frames() as f32 / 60.0;
    model.gpu.upload(
        queue,
        &model.smoothed_magnitudes,
        &model.tartini_gpu,
        view_proj.to_cols_array_2d(),
        cam_pos.to_array(),
        time,
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
