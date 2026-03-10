use crate::dsp;
use nannou::prelude::*;

/// Frequency window (Hz) for the tonnetz display. Must match shader constants in tonnetz.wgsl.
#[allow(dead_code)]
pub const MIN_FREQ: f32 = 40.0;
#[allow(dead_code)]
pub const MAX_FREQ: f32 = 18000.0;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniforms {
    view_proj: [[f32; 4]; 4],
    camera_pos: [f32; 3],
    time: f32,
    sample_rate: f32,
    fft_size: f32,
    beat_pulse: f32,
    num_tartini: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct TartiniGpu {
    pub diff_bin: u32,
    pub sum_bin: u32,
    pub magnitude: f32,
    pub _pad: u32,
}

pub struct GpuState {
    pub(crate) pipeline: wgpu::RenderPipeline,
    pub(crate) tartini_pipeline: wgpu::RenderPipeline,
    pub(crate) bind_group: wgpu::BindGroup,
    magnitude_buffer: wgpu::Buffer,
    tartini_buffer: wgpu::Buffer,
    uniform_buffer: wgpu::Buffer,
    sample_rate: f32,
}

impl GpuState {
    pub fn new(device: &wgpu::Device, msaa_samples: u32, sample_rate: f32) -> Self {
        let magnitude_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Magnitudes SSBO"),
            size: (dsp::NUM_BINS * std::mem::size_of::<f32>()) as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let tartini_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Tartini SSBO"),
            size: (dsp::MAX_TARTINI * std::mem::size_of::<TartiniGpu>()) as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Uniforms UBO"),
            size: std::mem::size_of::<Uniforms>() as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Tonnetz BGL"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Tonnetz BG"),
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: magnitude_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: tartini_buffer.as_entire_binding(),
                },
            ],
        });

        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Tonnetz PL"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Tonnetz Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/tonnetz.wgsl").into()),
        });

        let make_pipeline = |label, entry_point| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(&pl),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point,
                    buffers: &[],
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: "fs_main",
                    targets: &[Some(wgpu::ColorTargetState {
                        format: Frame::TEXTURE_FORMAT,
                        blend: Some(wgpu::BlendState {
                            color: wgpu::BlendComponent {
                                src_factor: wgpu::BlendFactor::SrcAlpha,
                                dst_factor: wgpu::BlendFactor::One,
                                operation: wgpu::BlendOperation::Add,
                            },
                            alpha: wgpu::BlendComponent::REPLACE,
                        }),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    strip_index_format: None,
                    front_face: wgpu::FrontFace::Ccw,
                    cull_mode: None,
                    polygon_mode: wgpu::PolygonMode::Fill,
                    unclipped_depth: false,
                    conservative: false,
                },
                depth_stencil: None,
                multisample: wgpu::MultisampleState {
                    count: msaa_samples,
                    mask: !0,
                    alpha_to_coverage_enabled: false,
                },
                multiview: None,
            })
        };

        let pipeline = make_pipeline("Spectrum Pipeline", "vs_main");
        let tartini_pipeline = make_pipeline("Tartini Pipeline", "vs_tartini");

        Self {
            pipeline,
            tartini_pipeline,
            bind_group,
            magnitude_buffer,
            tartini_buffer,
            uniform_buffer,
            sample_rate,
        }
    }

    pub fn upload(
        &self,
        queue: &wgpu::Queue,
        magnitudes: &[f32],
        tartini: &[TartiniGpu; dsp::MAX_TARTINI],
        view_proj: [[f32; 4]; 4],
        camera_pos: [f32; 3],
        time: f32,
        beat_pulse: f32,
        num_tartini: u32,
    ) {
        queue.write_buffer(&self.magnitude_buffer, 0, bytemuck::cast_slice(magnitudes));
        queue.write_buffer(&self.tartini_buffer, 0, bytemuck::cast_slice(tartini));
        let u = Uniforms {
            view_proj,
            camera_pos,
            time,
            sample_rate: self.sample_rate,
            fft_size: dsp::FFT_SIZE as f32,
            beat_pulse,
            num_tartini,
        };
        queue.write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&u));
    }
}
