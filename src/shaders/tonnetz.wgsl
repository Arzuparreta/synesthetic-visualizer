struct Uniforms {
    view_proj: mat4x4<f32>,
    camera_pos: vec3<f32>,
    time: f32,
    sample_rate: f32,
    fft_size: f32,
    beat_pulse: f32,
    num_tartini: u32,
}

struct TartiniTone {
    diff_bin: u32,
    sum_bin: u32,
    magnitude: f32,
    _pad: u32,
}

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec3<f32>,
    @location(1) alpha: f32,
    @location(2) uv: vec2<f32>,
}

@group(0) @binding(0) var<storage, read> magnitudes: array<f32>;
@group(0) @binding(1) var<uniform> uniforms: Uniforms;
@group(0) @binding(2) var<storage, read> tartini_data: array<TartiniTone>;

const PI: f32 = 3.14159265358979;
const MIN_FREQ: f32 = 40.0;
const MAX_FREQ: f32 = 18000.0;
const LOG_MIN: f32 = 5.32192809;   // log2(40.0)
const LOG_MAX: f32 = 14.1357093;   // log2(18000.0)

// ---- helpers ----

fn hsv_to_rgb(h: f32, s: f32, v: f32) -> vec3<f32> {
    let c = v * s;
    let hp = h / 60.0;
    let x = c * (1.0 - abs(hp % 2.0 - 1.0));
    let m = v - c;
    var rgb: vec3<f32>;
    if hp < 1.0 {
        rgb = vec3<f32>(c, x, 0.0);
    } else if hp < 2.0 {
        rgb = vec3<f32>(x, c, 0.0);
    } else if hp < 3.0 {
        rgb = vec3<f32>(0.0, c, x);
    } else if hp < 4.0 {
        rgb = vec3<f32>(0.0, x, c);
    } else if hp < 5.0 {
        rgb = vec3<f32>(x, 0.0, c);
    } else {
        rgb = vec3<f32>(c, 0.0, x);
    }
    return rgb + vec3<f32>(m, m, m);
}

fn quad_offset(vid: u32) -> vec2<f32> {
    var offsets = array<vec2<f32>, 6>(
        vec2<f32>(-0.5, -0.5),
        vec2<f32>( 0.5, -0.5),
        vec2<f32>( 0.5,  0.5),
        vec2<f32>(-0.5, -0.5),
        vec2<f32>( 0.5,  0.5),
        vec2<f32>(-0.5,  0.5),
    );
    return offsets[vid];
}

fn tonnetz_pos(bin: u32) -> vec3<f32> {
    let freq = f32(bin) * (uniforms.sample_rate / uniforms.fft_size);
    let safe_freq = max(freq, MIN_FREQ);
    let t = (log2(safe_freq) - LOG_MIN) / (LOG_MAX - LOG_MIN);
    let y_pos = (t - 0.5) * 8.0;
    let log_ratio = log2(safe_freq / 440.0);
    let semitones = 12.0 * log_ratio;
    let pitch_class = ((semitones % 12.0) + 12.0) % 12.0;
    let theta = (2.0 * PI / 12.0) * pitch_class;
    let r = 2.0;
    return vec3<f32>(r * cos(theta), y_pos, r * sin(theta));
}

fn billboard(base: vec3<f32>, q: vec2<f32>, size: f32) -> vec4<f32> {
    let to_cam = normalize(uniforms.camera_pos - base);
    let right = normalize(cross(vec3<f32>(0.0, 1.0, 0.0), to_cam));
    let up = cross(to_cam, right);
    let world_pos = base + right * q.x * size + up * q.y * size;
    return uniforms.view_proj * vec4<f32>(world_pos, 1.0);
}

// ---- entry points ----

@vertex
fn vs_main(
    @builtin(vertex_index) vid: u32,
    @builtin(instance_index) instance: u32,
) -> VertexOutput {
    var out: VertexOutput;

    let freq = f32(instance) * (uniforms.sample_rate / uniforms.fft_size);
    let mag = magnitudes[instance];

    if freq < MIN_FREQ || freq > MAX_FREQ {
        out.position = vec4<f32>(2.0, 2.0, 2.0, 1.0);
        out.color = vec3<f32>(0.0);
        out.alpha = 0.0;
        out.uv = vec2<f32>(0.5, 0.5);
        return out;
    }

    let base = tonnetz_pos(instance);
    let q = quad_offset(vid);
    let pulse = 1.0 + uniforms.beat_pulse * 0.7;
    let size = (0.03 + mag * 0.4) * pulse;
    out.position = billboard(base, q, size);
    out.uv = q + vec2<f32>(0.5, 0.5);

    let semitones = 12.0 * log2(freq / 440.0);
    let pitch_class = ((semitones % 12.0) + 12.0) % 12.0;
    let hue = (pitch_class / 12.0) * 360.0;
    out.color = hsv_to_rgb(hue, 0.85, 0.5 + clamp(mag * 5.0, 0.0, 0.5));
    out.alpha = clamp(mag * 10.0, 0.05, 1.0);

    return out;
}

@vertex
fn vs_tartini(
    @builtin(vertex_index) vid: u32,
    @builtin(instance_index) instance: u32,
) -> VertexOutput {
    var out: VertexOutput;

    let tone_idx = instance / 2u;
    let is_sum = instance % 2u;

    if tone_idx >= uniforms.num_tartini {
        out.position = vec4<f32>(2.0, 2.0, 2.0, 1.0);
        out.color = vec3<f32>(0.0);
        out.alpha = 0.0;
        out.uv = vec2<f32>(0.5, 0.5);
        return out;
    }

    let tone = tartini_data[tone_idx];
    var bin: u32;
    if is_sum == 0u {
        bin = tone.diff_bin;
    } else {
        bin = tone.sum_bin;
    }
    let mag = tone.magnitude;

    let freq = f32(bin) * (uniforms.sample_rate / uniforms.fft_size);
    if freq < MIN_FREQ || freq > MAX_FREQ {
        out.position = vec4<f32>(2.0, 2.0, 2.0, 1.0);
        out.color = vec3<f32>(0.0);
        out.alpha = 0.0;
        out.uv = vec2<f32>(0.5, 0.5);
        return out;
    }

    let base = tonnetz_pos(bin);
    let q = quad_offset(vid);
    let pulse = 1.0 + uniforms.beat_pulse * 0.8;
    let size = (0.05 + mag * 0.6) * pulse;
    out.position = billboard(base, q, size);
    out.uv = q + vec2<f32>(0.5, 0.5);

    if is_sum == 0u {
        out.color = vec3<f32>(1.0, 0.3, 0.8);
    } else {
        out.color = vec3<f32>(0.3, 1.0, 0.8);
    }
    out.alpha = clamp(mag * 15.0, 0.1, 0.9);

    return out;
}

@fragment
fn fs_main(vert: VertexOutput) -> @location(0) vec4<f32> {
    let dist = distance(vert.uv, vec2<f32>(0.5, 0.5));
    if dist > 0.5 {
        discard;
    }
    let glow = exp(-dist * dist * 20.0);
    return vec4<f32>(vert.color * glow, vert.alpha * glow);
}
