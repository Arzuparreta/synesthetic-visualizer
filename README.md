# Synesthetic Visualizer
Click the image below to watch demos:
[![Watch the demo](https://img.youtube.com/vi/rvp7EobO_14/maxresdefault.jpg)](https://www.youtube.com/watch?v=umtcc_3KsfI&list=PL4wUTace1gknGv08vVZ4xtOPwTX8arWlG&index=1)

While traditional visualizers simply map volume to a linear frequency spectrum (an engineering view), Synesthetic Visualizer visualizes harmonic relationships, movement, and musical form by rendering harmonic and combination-tone relationships as additive Gaussian splats in 3D space.

## Key Features

* **Synesthetic Tonnetz Mapping:** Pitches are mapped to a logarithmic spiral where geometric proximity equals harmonic consonance.
* **Tartini Combination Tones:** The DSP engine detects peak frequencies and mathematically derives their sum and difference tones, drawing them with distinct palettes and time-based pulsing.
* **Tempo-Driven Camera:** Instead of jarring, binary reactions to drum hits, the camera rotation is driven by a "Leaky Integrator" event-density model. It natively understands the tempo of the music, rotating smoothly and accelerating organically during fast, dense sections of a track.

## Architecture

* **Audio Thread:** Lock-free capture (via cpal) into a ring buffer. Zero allocations, zero mutexes. On Linux, the stream is dynamically routed via pactl move-source-output to target either the PipeWire loopback monitor or the configured default source (microphone).
* **DSP Thread:** Sliding-window FFT (Hanning, 4096 bins), perceptually weighted magnitude spectrum, peak detection, and Tartini derivation. Frames are sent by value over a bounded crossbeam channel to keep the hot loop heap-allocation free.
* **Main Thread:** EMA-smoothed magnitudes pass through a peak-tracking auto-gain stage. WGSL vertex shaders place billboard quads on the 3D spiral, applying procedural Gaussian falloff in the fragment stage. Additive blending (BlendFactor::One) physically sums the light of consonant frequencies without requiring an expensive Bloom pass.

## Stack

Rust; cpal, rustfft, ringbuf, crossbeam, nannou (wgpu). Single-pass, instanced rendering.

## Prerequisites

To compile this project, you need the standard Rust toolchain. 

**Linux Users:** You will need the ALSA and Udev development headers to compile cpal and wgpu.

Command: sudo apt install libasound2-dev libudev-dev pkg-config

(Note: pactl must be available at runtime. Use PulseAudio or PipeWire with the PulseAudio compatibility layer).

**Windows Users:** WASAPI loopback is used automatically. No extra dependencies are required.

## Usage

For smooth, real-time audio processing and rendering, you must compile and run in release mode:

``cargo run --release``

### Command-Line Flags

* --mic : Captures audio from the default microphone instead of the system audio loopback. 

(Note: When running via Cargo, pass application flags after --)

``cargo run --release -- --mic``

### Controls

* **Left-Click & Drag:** Manually orbit the 3D Tonnetz spiral.
* **Scroll Wheel:** Zoom in and out.
* (When the mouse is released, the camera smoothly resumes its tempo-driven automatic rotation).
