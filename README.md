# Synesthetic Visualizer

A real-time audio visualizer that maps frequency content onto a three-dimensional Tonnetz spiral. 

While traditional visualizers simply map volume to a linear frequency spectrum (an engineering view), this visualizes **harmonic relationships, movement, and musical form**. By rendering harmonic and combination-tone relationships as additive Gaussian splats in 3D space, it creates a fluid, perceptually grounded representation of a song's structural harmony.

## Key Features
* **Synesthetic Tonnetz Mapping:** Pitches are mapped to a logarithmic spiral where geometric proximity equals harmonic consonance.
* **Tartini Combination Tones:** The engine detects peak frequencies and mathematically derives their sum and difference tones, drawing them with distinct palettes and time-based pulsing.
* **Tempo-Driven Kinetic Camera:** Instead of jarring, binary reactions to drum hits, the camera rotation is driven by a "Leaky Integrator" event-density model. It natively understands the tempo of the music, rotating smoothly and accelerating organically during fast, dense sections of a track.

## Architecture

* **Audio Thread:** Lock-free capture (via `cpal`) into a ring buffer. Zero allocations, zero mutexes. On Linux, the stream is dynamically routed via `pactl move-source-output` to target either the PipeWire loopback monitor or the default microphone.
* **DSP Thread:** Sliding-window FFT (Hanning, 4096 bins), perceptually weighted magnitude spectrum, peak detection, and Tartini derivation. Frames are sent by value over a bounded `crossbeam` channel to keep the hot loop heap-allocation free.
* **Main Thread:** EMA-smoothed magnitudes pass through a peak-tracking auto-gain stage. WGSL vertex shaders place billboard quads on the 3D spiral, applying procedural Gaussian falloff in the fragment stage. Additive blending (`BlendFactor::One`) physically sums the light of consonant frequencies without requiring an expensive post-processing Bloom pass.

## Prerequisites

To compile this project, you need the standard Rust toolchain. 

**Linux Users:** You will need the ALSA and Udev development headers to compile `cpal` and `wgpu`.
```bash
sudo apt install libasound2-dev libudev-dev pkg-config
