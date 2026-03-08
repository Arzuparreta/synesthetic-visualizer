use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use ringbuf::traits::{Observer, Producer, Split};
use ringbuf::{HeapCons, HeapRb};

const RING_BUFFER_CAPACITY: usize = 8192;

pub struct AudioCapture {
    pub stream: cpal::Stream,
    pub consumer: HeapCons<f32>,
    pub sample_rate: u32,
}

pub fn start_capture() -> AudioCapture {
    let host = cpal::default_host();

    let (device, config) = select_device(&host);

    let sample_rate = config.sample_rate().0;
    let channels = config.channels() as usize;
    eprintln!("Audio config: {sample_rate}Hz, {channels} ch");

    let stream_config: cpal::StreamConfig = config.into();
    let (mut producer, consumer) = HeapRb::<f32>::new(RING_BUFFER_CAPACITY).split();

    let stream = device
        .build_input_stream(
            &stream_config,
            move |data: &[f32], _: &cpal::InputCallbackInfo| {
                if channels == 1 {
                    if producer.vacant_len() >= data.len() {
                        producer.push_slice(data);
                    }
                } else {
                    for chunk in data.chunks_exact(channels) {
                        if producer.vacant_len() < 1 {
                            break;
                        }
                        let _ = producer.try_push(chunk[0]);
                    }
                }
            },
            |err| eprintln!("audio error: {err}"),
            None,
        )
        .expect("failed to build input stream");

    stream.play().expect("failed to start audio stream");

    AudioCapture {
        stream,
        consumer,
        sample_rate,
    }
}

fn select_device(host: &cpal::Host) -> (cpal::Device, cpal::SupportedStreamConfig) {
    // Attempt 1: Output device loopback (WASAPI on Windows captures this natively).
    if let Some(dev) = host.default_output_device() {
        if let Ok(cfg) = dev.default_input_config() {
            eprintln!(
                "Loopback device: {}",
                dev.name().unwrap_or_default()
            );
            return (dev, cfg);
        }
    }

    // Attempt 2 (Linux): PulseAudio / PipeWire expose a "Monitor" source.
    if cfg!(target_os = "linux") {
        if let Ok(inputs) = host.input_devices() {
            for dev in inputs {
                let name = dev.name().unwrap_or_default();
                if name.to_ascii_lowercase().contains("monitor") {
                    if let Ok(cfg) = dev.default_input_config() {
                        eprintln!("Monitor device: {name}");
                        return (dev, cfg);
                    }
                }
            }
        }
    }

    // Attempt 3: Fallback to default microphone.
    let dev = host
        .default_input_device()
        .expect("no audio input device available");
    let cfg = dev
        .default_input_config()
        .expect("no default input config");
    eprintln!(
        "Fallback mic: {}",
        dev.name().unwrap_or_default()
    );
    (dev, cfg)
}
