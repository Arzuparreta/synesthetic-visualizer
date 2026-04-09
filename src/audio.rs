use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use ringbuf::traits::{Observer, Producer, Split};
use ringbuf::{HeapCons, HeapRb};

const RING_BUFFER_CAPACITY: usize = 8192;

pub struct AudioCapture {
    pub stream: cpal::Stream,
    pub consumer: HeapCons<f32>,
    pub sample_rate: u32,
}

pub fn start_capture(use_mic: bool) -> AudioCapture {
    let host = cpal::default_host();

    #[cfg(target_os = "linux")]
    let pre_ids = snapshot_source_output_ids();

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

    #[cfg(target_os = "linux")]
    redirect_stream(&pre_ids, use_mic);

    AudioCapture {
        stream,
        consumer,
        sample_rate,
    }
}

fn select_device(host: &cpal::Host) -> (cpal::Device, cpal::SupportedStreamConfig) {
    #[cfg(target_os = "windows")]
    {
        if let Some(dev) = host.default_output_device() {
            if let Ok(cfg) = dev.default_input_config() {
                eprintln!("Loopback device: {}", dev.name().unwrap_or_default());
                return (dev, cfg);
            }
        }
    }

    let dev = host
        .default_input_device()
        .expect("no audio input device available");
    let cfg = dev
        .default_input_config()
        .expect("no default input config");
    eprintln!("Input device: {}", dev.name().unwrap_or_default());
    (dev, cfg)
}

#[cfg(target_os = "linux")]
fn snapshot_source_output_ids() -> Vec<String> {
    use std::process::Command;
    Command::new("pactl")
        .args(["list", "short", "source-outputs"])
        .output()
        .ok()
        .map(|o| {
            String::from_utf8_lossy(&o.stdout)
                .lines()
                .filter_map(|l| l.split('\t').next().map(|id| id.to_string()))
                .collect()
        })
        .unwrap_or_default()
}

/// Diff source-output IDs against a pre-stream snapshot and move the new
/// stream to the target source. Target is the running monitor for loopback,
/// or the PipeWire default source for mic mode.
#[cfg(target_os = "linux")]
fn redirect_stream(pre_ids: &[String], use_mic: bool) {
    use std::process::Command;
    use std::thread;
    use std::time::Duration;

    let target = if use_mic {
        // Use the configured default source (the actual mic input).
        match Command::new("pactl").args(["get-default-source"]).output() {
            Ok(o) => {
                let name = String::from_utf8_lossy(&o.stdout).trim().to_string();
                if name.is_empty() {
                    eprintln!("Could not determine default source");
                    return;
                }
                eprintln!("Mic source: {name}");
                name
            }
            Err(_) => return,
        }
    } else {
        let sources = match Command::new("pactl")
            .args(["list", "short", "sources"])
            .output()
        {
            Ok(o) => String::from_utf8_lossy(&o.stdout).to_string(),
            Err(_) => return,
        };
        match sources
            .lines()
            .filter(|l| l.to_ascii_lowercase().contains("monitor"))
            .find(|l| l.contains("RUNNING"))
            .or_else(|| {
                sources
                    .lines()
                    .find(|l| l.to_ascii_lowercase().contains("monitor"))
            })
            .and_then(|l| l.split('\t').nth(1))
            .map(|s| s.to_string())
        {
            Some(m) => {
                eprintln!("Monitor source: {m}");
                m
            }
            None => {
                eprintln!("No monitor source found");
                return;
            }
        }
    };

    for attempt in 0..6 {
        thread::sleep(Duration::from_millis(if attempt == 0 { 100 } else { 200 }));

        let post_ids = snapshot_source_output_ids();
        if let Some(id) = post_ids.iter().find(|id| !pre_ids.contains(id)) {
            match Command::new("pactl")
                .args(["move-source-output", id, &target])
                .status()
            {
                Ok(s) if s.success() => {
                    eprintln!("Redirected stream #{id} -> {target}");
                    return;
                }
                _ => {}
            }
        }
    }

    eprintln!("Could not redirect stream to {target}");
}
