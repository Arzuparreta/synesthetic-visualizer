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
    redirect_to_monitor(&pre_ids);

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

/// Diff source-output IDs against a pre-stream snapshot, then move the new
/// entry to the monitor source.
#[cfg(target_os = "linux")]
fn redirect_to_monitor(pre_ids: &[String]) {
    use std::process::Command;
    use std::thread;
    use std::time::Duration;

    let sources = match Command::new("pactl")
        .args(["list", "short", "sources"])
        .output()
    {
        Ok(o) => String::from_utf8_lossy(&o.stdout).to_string(),
        Err(_) => return,
    };

    let monitor = sources
        .lines()
        .filter(|l| l.to_ascii_lowercase().contains("monitor"))
        .find(|l| l.contains("RUNNING"))
        .or_else(|| {
            sources
                .lines()
                .find(|l| l.to_ascii_lowercase().contains("monitor"))
        })
        .and_then(|l| l.split('\t').nth(1))
        .map(|s| s.to_string());

    let monitor = match monitor {
        Some(m) => m,
        None => {
            eprintln!("No monitor source found, using mic");
            return;
        }
    };

    for attempt in 0..6 {
        thread::sleep(Duration::from_millis(if attempt == 0 { 100 } else { 200 }));

        let post_ids = snapshot_source_output_ids();
        let new_id = post_ids.iter().find(|id| !pre_ids.contains(id));

        if let Some(id) = new_id {
            match Command::new("pactl")
                .args(["move-source-output", id, &monitor])
                .status()
            {
                Ok(s) if s.success() => {
                    eprintln!("Redirected to monitor: {monitor}");
                    return;
                }
                _ => eprintln!("move-source-output failed"),
            }
        }
    }

    eprintln!("Stream not registered in PipeWire, using mic");
}
