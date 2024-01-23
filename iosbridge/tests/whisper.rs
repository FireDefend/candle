use std::{sync::{Arc, Mutex}, fmt::Debug};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use rubato::{Resampler, SincFixedIn, SincInterpolationType, SincInterpolationParameters, WindowFunction};
use cpal::Stream;
use candle::Device;
use anyhow::Error;
use iosbridge::ioswhisper::{IOSWhisperModel,Recorder};
use candle_transformers::models::whisper::{self as m};
extern crate sdl2;
use sdl2::audio::{AudioCallback, AudioSpecDesired};
use std::time::Duration;



#[test]
fn test_whisper() ->Result<(), Error> {
    let device = Device::Cpu;
    let mut ioswhisper = IOSWhisperModel::new("/Users/xigsun/Documents/repo/whisper-base/", &device)?;
    ioswhisper.record();


    // Set up the resampler
    // let params = SincInterpolationParameters {
    //         sinc_len: 256,
    //         f_cutoff: 0.95,
    //         interpolation: SincInterpolationType::Linear,
    //         oversampling_factor: 256,
    //         window: WindowFunction::BlackmanHarris2,
    // };
    // let mut resampler = SincFixedIn::<f32>::new(
    //         sample_rate_out as f64 / sample_rate_in as f64,
    //         2.0,
    //         params,
    //         1024,
    //         1,
    // ).unwrap();


    // 阻塞主线程，使音频流保持活动状态
    println!("recording");
    std::thread::sleep(std::time::Duration::from_secs(8));
    ioswhisper.stoprecord();
    let token = ioswhisper.detectLanguage();
    ioswhisper.inferenceMel(token, None);
    // let waves_in = vec![final_data;1];
    // let waves_out = resampler.process(&waves_in, None).unwrap();
    // let result = waves_out[0].clone();

    // let output_device = host.default_output_device().expect("Failed to find output device");
    // let mut supported_output_configs_range = device.supported_output_configs()?;
    // let all_output_configs:Vec<_> = supported_output_configs_range.collect();
    // let mut output_config = output_device.default_output_config()?.config();
    // output_config.channels = 1;
    // let output_buffer = Arc::clone(&alldata);
    // let mut reversed_data = {
    //     let lock = output_buffer.lock().unwrap();
    //     let mut cloned_data = lock.clone();
    //     cloned_data.reverse();
    //     cloned_data
    // };
    // let output_stream = output_device.build_output_stream(
    //     &output_config,
    //     move |data: &mut [f32], _: &cpal::OutputCallbackInfo| {
    //         for elem in data.iter_mut() {
    //             *elem = reversed_data.pop().unwrap_or(0.0);
    //         }
    //     },
    //     |err| {
    //         eprintln!("Output error: {}", err);
    //     },
    //     None,
    // )?;
    // output_stream.play()?;
    // std::thread::sleep(std::time::Duration::from_secs(10));

    // let mut input = std::fs::File::open(std::path::PathBuf::from("/Users/xigsun/Downloads/samples_jfk.wav"))?;
    // let (header, data) = wav::read(&mut input)?;
    // println!("loaded wav data: {header:?}");
    // if header.sampling_rate != m::SAMPLE_RATE as u32 {
    //     anyhow::bail!("wav file must have a {} sampling rate", m::SAMPLE_RATE)
    // }
    // let data = data.as_sixteen().expect("expected 16 bit wav file");
    // let pcm_data: Vec<_> = data[..data.len() / header.channel_count as usize]
    //     .iter()
    //     .map(|v| *v as f32 / 32768.)
    //     .collect();
    Ok(())
}

