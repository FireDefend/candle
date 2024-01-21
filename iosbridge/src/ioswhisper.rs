use anyhow::{Error as E, Result};
use candle::{Device, IndexOp, Tensor};
use candle_nn::{ops::softmax, VarBuilder};
use clap::{Parser, ValueEnum};
use rand::{distributions::Distribution, SeedableRng};
use sdl2::AudioSubsystem;
use tokenizers::Tokenizer;
use std::os::raw::{c_char};
use std::ffi::{CString, CStr};
use std::sync::atomic::{Ordering, AtomicBool};
use std::sync::{Mutex, Arc};
use std::time::Instant;
use sdl2::audio::{AudioCallback, AudioSpecDesired};

use candle_transformers::models::whisper::{self as m, audio, Config};
mod multilingual;
pub enum Model {
    Normal(m::model::Whisper),
    Quantized(m::quantized_model::Whisper),
}
// Maybe we should use some traits rather than doing the dispatch for all these.
impl Model {
    pub fn config(&self) -> &Config {
        match self {
            Self::Normal(m) => &m.config,
            Self::Quantized(m) => &m.config,
        }
    }

    pub fn encoder_forward(&mut self, x: &Tensor, flush: bool) -> candle::Result<Tensor> {
        match self {
            Self::Normal(m) => m.encoder.forward(x, flush),
            Self::Quantized(m) => m.encoder.forward(x, flush),
        }
    }

    pub fn decoder_forward(
        &mut self,
        x: &Tensor,
        xa: &Tensor,
        flush: bool,
    ) -> candle::Result<Tensor> {
        match self {
            Self::Normal(m) => m.decoder.forward(x, xa, flush),
            Self::Quantized(m) => m.decoder.forward(x, xa, flush),
        }
    }

    pub fn decoder_final_linear(&self, x: &Tensor) -> candle::Result<Tensor> {
        match self {
            Self::Normal(m) => m.decoder.final_linear(x),
            Self::Quantized(m) => m.decoder.final_linear(x),
        }
    }
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
struct DecodingResult {
    tokens: Vec<u32>,
    text: String,
    avg_logprob: f64,
    no_speech_prob: f64,
    temperature: f64,
    compression_ratio: f64,
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
struct Segment {
    start: f64,
    duration: f64,
    dr: DecodingResult,
}

struct Decoder {
    model: Model,
    rng: rand::rngs::StdRng,
    task: Option<Task>,
    timestamps: bool,
    verbose: bool,
    tokenizer: Tokenizer,
    suppress_tokens: Tensor,
    sot_token: u32,
    transcribe_token: u32,
    translate_token: u32,
    eot_token: u32,
    no_speech_token: u32,
    no_timestamps_token: u32,
    language_token: Option<u32>,
}

impl Decoder {
    #[allow(clippy::too_many_arguments)]
    fn new(
        model: Model,
        tokenizer: Tokenizer,
        seed: u64,
        device: &Device,
        language_token: Option<u32>,
        task: Option<Task>,
        timestamps: bool,
        verbose: bool,
    ) -> Result<Self> {
        let no_timestamps_token = token_id(&tokenizer, m::NO_TIMESTAMPS_TOKEN)?;
        // Suppress the notimestamps token when in timestamps mode.
        // https://github.com/openai/whisper/blob/e8622f9afc4eba139bf796c210f5c01081000472/whisper/decoding.py#L452
        let suppress_tokens: Vec<f32> = (0..model.config().vocab_size as u32)
            .map(|i| {
                if model.config().suppress_tokens.contains(&i)
                    || timestamps && i == no_timestamps_token
                {
                    f32::NEG_INFINITY
                } else {
                    0f32
                }
            })
            .collect();
        let suppress_tokens = Tensor::new(suppress_tokens.as_slice(), device)?;
        let sot_token = token_id(&tokenizer, m::SOT_TOKEN)?;
        let transcribe_token = token_id(&tokenizer, m::TRANSCRIBE_TOKEN)?;
        let translate_token = token_id(&tokenizer, m::TRANSLATE_TOKEN)?;
        let eot_token = token_id(&tokenizer, m::EOT_TOKEN)?;
        let no_speech_token = m::NO_SPEECH_TOKENS
            .iter()
            .find_map(|token| token_id(&tokenizer, token).ok());
        let no_speech_token = match no_speech_token {
            None => anyhow::bail!("unable to find any non-speech token"),
            Some(n) => n,
        };
        Ok(Self {
            model,
            rng: rand::rngs::StdRng::seed_from_u64(seed),
            tokenizer,
            task,
            timestamps,
            verbose,
            suppress_tokens,
            sot_token,
            transcribe_token,
            translate_token,
            eot_token,
            no_speech_token,
            language_token,
            no_timestamps_token,
        })
    }

    fn decode(&mut self, mel: &Tensor, t: f64, predictionStringCallback: Option<extern "C" fn(*const c_char)>,) -> Result<DecodingResult> {
        let model = &mut self.model;
        let audio_features = model.encoder_forward(mel, true)?;
        if self.verbose {
            println!("audio features: {:?}", audio_features.dims());
        }
        let sample_len = model.config().max_target_positions / 2;
        let mut sum_logprob = 0f64;
        let mut no_speech_prob = f64::NAN;
        let mut tokens = vec![self.sot_token];
        if let Some(language_token) = self.language_token {
            tokens.push(language_token);
        }
        match self.task {
            None | Some(Task::Transcribe) => tokens.push(self.transcribe_token),
            Some(Task::Translate) => tokens.push(self.translate_token),
        }
        if !self.timestamps {
            tokens.push(self.no_timestamps_token);
        }
        for i in 0..sample_len {
            let tokens_t = Tensor::new(tokens.as_slice(), mel.device())?;

            // The model expects a batch dim but this inference loop does not handle
            // it so we add it at this point.
            let tokens_t = tokens_t.unsqueeze(0)?;
            let ys = model.decoder_forward(&tokens_t, &audio_features, i == 0)?;

            // Extract the no speech probability on the first iteration by looking at the first
            // token logits and the probability for the according token.
            if i == 0 {
                let logits = model.decoder_final_linear(&ys.i(..1)?)?.i(0)?.i(0)?;
                no_speech_prob = softmax(&logits, 0)?
                    .i(self.no_speech_token as usize)?
                    .to_scalar::<f32>()? as f64;
            }

            let (_, seq_len, _) = ys.dims3()?;
            let logits = model
                .decoder_final_linear(&ys.i((..1, seq_len - 1..))?)?
                .i(0)?
                .i(0)?;
            // TODO: Besides suppress tokens, we should apply the heuristics from
            // ApplyTimestampRules, i.e.:
            // - Timestamps come in pairs, except before EOT.
            // - Timestamps should be non-decreasing.
            // - If the sum of the probabilities of timestamps is higher than any other tokens,
            //   only consider timestamps when sampling.
            // https://github.com/openai/whisper/blob/e8622f9afc4eba139bf796c210f5c01081000472/whisper/decoding.py#L439
            let logits = logits.broadcast_add(&self.suppress_tokens)?;
            let next_token = if t > 0f64 {
                let prs = softmax(&(&logits / t)?, 0)?;
                let logits_v: Vec<f32> = prs.to_vec1()?;
                let distr = rand::distributions::WeightedIndex::new(&logits_v)?;
                distr.sample(&mut self.rng) as u32
            } else {
                let logits_v: Vec<f32> = logits.to_vec1()?;
                logits_v
                    .iter()
                    .enumerate()
                    .max_by(|(_, u), (_, v)| u.total_cmp(v))
                    .map(|(i, _)| i as u32)
                    .unwrap()
            };
            tokens.push(next_token);
            let prob = softmax(&logits, candle::D::Minus1)?
                .i(next_token as usize)?
                .to_scalar::<f32>()? as f64;
            match predictionStringCallback{
                Some(callback) =>{
                    let re = self.tokenizer.decode(&tokens, true).map_err(E::msg)?;
                    let c_string = CString::new(re).expect("CString::new failed in call back");
                    callback(c_string.as_ptr());
                },
                None => {},
            }
            if next_token == self.eot_token || tokens.len() > model.config().max_target_positions {
                break;
            }
            sum_logprob += prob.ln();
        }
        let text = self.tokenizer.decode(&tokens, true).map_err(E::msg)?;
        let avg_logprob = sum_logprob / tokens.len() as f64;

        Ok(DecodingResult {
            tokens,
            text,
            avg_logprob,
            no_speech_prob,
            temperature: t,
            compression_ratio: f64::NAN,
        })
    }

    fn decode_with_fallback(&mut self, segment: &Tensor, predictionStringCallback: Option<extern "C" fn(*const c_char)>,) -> Result<DecodingResult> {
        for (i, &t) in m::TEMPERATURES.iter().enumerate() {
            let dr: Result<DecodingResult> = self.decode(segment, t, predictionStringCallback);
            if i == m::TEMPERATURES.len() - 1 {
                return dr;
            }
            // On errors, we try again with a different temperature.
            match dr {
                Ok(dr) => {
                    let needs_fallback = dr.compression_ratio > m::COMPRESSION_RATIO_THRESHOLD
                        || dr.avg_logprob < m::LOGPROB_THRESHOLD;
                    if !needs_fallback || dr.no_speech_prob > m::NO_SPEECH_THRESHOLD {
                        return Ok(dr);
                    }
                }
                Err(err) => {
                    println!("Error running at {t}: {err}")
                }
            }
        }
        unreachable!()
    }

    fn run(&mut self, mel: &Tensor,language_token: Option<u32>, predictionStringCallback: Option<extern "C" fn(*const c_char)>,) -> Result<String> {
        self.language_token = language_token;
        let (_, _, content_frames) = mel.dims3()?;
        let mut seek = 0;
        let mut segments = vec![];
        let mut result =  String::from("");
        while seek < content_frames {
            let start = std::time::Instant::now();
            
            let time_offset = (seek * self.model.config().num_mel_bins) as f64 / m::SAMPLE_RATE as f64;
            let segment_size = usize::min(content_frames - seek, m::N_FRAMES);
            let mel_segment = mel.narrow(2, seek, segment_size)?;
            let segment_duration = (segment_size * self.model.config().num_mel_bins) as f64 / m::SAMPLE_RATE as f64;
            let dr = self.decode_with_fallback(&mel_segment, predictionStringCallback)?;
            seek += segment_size;
            if dr.no_speech_prob > m::NO_SPEECH_THRESHOLD && dr.avg_logprob < m::LOGPROB_THRESHOLD {
                println!("no speech detected, skipping {seek} {dr:?}");
                continue;
            }
            let segment = Segment {
                start: time_offset,
                duration: segment_duration,
                dr,
            };
            if self.timestamps {
                println!(
                    "{:.1}s -- {:.1}s",
                    segment.start,
                    segment.start + segment.duration,
                );
                let mut tokens_to_decode = vec![];
                let mut prev_timestamp_s = 0f32;
                for &token in segment.dr.tokens.iter() {
                    if token == self.sot_token || token == self.eot_token {
                        continue;
                    }
                    // The no_timestamp_token is the last before the timestamp ones.
                    if token > self.no_timestamps_token {
                        let timestamp_s = (token - self.no_timestamps_token + 1) as f32 / 50.;
                        if !tokens_to_decode.is_empty() {
                            let text = self
                                .tokenizer
                                .decode(&tokens_to_decode, true)
                                .map_err(E::msg)?;
                            println!("  {:.1}s-{:.1}s: {}", prev_timestamp_s, timestamp_s, text);
                            tokens_to_decode.clear()
                        }
                        prev_timestamp_s = timestamp_s;
                    } else {
                        tokens_to_decode.push(token)
                    }
                }
                if !tokens_to_decode.is_empty() {
                    let text = self
                        .tokenizer
                        .decode(&tokens_to_decode, true)
                        .map_err(E::msg)?;
                    if !text.is_empty() {
                        println!("  {:.1}s-...: {}", prev_timestamp_s, text);
                    }
                    tokens_to_decode.clear()
                }
            } else {
                println!(
                    "{:.1}s -- {:.1}s: {}",
                    segment.start,
                    segment.start + segment.duration,
                    segment.dr.text,
                );
                result = result + &segment.dr.text;
            }
            if self.verbose {
                println!("{seek}: {segment:?}, in {:?}", start.elapsed());
            }
            segments.push(segment)
        }
        Ok(result)
    }
}

pub fn token_id(tokenizer: &Tokenizer, token: &str) -> candle::Result<u32> {
    match tokenizer.token_to_id(token) {
        None => candle::bail!("no token-id for {token}"),
        Some(id) => Ok(id),
    }
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum Task {
    Transcribe,
    Translate,
}

pub struct Recorder {
    pub buffer: Arc<Mutex<Vec<f32>>>,
}

impl AudioCallback for Recorder {
    type Channel = f32;

    fn callback(&mut self, input: &mut [f32]) {
        let mut buffer = self.buffer.lock().unwrap();
        buffer.extend_from_slice(input);
    }
}


pub struct IOSWhisperModel {
    decoder: Decoder,
    config: Config,
    device: Device,
    mel_filters: Vec<f32>,
    audio_subsystem: AudioSubsystem,
    desired_spec: AudioSpecDesired,
    recoderdata: Arc<Mutex<Vec<f32>>>,
    capture_device: Option<sdl2::audio::AudioDevice<Recorder>>,
    start: Instant,
}

impl IOSWhisperModel {
    pub fn new(path:&str, devicein: &Device) -> Result<Self,E> {
        let folder_path = std::path::PathBuf::from(path.to_owned());
        let mut tokenizer_path = std::path::PathBuf::from(path.to_owned() + "tokenizer.json");
        let tokenizer = Tokenizer::from_file(tokenizer_path).map_err(E::msg)?;
        let config = serde_json::from_str(&std::fs::read_to_string(path.to_owned() + "config.json")?)?;
        let model_path = std::path::PathBuf::from(path.to_owned() + "model.safetensors");
        let mut model =  {
            let vb =
                unsafe { VarBuilder::from_mmaped_safetensors(&[model_path], m::DTYPE, &devicein)? };
            Model::Normal(m::model::Whisper::load(&vb, config)?)
        };
        let config: Config = serde_json::from_str(&std::fs::read_to_string(path.to_owned() + "config.json")?)?;
        let mel_bytes = match config.num_mel_bins {
            80 => include_bytes!("melfilters.bytes").as_slice(),
            128 => include_bytes!("melfilters128.bytes").as_slice(),
            nmel => anyhow::bail!("unexpected num_mel_bins {nmel}"),
        };
        let mut mel_filters = vec![0f32; mel_bytes.len() / 4];
        <byteorder::LittleEndian as byteorder::ByteOrder>::read_f32_into(mel_bytes, &mut mel_filters);
        let mut decoder = Decoder::new(
            model,
            tokenizer,
            299792458, //seed
            &devicein,
            None,
            Some(Task::Transcribe),
            false,
            true,
        )?;

        let audio_subsystem = sdl2::init().expect("Failed to find input device").audio().expect("Failed to find input audio device");
        println!("SDL2 Freq: 16000 Buffer Len: 4096");
        let desired_spec = AudioSpecDesired {
            freq: Some(16000),
            channels: Some(1), // Mono
            samples: Some(4096),     // Default sample size
        };
        let recoderdata = Arc::new(Mutex::new(Vec::new()));
        Ok(Self {
            decoder,
            config,
            device:devicein.clone(),
            mel_filters,
            audio_subsystem,
            desired_spec,
            recoderdata,
            capture_device:None,
            start:Instant::now(),
        })
    }

    pub fn inference(
        &mut self,
        input:Vec<f32>,
        languagetoken: Option<String>,
        predictionStringCallback: Option<extern "C" fn(*const c_char)>,
    ) -> Result<String,E> {
        
        let mel = audio::pcm_to_mel(&self.config, &input, &self.mel_filters);
        //let mel :Vec<f32>= input.iter().take(self.config.num_mel_bins * (input.len() / self.config.num_mel_bins)).cloned().collect();//audio::pcm_to_mel(&self.config, &input, &self.mel_filters);
        let mel_len = mel.len();
        let mel = Tensor::from_vec(
            mel,
            (1, self.config.num_mel_bins, mel_len / self.config.num_mel_bins),
            &self.device,
        )?;
        println!("loaded mel: {:?}", mel.dims());

        let languagetoken = match languagetoken{
            Some(languagetoken) =>{
                languagetoken
            },
            None => {                
                let re = multilingual::detect_language(&mut self.decoder.model, &self.decoder.tokenizer, &mel).expect("whidper detect_language failed");
                println!("detected language!");
                re
            },
        };

        let language = Some(crate::ioswhisper::token_id(&self.decoder.tokenizer, &languagetoken)?);

        return Ok(self.decoder.run(&mel,language, predictionStringCallback)?.to_owned());
    }

    pub fn recordandinference(
        &mut self,
        languagetoken: Option<String>,
        predictionStringCallback: Option<extern "C" fn(*const c_char)>,
    )  {
        self.recoderdata = Arc::new(Mutex::new(Vec::new()));
        let alldata_clone = Arc::clone(&self.recoderdata);
        self.capture_device = Some(self.audio_subsystem.open_capture(None, &self.desired_spec, |_| {
            Recorder { buffer: alldata_clone }
        }).expect("Failed to find input capture audio device"));
        // Start recording and playback
        self.start = Instant::now();
        (self.capture_device).as_mut().expect("Failed to find input capture audio device").resume();

    }
    pub fn stoprecord(
        &mut self,
        languagetoken: Option<String>,
    ) -> Result<String,E> {
        (self.capture_device).as_mut().expect("Failed to find input capture audio device").pause();
        self.capture_device = None;
        let final_data = {
            let alldata = self.recoderdata.lock().unwrap();
            alldata.clone()
        };
        println!("data size {:?}",final_data.len());
        let elapsed = self.start.elapsed();

        // 打印出所用的时间
        println!("Elapsed time: {:.2?}", elapsed);
        return Ok(self.inference(final_data,languagetoken,None)?.to_owned());
    }

    pub fn play(
        &mut self,
    ) -> Result<String,E> {
        // Set up playback device
        let playback_finished = Arc::new(AtomicBool::new(false));

        let playback_device = self.audio_subsystem.open_playback(None, &self.desired_spec,  |_| {
            Playback {
                buffer: Arc::clone(&self.recoderdata),
                position: 0,
                finished: Arc::clone(&playback_finished),
            }
        }).expect("Failed to open playback device");

        playback_device.resume(); // Start playing
        

        loop {
            if playback_finished.load(Ordering::Relaxed) {
                break; // Stop the loop if playback is finished
            }
            // You can adjust the sleep duration as needed
            std::thread::sleep(std::time::Duration::from_millis(100));
        }
    
        // Clean up
        drop(playback_device);
        return Ok("".to_owned());
    }
}

struct Playback{
    buffer: Arc<Mutex<Vec<f32>>>,
    position: usize,
    finished: Arc<AtomicBool>,
}

impl AudioCallback for Playback{
    type Channel = f32;

    fn callback(&mut self, out: &mut [f32]) {
        let mut buffer = self.buffer.lock().unwrap();
        let buffer_len = buffer.len();

        for dst in out.iter_mut() {
            if self.position >= buffer_len {
                *dst = 0.0; // Output silence if we've played all data
                self.finished.store(true, Ordering::Relaxed); // Signal that playback is finished
                break;
            } else {
                *dst = buffer[self.position];
                self.position += 1;
            }
        }
    }
}

#[no_mangle]
pub extern "C" fn ios_whisper_model_new(path: *const c_char, gpu: bool) -> *mut IOSWhisperModel {
    // 从 C 字符串转换为 Rust 字符串
    let path_str = unsafe { CStr::from_ptr(path).to_string_lossy().into_owned() };
    let device;
    if(gpu){
        device = match Device::new_metal(0) {
            Ok(result) => {
                println!("gpu enabled");
                result
            },
            Err(e) =>{ 
                eprintln!("gpu enabled failed, using cpu");
                Device::Cpu
            },
        };
    }else{
        device = Device::Cpu;
    }
    match IOSWhisperModel::new(&path_str, &device) {
        Ok(model) => Box::into_raw(Box::new(model)),
        Err(e) => { 
            eprintln!("{}", e);
            std::ptr::null_mut()
        },
    }
}

#[no_mangle]
pub extern "C" fn ios_whisper_model_inference(ptr: *mut IOSWhisperModel, languagetoken: *const c_char){
    if ptr.is_null() {
        eprintln!("Error: iosmt_model_inference_new null");
        return ;
    }
    let languagetoken_str: Option<String>;
    if(languagetoken.is_null()){
        languagetoken_str = None;
    }else{
        languagetoken_str = Some(unsafe { CStr::from_ptr(languagetoken).to_string_lossy().into_owned() });
    }
    // 把原始指针转换回 Box，这将确保资源被正确释放
    let mut model_box = unsafe { Box::from_raw(ptr) };
    model_box.recordandinference(languagetoken_str,None);
    std::mem::forget(model_box);
}

#[no_mangle]
pub extern "C" fn ios_whisper_model_stop_record(ptr: *mut IOSWhisperModel, languagetoken: *const c_char) -> *mut c_char {
    if ptr.is_null() {
        eprintln!("Error: iosmt_model_inference null");
        return std::ptr::null_mut();
    }
    let languagetoken_str: Option<String>;
    if(languagetoken.is_null()){
        languagetoken_str = None;
    }else{
        languagetoken_str = Some(unsafe { CStr::from_ptr(languagetoken).to_string_lossy().into_owned() });
    }
    let mut model_box = unsafe { Box::from_raw(ptr) };
    
    let result_ptr = match model_box.stoprecord(languagetoken_str){
        Ok(result) => {
            match CString::new(result) {
                Ok(c_string) => c_string.into_raw(),
                Err(_) => std::ptr::null_mut(),
            }
        },
        Err(e) =>{ 
            eprintln!("{}", e);
            std::ptr::null_mut()
        },
    };
    std::mem::forget(model_box);
    result_ptr
}

#[no_mangle]
pub extern "C" fn ios_whisper_model_record_play(ptr: *mut IOSWhisperModel){
    if ptr.is_null() {
        eprintln!("Error: iosmt_model_inference null");
        return ;
    }
    let mut model_box = unsafe { Box::from_raw(ptr) };
    model_box.play();
    std::mem::forget(model_box);
}

#[no_mangle]
pub extern "C" fn ios_whisper_model_free(ptr: *mut IOSWhisperModel) {
    if ptr.is_null() {
        // 处理错误或提前返回
        eprintln!("Error: iosmt_model_free null");
        return;
    }
    // 把原始指针转换回 Box，这将确保资源被正确释放
    unsafe { Box::from_raw(ptr) };
}