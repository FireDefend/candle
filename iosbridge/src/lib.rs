pub mod ioswhisper;

use std::os::raw::{c_char};
use std::ffi::{CString, CStr};
use anyhow::Error as E;
use tokenizers::{Tokenizer, InputSequence};
use candle_transformers::models::marian::{MTModel, self};
use candle_nn::VarBuilder;
use candle::{DType, Tensor, Device};
use candle_examples::token_output_stream::TokenOutputStream;
use std::time::Instant;

pub struct IOSMTModel {
    model: MTModel,
    inputtokenizer: Tokenizer,
    outputtokenizer: Tokenizer,
    config: marian::Config,
    device: Device,
}
fn decode_output(inputtokenizer: &Tokenizer,outputtokenizer: &Tokenizer,tokens: &[u32])-> Result<String,E> {
    let mut decode_result: String =  String::new();
    for token in 0..tokens.len() {
        let mut t = outputtokenizer.decode(&tokens[token..token+1], true).map_err(E::msg)?;
        if(t.eq("<NIL>")){
            t = inputtokenizer.decode(&tokens[token..token+1], true).map_err(E::msg)?;
        }
        decode_result = decode_result + &t+ " ";
    }
    Ok(decode_result)
}
impl IOSMTModel {
    pub fn new(path:&str, devicein: &Device) -> Result<Self,E> {
        let folder_path = std::path::PathBuf::from(path.to_owned());
        let folder_name = folder_path.file_name().unwrap().to_string_lossy().to_string();
        let folder_name_parts: Vec<&str> = folder_name.split('-').collect();
        let mut tokenizer_path = std::path::PathBuf::from(path.to_owned() + &format!("tokenizer-marian-base-{}.json", folder_name_parts[folder_name_parts.len()-2]));
        let inputtokenizer = {
            Tokenizer::from_file(&tokenizer_path).map_err(E::msg)?
        };
        let mut tokenizer_path = std::path::PathBuf::from(path.to_owned() + &format!("tokenizer-marian-base-{}.json","big")); //folder_name_parts[folder_name_parts.len()-1]));
        let outputtokenizer = {
            Tokenizer::from_file(&tokenizer_path).map_err(E::msg)?
        };
        tokenizer_path = std::path::PathBuf::from(path.to_owned() + "model.safetensors");
        let vb = { unsafe { VarBuilder::from_pth(path.to_owned() + "model.pth", DType::F32, &devicein)? }
        };
        let config = marian::Config::read_from_file(path.to_owned() + "config.json");
        let model: MTModel = marian::MTModel::new(&config, vb)?;
        Ok(Self {
            model,
            inputtokenizer,
            outputtokenizer,
            config,
            device:devicein.clone(),
        })
    }
    // can not be run in paralized since model has kv_cache
    pub fn inference(
        &mut self,
        input:String,
        predictionStringCallback: Option<extern "C" fn(*const c_char)>,
    ) -> Result<String,E> {
        let device = &self.device;
        self.model.reset_kv_cache();
        let mut tokenizer_dec = TokenOutputStream::new(self.outputtokenizer.clone());
        let mut logits_processor =
        candle_transformers::generation::LogitsProcessor::new(2666, Some(1.1), Some(0.5));

        let encoder_xs = {
            let mut tokens = self.inputtokenizer
                .encode(String::from(input), true)
                .map_err(E::msg)?
                .get_ids()
                .to_vec();
            tokens.push(self.config.eos_token_id);
            let tokens = Tensor::new(tokens.as_slice(), &device)?.unsqueeze(0)?;
            self.model.encoder().forward(&tokens, 0)?
        };
        let mut result =  String::from("");

        match self.config.num_beams {
            Some(num_beams) =>{
                let mut token_ids = vec![self.config.decoder_start_token_id];
                let mut beams = vec![(token_ids.clone(), 0.0)];
                for length in 0..1000 {
                    let mut new_beams = Vec::new();
                    for beam in beams.iter() {
                        // Get current sequence and score
                        let (seq, score) = beam;

                        // Check if sequence is complete (e.g., contains EOS)
                        if seq.last() == Some(&self.config.eos_token_id) || seq.last()  == Some(&self.config.eos_token_id) {
                            new_beams.push(beam.clone());
                            continue;
                        }

                        // Get model prediction
                        // self.model.decoder().reset_kv_cache();
                        // let mut logits:Tensor = Tensor::new(&seq[0..0], &device)?.unsqueeze(0)?;
                        // // to do: optimize
                        // for seq_token_index in 0..seq.len(){
                        //     let input_ids = Tensor::new(&seq[seq_token_index..(seq_token_index+1)], &device)?.unsqueeze(0)?;
                        //     logits = self.model.decode(&input_ids, &encoder_xs, seq_token_index)?;

                        // }
                        self.model.decoder().reset_kv_cache();
                        let context_size = seq.len();
                        let start_pos = seq.len().saturating_sub(context_size);
                        let input_ids = Tensor::new(&seq[start_pos..], &device)?.unsqueeze(0)?;
                        let mut logits = self.model.decode(&input_ids, &encoder_xs, start_pos)?;
                        logits = logits.squeeze(0)?;
                        logits = logits.get(logits.dim(0)? - 1)?;
                        let tokens_prob = logits_processor.sample_beam(&logits, num_beams)?;

                        // Create new beams for each candidate word
                        for (prob, idx) in tokens_prob {
                            let mut new_seq = seq.clone();
                            new_seq.push(idx);
                            let new_score = score + prob.log(std::f32::consts::E);
                            new_beams.push((new_seq, new_score));
                        }
                        
                    }

                    // Keep top K beams with highest total scores

                    //let mut argsort_indices = (0..new_beams.len()).collect::<Vec<_>>();
                    new_beams.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
                    beams = new_beams.into_iter().take(num_beams as usize).collect();
                    match predictionStringCallback{
                        Some(callback) =>{
                            let re = tokenizer_dec.decode( beams[0].0.as_slice())?;
                            let c_string = CString::new(re).expect("CString::new failed");
                            callback(c_string.as_ptr());
                        },
                        None => {},
                    }
                    //println!("{:?} ", beams[0]);
                    //let resulttmp = decode_output(&(self.inputtokenizer), &(self.outputtokenizer),beams[0].0.as_slice())?;
                    let resulttmp = tokenizer_dec.decode( beams[0].0.as_slice())?;
                    println!("{resulttmp}");
                }
                
                // Select sequence with highest score
                let best_beam = beams.iter().max_by(|a, b| a.1.partial_cmp(&b.1).unwrap()).unwrap();
                result = tokenizer_dec.decode( best_beam.0.as_slice())?;
                print!("{result}");
            }
            None => {
                let mut token_ids = vec![self.config.decoder_start_token_id];
                for index in 0..1000 {
                    let context_size = if index >= 1 { 1 } else { token_ids.len() };
                    let start_pos = token_ids.len().saturating_sub(context_size);
                    let input_ids = Tensor::new(&token_ids[start_pos..], &device)?.unsqueeze(0)?;
                    let logits = self.model.decode(&input_ids, &encoder_xs, start_pos)?;
                    let logits = logits.squeeze(0)?;
                    let logits = logits.get(logits.dim(0)? - 1)?;
                    let token = logits_processor.sample(&logits)?;
                    token_ids.push(token);
                    if let Some(t) = tokenizer_dec.next_token(token)? {
                        use std::io::Write;
                        print!("{t}");
                        result = result + &t;
                        match predictionStringCallback{
                            Some(callback) =>{
                                let c_string = CString::new(result).expect("CString::new failed");
                                callback(c_string.as_ptr());
                                result = c_string.into_string()?;
                            },
                            None => {},
                        }
                        std::io::stdout().flush()?;
                    }

                    if token == self.config.eos_token_id || token == self.config.forced_eos_token_id {
                        break;
                    }
                }
                if let Some(rest) = tokenizer_dec.decode_rest().map_err(E::msg)? {
                    print!("{rest}");
                    result = result + &rest;
                    match predictionStringCallback{
                        Some(callback) =>{
                            let c_string = CString::new(result).expect("CString::new failed");
                            callback(c_string.as_ptr());
                            result = c_string.into_string()?;
                        },
                        None => {},
                    }
                }
            }
        }

        println!();
        Ok(result)
    }
}

//std::mem::forget(model_box);

#[no_mangle]
pub extern "C" fn iosmt_model_new(path: *const c_char, gpu: bool) -> *mut IOSMTModel {
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
    match IOSMTModel::new(&path_str, &device) {
        Ok(model) => Box::into_raw(Box::new(model)),
        Err(e) => { 
            eprintln!("{}", e);
            std::ptr::null_mut()
        },
    }
}

#[no_mangle]
pub extern "C" fn iosmt_model_inference_new(ptr: *mut IOSMTModel, input: *const c_char, predictionStringCallback: extern "C" fn(*const c_char)) -> *mut c_char {
    if ptr.is_null() {
        eprintln!("Error: iosmt_model_inference_new null");
        return std::ptr::null_mut();
    }
    let input_str = unsafe { CStr::from_ptr(input).to_string_lossy().into_owned() };
    // 把原始指针转换回 Box，这将确保资源被正确释放
    let mut model_box = unsafe { Box::from_raw(ptr) };
    
    let result_ptr = match model_box.inference(input_str,Some(predictionStringCallback)){
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
pub extern "C" fn iosmt_model_inference(ptr: *mut IOSMTModel, input: *const c_char) -> *mut c_char {
    if ptr.is_null() {
        eprintln!("Error: iosmt_model_inference null");
        return std::ptr::null_mut();
    }
    let input_str = unsafe { CStr::from_ptr(input).to_string_lossy().into_owned() };
    // 把原始指针转换回 Box，这将确保资源被正确释放
    let mut model_box = unsafe { Box::from_raw(ptr) };
    
    let result_ptr = match model_box.inference(input_str,None){
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
pub extern "C" fn iosmt_model_free(ptr: *mut IOSMTModel) {
    if ptr.is_null() {
        // 处理错误或提前返回
        eprintln!("Error: iosmt_model_free null");
        return;
    }
    // 把原始指针转换回 Box，这将确保资源被正确释放
    unsafe { Box::from_raw(ptr) };
}

#[no_mangle]
pub extern fn rust_greeting(to: *const c_char) -> *mut c_char {
    let c_str = unsafe { CStr::from_ptr(to) };
    let recipient = match c_str.to_str() {
        Err(_) => "there",
        Ok(string) => string,
    };

    CString::new("Hello ".to_owned() + recipient).unwrap().into_raw()
}

#[no_mangle]
pub extern fn inference_free(s: *mut c_char) {
    unsafe {
        if s.is_null() { return }
        let _ = CString::from_raw(s);
    };
}

pub fn safe_load_model_inference(path: &str,input: &str) {
    let c_path = CString::new(path).expect("CString::new failed");
    let c_input = CString::new(input).expect("CString::new failed");
    let model = unsafe { iosmt_model_new(c_path.as_ptr(), true) };
    let result = iosmt_model_inference(model,c_input.as_ptr());
    inference_free(result);
    iosmt_model_free(model);

}

pub fn signal_test_load_model_inference(path: &str,input: &str) -> Result<String,E>{
    //let c_path = CString::new(path).expect("CString::new failed");
    //let model11 = unsafe { iosmt_model_new(c_path.as_ptr(), true) };
    let device = Device::new_metal(0)?;
    let device = Device::Cpu;
    let mut model =IOSMTModel::new(path,&device)?;
       
    model.model.reset_kv_cache();
    
    let mut tokenizer_dec = TokenOutputStream::new(model.outputtokenizer.clone());
    let mut logits_processor =
        candle_transformers::generation::LogitsProcessor::new(2666, Some(1.1), Some(0.5));
        
    let encoder_xs = {
            let mut tokens = model.inputtokenizer
                .encode(String::from(input), true)
                .map_err(E::msg)?
                .get_ids()
                .to_vec();
            tokens.push(model.config.eos_token_id);
            let tokens = Tensor::new(tokens.as_slice(), &device)?.unsqueeze(0)?;
            model.model.encoder().forward(&tokens, 0)?
        };
    let mut result: String =  String::from("");
    let seq: Vec<u32> = vec![65000, 21802, 5194]; 
    let input_ids = Tensor::new(&seq[0..], &device)?.unsqueeze(0)?;
    let seq2: Vec<u32> = vec![65000, 21802, 2222]; 
    let input_ids2 = Tensor::new(&seq2[0..], &device)?.unsqueeze(0)?;
    
    let cc: Tensor = Tensor::cat(&[&input_ids, &input_ids2,&input_ids, &input_ids2], 0)?.contiguous()?;
    
    let encoder_xs_cc = Tensor::cat(&[&encoder_xs, &encoder_xs,&encoder_xs, &encoder_xs], 0)?.contiguous()?;
    
    let mut logits:Tensor = Tensor::new(&seq[0..1], &device)?.unsqueeze(0)?;
    // // to do: optimize
    // for seq_token_index in 0..seq.len(){
    //     let input_ids = Tensor::new(&seq[seq_token_index..(seq_token_index+1)], &device)?.unsqueeze(0)?;
    //     logits = model.model.decode(&input_ids, &encoder_xs, seq_token_index)?;

    // }
    // logits = logits.get(logits.dim(0)? - 1)?.clone();
    // let logits = logits.to_dtype(DType::F32)?;
    // let prs = candle_nn::ops::softmax_last_dim(&logits)?.to_string();

    let start = Instant::now();

    // 这里放置您要测量执行时间的代码
    // 例如：some_function();
    for times in 0..500{
        model.model.decoder().reset_kv_cache();

        let context_size = seq.len();
        let start_pos = seq.len().saturating_sub(context_size);
        //let input_ids = Tensor::new(&cc[start_pos..], &device)?.unsqueeze(0)?;
        logits = model.model.decode(&input_ids, &encoder_xs, start_pos)?;
    
        logits = logits.squeeze(0)?;
    }

    // 获取当前时间，并与开始时间相减得到经过的时间
    let elapsed = start.elapsed();

    // 打印出所用的时间
    println!("Elapsed time: {:.2?}", elapsed);


    logits = logits.get(logits.dim(0)? - 1)?;
    let tokens_prob = logits_processor.sample_beam(&logits, 6)?;
    Ok(String::from(""))
}