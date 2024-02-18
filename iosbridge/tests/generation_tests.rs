
extern crate iosbridge;

use iosbridge::{safe_load_model_inference, signal_test_load_model_inference};
use iosbridge::iosmbart::IOSMBartModel;
use candle_transformers::models::marian::{MTModel, self};
use std::time::Instant;
use tokenizers::{Tokenizer, InputSequence};
use anyhow::{Error as E, Result};
use candle::{DType, Tensor, Device};
#[test]
fn test_load_model1() {
    let start = Instant::now();
    safe_load_model_inference("/Users/xigsun/Documents/repo/candle/candle-examples/examples/marian-mt/opus-mt-zh-en/","在上一篇文章中我们介绍了注意力机制—目前在深度学习中被广泛应用。注意力机制能够显著提高神经机器翻译任务的性能。本文将会看一看Transformer---加速训练注意力模型的方法。Transformers在很多特定任务上已经优于Google神经机器翻译模型了。不过其最大的优点在于它的并行化训练。Google云强烈建议使用TPU云提供的Transformer模型。我们赶紧撸起袖子拆开模型看一看内部究竟如何吧。");
    // 进行测试逻辑
    // 获取当前时间，并与开始时间相减得到经过的时间
    let elapsed = start.elapsed();

    // 打印出所用的时间
    println!("Elapsed time: {:.2?}", elapsed);
}


#[test]
fn test_load_model3() {
    let start = Instant::now();
    safe_load_model_inference("/Users/xigsun/Documents/repo/candle/candle-examples/examples/marian-mt/opus-mt-en-zh/","who are you");
    // 进行测试逻辑
    // 获取当前时间，并与开始时间相减得到经过的时间
    let elapsed = start.elapsed();

    // 打印出所用的时间
    println!("Elapsed time: {:.2?}", elapsed);
}


#[test]
fn test_load_model4() {
    let start = Instant::now();
    safe_load_model_inference("/Users/xigsun/Documents/repo/mt-language/opus-mt-de-en/","What's the weather today?");
    // 进行测试逻辑
    // 获取当前时间，并与开始时间相减得到经过的时间
    let elapsed = start.elapsed();

    // 打印出所用的时间
    println!("Elapsed time: {:.2?}", elapsed);
}

#[cfg(feature = "metal")]
#[test]
fn test_load_model2() {
    signal_test_load_model_inference("/Users/xigsun/Documents/repo/candle/candle-examples/examples/marian-mt/opus-mt-zh-en/","求真务实是中国共产党人的重要思想和工作方法。前不久举行的中央经济工作会议上，习近平总书记着眼于做好明年经济工作、巩固和增强经济回升向好态势，对抓落实提出了明确要求，强调“要求真务实抓落实”“坚决纠治形式主义、官僚主义”。");
    // 进行测试逻辑
}

#[test]
fn test_mbart_model() -> Result<()>{
    let device = Device::Cpu;
    let mut tokenizer_path = std::path::PathBuf::from("/Users/xigsun/Documents/repo/mbart-large-50-many-to-many-mmt/tokenizer-base.json");
    let inputtokenizer = Tokenizer::from_file(&tokenizer_path).map_err(E::msg)?;
    let input = "The model can translate directly between any pair of 50 languages. To translate into a target language, the target language id is forced as the first generated token. ";
    let mut tokens = inputtokenizer
                .encode(String::from(input), true)
                .map_err(E::msg)?
                .get_ids()
                .to_vec();
    let mut model = IOSMBartModel::new("/Users/xigsun/Documents/repo/mbart-large-50-many-to-many-mmt/",&device)?;
    //let re = model.inference(String::from(input), None);
    return Ok(());
}