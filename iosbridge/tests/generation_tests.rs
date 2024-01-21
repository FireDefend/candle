
extern crate iosbridge;

use iosbridge::{safe_load_model_inference, signal_test_load_model_inference};
use candle_transformers::models::marian::{MTModel, self};
use std::time::Instant;

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
fn test_load_model2() {
    signal_test_load_model_inference("/Users/xigsun/Documents/repo/candle/candle-examples/examples/marian-mt/opus-mt-zh-en/","求真务实是中国共产党人的重要思想和工作方法。前不久举行的中央经济工作会议上，习近平总书记着眼于做好明年经济工作、巩固和增强经济回升向好态势，对抓落实提出了明确要求，强调“要求真务实抓落实”“坚决纠治形式主义、官僚主义”。");
    // 进行测试逻辑
}