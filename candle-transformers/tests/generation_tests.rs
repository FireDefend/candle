use candle::{DType, Device, Result, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::generation::LogitsProcessor;
use candle_transformers::models::mbart::{self};
#[test]
fn sample_with_zero_temperature() -> Result<()> {
    let mut logits_process = LogitsProcessor::new(1337, None, None);
    let logits = Tensor::new(&[0.1, 0.2, 0.3, 0.4], &Device::Cpu)?;
    let token = logits_process.sample(&logits)?;
    assert_eq!(token, 3);
    Ok(())
}

#[test]
fn sample_with_temperature() -> Result<()> {
    let mut logits_process = LogitsProcessor::new(42, Some(0.9), None);
    let logits = Tensor::new(&[0.1, 0.2, 0.3, 0.4], &Device::Cpu)?;
    let token = logits_process.sample(&logits)?;
    assert_eq!(token, 0);
    Ok(())
}

#[test]
fn sample_with_top_p() -> Result<()> {
    let mut logits_process = LogitsProcessor::new(42, Some(1.0), Some(0.5));
    let logits = Tensor::new(&[0.1, 0.2, 0.3, 0.4], &Device::Cpu)?;
    let token = logits_process.sample(&logits)?;
    assert_eq!(token, 2);
    Ok(())
}

#[test]
fn mbart_test() -> Result<()> {
    let device = Device::Cpu;
    let model_path = std::path::PathBuf::from( "/Users/xigsun/Documents/repo/mbart-large-50-many-to-many-mmt/model.safetensors");
    let vb = { unsafe { VarBuilder::from_mmaped_safetensors(&[model_path], DType::F32, &device)? }
    };

    let config = mbart::Config::read_from_file("/Users/xigsun/Documents/repo/mbart-large-50-many-to-many-mmt/config.json");
    let model: mbart::MBartModel = mbart::MBartModel::new(&config, vb)?;
    Ok(())
}