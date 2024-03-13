use candle::{DType, Error, Result, Tensor};
use rand::{distributions::Distribution, SeedableRng};

pub struct LogitsProcessor {
    rng: rand::rngs::StdRng,
    temperature: Option<f64>,
    top_p: Option<f64>,
}

impl LogitsProcessor {
    pub fn new(seed: u64, temperature: Option<f64>, top_p: Option<f64>) -> Self {
        let temperature = if temperature.map_or(true, |v| v < 1e-7) {
            None
        } else {
            temperature
        };
        Self {
            rng: rand::rngs::StdRng::seed_from_u64(seed),
            temperature,
            top_p,
        }
    }

    fn sample_argmax(&mut self, logits: Tensor) -> Result<u32> {
        let logits_v: Vec<f32> = logits.to_vec1()?;
        let next_token = logits_v
            .iter()
            .enumerate()
            .max_by(|(_, u), (_, v)| u.total_cmp(v))
            .map(|(i, _)| i as u32)
            .unwrap();
        Ok(next_token)
    }

    fn sample_multinomial(&mut self, prs: &Vec<f32>) -> Result<u32> {
        let distr = rand::distributions::WeightedIndex::new(prs).map_err(Error::wrap)?;
        let next_token = distr.sample(&mut self.rng) as u32;
        Ok(next_token)
    }

    fn sample_topp(&mut self, prs: &mut Vec<f32>, top_p: f32) -> Result<u32> {
        // top-p sampling (or "nucleus sampling") samples from the smallest set of
        // tokens that exceed probability top_p. This way we never sample tokens that
        // have very low probabilities and are less likely to go "off the rails".
        let mut argsort_indices = (0..prs.len()).collect::<Vec<_>>();

        // Sort by descending probability.
        argsort_indices.sort_by(|&i, &j| prs[j].partial_cmp(&prs[i]).unwrap());

        // Clamp smaller probabilities to zero.
        let mut cumsum = 0.;
        for index in &argsort_indices {
            if cumsum >= top_p {
                prs[*index] = 0.0;
            } else {
                cumsum += prs[*index];
            }
        }
        // Sample with clamped probabilities.
        self.sample_multinomial(prs)
    }

    pub fn sample_beam(&mut self, logits: &Tensor, beams:usize) -> Result<Vec<(f32, u32)>> {
        let logits = logits.to_dtype(DType::F32)?;
        let prs = candle_nn::ops::softmax_last_dim(&logits)?;
        let prs: Vec<f32> = prs.to_vec1()?;
        let mut argsort_indices = (0..prs.len()).collect::<Vec<_>>();

        // Sort by descending probability.
        argsort_indices.sort_by(|&i, &j| prs[j].partial_cmp(&prs[i]).unwrap());
        let mut vec_f32: Vec<f32> = vec![]; // 示例数据
        let mut vec_u32: Vec<u32> = vec![];       // 示例数据
        if argsort_indices.len() < beams{
            return Err(Error::Msg(format!("beams over output")));
        }
        // Clamp smaller probabilities to zero.
        for i in 0..beams {
            let index = argsort_indices[i];
            vec_u32.push(index as u32);
            vec_f32.push(prs[index]);

        }

        Ok(vec_f32.into_iter().zip(vec_u32.into_iter()).collect::<Vec<(f32, u32)>>())
    }

    pub fn sample_new_beam(&mut self, logits: &Tensor, beams: Vec<(Vec<u32>, f32)>) -> Result<Vec<(Vec<u32>, f32, usize)>> {
        let logits = logits.to_dtype(DType::F32)?;
        let cur_len = beams[0].0.len();
        let prs = candle_nn::ops::softmax_last_dim(&logits)?;
        let mut prs: Vec<Vec<f32>> = prs.to_vec2()?;
        let target_beam_size = beams.len();
        let mut new_beams = Vec::with_capacity(target_beam_size * target_beam_size);
        if(cur_len == 1){
            prs = vec![prs[0].clone()];
        }
        for (pos, individual_beam_prob) in prs.iter().enumerate() {
            let mut argsort_indices = (0..individual_beam_prob.len()).collect::<Vec<_>>();
            argsort_indices.sort_by(|&i, &j| individual_beam_prob[j].partial_cmp(&individual_beam_prob[i]).unwrap());
            for i in 0..target_beam_size {
                let index = argsort_indices[i];
                let mut new_seq = beams[pos].0.clone();
                new_seq.push(index as u32);
                let prob: f32 = individual_beam_prob[index];
                // 6 0.25
                // 30 0.118
                let new_score = beams[pos].1 + prob.log(std::f32::consts::E);
                new_beams.push((new_seq, new_score, pos));
    
            }
        }
        new_beams.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        Ok(new_beams.into_iter().take(target_beam_size as usize).collect())
    }

    pub fn sample(&mut self, logits: &Tensor) -> Result<u32> {
        let logits = logits.to_dtype(DType::F32)?;
        let next_token = match self.temperature {
            None => self.sample_argmax(logits)?,
            Some(temperature) => {
                let logits = &(&logits / temperature)?;
                let prs = candle_nn::ops::softmax_last_dim(logits)?;
                let mut prs: Vec<f32> = prs.to_vec1()?;
                let top_p = self.top_p.unwrap_or(1.);
                if top_p <= 0.0 || top_p >= 1.0 {
                    // simply sample from the predicted probability distribution
                    self.sample_multinomial(&prs)?
                } else {
                    // top-p (nucleus) sampling, clamping the least likely tokens to zero
                    self.sample_topp(&mut prs, top_p as f32)?
                }
            }
        };
        Ok(next_token)
    }
}
