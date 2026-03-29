use half::f16;

use crate::bridge::{GpuBuffer, ModelWeights, Result};

pub(crate) fn get_or_zeros(weights: &ModelWeights, name: &str, shape: &[usize]) -> GpuBuffer<f16> {
    weights
        .get_as_buffer(name)
        .unwrap_or_else(|_| GpuBuffer::zeros(shape))
}

pub(crate) fn embed_tokens(
    embed: &GpuBuffer<f16>,
    token_ids: &[u32],
    hidden: usize,
) -> GpuBuffer<f16> {
    let mut out = Vec::with_capacity(token_ids.len() * hidden);
    for &tid in token_ids {
        let start = tid as usize * hidden;
        let end = start + hidden;
        if end <= embed.len() {
            out.extend_from_slice(&embed.data[start..end]);
        } else {
            out.extend(std::iter::repeat_n(f16::ZERO, hidden));
        }
    }
    GpuBuffer::from_vec(out, vec![token_ids.len(), hidden])
}

pub(crate) fn add_inplace(a: &mut GpuBuffer<f16>, b: &GpuBuffer<f16>) {
    for (x, y) in a.data.iter_mut().zip(b.data.iter()) {
        *x = f16::from_f32(x.to_f32() + y.to_f32());
    }
}

pub(crate) fn lm_head(
    hidden: &GpuBuffer<f16>,
    weight: &GpuBuffer<f16>,
    num_tokens: usize,
    vocab_size: usize,
) -> Result<GpuBuffer<f32>> {
    let h = hidden.len() / num_tokens;
    let mut logits = Vec::with_capacity(num_tokens * vocab_size);

    for t in 0..num_tokens {
        let row_start = t * h;
        for v in 0..vocab_size {
            let col_start = v * h;
            let mut acc = 0.0f32;
            for i in 0..h {
                acc += hidden.data[row_start + i].to_f32() * weight.data[col_start + i].to_f32();
            }
            logits.push(acc);
        }
    }

    Ok(GpuBuffer::from_vec(logits, vec![num_tokens, vocab_size]))
}
