use rvllm_core::error::{LLMError, Result};
use tracing::{debug, info};

use crate::weights::{GpuAllocator, ModelWeights, WeightTensor};

/// Loads only the shard belonging to a given rank from a set of model weights.
///
/// For tensor parallelism, certain weight matrices must be split along specific
/// dimensions. This loader takes fully-loaded weights and extracts the appropriate
/// slice for the given rank.
pub struct ShardedLoader;

impl ShardedLoader {
    /// Shard weights for the given rank in tensor-parallel group.
    ///
    /// - Column-parallel weights (q, k, v, gate, up) are split along dim 0
    /// - Row-parallel weights (o, down) are split along dim 1
    /// - Embeddings and norms are replicated (not sharded)
    pub fn shard(
        weights: ModelWeights,
        tp_size: usize,
        rank: usize,
        gpu: &dyn GpuAllocator,
    ) -> Result<ModelWeights> {
        if tp_size <= 1 {
            return Ok(weights);
        }

        if rank >= tp_size {
            return Err(LLMError::ConfigError(format!(
                "rank {} >= tensor_parallel_size {}",
                rank, tp_size
            )));
        }

        info!(
            "sharding {} weights for rank {}/{}",
            weights.num_weights(),
            rank,
            tp_size
        );

        let names: Vec<String> = weights.names().map(String::from).collect();
        let mut sharded = ModelWeights::new();

        for name in &names {
            let tensor = weights.get(name).unwrap();
            let shard_dim = classify_shard_dim(name, tensor.shape());

            match shard_dim {
                ShardDim::Along(dim) => {
                    let sharded_tensor = shard_along_dim(tensor, dim, tp_size, rank, gpu)?;
                    debug!(tensor = name.as_str(), dim, "tensor-parallel shard");
                    sharded.insert(sharded_tensor);
                }
                ShardDim::Replicate => sharded.insert(tensor.clone()),
            }
        }

        info!(
            "sharded {} weights for rank {}",
            sharded.num_weights(),
            rank
        );
        Ok(sharded)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ShardDim {
    Along(usize),
    Replicate,
}

/// Classify how a weight should be sharded based on its name.
fn classify_shard_dim(name: &str, shape: &[usize]) -> ShardDim {
    // GPT-OSS expert tensors are 3D: [experts, out, in] or [experts, hidden, intermediate].
    // Shard the projection dimension and keep expert routing replicated.
    if name.contains("mlp.experts.gate_up_proj") {
        return match shape.len() {
            0 | 1 => ShardDim::Replicate,
            2 => ShardDim::Along(1),
            _ => ShardDim::Along(1),
        };
    }
    if name.contains("mlp.experts.down_proj") {
        return match shape.len() {
            0 | 1 => ShardDim::Replicate,
            2 => ShardDim::Replicate,
            _ => ShardDim::Along(2),
        };
    }
    if name.contains("mlp.router.") {
        return ShardDim::Replicate;
    }

    // Column-parallel: output dimension split (q, k, v, gate, up projections)
    if name.contains("attn.q")
        || name.contains("attn.k")
        || name.contains("attn.v")
        || name.contains("q_proj")
        || name.contains("k_proj")
        || name.contains("v_proj")
        || name.contains("gate_proj")
        || name.contains("up_proj")
        || name.contains("ffn.gate")
        || name.contains("ffn.up")
        || name.contains("c_attn")
        || name.contains("c_fc")
    {
        return ShardDim::Along(0);
    }

    // Row-parallel: input dimension split (o, down projections)
    if name.contains("attn.o")
        || name.contains("o_proj")
        || name.contains("down_proj")
        || name.contains("ffn.down")
        || name.contains("c_proj")
    {
        return if shape.len() >= 2 {
            ShardDim::Along(1)
        } else {
            ShardDim::Replicate
        };
    }

    // Everything else (embeddings, norms, biases) gets replicated
    ShardDim::Replicate
}

/// Split a tensor along the given dimension.
fn shard_along_dim(
    tensor: &WeightTensor,
    dim: usize,
    tp_size: usize,
    rank: usize,
    gpu: &dyn GpuAllocator,
) -> Result<WeightTensor> {
    let shape = tensor.shape();
    if dim >= shape.len() {
        return Err(LLMError::ModelError(format!(
            "cannot shard {} (shape {:?}) along dim {}",
            tensor.name(),
            shape,
            dim
        )));
    }

    let dim_size = shape[dim];
    if !dim_size.is_multiple_of(tp_size) {
        return Err(LLMError::ModelError(format!(
            "tensor {} dim {} size {} not divisible by tp_size {}",
            tensor.name(),
            dim,
            dim_size,
            tp_size
        )));
    }

    let shard_size = dim_size / tp_size;
    let mut new_shape = shape.to_vec();
    new_shape[dim] = shard_size;

    let elem_size = tensor.dtype().size_of();
    let data = tensor.data().as_bytes();

    let outer_size = shape[..dim].iter().product::<usize>();
    let inner_size = shape[dim + 1..].iter().product::<usize>();
    let prefix_bytes = dim_size * inner_size * elem_size;
    let shard_bytes = shard_size * inner_size * elem_size;
    let shard_offset_bytes = rank * shard_bytes;

    let shard_data = if shape.len() == 1 {
        let start = shard_offset_bytes;
        let end = start + shard_bytes;
        data[start..end].to_vec()
    } else {
        let mut out = Vec::with_capacity(outer_size * shard_bytes);
        for outer_idx in 0..outer_size {
            let base = outer_idx * prefix_bytes;
            let start = base + shard_offset_bytes;
            let end = start + shard_bytes;
            out.extend_from_slice(&data[start..end]);
        }
        out
    };

    let gpu_buf = gpu.upload(&shard_data)?;
    Ok(WeightTensor::new(
        tensor.name().to_string(),
        new_shape,
        tensor.dtype(),
        gpu_buf,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dtype::DType;
    use crate::weights::{GpuBuffer, MockGpuAllocator};

    fn make_tensor(name: &str, shape: Vec<usize>, dtype: DType, data: Vec<u8>) -> WeightTensor {
        WeightTensor::new(name.into(), shape, dtype, GpuBuffer::from_bytes(data))
    }

    #[test]
    fn tp1_passthrough() {
        let mut weights = ModelWeights::new();
        weights.insert(make_tensor("w", vec![4], DType::F32, vec![0u8; 16]));

        let gpu = MockGpuAllocator;
        let result = ShardedLoader::shard(weights, 1, 0, &gpu).unwrap();
        assert_eq!(result.num_weights(), 1);
    }

    #[test]
    fn rank_out_of_bounds() {
        let weights = ModelWeights::new();
        let gpu = MockGpuAllocator;
        let result = ShardedLoader::shard(weights, 2, 3, &gpu);
        assert!(result.is_err());
    }

    #[test]
    fn column_parallel_shard() {
        // 4x4 F32 matrix = 64 bytes, split along dim 0 into 2 shards
        let data: Vec<u8> = (0..64).collect();
        let mut weights = ModelWeights::new();
        weights.insert(make_tensor("attn.q.weight", vec![4, 4], DType::U8, data));

        let gpu = MockGpuAllocator;
        let rank0 = ShardedLoader::shard(weights, 2, 0, &gpu).unwrap();
        let w = rank0.get("attn.q.weight").unwrap();
        assert_eq!(w.shape(), &[2, 4]);
        assert_eq!(w.size_bytes(), 8);
        // First 8 bytes = rows 0..2
        assert_eq!(w.data().as_bytes(), &(0u8..8).collect::<Vec<u8>>());
    }

    #[test]
    fn row_parallel_shard() {
        // 4x4 U8 matrix, split along dim 1 into 2 shards
        let data: Vec<u8> = (0..16).collect();
        let mut weights = ModelWeights::new();
        weights.insert(make_tensor("attn.o.weight", vec![4, 4], DType::U8, data));

        let gpu = MockGpuAllocator;
        let rank1 = ShardedLoader::shard(weights, 2, 1, &gpu).unwrap();
        let w = rank1.get("attn.o.weight").unwrap();
        assert_eq!(w.shape(), &[4, 2]);
        // Each row: take columns [2..4]
        let expected: Vec<u8> = vec![2, 3, 6, 7, 10, 11, 14, 15];
        assert_eq!(w.data().as_bytes(), &expected);
    }

    #[test]
    fn replicated_weights() {
        let data = vec![1u8; 8];
        let mut weights = ModelWeights::new();
        weights.insert(make_tensor(
            "embed_tokens.weight",
            vec![4, 2],
            DType::U8,
            data.clone(),
        ));
        weights.insert(make_tensor("norm.weight", vec![8], DType::U8, vec![2u8; 8]));

        let gpu = MockGpuAllocator;
        let result = ShardedLoader::shard(weights, 2, 0, &gpu).unwrap();
        // Both should be replicated (unchanged)
        assert_eq!(result.get("embed_tokens.weight").unwrap().shape(), &[4, 2]);
        assert_eq!(result.get("norm.weight").unwrap().shape(), &[8]);
    }

    #[test]
    fn classify_names() {
        assert_eq!(
            classify_shard_dim("layers.0.attn.q.weight", &[4, 4]),
            ShardDim::Along(0)
        );
        assert_eq!(
            classify_shard_dim("layers.0.attn.o.weight", &[4, 4]),
            ShardDim::Along(1)
        );
        assert_eq!(
            classify_shard_dim("layers.0.ffn.gate.weight", &[4, 4]),
            ShardDim::Along(0)
        );
        assert_eq!(
            classify_shard_dim("layers.0.ffn.down.weight", &[4, 4]),
            ShardDim::Along(1)
        );
        assert_eq!(
            classify_shard_dim("model.layers.0.mlp.experts.gate_up_proj", &[8, 16, 4]),
            ShardDim::Along(1)
        );
        assert_eq!(
            classify_shard_dim("model.layers.0.mlp.experts.gate_up_proj_bias", &[8, 16]),
            ShardDim::Along(1)
        );
        assert_eq!(
            classify_shard_dim("model.layers.0.mlp.experts.down_proj", &[8, 4, 16]),
            ShardDim::Along(2)
        );
        assert_eq!(
            classify_shard_dim("model.layers.0.mlp.experts.down_proj_bias", &[8, 4]),
            ShardDim::Replicate
        );
        assert_eq!(
            classify_shard_dim("model.layers.0.mlp.router.weight", &[8, 4]),
            ShardDim::Replicate
        );
        assert_eq!(
            classify_shard_dim("embed_tokens.weight", &[4, 2]),
            ShardDim::Replicate
        );
        assert_eq!(classify_shard_dim("norm.weight", &[8]), ShardDim::Replicate);
    }

    #[test]
    fn shard_3d_gate_up_projection_along_output_dim() {
        let data: Vec<u8> = (0..32).collect();
        let mut weights = ModelWeights::new();
        weights.insert(make_tensor(
            "model.layers.0.mlp.experts.gate_up_proj",
            vec![2, 4, 4],
            DType::U8,
            data,
        ));

        let gpu = MockGpuAllocator;
        let rank1 = ShardedLoader::shard(weights, 2, 1, &gpu).unwrap();
        let w = rank1
            .get("model.layers.0.mlp.experts.gate_up_proj")
            .unwrap();
        assert_eq!(w.shape(), &[2, 2, 4]);
        assert_eq!(
            w.data().as_bytes(),
            &vec![8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31]
        );
    }

    #[test]
    fn shard_3d_down_projection_along_input_dim() {
        let data: Vec<u8> = (0..32).collect();
        let mut weights = ModelWeights::new();
        weights.insert(make_tensor(
            "model.layers.0.mlp.experts.down_proj",
            vec![2, 4, 4],
            DType::U8,
            data,
        ));

        let gpu = MockGpuAllocator;
        let rank0 = ShardedLoader::shard(weights, 2, 0, &gpu).unwrap();
        let w = rank0.get("model.layers.0.mlp.experts.down_proj").unwrap();
        assert_eq!(w.shape(), &[2, 4, 2]);
        assert_eq!(
            w.data().as_bytes(),
            &vec![0, 1, 4, 5, 8, 9, 12, 13, 16, 17, 20, 21, 24, 25, 28, 29]
        );
    }
}
