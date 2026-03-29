//! Tensor-parallel collective hooks for the CUDA model runner.
//!
//! The first implementation is intentionally a local no-op: it marks the
//! exact places where TP collectives must happen without forcing the engine
//! to pretend multi-rank math is already fully wired end-to-end.

use std::sync::Arc;

use cudarc::driver::CudaSlice;
use half::f16;
use tracing::debug;

use crate::bridge::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TensorParallelCollective {
    AllReduceF32,
    AllReduceF16,
    AllGatherF32,
}

pub trait TensorParallelComm: Send + Sync {
    fn rank(&self) -> usize;
    fn world_size(&self) -> usize;

    fn all_reduce_f32(
        &self,
        _tensor: &CudaSlice<f32>,
        _len: usize,
        _label: &'static str,
    ) -> Result<()> {
        Ok(())
    }

    fn all_reduce_f16(
        &self,
        _tensor: &CudaSlice<f16>,
        _len: usize,
        _label: &'static str,
    ) -> Result<()> {
        Ok(())
    }

    fn all_gather_f32(
        &self,
        tensor: &CudaSlice<f32>,
        _rows: usize,
        _cols_per_rank: usize,
        _label: &'static str,
    ) -> Result<CudaSlice<f32>> {
        Ok(tensor.clone())
    }
}

#[derive(Debug, Default)]
pub struct NoopTensorParallelComm {
    rank: usize,
    world_size: usize,
}

impl NoopTensorParallelComm {
    pub fn new(rank: usize, world_size: usize) -> Self {
        Self { rank, world_size }
    }

    fn trace_collective(
        &self,
        collective: TensorParallelCollective,
        label: &'static str,
        len: usize,
    ) {
        if self.world_size > 1 {
            debug!(
                rank = self.rank,
                world_size = self.world_size,
                ?collective,
                label,
                len,
                "tensor-parallel collective hook reached with local no-op implementation"
            );
        }
    }
}

impl TensorParallelComm for NoopTensorParallelComm {
    fn rank(&self) -> usize {
        self.rank
    }

    fn world_size(&self) -> usize {
        self.world_size
    }

    fn all_reduce_f32(
        &self,
        _tensor: &CudaSlice<f32>,
        len: usize,
        label: &'static str,
    ) -> Result<()> {
        self.trace_collective(TensorParallelCollective::AllReduceF32, label, len);
        Ok(())
    }

    fn all_reduce_f16(
        &self,
        _tensor: &CudaSlice<f16>,
        len: usize,
        label: &'static str,
    ) -> Result<()> {
        self.trace_collective(TensorParallelCollective::AllReduceF16, label, len);
        Ok(())
    }

    fn all_gather_f32(
        &self,
        tensor: &CudaSlice<f32>,
        rows: usize,
        cols_per_rank: usize,
        label: &'static str,
    ) -> Result<CudaSlice<f32>> {
        self.trace_collective(
            TensorParallelCollective::AllGatherF32,
            label,
            rows.saturating_mul(cols_per_rank),
        );
        Ok(tensor.clone())
    }
}

pub fn local_tensor_parallel_comm(rank: usize, world_size: usize) -> Arc<dyn TensorParallelComm> {
    Arc::new(NoopTensorParallelComm::new(rank, world_size))
}
