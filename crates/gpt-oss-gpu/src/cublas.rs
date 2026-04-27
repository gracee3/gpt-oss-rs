//! cuBLAS GEMM operations for linear algebra.

use cudarc::cublas::sys::cublasOperation_t;
use cudarc::cublas::{CudaBlas, Gemm as _, GemmConfig, Gemv as _, GemvConfig};
use cudarc::driver::{CudaSlice, CudaStream, DevicePtr, DevicePtrMut};
use half::bf16;
use serde::{Deserialize, Serialize};
use std::mem::MaybeUninit;
use std::sync::Arc;

use crate::Result;

/// Default cuBLAS workspace size for graph capture (4 MiB).
/// NVIDIA recommends at least 4 KiB; 4 MiB covers all GEMM tile configs.
const CUBLAS_GRAPH_WORKSPACE_BYTES: usize = 4 * 1024 * 1024;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CublasHandleState {
    pub pointer_mode: String,
    pub math_mode: String,
    pub atomics_mode: String,
    pub graph_workspace_registered: bool,
    pub graph_workspace_bytes: usize,
}

/// Wrapper around cuBLAS for matrix operations.
pub struct CublasHandle {
    blas: CudaBlas,
    stream: Arc<CudaStream>,
    /// Pre-allocated workspace buffer for CUDA graph capture.
    /// cuBLAS requires an explicit workspace via `cublasSetWorkspace_v2`
    /// before any GEMM call inside a graph capture region, otherwise it
    /// tries to allocate internally with `cudaMalloc` which is forbidden.
    graph_workspace: Option<CudaSlice<u8>>,
}

impl CublasHandle {
    pub fn new(stream: Arc<CudaStream>) -> Result<Self> {
        let blas = CudaBlas::new(stream.clone())
            .map_err(|e| crate::LLMError::GpuError(format!("cuBLAS init failed: {e}")))?;
        Ok(Self {
            blas,
            stream,
            graph_workspace: None,
        })
    }

    /// Returns a reference to the underlying stream.
    pub fn stream(&self) -> &Arc<CudaStream> {
        &self.stream
    }

    #[cfg(feature = "cuda")]
    pub fn snapshot_state(&self) -> Result<CublasHandleState> {
        use cudarc::cublas::sys::{
            cublasAtomicsMode_t, cublasMath_t, cublasStatus_t::CUBLAS_STATUS_SUCCESS,
        };

        let pointer_mode = self
            .blas
            .get_pointer_mode()
            .map_err(|e| crate::LLMError::GpuError(format!("cublas pointer mode query failed: {e}")))?;

        let mut math_mode = MaybeUninit::<cublasMath_t>::uninit();
        let mut atomics_mode = MaybeUninit::<cublasAtomicsMode_t>::uninit();
        unsafe {
            let math_status = cudarc::cublas::sys::cublasGetMathMode(
                *self.blas.handle(),
                math_mode.as_mut_ptr(),
            );
            if math_status != CUBLAS_STATUS_SUCCESS {
                return Err(crate::LLMError::GpuError(format!(
                    "cublas math mode query failed: {math_status:?}"
                )));
            }
            let atomics_status = cudarc::cublas::sys::cublasGetAtomicsMode(
                *self.blas.handle(),
                atomics_mode.as_mut_ptr(),
            );
            if atomics_status != CUBLAS_STATUS_SUCCESS {
                return Err(crate::LLMError::GpuError(format!(
                    "cublas atomics mode query failed: {atomics_status:?}"
                )));
            }
        }

        Ok(CublasHandleState {
            pointer_mode: format!("{pointer_mode:?}"),
            math_mode: format!("{:?}", unsafe { math_mode.assume_init() }),
            atomics_mode: format!("{:?}", unsafe { atomics_mode.assume_init() }),
            graph_workspace_registered: self.graph_workspace.is_some(),
            graph_workspace_bytes: self.graph_workspace.as_ref().map(|ws| ws.len()).unwrap_or(0),
        })
    }

    /// Pre-allocate and register a cuBLAS workspace for CUDA graph capture.
    ///
    /// Must be called BEFORE `cuStreamBeginCapture`. The workspace stays
    /// registered for the lifetime of this handle; subsequent captures
    /// reuse the same buffer.
    #[cfg(feature = "cuda")]
    pub fn prepare_for_graph_capture(&mut self) -> Result<()> {
        if self.graph_workspace.is_some() {
            return Ok(()); // already prepared
        }

        tracing::info!(
            bytes = CUBLAS_GRAPH_WORKSPACE_BYTES,
            "allocating cuBLAS graph workspace"
        );

        let mut ws = self
            .stream
            .alloc_zeros::<u8>(CUBLAS_GRAPH_WORKSPACE_BYTES)
            .map_err(|e| crate::LLMError::GpuError(format!("cuBLAS workspace alloc: {e}")))?;

        // Get raw device pointer and call cublasSetWorkspace_v2 in a scoped
        // borrow so we can move `ws` into self.graph_workspace afterwards.
        {
            let (raw_ptr, _guard) = DevicePtrMut::device_ptr_mut(&mut ws, &self.stream);
            unsafe {
                let status = cudarc::cublas::sys::cublasSetWorkspace_v2(
                    *self.blas.handle(),
                    raw_ptr as *mut std::ffi::c_void,
                    CUBLAS_GRAPH_WORKSPACE_BYTES,
                );
                if status != cudarc::cublas::sys::cublasStatus_t::CUBLAS_STATUS_SUCCESS {
                    return Err(crate::LLMError::GpuError(format!(
                        "cublasSetWorkspace_v2 failed: {status:?}"
                    )));
                }
            }
        }

        self.graph_workspace = Some(ws);
        tracing::info!("cuBLAS graph workspace registered");
        Ok(())
    }

    /// No-op when cuda feature is off.
    #[cfg(not(feature = "cuda"))]
    pub fn prepare_for_graph_capture(&mut self) -> Result<()> {
        Ok(())
    }

    /// Pre-warm cuBLAS algorithm cache for all GEMM shapes used in the model.
    /// Runs 3 dummy GEMMs per shape to trigger cuBLAS's internal algo selection.
    #[cfg(feature = "cuda")]
    pub fn warmup_gemm_shapes(&self, shapes: &[(usize, usize, usize)]) -> Result<()> {
        use cudarc::cublas::sys::{
            cublasComputeType_t::CUBLAS_COMPUTE_32F,
            cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT_TENSOR_OP,
            cublasOperation_t::{CUBLAS_OP_N, CUBLAS_OP_T},
            cublasStatus_t::CUBLAS_STATUS_SUCCESS,
            cudaDataType_t::CUDA_R_16F,
        };

        // Find max element counts needed across all shapes so we allocate once.
        let max_a = shapes.iter().map(|&(m, _, k)| m * k).max().unwrap_or(0);
        let max_b = shapes.iter().map(|&(_, n, k)| n * k).max().unwrap_or(0);
        let max_c = shapes.iter().map(|&(m, n, _)| m * n).max().unwrap_or(0);

        if max_a == 0 {
            return Ok(());
        }

        let a_buf = self
            .stream
            .alloc_zeros::<half::f16>(max_a)
            .map_err(|e| crate::LLMError::GpuError(format!("warmup alloc A: {e}")))?;
        let b_buf = self
            .stream
            .alloc_zeros::<half::f16>(max_b)
            .map_err(|e| crate::LLMError::GpuError(format!("warmup alloc B: {e}")))?;
        let mut c_buf = self
            .stream
            .alloc_zeros::<half::f16>(max_c)
            .map_err(|e| crate::LLMError::GpuError(format!("warmup alloc C: {e}")))?;

        let alpha: f32 = 1.0;
        let beta: f32 = 0.0;

        let (a_ptr, _ag) = DevicePtr::device_ptr(&a_buf, &self.stream);
        let (b_ptr, _bg) = DevicePtr::device_ptr(&b_buf, &self.stream);
        let (c_ptr, _cg) = DevicePtrMut::device_ptr_mut(&mut c_buf, &self.stream);

        for &(m, n, k) in shapes {
            tracing::debug!(m, n, k, "warming up cuBLAS GEMM shape");
            for _ in 0..3 {
                unsafe {
                    let status = cudarc::cublas::sys::cublasGemmEx(
                        *self.blas.handle(),
                        CUBLAS_OP_T,
                        CUBLAS_OP_N,
                        n as i32,
                        m as i32,
                        k as i32,
                        &alpha as *const f32 as *const std::ffi::c_void,
                        b_ptr as *const std::ffi::c_void,
                        CUDA_R_16F,
                        k as i32,
                        a_ptr as *const std::ffi::c_void,
                        CUDA_R_16F,
                        k as i32,
                        &beta as *const f32 as *const std::ffi::c_void,
                        c_ptr as *mut std::ffi::c_void,
                        CUDA_R_16F,
                        n as i32,
                        CUBLAS_COMPUTE_32F,
                        CUBLAS_GEMM_DEFAULT_TENSOR_OP,
                    );
                    if status != CUBLAS_STATUS_SUCCESS {
                        return Err(crate::LLMError::GpuError(format!(
                            "warmup cublasGemmEx failed for ({m},{n},{k}): {status:?}"
                        )));
                    }
                }
            }
        }

        self.stream
            .synchronize()
            .map_err(|e| crate::LLMError::GpuError(format!("warmup sync: {e}")))?;

        tracing::info!(count = shapes.len(), "cuBLAS GEMM warmup complete");
        Ok(())
    }

    #[cfg(not(feature = "cuda"))]
    pub fn warmup_gemm_shapes(&self, _shapes: &[(usize, usize, usize)]) -> Result<()> {
        Ok(())
    }

    /// SGEMM: C[m,n] = A[m,k] @ B[n,k]^T
    ///
    /// A is activations in row-major [m, k].
    /// B is weights in PyTorch layout row-major [n, k].
    /// C is output row-major [m, n].
    pub fn sgemm(
        &self,
        m: usize,
        n: usize,
        k: usize,
        alpha: f32,
        a: &CudaSlice<f32>,
        b: &CudaSlice<f32>,
        beta: f32,
        c: &mut CudaSlice<f32>,
    ) -> Result<()> {
        // b row[n,k] = col[k,n], OP_T -> [n,k]. lda=k.
        // a row[m,k] = col[k,m], OP_N -> [k,m]. ldb=k.
        // C_col[n,m] = row C[m,n]. ldc=n.
        unsafe {
            self.blas
                .gemm(
                    GemmConfig {
                        transa: cublasOperation_t::CUBLAS_OP_T,
                        transb: cublasOperation_t::CUBLAS_OP_N,
                        m: n as i32,
                        n: m as i32,
                        k: k as i32,
                        alpha,
                        lda: k as i32,
                        ldb: k as i32,
                        beta,
                        ldc: n as i32,
                    },
                    b,
                    a,
                    c,
                )
                .map_err(|e| crate::LLMError::GpuError(format!("cuBLAS sgemm failed: {e}")))?;
        }
        Ok(())
    }

    /// HGEMM with f32 output: C[m,n] = A[m,k] @ B[n,k]^T
    ///
    /// A is f16 activations in row-major [m, k].
    /// B is f16 weights in PyTorch layout row-major [n, k].
    /// C is f32 output row-major [m, n].
    ///
    /// Uses `cublasGemmEx` with A/B as f16 and C as f32. This eliminates
    /// the output f16->f32 cast kernel (caller still casts input f32->f16,
    /// but saves one cast + one alloc per linear vs the old hgemm path).
    /// Compute in f32 with tensor-op auto-selection.
    #[cfg(feature = "cuda")]
    pub fn hgemm_f32_output(
        &self,
        m: usize,
        n: usize,
        k: usize,
        alpha: f32,
        a: &CudaSlice<half::f16>,
        b: &CudaSlice<half::f16>,
        beta: f32,
        c: &mut CudaSlice<f32>,
    ) -> Result<()> {
        use cudarc::cublas::sys::{
            cublasComputeType_t::CUBLAS_COMPUTE_32F,
            cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT_TENSOR_OP,
            cublasOperation_t::{CUBLAS_OP_N, CUBLAS_OP_T},
            cublasStatus_t::CUBLAS_STATUS_SUCCESS,
            cudaDataType_t::{CUDA_R_16F, CUDA_R_32F},
        };

        // Same row-major -> col-major mapping as sgemm/hgemm:
        // b row[n,k] = col[k,n], OP_T -> [n,k]. lda=k. (A = f16)
        // a row[m,k] = col[k,m], OP_N -> [k,m]. ldb=k. (B = f16)
        // C_col[n,m] = row C[m,n]. ldc=n. (C = f32)
        let (b_ptr, _b_guard) = DevicePtr::device_ptr(b, &self.stream);
        let (a_ptr, _a_guard) = DevicePtr::device_ptr(a, &self.stream);
        let (c_ptr, _c_guard) = DevicePtrMut::device_ptr_mut(c, &self.stream);

        unsafe {
            let status = cudarc::cublas::sys::cublasGemmEx(
                *self.blas.handle(),
                CUBLAS_OP_T,
                CUBLAS_OP_N,
                n as i32,
                m as i32,
                k as i32,
                &alpha as *const f32 as *const std::ffi::c_void,
                b_ptr as *const std::ffi::c_void,
                CUDA_R_16F,
                k as i32,
                a_ptr as *const std::ffi::c_void,
                CUDA_R_16F,
                k as i32,
                &beta as *const f32 as *const std::ffi::c_void,
                c_ptr as *mut std::ffi::c_void,
                CUDA_R_32F,
                n as i32,
                CUBLAS_COMPUTE_32F,
                CUBLAS_GEMM_DEFAULT_TENSOR_OP,
            );
            if status != CUBLAS_STATUS_SUCCESS {
                return Err(crate::LLMError::GpuError(format!(
                    "cublasGemmEx (f16xf16->f32) failed: {status:?}"
                )));
            }
        }
        Ok(())
    }

    /// No-op stub when cuda feature is off.
    #[cfg(not(feature = "cuda"))]
    pub fn hgemm_f32_output(
        &self,
        _m: usize,
        _n: usize,
        _k: usize,
        _alpha: f32,
        _a: &CudaSlice<half::f16>,
        _b: &CudaSlice<half::f16>,
        _beta: f32,
        _c: &mut CudaSlice<f32>,
    ) -> Result<()> {
        Ok(())
    }

    /// HGEMM: half-precision GEMM for f16.
    ///
    /// Same layout conventions as [`sgemm`](Self::sgemm) but operates on f16
    /// tensors. Internally uses f32 accumulation for numerical stability
    /// (matching cuBLAS CUBLAS_COMPUTE_32F behavior on Ampere+).
    ///
    /// This halves memory bandwidth for weight-bound operations (all linear
    /// projections in the transformer), which is the primary bottleneck for
    /// inference at moderate batch sizes.
    pub fn hgemm(
        &self,
        m: usize,
        n: usize,
        k: usize,
        alpha: half::f16,
        a: &CudaSlice<half::f16>,
        b: &CudaSlice<half::f16>,
        beta: half::f16,
        c: &mut CudaSlice<half::f16>,
    ) -> Result<()> {
        // Same mapping as sgemm: C[m,n] = A[m,k] @ B[n,k]^T
        unsafe {
            self.blas
                .gemm(
                    GemmConfig {
                        transa: cublasOperation_t::CUBLAS_OP_T,
                        transb: cublasOperation_t::CUBLAS_OP_N,
                        m: n as i32,
                        n: m as i32,
                        k: k as i32,
                        alpha,
                        lda: k as i32,
                        ldb: k as i32,
                        beta,
                        ldc: n as i32,
                    },
                    b,
                    a,
                    c,
                )
                .map_err(|e| crate::LLMError::GpuError(format!("cuBLAS hgemm failed: {e}")))?;
        }
        Ok(())
    }

    /// HGEMM: bfloat16 precision GEMM for bf16.
    ///
    /// Same layout conventions as [`hgemm`](Self::hgemm) but operates on bf16
    /// tensors. Internally uses f32 accumulation for numerical stability.
    #[cfg(feature = "cuda")]
    pub fn hgemm_bf16(
        &self,
        m: usize,
        n: usize,
        k: usize,
        alpha: bf16,
        a: &CudaSlice<bf16>,
        b: &CudaSlice<bf16>,
        beta: bf16,
        c: &mut CudaSlice<bf16>,
    ) -> Result<()> {
        use cudarc::cublas::sys::{
            cublasComputeType_t::CUBLAS_COMPUTE_32F,
            cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT_TENSOR_OP,
            cublasOperation_t::{CUBLAS_OP_N, CUBLAS_OP_T},
            cublasStatus_t::CUBLAS_STATUS_SUCCESS,
            cudaDataType_t::CUDA_R_16BF,
        };

        let alpha_f32 = alpha.to_f32();
        let beta_f32 = beta.to_f32();
        let (b_ptr, _bg) = DevicePtr::device_ptr(b, &self.stream);
        let (a_ptr, _ag) = DevicePtr::device_ptr(a, &self.stream);
        let (c_ptr, _cg) = DevicePtrMut::device_ptr_mut(c, &self.stream);

        unsafe {
            let status = cudarc::cublas::sys::cublasGemmEx(
                *self.blas.handle(),
                CUBLAS_OP_T,
                CUBLAS_OP_N,
                n as i32,
                m as i32,
                k as i32,
                &alpha_f32 as *const f32 as *const std::ffi::c_void,
                b_ptr as *const std::ffi::c_void,
                CUDA_R_16BF,
                k as i32,
                a_ptr as *const std::ffi::c_void,
                CUDA_R_16BF,
                k as i32,
                &beta_f32 as *const f32 as *const std::ffi::c_void,
                c_ptr as *mut std::ffi::c_void,
                CUDA_R_16BF,
                n as i32,
                CUBLAS_COMPUTE_32F,
                CUBLAS_GEMM_DEFAULT_TENSOR_OP,
            );
            if status != CUBLAS_STATUS_SUCCESS {
                return Err(crate::LLMError::GpuError(format!(
                    "cublasGemmEx (bf16xbf16->bf16) failed: {status:?}"
                )));
            }
        }
        Ok(())
    }

    /// BF16 GEMM using pedantic f32 accumulation and the non-tensor-op default algorithm.
    ///
    /// This preserves the same row-major logical contract as [`hgemm_bf16`](Self::hgemm_bf16)
    /// but avoids tensor-op math for narrow parity probes that require CPU-BF16 agreement.
    #[cfg(feature = "cuda")]
    pub fn hgemm_bf16_pedantic_no_tensor_op(
        &self,
        m: usize,
        n: usize,
        k: usize,
        alpha: bf16,
        a: &CudaSlice<bf16>,
        b: &CudaSlice<bf16>,
        beta: bf16,
        c: &mut CudaSlice<bf16>,
    ) -> Result<()> {
        use cudarc::cublas::sys::{
            cublasAtomicsMode_t::{self, CUBLAS_ATOMICS_NOT_ALLOWED},
            cublasComputeType_t::CUBLAS_COMPUTE_32F_PEDANTIC,
            cublasGemmAlgo_t::CUBLAS_GEMM_DFALT,
            cublasMath_t::{self, CUBLAS_PEDANTIC_MATH},
            cublasOperation_t::{CUBLAS_OP_N, CUBLAS_OP_T},
            cublasStatus_t::CUBLAS_STATUS_SUCCESS,
            cudaDataType_t::CUDA_R_16BF,
        };

        let alpha_f32 = alpha.to_f32();
        let beta_f32 = beta.to_f32();
        let (b_ptr, _bg) = DevicePtr::device_ptr(b, &self.stream);
        let (a_ptr, _ag) = DevicePtr::device_ptr(a, &self.stream);
        let (c_ptr, _cg) = DevicePtrMut::device_ptr_mut(c, &self.stream);

        unsafe {
            let mut previous_math_mode = MaybeUninit::<cublasMath_t>::uninit();
            let previous_math_status = cudarc::cublas::sys::cublasGetMathMode(
                *self.blas.handle(),
                previous_math_mode.as_mut_ptr(),
            );
            if previous_math_status != CUBLAS_STATUS_SUCCESS {
                return Err(crate::LLMError::GpuError(format!(
                    "cublasGetMathMode before pedantic GEMM failed: {previous_math_status:?}"
                )));
            }
            let previous_math_mode = previous_math_mode.assume_init();
            let mut previous_atomics_mode = MaybeUninit::<cublasAtomicsMode_t>::uninit();
            let previous_atomics_status = cudarc::cublas::sys::cublasGetAtomicsMode(
                *self.blas.handle(),
                previous_atomics_mode.as_mut_ptr(),
            );
            if previous_atomics_status != CUBLAS_STATUS_SUCCESS {
                return Err(crate::LLMError::GpuError(format!(
                    "cublasGetAtomicsMode before pedantic GEMM failed: {previous_atomics_status:?}"
                )));
            }
            let previous_atomics_mode = previous_atomics_mode.assume_init();

            let math_status = cudarc::cublas::sys::cublasSetMathMode(
                *self.blas.handle(),
                CUBLAS_PEDANTIC_MATH,
            );
            if math_status != CUBLAS_STATUS_SUCCESS {
                return Err(crate::LLMError::GpuError(format!(
                    "cublasSetMathMode(CUBLAS_PEDANTIC_MATH) failed: {math_status:?}"
                )));
            }
            let atomics_status = cudarc::cublas::sys::cublasSetAtomicsMode(
                *self.blas.handle(),
                CUBLAS_ATOMICS_NOT_ALLOWED,
            );
            if atomics_status != CUBLAS_STATUS_SUCCESS {
                let _ = cudarc::cublas::sys::cublasSetMathMode(
                    *self.blas.handle(),
                    previous_math_mode,
                );
                return Err(crate::LLMError::GpuError(format!(
                    "cublasSetAtomicsMode(CUBLAS_ATOMICS_NOT_ALLOWED) failed: {atomics_status:?}"
                )));
            }

            let status = cudarc::cublas::sys::cublasGemmEx(
                *self.blas.handle(),
                CUBLAS_OP_T,
                CUBLAS_OP_N,
                n as i32,
                m as i32,
                k as i32,
                &alpha_f32 as *const f32 as *const std::ffi::c_void,
                b_ptr as *const std::ffi::c_void,
                CUDA_R_16BF,
                k as i32,
                a_ptr as *const std::ffi::c_void,
                CUDA_R_16BF,
                k as i32,
                &beta_f32 as *const f32 as *const std::ffi::c_void,
                c_ptr as *mut std::ffi::c_void,
                CUDA_R_16BF,
                n as i32,
                CUBLAS_COMPUTE_32F_PEDANTIC,
                CUBLAS_GEMM_DFALT,
            );

            let restore_atomics_status = cudarc::cublas::sys::cublasSetAtomicsMode(
                *self.blas.handle(),
                previous_atomics_mode,
            );
            let restore_math_status = cudarc::cublas::sys::cublasSetMathMode(
                *self.blas.handle(),
                previous_math_mode,
            );
            if status != CUBLAS_STATUS_SUCCESS {
                return Err(crate::LLMError::GpuError(format!(
                    "cublasGemmEx pedantic bf16xbf16->bf16 failed: {status:?}"
                )));
            }
            if restore_atomics_status != CUBLAS_STATUS_SUCCESS {
                return Err(crate::LLMError::GpuError(format!(
                    "restore cublas atomics mode failed: {restore_atomics_status:?}"
                )));
            }
            if restore_math_status != CUBLAS_STATUS_SUCCESS {
                return Err(crate::LLMError::GpuError(format!(
                    "restore cublas math mode failed: {restore_math_status:?}"
                )));
            }
        }
        Ok(())
    }

    /// No-op stub when cuda feature is off.
    #[cfg(not(feature = "cuda"))]
    pub fn hgemm_bf16(
        &self,
        _m: usize,
        _n: usize,
        _k: usize,
        _alpha: bf16,
        _a: &CudaSlice<bf16>,
        _b: &CudaSlice<bf16>,
        _beta: bf16,
        _c: &mut CudaSlice<bf16>,
    ) -> Result<()> {
        Ok(())
    }

    #[cfg(not(feature = "cuda"))]
    pub fn hgemm_bf16_pedantic_no_tensor_op(
        &self,
        _m: usize,
        _n: usize,
        _k: usize,
        _alpha: bf16,
        _a: &CudaSlice<bf16>,
        _b: &CudaSlice<bf16>,
        _beta: bf16,
        _c: &mut CudaSlice<bf16>,
    ) -> Result<()> {
        Ok(())
    }

    /// HGEMM into a CudaViewMut (for writing into sub-slices of a larger buffer).
    /// Same math as hgemm but uses cublasGemmEx with raw pointers to accept views.
    #[cfg(feature = "cuda")]
    pub fn hgemm_into(
        &self,
        m: usize,
        n: usize,
        k: usize,
        alpha: f32,
        a: &impl DevicePtr<half::f16>,
        b: &impl DevicePtr<half::f16>,
        beta: f32,
        c: &mut impl DevicePtrMut<half::f16>,
    ) -> Result<()> {
        use cudarc::cublas::sys::{
            cublasComputeType_t::CUBLAS_COMPUTE_32F,
            cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT_TENSOR_OP,
            cublasOperation_t::{CUBLAS_OP_N, CUBLAS_OP_T},
            cublasStatus_t::CUBLAS_STATUS_SUCCESS,
            cudaDataType_t::CUDA_R_16F,
        };
        let (b_ptr, _bg) = DevicePtr::device_ptr(b, &self.stream);
        let (a_ptr, _ag) = DevicePtr::device_ptr(a, &self.stream);
        let (c_ptr, _cg) = DevicePtrMut::device_ptr_mut(c, &self.stream);
        unsafe {
            let status = cudarc::cublas::sys::cublasGemmEx(
                *self.blas.handle(),
                CUBLAS_OP_T,
                CUBLAS_OP_N,
                n as i32,
                m as i32,
                k as i32,
                &alpha as *const f32 as *const std::ffi::c_void,
                b_ptr as *const std::ffi::c_void,
                CUDA_R_16F,
                k as i32,
                a_ptr as *const std::ffi::c_void,
                CUDA_R_16F,
                k as i32,
                &beta as *const f32 as *const std::ffi::c_void,
                c_ptr as *mut std::ffi::c_void,
                CUDA_R_16F,
                n as i32,
                CUBLAS_COMPUTE_32F,
                CUBLAS_GEMM_DEFAULT_TENSOR_OP,
            );
            if status != CUBLAS_STATUS_SUCCESS {
                return Err(crate::LLMError::GpuError(format!(
                    "hgemm_into failed: {status:?}"
                )));
            }
        }
        Ok(())
    }

    #[cfg(not(feature = "cuda"))]
    pub fn hgemm_into(
        &self,
        _m: usize,
        _n: usize,
        _k: usize,
        _alpha: f32,
        _a: &impl DevicePtr<half::f16>,
        _b: &impl DevicePtr<half::f16>,
        _beta: f32,
        _c: &mut impl DevicePtrMut<half::f16>,
    ) -> Result<()> {
        Ok(())
    }

    /// Strided batched HGEMM: multiple independent f16 GEMMs in one cuBLAS call.
    /// C[i] = alpha * A[i] @ B[i]^T + beta * C[i] for i in 0..batch_count
    /// All matrices share the same m,n,k but sit at strided offsets from the
    /// base pointers. Eliminates per-GEMM launch overhead (~15us each).
    #[cfg(feature = "cuda")]
    pub fn hgemm_strided_batched(
        &self,
        m: usize,
        n: usize,
        k: usize,
        alpha: f32,
        a: &impl DevicePtr<half::f16>,
        stride_a: i64,
        b: &impl DevicePtr<half::f16>,
        stride_b: i64,
        beta: f32,
        c: &mut impl DevicePtrMut<half::f16>,
        stride_c: i64,
        batch_count: usize,
    ) -> Result<()> {
        use cudarc::cublas::sys::{
            cublasComputeType_t::CUBLAS_COMPUTE_32F,
            cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT_TENSOR_OP,
            cublasOperation_t::{CUBLAS_OP_N, CUBLAS_OP_T},
            cublasStatus_t::CUBLAS_STATUS_SUCCESS,
            cudaDataType_t::CUDA_R_16F,
        };

        // Same row-major -> col-major mapping as hgemm_into:
        // b row[n,k] = col[k,n], OP_T -> [n,k]. lda=k.
        // a row[m,k] = col[k,m], OP_N -> [k,m]. ldb=k.
        // C_col[n,m] = row C[m,n]. ldc=n.
        let (b_ptr, _bg) = DevicePtr::device_ptr(b, &self.stream);
        let (a_ptr, _ag) = DevicePtr::device_ptr(a, &self.stream);
        let (c_ptr, _cg) = DevicePtrMut::device_ptr_mut(c, &self.stream);

        unsafe {
            let status = cudarc::cublas::sys::cublasGemmStridedBatchedEx(
                *self.blas.handle(),
                CUBLAS_OP_T,
                CUBLAS_OP_N,
                n as i32,
                m as i32,
                k as i32,
                &alpha as *const f32 as *const std::ffi::c_void,
                b_ptr as *const std::ffi::c_void,
                CUDA_R_16F,
                k as i32,
                stride_b,
                a_ptr as *const std::ffi::c_void,
                CUDA_R_16F,
                k as i32,
                stride_a,
                &beta as *const f32 as *const std::ffi::c_void,
                c_ptr as *mut std::ffi::c_void,
                CUDA_R_16F,
                n as i32,
                stride_c,
                batch_count as i32,
                CUBLAS_COMPUTE_32F,
                CUBLAS_GEMM_DEFAULT_TENSOR_OP,
            );
            if status != CUBLAS_STATUS_SUCCESS {
                return Err(crate::LLMError::GpuError(format!(
                    "hgemm_strided_batched failed: {status:?}"
                )));
            }
        }
        Ok(())
    }

    #[cfg(not(feature = "cuda"))]
    pub fn hgemm_strided_batched(
        &self,
        _m: usize,
        _n: usize,
        _k: usize,
        _alpha: f32,
        _a: &impl DevicePtr<half::f16>,
        _stride_a: i64,
        _b: &impl DevicePtr<half::f16>,
        _stride_b: i64,
        _beta: f32,
        _c: &mut impl DevicePtrMut<half::f16>,
        _stride_c: i64,
        _batch_count: usize,
    ) -> Result<()> {
        Ok(())
    }

    /// SGEMM (no transpose): C[m,n] = A[m,k] @ B[k,n]
    ///
    /// Both A and B are row-major. No transpose on either operand.
    /// Used for attention: probs[tokens, kv_len] @ V[kv_len, head_dim].
    pub fn sgemm_nn(
        &self,
        m: usize,
        n: usize,
        k: usize,
        alpha: f32,
        a: &CudaSlice<f32>,
        b: &CudaSlice<f32>,
        beta: f32,
        c: &mut CudaSlice<f32>,
    ) -> Result<()> {
        // Row-major C[m,n] = A[m,k] @ B[k,n]
        // cuBLAS col-major: C_col[n,m] = B_col[n,k] @ A_col[k,m]
        // B row[k,n] = col[n,k], OP_N -> [n,k]. lda=n.
        // A row[m,k] = col[k,m], OP_N -> [k,m]. ldb=k.
        // C_col[n,m], ldc=n.
        unsafe {
            self.blas
                .gemm(
                    GemmConfig {
                        transa: cublasOperation_t::CUBLAS_OP_N,
                        transb: cublasOperation_t::CUBLAS_OP_N,
                        m: n as i32,
                        n: m as i32,
                        k: k as i32,
                        alpha,
                        lda: n as i32,
                        ldb: k as i32,
                        beta,
                        ldc: n as i32,
                    },
                    b,
                    a,
                    c,
                )
                .map_err(|e| crate::LLMError::GpuError(format!("cuBLAS sgemm_nn failed: {e}")))?;
        }
        Ok(())
    }

    /// Batched SGEMM for multiple independent matrix multiplications (e.g. multi-head attention).
    ///
    /// Each triple (a_batch[i], b_batch[i], c_batch[i]) is an independent GEMM with
    /// the same m/n/k dimensions.
    pub fn sgemm_batched(
        &self,
        _m: usize,
        _n: usize,
        _k: usize,
        _alpha: f32,
        _a_batch: &[&CudaSlice<f32>],
        _b_batch: &[&CudaSlice<f32>],
        _beta: f32,
        _c_batch: &mut [&mut CudaSlice<f32>],
    ) -> Result<()> {
        // TODO: implement via cublasSgemmBatched or cublasSgemmStridedBatched
        Err(crate::LLMError::GpuError(
            "sgemm_batched not yet implemented".into(),
        ))
    }

    /// SGEMV: y = alpha * A * x + beta * y
    ///
    /// A: [m, n] row-major, x: [n], y: [m].
    ///
    /// For row-major A, cuBLAS (column-major) sees A^T, so we pass CUBLAS_OP_T
    /// to get the correct row-major matrix-vector product.
    pub fn sgemv(
        &self,
        m: usize,
        n: usize,
        alpha: f32,
        a: &CudaSlice<f32>,
        x: &CudaSlice<f32>,
        beta: f32,
        y: &mut CudaSlice<f32>,
    ) -> Result<()> {
        // SAFETY: cuBLAS reads/writes device memory through valid CudaSlice handles.
        // Row-major A stored contiguously is column-major A^T with dims (n, m).
        // We want y = A * x  =>  cublas: y = Op(A_col) * x  where A_col is (n,m).
        // Op = CUBLAS_OP_T gives us A^T_col = A_row which is what we want.
        unsafe {
            self.blas
                .gemv(
                    GemvConfig {
                        trans: cublasOperation_t::CUBLAS_OP_T,
                        m: n as i32,
                        n: m as i32,
                        alpha,
                        lda: n as i32,
                        incx: 1,
                        beta,
                        incy: 1,
                    },
                    a,
                    x,
                    y,
                )
                .map_err(|e| crate::LLMError::GpuError(format!("cuBLAS sgemv failed: {e}")))?;
        }
        Ok(())
    }

    /// HGEMV for M=1 decode: output[n] = weight[n,k] @ input[k] (f16)
    ///
    /// Lower launch overhead than GEMM at M=1. Uses cublasGemmEx with m=1
    /// which cuBLAS internally dispatches as a GEMV kernel. F32 accumulation
    /// for numerical stability.
    ///
    /// Weight layout: row-major [n, k] (PyTorch [out_features, in_features]).
    #[cfg(feature = "cuda")]
    pub fn hgemv_f16(
        &self,
        n: usize,
        k: usize,
        alpha: f32,
        weight: &impl DevicePtr<half::f16>,
        input: &impl DevicePtr<half::f16>,
        beta: f32,
        output: &mut impl DevicePtrMut<half::f16>,
    ) -> Result<()> {
        use cudarc::cublas::sys::{
            cublasComputeType_t::CUBLAS_COMPUTE_32F,
            cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT_TENSOR_OP,
            cublasOperation_t::{CUBLAS_OP_N, CUBLAS_OP_T},
            cublasStatus_t::CUBLAS_STATUS_SUCCESS,
            cudaDataType_t::CUDA_R_16F,
        };

        // This is hgemm_into with m=1. Row-major -> col-major mapping:
        //   weight row[n,k] = col[k,n], OP_T -> [n,k], lda=k
        //   input  row[1,k] = col[k,1], OP_N -> [k,1], ldb=k
        //   output col[n,1] = row[1,n], ldc=n
        let (w_ptr, _wg) = DevicePtr::device_ptr(weight, &self.stream);
        let (x_ptr, _xg) = DevicePtr::device_ptr(input, &self.stream);
        let (y_ptr, _yg) = DevicePtrMut::device_ptr_mut(output, &self.stream);

        unsafe {
            let status = cudarc::cublas::sys::cublasGemmEx(
                *self.blas.handle(),
                CUBLAS_OP_T,
                CUBLAS_OP_N,
                n as i32,
                1i32,
                k as i32,
                &alpha as *const f32 as *const std::ffi::c_void,
                w_ptr as *const std::ffi::c_void,
                CUDA_R_16F,
                k as i32,
                x_ptr as *const std::ffi::c_void,
                CUDA_R_16F,
                k as i32,
                &beta as *const f32 as *const std::ffi::c_void,
                y_ptr as *mut std::ffi::c_void,
                CUDA_R_16F,
                n as i32,
                CUBLAS_COMPUTE_32F,
                CUBLAS_GEMM_DEFAULT_TENSOR_OP,
            );
            if status != CUBLAS_STATUS_SUCCESS {
                return Err(crate::LLMError::GpuError(format!(
                    "hgemv_f16 failed: {status:?}"
                )));
            }
        }
        Ok(())
    }

    #[cfg(not(feature = "cuda"))]
    pub fn hgemv_f16(
        &self,
        _n: usize,
        _k: usize,
        _alpha: f32,
        _weight: &impl DevicePtr<half::f16>,
        _input: &impl DevicePtr<half::f16>,
        _beta: f32,
        _output: &mut impl DevicePtrMut<half::f16>,
    ) -> Result<()> {
        Ok(())
    }
}

#[cfg(test)]
#[cfg(feature = "cuda")]
mod tests {
    use super::*;
    use cudarc::driver::CudaContext;
    use half::bf16;

    #[test]
    fn sgemm_a_times_bt() {
        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.new_stream().unwrap();
        let handle = CublasHandle::new(stream.clone()).unwrap();

        // A[2,3] row-major (activations)
        let a_host: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        // B[4,3] row-major (weights in PyTorch [out, in] layout)
        let b_host: Vec<f32> = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ];

        let a_gpu = stream.clone_htod(&a_host).unwrap();
        let b_gpu = stream.clone_htod(&b_host).unwrap();
        let mut c_gpu = stream.alloc_zeros::<f32>(2 * 4).unwrap();

        // sgemm(m=2, n=4, k=3): C[2,4] = A[2,3] @ B[4,3]^T
        handle
            .sgemm(2, 4, 3, 1.0, &a_gpu, &b_gpu, 0.0, &mut c_gpu)
            .unwrap();

        let c_host = stream.clone_dtoh(&c_gpu).unwrap();

        // CPU reference: C[i,j] = sum_k A[i,k] * B[j,k]
        let mut expected = vec![0.0f32; 8];
        for i in 0..2 {
            for j in 0..4 {
                for kk in 0..3 {
                    expected[i * 4 + j] += a_host[i * 3 + kk] * b_host[j * 3 + kk];
                }
            }
        }

        for (idx, (got, exp)) in c_host.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 1e-4,
                "mismatch at index {idx}: got {got}, expected {exp}"
            );
        }
    }

    #[test]
    fn hgemm_bf16_a_times_bt() {
        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.new_stream().unwrap();
        let handle = CublasHandle::new(stream.clone()).unwrap();

        let a_host: Vec<bf16> = vec![
            bf16::from_f32(1.0),
            bf16::from_f32(2.0),
            bf16::from_f32(3.0),
            bf16::from_f32(4.0),
            bf16::from_f32(5.0),
            bf16::from_f32(6.0),
        ];
        let b_host: Vec<bf16> = vec![
            bf16::from_f32(1.0),
            bf16::from_f32(2.0),
            bf16::from_f32(3.0),
            bf16::from_f32(4.0),
            bf16::from_f32(5.0),
            bf16::from_f32(6.0),
            bf16::from_f32(7.0),
            bf16::from_f32(8.0),
            bf16::from_f32(9.0),
            bf16::from_f32(10.0),
            bf16::from_f32(11.0),
            bf16::from_f32(12.0),
        ];

        let a_gpu = stream.clone_htod(&a_host).unwrap();
        let b_gpu = stream.clone_htod(&b_host).unwrap();
        let mut c_gpu = stream.alloc_zeros::<bf16>(2 * 4).unwrap();

        handle
            .hgemm_bf16(2, 4, 3, bf16::ONE, &a_gpu, &b_gpu, bf16::ZERO, &mut c_gpu)
            .unwrap();

        let c_host = stream.clone_dtoh(&c_gpu).unwrap();

        let mut expected = vec![0.0f32; 8];
        for i in 0..2 {
            for j in 0..4 {
                for kk in 0..3 {
                    expected[i * 4 + j] +=
                        a_host[i * 3 + kk].to_f32() * b_host[j * 3 + kk].to_f32();
                }
            }
        }

        for (idx, (got, exp)) in c_host.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got.to_f32() - exp).abs() < 1e-2,
                "mismatch at index {idx}: got {}, expected {exp}",
                got.to_f32()
            );
        }
    }
}
