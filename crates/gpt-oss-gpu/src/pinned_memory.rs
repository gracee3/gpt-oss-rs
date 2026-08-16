//! Pinned (page-locked) host memory for faster HtoD/DtoH transfers.
//!
//! CUDA pinned memory enables DMA transfers that bypass the CPU page tables,
//! achieving ~2x higher bandwidth compared to pageable memory. This is critical
//! for the embedding lookup upload, logits download, and any swap operations.
//!
//! Under `mock-gpu`, this falls back to normal heap allocation.
//! Under `cuda`, uses portable `cuMemHostAlloc` memory so one bounded relay
//! allocation can be used by both CUDA contexts.

use bytemuck::Pod;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;

use crate::Result;

/// A buffer backed by page-locked (pinned) host memory.
///
/// On CUDA systems, this memory is registered with the GPU driver for DMA,
/// providing ~2x faster HtoD/DtoH transfer rates. On mock-gpu, this is a
/// regular heap allocation.
pub struct PinnedBuffer<T: Pod + Send> {
    #[cfg(feature = "cuda")]
    ptr: *mut T,
    #[cfg(feature = "cuda")]
    len: usize,
    #[cfg(all(feature = "mock-gpu", not(feature = "cuda")))]
    data: Vec<T>,
    _marker: std::marker::PhantomData<T>,
}

// SAFETY: Pinned memory is a host allocation; it is safe to send/sync
// just like Vec<T>. The CUDA driver handles the DMA registration.
unsafe impl<T: Pod + Send> Send for PinnedBuffer<T> {}
unsafe impl<T: Pod + Send> Sync for PinnedBuffer<T> {}

impl<T: Pod + Send> PinnedBuffer<T> {
    /// Allocate a pinned buffer of `count` elements, zeroed.
    ///
    /// CUDA builds require a live CUDA context to be bound to the calling
    /// thread for the lifetime of the allocation.
    pub fn new(count: usize) -> Result<Self> {
        if count == 0 {
            #[cfg(feature = "cuda")]
            return Ok(Self {
                ptr: std::ptr::null_mut(),
                len: 0,
                _marker: std::marker::PhantomData,
            });
            #[cfg(all(feature = "mock-gpu", not(feature = "cuda")))]
            return Ok(Self {
                data: Vec::new(),
                _marker: std::marker::PhantomData,
            });
        }

        #[cfg(feature = "cuda")]
        {
            let bytes = count.checked_mul(std::mem::size_of::<T>()).ok_or_else(|| {
                crate::LLMError::MemoryError("PinnedBuffer allocation size overflow".to_string())
            })?;
            let mut ptr: *mut std::ffi::c_void = std::ptr::null_mut();

            // SAFETY: cuMemHostAlloc allocates page-locked memory on the host.
            // PORTABLE is required by H4 because GPU0 and GPU1 have separate
            // contexts but share the same bounded relay leases.
            // The pointer is valid until cuMemFreeHost is called (in Drop).
            let result = unsafe {
                cudarc::driver::sys::cuMemHostAlloc(
                    &mut ptr,
                    bytes,
                    cudarc::driver::sys::CU_MEMHOSTALLOC_PORTABLE,
                )
            };

            if result != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
                return Err(crate::LLMError::MemoryError(format!(
                    "portable cuMemHostAlloc failed for {} bytes: {:?}",
                    bytes, result
                )));
            }

            // Zero the memory
            // SAFETY: ptr is a valid allocation of `bytes` size.
            unsafe {
                std::ptr::write_bytes(ptr as *mut u8, 0, bytes);
            }

            Ok(Self {
                ptr: ptr as *mut T,
                len: count,
                _marker: std::marker::PhantomData,
            })
        }

        #[cfg(all(feature = "mock-gpu", not(feature = "cuda")))]
        {
            Ok(Self {
                data: vec![T::zeroed(); count],
                _marker: std::marker::PhantomData,
            })
        }
    }

    pub fn len(&self) -> usize {
        #[cfg(feature = "cuda")]
        {
            self.len
        }
        #[cfg(all(feature = "mock-gpu", not(feature = "cuda")))]
        {
            self.data.len()
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn size_bytes(&self) -> usize {
        self.len() * std::mem::size_of::<T>()
    }

    pub fn as_slice(&self) -> &[T] {
        #[cfg(feature = "cuda")]
        {
            if self.ptr.is_null() || self.len == 0 {
                return &[];
            }
            // SAFETY: ptr was allocated with cuMemHostAlloc for self.len elements.
            unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
        }
        #[cfg(all(feature = "mock-gpu", not(feature = "cuda")))]
        {
            &self.data
        }
    }

    pub fn as_mut_slice(&mut self) -> &mut [T] {
        #[cfg(feature = "cuda")]
        {
            if self.ptr.is_null() || self.len == 0 {
                return &mut [];
            }
            // SAFETY: ptr was allocated with cuMemHostAlloc for self.len elements.
            unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
        }
        #[cfg(all(feature = "mock-gpu", not(feature = "cuda")))]
        {
            &mut self.data
        }
    }

    pub fn as_ptr(&self) -> *const T {
        #[cfg(feature = "cuda")]
        {
            self.ptr as *const T
        }
        #[cfg(all(feature = "mock-gpu", not(feature = "cuda")))]
        {
            self.data.as_ptr()
        }
    }

    pub fn as_mut_ptr(&mut self) -> *mut T {
        #[cfg(feature = "cuda")]
        {
            self.ptr
        }
        #[cfg(all(feature = "mock-gpu", not(feature = "cuda")))]
        {
            self.data.as_mut_ptr()
        }
    }

    /// Copy from a host slice into this pinned buffer.
    pub fn copy_from_slice(&mut self, src: &[T]) -> Result<()> {
        if src.len() != self.len() {
            return Err(crate::LLMError::MemoryError(format!(
                "PinnedBuffer copy_from_slice: src len {} != buf len {}",
                src.len(),
                self.len()
            )));
        }
        self.as_mut_slice().copy_from_slice(src);
        Ok(())
    }

    /// Copy this pinned buffer to a new Vec.
    pub fn to_vec(&self) -> Vec<T> {
        self.as_slice().to_vec()
    }
}

impl<T: Pod + Send> Drop for PinnedBuffer<T> {
    fn drop(&mut self) {
        #[cfg(feature = "cuda")]
        {
            if !self.ptr.is_null() && self.len > 0 {
                // SAFETY: ptr was allocated with cuMemHostAlloc.
                unsafe {
                    let _ = cudarc::driver::sys::cuMemFreeHost(self.ptr as *mut std::ffi::c_void);
                }
            }
        }
        // mock-gpu: Vec drops automatically
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BoundedPinnedPoolStats {
    pub capacity: usize,
    pub available: usize,
    pub checked_out: usize,
    pub high_water: usize,
    pub fixed_allocations: usize,
    pub exhaustions: u64,
    pub quarantined: u64,
    pub bytes_per_buffer: usize,
}

struct BoundedPinnedPoolInner<T: Pod + Send> {
    buffers: parking_lot::Mutex<Vec<PinnedBuffer<T>>>,
    elem_count: usize,
    capacity: usize,
    checked_out: AtomicUsize,
    high_water: AtomicUsize,
    exhaustions: AtomicU64,
    quarantined: AtomicU64,
}

/// A fixed-capacity pinned pool that never allocates after construction.
///
/// Unlike [`PinnedPool`], exhaustion is an error. Callers must reserve every
/// required lease before enqueueing work and return leases only after every
/// CPU reader and CUDA event that can reference them is terminal.
#[derive(Clone)]
pub struct BoundedPinnedPool<T: Pod + Send> {
    inner: Arc<BoundedPinnedPoolInner<T>>,
}

impl<T: Pod + Send> BoundedPinnedPool<T> {
    pub fn warm_exact(elem_count: usize, capacity: usize) -> Result<Self> {
        if capacity == 0 {
            return Err(crate::LLMError::MemoryError(
                "bounded pinned pool capacity must be nonzero".into(),
            ));
        }
        let mut buffers = Vec::with_capacity(capacity);
        for _ in 0..capacity {
            buffers.push(PinnedBuffer::new(elem_count)?);
        }
        Ok(Self {
            inner: Arc::new(BoundedPinnedPoolInner {
                buffers: parking_lot::Mutex::new(buffers),
                elem_count,
                capacity,
                checked_out: AtomicUsize::new(0),
                high_water: AtomicUsize::new(0),
                exhaustions: AtomicU64::new(0),
                quarantined: AtomicU64::new(0),
            }),
        })
    }

    pub fn try_acquire(&self, generation: u64) -> Result<BoundedPinnedLease<T>> {
        let buffer = self.inner.buffers.lock().pop().ok_or_else(|| {
            self.inner.exhaustions.fetch_add(1, Ordering::Relaxed);
            crate::LLMError::MemoryError("bounded pinned pool exhausted".into())
        })?;
        let current = self.inner.checked_out.fetch_add(1, Ordering::AcqRel) + 1;
        self.inner.high_water.fetch_max(current, Ordering::AcqRel);
        Ok(BoundedPinnedLease {
            pool: Arc::clone(&self.inner),
            buffer: Some(buffer),
            generation,
        })
    }

    pub fn stats(&self) -> BoundedPinnedPoolStats {
        BoundedPinnedPoolStats {
            capacity: self.inner.capacity,
            available: self.inner.buffers.lock().len(),
            checked_out: self.inner.checked_out.load(Ordering::Acquire),
            high_water: self.inner.high_water.load(Ordering::Acquire),
            fixed_allocations: self.inner.capacity,
            exhaustions: self.inner.exhaustions.load(Ordering::Acquire),
            quarantined: self.inner.quarantined.load(Ordering::Acquire),
            bytes_per_buffer: self
                .inner
                .elem_count
                .saturating_mul(std::mem::size_of::<T>()),
        }
    }

    pub fn elem_count(&self) -> usize {
        self.inner.elem_count
    }
}

/// One generation-tagged buffer borrowed from a [`BoundedPinnedPool`].
///
/// A lease has no automatic reusable return path. `release_drained` is the
/// sole pool-return operation. Dropping an active lease frees/quarantines its
/// allocation, which is safe but permanently reduces capacity and is a gate
/// failure for the heterogeneous proof.
pub struct BoundedPinnedLease<T: Pod + Send> {
    pool: Arc<BoundedPinnedPoolInner<T>>,
    buffer: Option<PinnedBuffer<T>>,
    generation: u64,
}

impl<T: Pod + Send> BoundedPinnedLease<T> {
    pub const fn generation(&self) -> u64 {
        self.generation
    }

    pub fn as_slice(&self) -> &[T] {
        self.buffer
            .as_ref()
            .expect("bounded pinned lease buffer is present")
            .as_slice()
    }

    pub fn as_mut_slice(&mut self) -> &mut [T] {
        self.buffer
            .as_mut()
            .expect("bounded pinned lease buffer is present")
            .as_mut_slice()
    }

    pub fn release_drained(mut self) -> Result<()> {
        let buffer = self.buffer.as_ref().ok_or_else(|| {
            crate::LLMError::MemoryError("bounded pinned lease already released".into())
        })?;
        if buffer.len() != self.pool.elem_count {
            return Err(crate::LLMError::MemoryError(
                "bounded pinned lease size changed".into(),
            ));
        }
        let mut available = self.pool.buffers.lock();
        if available.len() >= self.pool.capacity {
            return Err(crate::LLMError::MemoryError(
                "bounded pinned pool return exceeds fixed capacity".into(),
            ));
        }
        let buffer = self
            .buffer
            .take()
            .expect("bounded pinned lease was checked before return");
        available.push(buffer);
        drop(available);
        self.pool.checked_out.fetch_sub(1, Ordering::AcqRel);
        Ok(())
    }
}

impl<T: Pod + Send> Drop for BoundedPinnedLease<T> {
    fn drop(&mut self) {
        if self.buffer.is_some() {
            self.pool.quarantined.fetch_add(1, Ordering::Relaxed);
            self.pool.checked_out.fetch_sub(1, Ordering::AcqRel);
        }
    }
}

/// A reusable pool of pinned buffers for amortizing allocation cost.
///
/// Keeps a stack of previously-used buffers. When a buffer of the right
/// size is requested, it pops from the pool instead of allocating.
/// When returned, buffers go back onto the pool.
pub struct PinnedPool<T: Pod + Send> {
    buffers: parking_lot::Mutex<Vec<PinnedBuffer<T>>>,
    elem_count: usize,
}

impl<T: Pod + Send> PinnedPool<T> {
    /// Create a pool that manages buffers of `elem_count` elements each.
    pub fn new(elem_count: usize) -> Self {
        Self {
            buffers: parking_lot::Mutex::new(Vec::new()),
            elem_count,
        }
    }

    /// Pre-allocate `n` buffers into the pool.
    pub fn warm(&self, n: usize) -> Result<()> {
        let mut pool = self.buffers.lock();
        for _ in 0..n {
            pool.push(PinnedBuffer::new(self.elem_count)?);
        }
        Ok(())
    }

    /// Get a buffer from the pool, or allocate a new one.
    pub fn acquire(&self) -> Result<PinnedBuffer<T>> {
        let mut pool = self.buffers.lock();
        match pool.pop() {
            Some(buf) => Ok(buf),
            None => PinnedBuffer::new(self.elem_count),
        }
    }

    /// Return a buffer to the pool for reuse.
    pub fn release(&self, buf: PinnedBuffer<T>) {
        if buf.len() == self.elem_count {
            let mut pool = self.buffers.lock();
            pool.push(buf);
        }
        // Wrong-sized buffers are dropped (freed).
    }

    /// Number of buffers currently in the pool (not in use).
    pub fn available(&self) -> usize {
        self.buffers.lock().len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "cuda")]
    fn cuda_context() -> std::sync::Arc<cudarc::driver::CudaContext> {
        cudarc::driver::CudaContext::new(0).unwrap()
    }

    #[test]
    fn pinned_buffer_basic() {
        #[cfg(feature = "cuda")]
        let _context = cuda_context();
        let mut buf = PinnedBuffer::<f32>::new(16).unwrap();
        assert_eq!(buf.len(), 16);
        assert_eq!(buf.size_bytes(), 64);

        let data: Vec<f32> = (0..16).map(|i| i as f32).collect();
        buf.copy_from_slice(&data).unwrap();
        assert_eq!(buf.to_vec(), data);
    }

    #[test]
    fn pinned_buffer_empty() {
        let buf = PinnedBuffer::<u8>::new(0).unwrap();
        assert!(buf.is_empty());
        assert_eq!(buf.size_bytes(), 0);
    }

    #[test]
    fn pinned_pool_reuse() {
        #[cfg(feature = "cuda")]
        let _context = cuda_context();
        let pool = PinnedPool::<f32>::new(64);
        pool.warm(2).unwrap();
        assert_eq!(pool.available(), 2);

        let buf1 = pool.acquire().unwrap();
        assert_eq!(pool.available(), 1);
        assert_eq!(buf1.len(), 64);

        pool.release(buf1);
        assert_eq!(pool.available(), 2);
    }

    #[test]
    fn pinned_buffer_size_mismatch() {
        #[cfg(feature = "cuda")]
        let _context = cuda_context();
        let mut buf = PinnedBuffer::<f32>::new(4).unwrap();
        assert!(buf.copy_from_slice(&[1.0, 2.0]).is_err());
    }

    #[test]
    fn bounded_pool_exhaustion_never_allocates_and_drained_return_reuses() {
        #[cfg(feature = "cuda")]
        let _context = cuda_context();
        let pool = BoundedPinnedPool::<u8>::warm_exact(128, 1).unwrap();
        let lease = pool.try_acquire(7).unwrap();
        assert_eq!(lease.generation(), 7);
        let before = pool.stats();
        assert!(pool.try_acquire(8).is_err());
        let exhausted = pool.stats();
        assert_eq!(exhausted.fixed_allocations, before.fixed_allocations);
        assert_eq!(exhausted.available, 0);
        assert_eq!(exhausted.exhaustions, 1);
        lease.release_drained().unwrap();
        let reused = pool.try_acquire(9).unwrap();
        assert_eq!(reused.generation(), 9);
        reused.release_drained().unwrap();
        let final_stats = pool.stats();
        assert_eq!(final_stats.available, 1);
        assert_eq!(final_stats.checked_out, 0);
        assert_eq!(final_stats.high_water, 1);
        assert_eq!(final_stats.quarantined, 0);
    }

    #[test]
    fn dropped_bounded_lease_is_quarantined_not_reused() {
        #[cfg(feature = "cuda")]
        let _context = cuda_context();
        let pool = BoundedPinnedPool::<u8>::warm_exact(64, 1).unwrap();
        drop(pool.try_acquire(3).unwrap());
        let stats = pool.stats();
        assert_eq!(stats.available, 0);
        assert_eq!(stats.checked_out, 0);
        assert_eq!(stats.quarantined, 1);
        assert!(pool.try_acquire(4).is_err());
        assert_eq!(pool.stats().fixed_allocations, 1);
    }
}
