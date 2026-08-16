//! Fixed-capacity pinned relay reservations and GPU0 result-slot uploads.
//!
//! H4 owns five prewarmed host buffers. Reservations never allocate and are
//! all-or-none. A lease can return only after every CPU/CUDA reference drains.

use cudarc::driver::{sys::CUevent_flags, CudaSlice, CudaStream};
use gpt_oss_core::error::{LLMError, Result};
use gpt_oss_gpu::event::CorrelatedTimeline;
use gpt_oss_gpu::pinned_memory::{BoundedPinnedLease, BoundedPinnedPool, BoundedPinnedPoolStats};

use super::contract::{GPT_OSS_HIDDEN_SIZE, GPT_OSS_ROUTE_WIRE_V1_BYTES, GPT_OSS_TOP_K};
use super::packing::{
    PackedDispatchPlan, RelayBytePlan, H4_DECODE_PINNED_CAP_BYTES, H4_PREFILL_MAX_ROWS,
    H4_PREFILL_PINNED_CAP_BYTES, H4_ROUTE_DESCRIPTOR_MAX_BYTES,
};
use super::router::CudaExactRouter;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RelayPinnedPoolStats {
    pub source_activation: BoundedPinnedPoolStats,
    pub route_descriptors: BoundedPinnedPoolStats,
    pub remote_gpu_input: BoundedPinnedPoolStats,
    pub remote_gpu_result: BoundedPinnedPoolStats,
    pub cpu_result: BoundedPinnedPoolStats,
    pub raw_capacity_bytes: usize,
    pub hard_cap_bytes: usize,
}

/// Exactly five capacity-one pools sized for one decode row or one bounded
/// prefill chunk. Their capacities do not depend on the observed route mix.
pub struct RelayPinnedPools {
    max_rows: usize,
    source_activation: BoundedPinnedPool<u16>,
    route_descriptors: BoundedPinnedPool<u8>,
    remote_gpu_input: BoundedPinnedPool<u16>,
    remote_gpu_result: BoundedPinnedPool<u16>,
    cpu_result: BoundedPinnedPool<u16>,
}

impl RelayPinnedPools {
    pub fn warm_exact(router: &CudaExactRouter, max_rows: usize) -> Result<Self> {
        if max_rows == 0 || max_rows > H4_PREFILL_MAX_ROWS {
            return Err(LLMError::MemoryError(format!(
                "relay pool rows {max_rows} outside 1..={H4_PREFILL_MAX_ROWS}"
            )));
        }
        router
            .relay_stream()
            .context()
            .bind_to_thread()
            .map_err(cuda_error("relay pool context bind"))?;
        let route_capacity = max_rows * GPT_OSS_TOP_K;
        let pools = Self {
            max_rows,
            source_activation: BoundedPinnedPool::warm_exact(max_rows * GPT_OSS_HIDDEN_SIZE, 1)?,
            route_descriptors: BoundedPinnedPool::warm_exact(
                route_capacity * GPT_OSS_ROUTE_WIRE_V1_BYTES,
                1,
            )?,
            remote_gpu_input: BoundedPinnedPool::warm_exact(
                route_capacity * GPT_OSS_HIDDEN_SIZE,
                1,
            )?,
            remote_gpu_result: BoundedPinnedPool::warm_exact(
                route_capacity * GPT_OSS_HIDDEN_SIZE,
                1,
            )?,
            cpu_result: BoundedPinnedPool::warm_exact(route_capacity * GPT_OSS_HIDDEN_SIZE, 1)?,
        };
        let stats = pools.stats();
        if stats.raw_capacity_bytes > stats.hard_cap_bytes {
            return Err(LLMError::MemoryError(format!(
                "relay fixed capacity {} exceeds hard cap {}",
                stats.raw_capacity_bytes, stats.hard_cap_bytes
            )));
        }
        Ok(pools)
    }

    /// Reserve all five fixed buffers before any CPU task or CUDA enqueue.
    /// On exhaustion, every earlier reservation is returned immediately.
    pub fn try_reserve_all(&self, generation: u64) -> Result<RelayPinnedReservation> {
        let mut source_activation = None;
        let mut route_descriptors = None;
        let mut remote_gpu_input = None;
        let mut remote_gpu_result = None;
        let mut cpu_result = None;
        let acquired = (|| -> Result<()> {
            source_activation = Some(self.source_activation.try_acquire(generation)?);
            route_descriptors = Some(self.route_descriptors.try_acquire(generation)?);
            remote_gpu_input = Some(self.remote_gpu_input.try_acquire(generation)?);
            remote_gpu_result = Some(self.remote_gpu_result.try_acquire(generation)?);
            cpu_result = Some(self.cpu_result.try_acquire(generation)?);
            Ok(())
        })();
        if let Err(error) = acquired {
            release_if_present(cpu_result)?;
            release_if_present(remote_gpu_result)?;
            release_if_present(remote_gpu_input)?;
            release_if_present(route_descriptors)?;
            release_if_present(source_activation)?;
            return Err(error);
        }
        Ok(RelayPinnedReservation {
            generation,
            source_activation: source_activation.expect("source lease acquired"),
            route_descriptors: route_descriptors.expect("descriptor lease acquired"),
            remote_gpu_input: remote_gpu_input.expect("remote input lease acquired"),
            remote_gpu_result: remote_gpu_result.expect("remote result lease acquired"),
            cpu_result: cpu_result.expect("CPU result lease acquired"),
        })
    }

    pub fn stats(&self) -> RelayPinnedPoolStats {
        let source_activation = self.source_activation.stats();
        let route_descriptors = self.route_descriptors.stats();
        let remote_gpu_input = self.remote_gpu_input.stats();
        let remote_gpu_result = self.remote_gpu_result.stats();
        let cpu_result = self.cpu_result.stats();
        let raw_capacity_bytes = source_activation.bytes_per_buffer
            + route_descriptors.bytes_per_buffer
            + remote_gpu_input.bytes_per_buffer
            + remote_gpu_result.bytes_per_buffer
            + cpu_result.bytes_per_buffer;
        let hard_cap_bytes = if self.max_rows == 1 {
            H4_DECODE_PINNED_CAP_BYTES
        } else {
            H4_PREFILL_PINNED_CAP_BYTES
        };
        RelayPinnedPoolStats {
            source_activation,
            route_descriptors,
            remote_gpu_input,
            remote_gpu_result,
            cpu_result,
            raw_capacity_bytes,
            hard_cap_bytes,
        }
    }

    /// Hold the second pool so a test can prove that failure after acquiring
    /// the source lease rolls that earlier acquisition back without allocating.
    #[cfg(feature = "heterogeneous-test-faults")]
    pub fn hold_route_descriptors_for_test(
        &self,
        generation: u64,
    ) -> Result<BoundedPinnedLease<u8>> {
        self.route_descriptors.try_acquire(generation)
    }
}

fn release_if_present<T: bytemuck::Pod + Send>(lease: Option<BoundedPinnedLease<T>>) -> Result<()> {
    if let Some(lease) = lease {
        lease.release_drained()?;
    }
    Ok(())
}

pub struct RelayPinnedReservation {
    generation: u64,
    pub source_activation: BoundedPinnedLease<u16>,
    pub route_descriptors: BoundedPinnedLease<u8>,
    pub remote_gpu_input: BoundedPinnedLease<u16>,
    pub remote_gpu_result: BoundedPinnedLease<u16>,
    pub cpu_result: BoundedPinnedLease<u16>,
}

impl RelayPinnedReservation {
    pub const fn generation(&self) -> u64 {
        self.generation
    }

    pub fn release_drained(self) -> Result<()> {
        // Continue releasing if a defensive invariant fails so no later lease
        // is accidentally quarantined merely because an earlier return failed.
        let mut first_error = None;
        for result in [
            self.cpu_result.release_drained(),
            self.remote_gpu_result.release_drained(),
            self.remote_gpu_input.release_drained(),
            self.route_descriptors.release_drained(),
            self.source_activation.release_drained(),
        ] {
            if first_error.is_none() {
                first_error = result.err();
            }
        }
        if let Some(error) = first_error {
            Err(error)
        } else {
            Ok(())
        }
    }
}

/// Copy canonical source rows into the stable remote-owner route slots. The
/// source arena remains full row-major; no host compaction changes row identity.
pub fn pack_remote_inputs(
    plan: &PackedDispatchPlan,
    source_activation: &BoundedPinnedLease<u16>,
    remote_gpu_input: &mut BoundedPinnedLease<u16>,
) -> Result<()> {
    let source_required = plan.rows as usize * GPT_OSS_HIDDEN_SIZE;
    if source_activation.as_slice().len() < source_required {
        return Err(LLMError::MemoryError(
            "relay source activation lease is undersized".into(),
        ));
    }
    remote_gpu_input.as_mut_slice().fill(0);
    for owner in &plan.remote_gpu {
        for route in &owner.routes {
            let source_row = route.relay_activation_slot as usize;
            if source_row != route.descriptor.route.source_row as usize {
                return Err(LLMError::ModelError(
                    "relay route lost canonical source-row identity".into(),
                ));
            }
            let source_start = source_row * GPT_OSS_HIDDEN_SIZE;
            let destination_start = route.owner_route_slot as usize * GPT_OSS_HIDDEN_SIZE;
            let source =
                &source_activation.as_slice()[source_start..source_start + GPT_OSS_HIDDEN_SIZE];
            remote_gpu_input.as_mut_slice()
                [destination_start..destination_start + GPT_OSS_HIDDEN_SIZE]
                .copy_from_slice(source);
        }
    }
    Ok(())
}

#[cfg(feature = "heterogeneous-test-faults")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResultRelayInjectedFault {
    AfterFirstResultEnqueue,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResultRelayExecution {
    pub cpu_h2d_bytes: usize,
    pub remote_gpu_h2d_bytes: usize,
    pub evidence_d2h_bytes: usize,
}

/// Detached H4 result uploader. It places CPU and GPU1 outputs into canonical
/// GPU0 route slots but deliberately performs no weighting or reduction.
pub struct CudaResultRelay {
    stream: std::sync::Arc<CudaStream>,
    contribution_arena: CudaSlice<u16>,
    max_routes: usize,
    #[cfg(feature = "heterogeneous-test-faults")]
    injected_fault: Option<ResultRelayInjectedFault>,
    #[cfg(feature = "heterogeneous-test-faults")]
    last_fault_drained: bool,
}

impl CudaResultRelay {
    pub fn new(router: &CudaExactRouter, max_rows: usize) -> Result<Self> {
        if max_rows == 0 || max_rows > H4_PREFILL_MAX_ROWS {
            return Err(LLMError::GpuError(format!(
                "result relay rows {max_rows} outside 1..={H4_PREFILL_MAX_ROWS}"
            )));
        }
        let stream = std::sync::Arc::clone(router.relay_stream());
        let max_routes = max_rows * GPT_OSS_TOP_K;
        let contribution_arena = stream
            .alloc_zeros::<u16>(max_routes * GPT_OSS_HIDDEN_SIZE)
            .map_err(cuda_error("result relay arena allocation"))?;
        stream
            .synchronize()
            .map_err(cuda_error("result relay construction drain"))?;
        Ok(Self {
            stream,
            contribution_arena,
            max_routes,
            #[cfg(feature = "heterogeneous-test-faults")]
            injected_fault: None,
            #[cfg(feature = "heterogeneous-test-faults")]
            last_fault_drained: false,
        })
    }

    #[cfg(feature = "heterogeneous-test-faults")]
    pub fn inject_next_failure(&mut self, fault: ResultRelayInjectedFault) -> Result<()> {
        if self.injected_fault.is_some() {
            return Err(LLMError::GpuError(
                "result relay already has an armed test fault".into(),
            ));
        }
        self.injected_fault = Some(fault);
        self.last_fault_drained = false;
        Ok(())
    }

    #[cfg(feature = "heterogeneous-test-faults")]
    pub const fn last_fault_drained(&self) -> bool {
        self.last_fault_drained
    }

    /// Upload actual CPU/GPU1 result rows and download the canonical arena into
    /// the no-longer-needed remote-input lease for bounded evidence.
    pub fn upload_results(
        &mut self,
        plan: &PackedDispatchPlan,
        reservation: &mut RelayPinnedReservation,
        timeline: Option<&CorrelatedTimeline>,
    ) -> Result<ResultRelayExecution> {
        let route_count = plan.rows as usize * GPT_OSS_TOP_K;
        if route_count > self.max_routes
            || reservation.remote_gpu_input.as_slice().len() < route_count * GPT_OSS_HIDDEN_SIZE
        {
            return Err(LLMError::GpuError(
                "result relay reservation is smaller than canonical arena".into(),
            ));
        }
        #[cfg(feature = "heterogeneous-test-faults")]
        let injected_fault = self.injected_fault.take();
        let submitted = (|| -> Result<(usize, usize, cudarc::driver::CudaEvent)> {
            self.stream
                .memset_zeros(&mut self.contribution_arena)
                .map_err(cuda_error("result relay arena clear"))?;
            if let Some(timeline) = timeline {
                timeline.enqueue_cuda_marker(&self.stream, "gpu0_relay", "result_h2d_begin")?;
            }
            let mut cpu_bytes = 0;
            let mut remote_bytes = 0;
            #[cfg(feature = "heterogeneous-test-faults")]
            let mut enqueued = 0;
            for (owners, source, bytes) in [
                (&plan.cpu, reservation.cpu_result.as_slice(), &mut cpu_bytes),
                (
                    &plan.remote_gpu,
                    reservation.remote_gpu_result.as_slice(),
                    &mut remote_bytes,
                ),
            ] {
                for owner in owners {
                    for route in &owner.routes {
                        let source_start = route.owner_route_slot as usize * GPT_OSS_HIDDEN_SIZE;
                        let destination_start =
                            route.descriptor.canonical_result_slot as usize * GPT_OSS_HIDDEN_SIZE;
                        self.stream
                            .memcpy_htod(
                                &source[source_start..source_start + GPT_OSS_HIDDEN_SIZE],
                                &mut self.contribution_arena.slice_mut(
                                    destination_start..destination_start + GPT_OSS_HIDDEN_SIZE,
                                ),
                            )
                            .map_err(cuda_error("result relay contribution H2D"))?;
                        *bytes += GPT_OSS_HIDDEN_SIZE * size_of::<u16>();
                        #[cfg(feature = "heterogeneous-test-faults")]
                        {
                            enqueued += 1;
                            if injected_fault
                                == Some(ResultRelayInjectedFault::AfterFirstResultEnqueue)
                                && enqueued == 1
                            {
                                return Err(LLMError::GpuError(
                                    "injected result relay post-enqueue failure".into(),
                                ));
                            }
                        }
                    }
                }
            }
            if let Some(timeline) = timeline {
                timeline.enqueue_cuda_marker(&self.stream, "gpu0_relay", "result_h2d_end")?;
            }
            self.stream
                .memcpy_dtoh(
                    &self
                        .contribution_arena
                        .slice(..route_count * GPT_OSS_HIDDEN_SIZE),
                    &mut reservation.remote_gpu_input.as_mut_slice()
                        [..route_count * GPT_OSS_HIDDEN_SIZE],
                )
                .map_err(cuda_error("result relay evidence D2H"))?;
            let terminal = self
                .stream
                .record_event(Some(CUevent_flags::CU_EVENT_DEFAULT))
                .map_err(cuda_error("result relay terminal event"))?;
            Ok((cpu_bytes, remote_bytes, terminal))
        })();
        let (cpu_h2d_bytes, remote_gpu_h2d_bytes, terminal) = match submitted {
            Ok(value) => value,
            Err(primary) => {
                let drained = self.stream.synchronize();
                #[cfg(feature = "heterogeneous-test-faults")]
                if injected_fault == Some(ResultRelayInjectedFault::AfterFirstResultEnqueue)
                    && drained.is_ok()
                {
                    self.last_fault_drained = true;
                }
                if let Err(drain) = drained {
                    return Err(LLMError::GpuError(format!(
                        "result relay submit failed ({primary}); mandatory drain failed ({drain})"
                    )));
                }
                return Err(primary);
            }
        };
        if let Err(error) = terminal.synchronize() {
            let primary = cuda_error("result relay terminal drain")(error);
            return match self.stream.synchronize() {
                Ok(()) => Err(primary),
                Err(drain) => Err(LLMError::GpuError(format!(
                    "result relay terminal failed ({primary}); mandatory drain failed ({drain})"
                ))),
            };
        }
        Ok(ResultRelayExecution {
            cpu_h2d_bytes,
            remote_gpu_h2d_bytes,
            evidence_d2h_bytes: route_count * GPT_OSS_HIDDEN_SIZE * size_of::<u16>(),
        })
    }
}

pub fn fixed_relay_byte_plan(max_rows: usize) -> Result<RelayBytePlan> {
    if max_rows == 0 || max_rows > H4_PREFILL_MAX_ROWS {
        return Err(LLMError::MemoryError(
            "fixed relay byte plan rows outside 1..=64".into(),
        ));
    }
    let row_bytes = GPT_OSS_HIDDEN_SIZE * size_of::<u16>();
    let source_activation_capacity = max_rows * row_bytes;
    let route_descriptor_capacity = max_rows * GPT_OSS_TOP_K * H4_ROUTE_DESCRIPTOR_MAX_BYTES;
    let route_arena = max_rows * GPT_OSS_TOP_K * row_bytes;
    let raw_pinned_bytes = source_activation_capacity + route_descriptor_capacity + route_arena * 3;
    let hard_cap_bytes = if max_rows == 1 {
        H4_DECODE_PINNED_CAP_BYTES
    } else {
        H4_PREFILL_PINNED_CAP_BYTES
    };
    Ok(RelayBytePlan {
        source_activation_d2h: source_activation_capacity,
        route_descriptor_d2h: route_descriptor_capacity,
        remote_gpu_h2d: 0,
        remote_gpu_d2h: 0,
        cpu_result_bytes: 0,
        source_activation_capacity,
        route_descriptor_capacity,
        remote_gpu_input_capacity: route_arena,
        remote_gpu_result_capacity: route_arena,
        cpu_result_capacity: route_arena,
        raw_pinned_bytes,
        hard_cap_bytes,
    })
}

fn cuda_error(stage: &'static str) -> impl FnOnce(cudarc::driver::DriverError) -> LLMError {
    move |error| LLMError::GpuError(format!("{stage}: {error}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixed_five_buffer_plan_matches_phase_2_arithmetic() {
        let decode = fixed_relay_byte_plan(1).unwrap();
        assert_eq!(decode.raw_pinned_bytes, 74_944);
        assert_eq!(decode.hard_cap_bytes, 128 * 1024);
        let prefill = fixed_relay_byte_plan(64).unwrap();
        assert_eq!(prefill.raw_pinned_bytes, 4_796_416);
        assert_eq!(prefill.hard_cap_bytes, 8 * 1024 * 1024);
    }
}
