//! Globally correlated host/CUDA stream markers for heterogeneous evidence.

use std::sync::Arc;
use std::time::Instant;

use parking_lot::Mutex;
use serde::{Deserialize, Serialize};

#[cfg(feature = "cuda")]
use crate::{LLMError, Result};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TimelinePoint {
    pub label: String,
    pub actor: String,
    pub monotonic_ns: u64,
}

struct TimelineInner {
    origin: Instant,
    points: Mutex<Vec<TimelinePoint>>,
}

/// A single process-monotonic clock shared by CPU workers and CUDA host
/// callbacks on every device context.
#[derive(Clone)]
pub struct CorrelatedTimeline {
    inner: Arc<TimelineInner>,
}

impl CorrelatedTimeline {
    pub fn new() -> Self {
        Self {
            inner: Arc::new(TimelineInner {
                origin: Instant::now(),
                points: Mutex::new(Vec::new()),
            }),
        }
    }

    pub fn record_host(&self, actor: impl Into<String>, label: impl Into<String>) {
        record_point(&self.inner, actor.into(), label.into());
    }

    pub fn points(&self) -> Vec<TimelinePoint> {
        let mut points = self.inner.points.lock().clone();
        points.sort_by_key(|point| point.monotonic_ns);
        points
    }

    #[cfg(feature = "cuda")]
    pub fn enqueue_cuda_marker(
        &self,
        stream: &cudarc::driver::CudaStream,
        actor: impl Into<String>,
        label: impl Into<String>,
    ) -> Result<()> {
        stream
            .context()
            .bind_to_thread()
            .map_err(|error| LLMError::GpuError(format!("timeline context bind: {error}")))?;
        let payload = Box::new(CudaMarkerPayload {
            timeline: Arc::clone(&self.inner),
            actor: actor.into(),
            label: label.into(),
        });
        let payload = Box::into_raw(payload);
        // SAFETY: CUDA invokes the callback exactly once after all prior work
        // in this stream. The boxed payload owns its timeline and strings until
        // the callback reclaims it. On launch failure, this function reclaims
        // the box before returning.
        let result = unsafe {
            cudarc::driver::sys::cuLaunchHostFunc(
                stream.cu_stream(),
                Some(cuda_marker_callback),
                payload.cast(),
            )
        };
        if result != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
            // SAFETY: the callback was not accepted, so ownership remains here.
            unsafe {
                drop(Box::from_raw(payload));
            }
            return Err(LLMError::GpuError(format!(
                "CUDA timeline host marker enqueue failed: {result:?}"
            )));
        }
        Ok(())
    }
}

impl Default for CorrelatedTimeline {
    fn default() -> Self {
        Self::new()
    }
}

fn record_point(inner: &TimelineInner, actor: String, label: String) {
    let elapsed = inner.origin.elapsed().as_nanos();
    inner.points.lock().push(TimelinePoint {
        label,
        actor,
        monotonic_ns: u64::try_from(elapsed).unwrap_or(u64::MAX),
    });
}

#[cfg(feature = "cuda")]
struct CudaMarkerPayload {
    timeline: Arc<TimelineInner>,
    actor: String,
    label: String,
}

#[cfg(feature = "cuda")]
unsafe extern "C" fn cuda_marker_callback(payload: *mut std::ffi::c_void) {
    // SAFETY: `enqueue_cuda_marker` passes exactly one Box allocation and CUDA
    // invokes this callback at most once for an accepted marker.
    let payload = unsafe { Box::from_raw(payload.cast::<CudaMarkerPayload>()) };
    record_point(
        &payload.timeline,
        payload.actor.clone(),
        payload.label.clone(),
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn host_points_share_one_monotonic_order() {
        let timeline = CorrelatedTimeline::new();
        timeline.record_host("cpu", "begin");
        timeline.record_host("coordinator", "end");
        let points = timeline.points();
        assert_eq!(points.len(), 2);
        assert_eq!(points[0].label, "begin");
        assert_eq!(points[1].label, "end");
        assert!(points[0].monotonic_ns <= points[1].monotonic_ns);
    }
}
