//! Stable lifecycle, failure, committed-event, lease, and delivery contracts.

use gpt_oss_core::prelude::{FinishReason, LogProb, RequestId, TokenId};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use tokio::sync::{mpsc, watch, Notify};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ServiceState {
    Starting,
    Ready,
    Draining,
    Failed,
    Stopped,
}
impl ServiceState {
    pub const fn is_ready(self) -> bool {
        matches!(self, Self::Ready)
    }

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Starting => "starting",
            Self::Ready => "ready",
            Self::Draining => "draining",
            Self::Failed => "failed",
            Self::Stopped => "stopped",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FailurePhase {
    Envelope,
    Validation,
    Tokenization,
    Admission,
    Queue,
    Execution,
    Commit,
    Delivery,
    Storage,
    Startup,
    Shutdown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StableFailureCode {
    InvalidRequest,
    BodyTooLarge,
    ContextLimit,
    ModelNotFound,
    StoredResponseNotFound,
    UnsupportedOption,
    OverloadedRequests,
    OverloadedTokens,
    OverloadedMemory,
    OverloadedDelivery,
    NotReady,
    Draining,
    EngineFailed,
    OwnerStopped,
    Shutdown,
    ExecutionFailed,
    SerializationFailed,
    ClientCancelled,
    SlowConsumer,
    DeliveryFailed,
}

impl StableFailureCode {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::InvalidRequest => "invalid_request",
            Self::BodyTooLarge => "body_too_large",
            Self::ContextLimit => "context_limit",
            Self::ModelNotFound => "model_not_found",
            Self::StoredResponseNotFound => "stored_response_not_found",
            Self::UnsupportedOption => "unsupported_option",
            Self::OverloadedRequests => "overloaded_requests",
            Self::OverloadedTokens => "overloaded_tokens",
            Self::OverloadedMemory => "overloaded_memory",
            Self::OverloadedDelivery => "overloaded_delivery",
            Self::NotReady => "not_ready",
            Self::Draining => "draining",
            Self::EngineFailed => "engine_failed",
            Self::OwnerStopped => "owner_stopped",
            Self::Shutdown => "shutdown",
            Self::ExecutionFailed => "execution_failed",
            Self::SerializationFailed => "serialization_failed",
            Self::ClientCancelled => "client_cancelled",
            Self::SlowConsumer => "slow_consumer",
            Self::DeliveryFailed => "delivery_failed",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, thiserror::Error)]
#[error("{code:?} during {phase:?}: {message}")]
pub struct StableFailure {
    pub code: StableFailureCode,
    pub phase: FailurePhase,
    pub retryable: bool,
    pub message: String,
}

impl StableFailure {
    pub fn new(
        code: StableFailureCode,
        phase: FailurePhase,
        retryable: bool,
        message: impl Into<String>,
    ) -> Self {
        Self {
            code,
            phase,
            retryable,
            message: message.into(),
        }
    }

    pub fn unavailable(state: ServiceState) -> Self {
        match state {
            ServiceState::Starting => Self::new(
                StableFailureCode::NotReady,
                FailurePhase::Admission,
                true,
                "service is still starting",
            ),
            ServiceState::Draining => Self::new(
                StableFailureCode::Draining,
                FailurePhase::Admission,
                true,
                "service is draining",
            ),
            ServiceState::Failed => Self::new(
                StableFailureCode::EngineFailed,
                FailurePhase::Admission,
                false,
                "engine owner failed",
            ),
            ServiceState::Stopped => Self::new(
                StableFailureCode::OwnerStopped,
                FailurePhase::Admission,
                false,
                "engine owner stopped",
            ),
            ServiceState::Ready => Self::new(
                StableFailureCode::InvalidRequest,
                FailurePhase::Admission,
                false,
                "service is ready",
            ),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum CommittedEvent {
    Delta {
        choice: u32,
        text: String,
        token_ids: Vec<TokenId>,
        #[serde(skip_serializing_if = "Option::is_none")]
        logprobs: Option<Vec<Vec<(TokenId, LogProb)>>>,
    },
    Usage {
        committed_prompt: u64,
        committed_completion: u64,
    },
    Finish {
        choice: u32,
        reason: FinishReason,
    },
    Error {
        failure: StableFailure,
    },
    Done,
}

impl CommittedEvent {
    pub const fn is_terminal_control(&self) -> bool {
        matches!(
            self,
            Self::Usage { .. } | Self::Finish { .. } | Self::Error { .. } | Self::Done
        )
    }

    pub fn serialized_bytes(&self) -> Result<usize, StableFailure> {
        serde_json::to_vec(self)
            .map(|bytes| bytes.len())
            .map_err(|error| {
                StableFailure::new(
                    StableFailureCode::SerializationFailed,
                    FailurePhase::Delivery,
                    false,
                    error.to_string(),
                )
            })
    }
}

struct LeaseInner {
    request_id: RequestId,
    cancelled: AtomicBool,
    cancel_tx: mpsc::UnboundedSender<RequestId>,
}

/// Cancellation capability associated with one admitted request. All clones
/// share one idempotence bit; dropping the last clone cancels unfinished work.
#[derive(Clone)]
pub struct RequestLease {
    inner: Arc<LeaseInner>,
}

impl std::fmt::Debug for RequestLease {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("RequestLease")
            .field("request_id", &self.request_id())
            .field("cancelled", &self.is_cancelled())
            .finish()
    }
}

impl RequestLease {
    pub fn new(request_id: RequestId, cancel_tx: mpsc::UnboundedSender<RequestId>) -> Self {
        Self {
            inner: Arc::new(LeaseInner {
                request_id,
                cancelled: AtomicBool::new(false),
                cancel_tx,
            }),
        }
    }

    pub fn request_id(&self) -> RequestId {
        self.inner.request_id
    }

    pub fn is_cancelled(&self) -> bool {
        self.inner.cancelled.load(Ordering::Acquire)
    }

    pub fn cancel(&self) -> bool {
        if self
            .inner
            .cancelled
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .is_ok()
        {
            let _ = self.inner.cancel_tx.send(self.inner.request_id);
            true
        } else {
            false
        }
    }

    /// Mark terminal completion so dropping route state does not send a late cancellation.
    pub fn complete(&self) -> bool {
        self.inner
            .cancelled
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .is_ok()
    }
}

impl Drop for RequestLease {
    fn drop(&mut self) {
        if Arc::strong_count(&self.inner) == 1 {
            self.cancel();
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ServiceStatus {
    pub state: ServiceState,
    pub served_model_id: String,
    pub runtime_snapshot_sha256: Option<String>,
    pub failure: Option<StableFailure>,
}

#[derive(Clone, Debug)]
pub struct ServiceLifecycle {
    tx: watch::Sender<ServiceStatus>,
}

impl ServiceLifecycle {
    pub fn starting(served_model_id: impl Into<String>) -> Self {
        let (tx, _) = watch::channel(ServiceStatus {
            state: ServiceState::Starting,
            served_model_id: served_model_id.into(),
            runtime_snapshot_sha256: None,
            failure: None,
        });
        crate::telemetry::metrics::record_service_state(ServiceState::Starting);
        Self { tx }
    }

    pub fn subscribe(&self) -> watch::Receiver<ServiceStatus> {
        self.tx.subscribe()
    }

    pub fn status(&self) -> ServiceStatus {
        self.tx.borrow().clone()
    }

    pub fn mark_ready(&self, runtime_snapshot_sha256: String) -> Result<(), StableFailure> {
        if runtime_snapshot_sha256.len() != 64
            || !runtime_snapshot_sha256
                .bytes()
                .all(|byte| byte.is_ascii_hexdigit())
        {
            return Err(StableFailure::new(
                StableFailureCode::InvalidRequest,
                FailurePhase::Startup,
                false,
                "runtime snapshot hash must be 64 hexadecimal characters",
            ));
        }
        self.transition(ServiceState::Ready, Some(runtime_snapshot_sha256), None)
    }

    pub fn begin_draining(&self) -> Result<(), StableFailure> {
        self.transition(ServiceState::Draining, None, None)
    }

    pub fn mark_failed(&self, failure: StableFailure) -> Result<(), StableFailure> {
        self.transition(ServiceState::Failed, None, Some(failure))
    }

    pub fn mark_stopped(&self) -> Result<(), StableFailure> {
        self.transition(ServiceState::Stopped, None, None)
    }

    fn transition(
        &self,
        target: ServiceState,
        snapshot_hash: Option<String>,
        failure: Option<StableFailure>,
    ) -> Result<(), StableFailure> {
        let current = self.status();
        let valid = matches!(
            (current.state, target),
            (ServiceState::Starting, ServiceState::Ready)
                | (ServiceState::Starting, ServiceState::Failed)
                | (ServiceState::Starting, ServiceState::Stopped)
                | (ServiceState::Ready, ServiceState::Draining)
                | (ServiceState::Ready, ServiceState::Failed)
                | (ServiceState::Draining, ServiceState::Failed)
                | (ServiceState::Draining, ServiceState::Stopped)
                | (ServiceState::Failed, ServiceState::Stopped)
        );
        if !valid {
            return Err(StableFailure::new(
                StableFailureCode::InvalidRequest,
                FailurePhase::Shutdown,
                false,
                format!(
                    "invalid service transition {:?} -> {:?}",
                    current.state, target
                ),
            ));
        }
        self.tx.send_modify(|status| {
            status.state = target;
            if let Some(hash) = snapshot_hash {
                status.runtime_snapshot_sha256 = Some(hash);
            }
            if failure.is_some() {
                status.failure = failure;
            }
        });
        crate::telemetry::metrics::record_service_state(target);
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeliveryLimits {
    pub per_request_queued_bytes: usize,
    pub terminal_control_allowance: usize,
    pub global_queued_bytes: usize,
    pub max_event_bytes: usize,
}

impl Default for DeliveryLimits {
    fn default() -> Self {
        Self {
            per_request_queued_bytes: 1024 * 1024,
            terminal_control_allowance: 16 * 1024,
            global_queued_bytes: 1024 * 1024,
            max_event_bytes: 256 * 1024,
        }
    }
}

#[derive(Debug)]
pub struct GlobalDeliveryBudget {
    limit: usize,
    queued: AtomicUsize,
}

impl GlobalDeliveryBudget {
    pub fn new(limit: usize) -> Arc<Self> {
        Arc::new(Self {
            limit,
            queued: AtomicUsize::new(0),
        })
    }

    pub fn queued_bytes(&self) -> usize {
        self.queued.load(Ordering::Acquire)
    }

    fn try_charge(&self, bytes: usize) -> bool {
        let mut current = self.queued.load(Ordering::Acquire);
        loop {
            let Some(next) = current.checked_add(bytes) else {
                return false;
            };
            if next > self.limit {
                return false;
            }
            match self.queued.compare_exchange_weak(
                current,
                next,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => return true,
                Err(observed) => current = observed,
            }
        }
    }

    fn release(&self, bytes: usize) {
        let previous = self.queued.fetch_sub(bytes, Ordering::AcqRel);
        debug_assert!(previous >= bytes);
    }
}

#[derive(Debug)]
struct QueuedEvent {
    event: CommittedEvent,
    bytes: usize,
}

#[derive(Debug, Default)]
struct DeliveryQueue {
    events: VecDeque<QueuedEvent>,
    queued_bytes: usize,
    abandoned: bool,
    terminal_enqueued: bool,
}

#[derive(Debug)]
struct DeliveryShared {
    limits: DeliveryLimits,
    global: Arc<GlobalDeliveryBudget>,
    queue: Mutex<DeliveryQueue>,
    notify: Notify,
    lease: RequestLease,
    publishers: AtomicUsize,
}

#[derive(Debug)]
pub struct DeliveryPublisher {
    shared: Arc<DeliveryShared>,
}

#[derive(Debug)]
pub struct DeliveryReceiver {
    shared: Arc<DeliveryShared>,
    terminal_seen: bool,
}

pub fn delivery_session(
    limits: DeliveryLimits,
    global: Arc<GlobalDeliveryBudget>,
    lease: RequestLease,
) -> Result<(DeliveryPublisher, DeliveryReceiver), StableFailure> {
    if limits.terminal_control_allowance > limits.per_request_queued_bytes
        || limits.max_event_bytes == 0
        || limits.global_queued_bytes != global.limit
    {
        return Err(StableFailure::new(
            StableFailureCode::InvalidRequest,
            FailurePhase::Startup,
            false,
            "invalid delivery limits",
        ));
    }
    let shared = Arc::new(DeliveryShared {
        limits,
        global,
        queue: Mutex::new(DeliveryQueue::default()),
        notify: Notify::new(),
        lease,
        publishers: AtomicUsize::new(1),
    });
    Ok((
        DeliveryPublisher {
            shared: shared.clone(),
        },
        DeliveryReceiver {
            shared,
            terminal_seen: false,
        },
    ))
}

impl DeliveryPublisher {
    /// Nonblocking publication. Adjacent text for one choice is coalesced.
    pub fn try_publish(&self, event: CommittedEvent) -> Result<(), StableFailure> {
        let bytes = event.serialized_bytes()?;
        if bytes > self.shared.limits.max_event_bytes {
            return Err(self.abandon("serialized stream event exceeds limit"));
        }
        let terminal = event.is_terminal_control();
        let ordinary_limit = self
            .shared
            .limits
            .per_request_queued_bytes
            .saturating_sub(self.shared.limits.terminal_control_allowance);
        let mut queue = self.shared.queue.lock().expect("delivery queue poisoned");
        if queue.abandoned {
            return Err(StableFailure::new(
                StableFailureCode::DeliveryFailed,
                FailurePhase::Delivery,
                false,
                "delivery session is abandoned",
            ));
        }

        if let (
            Some(QueuedEvent {
                event:
                    CommittedEvent::Delta {
                        choice: queued_choice,
                        text: queued_text,
                        token_ids: queued_tokens,
                        logprobs: queued_logprobs,
                    },
                bytes: queued_bytes,
            }),
            CommittedEvent::Delta {
                choice,
                text,
                token_ids,
                logprobs,
            },
        ) = (queue.events.back_mut(), &event)
        {
            if *queued_choice == *choice {
                let mut combined = CommittedEvent::Delta {
                    choice: *choice,
                    text: format!("{queued_text}{text}"),
                    token_ids: queued_tokens.iter().chain(token_ids).copied().collect(),
                    logprobs: merge_logprobs(queued_logprobs.as_ref(), logprobs.as_ref()),
                };
                let combined_bytes = combined.serialized_bytes()?;
                let extra = combined_bytes.saturating_sub(*queued_bytes);
                if combined_bytes <= self.shared.limits.max_event_bytes
                    && queue
                        .queued_bytes
                        .checked_add(extra)
                        .is_some_and(|total| total <= ordinary_limit)
                    && self.shared.global.try_charge(extra)
                {
                    if let Some(back) = queue.events.back_mut() {
                        std::mem::swap(&mut back.event, &mut combined);
                        back.bytes = combined_bytes;
                    }
                    queue.queued_bytes += extra;
                    metrics::counter!(
                        crate::telemetry::metrics::DELIVERY_COALESCES_TOTAL,
                        "backend" => crate::telemetry::metrics::BackendClass::Cpu.as_str(),
                        "result" => crate::telemetry::metrics::ResultClass::Accepted.as_str()
                    )
                    .increment(1);
                    metrics::counter!(
                        crate::telemetry::metrics::DELIVERY_BYTES_TOTAL,
                        "backend" => crate::telemetry::metrics::BackendClass::Cpu.as_str(),
                        "result" => crate::telemetry::metrics::ResultClass::Accepted.as_str()
                    )
                    .increment(extra as u64);
                    return Ok(());
                }
            }
        }

        let request_limit = if terminal {
            self.shared.limits.per_request_queued_bytes
        } else {
            ordinary_limit
        };
        if queue
            .queued_bytes
            .checked_add(bytes)
            .is_none_or(|total| total > request_limit)
            || !self.shared.global.try_charge(bytes)
        {
            drop(queue);
            return Err(self.abandon("delivery byte budget exhausted"));
        }
        queue.queued_bytes += bytes;
        queue.terminal_enqueued |=
            matches!(event, CommittedEvent::Done | CommittedEvent::Error { .. });
        queue.events.push_back(QueuedEvent { event, bytes });
        metrics::counter!(
            crate::telemetry::metrics::DELIVERY_BYTES_TOTAL,
            "backend" => crate::telemetry::metrics::BackendClass::Cpu.as_str(),
            "result" => crate::telemetry::metrics::ResultClass::Accepted.as_str()
        )
        .increment(bytes as u64);
        drop(queue);
        self.shared.notify.notify_one();
        Ok(())
    }

    pub fn is_abandoned(&self) -> bool {
        self.shared
            .queue
            .lock()
            .expect("delivery queue poisoned")
            .abandoned
    }

    fn abandon(&self, message: &str) -> StableFailure {
        let mut queue = self.shared.queue.lock().expect("delivery queue poisoned");
        if !queue.abandoned {
            queue.abandoned = true;
            let bytes = queue.queued_bytes;
            queue.queued_bytes = 0;
            queue.events.clear();
            metrics::counter!(
                crate::telemetry::metrics::DELIVERY_BYTES_TOTAL,
                "backend" => crate::telemetry::metrics::BackendClass::Cpu.as_str(),
                "result" => crate::telemetry::metrics::ResultClass::Abandoned.as_str()
            )
            .increment(bytes as u64);
            self.shared.global.release(bytes);
            self.shared.lease.cancel();
            self.shared.notify.notify_waiters();
        }
        StableFailure::new(
            StableFailureCode::SlowConsumer,
            FailurePhase::Delivery,
            false,
            message,
        )
    }
}

impl Clone for DeliveryPublisher {
    fn clone(&self) -> Self {
        self.shared.publishers.fetch_add(1, Ordering::AcqRel);
        Self {
            shared: self.shared.clone(),
        }
    }
}

impl Drop for DeliveryPublisher {
    fn drop(&mut self) {
        let previous = self.shared.publishers.fetch_sub(1, Ordering::AcqRel);
        debug_assert!(previous > 0);
        if previous == 1 {
            self.shared.notify.notify_waiters();
        }
    }
}

impl DeliveryReceiver {
    pub async fn recv(&mut self) -> Option<CommittedEvent> {
        loop {
            let notified = self.shared.notify.notified();
            {
                let mut queue = self.shared.queue.lock().expect("delivery queue poisoned");
                if let Some(queued) = queue.events.pop_front() {
                    queue.queued_bytes -= queued.bytes;
                    self.shared.global.release(queued.bytes);
                    self.terminal_seen |= matches!(
                        queued.event,
                        CommittedEvent::Done | CommittedEvent::Error { .. }
                    );
                    if self.terminal_seen {
                        self.shared.lease.complete();
                    }
                    if let CommittedEvent::Delta { token_ids, .. } = &queued.event {
                        metrics::counter!(
                            crate::telemetry::metrics::TOKENS_TOTAL,
                            "backend" => crate::telemetry::metrics::BackendClass::Cpu.as_str(),
                            "kind" => crate::telemetry::metrics::TokenClass::Delivered.as_str()
                        )
                        .increment(token_ids.len() as u64);
                    }
                    return Some(queued.event);
                }
                if queue.abandoned
                    || (queue.terminal_enqueued && self.terminal_seen)
                    || self.shared.publishers.load(Ordering::Acquire) == 0
                {
                    return None;
                }
            }
            notified.await;
        }
    }
}

impl Drop for DeliveryReceiver {
    fn drop(&mut self) {
        if self.terminal_seen {
            return;
        }
        let mut queue = self.shared.queue.lock().expect("delivery queue poisoned");
        if !queue.abandoned {
            queue.abandoned = true;
            let bytes = queue.queued_bytes;
            queue.queued_bytes = 0;
            queue.events.clear();
            metrics::counter!(
                crate::telemetry::metrics::DELIVERY_BYTES_TOTAL,
                "backend" => crate::telemetry::metrics::BackendClass::Cpu.as_str(),
                "result" => crate::telemetry::metrics::ResultClass::Abandoned.as_str()
            )
            .increment(bytes as u64);
            self.shared.global.release(bytes);
            self.shared.lease.cancel();
            self.shared.notify.notify_waiters();
        }
    }
}

fn merge_logprobs(
    left: Option<&Vec<Vec<(TokenId, LogProb)>>>,
    right: Option<&Vec<Vec<(TokenId, LogProb)>>>,
) -> Option<Vec<Vec<(TokenId, LogProb)>>> {
    match (left, right) {
        (None, None) => None,
        (left, right) => Some(
            left.into_iter()
                .flatten()
                .chain(right.into_iter().flatten())
                .cloned()
                .collect(),
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lifecycle_requires_snapshot_before_ready_and_rejects_invalid_transitions() {
        let lifecycle = ServiceLifecycle::starting("served-model");
        assert_eq!(lifecycle.status().state, ServiceState::Starting);
        assert!(lifecycle.mark_ready("bad".into()).is_err());
        lifecycle.mark_ready("a".repeat(64)).unwrap();
        assert_eq!(lifecycle.status().state, ServiceState::Ready);
        assert!(lifecycle.mark_ready("b".repeat(64)).is_err());
        lifecycle.begin_draining().unwrap();
        lifecycle.mark_stopped().unwrap();
    }

    #[test]
    fn lease_cancellation_is_idempotent_across_clones_and_drop() {
        let (tx, mut rx) = mpsc::unbounded_channel();
        let lease = RequestLease::new(RequestId(7), tx);
        let clone = lease.clone();
        assert!(lease.cancel());
        assert!(!clone.cancel());
        drop(lease);
        drop(clone);
        assert_eq!(rx.try_recv(), Ok(RequestId(7)));
        assert!(rx.try_recv().is_err());
    }

    #[tokio::test]
    async fn delivery_coalesces_text_and_preserves_control_boundaries() {
        let (cancel_tx, _cancel_rx) = mpsc::unbounded_channel();
        let lease = RequestLease::new(RequestId(1), cancel_tx);
        let limits = DeliveryLimits {
            per_request_queued_bytes: 4096,
            terminal_control_allowance: 1024,
            global_queued_bytes: 4096,
            max_event_bytes: 2048,
        };
        let global = GlobalDeliveryBudget::new(limits.global_queued_bytes);
        let (publisher, mut receiver) = delivery_session(limits, global.clone(), lease).unwrap();
        for text in ["a", "b"] {
            publisher
                .try_publish(CommittedEvent::Delta {
                    choice: 0,
                    text: text.into(),
                    token_ids: vec![],
                    logprobs: None,
                })
                .unwrap();
        }
        publisher
            .try_publish(CommittedEvent::Finish {
                choice: 0,
                reason: FinishReason::Stop,
            })
            .unwrap();
        publisher.try_publish(CommittedEvent::Done).unwrap();
        assert!(matches!(
            receiver.recv().await,
            Some(CommittedEvent::Delta { text, .. }) if text == "ab"
        ));
        assert!(matches!(
            receiver.recv().await,
            Some(CommittedEvent::Finish { .. })
        ));
        assert!(matches!(receiver.recv().await, Some(CommittedEvent::Done)));
        assert_eq!(global.queued_bytes(), 0);
    }

    #[tokio::test]
    async fn slow_consumer_abandons_only_its_request_and_cancels_owner() {
        let (cancel_tx, mut cancel_rx) = mpsc::unbounded_channel();
        let lease = RequestLease::new(RequestId(9), cancel_tx);
        let limits = DeliveryLimits {
            per_request_queued_bytes: 128,
            terminal_control_allowance: 64,
            global_queued_bytes: 128,
            max_event_bytes: 128,
        };
        let global = GlobalDeliveryBudget::new(128);
        let (publisher, _receiver) = delivery_session(limits, global.clone(), lease).unwrap();
        let result = publisher.try_publish(CommittedEvent::Delta {
            choice: 0,
            text: "x".repeat(80),
            token_ids: vec![],
            logprobs: None,
        });
        assert_eq!(result.unwrap_err().code, StableFailureCode::SlowConsumer);
        assert_eq!(cancel_rx.recv().await, Some(RequestId(9)));
        assert_eq!(global.queued_bytes(), 0);
    }
}
