//! Managed native-CPU owner with bounded admission and byte-charged delivery.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use tokio::sync::{mpsc, oneshot, Mutex, OwnedSemaphorePermit, Semaphore};
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;
use tracing::{debug, error, info};

use gpt_oss_core::prelude::{RequestId, SamplingParams};

use crate::service::{
    delivery_session, CommittedEvent, DeliveryLimits, DeliveryPublisher, DeliveryReceiver,
    FailurePhase, GlobalDeliveryBudget, RequestLease, ServiceLifecycle, ServiceState,
    StableFailure, StableFailureCode,
};
use crate::telemetry::metrics::{
    self as service_metrics, BackendClass, Phase, ReasonCode, ResultClass, TokenClass,
};
use crate::CpuBatchEngine;

enum CpuEngineCommand {
    Generate {
        request_id: RequestId,
        prompt: String,
        prompt_token_ids: Vec<u32>,
        sampling_params: SamplingParams,
        delivery: DeliveryPublisher,
        permit: OwnedSemaphorePermit,
        response_tx: oneshot::Sender<std::result::Result<(), StableFailure>>,
    },
}

struct ActiveDelivery {
    publisher: DeliveryPublisher,
    _permit: OwnedSemaphorePermit,
}

/// One managed request. Dropping it before terminal delivery idempotently
/// notifies the canonical owner through its lease.
#[derive(Debug)]
pub struct ManagedRequest {
    pub request_id: RequestId,
    pub events: DeliveryReceiver,
    pub lease: RequestLease,
    pub lifecycle: ServiceLifecycle,
}

/// Awaitable native-CPU engine lifecycle.
pub struct AsyncCpuBatchEngine {
    cmd_tx: mpsc::Sender<CpuEngineCommand>,
    cancel_tx: mpsc::UnboundedSender<RequestId>,
    shutdown: CancellationToken,
    next_request_id: AtomicU64,
    admission: Arc<Semaphore>,
    admission_tokenizer: Arc<gpt_oss_tokenizer::Tokenizer>,
    delivery_limits: DeliveryLimits,
    global_delivery: Arc<GlobalDeliveryBudget>,
    lifecycle: ServiceLifecycle,
    owner: Mutex<Option<JoinHandle<std::result::Result<(), StableFailure>>>>,
}

impl AsyncCpuBatchEngine {
    /// Test/programmatic constructor. Production startup should use
    /// [`Self::with_service`] and mark the supplied lifecycle ready only after
    /// its effective runtime snapshot is frozen.
    pub fn new(engine: CpuBatchEngine) -> Self {
        let max_requests = engine.max_num_seqs();
        let lifecycle = ServiceLifecycle::starting("test-model");
        lifecycle
            .mark_ready("0".repeat(64))
            .expect("static test snapshot hash is valid");
        let mut limits = DeliveryLimits::default();
        limits.global_queued_bytes = limits.per_request_queued_bytes.saturating_mul(max_requests);
        Self::with_service(engine, lifecycle, limits, Duration::from_secs(30))
            .expect("default CPU service limits are valid")
    }

    pub fn with_service(
        engine: CpuBatchEngine,
        lifecycle: ServiceLifecycle,
        delivery_limits: DeliveryLimits,
        drain_deadline: Duration,
    ) -> std::result::Result<Self, StableFailure> {
        let max_requests = engine.max_num_seqs();
        let admission_tokenizer = Arc::new(engine.tokenizer_clone());
        if max_requests == 0 || drain_deadline.is_zero() {
            return Err(StableFailure::new(
                StableFailureCode::InvalidRequest,
                FailurePhase::Startup,
                false,
                "CPU service requires positive admission and drain limits",
            ));
        }
        let (cmd_tx, cmd_rx) = mpsc::channel(max_requests.max(1));
        let (cancel_tx, cancel_rx) = mpsc::unbounded_channel();
        let shutdown = CancellationToken::new();
        let global_delivery = GlobalDeliveryBudget::new(delivery_limits.global_queued_bytes);
        let owner_shutdown = shutdown.clone();
        let owner_lifecycle = lifecycle.clone();
        let worker = tokio::spawn(Self::background_loop(
            engine,
            cmd_rx,
            cancel_rx,
            owner_shutdown,
            drain_deadline,
        ));
        let owner = tokio::spawn(async move {
            let result = match worker.await {
                Ok(result) => result,
                Err(join_error) => Err(StableFailure::new(
                    StableFailureCode::EngineFailed,
                    FailurePhase::Execution,
                    false,
                    format!("native CPU owner task failed: {join_error}"),
                )),
            };
            if let Err(failure) = &result {
                let _ = owner_lifecycle.mark_failed(failure.clone());
            }
            result
        });
        Ok(Self {
            cmd_tx,
            cancel_tx,
            shutdown,
            next_request_id: AtomicU64::new(1),
            admission: Arc::new(Semaphore::new(max_requests)),
            admission_tokenizer,
            delivery_limits,
            global_delivery,
            lifecycle,
            owner: Mutex::new(Some(owner)),
        })
    }

    pub fn lifecycle(&self) -> &ServiceLifecycle {
        &self.lifecycle
    }

    pub async fn generate(
        &self,
        prompt: String,
        sampling_params: SamplingParams,
    ) -> std::result::Result<ManagedRequest, StableFailure> {
        let status = self.lifecycle.status();
        if status.state != ServiceState::Ready {
            return Err(StableFailure::unavailable(status.state));
        }
        let permit = self.admission.clone().try_acquire_owned().map_err(|_| {
            service_metrics::record_admission(
                BackendClass::Cpu,
                ResultClass::Rejected,
                ReasonCode::Failure(StableFailureCode::OverloadedRequests),
            );
            StableFailure::new(
                StableFailureCode::OverloadedRequests,
                FailurePhase::Admission,
                true,
                "native CPU request capacity is full",
            )
        })?;
        let tokenizer = self.admission_tokenizer.clone();
        let tokenization_started = Instant::now();
        let (prompt, prompt_token_ids) = tokio::task::spawn_blocking(move || {
            let prompt_token_ids = tokenizer.encode(&prompt)?;
            Ok::<_, gpt_oss_core::prelude::LLMError>((prompt, prompt_token_ids))
        })
        .await
        .map_err(|error| {
            StableFailure::new(
                StableFailureCode::EngineFailed,
                FailurePhase::Tokenization,
                false,
                format!("CPU tokenization task failed: {error}"),
            )
        })?
        .map_err(|error| {
            StableFailure::new(
                StableFailureCode::InvalidRequest,
                FailurePhase::Tokenization,
                false,
                error.to_string(),
            )
        })?;
        service_metrics::record_phase_duration(
            BackendClass::Cpu,
            Phase::Tokenization,
            ResultClass::Completed,
            tokenization_started.elapsed(),
        );
        service_metrics::record_tokens(
            BackendClass::Cpu,
            TokenClass::Prompt,
            prompt_token_ids.len(),
        );
        let request_id = RequestId(self.next_request_id.fetch_add(1, Ordering::Relaxed));
        let lease = RequestLease::new(request_id, self.cancel_tx.clone());
        let (publisher, events) = delivery_session(
            self.delivery_limits,
            self.global_delivery.clone(),
            lease.clone(),
        )?;
        let (response_tx, response_rx) = oneshot::channel();
        self.cmd_tx
            .try_send(CpuEngineCommand::Generate {
                request_id,
                prompt,
                prompt_token_ids,
                sampling_params,
                delivery: publisher,
                permit,
                response_tx,
            })
            .map_err(|error| {
                let full = matches!(error, mpsc::error::TrySendError::Full(_));
                let code = if full {
                    StableFailureCode::OverloadedRequests
                } else {
                    StableFailureCode::OwnerStopped
                };
                StableFailure::new(
                    code,
                    FailurePhase::Admission,
                    full,
                    "native CPU admission queue is unavailable",
                )
            })?;
        response_rx.await.map_err(|_| {
            StableFailure::new(
                StableFailureCode::OwnerStopped,
                FailurePhase::Admission,
                false,
                "native CPU admission response was dropped",
            )
        })??;
        service_metrics::record_admission(
            BackendClass::Cpu,
            ResultClass::Accepted,
            ReasonCode::None,
        );
        service_metrics::adjust_current_requests(BackendClass::Cpu, 1.0);
        Ok(ManagedRequest {
            request_id,
            events,
            lease,
            lifecycle: self.lifecycle.clone(),
        })
    }

    pub fn begin_shutdown(&self) -> std::result::Result<(), StableFailure> {
        let state = self.lifecycle.status().state;
        match state {
            ServiceState::Starting | ServiceState::Ready => {
                if state == ServiceState::Starting {
                    // A startup cancellation has no admitted work and moves
                    // directly through failure-safe owner shutdown.
                    let failure = StableFailure::new(
                        StableFailureCode::Shutdown,
                        FailurePhase::Shutdown,
                        false,
                        "shutdown requested during startup",
                    );
                    self.lifecycle.mark_failed(failure)?;
                } else {
                    self.lifecycle.begin_draining()?;
                }
                self.admission.close();
                self.shutdown.cancel();
                Ok(())
            }
            ServiceState::Draining | ServiceState::Stopped => Ok(()),
            ServiceState::Failed => {
                self.admission.close();
                self.shutdown.cancel();
                Ok(())
            }
        }
    }

    /// Drain (up to the configured deadline) and await the canonical owner.
    pub async fn shutdown(&self) -> std::result::Result<(), StableFailure> {
        self.begin_shutdown()?;
        self.wait().await
    }

    pub async fn wait(&self) -> std::result::Result<(), StableFailure> {
        let Some(owner) = self.owner.lock().await.take() else {
            return Ok(());
        };
        match owner.await {
            Ok(Ok(())) => {
                if self.lifecycle.status().state != ServiceState::Stopped {
                    let _ = self.lifecycle.mark_stopped();
                }
                Ok(())
            }
            Ok(Err(failure)) => Err(failure),
            Err(join_error) => {
                let failure = StableFailure::new(
                    StableFailureCode::EngineFailed,
                    FailurePhase::Execution,
                    false,
                    format!("native CPU owner task failed: {join_error}"),
                );
                let _ = self.lifecycle.mark_failed(failure.clone());
                Err(failure)
            }
        }
    }

    async fn background_loop(
        mut engine: CpuBatchEngine,
        mut cmd_rx: mpsc::Receiver<CpuEngineCommand>,
        mut cancel_rx: mpsc::UnboundedReceiver<RequestId>,
        shutdown: CancellationToken,
        drain_deadline: Duration,
    ) -> std::result::Result<(), StableFailure> {
        let mut deliveries = HashMap::<RequestId, ActiveDelivery>::new();
        let mut draining_since = None::<Instant>;

        loop {
            Self::drain_commands(
                &mut engine,
                &mut deliveries,
                &mut cmd_rx,
                draining_since.is_some(),
            );
            Self::drain_cancellations(&mut engine, &mut deliveries, &mut cancel_rx);

            if shutdown.is_cancelled() && draining_since.is_none() {
                draining_since = Some(Instant::now());
                cmd_rx.close();
            }
            if let Some(started) = draining_since {
                if !engine.has_unfinished() || started.elapsed() >= drain_deadline {
                    break;
                }
            }

            if !engine.has_unfinished() {
                tokio::select! {
                    _ = shutdown.cancelled(), if draining_since.is_none() => {
                        draining_since = Some(Instant::now());
                        cmd_rx.close();
                    }
                    Some(request_id) = cancel_rx.recv() => {
                        Self::cancel_one(&mut engine, &mut deliveries, request_id, StableFailure::new(
                            StableFailureCode::ClientCancelled,
                            FailurePhase::Delivery,
                            false,
                            "request lease was cancelled",
                        ));
                    }
                    command = cmd_rx.recv(), if draining_since.is_none() => {
                        if let Some(command) = command {
                            Self::process_command(&mut engine, &mut deliveries, command, false);
                        } else {
                            draining_since = Some(Instant::now());
                        }
                    }
                    else => tokio::task::yield_now().await,
                }
                continue;
            }

            let reservation = match engine.reserve() {
                Ok(reservation) => reservation,
                Err(error) => {
                    let failure = StableFailure::new(
                        StableFailureCode::ExecutionFailed,
                        FailurePhase::Queue,
                        false,
                        error.to_string(),
                    );
                    return Self::fail_owner(&mut engine, &mut deliveries, failure);
                }
            };
            let Some(reservation) = reservation else {
                tokio::task::yield_now().await;
                continue;
            };
            let execute_started = Instant::now();
            let prepared = tokio::task::block_in_place(|| engine.execute(reservation));
            service_metrics::record_phase_duration(
                BackendClass::Cpu,
                Phase::Execute,
                if prepared.is_ok() {
                    ResultClass::Completed
                } else {
                    ResultClass::Failed
                },
                execute_started.elapsed(),
            );

            // Commands and disconnects that occurred during the bounded kernel
            // slice become tombstones before commit validation.
            Self::drain_commands(
                &mut engine,
                &mut deliveries,
                &mut cmd_rx,
                draining_since.is_some(),
            );
            Self::drain_cancellations(&mut engine, &mut deliveries, &mut cancel_rx);

            let prepared = match prepared {
                Ok(prepared) => prepared,
                Err(error) => {
                    let failure = StableFailure::new(
                        StableFailureCode::ExecutionFailed,
                        FailurePhase::Execution,
                        false,
                        error.to_string(),
                    );
                    return Self::fail_owner(&mut engine, &mut deliveries, failure);
                }
            };
            let commit_started = Instant::now();
            let committed = match engine.commit(prepared) {
                Ok(committed) => committed,
                Err(error) => {
                    service_metrics::record_phase_duration(
                        BackendClass::Cpu,
                        Phase::Commit,
                        ResultClass::Failed,
                        commit_started.elapsed(),
                    );
                    let failure = StableFailure::new(
                        StableFailureCode::ExecutionFailed,
                        FailurePhase::Commit,
                        false,
                        error.to_string(),
                    );
                    return Self::fail_owner(&mut engine, &mut deliveries, failure);
                }
            };
            service_metrics::record_phase_duration(
                BackendClass::Cpu,
                Phase::Commit,
                ResultClass::Completed,
                commit_started.elapsed(),
            );

            for request_id in committed.cancelled_requests {
                if deliveries.remove(&request_id).is_some() {
                    service_metrics::adjust_current_requests(BackendClass::Cpu, -1.0);
                    service_metrics::record_terminal(
                        BackendClass::Cpu,
                        ResultClass::Cancelled,
                        ReasonCode::Failure(StableFailureCode::ClientCancelled),
                    );
                }
            }
            for (request_id, event) in committed.events {
                let committed_tokens = match &event {
                    CommittedEvent::Delta { token_ids, .. } => token_ids.len(),
                    _ => 0,
                };
                service_metrics::record_tokens(
                    BackendClass::Cpu,
                    TokenClass::Committed,
                    committed_tokens,
                );
                let terminal = matches!(event, CommittedEvent::Done | CommittedEvent::Error { .. });
                let terminal_outcome = match &event {
                    CommittedEvent::Done => Some((ResultClass::Completed, ReasonCode::None)),
                    CommittedEvent::Error { failure } => {
                        Some((ResultClass::Failed, ReasonCode::Failure(failure.code)))
                    }
                    _ => None,
                };
                let delivery_started = Instant::now();
                let failed = deliveries
                    .get(&request_id)
                    .is_some_and(|delivery| delivery.publisher.try_publish(event).is_err());
                service_metrics::record_phase_duration(
                    BackendClass::Cpu,
                    Phase::Delivery,
                    if failed {
                        ResultClass::Abandoned
                    } else {
                        ResultClass::Completed
                    },
                    delivery_started.elapsed(),
                );
                if failed {
                    service_metrics::record_tokens(
                        BackendClass::Cpu,
                        TokenClass::Abandoned,
                        committed_tokens,
                    );
                    debug!(%request_id, "CPU delivery budget exhausted after commit");
                    if let Err(error) = engine.cancel_request(request_id) {
                        let failure = StableFailure::new(
                            StableFailureCode::ExecutionFailed,
                            FailurePhase::Commit,
                            false,
                            error.to_string(),
                        );
                        return Self::fail_owner(&mut engine, &mut deliveries, failure);
                    }
                    if deliveries.remove(&request_id).is_some() {
                        service_metrics::adjust_current_requests(BackendClass::Cpu, -1.0);
                        service_metrics::record_terminal(
                            BackendClass::Cpu,
                            ResultClass::Abandoned,
                            ReasonCode::Failure(StableFailureCode::SlowConsumer),
                        );
                    }
                    continue;
                }
                if terminal && deliveries.remove(&request_id).is_some() {
                    service_metrics::adjust_current_requests(BackendClass::Cpu, -1.0);
                    if let Some((result, reason)) = terminal_outcome {
                        service_metrics::record_terminal(BackendClass::Cpu, result, reason);
                    }
                }
            }
            tokio::task::yield_now().await;
        }

        let shutdown_failure = StableFailure::new(
            StableFailureCode::Shutdown,
            FailurePhase::Shutdown,
            false,
            "native CPU service stopped before request completion",
        );
        let active_ids = deliveries.keys().copied().collect::<Vec<_>>();
        for request_id in active_ids {
            Self::cancel_one(
                &mut engine,
                &mut deliveries,
                request_id,
                shutdown_failure.clone(),
            );
        }
        engine.shutdown().map_err(|error| {
            StableFailure::new(
                StableFailureCode::ExecutionFailed,
                FailurePhase::Shutdown,
                false,
                error.to_string(),
            )
        })?;
        info!("AsyncCpuBatchEngine canonical owner exited");
        Ok(())
    }

    fn drain_commands(
        engine: &mut CpuBatchEngine,
        deliveries: &mut HashMap<RequestId, ActiveDelivery>,
        cmd_rx: &mut mpsc::Receiver<CpuEngineCommand>,
        draining: bool,
    ) {
        while let Ok(command) = cmd_rx.try_recv() {
            Self::process_command(engine, deliveries, command, draining);
        }
    }

    fn process_command(
        engine: &mut CpuBatchEngine,
        deliveries: &mut HashMap<RequestId, ActiveDelivery>,
        command: CpuEngineCommand,
        draining: bool,
    ) {
        match command {
            CpuEngineCommand::Generate {
                request_id,
                prompt,
                prompt_token_ids,
                sampling_params,
                delivery,
                permit,
                response_tx,
            } => {
                if draining {
                    let _ = response_tx.send(Err(StableFailure::new(
                        StableFailureCode::Draining,
                        FailurePhase::Admission,
                        true,
                        "service began draining before admission",
                    )));
                    return;
                }
                match engine.add_tokenized_request(
                    request_id,
                    prompt,
                    prompt_token_ids,
                    sampling_params,
                ) {
                    Ok(_) => {
                        deliveries.insert(
                            request_id,
                            ActiveDelivery {
                                publisher: delivery,
                                _permit: permit,
                            },
                        );
                        let _ = response_tx.send(Ok(()));
                    }
                    Err(error) => {
                        let failure = classify_admission_error(error.to_string());
                        let _ = response_tx.send(Err(failure));
                    }
                }
            }
        }
    }

    fn drain_cancellations(
        engine: &mut CpuBatchEngine,
        deliveries: &mut HashMap<RequestId, ActiveDelivery>,
        cancel_rx: &mut mpsc::UnboundedReceiver<RequestId>,
    ) {
        while let Ok(request_id) = cancel_rx.try_recv() {
            Self::cancel_one(
                engine,
                deliveries,
                request_id,
                StableFailure::new(
                    StableFailureCode::ClientCancelled,
                    FailurePhase::Delivery,
                    false,
                    "request lease was cancelled",
                ),
            );
        }
    }

    fn cancel_one(
        engine: &mut CpuBatchEngine,
        deliveries: &mut HashMap<RequestId, ActiveDelivery>,
        request_id: RequestId,
        failure: StableFailure,
    ) {
        if let Err(error) = engine.cancel_request(request_id) {
            error!(%request_id, %error, "failed to tombstone CPU request");
        }
        if let Some(delivery) = deliveries.remove(&request_id) {
            let failure_code = failure.code;
            let _ = delivery
                .publisher
                .try_publish(CommittedEvent::Error { failure });
            service_metrics::adjust_current_requests(BackendClass::Cpu, -1.0);
            let result = if matches!(
                failure_code,
                StableFailureCode::ClientCancelled | StableFailureCode::Shutdown
            ) {
                ResultClass::Cancelled
            } else {
                ResultClass::Failed
            };
            service_metrics::record_terminal(
                BackendClass::Cpu,
                result,
                ReasonCode::Failure(failure_code),
            );
        }
    }

    fn fail_owner(
        engine: &mut CpuBatchEngine,
        deliveries: &mut HashMap<RequestId, ActiveDelivery>,
        failure: StableFailure,
    ) -> std::result::Result<(), StableFailure> {
        metrics::counter!(
            crate::telemetry::metrics::OWNER_FAILURES_TOTAL,
            "backend" => "cpu",
            "reason" => failure.code.as_str()
        )
        .increment(1);
        let active = deliveries.keys().copied().collect::<Vec<_>>();
        for request_id in active {
            Self::cancel_one(engine, deliveries, request_id, failure.clone());
        }
        let _ = engine.shutdown();
        Err(failure)
    }
}

fn classify_admission_error(message: String) -> StableFailure {
    let (code, phase) = if message.contains("context cap") {
        (StableFailureCode::ContextLimit, FailurePhase::Admission)
    } else if message.contains("best-of") || message.contains("beam") {
        (
            StableFailureCode::UnsupportedOption,
            FailurePhase::Validation,
        )
    } else if message.contains("token") {
        (
            StableFailureCode::InvalidRequest,
            FailurePhase::Tokenization,
        )
    } else {
        (StableFailureCode::InvalidRequest, FailurePhase::Admission)
    };
    StableFailure::new(code, phase, false, message)
}
