//! Async delivery and cancellation owner for [`crate::CpuBatchEngine`].

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};

use tokio::sync::{mpsc, oneshot};
use tokio_stream::wrappers::ReceiverStream;
use tokio_util::sync::CancellationToken;
use tracing::{debug, error, info};

use gpt_oss_core::prelude::{LLMError, RequestId, RequestOutput, Result, SamplingParams};

use crate::CpuBatchEngine;

enum CpuEngineCommand {
    Generate {
        request_id: RequestId,
        prompt: String,
        sampling_params: SamplingParams,
        output_tx: mpsc::Sender<RequestOutput>,
        response_tx: oneshot::Sender<Result<()>>,
    },
    AddRequest {
        request_id: RequestId,
        prompt: String,
        sampling_params: SamplingParams,
        response_tx: oneshot::Sender<Result<()>>,
    },
    AbortRequest {
        request_id: RequestId,
    },
}

/// Async native-CPU engine using one background owner and one canonical table.
pub struct AsyncCpuBatchEngine {
    cmd_tx: mpsc::Sender<CpuEngineCommand>,
    cancel: CancellationToken,
    next_request_id: AtomicU64,
}

impl AsyncCpuBatchEngine {
    pub fn new(engine: CpuBatchEngine) -> Self {
        let (cmd_tx, cmd_rx) = mpsc::channel(256);
        let cancel = CancellationToken::new();
        let cancel_background = cancel.clone();
        tokio::spawn(async move {
            Self::background_loop(engine, cmd_rx, cancel_background).await;
        });
        Self {
            cmd_tx,
            cancel,
            next_request_id: AtomicU64::new(1),
        }
    }

    pub async fn generate(
        &self,
        prompt: String,
        sampling_params: SamplingParams,
    ) -> Result<(RequestId, ReceiverStream<RequestOutput>)> {
        let request_id = RequestId(self.next_request_id.fetch_add(1, Ordering::Relaxed));
        let (output_tx, output_rx) = mpsc::channel(64);
        let (response_tx, response_rx) = oneshot::channel();
        self.cmd_tx
            .send(CpuEngineCommand::Generate {
                request_id,
                prompt,
                sampling_params,
                output_tx,
                response_tx,
            })
            .await
            .map_err(|_| LLMError::SchedulerError("CPU engine task stopped".into()))?;
        response_rx
            .await
            .map_err(|_| LLMError::SchedulerError("CPU admission response dropped".into()))??;
        Ok((request_id, ReceiverStream::new(output_rx)))
    }

    pub async fn add_request(
        &self,
        request_id: RequestId,
        prompt: String,
        sampling_params: SamplingParams,
    ) -> Result<()> {
        let (response_tx, response_rx) = oneshot::channel();
        self.cmd_tx
            .send(CpuEngineCommand::AddRequest {
                request_id,
                prompt,
                sampling_params,
                response_tx,
            })
            .await
            .map_err(|_| LLMError::SchedulerError("CPU engine task stopped".into()))?;
        response_rx
            .await
            .map_err(|_| LLMError::SchedulerError("CPU admission response dropped".into()))?
    }

    pub async fn abort_request(&self, request_id: RequestId) {
        let _ = self
            .cmd_tx
            .send(CpuEngineCommand::AbortRequest { request_id })
            .await;
    }

    pub fn shutdown(&self) {
        self.cancel.cancel();
    }

    async fn background_loop(
        mut engine: CpuBatchEngine,
        mut cmd_rx: mpsc::Receiver<CpuEngineCommand>,
        cancel: CancellationToken,
    ) {
        let mut output_channels = HashMap::<RequestId, mpsc::Sender<RequestOutput>>::new();
        loop {
            Self::drain_commands(&mut engine, &mut output_channels, &mut cmd_rx);
            Self::cancel_disconnected(&mut engine, &mut output_channels);

            if cancel.is_cancelled() {
                break;
            }
            if !engine.has_unfinished() {
                tokio::select! {
                    _ = cancel.cancelled() => break,
                    command = cmd_rx.recv() => {
                        let Some(command) = command else { break; };
                        Self::process_command(&mut engine, &mut output_channels, command);
                    }
                }
                continue;
            }

            let reservation = match engine.reserve() {
                Ok(Some(reservation)) => reservation,
                Ok(None) => {
                    tokio::task::yield_now().await;
                    continue;
                }
                Err(error) => {
                    error!(%error, "CPU reservation failed");
                    tokio::task::yield_now().await;
                    continue;
                }
            };
            let prepared = tokio::task::block_in_place(|| engine.execute(reservation));

            // Commands and receiver closure can occur while model kernels run.
            // Apply their tombstones before publishing prepared state.
            Self::drain_commands(&mut engine, &mut output_channels, &mut cmd_rx);
            Self::cancel_disconnected(&mut engine, &mut output_channels);
            if cancel.is_cancelled() {
                if let Ok(prepared) = prepared {
                    if let Err(error) = engine.discard(prepared) {
                        error!(%error, "failed to discard CPU work during shutdown");
                    }
                }
                break;
            }

            let committed = match prepared.and_then(|prepared| engine.commit(prepared)) {
                Ok(committed) => committed,
                Err(error) => {
                    error!(%error, "CPU iteration failed without committed progress");
                    tokio::task::yield_now().await;
                    continue;
                }
            };
            for request_id in committed.cancelled_requests {
                output_channels.remove(&request_id);
            }
            for output in committed.outputs {
                let request_id = output.request_id;
                let finished = output.finished;
                if let Some(channel) = output_channels.get(&request_id) {
                    if channel.send(output).await.is_err() {
                        debug!(%request_id, "CPU output receiver dropped after commit");
                        if let Err(error) = engine.cancel_request(request_id) {
                            error!(%request_id, %error, "failed to cancel disconnected CPU request");
                        }
                        output_channels.remove(&request_id);
                        continue;
                    }
                }
                if finished {
                    output_channels.remove(&request_id);
                }
            }
            tokio::task::yield_now().await;
        }

        if let Err(error) = engine.shutdown() {
            error!(%error, "CPU engine shutdown failed");
        }
        output_channels.clear();
        info!("AsyncCpuBatchEngine background loop exited");
    }

    fn drain_commands(
        engine: &mut CpuBatchEngine,
        output_channels: &mut HashMap<RequestId, mpsc::Sender<RequestOutput>>,
        cmd_rx: &mut mpsc::Receiver<CpuEngineCommand>,
    ) {
        while let Ok(command) = cmd_rx.try_recv() {
            Self::process_command(engine, output_channels, command);
        }
    }

    fn process_command(
        engine: &mut CpuBatchEngine,
        output_channels: &mut HashMap<RequestId, mpsc::Sender<RequestOutput>>,
        command: CpuEngineCommand,
    ) {
        match command {
            CpuEngineCommand::Generate {
                request_id,
                prompt,
                sampling_params,
                output_tx,
                response_tx,
            } => match engine.add_request(request_id, prompt, sampling_params) {
                Ok(_) => {
                    output_channels.insert(request_id, output_tx);
                    let _ = response_tx.send(Ok(()));
                }
                Err(error) => {
                    let _ = response_tx.send(Err(error));
                }
            },
            CpuEngineCommand::AddRequest {
                request_id,
                prompt,
                sampling_params,
                response_tx,
            } => {
                let result = engine
                    .add_request(request_id, prompt, sampling_params)
                    .map(|_| ());
                let _ = response_tx.send(result);
            }
            CpuEngineCommand::AbortRequest { request_id } => {
                if let Err(error) = engine.cancel_request(request_id) {
                    error!(%request_id, %error, "failed to cancel CPU request");
                }
                output_channels.remove(&request_id);
            }
        }
    }

    fn cancel_disconnected(
        engine: &mut CpuBatchEngine,
        output_channels: &mut HashMap<RequestId, mpsc::Sender<RequestOutput>>,
    ) {
        let disconnected = output_channels
            .iter()
            .filter_map(|(&request_id, channel)| channel.is_closed().then_some(request_id))
            .collect::<Vec<_>>();
        for request_id in disconnected {
            if let Err(error) = engine.cancel_request(request_id) {
                error!(%request_id, %error, "failed to cancel disconnected CPU request");
            }
            output_channels.remove(&request_id);
        }
    }
}
