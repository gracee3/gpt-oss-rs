#![cfg_attr(not(feature = "cuda"), forbid(unsafe_code))]
//! Single-GPU worker execution context for gpt-oss-rs.
//!
//! Provides the `Worker` type that owns a model runner, KV cache engine, and
//! GPU stream for a single device. The worker receives `WorkerInput` from the
//! executor, applies cache operations, runs the forward pass, samples tokens,
//! and returns `WorkerOutput`.

pub mod config;
#[cfg(feature = "cuda")]
pub mod gpu_worker;
pub mod graph_runner;
pub mod input;
pub mod metrics;
pub mod worker;

pub use config::WorkerConfig;
pub use gpt_oss_engine::sequence::{SequenceData, SequenceGroupMetadata};
pub use graph_runner::{GraphRunner, GraphRunnerConfig};
pub use input::prepare_input;
pub use worker::{SamplerOutput, Worker, WorkerInput, WorkerOutput};
