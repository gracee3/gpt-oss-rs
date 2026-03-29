#![cfg_attr(not(feature = "cuda"), forbid(unsafe_code))]
//! Main inference engine for gpt-oss-rs.
//!
//! Composes the scheduler, executor, and tokenizer into a synchronous
//! [`LLMEngine`] and an async [`AsyncLLMEngine`] that drives the
//! continuous-batching inference loop.
//!
//! Real dependency crate types are imported and adapted:
//! - `gpt_oss_executor` -- `Executor` trait, `ExecutorInput`, `SamplerOutput`, `ExecutorFactory`
//! - `gpt_oss_tokenizer` -- `Tokenizer` for encode/decode
//! - `gpt_oss_sequence` -- `Sequence`, `SequenceGroup`, `SequenceGroupMetadata`
//!
//! `ExecutorAdapter` bridges the real async `gpt_oss_engine::executor::Executor` into the
//! engine's sync step loop. A `SchedulerAdapter` for `gpt_oss_scheduler` is
//! pending that crate's API alignment with `gpt_oss_sequence` types.
extern crate self as gpt_oss_engine;

pub mod async_engine;
#[cfg(feature = "cuda")]
pub mod async_gpu_engine;
pub mod beam_search;
pub mod best_of_n;
pub mod block_manager;
pub mod config;
pub mod engine;
pub mod executor;
#[cfg(feature = "cuda")]
pub mod gpu_engine;
pub mod gpu_metrics;
#[cfg(any(feature = "cuda", test))]
mod hf_snapshot;
pub mod output;
pub mod scheduler;
pub mod sequence;
pub mod speculative;
pub mod stop_checker;
pub mod telemetry;
pub mod worker;

pub use async_engine::AsyncLLMEngine;
pub use beam_search::BeamSearchState;
pub use best_of_n::{build_best_of_n_output, select_best_of_n};
pub use block_manager::{
    BlockManager, BlockTable, Device, MemoryPool, PhysicalBlock, PrefixCache, SharedBlockManager,
};
pub use config::{
    load_config, validate, CacheConfigImpl, CliArgs, ConfigError, DeviceConfig, EngineConfig,
    ModelConfigImpl, ParallelConfigImpl, PreemptionMode, SchedulerConfigImpl, TelemetryConfig,
};
pub use engine::LLMEngine;
pub use engine::{Executor, ExecutorAdapter, Scheduler};
pub use engine::{ExecutorInput, SamplerOutput, SchedulerOutputs};
pub use executor::{ExecutorConfig, ExecutorFactory};
pub use output::OutputProcessor;
pub use sequence::{Sequence, SequenceData, SequenceGroup, SequenceGroupMetadata, SequenceStatus};
pub use stop_checker::StopChecker;
pub use telemetry::{init_telemetry, metrics_handler, MetricsRecorder, TelemetryGuard};
pub use worker::{GraphRunner, GraphRunnerConfig, Worker, WorkerConfig, WorkerInput, WorkerOutput};

#[cfg(feature = "cuda")]
pub use async_gpu_engine::AsyncGpuLLMEngine;
#[cfg(feature = "cuda")]
pub use gpu_engine::GpuLLMEngine;
