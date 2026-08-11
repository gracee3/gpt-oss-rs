//! Compatibility path for the canonical CPU scheduler.
//!
//! CPU scheduling state lives only in [`crate::cpu_scheduler::SequenceTable`].
//! This module intentionally contains no scheduler-local sequence groups.

pub use crate::cpu_scheduler::{
    CpuReservation, CpuScheduledPhase, CpuScheduledRow, CpuScheduler as Scheduler,
    CpuSchedulerConfig as SchedulerConfig, CpuSequenceLifecycle, CpuSequenceRecord, SequenceTable,
};
