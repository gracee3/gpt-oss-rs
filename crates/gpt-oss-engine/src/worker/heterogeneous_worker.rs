//! Capacity-one queue reservations for the detached heterogeneous H4 harness.

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;

use gpt_oss_core::error::{LLMError, Result};
use parking_lot::Mutex;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HeterogeneousQueueRole {
    Cpu,
    RemoteGpu,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CapacityOneQueueStats {
    pub occupied: bool,
    pub high_water: usize,
    pub reservations: u64,
    pub exhaustions: u64,
}

struct QueueInner {
    role: HeterogeneousQueueRole,
    generation: Mutex<Option<u64>>,
    high_water: AtomicUsize,
    reservations: AtomicU64,
    exhaustions: AtomicU64,
}

#[derive(Clone)]
pub struct CapacityOneOwnerQueue {
    inner: Arc<QueueInner>,
}

impl CapacityOneOwnerQueue {
    pub fn new(role: HeterogeneousQueueRole) -> Self {
        Self {
            inner: Arc::new(QueueInner {
                role,
                generation: Mutex::new(None),
                high_water: AtomicUsize::new(0),
                reservations: AtomicU64::new(0),
                exhaustions: AtomicU64::new(0),
            }),
        }
    }

    pub fn try_reserve(&self, generation: u64) -> Result<OwnerQueueTicket> {
        let mut occupied = self.inner.generation.lock();
        if occupied.is_some() {
            self.inner.exhaustions.fetch_add(1, Ordering::Relaxed);
            return Err(LLMError::ModelError(format!(
                "heterogeneous {:?} queue is full",
                self.inner.role
            )));
        }
        *occupied = Some(generation);
        self.inner.high_water.store(1, Ordering::Release);
        self.inner.reservations.fetch_add(1, Ordering::Relaxed);
        Ok(OwnerQueueTicket {
            queue: Arc::clone(&self.inner),
            generation,
            released: false,
        })
    }

    pub fn stats(&self) -> CapacityOneQueueStats {
        CapacityOneQueueStats {
            occupied: self.inner.generation.lock().is_some(),
            high_water: self.inner.high_water.load(Ordering::Acquire),
            reservations: self.inner.reservations.load(Ordering::Acquire),
            exhaustions: self.inner.exhaustions.load(Ordering::Acquire),
        }
    }
}

pub struct OwnerQueueTicket {
    queue: Arc<QueueInner>,
    generation: u64,
    released: bool,
}

impl OwnerQueueTicket {
    pub const fn generation(&self) -> u64 {
        self.generation
    }

    pub fn role(&self) -> HeterogeneousQueueRole {
        self.queue.role
    }

    pub fn release(mut self) -> Result<()> {
        self.release_inner()
    }

    fn release_inner(&mut self) -> Result<()> {
        if self.released {
            return Err(LLMError::ModelError(
                "heterogeneous queue ticket already released".into(),
            ));
        }
        let mut occupied = self.queue.generation.lock();
        if *occupied != Some(self.generation) {
            return Err(LLMError::ModelError(format!(
                "heterogeneous {:?} queue generation changed before release",
                self.queue.role
            )));
        }
        *occupied = None;
        self.released = true;
        Ok(())
    }
}

impl Drop for OwnerQueueTicket {
    fn drop(&mut self) {
        if !self.released {
            let mut occupied = self.queue.generation.lock();
            if *occupied == Some(self.generation) {
                *occupied = None;
            }
            self.released = true;
        }
    }
}

/// The CPU and remote-GPU queue slots are reserved before any job is
/// enqueued. Failure to reserve the second slot drops/releases the first, so a
/// caller can never observe a partial dispatch reservation.
pub fn reserve_owner_queues_all_or_none(
    cpu: &CapacityOneOwnerQueue,
    remote_gpu: &CapacityOneOwnerQueue,
    generation: u64,
) -> Result<(OwnerQueueTicket, OwnerQueueTicket)> {
    if cpu.inner.role != HeterogeneousQueueRole::Cpu
        || remote_gpu.inner.role != HeterogeneousQueueRole::RemoteGpu
    {
        return Err(LLMError::ModelError(
            "heterogeneous queue roles do not match CPU/remote reservation".into(),
        ));
    }
    let cpu_ticket = cpu.try_reserve(generation)?;
    let remote_ticket = remote_gpu.try_reserve(generation)?;
    Ok((cpu_ticket, remote_ticket))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn capacity_one_queue_is_generation_tagged_and_bounded() {
        let queue = CapacityOneOwnerQueue::new(HeterogeneousQueueRole::Cpu);
        let ticket = queue.try_reserve(11).unwrap();
        assert_eq!(ticket.generation(), 11);
        assert!(queue.try_reserve(12).is_err());
        ticket.release().unwrap();
        assert!(!queue.stats().occupied);
        assert_eq!(queue.stats().high_water, 1);
        assert_eq!(queue.stats().exhaustions, 1);
    }

    #[test]
    fn all_or_none_rolls_back_first_ticket_when_second_is_full() {
        let cpu = CapacityOneOwnerQueue::new(HeterogeneousQueueRole::Cpu);
        let remote = CapacityOneOwnerQueue::new(HeterogeneousQueueRole::RemoteGpu);
        let occupied = remote.try_reserve(7).unwrap();
        assert!(reserve_owner_queues_all_or_none(&cpu, &remote, 8).is_err());
        assert!(!cpu.stats().occupied);
        assert!(remote.stats().occupied);
        occupied.release().unwrap();
        let (cpu_ticket, remote_ticket) =
            reserve_owner_queues_all_or_none(&cpu, &remote, 9).unwrap();
        cpu_ticket.release().unwrap();
        remote_ticket.release().unwrap();
    }
}
