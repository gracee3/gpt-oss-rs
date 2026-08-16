use std::path::Path;

use gpt_oss_engine::worker::{
    reserve_owner_queues_all_or_none, CapacityOneOwnerQueue, HeterogeneousQueueRole,
};
use serde::Serialize;
use sha2::{Digest, Sha256};

#[derive(Serialize)]
struct QueueEvidence {
    schema: &'static str,
    repository_head: String,
    source_fingerprint_sha256: String,
    executable_sha256: String,
    capacity_per_queue: usize,
    cpu_high_water: usize,
    remote_gpu_high_water: usize,
    cpu_reservations: u64,
    remote_gpu_reservations: u64,
    cpu_exhaustions: u64,
    remote_gpu_exhaustions: u64,
    cpu_occupied_after_release: bool,
    remote_gpu_occupied_after_release: bool,
    partial_reservation_rolled_back: bool,
    passed: bool,
}

#[test]
fn capacity_one_owner_queues_are_all_or_none_and_emit_high_water() {
    let cpu = CapacityOneOwnerQueue::new(HeterogeneousQueueRole::Cpu);
    let remote = CapacityOneOwnerQueue::new(HeterogeneousQueueRole::RemoteGpu);

    let held_remote = remote.try_reserve(10).unwrap();
    assert!(reserve_owner_queues_all_or_none(&cpu, &remote, 11).is_err());
    let partial_reservation_rolled_back = !cpu.stats().occupied && remote.stats().occupied;
    assert!(partial_reservation_rolled_back);
    held_remote.release().unwrap();

    let (cpu_ticket, remote_ticket) = reserve_owner_queues_all_or_none(&cpu, &remote, 12).unwrap();
    assert!(cpu.try_reserve(13).is_err());
    assert!(remote.try_reserve(13).is_err());
    cpu_ticket.release().unwrap();
    remote_ticket.release().unwrap();

    let cpu_stats = cpu.stats();
    let remote_stats = remote.stats();
    assert_eq!(cpu_stats.high_water, 1);
    assert_eq!(remote_stats.high_water, 1);
    assert!(!cpu_stats.occupied);
    assert!(!remote_stats.occupied);

    if let Some(path) = std::env::var_os("GPT_OSS_H4_QUEUE_EVIDENCE") {
        let evidence = QueueEvidence {
            schema: "gpt-oss-rs.heterogeneous-h4-queues/v1",
            repository_head: required_env("GPT_OSS_H4_REPO_HEAD"),
            source_fingerprint_sha256: required_env("GPT_OSS_H4_SOURCE_FINGERPRINT"),
            executable_sha256: hash_file(&std::env::current_exe().unwrap()),
            capacity_per_queue: 1,
            cpu_high_water: cpu_stats.high_water,
            remote_gpu_high_water: remote_stats.high_water,
            cpu_reservations: cpu_stats.reservations,
            remote_gpu_reservations: remote_stats.reservations,
            cpu_exhaustions: cpu_stats.exhaustions,
            remote_gpu_exhaustions: remote_stats.exhaustions,
            cpu_occupied_after_release: cpu_stats.occupied,
            remote_gpu_occupied_after_release: remote_stats.occupied,
            partial_reservation_rolled_back,
            passed: true,
        };
        write_json(Path::new(&path), &evidence);
    }
}

fn required_env(name: &str) -> String {
    std::env::var(name).unwrap_or_else(|_| panic!("{name} is required when writing H4 evidence"))
}

fn hash_file(path: &Path) -> String {
    use std::io::Read;

    let mut file = std::fs::File::open(path).unwrap();
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        let read = file.read(&mut buffer).unwrap();
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    format!("{:x}", hasher.finalize())
}

fn write_json(path: &Path, value: &impl Serialize) {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).unwrap();
    }
    let mut bytes = serde_json::to_vec_pretty(value).unwrap();
    bytes.push(b'\n');
    std::fs::write(path, bytes).unwrap();
}
