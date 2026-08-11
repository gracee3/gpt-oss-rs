//! Read-only CPU topology observations for runtime diagnostics.

use std::collections::BTreeMap;
use std::fmt;
use std::fs;
use std::path::Path;

use serde::Serialize;

/// Logical processors that share one observed physical core.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct CpuPhysicalCore {
    pub package_id: Option<usize>,
    pub core_id: Option<usize>,
    pub logical_cpus: Vec<usize>,
}

/// CPUs observed under one Linux NUMA node.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct CpuNumaNode {
    pub node_id: usize,
    pub cpus: Vec<usize>,
    pub allowed_cpus: Vec<usize>,
}

/// Best-effort process and machine topology used only for diagnostics.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct CpuTopology {
    pub allowed_cpus: Vec<usize>,
    pub allowed_memory_nodes: Vec<usize>,
    pub physical_cores: Vec<CpuPhysicalCore>,
    pub numa_nodes: Vec<CpuNumaNode>,
    pub available_parallelism: usize,
    pub configured_worker_threads: usize,
}

impl CpuTopology {
    /// Observe Linux process masks and sysfs topology without changing either.
    pub fn observe(configured_worker_threads: usize) -> Self {
        let available_parallelism = std::thread::available_parallelism()
            .map(usize::from)
            .unwrap_or(1);
        let status = fs::read_to_string("/proc/self/status").unwrap_or_default();
        let mut allowed_cpus = status_list(&status, "Cpus_allowed_list")
            .unwrap_or_else(|| (0..available_parallelism).collect());
        allowed_cpus.sort_unstable();
        allowed_cpus.dedup();
        let allowed_memory_nodes = status_list(&status, "Mems_allowed_list").unwrap_or_default();
        let physical_cores =
            observe_physical_cores_at(Path::new("/sys/devices/system/cpu"), &allowed_cpus);
        let numa_nodes =
            observe_numa_nodes_at(Path::new("/sys/devices/system/node"), &allowed_cpus);
        Self {
            allowed_cpus,
            allowed_memory_nodes,
            physical_cores,
            numa_nodes,
            available_parallelism,
            configured_worker_threads,
        }
    }

    pub fn summary(&self) -> String {
        format!(
            "allowed_logical_cpus={}, physical_cores={}, numa_nodes={}, available_parallelism={}, worker_threads={}",
            self.allowed_cpus.len(),
            self.physical_cores.len(),
            self.numa_nodes.len(),
            self.available_parallelism,
            self.configured_worker_threads
        )
    }
}

impl fmt::Display for CpuTopology {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.summary())
    }
}

fn status_list(status: &str, field: &str) -> Option<Vec<usize>> {
    status.lines().find_map(|line| {
        let (name, value) = line.split_once(':')?;
        (name == field)
            .then(|| parse_cpu_list(value.trim()))
            .flatten()
    })
}

fn parse_cpu_list(value: &str) -> Option<Vec<usize>> {
    if value.is_empty() {
        return Some(Vec::new());
    }
    let mut values = Vec::new();
    for item in value.split(',') {
        let item = item.trim();
        if let Some((start, end)) = item.split_once('-') {
            let start = start.parse::<usize>().ok()?;
            let end = end.parse::<usize>().ok()?;
            if start > end {
                return None;
            }
            values.extend(start..=end);
        } else {
            values.push(item.parse::<usize>().ok()?);
        }
    }
    values.sort_unstable();
    values.dedup();
    Some(values)
}

fn read_usize(path: impl AsRef<Path>) -> Option<usize> {
    fs::read_to_string(path).ok()?.trim().parse().ok()
}

fn observe_physical_cores_at(root: &Path, allowed_cpus: &[usize]) -> Vec<CpuPhysicalCore> {
    let mut cores = BTreeMap::<(usize, usize), CpuPhysicalCore>::new();
    for &cpu in allowed_cpus {
        let topology = root.join(format!("cpu{cpu}")).join("topology");
        let package_id = read_usize(topology.join("physical_package_id"));
        let core_id = read_usize(topology.join("core_id"));
        // Unknown identities remain distinct instead of collapsing unrelated
        // logical CPUs into one artificial core.
        let key = (package_id.unwrap_or(usize::MAX), core_id.unwrap_or(cpu));
        cores
            .entry(key)
            .or_insert_with(|| CpuPhysicalCore {
                package_id,
                core_id,
                logical_cpus: Vec::new(),
            })
            .logical_cpus
            .push(cpu);
    }
    cores.into_values().collect()
}

fn observe_numa_nodes_at(root: &Path, allowed_cpus: &[usize]) -> Vec<CpuNumaNode> {
    let Ok(entries) = fs::read_dir(root) else {
        return Vec::new();
    };
    let mut nodes = entries
        .filter_map(Result::ok)
        .filter_map(|entry| {
            let name = entry.file_name();
            let name = name.to_str()?;
            let node_id = name.strip_prefix("node")?.parse::<usize>().ok()?;
            let cpus = fs::read_to_string(entry.path().join("cpulist"))
                .ok()
                .and_then(|value| parse_cpu_list(value.trim()))?;
            let allowed_cpus = cpus
                .iter()
                .copied()
                .filter(|cpu| allowed_cpus.binary_search(cpu).is_ok())
                .collect();
            Some(CpuNumaNode {
                node_id,
                cpus,
                allowed_cpus,
            })
        })
        .collect::<Vec<_>>();
    nodes.sort_unstable_by_key(|node| node.node_id);
    nodes
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_linux_cpu_lists_and_rejects_reversed_ranges() {
        assert_eq!(
            parse_cpu_list("0-3,8,10-11"),
            Some(vec![0, 1, 2, 3, 8, 10, 11])
        );
        assert_eq!(parse_cpu_list("3,1,3"), Some(vec![1, 3]));
        assert_eq!(parse_cpu_list("4-2"), None);
        assert_eq!(parse_cpu_list("bad"), None);
    }

    #[test]
    fn extracts_process_masks_from_proc_status_text() {
        let status = "Name:\ttest\nCpus_allowed_list:\t2-3,8\nMems_allowed_list:\t0,2\n";
        assert_eq!(
            status_list(status, "Cpus_allowed_list"),
            Some(vec![2, 3, 8])
        );
        assert_eq!(status_list(status, "Mems_allowed_list"), Some(vec![0, 2]));
        assert_eq!(status_list(status, "missing"), None);
    }

    #[test]
    fn live_observation_is_read_only_and_reports_thread_count() {
        let topology = CpuTopology::observe(7);
        assert!(!topology.allowed_cpus.is_empty());
        assert!(topology.available_parallelism > 0);
        assert_eq!(topology.configured_worker_threads, 7);
        assert!(topology.summary().contains("worker_threads=7"));
    }

    #[test]
    fn synthetic_sysfs_groups_allowed_core_siblings_and_numa_cpus() {
        let temp = tempfile::tempdir().unwrap();
        let cpu_root = temp.path().join("cpu");
        for (cpu, package, core) in [(0, 0, 4), (1, 0, 4), (2, 1, 0)] {
            let topology = cpu_root.join(format!("cpu{cpu}/topology"));
            fs::create_dir_all(&topology).unwrap();
            fs::write(topology.join("physical_package_id"), package.to_string()).unwrap();
            fs::write(topology.join("core_id"), core.to_string()).unwrap();
        }
        let cores = observe_physical_cores_at(&cpu_root, &[0, 1, 2]);
        assert_eq!(cores.len(), 2);
        assert_eq!(cores[0].logical_cpus, vec![0, 1]);
        assert_eq!(cores[1].logical_cpus, vec![2]);

        let node_root = temp.path().join("node");
        fs::create_dir_all(node_root.join("node0")).unwrap();
        fs::create_dir_all(node_root.join("node1")).unwrap();
        fs::write(node_root.join("node0/cpulist"), "0-1").unwrap();
        fs::write(node_root.join("node1/cpulist"), "2-5").unwrap();
        let nodes = observe_numa_nodes_at(&node_root, &[0, 1, 2]);
        assert_eq!(nodes.len(), 2);
        assert_eq!(nodes[0].allowed_cpus, vec![0, 1]);
        assert_eq!(nodes[1].allowed_cpus, vec![2]);
    }
}
