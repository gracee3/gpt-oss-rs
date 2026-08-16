use std::fs::File;
use std::os::fd::AsRawFd;
use std::os::unix::fs::PermissionsExt;
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::{Mutex, MutexGuard};
use std::thread;
use std::time::{Duration, Instant};

use gpt_oss_bench::h8_watchdog::read_h8_process_scan;

static TEST_LOCK: Mutex<()> = Mutex::new(());

fn test_lock() -> MutexGuard<'static, ()> {
    TEST_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

fn run_watchdog_death_case(signal: &str) {
    let _lock = test_lock();
    let ready = std::env::temp_dir().join(format!(
        "gpt-oss-h8-watchdog-{}-{signal}.ready",
        std::process::id()
    ));
    let _ = std::fs::remove_file(&ready);
    let mut watchdog = Command::new(env!("CARGO_BIN_EXE_heterogeneous_h8_watchdog"))
        .args(["lifecycle-probe", "--ready-file"])
        .arg(&ready)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .unwrap();
    let child_pid = wait_for_ready(&ready, &mut watchdog);

    let status = Command::new("/usr/bin/kill")
        .args(["--signal", signal, "--", &watchdog.id().to_string()])
        .status()
        .unwrap();
    assert!(status.success());
    wait_for_exit(&mut watchdog);
    wait_for_process_absent(child_pid);
    let _ = std::fs::remove_file(ready);
}

fn wait_for_ready(path: &PathBuf, watchdog: &mut Child) -> u32 {
    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        if let Ok(text) = std::fs::read_to_string(path) {
            return text.trim().parse().unwrap();
        }
        if let Some(status) = watchdog.try_wait().unwrap() {
            panic!("lifecycle probe exited before ready: {status}");
        }
        if Instant::now() >= deadline {
            let _ = watchdog.kill();
            let _ = watchdog.wait();
            panic!("lifecycle probe did not become ready");
        }
        thread::sleep(Duration::from_millis(20));
    }
}

fn wait_for_exit(process: &mut Child) {
    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        if process.try_wait().unwrap().is_some() {
            return;
        }
        if Instant::now() >= deadline {
            let _ = process.kill();
            let _ = process.wait();
            panic!("watchdog did not exit after signal");
        }
        thread::sleep(Duration::from_millis(20));
    }
}

fn wait_for_process_absent(pid: u32) {
    let path = PathBuf::from(format!("/proc/{pid}"));
    let deadline = Instant::now() + Duration::from_secs(5);
    while path.exists() {
        if Instant::now() >= deadline {
            let _ = Command::new("/usr/bin/kill")
                .args(["--signal", "KILL", "--", &pid.to_string()])
                .status();
            panic!("guarded child remained after watchdog death");
        }
        thread::sleep(Duration::from_millis(20));
    }
}

fn wait_for_stopped(child: &mut Child) {
    let status_path = PathBuf::from(format!("/proc/{}/status", child.id()));
    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        if let Ok(status) = std::fs::read_to_string(&status_path) {
            if status
                .lines()
                .any(|line| line.starts_with("State:") && line.contains("T (stopped)"))
            {
                return;
            }
        }
        if let Some(status) = child.try_wait().unwrap() {
            panic!("fd-executed H8 probe exited before stopping: {status}");
        }
        if Instant::now() >= deadline {
            let _ = child.kill();
            let _ = child.wait();
            panic!("fd-executed H8 probe did not stop");
        }
        thread::sleep(Duration::from_millis(20));
    }
}

#[test]
fn termination_handler_kills_guarded_child() {
    run_watchdog_death_case("TERM");
}

#[test]
fn parent_death_signal_kills_guarded_child_on_watchdog_sigkill() {
    run_watchdog_death_case("KILL");
}

#[test]
fn fd_exec_h8_is_detected_by_running_executable_and_mode() {
    let _lock = test_lock();
    let unique = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let root = std::env::temp_dir().join(format!(
        "gpt-oss-h8-fd-detection-{}-{unique}",
        std::process::id()
    ));
    let executable = root.join("heterogeneous_construct");
    std::fs::create_dir_all(&root).unwrap();
    std::fs::copy("/bin/sh", &executable).unwrap();
    std::fs::set_permissions(&executable, std::fs::Permissions::from_mode(0o700)).unwrap();
    let opened = File::open(&executable).unwrap();
    let fd_path = format!("/proc/self/fd/{}", opened.as_raw_fd());
    let mut child = Command::new(fd_path)
        .args(["-c", "kill -STOP $$", "probe", "--mode", "h8"])
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .unwrap();
    wait_for_stopped(&mut child);
    let scan = read_h8_process_scan(None);
    let _ = child.kill();
    let _ = child.wait();
    std::fs::remove_dir_all(root).unwrap();
    let scan = scan.unwrap();
    assert!(scan.proc_scan_complete);
    assert!(scan.active_h8_process_found);
}
