use std::fs::File;
use std::os::fd::AsRawFd;
use std::os::unix::fs::PermissionsExt;
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::Mutex;
use std::thread;
use std::time::{Duration, Instant};

use gpt_oss_bench::h8_watchdog::read_host_snapshot;

static TEST_LOCK: Mutex<()> = Mutex::new(());

fn run_watchdog_death_case(signal: &str) {
    let _lock = TEST_LOCK.lock().unwrap();
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
    let _lock = TEST_LOCK.lock().unwrap();
    let root = std::env::temp_dir().join(format!("gpt-oss-h8-fd-detection-{}", std::process::id()));
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
    thread::sleep(Duration::from_millis(50));
    let snapshot = read_host_snapshot(None, 0).unwrap();
    let _ = child.kill();
    let _ = child.wait();
    std::fs::remove_dir_all(root).unwrap();
    assert!(snapshot.active_h8_process_found);
}
