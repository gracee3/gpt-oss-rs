use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::process::{Command as ProcessCommand, Output};

use anyhow::{bail, Context, Result};
use serde::Deserialize;

pub const DEFAULT_OFFICIAL_VISIBLE_DEVICES: &str = "0,1";

#[derive(Debug, Deserialize)]
struct PythonPreflight {
    missing: Vec<String>,
    cuda_device_count: usize,
    visible_devices: Option<String>,
}

pub fn selected_visible_devices_from_env() -> String {
    std::env::var("CUDA_VISIBLE_DEVICES")
        .unwrap_or_else(|_| DEFAULT_OFFICIAL_VISIBLE_DEVICES.to_string())
}

pub fn helper_script_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tools")
        .join("pinned_prompt_official_capture.py")
}

pub fn preflight_official_python(
    python: &Path,
    official_checkout: &Path,
    visible_devices: &str,
) -> Result<()> {
    let output = ProcessCommand::new(python)
        .env("PYTHONPATH", prepend_pythonpath(official_checkout))
        .env("CUDA_VISIBLE_DEVICES", visible_devices)
        .arg("-c")
        .arg(
            "import importlib.util as u, json, os; mods=['torch','safetensors','gpt_oss']; missing=[m for m in mods if u.find_spec(m) is None]; cuda_device_count=(__import__('torch').cuda.device_count() if 'torch' not in missing else 0); print(json.dumps({'missing': missing, 'cuda_device_count': cuda_device_count, 'visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES')}))",
        )
        .output()
        .with_context(|| format!("failed to start python interpreter '{}'", python.display()))?;

    if !output.status.success() {
        bail!(
            "official reference preflight failed for python '{}': {}",
            python.display(),
            String::from_utf8_lossy(&output.stderr).trim()
        );
    }

    let preflight: PythonPreflight =
        serde_json::from_slice(&output.stdout).context("parse official python preflight output")?;
    if !preflight.missing.is_empty() {
        bail!(
            "official reference preflight failed for python '{}': missing python modules: {}",
            python.display(),
            preflight.missing.join(", ")
        );
    }
    if preflight.cuda_device_count < 2 {
        bail!(
            "official reference preflight failed for python '{}': distributed PPP official capture requires at least 2 visible CUDA devices; CUDA_VISIBLE_DEVICES='{}' exposes {} device(s)",
            python.display(),
            preflight.visible_devices.as_deref().unwrap_or(visible_devices),
            preflight.cuda_device_count
        );
    }
    Ok(())
}

pub fn run_distributed_helper(
    python: &Path,
    official_checkout: &Path,
    visible_devices: &str,
    input_path: &Path,
    output_path: &Path,
) -> Result<Output> {
    let helper_path = helper_script_path();
    if !helper_path.is_file() {
        bail!(
            "missing PPP official capture helper at {}",
            helper_path.display()
        );
    }

    ProcessCommand::new(python)
        .env("PYTHONPATH", prepend_pythonpath(official_checkout))
        .env("CUDA_VISIBLE_DEVICES", visible_devices)
        .arg("-m")
        .arg("torch.distributed.run")
        .arg("--standalone")
        .arg("--nproc-per-node=2")
        .arg(&helper_path)
        .arg("--input")
        .arg(input_path)
        .arg("--output")
        .arg(output_path)
        .arg("--official-checkout")
        .arg(official_checkout)
        .output()
        .with_context(|| {
            format!(
                "failed to run distributed PPP helper {}",
                helper_path.display()
            )
        })
}

fn prepend_pythonpath(official_checkout: &Path) -> OsString {
    let mut entries = vec![official_checkout.as_os_str().to_os_string()];
    if let Some(existing) = std::env::var_os("PYTHONPATH") {
        entries.extend(std::env::split_paths(&existing).map(|path| path.into_os_string()));
    }
    std::env::join_paths(entries).expect("valid PYTHONPATH entries")
}

#[cfg(test)]
mod tests {
    use super::{
        helper_script_path, preflight_official_python, selected_visible_devices_from_env,
        DEFAULT_OFFICIAL_VISIBLE_DEVICES,
    };
    use std::io::Write;
    use std::os::unix::fs::PermissionsExt;
    use std::path::{Path, PathBuf};
    use std::process::Command as ProcessCommand;
    use std::sync::{Mutex, OnceLock};
    use std::time::{SystemTime, UNIX_EPOCH};

    fn env_lock() -> &'static Mutex<()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(()))
    }

    fn unique_temp_path(label: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock before unix epoch")
            .as_nanos();
        std::env::temp_dir().join(format!("gpt-oss-bench-{label}-{nanos}"))
    }

    fn write_fake_python_script(contents: &str) -> PathBuf {
        let script_path = unique_temp_path("fake-python");
        let mut handle = std::fs::File::create(&script_path).expect("create fake python");
        handle
            .write_all(contents.as_bytes())
            .expect("write fake python");
        let mut permissions = handle.metadata().expect("metadata").permissions();
        permissions.set_mode(0o755);
        std::fs::set_permissions(&script_path, permissions).expect("chmod fake python");
        script_path
    }

    fn write_json(path: &Path, value: &serde_json::Value) {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).expect("create parent");
        }
        std::fs::write(path, serde_json::to_vec(value).expect("serialize json"))
            .expect("write json");
    }

    #[test]
    fn selected_visible_devices_defaults_to_two_gpu_mapping() {
        let _guard = env_lock().lock().expect("lock env");
        let previous = std::env::var_os("CUDA_VISIBLE_DEVICES");
        std::env::remove_var("CUDA_VISIBLE_DEVICES");
        assert_eq!(
            selected_visible_devices_from_env(),
            DEFAULT_OFFICIAL_VISIBLE_DEVICES
        );
        if let Some(value) = previous {
            std::env::set_var("CUDA_VISIBLE_DEVICES", value);
        }
    }

    #[test]
    fn preflight_rejects_missing_modules() {
        let script = write_fake_python_script(
            "#!/bin/sh\nprintf '%s' '{\"missing\":[\"gpt_oss\"],\"cuda_device_count\":2,\"visible_devices\":\"0,1\"}'\n",
        );
        let checkout = unique_temp_path("official-checkout");
        std::fs::create_dir_all(&checkout).expect("create checkout");

        let err = preflight_official_python(&script, &checkout, "0,1").unwrap_err();
        let message = err.to_string();
        assert!(message.contains("missing python modules"));
        assert!(message.contains("gpt_oss"));
    }

    #[test]
    fn preflight_rejects_fewer_than_two_visible_gpus() {
        let script = write_fake_python_script(
            "#!/bin/sh\nprintf '%s' '{\"missing\":[],\"cuda_device_count\":1,\"visible_devices\":\"0\"}'\n",
        );
        let checkout = unique_temp_path("official-checkout");
        std::fs::create_dir_all(&checkout).expect("create checkout");

        let err = preflight_official_python(&script, &checkout, "0").unwrap_err();
        let message = err.to_string();
        assert!(message.contains("at least 2 visible CUDA devices"));
        assert!(message.contains("CUDA_VISIBLE_DEVICES='0'"));
    }

    #[test]
    fn helper_self_test_only_emits_rank_zero_output() {
        let helper = helper_script_path();
        assert!(
            helper.is_file(),
            "helper script missing at {}",
            helper.display()
        );

        let base = unique_temp_path("rank-ownership");
        let rank_zero_input = base.join("rank0-input.json");
        let rank_zero_output = base.join("rank0-output.json");
        let rank_one_input = base.join("rank1-input.json");
        let rank_one_output = base.join("rank1-output.json");
        write_json(&rank_zero_input, &serde_json::json!({"ignored": true}));
        write_json(&rank_one_input, &serde_json::json!({"ignored": true}));

        let rank_zero = ProcessCommand::new("python3")
            .env("PPP_OFFICIAL_CAPTURE_TEST_MODE", "rank-ownership")
            .env("WORLD_SIZE", "2")
            .env("RANK", "0")
            .arg(&helper)
            .arg("--input")
            .arg(&rank_zero_input)
            .arg("--output")
            .arg(&rank_zero_output)
            .arg("--official-checkout")
            .arg(".")
            .output()
            .expect("run rank0 self-test");
        assert!(
            rank_zero.status.success(),
            "rank0 self-test failed: stdout={} stderr={}",
            String::from_utf8_lossy(&rank_zero.stdout),
            String::from_utf8_lossy(&rank_zero.stderr)
        );
        assert!(rank_zero_output.is_file(), "rank0 output file missing");
        let rank_zero_stdout = String::from_utf8(rank_zero.stdout).expect("rank0 stdout utf8");
        assert!(rank_zero_stdout.contains("\"rank\": 0"));

        let rank_one = ProcessCommand::new("python3")
            .env("PPP_OFFICIAL_CAPTURE_TEST_MODE", "rank-ownership")
            .env("WORLD_SIZE", "2")
            .env("RANK", "1")
            .arg(&helper)
            .arg("--input")
            .arg(&rank_one_input)
            .arg("--output")
            .arg(&rank_one_output)
            .arg("--official-checkout")
            .arg(".")
            .output()
            .expect("run rank1 self-test");
        assert!(
            rank_one.status.success(),
            "rank1 self-test failed: stdout={} stderr={}",
            String::from_utf8_lossy(&rank_one.stdout),
            String::from_utf8_lossy(&rank_one.stderr)
        );
        assert!(String::from_utf8(rank_one.stdout)
            .expect("rank1 stdout utf8")
            .trim()
            .is_empty());
        assert!(
            !rank_one_output.exists(),
            "rank1 should not create an output file"
        );
    }
}
