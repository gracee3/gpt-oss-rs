use std::collections::BTreeMap;
use std::io::{Read, Write};
use std::net::{TcpStream, ToSocketAddrs};
use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::time::Duration;

use anyhow::{bail, Context, Result};
use clap::Parser;
use gpt_oss_evidence::{
    ArtifactRef, EvidenceStatus, ModelEvidence, RunManifestV1, SourceProvenance, WorkloadEvidence,
};
use serde::Serialize;
use serde_json::Value;
use sha2::{Digest, Sha256};

#[derive(Debug, Parser)]
#[command(about = "Model-independent CPU service contract probe with E1 evidence sidecar")]
struct Cli {
    /// HTTP origin with an explicit port, for example http://127.0.0.1:8000.
    #[arg(long)]
    base_url: String,

    /// Expected stable public model alias.
    #[arg(long)]
    served_model: String,

    /// Atomic raw capture destination. A .manifest.json sidecar is also written.
    #[arg(long)]
    output: PathBuf,

    /// Expect /metrics to be absent because telemetry exposition is disabled.
    #[arg(long)]
    metrics_disabled: bool,

    /// Additional smoke capture to hash into the sidecar, formatted ROLE=PATH.
    #[arg(long, value_name = "ROLE=PATH")]
    artifact: Vec<AdditionalArtifact>,
}

#[derive(Debug, Clone)]
struct AdditionalArtifact {
    role: String,
    path: PathBuf,
}

impl FromStr for AdditionalArtifact {
    type Err = String;

    fn from_str(value: &str) -> std::result::Result<Self, Self::Err> {
        let (role, path) = value
            .split_once('=')
            .ok_or_else(|| "artifact must use ROLE=PATH".to_string())?;
        if role.is_empty() || path.is_empty() {
            return Err("artifact role and path must be non-empty".into());
        }
        Ok(Self {
            role: role.to_string(),
            path: PathBuf::from(path),
        })
    }
}

#[derive(Debug, Serialize)]
struct EndpointCapture {
    status: u16,
    body: Value,
}

#[derive(Debug, Serialize)]
struct ServiceProbeCapture {
    schema: &'static str,
    served_model: String,
    runtime_snapshot_sha256: String,
    endpoints: BTreeMap<String, EndpointCapture>,
}

struct HttpResponse {
    status: u16,
    body: Vec<u8>,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let (host, port) = parse_origin(&cli.base_url)?;
    let mut endpoints = BTreeMap::new();

    let health = get(&host, port, "/health")?;
    require_status("/health", &health, 200)?;
    endpoints.insert("health".into(), capture_text(health)?);

    let ready = get(&host, port, "/ready")?;
    require_status("/ready", &ready, 200)?;
    let ready_capture = capture_json(ready)?;
    let runtime_snapshot_sha256 = ready_capture
        .body
        .get("runtime_snapshot_sha256")
        .and_then(Value::as_str)
        .context("/ready omitted runtime_snapshot_sha256")?
        .to_string();
    require_sha256(&runtime_snapshot_sha256, "runtime snapshot")?;
    let ready_model = ready_capture
        .body
        .get("model")
        .and_then(Value::as_str)
        .context("/ready omitted model")?;
    if ready_model != cli.served_model {
        bail!(
            "/ready exposed model {ready_model:?}, expected {:?}",
            cli.served_model
        );
    }
    endpoints.insert("ready".into(), ready_capture);

    let models = get(&host, port, "/v1/models")?;
    require_status("/v1/models", &models, 200)?;
    let models_capture = capture_json(models)?;
    let model_ids = models_capture
        .body
        .get("data")
        .and_then(Value::as_array)
        .context("/v1/models omitted data")?
        .iter()
        .filter_map(|model| model.get("id").and_then(Value::as_str))
        .collect::<Vec<_>>();
    if model_ids != [cli.served_model.as_str()] {
        bail!(
            "/v1/models exposed {model_ids:?}, expected only {:?}",
            cli.served_model
        );
    }
    endpoints.insert("models".into(), models_capture);

    let metrics = get(&host, port, "/metrics")?;
    let expected_metrics_status = if cli.metrics_disabled { 404 } else { 200 };
    require_status("/metrics", &metrics, expected_metrics_status)?;
    if !cli.metrics_disabled {
        let text = std::str::from_utf8(&metrics.body).context("/metrics was not UTF-8")?;
        if !text.contains("gpt_oss_service_state") {
            bail!("/metrics omitted gpt_oss_service_state");
        }
        if text.contains(&cli.served_model) {
            bail!("/metrics used the served model as a label value");
        }
    }
    endpoints.insert("metrics".into(), capture_text(metrics)?);

    let batches = get(&host, port, "/v1/batches")?;
    require_status("/v1/batches", &batches, 404)?;
    endpoints.insert("batches_unmounted".into(), capture_text(batches)?);

    let capture = ServiceProbeCapture {
        schema: "gpt-oss-rs.cpu-service-probe/v1",
        served_model: cli.served_model.clone(),
        runtime_snapshot_sha256: runtime_snapshot_sha256.clone(),
        endpoints,
    };
    let encoded = serde_json::to_vec_pretty(&capture)?;
    if let Some(parent) = cli.output.parent() {
        std::fs::create_dir_all(parent)?;
    }
    gpt_oss_evidence::atomic_write(&cli.output, &encoded)?;
    write_sidecar(&cli, runtime_snapshot_sha256, &encoded)?;
    println!("{}", String::from_utf8(encoded)?);
    Ok(())
}

fn parse_origin(origin: &str) -> Result<(String, u16)> {
    let authority = origin
        .strip_prefix("http://")
        .context("--base-url must start with http://")?
        .trim_end_matches('/');
    if authority.contains('/') {
        bail!("--base-url must not contain a path");
    }
    let (host, port) = authority
        .rsplit_once(':')
        .context("--base-url must include an explicit port")?;
    if host.is_empty() {
        bail!("--base-url host is empty");
    }
    let port = port.parse().context("invalid --base-url port")?;
    Ok((host.to_string(), port))
}

fn get(host: &str, port: u16, path: &str) -> Result<HttpResponse> {
    let address = (host, port)
        .to_socket_addrs()?
        .next()
        .context("service origin did not resolve")?;
    let mut stream = TcpStream::connect_timeout(&address, Duration::from_secs(5))?;
    stream.set_read_timeout(Some(Duration::from_secs(10)))?;
    stream.set_write_timeout(Some(Duration::from_secs(5)))?;
    write!(
        stream,
        "GET {path} HTTP/1.1\r\nHost: {host}:{port}\r\nConnection: close\r\nAccept: */*\r\n\r\n"
    )?;
    stream.flush()?;
    let mut raw = Vec::new();
    stream.read_to_end(&mut raw)?;
    parse_response(&raw)
}

fn parse_response(raw: &[u8]) -> Result<HttpResponse> {
    let separator = raw
        .windows(4)
        .position(|window| window == b"\r\n\r\n")
        .context("HTTP response omitted the header terminator")?;
    let header_bytes = &raw[..separator];
    let headers = std::str::from_utf8(header_bytes).context("HTTP headers were not UTF-8")?;
    let status = headers
        .lines()
        .next()
        .and_then(|line| line.split_whitespace().nth(1))
        .context("HTTP response omitted its status")?
        .parse()
        .context("invalid HTTP response status")?;
    let mut body = raw[separator + 4..].to_vec();
    if headers
        .lines()
        .any(|line| line.eq_ignore_ascii_case("transfer-encoding: chunked"))
    {
        body = decode_chunked(&body)?;
    }
    Ok(HttpResponse { status, body })
}

fn decode_chunked(raw: &[u8]) -> Result<Vec<u8>> {
    let mut offset = 0;
    let mut decoded = Vec::new();
    loop {
        let line_end = raw[offset..]
            .windows(2)
            .position(|window| window == b"\r\n")
            .map(|relative| offset + relative)
            .context("truncated chunk length")?;
        let length_text = std::str::from_utf8(&raw[offset..line_end])?
            .split(';')
            .next()
            .context("empty chunk length")?;
        let length = usize::from_str_radix(length_text, 16).context("invalid chunk length")?;
        offset = line_end + 2;
        if length == 0 {
            return Ok(decoded);
        }
        let end = offset
            .checked_add(length)
            .context("chunk length overflow")?;
        if end + 2 > raw.len() || &raw[end..end + 2] != b"\r\n" {
            bail!("truncated chunk payload");
        }
        decoded.extend_from_slice(&raw[offset..end]);
        offset = end + 2;
    }
}

fn require_status(path: &str, response: &HttpResponse, expected: u16) -> Result<()> {
    if response.status != expected {
        bail!(
            "{path} returned {}, expected {expected}: {}",
            response.status,
            String::from_utf8_lossy(&response.body)
        );
    }
    Ok(())
}

fn capture_json(response: HttpResponse) -> Result<EndpointCapture> {
    Ok(EndpointCapture {
        status: response.status,
        body: serde_json::from_slice(&response.body).context("endpoint returned invalid JSON")?,
    })
}

fn capture_text(response: HttpResponse) -> Result<EndpointCapture> {
    Ok(EndpointCapture {
        status: response.status,
        body: Value::String(String::from_utf8(response.body)?),
    })
}

fn require_sha256(value: &str, label: &str) -> Result<()> {
    if value.len() != 64 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        bail!("{label} is not a SHA-256 digest");
    }
    Ok(())
}

fn write_sidecar(cli: &Cli, runtime_snapshot_sha256: String, raw_bytes: &[u8]) -> Result<()> {
    let artifact = ArtifactRef::from_path("raw-output", &cli.output)?;
    if artifact.sha256 != sha256(raw_bytes) {
        bail!("written service probe does not match its in-memory capture");
    }
    let mut evidence = RunManifestV1::new(
        "cpu-service-probe",
        "service-contract",
        EvidenceStatus::Pass,
    );
    evidence.source = local_source_provenance();
    evidence.model = ModelEvidence {
        id: cli.served_model.clone(),
        revision: "served-runtime".into(),
        ..ModelEvidence::default()
    };
    evidence.runtime_snapshot_sha256 = runtime_snapshot_sha256;
    evidence.command.argv_redacted = std::env::args().collect();
    evidence.workload = WorkloadEvidence {
        id: "model-independent-public-endpoints".into(),
        prompt_sha256: None,
        seed: 0,
        repetitions: 1,
    };
    evidence.artifacts.push(artifact);
    for additional in &cli.artifact {
        evidence
            .artifacts
            .push(ArtifactRef::from_path(&additional.role, &additional.path)?);
    }
    evidence
        .limitations
        .push("probes public service metadata only; it does not run inference".into());
    evidence.write_atomic(sidecar_path(&cli.output)?)?;
    Ok(())
}

fn sidecar_path(output: &Path) -> Result<PathBuf> {
    let file_name = output
        .file_name()
        .and_then(|name| name.to_str())
        .context("output has no UTF-8 file name")?;
    Ok(output.with_file_name(format!("{file_name}.manifest.json")))
}

fn local_source_provenance() -> SourceProvenance {
    let repository_commit = command_output("git", &["rev-parse", "HEAD"])
        .filter(|value| value.len() == 40)
        .unwrap_or_default();
    let dirty = std::process::Command::new("git")
        .args(["status", "--porcelain"])
        .output()
        .ok()
        .is_some_and(|output| output.status.success() && !output.stdout.is_empty());
    SourceProvenance {
        repository_commit,
        dirty,
        branch_role: "candidate".into(),
        cargo_lock_sha256: std::fs::read("Cargo.lock")
            .ok()
            .map(|bytes| sha256(&bytes))
            .unwrap_or_default(),
        toolchain: command_output("rustc", &["--version"]).unwrap_or_else(|| "unknown".into()),
        profile: "release".into(),
        features: Vec::new(),
    }
}

fn command_output(program: &str, arguments: &[&str]) -> Option<String> {
    std::process::Command::new(program)
        .args(arguments)
        .output()
        .ok()
        .filter(|output| output.status.success())
        .and_then(|output| String::from_utf8(output.stdout).ok())
        .map(|value| value.trim().to_string())
}

fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_fixed_and_chunked_http_responses() {
        let fixed = parse_response(b"HTTP/1.1 200 OK\r\ncontent-length: 2\r\n\r\n{}").unwrap();
        assert_eq!(fixed.status, 200);
        assert_eq!(fixed.body, b"{}");

        let chunked = parse_response(
            b"HTTP/1.1 200 OK\r\ntransfer-encoding: chunked\r\n\r\n2\r\n{}\r\n0\r\n\r\n",
        )
        .unwrap();
        assert_eq!(chunked.body, b"{}");
    }

    #[test]
    fn origin_requires_plain_http_and_explicit_port() {
        assert_eq!(
            parse_origin("http://127.0.0.1:8000/").unwrap(),
            ("127.0.0.1".into(), 8000)
        );
        assert!(parse_origin("https://127.0.0.1:8000").is_err());
        assert!(parse_origin("http://127.0.0.1").is_err());
        assert!(parse_origin("http://127.0.0.1:8000/path").is_err());
        let artifact: AdditionalArtifact = "stream=/tmp/capture.sse".parse().unwrap();
        assert_eq!(artifact.role, "stream");
        assert_eq!(artifact.path, Path::new("/tmp/capture.sse"));
        assert!("missing-separator".parse::<AdditionalArtifact>().is_err());
    }
}
