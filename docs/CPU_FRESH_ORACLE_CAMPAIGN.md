# Fresh CPU Oracle Campaign

This is the only procedure allowed to create new authoritative CPU parity
claims. All prior official-oracle, C3, seven-scenario, and 28-cell captures are
retired historical records. Static fixture inputs may be reused only after
their bytes are rehashed; no prior output capture may enter this campaign.

## Fixed identity

- branch base ancestor: `f86674d6acf17484899f5d17e286dcb2c6d1f850`;
- official source: `gpt-oss` `v0.0.9`, revision
  `599476783c6f88508dab8577808b5ead5cbee8d2`, source-archive SHA-256
  `7306d68ae017f461f2ebb82d04628f8dcba7cc7b431ef28e8786c947510c6f6b`;
- model revision: `6cee5e81ee83917806bbde320786a8fb61efebee`;
- Python `3.12.12`, CPU-only PyTorch `2.12.1`, and SafeTensors `0.8.0`;
- llama.cpp `030ebb558a5820b444a8f836ed5cdd46c9b4bd7a`;
- image: private `ghcr.io/gracee3/gpt-oss-rs-cpu-oracle`, consumed only as
  `name@sha256:digest`.

Normal PyTorch CPU dispatch (`execution_mode=native`) is authoritative.
`ATEN_CPU_CAPABILITY=default` is a qualification diagnostic recorded as
`execution_mode=generic`; it can neither pass nor fail an official comparison.
The comparator rejects mixed modes and any differing oracle identity field.

## Two-commit freeze

1. Commit and push all image inputs, workflow, and harness changes. Record
   that commit as the image-input revision.
2. Dispatch `CPU oracle image` on that exact revision. It builds and publishes
   `linux/amd64`, attaches SBOM and provenance, runs five-repeat native and
   generic probes, and exports the exact pushed digest with a digest-pinned
   skopeo image. Download its artifact. Preserve the OCI archive, SBOM,
   provenance, probes, and material JSON under
   `/home/emmy/gpt-oss-rs-artifacts/oracle-images/`; do not commit them.
3. Generate the lock without modifying an existing lock:

   ```text
   python3 oracle/generate_cpu_oracle_lock.py \
     --material /home/emmy/gpt-oss-rs-artifacts/oracle-images/oracle-lock-material.json \
     --oci-archive /home/emmy/gpt-oss-rs-artifacts/oracle-images/cpu-oracle-INPUT_SHA.oci.tar \
     --sbom /home/emmy/gpt-oss-rs-artifacts/oracle-images/cpu-oracle.spdx.json \
     --provenance /home/emmy/gpt-oss-rs-artifacts/oracle-images/cpu-oracle.provenance.json
   ```

4. Verify the lock and archive, commit only `oracle/cpu-oracle.lock.json`, and
   push. This second commit is candidate A. The image label identifies the
   first commit; the lock proves which exact inputs candidate A selected.

Use a refreshed login or `newgrp docker` before local image work. Do not use
sudo and do not change Docker daemon configuration.

## Container contract

`oracle/cpu_oracle.py` rejects mutable/wrong references, wrong platform,
source/model/lock drift, changed image inputs or policy, corrupt archives,
unavailable Docker, visible CUDA, incomplete probes, and cross-mode host-key
changes. It runs the invoking UID/GID with a read-only root, no network, no
capabilities, no-new-privileges, read-only model/repository mounts, one
writable attempt mount, CPUs 0–3, four intra-op threads, one inter-op thread,
24 GiB memory, and no container swap beyond that memory limit.

The image contains no CUDA, Triton, torchvision, torchaudio, Transformers, or
installed `gpt-oss` distribution. The verified official source archive is
unpacked as source and imported through `PYTHONPATH`.

## Fresh root and preflight

Candidate A uses a new, initially nonexistent root on `/home` whose directory
name is the full candidate SHA. Keep the 40 GiB initialization gate and 20 GiB
reserve. Initialization creates a new empty cache and rehashes the model,
fixture/source/lock/image/archive/binary inputs. Example:

```text
newgrp docker
cargo build --release --locked -p gpt-oss-bench --bin cpu_validation
target/release/cpu_validation \
  --root /home/emmy/gpt-oss-rs-artifacts/cpu-validation/CANDIDATE_SHA \
  init \
  --oracle-lock oracle/cpu-oracle.lock.json \
  --oci-archive /home/emmy/gpt-oss-rs-artifacts/oracle-images/cpu-oracle-INPUT_SHA.oci.tar \
  --model /data/models/openai/gpt-oss-20b \
  --fixtures crates/gpt-oss-bench/fixtures/cpu_harmony_parity.json \
  --llama-source /home/emmy/src/cpu-runtime-research/llama.cpp-oracle-030ebb558 \
  --llama-model /data/models/llama-cpp/gpt-oss-20b/gpt-oss-20b-MXFP4.gguf
```

The preflight runs five repeat-identical BF16 fingerprints in both modes and
records Python, package/wheel pins, Torch version/git/configuration,
MKLDNN/OpenMP, dispatch capability, kernel/CPU/microcode/cgroup identity, and
the stable host key. A failed or unavailable preflight is terminal evidence,
not permission to fall back to a venv.

Every subsequent `run` command receives `GPT_OSS_ORACLE_IDENTITY_JSON`,
`GPT_OSS_CAMPAIGN_ROOT`, `GPT_OSS_ATTEMPT_DIR`, and
`GPT_OSS_ORACLE_LOCK`. Official captures run through the locked policy helper,
using paths inside the container such as `/model`, `/repo`, and `/attempt`.

## Ordered gates

Run only fresh outputs in this order:

1. native plus generic image/host qualification;
2. paired C3-X-001 capture and prefix localization;
3. 28 native/official comparisons and seven new llama.cpp scenario captures;
4. the complete model-free lifecycle/HTTP suite and one bounded 20B service
   session;
5. performance only after every correctness and service gate passes.

The driver locks the performance phase until the exact `c3-x-001` cell, all
seven scenarios crossed with `automatic`, `scalar`, `avx2`, and
`avx512-vnni`, seven exact `llama-cpp`/`ubatch-1` advisory cells, and both
`model-free-lifecycle-http` and `bounded-20b` service cells exist. The
`run_cpu_comparison.py`, `run_c3_x001.py`, `run_llama_cpu_batch.py`,
`select_llama_cpu_capture.py`, `run_model_free_service_suite.py`, and
`run_bounded_20b_service.py` workers keep their outputs in the attempt
directory and use the identity injected by the driver. A C3 mismatch triggers
five-repeat isolated prefix probes on scalar, AVX2, and AVX-512/VNNI.
Retries never inflate totals: finalization counts only the latest verified
terminal artifact per fresh cell. Generic comparisons never count.

If C3 requires a numerical correction, commit candidate B, create a new root
named with B's SHA, set A as its linked parent in the campaign record, and keep
the same oracle lock. Never overwrite A's evidence.

## Verification and closure

Before closure run the locked workspace build/tests, comparator and oracle
negative tests, formatting, warnings-denied Clippy for affected crates, forced
scalar/AVX2/AVX-512 lanes, portable AMX, image probes, documentation-link
validation, and the branch CPU workflow. A final report is valid only when it
names the candidate SHA and the final summary's E1 `artifact_set_sha256`.
Until the complete fresh matrix exists, the report must say incomplete and
must not repeat or combine any historical counts.
