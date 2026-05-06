# Oracle Forward Python Env Baseline Plan

## Classification

```text
oracle_forward_python_env_baseline_plan_recorded
```

## Scope

This is a docs-only plan for a forward Python/Torch baseline for future
oracle/source-attribution work. It does not create a virtual environment,
install packages, modify requirements files, rerun oracle probes, change
runtime code, clone PyTorch, or build PyTorch.

Future implementation branch:

```text
oracle/forward-python-env-baseline
```

Primary objective: create a reliable forward Python/Torch baseline for future
oracle/source-attribution work while preserving prior virtual environments for
historical artifact reproducibility.

Historical/provenance environments:

```text
/home/emmy/openai/gpt-oss/.venv
/data/models/.venv-awq
```

These environments produced prior artifacts and must not be overwritten,
repurposed, or silently replaced by the forward baseline.

## Workspace Layout

Recommended future layout:

```text
/home/emmy/openai/.venvs/gpt-oss-oracle-py312-cu130
/home/emmy/openai/.venvs/torch-wheel-attribution
/home/emmy/openai/.venvs/pytorch-src-cpu
/home/emmy/openai/pytorch
/home/emmy/openai/pytorch-research
```

Layout rules:

- `gpt-oss-rs` worktrees remain in `/home/emmy/openai/worktrees/`.
- `/home/emmy/openai/gpt-oss-rs` remains the main repository.
- `/home/emmy/openai/pytorch` is only for future PyTorch source checkout.
- `/home/emmy/openai/pytorch-research` is for source-attribution logs and
  statuses.
- source editable builds must not share the forward oracle environment.
- installed-wheel attribution and source editable builds should remain
  isolated.

## Baseline Recommendation

- Prefer Python 3.12 for the forward oracle/source-attribution environment.
- Treat Python 3.13 plus Torch `2.11.0+cu130` from
  `/home/emmy/openai/gpt-oss/.venv` as historical/provenance unless explicitly
  revalidated.
- Prefer a CUDA-enabled Torch wheel family only so CUDA availability can be
  recorded; oracle attribution remains CPU-first unless explicitly authorized.
- Single-GPU work, if ever required, defaults to GPU1 because displays are on
  GPU0.
- Do not load the full Torch model on GPU from this environment unless a
  separate GPU/sharding plan authorizes it.

## Planned Requirements Files

Do not add requirements files in this docs branch. The implementation branch
should create and validate them:

```text
requirements/oracle-forward-py312-cu130.in
requirements/oracle-forward-py312-cu130.txt
requirements/oracle-forward-py312-cu130.constraints.txt
requirements/oracle-legacy-observed.txt
```

Intended roles:

1. `oracle-forward-py312-cu130.in`

   Human-maintained desired package set. It should include `gpt-oss`,
   `transformers`, and attribution utilities.

2. `oracle-forward-py312-cu130.constraints.txt`

   Pinned package versions selected after validation. It should pin Torch,
   NumPy, Transformers, Accelerate, Triton, `kernels`, `safetensors`,
   `huggingface_hub`, `openai-harmony`, `gpt-oss`, and any probe dependencies.

3. `oracle-forward-py312-cu130.txt`

   Installable requirements entrypoint. It may use the constraints file and
   should avoid ambiguity around the Torch index URL.

4. `oracle-legacy-observed.txt`

   Records historical environments and observed versions for:

   ```text
   /home/emmy/openai/gpt-oss/.venv
   /data/models/.venv-awq
   ```

   These entries are for provenance/reproduction, not the forward baseline.

## Package Families To Plan

Plan these package families for the future baseline:

- Python 3.12
- Torch CUDA 13.0 family, if compatible and validated
- NumPy
- Transformers
- Accelerate
- Triton, with `triton==3.4` as the starting target unless newer `gpt-oss`
  guidance explicitly supersedes it
- `kernels`
- `safetensors`
- `huggingface_hub`
- `openai-harmony`
- `gpt-oss`
- `packaging`
- `pytest`, only if needed for probes
- `rich` or `tabulate`, only if already useful

Do not hard-code final exact package versions in this docs branch unless they
are already proven locally. The implementation branch must create the
environment, install candidate packages, run import/probe validation, freeze
exact known-good versions, update requirements files, and emit a status JSON.

## Future Status Contract

Future status path:

```text
/tmp/oracle_forward_python_env_baseline_status.json
```

Allowed future classifications:

- `oracle_forward_python_env_baseline_recorded`
- `oracle_forward_python_env_baseline_validated`
- `oracle_forward_python_env_baseline_blocked_by_python`
- `oracle_forward_python_env_baseline_blocked_by_torch_install`
- `oracle_forward_python_env_baseline_failed`

Future implementation validation must record:

- Python executable
- Python version
- `sys.prefix`
- pip version
- `torch.__version__`
- `torch.version.git_version`, if available
- Torch import path
- `torch.__config__.show()`
- `torch.cuda.is_available()`
- `cuda_used = false`
- tiny CPU BF16 `torch.addmm` sanity
- NumPy version
- Transformers version
- Accelerate version
- Triton version
- `kernels` import status
- `safetensors` version
- `huggingface_hub` version
- `openai_harmony` import status
- `gpt_oss` import status
- pip freeze output path

Required guard fields:

- `validation_only = true`
- `oracle_device = "cpu"`
- `cuda_used = false`
- `pytorch_clone_performed = false`
- `pytorch_build_performed = false`
- `backend_selected = false`
- `implementation_authorized = false`
- `consumer_revalidation_authorized = false`
- `runtime_behavior_changed = false`
- `production_routing_changed = false`
- `cuda_kernels_changed = false`
- `output_emitted = false`
- `ladder_continued = false`

## Future Tiny Sanity Checks

The future implementation branch should:

- import Torch
- import NumPy
- import Transformers
- import Accelerate
- import `safetensors`
- import `huggingface_hub`
- import `openai_harmony` if installed/import name is known
- import `gpt_oss` if installed/import name is known
- create tiny CPU BF16 tensors
- run `torch.addmm(bias, input, weight_t)`
- assert output device is CPU
- assert `cuda_used = false`

Do not require model loading in the baseline validation branch. Do not rerun
Workstream A oracle artifacts in the baseline branch. Do not compare new-env
outputs to old-env official artifacts until a later explicitly authorized
rebaseline branch.

## Rebaseline Policy

- Historical artifacts remain tied to their recorded environment.
- New forward artifacts must include forward environment identity.
- Cross-env comparisons require explicit status metadata.
- If new Torch/Python changes outputs, record that as new baseline behavior,
  not silent replacement.
- Old virtual environments must remain available until the project explicitly
  retires them.

## Future Branch Sequence

1. `docs/oracle-forward-python-env-baseline-plan`

   This docs-only branch.

2. `oracle/forward-python-env-baseline`

   Create and validate the forward virtual environment and requirements files.

3. `oracle/fused-linear-addmm-forward-env-smoke`

   Optional tiny sampled seam smoke, with no full rebaseline.

4. `oracle/fused-linear-addmm-forward-env-rebaseline`

   Only if explicitly approved.

## Relationship To Current Workstream A

The installed Torch wheel dispatch attribution selected
`/home/emmy/openai/gpt-oss/.venv/bin/python` with Torch `2.11.0+cu130` and
git version `70d99e998b4955e0049d13a98d77ae1b14db1f45`. That environment is
now historical/provenance for the recorded Workstream A artifacts and source
attribution status.

The forward baseline is for future oracle/source-attribution work. It must not
silently replace old artifacts or reinterpret previous exactness claims.

## Implementation Attempt Result

Implementation branch:

```text
oracle/forward-python-env-baseline
```

Status:

```text
/tmp/oracle_forward_python_env_baseline_status.json
```

Classification:

```text
oracle_forward_python_env_baseline_blocked_by_python
```

Result:

- `python3.12` was not available on the host.
- The forward environment
  `/home/emmy/openai/.venvs/gpt-oss-oracle-py312-cu130` was not created.
- No packages were installed.
- No requirements files were written because validation could not proceed.
- No pip freeze was written.
- Historical/provenance environments were observed without modification:
  - `/home/emmy/openai/gpt-oss/.venv`: Python 3.13.7, Torch
    `2.11.0+cu130`, Torch git
    `70d99e998b4955e0049d13a98d77ae1b14db1f45`, NumPy not importable.
  - `/data/models/.venv-awq`: Python 3.13.7, Torch `2.10.0+cu128`, Torch git
    `449b1768410104d3ed79d3bcfe4ba1d65c7f22c0`, NumPy `2.4.2`.

Next bounded step: make Python 3.12 available, then rerun the forward baseline
branch to create the environment, install candidate packages, validate imports,
freeze known-good versions, and write requirements files.

## Guardrails

- Docs-only.
- No virtual environment creation.
- No package installation.
- No requirements file changes in this branch.
- No PyTorch clone.
- No PyTorch build.
- No runtime implementation.
- No backend selected.
- No consumer revalidation.
- No CUDA mirror.
- No runtime/default/CUDA behavior change.
- No output emission.
- No ladder continuation.
- No correction metadata promotion.
- No tolerance pass.
- No final-logit/all-layer/server/4097 claim.
- No Torch runtime dependency in Rust.
