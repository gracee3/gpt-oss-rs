# CPU Validation Evidence-Closure Campaign

## Outcome

The 2026-08-12 campaign is **paused at preflight**. The implementation and
model-free validation harness are complete, but the exact pinned oracle
interpreter `/data/models/.venv-awq/bin/python` was absent. The campaign did
not install, update, or substitute an environment. Consequently this record
does not claim CPU validation closure, trusted-mode eligibility, a new C3
outcome, official parity, llama.cpp parity, model-scale service validation, or
performance promotion.

| Item | Recorded value |
| --- | --- |
| Branch | `agent/cpu-validation-closure` |
| Candidate A | `fbfefdb48f0e1cd7f756bd76522976559c7b4faa` |
| Required baseline | `ac3eea350c2e926087f0b4eb67afa75ee5eecde1` |
| Campaign | `cpu-validation-20260812T032520Z-fbfefdb48f0e1cd7f756bd76522976559c7b4faa` |
| Campaign-index snapshot SHA-256 | `e8f9d841286c65320d119b3a3b6a7b800346457e745d4236581450dfee1199bf` |
| Preflight artifact SHA-256 | `9fe0d510c092cb06c378842e82d2127fd279eabc74d40af690ade528b8139923` |
| C3 | Not executed; no candidate B |
| Official matrix | `0/28` new authoritative comparisons |
| Advisory llama.cpp | `0/7` new captures |
| Model-scale service session | Not executed |
| Performance | Gated; not executed |

The external campaign directory contains the private absolute-path manifests,
raw command captures, build products, and isolated cache. They are not tracked
in Git. Its redacted and private initial preflight manifests have the same
artifact hash.

## Preflight evidence

The clean starting branch was the actual `origin/main` at the required
baseline SHA. Candidate A is a descendant of that baseline and was clean and
synchronized with its remote branch before the recorded preflight. The
official source checkout was clean at
`7802bf263f902efd4c7d18fcceff3ba72f941e80`; the llama.cpp checkout was clean
at `030ebb558a5820b444a8f836ed5cdd46c9b4bd7a`. The model snapshot, fixture,
GGUF, `Cargo.lock`, and release binaries were hashed in the external evidence
set. The isolated repack-cache inventory was empty at campaign start.

The artifact filesystem had about 221 GiB free after the captures, satisfying
the 40 GiB initial-free-space and 20 GiB reserve gates. The host snapshot also
recorded memory and pre-existing swap use. The sole campaign stop was the
missing pinned oracle interpreter. Resumption must use the restored,
pre-existing environment at the exact path; installing or substituting
dependencies would invalidate the intended oracle identity.

## Implemented evidence surface

Candidate A adds resumable campaign and attempt identities, stable cell keys,
binary and artifact hashes, requested/effective dispatch, timing/resource
fields, related-run references, strict completeness checks, and create-new
atomic publication to the existing evidence schema. Standalone captures are
non-authoritative; only the comparator can assign authoritative official
parity after checking workload and provenance. Negative or partial inputs
retain their negative status.

The campaign driver supports initialization, phase execution, resumption, and
finalization. It verifies terminal hashes before skipping a completed attempt
and allocates a new attempt for interrupted work. Native and official workers
preserve failure envelopes. The pinned llama.cpp driver builds outside its
clean source checkout, runs CPU-only with micro-batch one and prompt-cache
reuse disabled, and uses exact fixture token IDs. The offline-only C3 surface
captures layer-0 normalized inputs, pre-RoPE K/V BF16 bits, selected weight
rows and bias, dispatch, and prefix accumulators; no HTTP validation endpoint
was added.

## Verification completed without the model oracle

- `cargo build --release --locked` completed for the workspace.
- Evidence, comparator/scanner, kernel, model-runner, engine, server, and HTTP
  contract tests passed, including paused-time lifecycle, cancellation,
  slow-consumer, draining, join, and ledger/store/delivery cleanup cases.
- Forced scalar, AVX2, and AVX-512/VNNI kernel lanes each passed 36 tests.
- Portable AMX compilation and its 41 kernel tests passed; no native AMX
  execution is claimed.
- Formatting and `git diff --check` passed.
- Warnings-denied Clippy remains non-green because the baseline contains
  unrelated existing lints (including two in `gpt-oss-core` and broad
  model-runner test/library lint debt). Those surfaces were not changed merely
  to manufacture a campaign pass.

These results characterize the model-free CPU service foundation. They do not
replace the required bounded 20B service session.

## Resume and acceptance

After the owner restores the exact pinned oracle environment, resume this same
campaign index. Valid terminal attempt hashes will be retained and incomplete
work will receive new attempt IDs. C3-X-001 must run first. A numerical change
is permitted only after exact first-boundary localization; it would create a
separately committed candidate B and a linked SHA-named campaign root.

Full closure still requires an accepted `localized` or `not_reproduced` C3
result, 28 authoritative native/official comparisons, seven advisory
llama.cpp captures, the bounded model-scale service matrix, and qualifying
performance evidence or an explicit evidence-based insufficient
classification. Historical i7 results remain historical and are not
superseded by this paused campaign. Trusted mode remains disabled, and no
dispatch policy, kernel threshold, T14, Iris Xe, or CUDA behavior changed.
