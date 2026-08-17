# R4 retained-20B comparison evidence

Verdict: `blocked_pre_model`

The authorized R4 implementation is commit `eb1d816b8c4ea8be809d0f82bc5c6fa897c7322c`
on branch `agent/r4-retained-20b-comparison`. That commit was pushed before the
release binaries were built. No PR or merge was created.

## Attempt identity

- R2 policy SHA-256:
  `f269a4c984bbfa0d2a18c037b42ded2c81330094b18c6fc8dc668b7ad81bb90f`
- supervisor SHA-256:
  `2a09c16598e3d926c1c6a54f08694165af52cf02adb1b31ca747fe896d0a88f0`
- immutable raw preflight:
  `/home/emmy/workspace/gpt-oss-rs-het-r4-20260817-011643/preflight.json`
- raw preflight SHA-256:
  `821c40dae25f9c9b197773ef4713591f13a039e2dfba62af8950fab0a5b87d65`
- sample-chain SHA-256:
  `2d6ed3c0325328e1b06df7587789ffdd6a694983a95bd8eb4e19a83f53d0ea3d`
- observation window: 2026-08-17 01:16:55–01:18:55 EDT; 120,024 ms;
  six samples

The full immutable record remains outside Git to avoid retaining an
unnecessarily broad host snapshot. [`preflight-summary.json`](preflight-summary.json)
contains the bounded decision facts and binds the raw record by hash.

## Admission result

The preflight failed both of these hard R2 predicates:

1. comparison-cgroup `memory.swap.current` was 1,146,880 bytes rather than
   exactly zero in every host sample; and
2. PSI `some/full avg10` was 0.07 at 120,000 and 120,024 ms.

No observation read failed. Global swap used stayed exactly 11,460,608 bytes,
target-tree swap stayed zero, `SwapFree` and `SwapCached` happened to remain
byte-stable, attribution was complete, and minimum `MemAvailable` was
96,249,933,824 bytes. The two GPUs stayed at 9 MiB and 1 MiB used, respectively,
with more than the frozen free-memory reserve. The protected NVMe stayed
read-only and unmounted.

The supervisor exited before its `run` subcommand. Consequently:

- no 20B model or checkpoint payload was opened;
- no monolithic or capacity-one construction cell ran;
- no new comparison cache was created;
- no H7 repeat ran;
- no H8 or 120B process/input/access began; and
- no host, cgroup, swap, network, storage, or GPU setting changed.

The pre-existing `/home/emmy/workspace/gpt-oss-rs-het-cache` inventory hash was
`06cbcaf72a2986ad918c8b64ba818e718d292ac0b7f3b89ee9aa9ae5a43255d6`
both before and after the attempt. [`validation.tsv`](validation.tsv) records
the implementation and live-gate results. `SHA256SUMS` authenticates the
bounded files in this directory except itself.

## Interpretation

This is a valid fail-closed admission result, not constructor evidence. It does
not select a fallback and does not weaken R2. A new attempt requires separate
authorization after the exact cgroup-swap and full-window PSI prerequisites are
true. H8/H9/H10 remain stopped.
