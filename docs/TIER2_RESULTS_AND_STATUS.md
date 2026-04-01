# Tier-2 Results And Status

## Current Status

Restricted-fp16 CUDA Tier 2 is no longer treated as an undirected runtime bug-hunt.

The current repo position is:

- raw global compare is telemetry
- runtime-emulated global compare is localization
- same-input local replay is ownership proof

That contract is now the operative standard for future Tier-2 claims.

## Current Findings

Recent shallow, mid, and late candidate owners did not survive exact or corrected same-input replay.

- `layer0`: no surviving local owner
- `layer1`: no surviving local owner
- `layer12`: no surviving local owner
- `layer23`: no surviving local owner

These overturned provisional owner claims that had looked plausible from global-only evidence.

## What The Investigation Showed

- A first-visible global delta is not enough to assign ownership.
- Runtime-emulated compare is useful for localization, but still not ownership proof.
- Same-input replay can invalidate apparently strong layer/path suspicions.
- The most valuable output of the recent debug loop is the harness and workflow, not a settled runtime root cause.

## What Is Resolved

- The Tier-2 workflow now has explicit seed-capture and local-replay hooks.
- The repo has an operational path for representative sentinel-layer checks.
- The repo has enough harness structure to keep future live-testing disciplined.
- The recent replay/debug work has been converted into a workflow and evidence standard rather than left as branch-local lore.

## What Is Still Unresolved

- No final restricted-fp16 CUDA runtime bug owner has been proven by same-input replay.
- The remaining global-vs-local gap is still real even when provisional local owners were overturned.
- Additional representative live testing may still expose a surviving owner, but that result has not been demonstrated yet.

## What Is Incomplete Rather Than Proven Wrong

- Forward runtime work is incomplete.
- Some integration and parity branches still contain potentially useful ideas or fixes.
- Additional ergonomics and harness polish remain worth doing even without a settled semantic conclusion.

Incomplete does not mean merged fact. Keep unfinished runtime/semantic work isolated until it clears the Tier-2 evidence bar.

## Immediate Next Steps

- Continue using representative seed capture and local replay instead of broad speculative frontier chasing.
- Prefer harness, docs, and live-testing workflow improvements on mainline paths.
- Keep future runtime/semantic experiments on isolated forward branches until same-input replay supports a real ownership claim.
