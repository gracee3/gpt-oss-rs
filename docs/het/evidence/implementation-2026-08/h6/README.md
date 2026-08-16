# H6 gate record — real one-layer owner shell and three-owner oracle

**Status:** H6a passed on 2026-08-16; H6b remains in progress. This record does
not claim the complete H6 gate or real concurrent CPU/GPU0/GPU1 execution yet.

H6a runs the real 20B layer-0 decode fixture at position 63. The retained
`ResidualQ8` runner is authority only through the dense, private K/V, attention,
post-attention residual, router-input, and real route/weight boundaries. The
four expert outputs are independently recomputed from the native MXFP4 views
with `exact_selected_expert_reference`; the retained optimized MoE output is
not used as the H6 exact target. GPU0 then performs the H5 strict rank-ordered
reduction and the owner shell applies the final BF16 residual.

All twelve retained/exact boundaries are bit-exact. The real route is
`[31,21,22,6]`, with selected BF16 weight bits
`[16128,15926,15915,15903]`. The owner-shell-to-router execution handoff and
reducer-to-residual handoff each move 5,760 bytes device-to-device and move zero
execution bytes through host memory. Router activation/descriptor and reducer
trace downloads remain bounded evidence-only transfers. H6a's oracle-only
control uploads 23,040 bytes of exact CPU-authority contributions; H6b must
replace that control with actual resident CPU/GPU0/GPU1 work.

The retained v3 record was built with `heterogeneous-test-faults`, which
implies `cuda`. The shell owns a fixed 309,504-byte host staging arena for
prior-K/V and RoPE H2D sources plus all nine boundary/final-output D2H targets.
Every copy batch has an explicit terminal event. Five injected failures cover
post-key-H2D, prefix terminal handling, post-first-boundary-D2H,
post-residual-D2D, and post-final-output-D2H; each mandatory fallback drain is
followed immediately by a clean retry. An unproven fallback drain poisons the
shell and retains its CUDA state and any referenced host arena rather than
releasing storage that work may still address.

The selected-expert native view identity used by the oracle is SHA-256 over the
six exact expert surfaces (gate/up blocks, scales, bias and down blocks, scales,
bias), not the broader checkpoint-mapping identity.

## Records

| Record | Result |
|---|---|
| [`h6a-owner-shell.json`](h6a-owner-shell.json) | v3: twelve bit-exact real layer boundaries, five owner-shell lifecycle faults/retries, fixed host staging, real router IDs/weights, exact expert/reduction hashes, and resident-edge byte evidence |
| [`h6a-run-manifest.json`](h6a-run-manifest.json) | v2: final source/binary/four-PTX identities, exact Cargo feature set, focused lifecycle/regression gates, and post-run cleanup |
| [`SHA256SUMS`](SHA256SUMS) | Bounded evidence identities |

The final hardened source fingerprint is
`c82b4700102a2b72135f64f7b70b26239d50f116a697d7b65d9329783dd036f8`.
It uses the same H5 convention: sorted tracked and
untracked, non-ignored paths outside `docs/`, with path and SHA-256 separated by
NUL bytes. H6a is a checkpoint only. H6 passes only after H6b executes the same
real route concurrently on CPU, GPU0, and GPU1, proves correlated overlap,
uses the H5 transaction barrier, repeats cleanly, and preserves exact output.

No model was copied or transformed, no 120B execution occurred, nothing was
pushed, swap remained zero, both GPUs returned to idle, and `/dev/nvme1n1`
remained read-only and unmounted.

The final-source regression closeout passed the synthetic selected-expert gate
on both GPUs, the real four-expert/two-GPU oracle, all four synthetic/native
E=32/E=128 router cases, both relay fault/real-x8 cases, locked workspace
check/tests, the three configured strict Clippy lanes, and a fourth targeted
warnings-denied Clippy lane for the exact fault-gated bench binary.
