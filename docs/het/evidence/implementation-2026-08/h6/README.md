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

The selected-expert native view identity used by the oracle is SHA-256 over the
six exact expert surfaces (gate/up blocks, scales, bias and down blocks, scales,
bias), not the broader checkpoint-mapping identity.

## Records

| Record | Result |
|---|---|
| [`h6a-owner-shell.json`](h6a-owner-shell.json) | Twelve bit-exact real layer boundaries, real router IDs/weights, exact expert/reduction hashes, and resident-edge byte evidence |
| [`h6a-run-manifest.json`](h6a-run-manifest.json) | Package-start source/binary/PTX identities, focused authority-upload lifecycle fault result, and post-run cleanup |
| [`SHA256SUMS`](SHA256SUMS) | Bounded evidence identities |

The source fingerprint uses the same H5 convention: sorted tracked and
untracked, non-ignored paths outside `docs/`, with path and SHA-256 separated by
NUL bytes. H6a is a checkpoint only. H6 passes only after H6b executes the same
real route concurrently on CPU, GPU0, and GPU1, proves correlated overlap,
uses the H5 transaction barrier, repeats cleanly, and preserves exact output.

No model was copied or transformed, no 120B execution occurred, nothing was
pushed, swap remained zero, both GPUs returned to idle, and `/dev/nvme1n1`
remained read-only and unmounted.
