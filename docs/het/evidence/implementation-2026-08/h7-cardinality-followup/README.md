# H7 follow-up — complete owner-cardinality support

**Status:** passed on 2026-08-16. The H7 control path now accepts every
nonnegative `[CPU, GPU0, GPU1]` expert-count triple summing to top-k four. It
preserves canonical rank, route, owner, weight, and result-slot identity; uses
one fixed all-or-nothing dispatch reservation; and performs no successful-path
heap allocation after the first enqueue.

Attempt 01 stopped before producing an artifact because its generalized GPU1
timeline actor did not match the retained gate. Attempt 02 is retained
unchanged as `attempt-02-numerical-pass-global-swap-nonpassing.json`: arithmetic,
overlap, and cleanup passed, but an external preflight detected 3,376 KiB of
unrelated global swap growth. Neither attempt is promoted.

After four byte-exact samples spanning 123 seconds, attempt 03 began and ended
with `SwapFree=104836504 kB`, `SwapCached=1712 kB`, zero PSI memory pressure,
and target `VmSwap=0`. Both repetitions produced exactly:

```text
[200005, 35644, 200008, 976, 1825, 5003, 25, 392]
```

Both also retained the real `[GPU0, CPU, GPU1, GPU0]` route, exact packed and
completion descriptors, strict three-way compute intersection, bounded pools,
and deterministic cleanup. The original recoverable and unproven faults passed.
The all-remote `[0,0,4]` fixture additionally faulted after one remote result had
already submitted: the recoverable case drained and retried cleanly, while the
unproven case quarantined all five pinned leases and rejected reuse.

`h7-followup-run-manifest.json` binds source, executables, PTX, models, commands,
and regression results. `SHA256SUMS` covers every retained record. Prior H7
evidence under `../h7/` was not modified. H8 was not retried and H9 was not
started.
