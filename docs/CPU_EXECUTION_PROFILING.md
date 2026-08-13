# Bounded CPU Execution Profiling

`gpt-oss-rs.execution-profile/v1` is an opt-in, model-content-free operation
trace. Enable it for the server with `--cpu-profile-output PATH`; the enabled
record slab defaults to 16 MiB and may be changed with
`--cpu-profile-cap-mib N`. A cap without an output path is rejected. The
`cpu_parity` capture tool exposes the same controls.

With no output path, `CpuExecutionContext` stores `None`: operation sites take
one option branch and perform no clock read, allocation, formatting, lock, or
I/O. Enabling profiling allocates one fixed record slab before execution.
Coarse operation boundaries append only numeric enums, dimensions, counters,
and monotonic nanoseconds. When full, the slab stops accepting records and
increments a dropped counter. A truncated capture is valid diagnostic output
but is rejected by the summarizer for crossover decisions.

Records include phase and mixed-batch row counts, operation and layer,
M/N/K, exact expert-bucket M, context length, requested/effective CPU and
matrix controls, thread count, projection and attention class,
preparation/residency/fallback state, scratch/resident high-water, and an
explicit transaction state. Failed preparation records are labeled `failed`;
successful model work is labeled `prepared`, not committed. No prompt text,
tokens, logits, output text, sequence ID, or expert ID is recorded.

Serialization occurs outside model loops during an explicit offline flush or
graceful service shutdown. Publication uses a synced temporary file plus a
no-replace hard-link operation, so an existing capture is never overwritten.
The document names source/model/hardware/runtime/dispatch identity, command,
timer semantics, capacity, dropped count, time bounds, and a hash of its
record array.

Summarize one or more complete captures with:

```text
python3 crates/gpt-oss-bench/tools/summarize_cpu_profile.py \
  capture-*.json --output combined.json --report combined.txt
```

The deterministic summary validates record hashes, rejects truncation, keeps
failed work out of committed-operation timing totals, and reports operation
shares, shapes, backend codes, expert-bucket distributions, and memory
high-water values.
