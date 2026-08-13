# Tiger Lake representative CPU corpus

Status: complete pre-promotion evidence; immutable external artifacts.

## Identity

- Source candidate: `12a3f5816f6864b484927a2a9e48c18a1fc92087`
- Binary SHA-256: `d19114198c63786a96ab3e9696c7f4cbf96521ebd61caefa8ee15baf174cefaa`
- Fixture SHA-256: `c79236fdcdd210a203139f32d4321322c106f7062680e31921c0709423d56f56`
- Artifact root: `/home/emmy/gpt-oss-rs-artifacts/tiger-lake-optimization/12a3f582f98fd6557f3c80e92c9a28ee5fdcb5b6/corpus-v2/`
- `SHA256SUMS` SHA-256: `860e3dae77a9471256b08abd418703e21e99a21af716a2484ca7e24859f15254`
- Warm summary SHA-256: `aca1150bd3cf6b8c3a47dd5ee3a5081adb784d9959be36b82b6131358051205e`

The root contains 28 fresh profiles: one warmup plus three warm repetitions
for each of the seven pinned scenarios. All 201 indexed files verify. The
profiles contain 261,924 fixed-width records with zero drops, zero truncation,
and zero failed transactions. Captures ran on CPUs 0-3 with four threads,
CPU-only Auto dispatch, layer-major prefill, eight bounded decode tokens, an
ordinary `powersave` host policy, AC power, and a 65 C cold-start admission
gate with 120-second cooldown intervals.

The source predates the topology correction in `42b7fdd`, so its profile
metadata reports the four-CPU task affinity as `logical_cpus=4`. This does not
alter the captured operation shapes or timings, but it is not the final
hardware-profile key. Promotion and final certification use the corrected
normalized 4C/8T, microcode `0xbe`, XCR0 identity.

## Observed workload

Gate/up and down had identical routed-bucket distributions. Across all 28
profiles, 40,188 projection buckets per role were observed at 260 distinct M
values spanning 1 through 412. Every observed value is retained in the raw
summary; grouped frequency is:

| Expert-bucket M | Events per role | Share |
| --- | ---: | ---: |
| 1 | 22,616 | 56.28% |
| 2 | 988 | 2.46% |
| 3 | 1,056 | 2.63% |
| 4-7 | 3,260 | 8.11% |
| 8-15 | 3,588 | 8.93% |
| 16+ | 8,680 | 21.60% |

The warm-only summary has the same shape: M=1 is 56.3%, M=4-7 is 8.1%,
M=8-15 is 8.9%, and M>=16 is 21.6%. The observed range and long sparse tail
rule out a single guessed crossover threshold; benchmarking must cover all
observed M values and use contiguous evidence-backed regions.

Exclusive coarse-operation time shares across the full corpus are:

| Operation | Share |
| --- | ---: |
| Gate/up MXFP4 projection | 65.62% |
| Down MXFP4 projection | 32.34% |
| Attention | 0.62% |
| Q and O BF16 projections | 0.81% combined |
| SwiGLU | 0.28% |
| Residual-Q8 preparation | 0.23% |
| All remaining recorded work | 0.11% |

Thus 97.95% of recorded time is in the two expert MXFP4 projections. The
preparation cost is small enough that a matrix-kernel win can affect complete
requests, but profiled operation durations alone are not used to rank kernels.

Warm wall-time medians were 193.08 s (`harmony_63`), 358.77 s
(`harmony_122`), 400.55 s (`harmony_136`), 770.89 s (`harmony_262`),
1,017.69 s (`harmony_346`), 1,308.64 s (`harmony_444`), and 529.24 s
(`tool_history_180`). Outliers are retained in the raw artifacts.

The pinned 63-token scenario is the smallest valid Harmony envelope near the
planned 32-token boundary; 122/136 cover the 128 neighborhood and 444 covers
the 512 neighborhood. The deterministic repeated-segment profiling control is
used separately for the 2048-token neighborhood without changing fixture
bytes or official-oracle cells.
