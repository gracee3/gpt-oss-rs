# Tiger Lake Optimization Foundation Closure

This record closes the Tiger Lake optimization-foundation sprint for the
immutable implementation candidate
`24577826d9a5bf186656a0d419e5e93237c66a4c`. Later documentation and
integration commits do not change that evidence key. No output from a prior
candidate is counted in its fresh CPU-oracle certification.

## Identity and disposition

| Coordinate | Value |
| --- | --- |
| Starting checkpoint | `debb9d74d01af5f78deefc013364aba1129b49c1` |
| Implementation candidate | `24577826d9a5bf186656a0d419e5e93237c66a4c` |
| Candidate root | `/home/emmy/gpt-oss-rs-artifacts/cpu-validation/24577826d9a5bf186656a0d419e5e93237c66a4c/` |
| E1 artifact-set SHA-256 | `cf833a1ccc5dffcf14575890d9eeca02c447a2471f30150da03d340f29092a0c` |
| Oracle lock SHA-256 | `ba1c8a8bf3e527d3bdf8bcf0fde9069b4970420a4d7a5603154f10248c26b674` |
| Official source | gpt-oss v0.0.9, `599476783c6f88508dab8577808b5ead5cbee8d2` |
| Model revision | `6cee5e81ee83917806bbde320786a8fb61efebee` |
| llama.cpp revision | `030ebb558a5820b444a8f836ed5cdd46c9b4bd7a` |
| Container policy SHA-256 | `cb1762bdc25f0f25fae20ddff7d2969fa4af8829da25a1250b6c59db41af4060` |

The final campaign summary is
`publish/final-summary.json` below the candidate root. It reports a complete
campaign: one accepted C3 gate, 42 official comparisons, seven llama.cpp
advisory captures, two service cells, two performance records, and 56
terminal attempts. The second performance record supersedes only the first
record's malformed derived summary; its three immutable raw runs remain
valid and are retained in E1.

The sprint outcomes are deliberately separated:

- **Implemented:** structured CPU identity and dispatch diagnostics; bounded
  execution profiling; deterministic corpus summaries; a genuine AVX-512/VNNI
  multi-row matrix kernel; six-cell/42-comparison campaign support; and the
  bounded OpenCL expert LRU.
- **Promoted:** no new M>1 CPU region and no Xe behavior. Existing M=1 AVX2 x8
  remains unchanged.
- **Forced-only:** AVX2 and AVX-512/VNNI matrix controls, and explicit-Xe
  residency with a nonzero capacity.
- **Negative result:** the isolated M=3 AVX2 matrix win regressed paired full
  requests; the Xe cache won isolated repeated projection reuse but regressed
  the representative full request at every tested capacity.
- **Deferred:** dense BF16 matrices, attention kernels, fusion, Xe decode,
  Level Zero production, general autotuning, trusted-mode changes, and
  automatic Xe promotion.

## Certified host and runtime

The final model-free diagnostic identifies GenuineIntel family 6, model 140,
stepping 1, four physical cores/eight logical CPUs, microcode `0xbe`, OSXSAVE,
XCR0 `0x2e7`, and hardware-profile key
`GenuineIntel-family6-model140-stepping1-cores4-logical8-microcodebe-xcr02e7`.
AVX2/FMA and AVX-512F/BW/VL/VNNI are legal; AVX-VNNI, AVX-512 BF16, and AMX
are not. The resolved Auto plan uses AVX-512/VNNI for BF16 matvec,
quantization, and RMSNorm, AVX2 x8 for M=1 MXFP4, and scalar for every M>1
MXFP4 matrix problem. The crossover-region list is empty.

The host is Ubuntu 26.04 LTS on kernel `7.0.0-29-generic`. Iris Xe remains
PCI `8086:9a49`, Dell subsystem `1028:0a42`, on i915. Intel NEO
`26.05.037020` provides OpenCL 3.0, subgroups 8/16/32, integer dot products,
unified memory, SVM, Intel USM, and SPIR-V through 1.5. The installed Level
Zero loader/runtime passed independent enumeration, shared-allocation,
immediate-list, and lifecycle probes. Repository Level Zero production work
remains deferred because the pinned header corpus is absent.

## Bounded profiler and representative corpus

`gpt-oss-rs.execution-profile/v1` stores 104-byte fixed records in a
preallocated slab. The default 16 MiB cap holds 161,319 records. With no
output path the context stores no profiler and operation sites perform no
clock read, allocation, formatting, lock, or I/O. Publication and
deterministic summarization occur outside hot loops; truncation is explicit
and disqualifies a corpus from promotion.

Five cool-start rotated `harmony_63` pairs compared the candidate-disabled
path with exact pre-profiler commit `364f940`. Disabled profiling changed
mean full-request time by +0.130% and median by +0.140%; the paired bootstrap
mean 95% interval was -0.340% to +0.657%, so no regression was demonstrated.
Enabled versus disabled changed mean by -0.182% and median by -0.109%; its
interval was -0.516% to +0.055%, so no enabled latency cost was detected in
this sample. Enabled median RSS was 10,621,556 KiB versus 10,561,732 KiB
disabled, about 58 MiB higher. Every enabled profile wrote 51,262 records
with zero drops or truncation. The artifact root is
`/home/emmy/gpt-oss-rs-artifacts/tiger-lake-optimization/91f5c93d03f9d1e3d6b4a775cd6ef45328f0dd59/profiler-overhead-v2/`;
its `SHA256SUMS` hashes to
`57925e2a04960ca7e847f4a095e2e1cb47af74dc9578e2cb2f13cd224ec340a9`.

The representative corpus contains 28 fresh profiles: a warmup and three warm
runs for each pinned scenario. Its 261,924 records have zero drops,
truncations, or failed transactions. The external root and detailed protocol
are recorded in [`TIGER_LAKE_CPU_CORPUS.md`](TIGER_LAKE_CPU_CORPUS.md); its
index hash is
`860e3dae77a9471256b08abd418703e21e99a21af716a2484ca7e24859f15254`.

Gate/up and down each observed 40,188 expert buckets over 260 distinct M
values from 1 through 412. M=1 accounts for 56.28%, M=2 for 2.46%, M=3 for
2.63%, M=4-7 for 8.11%, M=8-15 for 8.93%, and M>=16 for 21.60%. Exclusive
operation time is dominated by gate/up (65.62%) and down (32.34%); attention
is 0.62%, Q/O BF16 projections 0.81% combined, SwiGLU 0.28%, and residual-Q8
preparation 0.23%.

## CPU matrix result

The new candidate is an 8-input by 8-output ZMM microkernel using
`VPDPBUSD` and the existing `InterleavedSplitX8V2` layout. It allocates
nothing, uses caller-owned aligned scratch, supports Q8 and residual-Q8, and
preserves scalar bias and primary-then-residual accumulation boundaries.
Disassembly and benchmark metadata confirm that complete x8 output groups
execute the intended VNNI path; tails retain the canonical scalar contract.

At thermally valid residual-Q8 M=3, K=2880 shapes, forced AVX2 beat both
scalar and AVX-512/VNNI in isolated microbenchmarks: 5.028 ms for N=5760 and
2.464 ms for N=2880. The paired 95% intervals established both wins. The
indexed benchmark root hashes to
`106f81536bba49a22f8b7f6e64c85d9fbce9e55d8331f10382cb59bac446d989`.

Candidate B enabled only that region. Three cool-start, order-alternating
full-request pairs then measured regressions of 0.522%, 0.653%, and 0.345%;
the mean was +0.507% with 95% interval +0.345% to +0.653%. Its evidence index
hash is
`532f0b9ec2151043e5af829446b008ccfc7938d4c3f2087bb050df7d31d2a7eb`.
Candidate C therefore encodes a negative promotion record: there are no
promoted regions. Unknown profiles, different thread policies, unobserved or
thermally ineligible shapes, and every known M>1 shape resolve to scalar.

Candidate C's final three warm automatic `harmony_63` requests completed in
26.696, 27.058, and 26.773 seconds, median 26.773 seconds, with exact official
tokens. Start temperatures were 42, 43, and 45 C. The raw-capture index hashes
to `34dba188f8880f29326688524bfe10984444273b69559abe74a9c576fd500c97`.
Forced whole-runtime AVX2 remains an informative explicit comparator, but
changing the existing hybrid operation policy was outside this matrix-only
promotion and is retained as a documented follow-up opportunity.

## Xe residency result

The explicit OpenCL path now supports a strict byte-bounded deterministic LRU
of immutable expert weight/bias buffers. Its identity includes source tensor,
layer, expert, projection role, dimensions, layout/ABI/source/build, PCI, and
runtime facts. Misses repack lazily; hits avoid both repack and upload.
Oversized entries bypass, faults drain and trip the existing circuit breaker,
and one CPU recomputation preserves transactional output. The default is zero,
the option is legal only with explicit `--device xe`, and decode/SwiGLU/state
remain on CPU.

All 22 live OpenCL tests passed. At 128 MiB an isolated repeated layer-0 gate
measured 482/484 hits, avoided 3,194,156,160 upload bytes, held 13,253,760
resident bytes, and had no eviction or fault. M=4 gate/up fell from 10.868 ms
streaming to 1.797 ms resident; down fell from 7.336 to 1.607 ms.

The full-model result was negative. Zero cache completed `harmony_63` in
20.114 seconds. After cache priming, 128/256/512 MiB each produced zero hits,
846 misses, and 5,606,340,480 uploaded bytes in the measured prefill; total
times were 21.086, 21.228, and 21.138 seconds. All outputs remained exact and
fault-free. The relevant index hashes are
`e360ab26f721549ea92f92d3a58088c7a506cab3812b4e200afb3ac266492911`
for the isolated win and
`2476b97c21bc8bc6701d4749ad90154397eba94c160e6d6242ffb8ac4ef9a9c5`
for the full-model negative result. No capacity is selected and automatic Xe
remains disabled.

## Fresh correctness and service certification

The Candidate C root began nonexistent with an empty repack cache and 143 GiB
free, exceeding the 40-GiB initialization and 20-GiB reserve gates. Native
and generic image/host qualification passed against the unchanged published
OCI material. C3-X-001 did not reproduce at absolute row 267; native and
official K/V comparison passed, so the accepted outcome is
`insufficient_evidence` with no numerical correction.

All 42 official native/PyTorch comparisons passed exact generated-token
parity: seven scenarios crossed with `automatic/auto`, `scalar/auto`,
`avx2/auto`, `avx512-vnni/auto`, `automatic/avx2`, and
`automatic/avx512-vnni`. The last two are distinct forced matrix backends.
Seven fresh llama.cpp/ubatch-1 advisory cells completed in one candidate-local
session; the combined capture SHA-256 is
`a0820e05dcea7d85b42b374995ff94a28d072ae79982c06b81a4bedb2be8cc36`.
As in the historical campaign, this pinned server revision leaves the legacy
top-level `tokens` arrays empty; retained `completion_probabilities` contain
the generated-token evidence. The advisory captures neither fail nor waive
official parity.

The locked model-free lifecycle/HTTP suite passed. The bounded 20B service
became ready in 1.516 seconds, passed its lifecycle probe, returned HTTP 200
for the one-token chat request in 181.242 seconds, and shut down cleanly.

## Validation and limitations

Candidate C and the documentation closure passed `cargo fmt --all --
--check`, `git diff --check`, locked workspace check/test, all three 44-test
forced CPU-kernel lanes, three targeted AVX2 matrix tests, four targeted
AVX-512/VNNI matrix tests, the 49-test portable AMX lane, and AMX
warnings-denied Clippy. Warnings-denied Clippy also passes for the CPU kernel,
Xe, evidence, and benchmark crates. The 22-test opt-in live OpenCL suite
passes on the installed Intel stack. Default Xe-enabled and CPU-only
no-default release server builds pass, as do all CPU campaign binaries and the
Xe projection gate build. Dynamic-link inspection confirms the OpenCL loader
is resolved at runtime rather than linked into the server or gate binary.

All 82 repository Markdown files pass relative-link validation; all 35
campaign/comparator/summarizer Python tests and ten oracle negative tests
pass. The immutable lock/archive re-verifies. An independent closeout pass
rehashes all 56 terminal manifests and all 511 artifacts declared by them,
covering 30,285,043 bytes, and reproduces E1 exactly.

The official warnings-denied workflow lanes pass. Enabling warnings-denied
over broader legacy surfaces still exposes 88 model-runner findings, 24
engine findings, and one server `derivable_impls` finding. The dedicated
kernel, Xe, evidence, and benchmark lanes are clean; the broader lint debt is
not hidden, and changing it after evidence freeze would require another
implementation candidate and campaign. Performance evidence does not claim a
universal power-policy result, and thermally throttled matrix shapes remain
ineligible rather than averaged into a promotion claim.

The branch GitHub workflow and final integration disposition are attached to
the pull request created from the documentation closure commit.

Historical closure
[`CPU_FRESH_ORACLE_CLOSURE.md`](CPU_FRESH_ORACLE_CLOSURE.md) remains scoped to
`af6c0a2`. Candidate A `841a805` was superseded when fresh initialization
found a stale hard-coded fixture hash. Candidate B `91f5c93` was superseded
after its M=3 promotion failed the full-request gate. Candidate B's incomplete
campaign E1 is retained only as negative provenance and does not contribute
to the Candidate C counts above.
