# v0.1.0 evidence bundle

**Status:** complete; captured and independently revalidated on 2026-08-17.

This directory is the publication location for the bounded v0.1.0 CPU evidence.
The bundle contains:

- `publication-benchmark.json`: normalized full-model samples, exact tokens,
  source/model/binary and sanitized host identity, run order, aggregate
  analysis, and divergence policy;
- `mxfp4-gate-up.json` and `mxfp4-down.json`: exact-bit raw matrix samples;
- `mxfp4-analysis.json`: 10,000-iteration paired bootstrap result;
- `conversion.json`: local SafeTensors-to-GGUF provenance and dependency
  freeze identity;
- `environment.json`: release-time compiler, operating-system, CPU, and
  accelerator health without hostnames or serial numbers;
- `SHA256SUMS`: hashes of every published artifact; and
- [`schema/publication-benchmark-v1.schema.json`](schema/publication-benchmark-v1.schema.json):
  versioned report schema.

All four full-model lanes matched the official eight-token sequence in all five
measured trials. The only reportable Auto-versus-scalar speedup was decode
throughput: 6.05x with a paired 95% interval of 5.59x-6.23x. Auto prompt and
full-request latency were slower and received no speedup claim. Five MXFP4
candidate regions qualified; gate/up `M=8` and seven other candidate rows
remained non-qualifying.

The converted GGUF SHA-256 differs from the historical fixture SHA-256. Both
identities and the no-download policy are recorded in `conversion.json` and
`publication-benchmark.json`.

Model payloads, repack caches, build trees, raw logs, credentials, hostnames,
serial numbers, network identifiers, and local personal paths are not
published.

Validation command after capture:

```bash
python3 crates/gpt-oss-bench/tools/publication_benchmark.py validate \
  docs/research/evidence/v0.1.0
```

The JSON report is also checked against the versioned schema, and
`SHA256SUMS` covers every file in this directory except the checksum index
itself.
