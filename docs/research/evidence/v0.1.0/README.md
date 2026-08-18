# v0.1.0 evidence bundle

**Status:** awaiting final controlled capture.

This directory is the publication location for the bounded v0.1.0 CPU evidence.
The completed bundle will contain:

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

Model payloads, repack caches, build trees, raw logs, credentials, hostnames,
serial numbers, and local personal paths are not published.

Validation command after capture:

```bash
python3 crates/gpt-oss-bench/tools/publication_benchmark.py validate \
  docs/research/evidence/v0.1.0
```
