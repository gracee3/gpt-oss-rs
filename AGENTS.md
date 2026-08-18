# Contributor and agent guidance

`gpt-oss-rs` is a CPU-first Rust research implementation for GPT-OSS. Version
0.1.0 closes the planned research program; `main` is in maintenance for
correctness, reproducibility, security, attribution, and evidence repairs. The
named heterogeneous and multi-GPU branches are archives, not active runtime
surfaces to merge back casually.

Before changing code, read `README.md`, `CONTRIBUTING.md`,
`docs/PROJECT_INTENT.md`, `docs/NEXT_MILESTONES.md`, and
`docs/research/RESEARCH_ETHICS.md`, plus the documentation for the affected
crate or evidence bundle.

## Ordinary validation

The reviewed lightweight portfolio check is model-free and GPU-free:

```bash
python3 -m unittest discover -s crates/gpt-oss-bench/tools/tests -p 'test_*.py'
git diff --check
```

Broader Rust, model, CUDA, benchmark, evidence-bundle, or release checks must
match the changed surface and the repository's validation documentation. Do not
load models, run GPUs, build containers, prepare datasets, or execute benchmarks
without separate explicit authorization.

## Evidence, provenance, and delivery

- Keep claims bound to exact source, model, hardware, workload, and evidence
  identities. Preserve negative results and distinguish verified facts from
  inference or proposals.
- Do not commit model payloads, credentials, private data, raw host captures,
  local paths, generated caches, or unreviewed benchmark output.
- Retain inherited authorship, notices, citations, adapted-source attribution,
  and applicable model/software licenses. Do not rewrite archives to simplify
  the publication story.
- Use a focused feature branch. Commit and push the validated change and open a
  pull request; incomplete or higher-risk work stays draft.
- After publication, send the exact commit, PR, validation, outcome, risks, and
  next action to the repository's external coordination record. Do not claim
  completion until that remote handoff is verified.
