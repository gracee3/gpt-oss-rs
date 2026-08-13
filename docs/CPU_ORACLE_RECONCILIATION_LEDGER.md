# CPU Oracle Reconciliation Ledger

The fresh oracle lineage diverged from the unified CPU/Xe main line at
`f86674d6`. It was transplanted file-by-file onto baseline `6caf274` without a
merge or cherry-pick. Published closure from `af6c0a2` remains historical,
candidate-scoped evidence and does not certify this branch.

| Fresh commit | Disposition |
| --- | --- |
| `3c433a3` | Imported: resumable validation, comparison and capture tools, evidence schemas, lifecycle integration, and offline dense-boundary tracing. |
| `072d491` | Imported: validation-root parsing correction. |
| `f1937a3` | Imported: immutable campaign workflow, oracle image inputs, official campaign tools, bounded service suite, and link checks. Overlapping engine/model/server files were three-way composed with the unified main versions. |
| `af6c0a2` | Imported: immutable image lock, atomic publication checks, archive validation, and negative tests. The lock and image inputs retain their published bytes. |
| `72ed57f` | Imported as historical documentation only; its closure explicitly applies to `af6c0a2`, not the unified implementation candidate. |

Main's Xe capture/runtime fields, official greedy-token fixture changes, and
response-store grant-release behavior are independently present and retained.
The official source identity is the fresh lineage's GPT-OSS `v0.0.9` identity.
No fresh-lineage change was intentionally omitted; superseded contextual
documentation is retained with this ledger defining its scope.
