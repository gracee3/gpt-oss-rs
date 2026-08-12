# CPU Service Foundation Implementation

- Branch base: `main` at `d222d72`
- Implementation branch: `agent/cpu-service-foundation`
- Authorized scope: E1 observability, C2 logical reservations, and C1 native-CPU lifecycle/delivery
- Explicitly excluded: C3-C7, paging, trusted-mode promotion, kernel/dispatch tuning, and CUDA behavioral changes

## Implementation status

Implemented on `agent/cpu-service-foundation` in dependency order. Model-free
evidence, reservation, delivery, lifecycle, route, and CLI tests are green.
The bounded existing-20B closeout smoke passed readiness and served-identity
checks, concurrent suffix-only streaming and non-streaming requests, isolated
disconnect recovery, and SIGTERM drain/join. Captures and E1 manifests remain
outside Git. The release model-free paired delivery/Prometheus A/B passed with
0% median-throughput regression and 1.275% median p99-latency regression
against the 1%/2% gates. No model benchmark, oracle, or tuning campaign was
run.

## Dependency order

1. Establish the versioned evidence crate, canonical runtime snapshots,
   bounded diagnostics, and bounded metric vocabulary.
2. Add checked logical reservation accounting and separate process-memory
   observations from virtual mappings, disk bytes, and allocator omissions.
3. Replace cumulative CPU publication with committed suffix events and a
   byte-charged, nonblocking delivery handoff.
4. Add managed admission, readiness, owner supervision, draining, bounded
   terminal response storage, and route-level typed failures.
5. Verify model-free contracts first, then run the existing CPU workflow and
   the bounded tiny-model shutdown smoke described in the charter.

## Non-negotiable invariants

- A request is granted before entering the canonical CPU sequence table.
- Reservation arithmetic uses checked `u128`; denial is atomic and release is
  idempotent.
- The CPU owner never awaits a client or route-owned delivery queue.
- Published text is suffix-only and cannot later be retracted by a stop string.
- Terminal control events retain reserved delivery capacity and stay ordered.
- Local source paths never become served model IDs or metric label values.
- Readiness is true only after the effective runtime snapshot is frozen.
- Batch source remains present, but no batch route is mounted.
