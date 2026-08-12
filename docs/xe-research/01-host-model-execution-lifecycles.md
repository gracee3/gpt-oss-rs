# X1 — Host, Model, and Execution Lifecycles

- Result: decision-complete ownership model for a future forced-only path
- Current implementation scope: synchronous research harness only
- Safety rule: GPU output cannot alter committed model state before validation

## Connected ownership model

```text
XeApi (process)
  -> exact loader + selected driver/device
  -> XeContext (future: one per attached model/backend)
     -> XeModule + native cache identity
     -> queue/list pool + event/fence pool
     -> XeModelResources
        -> one compact persistent weight region or one bounded residency slab
        -> reusable activation/output scratch
     -> XeExecution
        -> borrowed model resources + exclusive queue/list lease
        -> completion token
        -> validated, uncommitted output
        -> commit OR discard and CPU recomputation
```

The research `Session` intentionally implements only the synchronous subset.
It is not `Clone`, does not expose its raw handle, and borrows every `Buffer` so
a buffer cannot outlive its context. Partial C initialization uses zeroed
handles and one reverse-order destroy path. Rust `Drop` calls that path exactly
once. The future production boundary must preserve these properties while
adding an explicit, idempotent `shutdown()` that first stops new leases, waits
for or invalidates in-flight work, releases executions, model allocations,
events, lists/queues, modules, context, and finally the API/driver handle.

## API object mapping

| Responsibility | OpenCL objects and completion | Level Zero objects and completion |
| --- | --- | --- |
| Selection | platform, exact GPU device | driver, exact GPU device |
| Context | `cl_context` | `ze_context_handle_t` |
| Artifact | `cl_program`, `cl_kernel` | `ze_module_handle_t`, `ze_kernel_handle_t` |
| Submission | in-order `cl_command_queue` | regular command queue/list or immediate list |
| Completion | blocking operations and profiling events | list/queue sync and timestamp event |
| Allocation | buffer, mapped buffer, coarse SVM | host, shared, or device allocation |
| Invalidation | context/device error invalidates attached resources | device/context error invalidates attached resources |

One context per model attachment is the smallest useful isolation boundary.
A process-global loader/device inventory may be cached, but a context is not
shared between independent model attachments in the pre-plan. That makes
teardown and CPU fallback local and prevents one failed attachment from owning
another model's resources.

The future queue/list pool is fixed and bounded. A lease is exclusive; no
command list is reset, no scratch buffer is reused, and no model resource is
released until its completion token is terminal. OpenCL queue access is
serialized. Level Zero regular lists are recycled only after queue completion;
immediate lists are leased the same way. Per-launch context/module/list
construction is rejected.

## Model and execution lifetimes

`XeModelResources` is created transactionally:

1. Validate the artifact ABI, device/cache identity, tensor shapes, and memory
   budget.
2. Allocate the narrow compact region or bounded slab and reusable scratch.
3. Copy and validate the selected tensor slice.
4. Publish the attachment only after every step passes.

Failure before publication destroys the partial resources in reverse order.
There is no general allocator, LRU, background migration, or second
model-scale derived representation.

An `XeExecution` owns or borrows all activation preparation, staging, output,
queue/list, event, and timing state through completion. Asynchronous Rust
support is deferred: the sprint uses blocking completion so it cannot hide a
borrowed-buffer lifetime behind an untracked callback or future.

## Failure, cancellation, and commit protocol

```text
prepare CPU-owned inputs
  -> submit Xe work
  -> wait and validate status + output
     -> success: convert output, then commit model-visible state once
     -> any pre-commit failure: discard all Xe bytes, recompute on CPU, commit once
```

A failed Xe operation may be recomputed on CPU only before model-state commit.
No logits, cache entries, router state, counters, streamed tokens, or other
externally visible result may be committed from an incomplete or unvalidated
GPU operation. After commit, the same operation is never silently replayed.

Cancellation before submission returns the lease without GPU work.
Cancellation after submission stops waiting only where the API provides a safe
bounded wait; it does not free in-flight resources or pretend to preempt an
arbitrary kernel. OpenCL core queue completion has no general bounded host
timeout, while Level Zero synchronization accepts a timeout. Both paths need a
drain-or-invalidate shutdown state.

Unsafe device-loss injection on the active display GPU is unavailable, not
simulated. A real device/context-loss result would atomically mark the
attachment unusable, reject new submissions, preserve resources until safe
teardown, and route only uncommitted operations to CPU. Automatic selection is
outside this sprint.

## Audited unsafe boundary

The smallest boundary is the checked-in C loader shim plus `src/ffi.rs` and
`src/runtime.rs`. It covers exact symbol resolution, handle creation,
argument-size calls, command submission, timing, allocation, and destruction.
Tensor parsing, ABI/shape validation, cache policy, numerical comparison, and
commit eligibility remain safe Rust. Adopting all of `opencl3`, `oneapi-rs`,
or a generated full API surface would enlarge the audited boundary without
evidence of a benefit for this forced-only lane.
