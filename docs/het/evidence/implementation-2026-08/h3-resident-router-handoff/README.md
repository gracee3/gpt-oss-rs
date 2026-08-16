# H3 resident exact-router handoff evidence

**Status:** source and synthetic-CUDA prerequisite validated for supervisor
review. This is not a model-backed construction or execution record and does
not pass H8, H9, or H10.

The final-source tests construct generated BF16 router surfaces in device byte
allocations, consume those allocations through the new handoff, and compare
the returned router bit-for-bit with the unchanged host-backed constructor.
The two cases deliberately cover `E=32` on one local RTX 3090 and `E=128` on
the other. No stable PCI identifier, UUID, hostname, or checkpoint-derived
value is retained here.

The retained [validation manifest](validation.json) binds the source diff and
both focused test executables. Its fault-enabled case proves three distinct
lifecycle outcomes:

- a successful terminal copy drain releases the source before constructor
  return;
- a recoverable post-enqueue failure drains, releases, and permits a fresh
  retry; and
- an injected fallback-drain failure returns no router, does not release the
  source, and retains every possibly referenced source/destination/stream/
  context handle for process life.

The implementation uses cudarc's checked-length unsafe `CudaSlice<u8>` to
borrowed `CudaView<u16>` transmute only for same-context D2D into ordinary
router-owned `CudaSlice<u16>` destinations. The existing router weight/bias
fields and projection launch remain unchanged. The complete boundary and its
remaining model-ownership dependency are documented in
[`../../../33-h3-resident-router-handoff.md`](../../../33-h3-resident-router-handoff.md).

All commands removed the three model-path variables and all known opt-in model
run gates. The real model-gated router test observed its unset gate and did no
model work. No model path was opened, statted, mapped, hashed, constructed, or
executed. `/dev/nvme1n1` was observed read-only and unmounted; it was not
opened for data I/O.

Verify the bounded record with:

```bash
cd docs/het/evidence/implementation-2026-08/h3-resident-router-handoff
sha256sum -c SHA256SUMS
jq -e . validation.json >/dev/null
```
