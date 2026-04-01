# Next Milestones

Current aligned checkpoint:

- `main`: `b4c2efa`
- `integration/mainline-alignment`: `b4c2efa`
- `harness/tier2-workflow`: `b4c2efa`
- `feature/runtime-forward`: `bd49d35`

Active workstreams:

1. `integration/mainline-alignment` at `~/openai/gpt-oss-rs`
2. `harness/tier2-workflow` at `~/openai/worktrees/tier2-workflow`
3. `feature/runtime-forward` at `~/openai/worktrees/runtime-forward`

Near-term milestones:

## M1. Keep the aligned mainline stable

- keep the three-worktree layout intact
- promote only small validated batches from active lanes
- keep workstream docs and TODOs current as branch tips move
- fix integration-safe validation harness defects without changing compare semantics
- current small ready extraction: `integration/tier01-lane` commit `391a975`

## M2. Make the Tier-2 harness operator-ready

- improve seed-capture and local-replay operator ergonomics
- document live restricted-fp16 CUDA execution steps
- add any narrow harness-only checks that reduce rerun cost without changing default behavior

## M3. Validate future runtime claims against the contract

- use raw compare as telemetry
- use runtime-emulated compare as localization
- require same-input local replay before claiming ownership of a runtime defect

## M4. Keep runtime-forward isolated

- continue incomplete runtime work only on `feature/runtime-forward`
- mine archived/debug branches selectively by commit when needed
- do not treat historical exploratory branches as active integration lanes
