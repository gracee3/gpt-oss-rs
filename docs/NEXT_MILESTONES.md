# Next Milestones

Current aligned checkpoint:

- `main`: `da8f86a`
- `integration/mainline-alignment`: `da8f86a`
- `harness/tier2-workflow`: `da8f86a`
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
