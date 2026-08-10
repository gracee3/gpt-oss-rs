# CPU Backend Agent Coordination

This directory is the shared coordination surface for CPU-backend work. It was
introduced on `agent/cpu-model-runtime` in PR #3; carry it forward to each
follow-up branch used by both hosts.

## File ownership

- `i7.md` is edited only by the i7-host agent.
- `t14.md` is edited only by the ThinkPad T14 agent.
- This `README.md` is the stable protocol. Change it only when both workstreams
  need a protocol change.

Each agent reads both status files, but writes only its own. This keeps routine
status updates merge-conflict free.

## Sync loop

Before claiming work:

1. Preserve any local changes; never discard another agent's work.
2. Fetch `origin` and rebase the local working branch onto its matching remote
   branch.
3. Read both host status files.
4. Record the proposed work and owned files in the host's own status file.
5. Commit and push the claim before changing implementation files.

Before every implementation push:

1. Run the relevant formatting, check, and test gates.
2. Commit the local work.
3. Fetch `origin` and rebase onto the matching remote branch.
4. Re-run affected tests after a non-trivial rebase.
5. Push normally. Never force-push the shared branch.

A typical clean sync is:

```bash
git fetch origin
git rebase "origin/$(git branch --show-current)"
git push origin HEAD
```

If the worktree is not clean, commit the coherent local work before rebasing.
Do not use destructive reset or checkout commands to make it clean.

## Coordination rules

- Do not concurrently edit a file claimed by the other host.
- Keep claims narrow and list exact files or directories when possible.
- Put questions and handoff requests in the `Requests for other host` section
  of the writer's status file.
- Clear or move completed claims promptly so the other host knows the surface
  is available.
- Repository commits remain the source of truth; status files summarize them
  and must include the current commit ID.

## Required status fields

Each host status file records:

- last update timestamp with timezone;
- current commit and worktree state;
- active objective;
- owned files or surfaces;
- validation completed;
- requests for the other host;
- immediate next step.
