# Final authorized H8 attempt

**Status:** construction not yet launched. The first passing preflight in this
directory is retained as a pre-fix admission diagnostic only.

`preflight.json` was produced from clean synchronized HEAD
`b4b7206692814233973b30e6ff690d5351b0a34d` by watchdog executable SHA-256
`80f516abb737b1e3421825cf63ad2727b38b062900c9a3b3ebb0ff7877f34b98`.
Its five samples span exactly 120,000 ms and pass every frozen preflight gate.
The subsequent `run` command rejected before spawning a child because the
process scanner confused the watchdog's own trailing child arguments with an
active constructor. No model was opened and no construction attempt occurred.

Commit `2082c41` narrows the readable-process identity fallback to the exact
Linux-truncated constructor name and adds the real watchdog-command regression.
The final preflight and, if admitted, the single authorized construction launch
must be generated from a clean committed HEAD containing that fix. They must
use new evidence paths; this diagnostic is immutable and is not an admission
artifact for the final run.

