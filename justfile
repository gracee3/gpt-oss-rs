default:
    just --list

venv:
    python3 -m venv .venv

install-python-tools:
    .venv/bin/python -m pip install --upgrade pip
    .venv/bin/python -m pip install -r requirements.txt

validate-final-readout artifact:
    python3 scripts/validate_runtime_forward_final_readout_status.py --artifact "{{artifact}}"

validate-final-readout-source:
    test -f /home/emmy/openai/worktrees/runtime-forward/.live/runtime-forward-final-readout-20260423/developer-message.runner-final-readout-direct-module-rerun-status.json
    python3 scripts/validate_runtime_forward_final_readout_status.py \
      --artifact /home/emmy/openai/worktrees/runtime-forward/.live/runtime-forward-final-readout-20260423/developer-message.runner-final-readout-direct-module-rerun-status.json \
      --check-manifest /home/emmy/openai/worktrees/runtime-forward/.live/pinned-prompt-parity-official-reference-20260424/LARGE_ARTIFACTS_MANIFEST.json

py-check:
    python3 -m py_compile scripts/validate_runtime_forward_final_readout_status.py

diff-check:
    git diff --check
