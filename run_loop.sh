#!/usr/bin/env bash
# Launch the autoresearch loop. Output is mirrored to progress.log.
# Unattended (survives logout):  nohup ./run_loop.sh > /dev/null 2>&1 &
# Smoke test (one experiment):   ./run_loop.sh --max-experiments 1
cd "$(dirname "$0")" || exit 1
uv run python -m driver.loop "$@" 2>&1 | tee -a progress.log
