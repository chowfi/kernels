#!/usr/bin/env bash
# Launch the autoresearch loop. The driver itself appends every status line to progress.log
# (and echoes it to stdout); nothing else may write that file.
# Unattended (survives logout):  nohup ./run_loop.sh > driver.out 2>&1 &
# Smoke test (one experiment):   ./run_loop.sh --max-experiments 1
cd "$(dirname "$0")" || exit 1
exec uv run --frozen python -m driver.loop "$@"
