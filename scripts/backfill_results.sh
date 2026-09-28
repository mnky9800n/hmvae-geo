#!/usr/bin/env bash
#
# Run every experiment whose result file the code expects and does not
# have. `check_results_current.py` lists them; this produces them, in
# increasing order of cost so the cheap gaps close first and a failure
# late in the queue does not cost the early results.
#
# Each run logs to logs/<name>.log and the queue continues past a
# failure, because one broken script should not hold up the rest.
#
# Usage:  bash scripts/backfill_results.sh
#
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
mkdir -p logs

run() {                       # run <log name> <command...>
    local name="$1"; shift
    echo "=== $name  $(date -u +%H:%M:%S) ==="
    if uv run "$@" > "logs/$name.log" 2>&1; then
        echo "    ok"
    else
        echo "    FAILED, see logs/$name.log"
        tail -3 "logs/$name.log" | sed 's/^/      /'
    fi
}

run raw_baselines   lilyvae-raw-baselines
run calibration     lilyvae-calibration
run depth_confound  lilyvae-depth-confound
run depth_bootstrap lilyvae-depth-bootstrap
run zeroshot_bootstrap lilyvae-zeroshot-bootstrap --iterations 30 --device cuda

echo "=== done  $(date -u +%H:%M:%S) ==="
python3 scripts/check_results_current.py
