#!/usr/bin/env bash
#
# Produce the result files the code reads and does not have.
# `check_results_current.py` lists them; this runs them.
#
# Order is by what the manuscript is blocked on, NOT by cost. An earlier
# version of this script ordered by guessed cost and put the zero-shot
# bootstrap last, behind raw_baselines, which reads as cheap and is not:
# it defaults to 30 iterations, retrains a VAE per fold and runs five
# arms including kNN over 237k rows. It ran for over an hour and a half
# while the file the headline figure needs had not been started.
#
# None of these are cheap. Budget hours, not minutes, and run it on the
# GPU box through deploy_gpu.sh so the results record a commit.
#
# A step whose result file already exists is skipped, so an interrupted
# queue can be restarted without repeating work. FORCE=1 runs everything.
#
# Usage:  bash scripts/backfill_results.sh
#         FORCE=1 bash scripts/backfill_results.sh
#
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
mkdir -p logs

run() {                       # run <result file> <log name> <command...>
    local result="$1" name="$2"; shift 2
    if [ -z "${FORCE:-}" ] && [ -f "data/results/$result" ]; then
        echo "=== $name  skipped, data/results/$result exists ==="
        return
    fi
    echo "=== $name  start $(date -u +%H:%M:%S) ==="
    # Unbuffered, or the log stays empty for hours and a running job is
    # indistinguishable from a hung one.
    if PYTHONUNBUFFERED=1 uv run "$@" > "logs/$name.log" 2>&1; then
        echo "    ok    $(date -u +%H:%M:%S)"
    else
        echo "    FAILED, see logs/$name.log"
        tail -3 "logs/$name.log" | sed 's/^/      /'
    fi
}

# First: the file behind Figure 5 and the two other holdout figures.
run zeroshot_bootstrap.json zeroshot_bootstrap \
    lilyvae-zeroshot-bootstrap --iterations 30 --device cuda
# Then the depth results, which Section 4.5 and the Discussion both use.
run depth_confound.json  depth_confound  lilyvae-depth-confound
run depth_bootstrap.json depth_bootstrap lilyvae-depth-bootstrap
# Then the comparisons that support the table rather than a figure.
run raw_baselines.json   raw_baselines   lilyvae-raw-baselines
run calibration.json     calibration     lilyvae-calibration

echo "=== done  $(date -u +%H:%M:%S) ==="
python3 scripts/check_results_current.py
