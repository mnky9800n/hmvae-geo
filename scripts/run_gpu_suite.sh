#!/usr/bin/env bash
#
# Everything the JGR:ML rewrite needs from the GPU workstation, in the
# order that matters. Each step writes its own JSON under data/results/
# and is skipped if that file already exists, so the script can be
# re-run after an interruption without repeating finished work.
#
# The zero-shot transfer bootstrap runs first because it gates the
# headline number; nothing in the Results section can be written until
# it lands.
#
# Usage:
#   bash scripts/run_gpu_suite.sh            # everything still missing
#   bash scripts/run_gpu_suite.sh --force    # redo everything
#   FOLDS=5 bash scripts/run_gpu_suite.sh    # quick pass at low fold count
#
set -uo pipefail

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)" || exit 1

FOLDS="${FOLDS:-30}"
RESULTS="data/results"
LOGS="$RESULTS/logs"
mkdir -p "$LOGS"

FORCE=0
[ "${1:-}" = "--force" ] && FORCE=1

failed=()

step() {   # step <output-json> <label> <command...>
    local out="$RESULTS/$1"; shift
    local label="$1"; shift
    local log="$LOGS/$label.log"

    if [ "$FORCE" -eq 0 ] && [ -f "$out" ]; then
        echo "[skip] $label  ($out exists)"
        return
    fi

    echo "[run ] $label  -> $out"
    local start=$SECONDS
    if "$@" > "$log" 2>&1; then
        echo "[done] $label  ($(( (SECONDS - start) / 60 )) min, log: $log)"
    else
        echo "[FAIL] $label  (log: $log)"
        tail -20 "$log" | sed 's/^/       /'
        failed+=("$label")
    fi
}

echo "=== LilyVAE GPU suite: $FOLDS folds ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || \
    echo "(no nvidia-smi; will fall back to whatever get_device picks)"

# 1. Gates the headline number.
step zeroshot_bootstrap.json zeroshot-transfer \
    uv run lilyvae-zeroshot-bootstrap --iterations "$FOLDS"

# 2. How hard is the task? Raw inputs vs the embedding, same folds.
step raw_baselines.json raw-baselines \
    uv run lilyvae-raw-baselines --iterations "$FOLDS"

# 3. Every table row under one protocol, instead of bootstrapped VAEs
#    beside single-run in-sample baselines.
step bootstrap_all_models.json bootstrap-all-models \
    uv run lilyvae-bootstrap --iterations "$FOLDS" --all-models \
        --output "$RESULTS/bootstrap_all_models.json"

# 4. Is the signal just depth and compaction?
step depth_confound.json depth-confound \
    uv run lilyvae-depth-confound --model unsupervised

# 5. Does releasing the clamp floor fix the calibration?
step calibration.json calibration \
    uv run lilyvae-calibration --clamp-sweep

echo
if [ ${#failed[@]} -eq 0 ]; then
    echo "=== all steps complete ==="
else
    echo "=== ${#failed[@]} step(s) failed: ${failed[*]} ==="
    exit 1
fi
