#!/usr/bin/env bash
#
# The bottleneck comparison, for the reviewer's first point.
#
# At ten latent dimensions for six inputs the encoder reconstructs at
# R^2 = 0.999, so the latent is close to an invertible transform of the
# inputs. Any label that is a function of them is already present, and
# supervision can only change the geometry rather than add anything. A
# narrower latent forces the model to discard, which is the only setting
# where lithology labels could steer what it keeps.
#
# Both VAEs at three dimensions, then at two, on the same 30 borehole
# folds as everything else.
#
# Usage:  bash scripts/run_bottleneck.sh
#
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
mkdir -p logs

for d in 3 2; do
    out="data/results/bootstrap_latent$d.json"
    if [ -f "$out" ]; then
        echo "=== latent $d  skipped, $out exists ==="
        continue
    fi
    echo "=== latent $d  start $(date -u +%H:%M) ==="
    if PYTHONUNBUFFERED=1 uv run lilyvae-bootstrap --latent-dim "$d" \
            --iterations 30 --device cuda --output "$out" \
            > "logs/bootstrap_latent$d.log" 2>&1; then
        echo "    ok  $(date -u +%H:%M)"
    else
        echo "    FAILED, see logs/bootstrap_latent$d.log"
        tail -3 "logs/bootstrap_latent$d.log" | sed 's/^/      /'
    fi
done
echo "=== done  $(date -u +%H:%M) ==="
