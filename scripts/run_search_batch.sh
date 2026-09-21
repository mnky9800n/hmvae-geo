#!/usr/bin/env bash
#
# Run a batch of proposed configurations through lilyvae-search, one at
# a time, appending each to the ledger.
#
# Each line of the proposals file is: <JSON config><TAB><rationale>
# Blank lines and lines starting with # are ignored. A configuration
# already in the ledger is skipped by lilyvae-search itself, so the
# batch is safe to re-run after an interruption.
#
# Usage:
#   bash scripts/run_search_batch.sh proposals.tsv [folds]
#
set -uo pipefail

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)" || exit 1

PROPOSALS="${1:?give a proposals file}"
FOLDS="${2:-3}"

[ -f "$PROPOSALS" ] || { echo "no such file: $PROPOSALS"; exit 1; }

n=0
while IFS=$'\t' read -r config rationale; do
    case "$config" in ''|'#'*) continue ;; esac
    n=$((n+1))
    echo
    echo "=============================================================="
    echo "batch item $n"
    echo "=============================================================="
    uv run lilyvae-search run \
        --config "$config" \
        --by agent \
        --rationale "${rationale:-}" \
        --folds "$FOLDS" || echo "  (trial failed, recorded in the ledger)"
done < "$PROPOSALS"

echo
echo "batch complete: $n proposal(s)"
uv run lilyvae-search status
