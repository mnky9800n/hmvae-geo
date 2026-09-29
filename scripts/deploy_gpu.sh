#!/usr/bin/env bash
#
# Ship the current commit to the GPU box and record which one it is.
#
# A run is only worth its wall-clock time if the number it produces can
# be traced back to code. Deploying an rsync of whatever happens to be in
# the working tree breaks that, and it already cost us one four-hour run
# that would have landed unstamped. So this refuses to deploy a dirty
# tree, and writes the SHA it shipped to .deployed_sha, which
# run_provenance() reads because the deployed copy has no .git.
#
# Usage:  bash scripts/deploy_gpu.sh [host]        (default: lambda-scalar)
#
set -euo pipefail

HOST="${1:-lambda-scalar}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [ -n "$(git status --porcelain)" ]; then
    echo "FAIL: working tree is dirty. Commit first, so the run records a"
    echo "      commit that exists. Uncommitted:"
    git status --short | sed 's/^/    /'
    exit 1
fi

SHA="$(git rev-parse HEAD)"
BRANCH="$(git rev-parse --abbrev-ref HEAD)"
echo "deploying $BRANCH $SHA to $HOST"

ssh "$HOST" "mkdir -p ~/repos/lilyvae"
# --delete, never --delete-excluded. The latter deletes the excluded
# paths on the receiver too, which wiped data/results/ and logs/ on the
# box on every deploy, once while a run was writing to a log there.
rsync -az --delete \
    --exclude 'data/raw' --exclude 'data/results' --exclude '.git' \
    --exclude '__pycache__' --exclude '.venv' --exclude 'logs' \
    --exclude 'an-unsupervised-embeddings-model-for-ocean-crust-lithology' \
    ./ "$HOST:~/repos/lilyvae/"

ssh "$HOST" "echo '$SHA' > ~/repos/lilyvae/.deployed_sha && cd ~/repos/lilyvae && uv sync --quiet"
echo "deployed. results written on $HOST will record $SHA"
