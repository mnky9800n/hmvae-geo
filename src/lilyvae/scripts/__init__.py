"""CLI entry points for the LilyVAE pipeline."""

from __future__ import annotations

import json
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def make_serializable(obj):
    """Recursively convert numpy types to Python built-ins for json.dump."""
    if isinstance(obj, (np.floating, np.integer)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [make_serializable(v) for v in obj]
    return obj


def run_provenance() -> dict:
    """Who made this file, from which code, and when.

    A number in the manuscript is only checkable if the run behind it can
    be found again. Recording the commit, whether the tree was dirty at
    the time, and the command that produced the file is what makes that
    possible a year later, when the only other evidence is the number
    itself.

    `dirty` matters as much as the SHA. A result produced from
    uncommitted work cannot be reproduced from the commit alone, and a
    file that says so is more use than one that implies a clean tree.
    """
    root = Path(__file__).resolve().parents[3]

    def git(*args: str) -> str | None:
        try:
            out = subprocess.run(("git", *args), cwd=root, capture_output=True,
                                 text=True, timeout=10)
        except (OSError, subprocess.SubprocessError):
            return None
        return out.stdout.strip() if out.returncode == 0 else None

    sha = git("rev-parse", "HEAD")
    status = git("status", "--porcelain")
    deployed = root / ".deployed_sha"

    # A deployed copy carries no .git, so a run on the GPU box would
    # record no commit at all. The deploy writes the SHA it shipped and
    # we read it here, which is the point of deploying a clean tree.
    if sha is None and deployed.exists():
        sha = deployed.read_text().strip() or None
        status = ""

    return {
        "git_sha": sha,
        "git_dirty": bool(status) if status is not None else None,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "host": socket.gethostname(),
        "command": " ".join(sys.argv),
        "python": sys.version.split()[0],
    }


def write_result(path, payload: dict) -> Path:
    """Write a result file with its provenance attached.

    Every script writes through here, so no result lands on disk without
    a record of the run that made it.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    body = make_serializable(payload)
    if isinstance(body, dict):
        body = {"_provenance": run_provenance(), **body}
    with open(path, "w") as f:
        json.dump(body, f, indent=2)
    return path
