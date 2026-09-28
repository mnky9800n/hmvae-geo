#!/usr/bin/env python3
"""Audit the result files behind the manuscript.

Two failures motivated this. A file that the figure code reads was
missing entirely, so the headline zero-shot figure could not be
regenerated from anything in the repo, and nobody noticed because a
stale PNG sits in the paper directory and looks fine. And numbers in
Section 4.4 came from a run nothing recorded, so there was no way to
tell which code produced them or whether they predate a change to the
dataset.

This reports, for every result file the code expects:

    MISSING       the code reads it and it is not there
    unstamped     no provenance block; it predates the stamping or was
                  written by code that did not go through write_result
    no commit     stamped, but the commit was not recorded
    dirty         produced from a working tree with uncommitted changes
    stale         src/ has changed since the commit that produced it

Exit code is 1 if any expected file is missing, which is the only state
that makes a figure unbuildable. Everything else is reported and does
not fail, because a stale result is a judgement call, not a build error.

    python3 scripts/check_results_current.py
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "data" / "results"


def git(*args: str) -> str:
    out = subprocess.run(("git", *args), cwd=ROOT, capture_output=True, text=True)
    return out.stdout.strip() if out.returncode == 0 else ""


def expected_files() -> dict[str, list[str]]:
    """Result files the code reads, and which modules read them."""
    wanted: dict[str, list[str]] = {}
    for py in sorted((ROOT / "src").rglob("*.py")):
        text = py.read_text()
        for name in re.findall(r'RESULTS_DIR\s*/\s*"([^"]+\.json)"', text):
            wanted.setdefault(name, []).append(
                str(py.relative_to(ROOT / "src" / "lilyvae"))
            )
    return wanted


def main() -> int:
    head = git("rev-parse", "HEAD")
    readers = expected_files()
    on_disk = {p.name for p in RESULTS.glob("*.json")}
    missing = 0

    rows = []
    for name in sorted(set(readers) | on_disk):
        who = ", ".join(sorted(set(readers.get(name, [])))) or "not read by any module"
        path = RESULTS / name
        if not path.exists():
            rows.append(("MISSING", name, who, ""))
            missing += 1
            continue
        try:
            prov = json.loads(path.read_text()).get("_provenance")
        except (json.JSONDecodeError, AttributeError):
            prov = None
        if not prov:
            rows.append(("unstamped", name, who, ""))
            continue
        sha, dirty = prov.get("git_sha"), prov.get("git_dirty")
        when = (prov.get("created_utc") or "")[:10]
        if not sha:
            rows.append(("no commit", name, who, when))
            continue
        behind = git("rev-list", "--count", f"{sha}..{head}", "--", "src")
        flags = []
        if dirty:
            flags.append("dirty")
        if behind and behind != "0":
            flags.append(f"stale, src changed {behind}x since")
        rows.append((", ".join(flags) or "ok", name, who, when))

    width = max(len(r[1]) for r in rows) if rows else 20
    print(f"{'file':<{width}}  {'state':<28}  made        read by")
    print("-" * (width + 58))
    for state, name, who, when in rows:
        print(f"{name:<{width}}  {state:<28}  {when or '-':<10}  {who}")

    if missing:
        print(f"\nFAIL: {missing} file(s) the code reads do not exist.")
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
