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
    stale         the code that writes it has changed since
    producer unknown
                  nothing here writes this name (usually --output)

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
    # Both trees: the package, and the loose scripts beside it. Walking
    # only src/ hid scripts/paired_comparisons.py, so the file it writes
    # looked like it had no producer at all.
    for base in (ROOT / "src", ROOT / "scripts"):
        for py in sorted(base.rglob("*.py")):
            text = py.read_text()
            for name in re.findall(r'RESULTS_DIR\s*/\s*"([^"]+\.json)"', text):
                rel = py.relative_to(ROOT / "src" / "lilyvae") if base.name == "src" \
                    else py.relative_to(ROOT)
                wanted.setdefault(name, []).append(str(rel))
    return wanted


def code_for(name: str, readers: dict[str, list[str]]) -> list[str]:
    """The paths whose history can make this result stale.

    Counting every commit to src/ marked a file stale because an
    unrelated module changed, which trains the reader to ignore the
    column. This counts the modules that name the file, plus the
    evaluation module each of them shares a name with, which is how this
    package pairs a script with the code it drives.
    """
    paths = set()
    for rel in readers.get(name, []):
        # A figure module reads a result; it cannot invalidate one. Only
        # the code that writes the file counts here.
        if rel.startswith("plotting/"):
            continue
        if rel.startswith("scripts/"):
            paths.add(rel)
            continue
        paths.add(f"src/lilyvae/{rel}")
        sibling = Path("src/lilyvae/evaluation") / Path(rel).name
        if (ROOT / sibling).exists():
            paths.add(str(sibling))
    # Empty means nothing here writes this name, usually because it was
    # produced with --output. Saying so beats counting every commit to
    # src/ and calling the result stale.
    return sorted(paths)


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
        producers = code_for(name, readers)
        flags = []
        if dirty:
            flags.append("dirty")
        if not producers:
            flags.append("producer unknown")
        else:
            behind = git("rev-list", "--count", f"{sha}..{head}", "--", *producers)
            if behind and behind != "0":
                flags.append(f"stale, writer changed {behind}x since")
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
